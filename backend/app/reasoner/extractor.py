"""
Slot extractor — the Reasoner's "sensory" layer.

Called asynchronously after each user turn boundary. Takes the recent
transcript (PersonaPlex's monologue stream + Scribe v2 ASR + Sarah's
read-back history) and returns updates to the ClaimReport.

Design:
  - Anthropic Claude Haiku via tool-use API for structured output.
  - Pydantic schema → JSON schema → tool input schema (single source of truth).
  - Merge semantics: only updates empty slots OR slots that were just verbally
    corrected after a read-back. Never silently overwrites confirmed values.
  - Async: never blocks the per-frame audio loop. The Reasoner reads whatever
    the most recent extraction said.

Latency budget: ~200-400ms per call (Haiku TTFT). Runs once per user-turn
boundary, not per frame. Async so the audio path doesn't wait.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from .schema import (
    ClaimReport,
    PolicyContext,
    ReadbackEvent,
    TranscriptTurn,
)


HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ─── System prompt for the extractor ─────────────────────────────────────────

EXTRACTOR_SYSTEM = """\
You are the slot extractor for an insurance claims voice agent. You read the \
ongoing call transcript and update a structured FNOL (First Notice of Loss) \
report.

Rules:

1. Only fill in fields that have NOT yet been captured, OR that have been \
explicitly corrected by the caller after Sarah read them back.

2. Never invent values. If the caller hasn't mentioned a field, leave it null. \
Do not guess, infer, or extrapolate beyond what was said.

3. The transcript may contain BOTH PersonaPlex's text monologue AND Scribe \
v2's ASR transcript for the same user turn. They may differ slightly. When \
they disagree on a critical entity (policy_number, license_plate, dates, \
phone, names, monetary amounts), report low confidence and wait for Sarah's \
read-back to resolve.

4. Read-back events are authoritative. If a slot has a "confirmed" \
ReadbackEvent, that value is final — do not overwrite it.

5. Pay attention to correction phrases ("actually", "I meant", "no, it's…", \
"wait, that's wrong"). When the caller corrects a previously stated value, \
update the slot.

6. Don't try to fill EVERY field every turn. Only update what's new in this \
turn's transcript. If nothing relevant was said, return an empty update.

Return a JSON object containing only the fields that should be UPDATED, plus \
a `reasoning` field explaining your decisions briefly.
"""


# ─── Compact extractor output schema ─────────────────────────────────────────
# We don't want Haiku to return the full ClaimReport schema — too many fields,
# slow, and most are null per turn. Instead we let it return a sparse update
# dict keyed by slot path.

class SlotUpdate(BaseModel):
    """One slot update proposed by the extractor."""
    slot_path: str = Field(
        description="Dotted path into ClaimReport, e.g. 'policy_number' or "
        "'other_party.license_plate' or 'incident_datetime'."
    )
    value: Any = Field(
        description="The extracted value. Use the type expected by the field "
        "in ClaimReport (string, datetime ISO format, bool, list, etc.)."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="0.0-1.0. High when caller stated clearly and both transcripts "
        "agree. Low when transcripts disagreed or speech was unclear."
    )
    source_quote: str = Field(
        description="The exact phrase from the caller that supports this value, "
        "for audit / read-back generation."
    )
    is_correction: bool = Field(
        default=False,
        description="True if this is overriding a previously stated value because "
        "the caller said something like 'actually' or 'I meant'."
    )


class ExtractorOutput(BaseModel):
    """Full response from one extractor invocation."""
    updates: list[SlotUpdate] = Field(default_factory=list)
    reasoning: str = Field(
        description="Brief explanation of what was extracted and any decisions "
        "about merging vs corrections."
    )


# ─── Extractor implementation ────────────────────────────────────────────────

class SlotExtractor:
    """Async slot extractor using Claude Haiku tool-use.

    Holds an Anthropic client and prompt-cached system prompt. Called once
    per user-turn boundary by the Reasoner orchestrator.
    """

    def __init__(self, api_key: str | None = None):
        self.client = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    async def extract(
        self,
        transcript_window: list[TranscriptTurn],
        current_report: ClaimReport,
        readbacks: list[ReadbackEvent],
        policy_context: PolicyContext | None = None,
        now: datetime | None = None,
    ) -> ExtractorOutput:
        """Run extraction over the recent transcript window.

        transcript_window: last ~10-20 turns (PersonaPlex monologue + Scribe ASR).
        current_report: the slots already filled in ClaimReport.
        readbacks: history of confirmed/corrected entities.
        policy_context: loaded policy info if authentication completed (used
            so the extractor knows what's already known and shouldn't be
            asked of the caller).
        now: tz-aware current time; needed to resolve relative time references
            in the caller's speech ("this morning", "yesterday"). Defaults to
            datetime.now(timezone.utc) — orchestrator should pass the
            call-start time for deterministic resolution.

        Returns ExtractorOutput with sparse updates + reasoning trace.
        """
        user_message = self._build_user_message(
            transcript_window, current_report, readbacks, policy_context,
            now=now or datetime.now(timezone.utc),
        )

        response = await self.client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": EXTRACTOR_SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # prompt cache
                }
            ],
            tools=[
                {
                    "name": "report_extraction",
                    "description": "Report extracted slot updates from the call transcript.",
                    "input_schema": ExtractorOutput.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "report_extraction"},
            messages=[{"role": "user", "content": user_message}],
        )

        # Find the tool_use block in the response
        for block in response.content:
            if block.type == "tool_use" and block.name == "report_extraction":
                return ExtractorOutput.model_validate(block.input)

        # If model didn't call the tool, return empty (shouldn't happen with tool_choice)
        return ExtractorOutput(reasoning="No tool call returned by the model.")

    @staticmethod
    def _build_user_message(
        transcript_window: list[TranscriptTurn],
        current_report: ClaimReport,
        readbacks: list[ReadbackEvent],
        policy_context: PolicyContext | None,
        now: datetime | None = None,
    ) -> str:
        """Format the conversation context for Haiku.

        Compact format — Haiku is fast but token cost adds up over a long call.
        """
        parts: list[str] = []

        # 0. Time context so the model can resolve "this morning", "yesterday", etc.
        if now is not None:
            parts.append(
                f"## Current time (for resolving relative references)\n"
                f"{now.isoformat()}\n"
                f"Use this to resolve phrases like 'this morning', "
                f"'yesterday', 'an hour ago' into concrete ISO datetimes."
            )

        # 1. What we already know from the DB (so extractor doesn't try to fill these)
        if policy_context is not None:
            parts.append(
                "## Already known from policy DB (do NOT ask caller for these)\n"
                f"- Policyholder: {policy_context.policyholder.full_name}\n"
                f"- Vehicle: {policy_context.make} {policy_context.model} "
                f"(plate {policy_context.license_plate}, VIN {policy_context.vin})\n"
                f"- Coverage: {policy_context.kasko_type.value}\n"
                f"- Policy number: {policy_context.policy_number}\n"
            )
        else:
            parts.append(
                "## Authentication phase — caller has not yet been identified.\n"
                "Priority: extract policy_number, license_plate, or "
                "(reporter_name + vehicle make) so the policy can be looked up.\n"
            )

        # 2. Current report state — what's already filled
        filled = current_report.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        if filled:
            parts.append(
                "## Already captured in this call\n"
                + json.dumps(filled, indent=2, ensure_ascii=False)
            )

        # 3. Confirmed entities via read-back (immutable)
        confirmed = [r for r in readbacks if r.caller_response == "confirmed"]
        if confirmed:
            parts.append(
                "## Confirmed via read-back (do NOT overwrite)\n"
                + "\n".join(f"- {r.slot_path}: {r.final_value}" for r in confirmed)
            )

        # 4. Recent transcript window
        parts.append("## Recent transcript")
        for turn in transcript_window:
            tag = "CALLER" if turn.role == "caller" else "SARAH"
            src = f"[{turn.source}]" if turn.source != "personaplex" else ""
            parts.append(f"{tag} {src}: {turn.text}")

        parts.append(
            "\n## Task\n"
            "Call `report_extraction` with the slot updates implied by the most "
            "recent caller turn(s) only. Skip fields that haven't been mentioned. "
            "Flag any disagreement between the two transcript sources with low confidence."
        )

        return "\n\n".join(parts)


# ─── Smoke test entry point (offline, no API call) ──────────────────────────
# Builds the prompt and prints it without invoking Haiku. Useful for prompt
# tuning without burning API credits.
#
# uv run python -m backend.app.reasoner.extractor

if __name__ == "__main__":
    from datetime import datetime, timezone

    from . import db

    # Fixture: caller authenticated as Anna Schmidt, mid-call describing accident.
    policy = db.lookup("B-AL-1234")
    assert policy is not None

    transcript = [
        TranscriptTurn(
            role="agent",
            text="Allianz Claims, this is Sarah. How can I help you today?",
            timestamp=datetime.now(timezone.utc),
        ),
        TranscriptTurn(
            role="caller",
            text="Hi yes, I had an accident this morning, my plate is B-AL-1234",
            timestamp=datetime.now(timezone.utc),
        ),
        TranscriptTurn(
            role="agent",
            text="Okay, B as in Berlin, A-L, 1-2-3-4 — is that right?",
            timestamp=datetime.now(timezone.utc),
        ),
        TranscriptTurn(
            role="caller",
            text="Yes that's correct.",
            timestamp=datetime.now(timezone.utc),
        ),
        TranscriptTurn(
            role="agent",
            text="Got it, I'm pulling up your file now... I see your VW Golf. "
            "Tell me what happened.",
            timestamp=datetime.now(timezone.utc),
        ),
        TranscriptTurn(
            role="caller",
            text="So I was on the A100 around eight thirty this morning, "
            "stopped at a red light, and the car behind rear-ended me. "
            "Nobody hurt but my back bumper is pretty smashed.",
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    report = ClaimReport(
        policy_number=policy.policy_number,
        license_plate=policy.license_plate,
    )

    readbacks = [
        ReadbackEvent(
            slot_path="license_plate",
            proposed_value="B-AL-1234",
            caller_response="confirmed",
            final_value="B-AL-1234",
            timestamp=datetime.now(timezone.utc),
        )
    ]

    user_msg = SlotExtractor._build_user_message(
        transcript, report, readbacks, policy
    )
    print("=" * 70)
    print("EXTRACTOR USER MESSAGE (would be sent to Haiku):")
    print("=" * 70)
    print(user_msg)
    print()
    print("=" * 70)
    print(f"Tokens (rough): ~{len(user_msg) // 4}")
    print("To actually invoke Haiku, set ANTHROPIC_API_KEY and call "
          "SlotExtractor().extract(...)")
