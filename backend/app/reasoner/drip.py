"""
Control-plane directive protocol — Reasoner → model_service.

The audio plane (24kHz PCM bidirectional) flows over its own WebSocket.
This module defines the *control* messages that the Reasoner pushes to
model_service to influence what PersonaPlex says next.

Two design rules from architecture.md / VAOS journal:

  1. Directives are pushed asynchronously and cached on model_service side.
     The model_service per-frame loop reads the latest cached directive
     locally — no per-frame RPC. This is what keeps injection latency at ~0.

  2. Text content must be drip-fed at the 12.5 Hz (80ms/frame) cadence on
     PersonaPlex's text monologue stream. Burst injection causes audio-head
     degeneration (token repetition collapse). Backend sends the *string*
     here; model_service does the actual tokenization and per-frame writing.

Special tokens we control directly: EPAD=0 (start speaking), PAD=3 (stay
silent), BOS=1, EOS=2 — see PersonaPlex offline.py:293.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ─── Special tokens (mirrored from PersonaPlex offline.py:293) ──────────────

class TextSpecialToken(StrEnum):
    EPAD = "EPAD"  # id=0; force agent to start speaking
    BOS = "BOS"    # id=1; beginning of sentence
    EOS = "EOS"    # id=2; end of sentence
    PAD = "PAD"    # id=3; stay silent


SPECIAL_TOKEN_IDS = {
    TextSpecialToken.EPAD: 0,
    TextSpecialToken.BOS: 1,
    TextSpecialToken.EOS: 2,
    TextSpecialToken.PAD: 3,
}


# ─── Directive types (discriminated union) ───────────────────────────────────

class _BaseDirective(BaseModel):
    """Common metadata. `seq` lets model_service order/deduplicate; `issued_at`
    helps with debugging when the gate fires multiple times in quick succession."""
    seq: int = Field(description="Monotonic per-session counter for ordering.")
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpeakDirective(_BaseDirective):
    """Force Sarah to start speaking the given content via drip-feed.

    model_service will:
      1. Inject EPAD on the next frame to break out of any PAD/silence state
      2. Tokenize `text` and drip the tokens at 12.5Hz (~20 chars/frame)
      3. After the drip ends, release control back to PersonaPlex's own sampling
         (so Sarah can naturally complete the sentence and continue the call).
    """
    type: Literal["speak"] = "speak"
    text: str = Field(description="Content to drip-feed onto the text monologue stream.")
    after_release: Literal["resume", "silent"] = Field(
        default="resume",
        description="What to do after the drip is exhausted. 'resume' = let "
        "PersonaPlex sample freely (default); 'silent' = force PAD until next "
        "directive (used when we want Sarah to wait for the caller).",
    )
    silent_frames_after: int = Field(
        default=0, ge=0, le=125,
        description="Bounded PAD window (frames @ 12.5Hz, max 10s) inserted "
        "AFTER the drip ends and BEFORE returning to after_release behavior. "
        "Use to give the caller a guaranteed response window before Sarah's "
        "text head free-samples — prevents 'predict the answer' hallucinations "
        "where the model continues a Q→A pattern by vocalizing the likely reply.",
    )
    reason: str = Field(
        default="",
        description="Human-readable reason this was emitted (logged, not sent to model).",
    )


class SilenceDirective(_BaseDirective):
    """Force PersonaPlex into silence by writing PAD on the text monologue
    stream for `duration_frames` frames (each frame = 80ms).

    Use cases:
      - Caller is mid-thought — give them space.
      - We're awaiting an async tool call (DB lookup, RAG) and don't want
        Sarah to fill the gap with hallucinated content.
    """
    type: Literal["silent"] = "silent"
    duration_frames: int = Field(
        ge=1, le=125,
        description="Number of 80ms frames to hold silence (max 10s = 125 frames).",
    )
    reason: str = ""


class ReleaseDirective(_BaseDirective):
    """Cancel any active speak/silent override and let PersonaPlex sample
    normally. Used when the gate has been holding Sarah back and conditions
    have changed (e.g. caller finished their tangent).
    """
    type: Literal["release"] = "release"
    reason: str = ""


class RescueClipDirective(_BaseDirective):
    """Play one of the pre-recorded "you're breaking up" rescue clips. Used
    when audio quality is bad or PersonaPlex's output has degenerated.

    model_service:
      1. Mutes PersonaPlex's audio output frames (server-side gating per VAOS)
      2. Streams the WAV from model_service/assets/rescue_clips/{clip_id}.wav
      3. Restores PersonaPlex output after clip ends + 0.5s cooldown
    """
    type: Literal["rescue_clip"] = "rescue_clip"
    clip_id: str = Field(
        description="Filename (without .wav) under model_service/assets/rescue_clips/."
    )
    reason: str = ""


class LoadPolicyContextDirective(_BaseDirective):
    """Inform model_service that a PolicyContext has been loaded for this
    session. The Reasoner emits this once after authentication; model_service
    holds it in session state for use as drip-feed context.

    The actual policy_brief text is sent as a SpeakDirective separately when
    Sarah is ready to "reference" the file (typically during the lead portion
    of her acknowledgement turn).
    """
    type: Literal["load_policy_context"] = "load_policy_context"
    policy_brief: str = Field(
        description="Compact fact summary from persona.policy_brief() — the "
        "INTERNAL: lines should never reach the audio stream, only the user-"
        "facing facts via subsequent SpeakDirectives."
    )
    reason: str = ""


# Discriminated union — model_service deserializes via the `type` field.
ReasonerDirective = Annotated[
    Union[
        SpeakDirective,
        SilenceDirective,
        ReleaseDirective,
        RescueClipDirective,
        LoadPolicyContextDirective,
    ],
    Field(discriminator="type"),
]


# ─── Directive builder (stateful seq counter) ────────────────────────────────

class DirectiveBuilder:
    """Per-session directive factory. Keeps a monotonic seq counter so
    model_service can order/dedupe."""

    def __init__(self) -> None:
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def speak(
        self,
        text: str,
        *,
        after_release: Literal["resume", "silent"] = "resume",
        silent_frames_after: int = 0,
        reason: str = "",
    ) -> SpeakDirective:
        return SpeakDirective(
            seq=self._next_seq(),
            text=text,
            after_release=after_release,
            silent_frames_after=silent_frames_after,
            reason=reason,
        )

    def silent(self, duration_frames: int, *, reason: str = "") -> SilenceDirective:
        return SilenceDirective(
            seq=self._next_seq(),
            duration_frames=duration_frames,
            reason=reason,
        )

    def release(self, *, reason: str = "") -> ReleaseDirective:
        return ReleaseDirective(seq=self._next_seq(), reason=reason)

    def rescue_clip(self, clip_id: str, *, reason: str = "") -> RescueClipDirective:
        return RescueClipDirective(
            seq=self._next_seq(),
            clip_id=clip_id,
            reason=reason,
        )

    def load_policy_context(
        self, policy_brief: str, *, reason: str = ""
    ) -> LoadPolicyContextDirective:
        return LoadPolicyContextDirective(
            seq=self._next_seq(),
            policy_brief=policy_brief,
            reason=reason,
        )


# ─── Read-back utility ───────────────────────────────────────────────────────
# Sarah's read-back protocol mostly emerges from her persona prompt, but for
# critical-entity slots the gate may want to inject a templated read-back to
# guarantee the protocol fires. Examples:

def render_readback(slot_label: str, value: str) -> str:
    """Generate a templated read-back string for the most common slot shapes.

    The persona prompt also instructs Sarah to read back naturally — this
    function is the deterministic fallback used by the gate when we want to
    *force* a read-back on a critical entity that the slot extractor flagged
    with low confidence.

    Different slot families need different rendering:
      - IDs (policy_number, license_plate, VIN, claim_number) → spell
        character-by-character so the caller can verify each digit/letter
      - Phone / email → spell similarly (digits, @, dots)
      - Datetime → natural English ("5 a.m. on April 26th"), NOT
        character-by-character ISO ("2 0 2 6 dash 0 4 dash 2 6 ...")
      - Names, addresses, free text → speak naturally, no spelling
    """
    if _is_datetime_slot(slot_label):
        return f"Okay, just to confirm — that's {_render_datetime(value)}, is that right?"
    if _is_id_slot(slot_label):
        return f"Okay so that's {_spell_for_voice(value)}, is that right?"
    # Default: read back as plain text without spelling.
    return f"Okay so that's {value}, is that right?"


def _is_datetime_slot(slot_label: str) -> bool:
    return (
        slot_label.endswith("_datetime")
        or slot_label.endswith("_at")
        or slot_label.endswith("_date")
        or slot_label.endswith("_time")
    )


def _is_id_slot(slot_label: str) -> bool:
    return slot_label in {
        "policy_number",
        "license_plate",
        "vin",
        "claim_number",
        "police_case_number",
    } or slot_label.endswith("_phone") or slot_label.endswith("_number")


def _spell_for_voice(value: str) -> str:
    """Render an alphanumeric ID into a more pronounceable form.

    'POL-2024-001' → 'P-O-L dash 2 0 2 4 dash 0 0 1'
    'B-AL-1234'    → 'B dash A-L dash 1 2 3 4'
    """
    out: list[str] = []
    for ch in value:
        if ch == "-":
            out.append("dash")
        elif ch.isspace():
            continue
        else:
            out.append(ch.upper())
    return " ".join(out)


# ─── Datetime rendering ──────────────────────────────────────────────────────

_BERLIN_TZ_NAME = "Europe/Berlin"


def _render_datetime(value: str) -> str:
    """Render an ISO datetime in spoken English aligned to Berlin local time.

    '2026-04-26T05:00:00+00:00' → '5 a.m. on April 26th'
    '2026-04-26T17:30:00Z'      → '7:30 p.m. on April 26th' (Berlin = UTC+2 in DST)

    Falls back to the input string if parsing fails.
    """
    try:
        from zoneinfo import ZoneInfo
        # Tolerate trailing Z (Python <3.11 compat plus some inputs).
        iso = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            # Treat naive as Berlin local — orchestrator usually emits UTC,
            # but be defensive.
            dt = dt.replace(tzinfo=ZoneInfo(_BERLIN_TZ_NAME))
        else:
            dt = dt.astimezone(ZoneInfo(_BERLIN_TZ_NAME))
    except (ValueError, TypeError, ImportError):
        return value

    return f"{_clock_phrase(dt)} on {_date_phrase(dt)}"


def _clock_phrase(dt: datetime) -> str:
    h, m = dt.hour, dt.minute
    if h == 0:
        return "midnight" if m == 0 else f"12:{m:02d} a.m."
    if h == 12:
        return "noon" if m == 0 else f"12:{m:02d} p.m."
    if h < 12:
        return f"{h} a.m." if m == 0 else f"{h}:{m:02d} a.m."
    h12 = h - 12
    return f"{h12} p.m." if m == 0 else f"{h12}:{m:02d} p.m."


def _date_phrase(dt: datetime) -> str:
    """'April 26th'. Ordinal suffix to feel natural in spoken form."""
    day = dt.day
    if 11 <= day <= 13:
        suffix = "th"
    elif day % 10 == 1:
        suffix = "st"
    elif day % 10 == 2:
        suffix = "nd"
    elif day % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"
    return f"{dt.strftime('%B')} {day}{suffix}"


# ─── Smoke test ─────────────────────────────────────────────────────────────
# uv run python -m backend.app.reasoner.drip

if __name__ == "__main__":
    import json

    b = DirectiveBuilder()

    print("=== Read-back rendering ===")
    print(f"  POL-2024-001 → {_spell_for_voice('POL-2024-001')}")
    print(f"  B-AL-1234    → {_spell_for_voice('B-AL-1234')}")
    print(f"  Templated:   {render_readback('policy_number', 'POL-2024-001')}")

    print()
    print("=== Directive serialization (control plane wire format) ===")

    examples = [
        b.load_policy_context(
            policy_brief="Caller is Anna Schmidt. Vehicle: VW Golf...",
            reason="DB lookup completed for plate B-AL-1234",
        ),
        b.speak(
            text="Okay let me pull that up for you...",
            reason="lead filler while async tool call runs",
        ),
        b.silent(
            duration_frames=25,  # 2 seconds
            reason="caller is mid-thought, give them space",
        ),
        b.speak(
            text=render_readback("license_plate", "B-AL-1234"),
            reason="forced readback: extractor confidence 0.72 on critical slot",
        ),
        b.rescue_clip(
            clip_id="breaking_up_1",
            reason="overlap detector + low SNR for >800ms",
        ),
        b.release(reason="caller finished tangent, return Sarah to free sampling"),
    ]

    for d in examples:
        wire = d.model_dump(mode="json")
        print(f"  seq={wire['seq']:2d} type={wire['type']:25s} reason={wire['reason']}")
        # Show full payload for one of them
    print()
    print("=== Full payload example (LoadPolicyContextDirective) ===")
    print(json.dumps(examples[0].model_dump(mode="json"), indent=2, default=str))
