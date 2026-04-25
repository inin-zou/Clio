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
        reason: str = "",
    ) -> SpeakDirective:
        return SpeakDirective(
            seq=self._next_seq(),
            text=text,
            after_release=after_release,
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
    """
    spaced = _spell_for_voice(value)
    return f"Okay so that's {spaced}, is that right?"


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
