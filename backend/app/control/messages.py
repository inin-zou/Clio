"""
Wire-format types for the control-plane WebSocket between backend and the
Modal-hosted LiveKit Agent.

Direction matters:
  - Backend → Modal: directives the Reasoner wants the Talker to apply.
    These re-export the existing Pydantic models from reasoner/drip.py so
    we have one source of truth for directive shape.
  - Modal → Backend: transcript turns + session lifecycle events. Backend
    feeds these into the Reasoner Session.

Audio NEVER traverses this WebSocket. Audio path is LiveKit Cloud → Modal
direct via WebRTC. See architecture.md for the rationale.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# Backend → Modal directives are the same Pydantic models the Reasoner already
# emits. Re-exported here so control/messages.py is the single import surface.
from ..reasoner.drip import (
    LoadPolicyContextDirective,
    ReleaseDirective,
    RescueClipDirective,
    SilenceDirective,
    SpeakDirective,
)


# ─── Backend → Modal: session bootstrap ──────────────────────────────────────

class SessionStart(BaseModel):
    """Sent first by backend to tell Modal what call it's joining.

    Modal uses this to:
      - Associate the WebSocket connection with a call_id
      - Pass system_prompt to PersonaPlex via step_system_prompts()
      - Choose which voice prompt (.pt) to load
      - Know which LiveKit room to join (Modal already has API key/secret
        as env; only needs the room name + agent participant token)
    """
    type: Literal["session_start"] = "session_start"
    call_id: str
    livekit_room: str
    livekit_agent_token: str
    system_prompt: str
    voice_prompt_id: str = "NATF1"


class SessionEnd(BaseModel):
    type: Literal["session_end"] = "session_end"
    call_id: str
    reason: str = ""


# ─── Backend → Modal: full discriminated union ───────────────────────────────

BackendMessage = Annotated[
    Union[
        SessionStart,
        SessionEnd,
        SpeakDirective,
        SilenceDirective,
        ReleaseDirective,
        RescueClipDirective,
        LoadPolicyContextDirective,
    ],
    Field(discriminator="type"),
]


# ─── Modal → Backend: lifecycle ──────────────────────────────────────────────

class SessionReady(BaseModel):
    """Modal has loaded persona prompt + voice and is ready to take audio."""
    type: Literal["session_ready"] = "session_ready"
    call_id: str
    voice_prompt_id: str
    cold_start_seconds: float = 0.0  # for telemetry


class SessionClosed(BaseModel):
    type: Literal["session_closed"] = "session_closed"
    call_id: str
    reason: str = ""
    duration_seconds: float = 0.0


# ─── Modal → Backend: transcript stream ──────────────────────────────────────

class TranscriptTurn(BaseModel):
    """One transcript turn pushed from Modal to backend.

    Sources:
      - "personaplex": from PersonaPlex's text monologue stream (primary).
        Token-level granularity; backend aggregates contiguous tokens into
        readable phrases.
      - "scribe": from ElevenLabs Scribe v2 (always-on ASR for entity
        verification). Phrase-level; produces final transcripts as the
        caller pauses.
    """
    type: Literal["transcript"] = "transcript"
    call_id: str
    role: Literal["caller", "agent"]
    text: str
    source: Literal["personaplex", "scribe"] = "personaplex"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_final: bool = True  # Scribe partial transcripts may set this False


class CallerTurnBoundary(BaseModel):
    """Modal detected the caller stopped speaking (VAD silence > N or Scribe
    finalize signal). Backend uses this to trigger a slot extractor run.

    Without this, backend would extract on every token which is wasteful.
    """
    type: Literal["caller_turn_boundary"] = "caller_turn_boundary"
    call_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReadbackOutcome(BaseModel):
    """Modal observed the caller responding to a Sarah read-back.

    The outcome was determined by Modal's local heuristic on the Scribe
    transcript (caller said "yes/correct/right" → confirmed; "no/actually/
    I meant" → corrected; ambiguous → unclear). Backend records this in
    the session's readback log.
    """
    type: Literal["readback_outcome"] = "readback_outcome"
    call_id: str
    slot_path: str
    proposed_value: str
    caller_response: Literal["confirmed", "corrected", "unclear"]
    final_value: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─── Modal → Backend: full discriminated union ───────────────────────────────

ModalMessage = Annotated[
    Union[
        SessionReady,
        SessionClosed,
        TranscriptTurn,
        CallerTurnBoundary,
        ReadbackOutcome,
    ],
    Field(discriminator="type"),
]
