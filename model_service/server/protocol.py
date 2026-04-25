"""
WebSocket protocol between backend (LiveKit Agent) and model_service.

For MVP1 we use a single multiplexed WebSocket carrying both audio and
control messages, distinguished by `type`. This is simpler than the
two-channel design described in architecture.md, which we'll split into
separate WSes if head-of-line blocking becomes an issue under load.

Message types are JSON for control + base64-PCM for audio. Messages are
discriminated unions on the `type` field; both sides parse with Pydantic.

The directive shapes here mirror backend/app/reasoner/drip.py exactly.
The two modules are kept independent (no cross-package imports) so
model_service can be deployed in isolation from backend. JSON wire format
is the contract; if drip.py changes, this file must change too.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ─── Backend → model_service ─────────────────────────────────────────────────

class SessionStart(BaseModel):
    """First message in any session. Loads persona + voice prompt into
    PersonaPlex's KV cache via lm_gen.step_system_prompts()."""
    type: Literal["session_start"] = "session_start"
    call_id: str
    system_prompt: str  # full text including BASE_PERSONA + time block
    voice_prompt_id: str  # e.g. "NATF1" — matches a .pt file in assets/voices/
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AudioFrame(BaseModel):
    """One frame of caller audio. PCM mono 24kHz, 80ms = 1920 samples = 3840
    bytes (int16). Sent as base64 to keep JSON-only transport simple for MVP1.
    """
    type: Literal["audio"] = "audio"
    pcm_base64: str
    seq: int  # monotonic per session, for ordering / debugging


class SpeakDirective(BaseModel):
    type: Literal["speak"] = "speak"
    seq: int
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str
    after_release: Literal["resume", "silent"] = "resume"
    reason: str = ""


class SilenceDirective(BaseModel):
    type: Literal["silent"] = "silent"
    seq: int
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_frames: int = Field(ge=1, le=125)
    reason: str = ""


class ReleaseDirective(BaseModel):
    type: Literal["release"] = "release"
    seq: int
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""


class RescueClipDirective(BaseModel):
    type: Literal["rescue_clip"] = "rescue_clip"
    seq: int
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    clip_id: str
    reason: str = ""


class LoadPolicyContextDirective(BaseModel):
    type: Literal["load_policy_context"] = "load_policy_context"
    seq: int
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    policy_brief: str
    reason: str = ""


class SessionEnd(BaseModel):
    type: Literal["session_end"] = "session_end"
    call_id: str


ClientMessage = Annotated[
    Union[
        SessionStart,
        AudioFrame,
        SpeakDirective,
        SilenceDirective,
        ReleaseDirective,
        RescueClipDirective,
        LoadPolicyContextDirective,
        SessionEnd,
    ],
    Field(discriminator="type"),
]


# ─── model_service → Backend ─────────────────────────────────────────────────

class ServerAudioFrame(BaseModel):
    """Outgoing audio frame from PersonaPlex (Sarah's voice) or rescue clip."""
    type: Literal["audio"] = "audio"
    pcm_base64: str
    seq: int
    is_rescue_clip: bool = False  # backend may want to know


class TranscriptToken(BaseModel):
    """One token emitted on PersonaPlex's text monologue stream. Backend
    accumulates these into Sarah's side of the transcript and feeds the
    Reasoner's slot extractor.
    """
    type: Literal["transcript"] = "transcript"
    seq: int
    token_id: int  # raw vocab id (0=EPAD, 1=BOS, 2=EOS, 3=PAD)
    text: str  # decoded form ("hello", "▁", "EPAD", etc.)
    is_special: bool = False  # True if special control token


class ServerError(BaseModel):
    """Something went wrong; backend may decide to play a rescue clip or
    end the call gracefully."""
    type: Literal["error"] = "error"
    code: str
    message: str
    fatal: bool = False


class SessionReady(BaseModel):
    """Sent in response to SessionStart once persona + voice are loaded."""
    type: Literal["session_ready"] = "session_ready"
    call_id: str
    voice_prompt_id: str


class SessionClosed(BaseModel):
    type: Literal["session_closed"] = "session_closed"
    call_id: str
    reason: str = ""


ServerMessage = Annotated[
    Union[
        ServerAudioFrame,
        TranscriptToken,
        ServerError,
        SessionReady,
        SessionClosed,
    ],
    Field(discriminator="type"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────
# Special token ids match PersonaPlex's offline.py:293 mapping.
EPAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3
SPECIAL_TOKEN_TEXT = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}
