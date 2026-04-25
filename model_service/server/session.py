"""
Per-call session orchestrator.

Owns one Talker instance + the directive cache + the per-frame loop. The WS
server (main.py) creates one Session per WebSocket connection, feeds it
ClientMessage objects, and forwards the ServerMessages it emits back to backend.

Drip-feed implementation:
  When a SpeakDirective arrives, we tokenize the text and queue the IDs.
  Each subsequent frame consumes one token from the queue and forces it
  via Talker.step(forced_text_token_id=...). When the queue empties we
  switch to "free sampling" (None) unless after_release == "silent".

This file is Talker-implementation-agnostic — works with MockTalker locally
and PersonaPlexTalker on Modal.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator

from .protocol import (
    AudioFrame,
    ClientMessage,
    EPAD_TOKEN_ID,
    LoadPolicyContextDirective,
    PAD_TOKEN_ID,
    ReleaseDirective,
    RescueClipDirective,
    ServerAudioFrame,
    ServerError,
    ServerMessage,
    SessionClosed,
    SessionEnd,
    SessionReady,
    SessionStart,
    SilenceDirective,
    SpeakDirective,
    SPECIAL_TOKEN_TEXT,
    TranscriptToken,
)
from .talker import BYTES_PER_FRAME_INT16, StepResult, Talker


# ─── Drip-feed state ────────────────────────────────────────────────────────

@dataclass
class _DripState:
    """What the per-frame loop should force on the next text-token slot."""
    queue: deque[int] = field(default_factory=deque)
    after_release: str = "resume"  # "resume" | "silent"
    silent_frames_remaining: int = 0
    rescue_clip_id: str | None = None  # if set, override audio output

    def force_silence(self, frames: int) -> None:
        self.queue.clear()
        self.silent_frames_remaining = frames

    def force_speak(self, token_ids: list[int], after_release: str) -> None:
        # Always lead with EPAD to break out of any active PAD state.
        self.queue.clear()
        self.queue.append(EPAD_TOKEN_ID)
        self.queue.extend(token_ids)
        self.after_release = after_release
        self.silent_frames_remaining = 0

    def force_release(self) -> None:
        self.queue.clear()
        self.silent_frames_remaining = 0
        self.after_release = "resume"

    def next_forced_token(self) -> int | None:
        """Return the token to force on the next frame, or None to let the
        Talker sample freely."""
        if self.silent_frames_remaining > 0:
            self.silent_frames_remaining -= 1
            return PAD_TOKEN_ID
        if self.queue:
            return self.queue.popleft()
        if self.after_release == "silent":
            return PAD_TOKEN_ID
        return None


# ─── Tokenization shim (Mock vs real) ───────────────────────────────────────

def _tokenize_for_drip(text: str) -> list[int]:
    """Convert a string into a list of text-token IDs to drip-feed.

    For MockTalker we use one token per character (id = ord(c)) so backend
    can verify the drip got through end-to-end. The real PersonaPlex talker
    will use SentencePiece via PersonaPlex's tokenizer once that's wired up
    (see PersonaPlexTalker.step in talker.py).
    """
    return [ord(c) for c in text]


# ─── Session ────────────────────────────────────────────────────────────────

class Session:
    """One LiveKit-room-per-call worth of state.

    The WS handler in main.py instantiates this on connect, calls
    `handle(message)` for every ClientMessage, and forwards everything that
    comes out of `outgoing()` back over the socket.
    """

    def __init__(self, talker: Talker) -> None:
        self.talker = talker
        self.call_id: str | None = None
        self.frame_seq_in = 0
        self.frame_seq_out = 0
        self.token_seq_out = 0
        self._drip = _DripState()
        self._outgoing: asyncio.Queue[ServerMessage] = asyncio.Queue()
        self._initialized = False
        self._closed = False

    # ── public consumer interface ──

    async def handle(self, message: ClientMessage) -> None:
        """Process one inbound message. Side-effects: enqueues outgoing
        messages, updates internal state."""
        if isinstance(message, SessionStart):
            await self._handle_session_start(message)
        elif isinstance(message, AudioFrame):
            await self._handle_audio_frame(message)
        elif isinstance(message, SpeakDirective):
            self._drip.force_speak(_tokenize_for_drip(message.text), message.after_release)
        elif isinstance(message, SilenceDirective):
            self._drip.force_silence(message.duration_frames)
        elif isinstance(message, ReleaseDirective):
            self._drip.force_release()
        elif isinstance(message, RescueClipDirective):
            self._drip.rescue_clip_id = message.clip_id
            # TODO: actually load + stream the WAV. For MVP1 we just flag it.
        elif isinstance(message, LoadPolicyContextDirective):
            # The brief stays in the Reasoner's purview for drip-feeding via
            # subsequent SpeakDirectives. We just acknowledge here.
            pass
        elif isinstance(message, SessionEnd):
            await self.close(reason="client requested end")

    async def outgoing(self) -> AsyncIterator[ServerMessage]:
        """Async generator of messages to send to the client.

        Continues draining until a SessionClosed message has been yielded.
        This avoids the race where close() flips _closed=True before the
        consumer has had a chance to receive the SessionClosed itself.
        """
        while True:
            try:
                msg = await asyncio.wait_for(self._outgoing.get(), timeout=0.5)
            except asyncio.TimeoutError:
                if self._closed and self._outgoing.empty():
                    return
                continue
            yield msg
            if isinstance(msg, SessionClosed):
                return

    async def close(self, reason: str = "") -> None:
        if self._closed:
            return
        await self.talker.close()
        await self._outgoing.put(SessionClosed(call_id=self.call_id or "", reason=reason))
        self._closed = True

    # ── handlers ──

    async def _handle_session_start(self, m: SessionStart) -> None:
        self.call_id = m.call_id
        try:
            await self.talker.init_session(
                call_id=m.call_id,
                system_prompt=m.system_prompt,
                voice_prompt_id=m.voice_prompt_id,
            )
            self._initialized = True
            await self._outgoing.put(
                SessionReady(call_id=m.call_id, voice_prompt_id=m.voice_prompt_id)
            )
        except Exception as e:
            await self._outgoing.put(
                ServerError(
                    code="session_init_failed",
                    message=str(e),
                    fatal=True,
                )
            )
            await self.close(reason="init failed")

    async def _handle_audio_frame(self, m: AudioFrame) -> None:
        if not self._initialized:
            await self._outgoing.put(
                ServerError(
                    code="not_initialized",
                    message="Audio frame received before session_start",
                    fatal=False,
                )
            )
            return

        try:
            user_pcm = base64.b64decode(m.pcm_base64)
        except Exception as e:
            await self._outgoing.put(
                ServerError(code="bad_audio_base64", message=str(e))
            )
            return

        if len(user_pcm) != BYTES_PER_FRAME_INT16:
            await self._outgoing.put(
                ServerError(
                    code="bad_frame_size",
                    message=f"Expected {BYTES_PER_FRAME_INT16} bytes, got {len(user_pcm)}",
                )
            )
            return

        forced = self._drip.next_forced_token()
        result: StepResult = await self.talker.step(
            user_audio_int16=user_pcm,
            forced_text_token_id=forced,
        )

        # Emit transcript token for backend's slot extractor.
        self.token_seq_out += 1
        await self._outgoing.put(
            TranscriptToken(
                seq=self.token_seq_out,
                token_id=result.text_token_id,
                text=result.text_token_decoded,
                is_special=result.text_token_id in SPECIAL_TOKEN_TEXT,
            )
        )

        # Emit agent audio frame.
        self.frame_seq_out += 1
        await self._outgoing.put(
            ServerAudioFrame(
                seq=self.frame_seq_out,
                pcm_base64=base64.b64encode(result.audio_pcm_int16).decode("ascii"),
                is_rescue_clip=self._drip.rescue_clip_id is not None,
            )
        )
        # Single rescue-clip frame for MVP1; clear flag after emitting.
        if self._drip.rescue_clip_id is not None:
            self._drip.rescue_clip_id = None
