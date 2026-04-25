"""
Talker abstraction — the audio-side model.

Two implementations:
  - PersonaPlexTalker — real PersonaPlex 7B inference. Only runs on Modal
    (or anywhere else with a CUDA-capable GPU + the moshi/personaplex deps
    installed via the [modal] extra).
  - MockTalker — emits silent audio + cycles through canned text tokens.
    Used for local dev so backend/livekit_agent.py and the WebSocket
    protocol can be developed without a GPU.

The interface is per-frame (12.5 Hz / 80 ms) and matches PersonaPlex's
LMGen.step() contract: feed one user-audio frame, optionally force a
text token, get one agent-audio frame + one agent-text token back.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .protocol import EPAD_TOKEN_ID, PAD_TOKEN_ID, SPECIAL_TOKEN_TEXT


# Frame parameters mirroring PersonaPlex / Mimi codec
SAMPLE_RATE = 24000  # Hz
FRAME_DURATION_S = 0.08  # 80 ms = 12.5 Hz
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_S)  # 1920
BYTES_PER_FRAME_INT16 = SAMPLES_PER_FRAME * 2  # 3840


@dataclass
class StepResult:
    """Output of one Talker.step() — Sarah's audio frame + her chosen text token."""
    audio_pcm_int16: bytes  # 3840 bytes (1920 int16 samples) at 24kHz mono
    text_token_id: int
    text_token_decoded: str  # "▁hello" or "EPAD" etc.


class Talker(Protocol):
    """Abstract per-frame voice model. PersonaPlex implementation goes on
    Modal; mock for local dev."""

    async def init_session(
        self, call_id: str, system_prompt: str, voice_prompt_id: str
    ) -> None:
        """Called once at session start. Loads persona + voice into model state."""
        ...

    async def step(
        self,
        user_audio_int16: bytes,
        forced_text_token_id: int | None = None,
    ) -> StepResult:
        """One frame of inference. Pass user_audio_int16 (3840 bytes / 1920 samples).
        If forced_text_token_id is not None, the model is teacher-forced on
        the text monologue stream — see PersonaPlex/lm.py:776-892."""
        ...

    async def close(self) -> None:
        """Release any model state (CUDA tensors, KV cache, etc.)."""
        ...


# ─── MockTalker — local dev / WS protocol testing ───────────────────────────

class MockTalker:
    """Drop-in Talker that returns silence + cycles through PAD/EPAD/text tokens.

    No torch, no GPU, no model load. Used to develop and integration-test the
    WebSocket protocol + Reasoner orchestration without GPU.

    Behavior:
      - Audio out: zeros (silence) at the correct frame size
      - Text out by default: PAD (silence)
      - When `forced_text_token_id` is provided, echoes it back (so backend can
        verify drip-feed got through)
      - Otherwise, alternates a few canned tokens to give the test something to
        observe (decoded as "...", "okay", "hmm" etc.)
    """

    # Canned tokens for when the model isn't being teacher-forced.
    # IDs are arbitrary (anything outside 0-3 is fine) — only the decoded text
    # matters for backend's transcript and the slot extractor.
    _CANNED_TOKENS: list[tuple[int, str]] = [
        (PAD_TOKEN_ID, "PAD"),
        (PAD_TOKEN_ID, "PAD"),
        (PAD_TOKEN_ID, "PAD"),
        (1024, "▁mm"),
        (1025, "▁hmm"),
        (PAD_TOKEN_ID, "PAD"),
        (PAD_TOKEN_ID, "PAD"),
    ]

    def __init__(self) -> None:
        self._call_id: str | None = None
        self._idx = 0

    async def init_session(
        self, call_id: str, system_prompt: str, voice_prompt_id: str
    ) -> None:
        self._call_id = call_id
        self._idx = 0
        # Simulate session-init load time so backend tests realistic timing
        await asyncio.sleep(0.05)

    async def step(
        self,
        user_audio_int16: bytes,
        forced_text_token_id: int | None = None,
    ) -> StepResult:
        if forced_text_token_id is not None:
            decoded = SPECIAL_TOKEN_TEXT.get(
                forced_text_token_id, f"<id={forced_text_token_id}>"
            )
            return StepResult(
                audio_pcm_int16=_silent_frame(),
                text_token_id=forced_text_token_id,
                text_token_decoded=decoded,
            )

        token_id, decoded = self._CANNED_TOKENS[self._idx % len(self._CANNED_TOKENS)]
        self._idx += 1
        return StepResult(
            audio_pcm_int16=_silent_frame(),
            text_token_id=token_id,
            text_token_decoded=decoded,
        )

    async def close(self) -> None:
        self._call_id = None


def _silent_frame() -> bytes:
    """1920 samples of int16 zeros = 3840 bytes."""
    return np.zeros(SAMPLES_PER_FRAME, dtype=np.int16).tobytes()


# ─── PersonaPlexTalker — Modal-only (lazy import of torch/moshi) ────────────

class PersonaPlexTalker:
    """Real PersonaPlex 7B inference. Only works where torch + moshi are
    installed (i.e. inside the Modal image — see deploy/modal_app.py).

    Implementation deferred until we have the Modal-side image building.
    The class is a stub here so the rest of the server can import it without
    pulling in torch on dev machines.
    """

    def __init__(self) -> None:
        self._lm_gen = None  # LMGen — set in init_session
        self._mimi = None
        self._other_mimi = None
        self._tokenizer = None

    async def init_session(
        self, call_id: str, system_prompt: str, voice_prompt_id: str
    ) -> None:
        try:
            import torch  # noqa: F401
            from moshi.models import LMGen, loaders  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "PersonaPlexTalker requires the [modal] extra (torch + moshi). "
                "Use MockTalker for local dev, or install on a GPU machine."
            ) from e

        # TODO: implement the actual loading sequence mirroring
        #   personaplex/moshi/moshi/offline.py:run_inference (lines 153–256):
        #   - load Mimi codecs
        #   - load tokenizer
        #   - load LM
        #   - construct LMGen with audio_silence_frame_cnt etc.
        #   - mimi.streaming_forever(1); lm_gen.streaming_forever(1)
        #   - warmup
        #   - load voice_prompt by id
        #   - tokenize system_prompt, set lm_gen.text_prompt_tokens
        #   - call lm_gen.step_system_prompts(mimi)
        # For now we raise to make accidental local use loud.
        raise NotImplementedError(
            "PersonaPlexTalker.init_session: implement after Modal image is up. "
            "See deploy/modal_app.py for the deployment story."
        )

    async def step(
        self,
        user_audio_int16: bytes,
        forced_text_token_id: int | None = None,
    ) -> StepResult:
        # TODO: actual implementation:
        #   1. Decode user_audio_int16 to torch.Tensor [1, 1, 1920], float32
        #   2. mimi.encode → [1, K_user_codebooks, 1] codes
        #   3. forced_tensor = torch.tensor([forced_text_token_id], device=...) if not None
        #   4. tokens = lm_gen.step(input_tokens=user_codes, text_token=forced_tensor)
        #   5. agent_audio = mimi.decode(tokens[:, 1:9])
        #   6. text_id = tokens[0, 0, 0].item()
        #   7. text_decoded = tokenizer.id_to_piece(text_id) or special-name
        #   8. return StepResult(...)
        raise NotImplementedError("PersonaPlexTalker.step")

    async def close(self) -> None:
        # Free CUDA tensors / streaming state
        self._lm_gen = None
        self._mimi = None
        self._other_mimi = None
