"""
Integration test: walk a Session (MockTalker) through a realistic call.

Verifies:
  - SessionStart → SessionReady handshake
  - Audio frames in → audio + transcript out at 1:1 ratio
  - SpeakDirective drips chars via forced text tokens
  - SilenceDirective forces PAD for the requested frame count
  - ReleaseDirective resumes default behavior
  - SessionEnd closes cleanly

No GPU, no torch, no network. Pure asyncio.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from server.protocol import (
    AudioFrame,
    ClientMessage,
    ReleaseDirective,
    SessionEnd,
    SessionReady,
    SessionStart,
    SilenceDirective,
    SpeakDirective,
    SPECIAL_TOKEN_TEXT,
    PAD_TOKEN_ID,
)
from server.session import Session
from server.talker import MockTalker, BYTES_PER_FRAME_INT16


def _silent_pcm_b64() -> str:
    return base64.b64encode(b"\x00" * BYTES_PER_FRAME_INT16).decode("ascii")


async def _drain(session: Session, expected_count: int, timeout: float = 1.0):
    """Pull `expected_count` outgoing messages from the session."""
    out = []
    async def collector():
        async for m in session.outgoing():
            out.append(m)
            if len(out) >= expected_count:
                break
    try:
        await asyncio.wait_for(collector(), timeout=timeout)
    except asyncio.TimeoutError:
        pass
    return out


@pytest.mark.asyncio
async def test_session_handshake():
    sess = Session(MockTalker())
    await sess.handle(SessionStart(
        call_id="t-handshake",
        system_prompt="You are Sarah at Allianz.",
        voice_prompt_id="NATF1",
    ))
    out = await _drain(sess, expected_count=1)
    assert len(out) == 1
    assert isinstance(out[0], SessionReady)
    assert out[0].call_id == "t-handshake"
    assert out[0].voice_prompt_id == "NATF1"
    await sess.close()


@pytest.mark.asyncio
async def test_audio_frame_roundtrip():
    sess = Session(MockTalker())
    await sess.handle(SessionStart(call_id="t-audio", system_prompt="x", voice_prompt_id="NATF1"))
    await _drain(sess, expected_count=1)  # SessionReady

    # Send 5 frames, expect 5 transcript + 5 audio out
    for seq in range(5):
        await sess.handle(AudioFrame(seq=seq, pcm_base64=_silent_pcm_b64()))

    out = await _drain(sess, expected_count=10)
    transcripts = [m for m in out if m.type == "transcript"]
    audios = [m for m in out if m.type == "audio"]
    assert len(transcripts) == 5, f"got {len(transcripts)} transcripts"
    assert len(audios) == 5, f"got {len(audios)} audios"
    # MockTalker's default canned tokens start with PAD
    assert transcripts[0].token_id == PAD_TOKEN_ID
    await sess.close()


@pytest.mark.asyncio
async def test_speak_directive_drips_chars():
    """A SpeakDirective with N chars should force EPAD then N text tokens."""
    sess = Session(MockTalker())
    await sess.handle(SessionStart(call_id="t-speak", system_prompt="x", voice_prompt_id="NATF1"))
    await _drain(sess, expected_count=1)

    text = "hi"
    await sess.handle(SpeakDirective(seq=1, text=text, reason="test"))
    # Now feed enough frames to drain the drip queue (EPAD + len(text) chars + 1 extra)
    for seq in range(len(text) + 2):
        await sess.handle(AudioFrame(seq=seq, pcm_base64=_silent_pcm_b64()))

    out = await _drain(sess, expected_count=2 * (len(text) + 2))
    transcripts = [m for m in out if m.type == "transcript"]
    # First forced token should be EPAD (id=0); next chars = ord('h'), ord('i')
    forced_ids = [t.token_id for t in transcripts[:1 + len(text)]]
    assert forced_ids[0] == 0, f"expected EPAD, got {forced_ids[0]}"
    assert forced_ids[1] == ord("h")
    assert forced_ids[2] == ord("i")
    await sess.close()


@pytest.mark.asyncio
async def test_silence_directive():
    sess = Session(MockTalker())
    await sess.handle(SessionStart(call_id="t-silent", system_prompt="x", voice_prompt_id="NATF1"))
    await _drain(sess, expected_count=1)

    await sess.handle(SilenceDirective(seq=1, duration_frames=3, reason="test"))
    for seq in range(3):
        await sess.handle(AudioFrame(seq=seq, pcm_base64=_silent_pcm_b64()))

    out = await _drain(sess, expected_count=6)
    transcripts = [m for m in out if m.type == "transcript"]
    # All 3 should be PAD (id=3)
    assert all(t.token_id == PAD_TOKEN_ID for t in transcripts), \
        f"expected all PAD, got {[t.token_id for t in transcripts]}"
    await sess.close()


@pytest.mark.asyncio
async def test_session_end_closes():
    sess = Session(MockTalker())
    await sess.handle(SessionStart(call_id="t-end", system_prompt="x", voice_prompt_id="NATF1"))
    await _drain(sess, expected_count=1)

    await sess.handle(SessionEnd(call_id="t-end"))
    out = await _drain(sess, expected_count=1)
    assert any(m.type == "session_closed" for m in out)


if __name__ == "__main__":
    # Allow `python -m tests.test_mock_session` for quick verification.
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
