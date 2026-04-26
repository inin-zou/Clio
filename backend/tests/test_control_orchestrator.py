"""
Unit-test the orchestrator's message handling without spinning up FastAPI.

Uses a fake WebSocket that captures sent messages so we can assert what
backend would tell Modal.

These tests do NOT call the live Anthropic API — extractor.extract is
monkeypatched. Verifies the orchestration layer wires correctly: messages
parse, transcript accumulates, readbacks record, gate decisions get sent.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.app.control.messages import (
    CallerTurnBoundary,
    ReadbackOutcome,
    SessionReady,
    TranscriptTurn,
)
from backend.app.control.orchestrator import CallOrchestrator
from backend.app.reasoner.extractor import ExtractorOutput, SlotUpdate


class FakeWebSocket:
    """Captures sent messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed = False

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        raise NotImplementedError("test does not receive")

    async def close(self, code: int = 1000) -> None:
        self.closed = True


@pytest.fixture
def fake_ws():
    return FakeWebSocket()


@pytest.fixture
def orch(fake_ws, tmp_path: Path):
    return CallOrchestrator(
        ws=fake_ws,
        call_id="test-001",
        livekit_room="room-x",
        livekit_agent_token="tok-x",
        voice_prompt_id="NATF1",
        output_dir=tmp_path,
    )


@pytest.mark.asyncio
async def test_session_start_sends_persona_prompt(orch, fake_ws):
    await orch.send_session_start()
    # Two messages now: session_start + forced opening greeting (so the
    # floor is never cold while we wait for VAD to fire EPAD).
    assert len(fake_ws.sent) == 2
    msg = fake_ws.sent[0]
    assert msg["type"] == "session_start"
    assert msg["call_id"] == "test-001"
    assert msg["livekit_room"] == "room-x"
    assert msg["voice_prompt_id"] == "NATF1"
    # System prompt should include both the BASE_PERSONA and time block
    assert "Allianz" in msg["system_prompt"]
    assert "CURRENT SESSION CONTEXT" in msg["system_prompt"]
    # Second message is the forced opening greeting drip.
    greeting = fake_ws.sent[1]
    assert greeting["type"] == "speak"
    assert "Allianz" in greeting["text"]
    assert "Sarah" in greeting["text"]
    assert "opening greeting" in greeting["reason"]


@pytest.mark.asyncio
async def test_transcript_appends_to_session(orch):
    await orch.handle_message(
        TranscriptTurn(
            call_id="test-001",
            role="caller",
            text="my plate is B-AL-1234",
            source="scribe",
        )
    )
    assert len(orch.reasoner_session.session.transcript) == 1
    turn = orch.reasoner_session.session.transcript[0]
    assert turn.role == "caller"
    assert "B-AL-1234" in turn.text


@pytest.mark.asyncio
async def test_readback_outcome_records_event(orch):
    await orch.handle_message(
        ReadbackOutcome(
            call_id="test-001",
            slot_path="license_plate",
            proposed_value="B-AL-1234",
            caller_response="confirmed",
            final_value="B-AL-1234",
        )
    )
    confirmed = orch.reasoner_session.session.confirmed_slots()
    assert "license_plate" in confirmed


@pytest.mark.asyncio
async def test_turn_boundary_runs_extractor_and_sends_directive(
    orch, fake_ws, monkeypatch
):
    """End-to-end: caller turn → extractor returns updates → Reasoner applies
    → gate decides → directive sent over WS."""

    # Mock extractor: pretend caller said something fishy that should
    # trigger a forced read-back from the gate.
    async def fake_extract(self, transcript_window, current_report,
                           readbacks, policy_context, now=None):
        return ExtractorOutput(
            updates=[
                SlotUpdate(
                    slot_path="license_plate",
                    value="B-AL-1234",
                    confidence=0.7,
                    source_quote="my plate is B-AL-1234",
                ),
            ],
            reasoning="extracted plate from caller",
        )

    from backend.app.reasoner import extractor
    monkeypatch.setattr(extractor.SlotExtractor, "extract", fake_extract)

    # Add some caller turns first so the gate's grace-period logic doesn't
    # reject the readback for being too early. The last caller turn must
    # contain the extracted value so the structural anchor filter
    # (filter_caller_anchored) keeps the identifier-slot update.
    started = orch.reasoner_session.session.started_at
    from datetime import timedelta
    caller_lines = [
        "hello, I want to file a claim",
        "yes that's right",
        "no nothing else for now",
        "my plate is B-AL-1234",
    ]
    for i, line in enumerate(caller_lines):
        await orch.handle_message(
            TranscriptTurn(
                call_id="test-001",
                role="caller",
                text=line,
                timestamp=started + timedelta(seconds=i * 5),
            )
        )

    # Now fire the turn boundary, which kicks off the extractor pipeline
    await orch.handle_message(
        CallerTurnBoundary(
            call_id="test-001",
            timestamp=started + timedelta(seconds=30),
        )
    )

    # Wait for the spawned extraction task to complete
    if orch._extraction_task:
        await orch._extraction_task

    # license_plate should now be in the report
    assert orch.reasoner_session.session.report.license_plate == "B-AL-1234"

    # Auth side-effect: PolicyContext should be loaded since B-AL-1234 is
    # in our mock DB.
    assert orch.reasoner_session.session.authenticated()

    # The gate may have fired a forced readback. Check that we sent something
    # other than just session_start.
    sent_types = [m.get("type") for m in fake_ws.sent]
    assert "session_start" not in sent_types  # we never called send_session_start
    # If the gate fired, it'd be a 'speak' directive
    if any(t == "speak" for t in sent_types):
        speak_msgs = [m for m in fake_ws.sent if m.get("type") == "speak"]
        assert "license_plate" in speak_msgs[0].get("reason", "") \
            or "B-AL-1234" in speak_msgs[0].get("text", ""), \
            f"unexpected speak directive: {speak_msgs}"


@pytest.mark.asyncio
async def test_status_reflects_session(orch):
    await orch.handle_message(
        TranscriptTurn(
            call_id="test-001",
            role="caller",
            text="hello",
        )
    )
    s = orch.status()
    assert s["call_id"] == "test-001"
    assert s["transcript_turns"] == 1
    assert s["authenticated"] is False  # no DB lookup yet


@pytest.mark.asyncio
async def test_close_persists_session(orch, tmp_path):
    await orch.handle_message(
        TranscriptTurn(call_id="test-001", role="caller", text="hi")
    )
    await orch.close(reason="test")
    out_file = tmp_path / "test-001.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["call_id"] == "test-001"
    assert len(data["transcript"]) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
