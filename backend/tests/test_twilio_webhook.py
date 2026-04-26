"""Twilio webhook smoke tests — no real Twilio, no real Modal, no real LiveKit.

What these cover:
  - POST /twilio/voice with mock form data registers a PendingCall
  - The Modal spawn function is invoked with the right kwargs
  - TwiML response includes <Dial><Sip> with the room name in the URI
  - Failure when LiveKit env is missing returns a polite TwiML hangup
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def env(monkeypatch):
    """Set env vars the webhook needs. AUTH_TOKEN absent + skip-flag on
    keeps signature validation off for tests."""
    # >= 32 bytes → no jwt InsecureKeyLengthWarning noise
    monkeypatch.setenv("LIVEKIT_API_KEY", "test_api_key_for_unit_tests_xx")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test_api_secret_thirty_two_bytes_min_xx")
    monkeypatch.setenv("LIVEKIT_SIP_URI", "test.sip.livekit.cloud")
    monkeypatch.setenv("BACKEND_PUBLIC_WS_URL", "wss://test.ngrok.app")
    monkeypatch.setenv("TWILIO_SKIP_SIGNATURE_VALIDATION", "1")
    monkeypatch.setenv("CLIO_SKIP_PREAMBLE", "1")  # cleaner TwiML in tests
    # Disable the hackathon fixed-room mode for these tests so each call
    # gets its own deterministic room name we can assert on.
    monkeypatch.setenv("CLIO_USE_FIXED_ROOM", "0")


@pytest.fixture
def mock_modal(monkeypatch):
    """Replace the lazy Modal handle with a mock that records spawn() calls."""
    from backend.app.telephony import twilio_webhook

    spawn_calls: list[dict] = []
    fake_process_call = MagicMock()
    fake_process_call.spawn = lambda **kw: spawn_calls.append(kw)

    monkeypatch.setattr(
        twilio_webhook, "_get_modal_process_call", lambda: fake_process_call
    )
    return spawn_calls


@pytest.fixture
def clean_pending():
    """Isolate PENDING_CALLS between tests by clearing the live registry
    (the webhook + server both read this same dict reference)."""
    from backend.app.control.registry import PENDING_CALLS

    PENDING_CALLS.clear()
    yield PENDING_CALLS
    PENDING_CALLS.clear()


@pytest.fixture
def client(env, mock_modal, clean_pending):
    from backend.app.control.server import app

    return TestClient(app)


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_voice_webhook_registers_pending_call(client, mock_modal, clean_pending):
    resp = client.post(
        "/twilio/voice",
        data={
            "CallSid": "CAtest1234",
            "From": "+4915123456789",
            "To": "+493012345678",
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(clean_pending) == 1
    call_id, pending = next(iter(clean_pending.items()))
    assert pending.livekit_room.startswith("clio-")
    assert pending.livekit_room == f"clio-{call_id}"
    assert len(pending.livekit_agent_token) > 100  # JWT-shaped


def test_voice_webhook_spawns_modal_with_right_kwargs(client, mock_modal):
    client.post("/twilio/voice", data={"CallSid": "CAtest", "From": "+49", "To": "+49"})
    assert len(mock_modal) == 1
    kw = mock_modal[0]
    assert set(kw) == {
        "call_id",
        "livekit_room",
        "livekit_agent_token",
        "backend_control_url",
    }
    assert kw["backend_control_url"] == "wss://test.ngrok.app"
    assert kw["livekit_room"] == f"clio-{kw['call_id']}"


def test_voice_webhook_returns_dial_sip_twiml(client):
    resp = client.post("/twilio/voice", data={"CallSid": "CAtest"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")

    root = ET.fromstring(resp.text)
    assert root.tag == "Response"
    sip = root.find(".//Sip")
    assert sip is not None, f"no <Sip> in TwiML: {resp.text}"
    assert sip.text.startswith("sip:clio-")
    assert "test.sip.livekit.cloud" in sip.text
    assert sip.text.endswith(";transport=tcp")


def test_voice_webhook_missing_backend_url_returns_failure_twiml(
    client, monkeypatch, clean_pending
):
    monkeypatch.delenv("BACKEND_PUBLIC_WS_URL", raising=False)
    resp = client.post("/twilio/voice", data={"CallSid": "CAtest"})
    assert resp.status_code == 200  # Twilio always wants 200 + TwiML
    root = ET.fromstring(resp.text)
    assert root.find(".//Hangup") is not None
    assert len(clean_pending) == 0  # nothing should have been registered


def test_status_webhook_returns_204(client):
    resp = client.post(
        "/twilio/status",
        data={
            "CallSid": "CAtest",
            "CallStatus": "completed",
            "CallDuration": "47",
        },
    )
    assert resp.status_code == 204


def test_signature_validation_rejects_when_token_missing(client, monkeypatch):
    """Without TWILIO_AUTH_TOKEN and without the skip flag, must refuse."""
    monkeypatch.delenv("TWILIO_SKIP_SIGNATURE_VALIDATION", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    resp = client.post("/twilio/voice", data={"CallSid": "CAtest"})
    assert resp.status_code == 500
