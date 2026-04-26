"""
Control-plane WebSocket server.

Backend exposes ws://host:port/control/{call_id} for Modal's LiveKit Agent
to connect to. One CallOrchestrator instance per connection. Audio path is
elsewhere (LiveKit Cloud → Modal direct via WebRTC); only directives and
transcripts traverse this WS.

Usage
-----
Local dev (FastAPI + uvicorn):
    uv run uvicorn backend.app.control.server:app --reload --port 8000

For end-to-end with Modal, expose this service on a public URL (e.g. via
ngrok during dev, Render/Vercel for production), then set the URL in
Modal's environment so its process_call() WS-clients-out to here.

Pending-calls registry
----------------------
Twilio webhook → backend mints LiveKit room + token → adds to PENDING_CALLS
→ spawns Modal process_call.spawn() → Modal opens WS to /control/{call_id}
→ this server pops PENDING_CALLS[call_id] for the setup info → constructs
CallOrchestrator and runs the message loop.

For hackathon scope, in-memory dict is fine. Production would use Redis or
similar so multiple backend replicas coordinate.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .orchestrator import CallOrchestrator
from .registry import (
    ACTIVE_CALLS,
    PENDING_CALLS,
    PendingCall,
    register_pending_call,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("clio.control.server")


# ─── Pending call registry ───────────────────────────────────────────────────
# PendingCall, PENDING_CALLS, ACTIVE_CALLS, and register_pending_call live in
# .registry so backend/app/telephony/twilio_webhook.py can populate them
# without a circular import on this server module. Re-exported above.

# Where session JSON gets written when calls end.
SESSION_OUTPUT_DIR = Path(
    os.environ.get("CLIO_SESSION_OUTPUT_DIR", "data/sessions")
)


# ─── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Clio Control Plane")

# Permissive CORS for local dev / monitoring frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "pending_calls": len(PENDING_CALLS),
        "active_calls": len(ACTIVE_CALLS),
    }


@app.get("/calls")
async def list_calls() -> dict:
    """Monitoring endpoint — what's currently running."""
    return {
        "pending": list(PENDING_CALLS.keys()),
        "active": [
            {"call_id": cid, **orch.status()}
            for cid, orch in ACTIVE_CALLS.items()
        ],
    }


@app.get("/calls/{call_id}")
async def get_call(call_id: str) -> dict:
    orch = ACTIVE_CALLS.get(call_id)
    if not orch:
        raise HTTPException(status_code=404, detail="call not active")
    return orch.status()


@app.websocket("/control/{call_id}")
async def control_ws(websocket: WebSocket, call_id: str) -> None:
    """One control-plane connection per call.

    Modal's process_call opens a client to this endpoint right after it
    connects to the LiveKit room. The lookup by call_id provides the
    setup data Modal needs (room name, agent token, voice prompt).
    """
    setup = PENDING_CALLS.pop(call_id, None)
    if setup is None:
        # Could happen if Modal lags or multiple containers race for the
        # same call_id. Reject loudly.
        await websocket.close(code=1008, reason=f"unknown call_id={call_id}")
        return

    await websocket.accept()
    orch = CallOrchestrator(
        ws=websocket,
        call_id=call_id,
        livekit_room=setup.livekit_room,
        livekit_agent_token=setup.livekit_agent_token,
        voice_prompt_id=setup.voice_prompt_id,
        output_dir=SESSION_OUTPUT_DIR,
    )
    ACTIVE_CALLS[call_id] = orch

    try:
        # Send the setup (system_prompt, livekit room, voice) immediately.
        await orch.send_session_start()

        # Inbound message loop.
        while True:
            raw = await websocket.receive_text()
            await orch.handle_raw_text(raw)

    except WebSocketDisconnect:
        logger.info("call %s: WebSocket disconnected", call_id)
    except Exception as e:
        logger.exception("call %s: unexpected error: %s", call_id, e)
    finally:
        await orch.close(reason="ws disconnected")
        ACTIVE_CALLS.pop(call_id, None)


# ─── Dev / smoke entry ───────────────────────────────────────────────────────

@app.post("/dev/register-mock-call")
async def register_mock_call(body: dict) -> dict:
    """Dev convenience — push a fake pending call so a test client can
    connect on /control/{call_id} without a real Twilio webhook."""
    setup = PendingCall(
        call_id=body["call_id"],
        livekit_room=body.get("livekit_room", "mock-room"),
        livekit_agent_token=body.get("livekit_agent_token", "mock-token"),
        voice_prompt_id=body.get("voice_prompt_id", "NATF1"),
    )
    register_pending_call(setup)
    return {"ok": True, "call_id": setup.call_id}


# ─── Telephony router ────────────────────────────────────────────────────────
# Late import: telephony.twilio_webhook depends on PendingCall +
# register_pending_call defined above. By the time this line runs, both
# symbols exist in this module's namespace, so the circular import resolves.
from ..telephony.twilio_webhook import router as twilio_router  # noqa: E402

app.include_router(twilio_router)

# ─── SSE / monitoring UI ─────────────────────────────────────────────────────

from .sse import router as sse_router  # noqa: E402

app.include_router(sse_router)
