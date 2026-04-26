"""Twilio inbound voice webhook + status callback.

Wire-up
-------
Twilio Console: Phone Number → Voice & Fax → "A Call Comes In":
    HTTP POST  https://<BACKEND_PUBLIC_HOST>/twilio/voice
Status Callback URL (optional but recommended):
    HTTP POST  https://<BACKEND_PUBLIC_HOST>/twilio/status

Flow
----
1. Caller dials Twilio DID → Twilio POSTs /twilio/voice with form fields
   (CallSid, From, To, ...).
2. We mint a unique call_id, derive room_name = "clio-<call_id>", and a
   short-lived AccessToken Modal will use to join that room.
3. We register the (call_id → room/token) into PENDING_CALLS so the
   control-plane WS endpoint can pop it when Modal connects back.
4. We call Modal `process_call.spawn(call_id, room, token, backend_ws_url)`
   to start the GPU container joining that room. spawn() returns
   immediately — the container starts async on Modal's side.
5. We return TwiML: <Say> a brief preamble (covers cold-start latency),
   then <Dial><Sip> the SIP URI whose user-part is the room name. LiveKit
   Cloud's inbound dispatch rule routes the caller to the matching room.

Failure modes
-------------
- LiveKit env missing → 500. Fail loud at webhook time, not after the
  caller pays the connect cost.
- Modal lookup fails → log + still return TwiML with a "we're having
  trouble" Say + Hangup. Avoids a confused dead-air call.
- Twilio signature invalid (when validation enabled) → 403.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from twilio.request_validator import RequestValidator
from twilio.twiml.voice_response import VoiceResponse

from ..control.registry import PendingCall, register_pending_call
from .livekit_sip import mint_call_setup

logger = logging.getLogger("clio.telephony")

router = APIRouter(prefix="/twilio", tags=["telephony"])


# ─── Modal handle (lazy) ─────────────────────────────────────────────────────
# Module-level cache so we don't re-resolve the deployed app on every call.
# Cls.from_name() is a network round-trip to Modal's control plane.

_modal_cls = None


def _get_modal_process_call():
    """Resolve the deployed Modal class + return its bound process_call.

    Done lazily so importing this module doesn't crash CI (where Modal
    creds aren't set). Webhook tests monkeypatch this whole function.
    """
    global _modal_cls
    if _modal_cls is None:
        import modal

        app_name = os.environ.get("MODAL_APP_NAME", "personaplex-clio")
        cls_name = os.environ.get("MODAL_CLS_NAME", "PersonaPlexService")
        _modal_cls = modal.Cls.from_name(app_name, cls_name)
        logger.info("resolved Modal Cls %s/%s", app_name, cls_name)
    return _modal_cls().process_call


# ─── Twilio signature validation ─────────────────────────────────────────────


async def _validate_twilio_signature(request: Request, form: dict[str, Any]) -> None:
    if os.environ.get("TWILIO_SKIP_SIGNATURE_VALIDATION") == "1":
        return
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not auth_token:
        # Misconfigured backend — refuse rather than accept unauthenticated
        # webhooks. Hint at the dev-time bypass.
        raise HTTPException(
            status_code=500,
            detail="TWILIO_AUTH_TOKEN unset; "
                   "set TWILIO_SKIP_SIGNATURE_VALIDATION=1 to bypass for dev",
        )
    signature = request.headers.get("X-Twilio-Signature", "")
    validator = RequestValidator(auth_token)
    # Twilio signs the FULL request URL (incl. query string). We reconstruct
    # from request.url; behind ngrok this matches if the public URL is what
    # you registered in Twilio Console.
    if not validator.validate(str(request.url), form, signature):
        logger.warning("Twilio signature mismatch for %s", request.url)
        raise HTTPException(status_code=403, detail="invalid Twilio signature")


# ─── Inbound voice ───────────────────────────────────────────────────────────


@router.post("/voice")
async def voice_webhook(request: Request) -> Response:
    """Twilio POSTs here when a call hits the DID."""
    form_data = await request.form()
    form = {k: str(v) for k, v in form_data.items()}
    await _validate_twilio_signature(request, form)

    twilio_call_sid = form.get("CallSid", "")
    caller_from = form.get("From", "")
    logger.info(
        "inbound call: CallSid=%s From=%s",
        twilio_call_sid, caller_from,
    )

    backend_ws_url = os.environ.get("BACKEND_PUBLIC_WS_URL")
    if not backend_ws_url:
        logger.error("BACKEND_PUBLIC_WS_URL not set — Modal can't connect back")
        return _twiml_failure("our system is unavailable, please try again later")

    # 1. Mint room + token for this call.
    try:
        setup = mint_call_setup()
    except RuntimeError as e:
        logger.error("LiveKit config missing: %s", e)
        return _twiml_failure("our system is unavailable, please try again later")

    # 2. Register into the in-process pending registry. Modal will pop this
    #    when it opens the control-plane WS shortly.
    register_pending_call(
        PendingCall(
            call_id=setup.call_id,
            livekit_room=setup.room_name,
            livekit_agent_token=setup.agent_token,
            voice_prompt_id=os.environ.get("CLIO_VOICE_PROMPT_ID", "NATF1"),
        )
    )

    # 3. Kick off the Modal container. spawn() returns a FunctionCall handle
    #    immediately; the actual @modal.method runs out-of-band on the GPU.
    try:
        process_call = _get_modal_process_call()
        process_call.spawn(
            call_id=setup.call_id,
            livekit_room=setup.room_name,
            livekit_agent_token=setup.agent_token,
            backend_control_url=backend_ws_url,
        )
        logger.info(
            "spawned Modal process_call for call_id=%s room=%s",
            setup.call_id, setup.room_name,
        )
    except Exception as e:
        logger.exception("failed to spawn Modal process_call: %s", e)
        return _twiml_failure("our system is having trouble, please try again later")

    # 4. Return TwiML: preamble (covers cold-start window) + SIP dial.
    # We pad the preamble because even with a warm GPU container there's
    # ~3-8s of network + LiveKit join + persona priming. With a cold start
    # the gap is 30s+; the warm-container path needs `min_containers=1` set
    # at deploy time via CLIO_DEMO_MODE=1.
    response = VoiceResponse()
    if os.environ.get("CLIO_SKIP_PREAMBLE") != "1":
        response.say(
            "Thank you for calling Allianz Claims.",
            voice="Polly.Joanna-Neural",
            language="en-US",
        )
        response.pause(length=1)
        response.say(
            "Please hold while I connect you to a claims representative.",
            voice="Polly.Joanna-Neural",
            language="en-US",
        )
        response.pause(length=2)
        response.say(
            "Connecting you now.",
            voice="Polly.Joanna-Neural",
            language="en-US",
        )
    # `timeout=60` keeps the SIP dial trying for up to a minute before
    # giving up. Default is 30s which can be tight on cold starts.
    dial = response.dial(timeout=60)
    dial.sip(setup.sip_destination)
    return Response(
        content=str(response),
        media_type="application/xml",
    )


# ─── Status callback ─────────────────────────────────────────────────────────


@router.post("/status")
async def status_webhook(request: Request) -> Response:
    """Twilio posts call-state updates here. We mostly use 'completed' to
    log; the WS disconnect from Modal handles actual session teardown."""
    form_data = await request.form()
    form = {k: str(v) for k, v in form_data.items()}
    await _validate_twilio_signature(request, form)

    call_sid = form.get("CallSid", "")
    call_status = form.get("CallStatus", "")
    duration = form.get("CallDuration", "")
    logger.info(
        "call status: CallSid=%s status=%s duration=%ss",
        call_sid, call_status, duration,
    )
    return Response(status_code=204)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _twiml_failure(message: str) -> Response:
    """Return TwiML that politely fails the call. Used when something
    upstream of the SIP dial is broken; avoids leaving the caller in
    silence wondering if they got through."""
    response = VoiceResponse()
    response.say(message, voice="Polly.Joanna-Neural", language="en-US")
    response.hangup()
    return Response(content=str(response), media_type="application/xml")
