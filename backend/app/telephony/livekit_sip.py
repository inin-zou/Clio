"""LiveKit SIP plumbing: room naming + agent participant token minting.

The webhook flow:
  1. Caller dials Twilio DID.
  2. Webhook derives a deterministic room_name = "clio-<call_id>".
  3. Webhook mints a short-lived AccessToken so Modal can join that room
     as the "sarah-agent" participant.
  4. Webhook returns TwiML telling Twilio to <Dial><Sip> into LiveKit Cloud,
     putting the user-part of the SIP URI = room_name. LiveKit's inbound
     SIP dispatch rule routes the caller to that room.
  5. Modal joins the same room (via the JWT) and PersonaPlex talks to the
     caller end-to-end over WebRTC.

The agent's SIP URI for inbound calls is configured on the LiveKit side; we
read it from $LIVEKIT_SIP_URI. See .claude/docs/livekit-sip-setup.md for the
one-time `livekit-cli` commands to wire up the trunk + dispatch rule.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import timedelta

from livekit.api import AccessToken, VideoGrants

# Agent participant identity inside the LiveKit room. PersonaPlex publishes
# its audio track under this identity; the caller (SIP participant) gets
# auto-named by LiveKit.
AGENT_IDENTITY = "sarah-agent"
AGENT_NAME = "Sarah"

# Token lifetime — long enough for a 1h call, short enough to be useless if
# leaked from Modal logs.
AGENT_TOKEN_TTL = timedelta(hours=2)

ROOM_NAME_PREFIX = "clio-"

# Hackathon mode: when CLIO_USE_FIXED_ROOM=1 (default), every call lands
# in CLIO_FIXED_ROOM_NAME (default "clio-active"). Required because
# LiveKit's `dispatchRuleDirect` routes all inbound SIP calls to one
# named room — the user-part of the SIP URI is ignored. For multi-call
# concurrency we'd need a dispatch rule that derives the room name from
# the SIP attributes, plus matching webhook-side logic.
DEFAULT_FIXED_ROOM_NAME = "clio-active"


@dataclass(frozen=True)
class CallSetup:
    """Everything the Twilio webhook produces and passes downstream."""

    call_id: str
    room_name: str
    agent_token: str
    sip_destination: str  # e.g. "sip:clio-abc123@sip.livekit.cloud"


def new_call_id() -> str:
    """16 hex chars of entropy — unique per inbound call. Used as the
    LiveKit room suffix and the WS path component."""
    return secrets.token_hex(8)


def mint_call_setup(
    call_id: str | None = None,
    *,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    livekit_sip_uri: str | None = None,
) -> CallSetup:
    """Produce the full setup the webhook needs to hand to Modal + Twilio.

    Reads keys from env unless overridden (override path is for tests).
    Raises RuntimeError on missing config so the webhook fails loud rather
    than silently issuing tokens against the wrong project.
    """
    api_key = livekit_api_key or os.environ.get("LIVEKIT_API_KEY")
    api_secret = livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET")
    sip_uri = livekit_sip_uri or os.environ.get("LIVEKIT_SIP_URI")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set")
    if not sip_uri:
        raise RuntimeError("LIVEKIT_SIP_URI not set (e.g. sip.livekit.cloud)")

    cid = call_id or new_call_id()
    # Fixed room when LiveKit's dispatch rule is `Direct` with a single
    # roomName. Per-call rooms only work if dispatch rule extracts the
    # room name from SIP attributes — not the case for the simple
    # hackathon setup. Read env at call time so tests can override.
    use_fixed = os.environ.get("CLIO_USE_FIXED_ROOM", "1") == "1"
    fixed = os.environ.get("CLIO_FIXED_ROOM_NAME", DEFAULT_FIXED_ROOM_NAME)
    room_name = fixed if use_fixed else f"{ROOM_NAME_PREFIX}{cid}"

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(AGENT_IDENTITY)
        .with_name(AGENT_NAME)
        .with_ttl(AGENT_TOKEN_TTL)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    # Twilio's <Sip> URI: user-part = room name. LiveKit's inbound dispatch
    # rule (configured once via livekit-cli) parses the user-part and routes
    # the caller into that room. Tolerate either bare host or full sip:// URL
    # in LIVEKIT_SIP_URI. transport=tcp matches LiveKit's recommended Twilio
    # config — UDP is default but flakes under packet loss.
    host = sip_uri
    for prefix in ("sip://", "sip:"):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    # Strip any params the user pasted (e.g. ";transport=tcp") so we don't
    # double them.
    host = host.split(";", 1)[0]
    sip_destination = f"sip:{room_name}@{host};transport=tcp"

    return CallSetup(
        call_id=cid,
        room_name=room_name,
        agent_token=token,
        sip_destination=sip_destination,
    )


if __name__ == "__main__":
    # uv run python -m backend.app.telephony.livekit_sip
    from dotenv import load_dotenv

    load_dotenv()
    setup = mint_call_setup()
    print(f"call_id      : {setup.call_id}")
    print(f"room_name    : {setup.room_name}")
    print(f"sip_dest     : {setup.sip_destination}")
    print(f"token (head) : {setup.agent_token[:60]}...")
