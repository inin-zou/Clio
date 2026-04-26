"""Pending + active call registries — extracted from server.py so the
telephony webhook can populate them without a circular import.

Hackathon-scoped: in-process dicts. A multi-replica deployment would back
PENDING_CALLS with Redis or similar.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("clio.control.registry")


@dataclass
class PendingCall:
    call_id: str
    livekit_room: str
    livekit_agent_token: str
    voice_prompt_id: str = "NATF1"


# Keyed by call_id. Items are popped when Modal connects to /control/{call_id}.
PENDING_CALLS: dict[str, PendingCall] = {}

# Active orchestrators by call_id. Populated by server.py once the WS
# accepts. Type-erased to dict to avoid importing CallOrchestrator here
# (which would re-introduce a cycle).
ACTIVE_CALLS: dict[str, object] = {}


def register_pending_call(setup: PendingCall) -> None:
    """Twilio webhook handler calls this before spawning Modal.process_call."""
    PENDING_CALLS[setup.call_id] = setup
    logger.info(
        "registered pending call %s for room %s",
        setup.call_id, setup.livekit_room,
    )
