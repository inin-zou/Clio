"""In-process pub/sub for streaming call events to UI subscribers (SSE).

The orchestrator publishes events as they happen (transcript, slot update,
gate firing, lifecycle); SSE subscribers pull from a per-subscription queue
and stream them to the browser.

Hackathon-scoped: in-memory only, single-process. A multi-replica deploy
would back this with Redis pub/sub.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("clio.control.eventbus")

# Sentinel call_id meaning "subscribe to events from ALL active calls".
ANY_CALL = "__any__"

# Bound the per-subscriber queue so a stuck SSE client can't grow memory
# unboundedly. ~1000 events ≈ 5–10 minutes of an active call; if the
# client falls behind that much we drop new events for it.
_QUEUE_MAX = 1000


class CallEventBus:
    """Per-call event channel for SSE/WebSocket UI subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, call_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.setdefault(call_id, []).append(q)
        logger.debug("event subscriber added: %s (now %d)",
                     call_id, len(self._subscribers[call_id]))
        return q

    def unsubscribe(self, call_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(call_id, [])
        if q in subs:
            subs.remove(q)
        if call_id in self._subscribers and not self._subscribers[call_id]:
            del self._subscribers[call_id]

    def publish(self, call_id: str, event: dict[str, Any]) -> None:
        """Push an event to every subscriber of this call_id AND every
        subscriber of ANY_CALL. Drops on full queue (slow consumer)."""
        # Direct call_id subscribers
        for q in list(self._subscribers.get(call_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("dropping event for slow subscriber on %s", call_id)
        # ANY_CALL subscribers also see the event, with call_id annotated
        if ANY_CALL in self._subscribers:
            payload = {"call_id": call_id, **event}
            for q in list(self._subscribers[ANY_CALL]):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    logger.warning("dropping event for slow ANY subscriber")


# Process-wide singleton.
EVENT_BUS = CallEventBus()
