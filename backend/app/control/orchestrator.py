"""
Per-call orchestrator — bridges control-plane WebSocket messages to the
Reasoner's Session state machine.

Lifecycle (one CallOrchestrator per call):
  1. Constructed when Modal's control WS connects with a known call_id.
  2. SessionStart sent to Modal (system_prompt + voice + LiveKit room info).
  3. Modal responds SessionReady → orchestrator is fully online.
  4. Modal pushes TranscriptTurn events as PersonaPlex / Scribe produce text.
  5. On CallerTurnBoundary, orchestrator runs the slot extractor (Haiku),
     applies updates to Session, runs the intervention gate, and pushes any
     resulting directive back over the WS.
  6. ReadbackOutcome events are recorded into Session.readbacks.
  7. SessionClosed → orchestrator persists the session JSON and cleans up.

Designed for one orchestrator per call. State is local (no cross-call sharing).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from ..reasoner import gate, persona
from ..reasoner.extractor import SlotExtractor
from ..reasoner.schema import (
    CRITICAL_SLOTS,
    ReadbackEvent,
)
from ..reasoner.state import Session
from .eventbus import EVENT_BUS
from .messages import (
    CallerTurnBoundary,
    ModalMessage,
    ReadbackOutcome,
    SessionClosed,
    SessionEnd,
    SessionReady,
    SessionStart,
    TranscriptTurn,
)

logger = logging.getLogger("clio.control")


_MODAL_MSG_ADAPTER = TypeAdapter(ModalMessage)


class WebSocketLike:
    """Minimal WS interface so this module is testable without a real WS.

    FastAPI's WebSocket conforms to this; tests can pass a mock.
    """

    async def send_json(self, data: dict[str, Any]) -> None: ...
    async def send_text(self, data: str) -> None: ...
    async def receive_text(self) -> str: ...
    async def close(self, code: int = 1000) -> None: ...


class CallOrchestrator:
    """Per-call coordinator. One per WebSocket connection.

    External code creates this with the call setup (system_prompt, voice
    prompt id, LiveKit room/token). The websocket message loop drives
    `handle_message()`. When extractor work is needed, the orchestrator
    spawns it as a background task so WS message processing isn't blocked.
    """

    def __init__(
        self,
        ws: WebSocketLike,
        call_id: str,
        livekit_room: str,
        livekit_agent_token: str,
        voice_prompt_id: str = "NATF1",
        output_dir: Path | None = None,
    ):
        self.ws = ws
        self.call_id = call_id
        self.livekit_room = livekit_room
        self.livekit_agent_token = livekit_agent_token
        self.voice_prompt_id = voice_prompt_id
        self.output_dir = output_dir or Path("data/sessions")

        self.reasoner_session = Session.create(call_id=call_id)
        self.gate = gate.InterventionGate()
        self.extractor = SlotExtractor()

        self._extraction_task: asyncio.Task | None = None
        self._closed = False
        self._cold_start_started_at: datetime | None = None

    # ─── Setup / teardown ────────────────────────────────────────────────

    async def send_session_start(self) -> None:
        """First message Modal sees on the WS. Loads persona + tells Modal
        which LiveKit room to join."""
        now = datetime.now(UTC)
        self._cold_start_started_at = now
        system_prompt = persona.session_system_prompt(now=now)

        msg = SessionStart(
            call_id=self.call_id,
            livekit_room=self.livekit_room,
            livekit_agent_token=self.livekit_agent_token,
            system_prompt=system_prompt,
            voice_prompt_id=self.voice_prompt_id,
        )
        await self._send(msg)

    async def close(self, reason: str = "") -> None:
        if self._closed:
            return
        self._closed = True

        # Cancel any in-flight extractor task
        if self._extraction_task and not self._extraction_task.done():
            self._extraction_task.cancel()

        # Persist final session state
        try:
            out_path = self.reasoner_session.save(self.output_dir)
            logger.info("call %s: session JSON saved to %s",
                        self.call_id, out_path)
        except Exception as e:
            logger.exception("failed to save session for call %s: %s",
                             self.call_id, e)

        EVENT_BUS.publish(self.call_id, {
            "type": "session_end",
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Close WS
        try:
            await self._send(SessionEnd(call_id=self.call_id, reason=reason))
        except Exception:
            pass

    # ─── Inbound message loop ────────────────────────────────────────────

    async def handle_raw_text(self, raw: str) -> None:
        """Parse + dispatch one inbound text message from Modal."""
        try:
            msg = _MODAL_MSG_ADAPTER.validate_json(raw)
        except ValidationError as e:
            logger.warning("call %s: bad message: %s", self.call_id, e)
            return
        await self.handle_message(msg)

    async def handle_message(self, msg: ModalMessage) -> None:
        if isinstance(msg, SessionReady):
            await self._on_session_ready(msg)
        elif isinstance(msg, TranscriptTurn):
            await self._on_transcript(msg)
        elif isinstance(msg, CallerTurnBoundary):
            await self._on_turn_boundary(msg)
        elif isinstance(msg, ReadbackOutcome):
            await self._on_readback(msg)
        elif isinstance(msg, SessionClosed):
            await self._on_session_closed(msg)

    # ─── Handlers ────────────────────────────────────────────────────────

    async def _on_session_ready(self, msg: SessionReady) -> None:
        if self._cold_start_started_at:
            elapsed = (datetime.now(UTC) - self._cold_start_started_at).total_seconds()
            logger.info("call %s: ready after %.1fs", self.call_id, elapsed)
        EVENT_BUS.publish(self.call_id, {
            "type": "session_ready",
            "voice_prompt_id": msg.voice_prompt_id,
            "cold_start_seconds": msg.cold_start_seconds,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    async def _on_transcript(self, msg: TranscriptTurn) -> None:
        # Append to session transcript regardless of source.
        # The slot extractor will see both PersonaPlex monologue and Scribe
        # transcripts when it runs; agreement = high confidence.
        self.reasoner_session.add_turn(
            role=msg.role,
            text=msg.text,
            source=msg.source,
            timestamp=msg.timestamp,
        )
        EVENT_BUS.publish(self.call_id, {
            "type": "transcript",
            "role": msg.role,
            "source": msg.source,
            "text": msg.text,
            "is_final": msg.is_final,
            "timestamp": msg.timestamp.isoformat(),
        })

    async def _on_turn_boundary(self, msg: CallerTurnBoundary) -> None:
        """Caller paused — good moment to run the slot extractor."""
        if self._extraction_task and not self._extraction_task.done():
            # Previous extraction still running. Skip this one — the next
            # turn boundary will trigger a fresh extraction with newer state.
            logger.debug("call %s: skipping extraction (previous still in-flight)",
                         self.call_id)
            return
        self._extraction_task = asyncio.create_task(self._run_extraction())

    async def _on_readback(self, msg: ReadbackOutcome) -> None:
        event = ReadbackEvent(
            slot_path=msg.slot_path,
            proposed_value=msg.proposed_value,
            caller_response=msg.caller_response,
            final_value=msg.final_value,
            timestamp=msg.timestamp,
            attempt=self.reasoner_session.attempts_for_slot(msg.slot_path) + 1,
        )
        self.reasoner_session.record_readback(event)
        logger.info("call %s: readback %s = %r (%s)", self.call_id,
                    msg.slot_path, msg.final_value, msg.caller_response)
        EVENT_BUS.publish(self.call_id, {
            "type": "readback",
            "slot_path": msg.slot_path,
            "proposed_value": msg.proposed_value,
            "caller_response": msg.caller_response,
            "final_value": msg.final_value,
            "timestamp": msg.timestamp.isoformat(),
        })

    async def _on_session_closed(self, msg: SessionClosed) -> None:
        await self.close(reason=msg.reason or "modal closed session")

    # ─── Extraction + gate pipeline ──────────────────────────────────────

    async def _run_extraction(self) -> None:
        """Background task: extract slots, apply, run gate, send directive."""
        try:
            session = self.reasoner_session
            now = datetime.now(UTC)

            # 1. Run extractor on the recent transcript window
            window = session.recent_turns(n=12)
            if not window:
                return

            try:
                result = await self.extractor.extract(
                    transcript_window=window,
                    current_report=session.session.report,
                    readbacks=session.session.readbacks,
                    policy_context=session.session.policy,
                    now=now,
                )
            except Exception as e:
                logger.exception("call %s: extractor failed: %s",
                                 self.call_id, e)
                return

            logger.info("call %s: extractor → %d updates (%s)", self.call_id,
                        len(result.updates), result.reasoning[:120])

            # 2. Apply with merge semantics
            applied, rejected = session.apply_updates(result.updates)
            if rejected:
                logger.info("call %s: %d updates rejected: %s", self.call_id,
                            len(rejected),
                            [f"{r.slot_path}: {r.reason}" for r in rejected[:3]])
            if applied:
                # `applied` is list[str] of slot paths. Look up the current
                # value from the session report so the UI can display it.
                report_dump = session.session.report.model_dump(mode="json")
                EVENT_BUS.publish(self.call_id, {
                    "type": "slot_updates",
                    "applied": [
                        {
                            "slot_path": path,
                            "value": _resolve_path(report_dump, path),
                        }
                        for path in applied
                    ],
                    "rejected": [
                        {"slot_path": r.slot_path, "reason": r.reason}
                        for r in rejected
                    ],
                    "extractor_reasoning": result.reasoning,
                    "timestamp": now.isoformat(),
                })

            # 3. Run the intervention gate
            directive = self.gate.decide(session, now=now)
            if directive is not None:
                logger.info("call %s: gate fires %s — %s", self.call_id,
                            directive.type, directive.reason)
                await self._send(directive)
                EVENT_BUS.publish(self.call_id, {
                    "type": "gate_directive",
                    "directive_type": directive.type,
                    "reason": directive.reason,
                    "text": getattr(directive, "text", None),
                    "timestamp": now.isoformat(),
                })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("call %s: extraction pipeline crashed: %s",
                             self.call_id, e)

    # ─── Sending ─────────────────────────────────────────────────────────

    async def _send(self, msg: Any) -> None:
        if self._closed:
            return
        # Pydantic models have model_dump(); fall back for plain dicts.
        if hasattr(msg, "model_dump"):
            payload = msg.model_dump(mode="json")
        else:
            payload = msg
        await self.ws.send_json(payload)

    # ─── Read-only inspection (for monitoring API later) ─────────────────

    def status(self) -> dict[str, Any]:
        s = self.reasoner_session.session
        return {
            "call_id": self.call_id,
            "started_at": s.started_at.isoformat(),
            "authenticated": s.authenticated(),
            "policyholder": (s.policy.policyholder.full_name if s.policy else None),
            "transcript_turns": len(s.transcript),
            "readbacks": len(s.readbacks),
            "confirmed_slots": list(s.confirmed_slots()),
            "filled_critical_slots": [
                p for p in CRITICAL_SLOTS
                if _has_value(s.report, p)
            ],
        }


def _has_value(report: Any, path: str) -> bool:
    cursor = report.model_dump(mode="python")
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return False
        cursor = cursor.get(part)
        if cursor is None:
            return False
    return cursor not in (None, "", [], {})


def _resolve_path(report_dump: dict, path: str) -> Any:
    """Walk a dotted slot path through a model_dump dict; return None if any
    intermediate is missing. Used by the event publisher to surface the
    current value of a just-applied slot to the UI."""
    cursor: Any = report_dump
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
        if cursor is None:
            return None
    return cursor
