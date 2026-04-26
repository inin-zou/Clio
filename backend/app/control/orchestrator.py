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
from ..reasoner.extractor import SlotExtractor, filter_caller_anchored
from ..reasoner.schema import (
    CRITICAL_SLOTS,
    ReadbackEvent,
)
from ..reasoner.state import Session
from . import supabase_writer
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

        # Backend-side ReadbackOutcome detection. When the gate forces a
        # read-back, we APPEND (slot_path, value) here. The caller's next
        # transcript is pattern-matched against confirm/correct/both/all;
        # on a match we synthesize a ReadbackEvent locally so the gate's
        # confirmed_slots() updates and it stops re-firing.
        #
        # FIFO list (not single-slot) because the gate can issue several
        # readbacks back-to-back (e.g. policy_number + license_plate),
        # and a caller's "Yeah" needs to confirm the right one. "Both"
        # / "all" / "all of them" → confirm every pending entry at once.
        self._pending_readbacks: list[dict] = []

    # ─── Setup / teardown ────────────────────────────────────────────────

    async def send_session_start(self) -> None:
        """First message Modal sees on the WS. Loads persona + tells Modal
        which LiveKit room to join."""
        now = datetime.now(UTC)
        self._cold_start_started_at = now
        system_prompt = persona.session_system_prompt(now=now)

        # Create the calls row up-front so realtime subscribers see the call
        # appear immediately, even before any transcript turns arrive.
        await asyncio.to_thread(supabase_writer.insert_call, self.call_id)

        msg = SessionStart(
            call_id=self.call_id,
            livekit_room=self.livekit_room,
            livekit_agent_token=self.livekit_agent_token,
            system_prompt=system_prompt,
            voice_prompt_id=self.voice_prompt_id,
        )
        await self._send(msg)

        # Forced opening greeting so the floor is never cold. Without this
        # Sarah waits for VAD to detect caller-side audio before EPAD fires
        # — fragile (depends on line noise / caller's "hi") and adds 1-2s
        # of dead air at call connect. Drip queues immediately; Modal will
        # play it as soon as the snapshot restore completes.
        await self._send(self.gate.directives.speak(
            text=persona.opening_greeting(now=now),
            reason="opening greeting (call connect)",
        ))

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

        # Persist final FNOL state to Supabase so the UI shows it past
        # session end (and so we have a queryable record of the call).
        try:
            fnol_dump = self.reasoner_session.session.report.model_dump(mode="json")
            policy_number = None
            if self.reasoner_session.session.policy:
                policy_number = self.reasoner_session.session.policy.policy_number
            await asyncio.to_thread(
                supabase_writer.update_call_ended,
                self.call_id,
                fnol=fnol_dump,
                reason_ended=reason,
                policy_number=policy_number,
            )
        except Exception as e:
            logger.warning("supabase update_call_ended failed: %s", e)

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

        # If we forced a read-back recently and this is the caller's
        # response, try to match it against confirm/correct/unclear/
        # confirm-all and synthesize ReadbackEvent(s) locally. Without
        # this loop the gate never sees the slot as confirmed and
        # re-fires until it gives up.
        if (
            msg.role == "caller"
            and msg.source == "scribe"  # Scribe transcripts are the clean signal
            and self._pending_readbacks
        ):
            outcome = _classify_readback_response(msg.text)
            confirm_all = _matches_confirm_all(msg.text)
            if outcome is not None or confirm_all:
                # confirm_all always confirms every pending entry. Single
                # confirm/correct only consumes the OLDEST pending — the
                # rest stay queued for the next caller utterance.
                if confirm_all:
                    to_resolve = list(self._pending_readbacks)
                    self._pending_readbacks = []
                else:
                    to_resolve = [self._pending_readbacks.pop(0)]

                for pending in to_resolve:
                    response = (
                        outcome["response"] if outcome else "confirmed"
                    )
                    final_value = (
                        (outcome and outcome["final_value"])
                        or pending["value"]
                    )
                    event = ReadbackEvent(
                        slot_path=pending["slot_path"],
                        proposed_value=pending["value"],
                        caller_response=response,
                        final_value=final_value,
                        timestamp=msg.timestamp,
                        attempt=self.reasoner_session.attempts_for_slot(
                            pending["slot_path"]
                        ) + 1,
                    )
                    self.reasoner_session.record_readback(event)
                    logger.info(
                        "call %s: auto-detected readback %s = %r (%s%s)",
                        self.call_id,
                        event.slot_path,
                        event.final_value,
                        event.caller_response,
                        ", via confirm-all" if confirm_all else "",
                    )
                    payload = event.model_dump(mode="json")
                    EVENT_BUS.publish(self.call_id, {
                        "type": "readback",
                        **payload,
                    })
                    await asyncio.to_thread(
                        supabase_writer.insert_event,
                        self.call_id,
                        type="readback",
                        payload=payload,
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
        # Mirror to Supabase so the UI can subscribe to a durable channel
        # rather than the in-process EVENT_BUS (which doesn't survive a
        # container restart or scale event).
        await asyncio.to_thread(
            supabase_writer.insert_message,
            self.call_id,
            role=msg.role,
            text=msg.text,
            source=msg.source,
            timestamp=msg.timestamp,
        )

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
        readback_payload = {
            "slot_path": msg.slot_path,
            "proposed_value": msg.proposed_value,
            "caller_response": msg.caller_response,
            "final_value": msg.final_value,
        }
        EVENT_BUS.publish(self.call_id, {
            "type": "readback",
            **readback_payload,
            "timestamp": msg.timestamp.isoformat(),
        })
        await asyncio.to_thread(
            supabase_writer.insert_event,
            self.call_id,
            type="readback",
            payload=readback_payload,
            timestamp=msg.timestamp,
        )

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

            # 2. Structural anchor filter: drop identifier-slot updates whose
            # value isn't anchored to caller text. Catches the predict-the-
            # answer pattern where Sarah free-samples a value (e.g. "On June
            # 15.") and Haiku stores it despite Rule 10.
            kept_updates, anchor_dropped = filter_caller_anchored(
                result.updates, window
            )
            if anchor_dropped:
                logger.warning(
                    "call %s: anchor filter dropped %d hallucinated identifier(s): %s",
                    self.call_id, len(anchor_dropped),
                    [(u.slot_path, u.value) for u, _ in anchor_dropped],
                )

            # 3. Apply with merge semantics
            applied, rejected = session.apply_updates(kept_updates)
            if rejected:
                logger.info("call %s: %d updates rejected: %s", self.call_id,
                            len(rejected),
                            [f"{r.slot_path}: {r.reason}" for r in rejected[:3]])
            if applied:
                # `applied` is list[str] of slot paths. Look up the current
                # value from the session report so the UI can display it.
                report_dump = session.session.report.model_dump(mode="json")
                slot_payload = {
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
                }
                EVENT_BUS.publish(self.call_id, {
                    "type": "slot_updates",
                    **slot_payload,
                    "timestamp": now.isoformat(),
                })
                await asyncio.to_thread(
                    supabase_writer.insert_event,
                    self.call_id,
                    type="slot_update",
                    payload=slot_payload,
                    timestamp=now,
                )

            # 3. Run the intervention gate
            directive = self.gate.decide(session, now=now)
            if directive is not None:
                logger.info("call %s: gate fires %s — %s", self.call_id,
                            directive.type, directive.reason)
                await self._send(directive)

                # If the gate fired a read-back, append to the pending
                # queue so the next caller transcript can be pattern-
                # matched and turned into a ReadbackEvent.
                if "forced readback" in directive.reason:
                    slot_path = _slot_from_readback_reason(directive.reason)
                    if slot_path:
                        value = _resolve_path(
                            session.session.report.model_dump(mode="json"),
                            slot_path,
                        )
                        # Drop any older pending entry for the SAME slot
                        # (a re-fire supersedes its own previous attempt).
                        self._pending_readbacks = [
                            p for p in self._pending_readbacks
                            if p["slot_path"] != slot_path
                        ]
                        self._pending_readbacks.append({
                            "slot_path": slot_path,
                            "value": str(value) if value is not None else "",
                            "issued_at": now,
                        })
                        logger.debug(
                            "call %s: queued readback %s = %r (pending=%d)",
                            self.call_id, slot_path, value,
                            len(self._pending_readbacks),
                        )

                gate_payload = {
                    "directive_type": directive.type,
                    "reason": directive.reason,
                    "text": getattr(directive, "text", None),
                }
                EVENT_BUS.publish(self.call_id, {
                    "type": "gate_directive",
                    **gate_payload,
                    "timestamp": now.isoformat(),
                })
                await asyncio.to_thread(
                    supabase_writer.insert_event,
                    self.call_id,
                    type="gate_fired",
                    payload=gate_payload,
                    timestamp=now,
                )

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


# ─── Readback auto-detection helpers ─────────────────────────────────────


import re

# Reasons emitted by gate.py look like:
#   "forced readback (attempt 1/3): policy_number unconfirmed"
_READBACK_REASON_RE = re.compile(r"forced readback[^:]*:\s*([\w.]+)\s+unconfirmed")


def _slot_from_readback_reason(reason: str) -> str | None:
    m = _READBACK_REASON_RE.search(reason)
    return m.group(1) if m else None


# Phrase patterns the caller might use after a read-back. Order matters: we
# check correction first because "no actually" should NOT trigger confirm.
_CORRECT_PATTERNS = (
    r"\bno\b",
    r"\bnope\b",
    r"\bwrong\b",
    r"\bactually\b",
    r"\bi meant\b",
    r"\bit's\b",
    r"\bit is\b",
    r"\bcorrect.{0,5}me\b",
)
_CONFIRM_PATTERNS = (
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\byup\b",
    r"\bthat'?s right\b",
    r"\bthat'?s correct\b",
    r"\bcorrect\b",
    r"\bright\b",
    r"\bexactly\b",
    r"\bperfect\b",
    r"\bgood\b",
    r"\baffirmative\b",
    r"\bmm-?hm+\b",
)


_CONFIRM_ALL_PATTERNS = (
    r"\ball of (them|those|that)\b",
    r"\bboth\b",
    r"\beverything\b",
    r"\bcorrect.{0,15}both\b",
    r"\byes.{0,5}both\b",
    r"\byeah.{0,5}both\b",
    r"\bboth.{0,15}correct\b",
    r"\bboth.{0,15}right\b",
)


def _matches_confirm_all(text: str) -> bool:
    """Detect 'both/all/everything' style confirmations that should
    accept every pending readback at once. Used in addition to the
    per-slot classifier so 'Correct, both of them' counts."""
    if not text:
        return False
    t = text.lower().strip()
    return any(re.search(p, t) for p in _CONFIRM_ALL_PATTERNS)


def _classify_readback_response(text: str) -> dict | None:
    """Classify a caller utterance as confirm / correct / unclear.

    Returns None if the text doesn't look like a response to a read-back at
    all (so we don't accidentally lock in a non-confirmation as confirmed).
    """
    if not text:
        return None
    t = text.lower().strip()

    correcting = any(re.search(p, t) for p in _CORRECT_PATTERNS)
    confirming = any(re.search(p, t) for p in _CONFIRM_PATTERNS)

    if correcting and not confirming:
        # Try to extract the corrected value — best-effort, falls back to
        # full text. Better correction parsing is a future improvement.
        return {"response": "corrected", "final_value": text.strip()}
    if confirming and not correcting:
        return {"response": "confirmed", "final_value": None}
    if confirming and correcting:
        # Mixed signal ("no, that's right"?) — ambiguous
        return {"response": "unclear", "final_value": None}
    # No clear signal; don't classify.
    return None
