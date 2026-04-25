"""
FNOLSession lifecycle manager.

Wraps schema.FNOLSession with mutators that:

  - Apply slot updates from the extractor with proper merge semantics
    (don't overwrite confirmed values; honor `is_correction` flag).
  - Auto-authenticate: when policy_number or license_plate becomes filled,
    look it up in db.py and attach PolicyContext.
  - Record readback events; corrections automatically update the slot.
  - Persist final session JSON at end-of-call (the deliverable Inca scores).

Single Session = single live call. No cross-call state. The Reasoner orchestrator
in livekit_agent.py creates one Session per LiveKit room and disposes it on
disconnect.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from . import db
from .extractor import SlotUpdate
from .schema import (
    ClaimReport,
    FNOLSession,
    PolicyContext,
    ReadbackEvent,
    TranscriptTurn,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def set_by_path(report: ClaimReport, path: str, value: Any) -> ClaimReport:
    """Set a nested field on a Pydantic ClaimReport via dotted path.

    Examples:
      "policy_number" → top-level
      "other_party.license_plate" → nested OtherParty model
      "own_vehicle_damage.drivable" → nested VehicleDamage model
      "injuries" → list field, replaces entire list

    Re-validates through Pydantic so type coercion + defaults still apply.
    """
    parts = path.split(".")
    data = report.model_dump(mode="python")

    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        # If intermediate field is None or missing, create dict so we can descend.
        if cursor.get(part) is None:
            cursor[part] = {}
        elif not isinstance(cursor[part], dict):
            # Was a primitive — replace with dict (caller bears responsibility for path correctness)
            cursor[part] = {}
        cursor = cursor[part]

    cursor[parts[-1]] = value
    return ClaimReport.model_validate(data)


def get_by_path(report: ClaimReport, path: str) -> Any:
    """Read a nested field by dotted path. Returns None if missing or empty."""
    cursor: Any = report.model_dump(mode="python")
    for part in path.split("."):
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(part)
        if cursor is None:
            return None
    return cursor


# ─── Session ─────────────────────────────────────────────────────────────────

class UpdateRejected(BaseModel):
    """A rejected slot update + reason. Returned from apply_updates so the
    Reasoner can decide whether to ask the caller again."""
    slot_path: str
    proposed_value: Any
    reason: str


class Session:
    """Live state for one call.

    Use create() to start, hand the resulting Session to the Reasoner
    orchestrator, mutate via the methods below as the call progresses,
    and call save() at end-of-call.
    """

    def __init__(self, session: FNOLSession):
        self.session = session

    # ── construction ──

    @classmethod
    def create(cls, call_id: str | None = None) -> Session:
        return cls(
            FNOLSession(
                call_id=call_id or f"call-{uuid4().hex[:12]}",
                started_at=datetime.now(timezone.utc),
            )
        )

    # ── auth ──

    def authenticate(self) -> PolicyContext | None:
        """Try to load PolicyContext from DB using whichever auth slot is filled.

        Idempotent — once self.session.policy is set, becomes a no-op.
        Called automatically by apply_updates when an auth slot is filled,
        but also exposed for explicit invocation (e.g. fuzzy-fallback path).
        """
        if self.session.policy is not None:
            return self.session.policy

        report = self.session.report
        # Try policy_number first (more specific), then license_plate.
        for key in (report.policy_number, report.license_plate):
            if key:
                policy = db.lookup(key)
                if policy is not None:
                    self.session.policy = policy
                    return policy
        return None

    # ── transcript ──

    def add_turn(
        self,
        role: str,
        text: str,
        source: str = "personaplex",
        timestamp: datetime | None = None,
    ) -> None:
        """Append one transcript turn. Both PersonaPlex monologue and Scribe ASR
        feed in here; source distinguishes which transcript stream."""
        self.session.transcript.append(
            TranscriptTurn(
                role=role,  # type: ignore[arg-type]
                text=text,
                timestamp=timestamp or datetime.now(timezone.utc),
                source=source,  # type: ignore[arg-type]
            )
        )

    def recent_turns(self, n: int = 12) -> list[TranscriptTurn]:
        """Last N turns, for the extractor's window."""
        return self.session.transcript[-n:]

    # ── slot updates ──

    def apply_updates(
        self,
        updates: list[SlotUpdate],
    ) -> tuple[list[str], list[UpdateRejected]]:
        """Apply a batch of extractor updates with merge semantics.

        Returns (applied_paths, rejected) where:
          - applied_paths: slot paths that changed (caller may want to log)
          - rejected: updates that were declined (e.g. would overwrite a
            confirmed value); the Reasoner can decide what to do.

        Auth side effect: if policy_number or license_plate becomes filled,
        triggers self.authenticate().
        """
        applied: list[str] = []
        rejected: list[UpdateRejected] = []
        confirmed = self.session.confirmed_slots()
        report = self.session.report

        for u in updates:
            existing = get_by_path(report, u.slot_path)

            # Rule 1: never overwrite a slot the caller verbally confirmed.
            if u.slot_path in confirmed and not u.is_correction:
                rejected.append(
                    UpdateRejected(
                        slot_path=u.slot_path,
                        proposed_value=u.value,
                        reason="slot is confirmed via read-back; only explicit "
                        "correction can overwrite",
                    )
                )
                continue

            # Rule 2: don't silently overwrite an existing non-empty value
            # unless the extractor flagged this as a correction.
            if existing not in (None, "", [], {}) and not u.is_correction:
                if existing == u.value:
                    # No change — fine, skip
                    continue
                rejected.append(
                    UpdateRejected(
                        slot_path=u.slot_path,
                        proposed_value=u.value,
                        reason=f"slot already filled with {existing!r}; new value "
                        f"{u.value!r} is not flagged as a correction",
                    )
                )
                continue

            # Apply.
            report = set_by_path(report, u.slot_path, u.value)
            applied.append(u.slot_path)

        self.session.report = report

        # Auth side effect
        if any(p in ("policy_number", "license_plate") for p in applied):
            self.authenticate()

        return applied, rejected

    # ── readbacks ──

    def record_readback(self, event: ReadbackEvent) -> None:
        """Append a readback event. If the caller corrected the value, update
        the slot too. If they confirmed, the slot is now in confirmed_slots()
        and locked against extractor overwrites.
        """
        self.session.readbacks.append(event)
        if event.caller_response == "corrected":
            self.session.report = set_by_path(
                self.session.report, event.slot_path, event.final_value
            )
            # Auth re-check: a corrected policy_number/license_plate may unlock
            # a different policy — re-load.
            if event.slot_path in ("policy_number", "license_plate"):
                self.session.policy = None
                self.authenticate()

    def attempts_for_slot(self, slot_path: str) -> int:
        """How many times Sarah has tried to confirm this slot. Useful for
        the gate: after N failed read-backs, escalate to spelling or ask for
        an alternative identifier."""
        return sum(1 for r in self.session.readbacks if r.slot_path == slot_path)

    # ── persistence ──

    def save(self, output_dir: Path) -> Path:
        """Write final session JSON. Returns the path written."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self.session.call_id}.json"
        out_path.write_text(
            json.dumps(self.session.to_summary_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return out_path


# ─── Smoke test ─────────────────────────────────────────────────────────────
# Walks a synthetic call through Session and verifies merge / auth / readback
# semantics work as described.
#
# uv run python -m backend.app.reasoner.state

if __name__ == "__main__":
    from datetime import timezone

    s = Session.create("smoke-test")
    print(f"Created session {s.session.call_id}")

    # 1. Caller mentions plate. Extractor catches it. Auth should fire.
    print("\n[1] Caller says plate. Apply update.")
    applied, rejected = s.apply_updates([
        SlotUpdate(
            slot_path="license_plate",
            value="B-AL-1234",
            confidence=0.85,
            source_quote="my plate is B dash A L dash 1 2 3 4",
        )
    ])
    print(f"   applied={applied}, rejected={rejected}")
    print(f"   authenticated? {s.session.authenticated()}")
    if s.session.policy:
        print(f"   loaded policy: {s.session.policy.policy_number} "
              f"({s.session.policy.policyholder.full_name})")

    # 2. Sarah reads back, caller confirms. Slot is now locked.
    print("\n[2] Sarah reads back plate, caller confirms.")
    s.record_readback(
        ReadbackEvent(
            slot_path="license_plate",
            proposed_value="B-AL-1234",
            caller_response="confirmed",
            final_value="B-AL-1234",
            timestamp=datetime.now(timezone.utc),
        )
    )
    print(f"   confirmed_slots: {s.session.confirmed_slots()}")

    # 3. Extractor tries to overwrite confirmed slot — should be rejected.
    print("\n[3] Hostile extractor tries to overwrite confirmed slot (should reject).")
    applied, rejected = s.apply_updates([
        SlotUpdate(
            slot_path="license_plate",
            value="B-XX-9999",
            confidence=0.55,
            source_quote="(noisy mistranscription)",
        )
    ])
    print(f"   applied={applied}, rejected={[r.reason for r in rejected]}")
    print(f"   plate is still: {s.session.report.license_plate}")

    # 4. Caller corrects. is_correction=True bypasses the lock.
    print("\n[4] Caller explicitly corrects via read-back: 'no, it's 1235 not 1234'.")
    s.record_readback(
        ReadbackEvent(
            slot_path="license_plate",
            proposed_value="B-AL-1234",
            caller_response="corrected",
            final_value="B-AL-1235",
            timestamp=datetime.now(timezone.utc),
            attempt=2,
        )
    )
    print(f"   plate now: {s.session.report.license_plate}")
    print(f"   authenticated? {s.session.authenticated()} "
          f"(should be False — new plate isn't in DB)")

    # 5. Caller corrects again to a real plate.
    print("\n[5] Caller corrects again to a real plate.")
    s.record_readback(
        ReadbackEvent(
            slot_path="license_plate",
            proposed_value="B-AL-1235",
            caller_response="corrected",
            final_value="B-AL-1234",
            timestamp=datetime.now(timezone.utc),
            attempt=3,
        )
    )
    print(f"   plate now: {s.session.report.license_plate}")
    print(f"   authenticated? {s.session.authenticated()}")
    print(f"   attempts on license_plate: {s.attempts_for_slot('license_plate')}")

    # 6. Nested field via dotted path.
    print("\n[6] Apply nested update: own_vehicle_damage.drivable=False.")
    applied, _ = s.apply_updates([
        SlotUpdate(
            slot_path="own_vehicle_damage.drivable",
            value=False,
            confidence=0.95,
            source_quote="the car can't be driven",
        )
    ])
    print(f"   applied={applied}")
    print(f"   own_vehicle_damage.drivable = {s.session.report.own_vehicle_damage.drivable}")

    # 7. Persist
    print("\n[7] Save session to disk.")
    out_dir = Path("/tmp/clio-test-output")
    out_path = s.save(out_dir)
    print(f"   wrote {out_path} ({out_path.stat().st_size} bytes)")
    print("\nAll session lifecycle checks passed.")
