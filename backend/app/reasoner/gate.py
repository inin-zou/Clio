"""
Intervention gate — decides when the Reasoner should override Sarah's
free sampling and force a specific behavior.

Most of the time `decide()` returns None and Sarah talks freely (the entire
point of the passive-checklist architecture, see architecture-decision.md).
The gate fires only when one of these triggers matches, in priority order:

  1. RESCUE — audio degeneration / extreme overlap / SNR collapse.
  2. PENDING_READBACK — a critical entity slot has a value but no confirmation
     after the natural read-back window expired.
  3. WRAP_UP — caller signaled end-of-call AND critical slots still missing.
  4. COMPLIANCE_DEADLINE — call has been running long enough that we MUST ask
     a regulatorily-required slot before it's too late (e.g. injuries).
  5. DRIFT — caller off-topic for too long with critical slots missing.

Each trigger emits a `ReasonerDirective` for the model_service to apply.
The gate itself is mostly stateless — it reads Session state and time.
A small per-slot bookkeeping dict prevents double-firing on the same slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .drip import (
    DirectiveBuilder,
    ReasonerDirective,
    render_readback,
)
from .schema import CRITICAL_SLOTS, ENTITY_VERIFICATION_SLOTS, FNOLSession
from .state import Session, get_by_path


# ─── Tunable thresholds ──────────────────────────────────────────────────────

# How many transcript turns can pass after a critical slot is filled before
# the gate forces a read-back if Sarah hasn't done one herself. Tighter is
# better for entity accuracy: a 1-turn grace gives Sarah's persona prompt
# one shot to handle it before we force the issue.
READBACK_GRACE_TURNS = 1

# How many caller turns can pass after we forced a read-back before we
# re-fire it (because the caller still hasn't confirmed). Without this we
# only get one shot per slot and the conversation moves on with unconfirmed
# entities. Capped so we don't loop forever on a stubbornly-noisy slot.
READBACK_REFIRE_AFTER_TURNS = 3
READBACK_MAX_ATTEMPTS = 3

# Throttle between any two gate-issued directives. Sarah's last drip-feed
# typically takes 3-6 seconds to play out; queueing another directive
# during that window causes back-to-back utterances that sound like
# stuttering ("Okay so that's X Y Z 1 2 3, is that right? Okay so
# that's A B C D 2 2, is that right?"). 5 seconds gives the prior
# read-back room to land + the caller a moment to respond.
DIRECTIVE_COOLDOWN_SEC = 5.0

# How many seconds of silence from the caller (no new turns) implies they're
# wrapping up. If any critical slot is still missing at that point, fire wrap-up.
WRAPUP_SILENCE_SECONDS = 25.0  # was 6.0 — too aggressive; real callers need time to think between turns

# Wrap-up phrase patterns. Any of these uttered by the caller fires the gate
# even before the silence threshold.
WRAPUP_PHRASES = (
    "okay thanks",
    "alright thanks",
    "thanks for your help",
    "is that all",
    "are we good",
    "we good",
    "anything else",
    "that's all",
    "that's it",
    "i think that's everything",
    "do you need anything else",
)

# Compliance: any critical slot still empty by this many seconds into the
# call MUST be asked. Tuned per-slot.
COMPLIANCE_DEADLINES_SEC: dict[str, float] = {
    "any_injuries": 240.0,        # 4 minutes — duty of care, ask early
    "incident_datetime": 180.0,   # 3 minutes — needed for fraud signals
    "incident_type": 120.0,       # 2 minutes — drives all conditional flow
}

# Wrap-up iteration cap. Each time the caller signals close (or sustains
# silence past WRAPUP_SILENCE_SECONDS) and critical slots are still empty,
# the gate fires one wrap-up question targeting the next un-asked missing
# slot. After this many attempts, the gate stops nagging — caller has
# legitimate need to leave at some point.
WRAPUP_MAX_ATTEMPTS = 2

# Rescue trigger: caller indicates they can't hear Sarah ("hello hello?",
# "what did you say", "you broke up"). Sarah responds with an apologetic
# "couldn't hear you" prompt instead of plowing ahead. Cooldown so we
# don't spam the caller if they keep saying "hello".
RESCUE_COOLDOWN_SEC = 20.0
RESCUE_MAX_PER_CALL = 3

_RESCUE_TRIGGER_PHRASES = (
    "didn't hear",
    "did not hear",
    "can't hear",
    "cannot hear",
    "what did you say",
    "could you repeat",
    "say that again",
    "say again",
    "are you there",
    "you broke up",
    "you're breaking up",
)

_RESCUE_RESPONSES = (
    "Sorry, I'm having a bit of trouble hearing you — could you say that again?",
    "Apologies, I didn't quite catch that — what was it?",
    "Sorry, you're a little unclear on my end — could you repeat that?",
)


def _looks_like_rescue_signal(session: FNOLSession) -> bool:
    """True if the most recent caller turn suggests they can't hear Sarah
    or are confused. Triggers on repeated "hello" or explicit can't-hear
    phrases — both are unambiguous human signals of audio trouble."""
    last_caller_text = ""
    for turn in reversed(session.transcript):
        if turn.role == "caller":
            last_caller_text = turn.text.lower()
            break
    if not last_caller_text:
        return False
    if last_caller_text.count("hello") >= 2:
        return True
    return any(p in last_caller_text for p in _RESCUE_TRIGGER_PHRASES)

# Drift detection: if the last N turns from the caller don't introduce any
# new entity matching CRITICAL_SLOTS, that's drift. Disabled for MVP1; see
# DRIFT_ENABLED below.
DRIFT_ENABLED = False


# ─── Gate state (per-session bookkeeping) ────────────────────────────────────

@dataclass
class _GateMemory:
    """Track which slots/triggers we've already fired on, so we don't double-fire."""
    # slot_path → {attempts: int, last_attempt_turn_idx: int}.
    # Lets us re-fire read-back if the caller hasn't confirmed within
    # READBACK_REFIRE_AFTER_TURNS of caller turns since our last attempt.
    readback_attempts: dict[str, dict] = field(default_factory=dict)
    fired_compliance: set[str] = field(default_factory=set)
    wrapup_attempts: int = 0
    wrapup_asked: set[str] = field(default_factory=set)
    fired_rescue_count: int = 0
    last_rescue_at: datetime | None = None


# ─── Trigger detection helpers ───────────────────────────────────────────────

def _missing_critical_slots(session: FNOLSession) -> list[str]:
    """Critical-tier slots whose ClaimReport value is None / empty."""
    missing: list[str] = []
    for path in CRITICAL_SLOTS:
        value = get_by_path(session.report, path)
        if value in (None, "", [], {}):
            missing.append(path)
    return missing


def _slots_needing_readback(session: FNOLSession) -> list[str]:
    """Entity-verification slots with a value but no confirmed ReadbackEvent yet.

    Uses ENTITY_VERIFICATION_SLOTS, NOT CRITICAL_SLOTS — they're related but
    different concerns. Read-back is about accuracy of identifier-shaped values
    (plates, policy numbers, dates); critical is about call completeness.
    """
    confirmed = session.confirmed_slots()
    needing: list[str] = []
    for path in ENTITY_VERIFICATION_SLOTS:
        value = get_by_path(session.report, path)
        if value not in (None, "", [], {}) and path not in confirmed:
            needing.append(path)
    return needing


def _caller_signaled_wrapup(session: FNOLSession) -> bool:
    """True if the caller's last utterance contains a wrap-up phrase."""
    for turn in reversed(session.transcript):
        if turn.role != "caller":
            continue
        text_lc = turn.text.lower()
        return any(phrase in text_lc for phrase in WRAPUP_PHRASES)
    return False


def _seconds_since_last_caller_turn(session: FNOLSession, now: datetime) -> float:
    """Seconds since the caller last said anything. Inf if never."""
    for turn in reversed(session.transcript):
        if turn.role == "caller":
            return (now - turn.timestamp).total_seconds()
    return float("inf")


def _seconds_since_call_start(session: FNOLSession, now: datetime) -> float:
    return (now - session.started_at).total_seconds()


def _turns_since_slot_filled(session: FNOLSession, slot_path: str) -> int:
    """Count caller turns since the slot was first filled.

    Hackathon shortcut: we don't track when the slot was filled; we count
    caller turns total since SOMETHING was extracted. Good enough for the
    grace-period heuristic.

    The gate fires PENDING_READBACK only after this exceeds READBACK_GRACE_TURNS,
    giving Sarah a window to read back herself per her persona prompt.
    """
    # If extractor wrote into this slot recently, we don't know exactly which
    # turn. Approximate: caller turns total. If we want precision we'd track
    # extraction timestamps in Session.
    return sum(1 for t in session.transcript if t.role == "caller")


# ─── Main gate ───────────────────────────────────────────────────────────────

class InterventionGate:
    """Stateful per-call gate. Call `decide(session, now=...)` periodically.

    Returns a ReasonerDirective when a trigger fires, or None otherwise.
    The orchestrator forwards directives to model_service over the control WS.
    """

    def __init__(self) -> None:
        self.memory = _GateMemory()
        self.builder = DirectiveBuilder()
        # Timestamp of the most recent directive we sent. Used to throttle
        # back-to-back firings so Sarah's previous utterance has time to
        # play out before we queue another (otherwise her drips overlap
        # and she sounds like she's stuttering).
        self._last_fired_at: datetime | None = None

    def _stamp(self, directive: ReasonerDirective, now: datetime) -> ReasonerDirective:
        """Mark when we last fired so the throttle in decide() works on
        the next call."""
        self._last_fired_at = now
        return directive

    # The builder is exposed so other code (e.g. the orchestrator's auth flow)
    # can emit directives that share the seq counter.
    @property
    def directives(self) -> DirectiveBuilder:
        return self.builder

    def decide(
        self,
        session: Session,
        now: datetime | None = None,
    ) -> ReasonerDirective | None:
        """Inspect session state and return a directive if any trigger fires.

        Priority: rescue > pending readback > compliance > wrap-up > drift.
        """
        now = now or datetime.now(timezone.utc)
        s = session.session  # FNOLSession

        # Throttle: if we issued a directive in the last DIRECTIVE_COOLDOWN
        # seconds, hold off so Sarah's previous utterance has time to play
        # out. Without this we queue back-to-back read-backs and Sarah
        # stutters between two sentences.
        if self._last_fired_at is not None:
            since = (now - self._last_fired_at).total_seconds()
            if since < DIRECTIVE_COOLDOWN_SEC:
                return None

        # ── 0. Audio rescue ──────────────────────────────────────────────
        # Caller signaled they can't hear / are confused ("hello hello?",
        # "what did you say"). Fire BEFORE readback because confirming a
        # value the caller can't hear is pointless. Capped per call and
        # cooldown'd so we don't spam if the caller keeps saying "hello".
        if (
            self.memory.fired_rescue_count < RESCUE_MAX_PER_CALL
            and (
                self.memory.last_rescue_at is None
                or (now - self.memory.last_rescue_at).total_seconds()
                > RESCUE_COOLDOWN_SEC
            )
            and _looks_like_rescue_signal(s)
        ):
            text = _RESCUE_RESPONSES[
                self.memory.fired_rescue_count % len(_RESCUE_RESPONSES)
            ]
            self.memory.fired_rescue_count += 1
            self.memory.last_rescue_at = now
            return self._stamp(self.builder.speak(
                text=text,
                reason=(
                    f"rescue (attempt {self.memory.fired_rescue_count}/"
                    f"{RESCUE_MAX_PER_CALL}): caller indicates can't hear / confused"
                ),
            ), now)

        # ── 1. Pending read-back ─────────────────────────────────────────
        # Fires when an entity-verification slot is filled but unconfirmed.
        # Re-fires every READBACK_REFIRE_AFTER_TURNS caller turns until
        # confirmed, capped by READBACK_MAX_ATTEMPTS.
        caller_turn_idx = sum(1 for t in s.transcript if t.role == "caller")
        for slot_path in _slots_needing_readback(s):
            attempts = self.memory.readback_attempts.get(slot_path)
            if attempts is None:
                # First time we see this slot unconfirmed.
                if _turns_since_slot_filled(s, slot_path) <= READBACK_GRACE_TURNS:
                    continue  # let Sarah's persona prompt try first
                attempts = {"attempts": 0, "last_attempt_turn_idx": -1}
            else:
                if attempts["attempts"] >= READBACK_MAX_ATTEMPTS:
                    continue  # gave up; downstream wrap-up gate will handle it
                turns_since_attempt = caller_turn_idx - attempts["last_attempt_turn_idx"]
                if turns_since_attempt < READBACK_REFIRE_AFTER_TURNS:
                    continue  # waiting on caller to respond to last attempt

            value = get_by_path(s.report, slot_path)
            attempts["attempts"] += 1
            attempts["last_attempt_turn_idx"] = caller_turn_idx
            self.memory.readback_attempts[slot_path] = attempts
            return self._stamp(self.builder.speak(
                text=render_readback(slot_path, str(value)),
                after_release="silent",  # hold and wait for caller's confirmation
                reason=(
                    f"forced readback (attempt {attempts['attempts']}/"
                    f"{READBACK_MAX_ATTEMPTS}): {slot_path} unconfirmed"
                ),
            ), now)

        # ── 2. Compliance deadlines ──────────────────────────────────────
        elapsed = _seconds_since_call_start(s, now)
        for slot_path, deadline_sec in COMPLIANCE_DEADLINES_SEC.items():
            if slot_path in self.memory.fired_compliance:
                continue
            if elapsed < deadline_sec:
                continue
            value = get_by_path(s.report, slot_path)
            if value not in (None, "", [], {}):
                continue  # already filled, no need
            self.memory.fired_compliance.add(slot_path)
            return self._stamp(self.builder.speak(
                text=_compliance_phrasing(slot_path),
                reason=f"compliance deadline ({deadline_sec:.0f}s) hit for {slot_path}",
            ), now)

        # ── 3. Wrap-up gate ──────────────────────────────────────────────
        # Only fires after the caller has spoken at least once — otherwise an
        # empty session looks like infinite silence. Iterates through missing
        # critical slots up to WRAPUP_MAX_ATTEMPTS; each wrap-up signal from
        # the caller picks the next un-asked missing slot. Without this, the
        # gate would fire once for the first missing slot and let Sarah close
        # the call with the remaining slots empty (the under-asking pattern).
        caller_has_spoken = any(t.role == "caller" for t in s.transcript)
        if (
            self.memory.wrapup_attempts < WRAPUP_MAX_ATTEMPTS
            and caller_has_spoken
        ):
            silence = _seconds_since_last_caller_turn(s, now)
            wrapup_intent = (
                _caller_signaled_wrapup(s)
                or silence > WRAPUP_SILENCE_SECONDS
            )
            if wrapup_intent:
                missing = _missing_critical_slots(s)
                # Pick a missing slot we haven't already asked about via wrap-up.
                unasked = [p for p in missing if p not in self.memory.wrapup_asked]
                if unasked:
                    target = unasked[0]
                    self.memory.wrapup_attempts += 1
                    self.memory.wrapup_asked.add(target)
                    return self._stamp(self.builder.speak(
                        text=_wrapup_phrasing(target),
                        after_release="resume",
                        reason=(
                            f"wrap-up gate (attempt "
                            f"{self.memory.wrapup_attempts}/{WRAPUP_MAX_ATTEMPTS}): "
                            f"missing {missing}, asking about {target}"
                        ),
                    ), now)
                # else: every missing slot has already been asked once via
                # wrap-up, OR all critical slots are filled. Let Sarah close.

        # ── 4. Drift (disabled for MVP1) ─────────────────────────────────
        if DRIFT_ENABLED:
            pass  # TODO

        # Default: no override.
        return None


# ─── Phrasing helpers ────────────────────────────────────────────────────────

_COMPLIANCE_PHRASES: dict[str, str] = {
    "any_injuries": (
        "Sorry to ask — was anyone injured in the accident? Even minor stuff, "
        "I just need to note it for your file."
    ),
    "incident_datetime": (
        "Just so I get the timing right on the report — when did this happen, "
        "exactly? Date and roughly the time?"
    ),
    "incident_type": (
        "Could you tell me a bit more about what kind of accident this was — "
        "was it a collision with another vehicle, or something else?"
    ),
}


def _compliance_phrasing(slot_path: str) -> str:
    return _COMPLIANCE_PHRASES.get(
        slot_path,
        f"One thing I still need from you — could you tell me about the {slot_path.replace('_', ' ')}?",
    )


_WRAPUP_PHRASES_TEMPLATE: dict[str, str] = {
    "reporter_role": (
        "One quick thing — are you the policyholder, or are you reporting "
        "on someone else's behalf?"
    ),
    "incident_datetime": (
        "I still need the date and roughly the time the accident happened "
        "— could you give me that?"
    ),
    "location.full_address": (
        "What was the address or street where the accident happened?"
    ),
    "incident_type": (
        "I want to make sure I categorize this correctly — was it a "
        "collision with another vehicle, parking damage, wildlife, or "
        "something else?"
    ),
    "description": (
        "Could you give me one more sentence about what happened, in your "
        "own words? Just so I have your version on file."
    ),
    "any_injuries": (
        "Was anyone injured? I want to make sure I note that."
    ),
    "driver_was_policyholder": (
        "One more thing — were you driving, or was someone else behind "
        "the wheel?"
    ),
    "own_vehicle_damage.drivable": (
        "Is the car drivable right now, or do we need to arrange a tow?"
    ),
}


def _wrapup_phrasing(slot_path: str) -> str:
    return _WRAPUP_PHRASES_TEMPLATE.get(
        slot_path,
        f"One more thing — could you tell me about the "
        f"{slot_path.replace('_', ' ').replace('.', ' ')}?",
    )


# ─── Smoke test ─────────────────────────────────────────────────────────────
# Walks a synthetic session through a few gate decisions and shows what fires.
#
# uv run python -m backend.app.reasoner.gate

if __name__ == "__main__":
    from datetime import timedelta

    from .extractor import SlotUpdate
    from .schema import ReadbackEvent

    s = Session.create("gate-test")
    g = InterventionGate()

    # Anchor: align all simulated turns to s.session.started_at + offset so
    # silence math is deterministic in the smoke test.
    T0 = s.session.started_at

    print("=" * 70)
    print("[T+0s] Empty session, no slots filled → no directive expected")
    print("=" * 70)
    d = g.decide(s, now=T0)
    print(f"  result: {d.type if d else 'None (Sarah free-samples)'}")

    print()
    print("=" * 70)
    print("[T+5s] Caller mentions plate, within grace window → no directive")
    print("=" * 70)
    s.add_turn("agent", "Allianz Claims, this is Sarah.",
               timestamp=T0 + timedelta(seconds=2))
    s.add_turn("caller", "Hi, my plate is B-AL-1234.",
               timestamp=T0 + timedelta(seconds=5))
    s.apply_updates([
        SlotUpdate(
            slot_path="license_plate",
            value="B-AL-1234",
            confidence=0.85,
            source_quote="my plate is B-AL-1234",
        ),
    ])
    d = g.decide(s, now=T0 + timedelta(seconds=5))
    print(f"  result: {d.type if d else 'None'} "
          f"(expected None — Sarah still has grace turns)")

    print()
    print("=" * 70)
    print("[T+30s] Grace window passed, plate still unconfirmed → forced readback")
    print("=" * 70)
    for i in range(3):
        s.add_turn("caller", f"and another sentence #{i}",
                   timestamp=T0 + timedelta(seconds=8 + i * 2))
        s.add_turn("agent", "mm-hmm",
                   timestamp=T0 + timedelta(seconds=9 + i * 2))
    d = g.decide(s, now=T0 + timedelta(seconds=30))
    if d and d.type == "speak":
        print(f"  result: speak — {d.text!r}")
        print(f"  reason: {d.reason}")

    print()
    print("=" * 70)
    print("[T+35s] Caller confirms read-back → slot locked, no further nudge")
    print("=" * 70)
    s.record_readback(
        ReadbackEvent(
            slot_path="license_plate",
            proposed_value="B-AL-1234",
            caller_response="confirmed",
            final_value="B-AL-1234",
            timestamp=T0 + timedelta(seconds=33),
        )
    )
    s.add_turn("caller", "yes that's right",
               timestamp=T0 + timedelta(seconds=33))
    d = g.decide(s, now=T0 + timedelta(seconds=35))
    print(f"  result: {d.type if d else 'None (correct — slot confirmed)'}")

    print()
    print("=" * 70)
    print("[T+125s] Compliance: incident_type still empty after 120s deadline")
    print("=" * 70)
    s.add_turn("caller", "still talking about other stuff",
               timestamp=T0 + timedelta(seconds=120))
    d = g.decide(s, now=T0 + timedelta(seconds=125))
    if d and d.type == "speak":
        print(f"  result: speak — {d.text!r}")
        print(f"  reason: {d.reason}")

    print()
    print("=" * 70)
    print("[T+200s] Caller says 'okay thanks', critical slots still missing")
    print("=" * 70)
    s.add_turn("caller", "okay thanks I appreciate it",
               timestamp=T0 + timedelta(seconds=199))
    d = g.decide(s, now=T0 + timedelta(seconds=200))
    if d and d.type == "speak":
        print(f"  result: speak — {d.text!r}")
        print(f"  reason: {d.reason}")

    print()
    print("All gate decisions ran with the priority order intact.")
    print("(Compliance > wrap-up — that's why some wrap-up scenarios fire compliance instead.)")
