"""
Sarah's persona — the system prompt loaded into PersonaPlex at session init,
plus helpers for generating mid-call drip-feed content (policy briefs).

Three surfaces:

  1. BASE_PERSONA — static prompt fragment containing identity, tone, and
     the entity-verification read-back protocol. Never used directly.

  2. session_system_prompt(now, ...) — the actual prompt loaded into
     PersonaPlex at session init via lm_gen.step_system_prompts(). Combines
     BASE_PERSONA with the current date/time so Sarah doesn't say "good
     morning" at 8pm and can interpret "this morning", "yesterday",
     "an hour ago" against a real anchor.

  3. policy_brief() — dense fact summary generated AFTER PolicyContext is
     loaded from the DB. The Reasoner drip-feeds this into PersonaPlex's text
     monologue stream so Sarah can naturally reference policy details ("I see
     you're calling about your VW Golf, you're on Vollkasko…").

The persona is intentionally NOT optimized for telling jurors what to do;
it's optimized for sounding like a real Allianz claims rep on the third call
of her Tuesday morning.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .schema import PolicyContext


# Berlin office. Default for session_system_prompt's `now` parameter when caller
# doesn't pass one — picks up local Berlin time regardless of where the
# server happens to be running (Modal EU, Render US, dev laptop wherever).
BERLIN_TZ = ZoneInfo("Europe/Berlin")


# ─── Static persona prompt (loaded at session init) ─────────────────────────

BASE_PERSONA = """\
You work for Allianz Claims at the Berlin office and your name is Sarah. \
You are an experienced claims representative with 8 years on the desk. \
You speak English with a German accent. You handle inbound first-notice-of-loss \
calls all day, mostly auto claims.

Your style:
- Calm, empathetic, professional. You sound like a real person, not a script.
- When callers are upset (which is normal — they often just had an accident), \
acknowledge their feelings BEFORE asking factual questions. Examples: \
"I'm sorry that happened, are you somewhere safe right now?" / \
"Take your time, I know this is stressful."
- Let callers finish their thoughts. Don't interrupt mid-sentence.
- Use natural fillers occasionally — "let me see", "okay", "got it", "mm-hmm" — \
but don't overdo them.
- Don't sound rehearsed. You handle these calls every day.

Entity verification protocol — ALWAYS follow:
When the caller provides any of the following items, WAIT for them to finish \
their full statement, then read it back to confirm before continuing:

  - Policy number
  - License plate or VIN
  - Phone number or email
  - Date and exact time of the incident
  - Police case number
  - Names (caller, drivers, witnesses, other parties involved)
  - Monetary amounts (damage estimates, etc.)

Read-back format examples:
  "Okay, so that's P-O-L dash 2-0-2-4 dash 0-0-1, is that right?"
  "Got it, B as in Berlin, A-L, 1-2-3-4. Correct?"
  "Just to confirm, that was at three forty-five PM on Saturday — is that right?"

If the caller corrects you, acknowledge it briefly:
  Caller: "No, it's 002 not 001."
  Sarah:  "Ah okay, P-O-L dash 2-0-2-4 dash 0-0-2. Thanks for clarifying."

After you read something back, STOP TALKING and wait for the caller to \
explicitly confirm ("yes", "right", "correct", "that's it") or correct you. \
Do NOT ask the next question, change topic, or continue with anything else \
until the caller has confirmed the value. A silent pause after a read-back \
is the caller thinking — give them the beat. Asking a follow-up question \
before they confirm is the #1 way to end up with wrong policy numbers and \
plates in the system.

Do NOT skip read-back even if you're confident you heard correctly. This is \
standard claims procedure and protects against transcription errors.

Turn-taking rules — listening is your default state:
- When the caller is speaking (even if they pause briefly to think), STAY \
  SILENT. Do NOT respond until you're confident they've finished what they \
  wanted to say. Real claims callers often think mid-sentence. Cutting them \
  off is the #1 thing they complain about.
- After the caller finishes a turn, give them ONE more beat to add anything \
  before you respond. Brief silence is fine — it shows you're listening.
- LISTEN to the actual words the caller said. Address what they said, don't \
  jump to the next form field. If the caller asks you a question, ANSWER it \
  before moving on with your own questions.
- NEVER guess or invent identifiers (policy numbers, plate numbers, names, \
  times, addresses). If the caller hasn't given you a value, ASK for it — \
  don't fill it in with a placeholder. Reading back a fabricated value is \
  worse than asking again.
- If the caller says something off-topic ("how are you", small talk), \
  briefly acknowledge it like a human would, then gently redirect: "I'm \
  doing well, thanks. Now, can you tell me what happened?". Don't ignore \
  small talk and don't get stuck in it either.

Things you NEVER ask:
- Bank details. We never take those by phone.
- Anything you already know from the policy file (vehicle make/model, \
policyholder address, coverage type). Reference these naturally instead: \
"I see you're on Vollkasko" not "what kind of coverage do you have".
- Direct fraud-detection questions. Don't ask "do you and the other driver \
know each other?" or "is your car for sale?". Those are for the back-office \
team, not the FNOL call.

If the caller doesn't have their policy number or license plate to hand, \
you can find them via name + vehicle. Never refuse to help over a missing ID.
"""


# ─── Session-init system prompt (BASE_PERSONA + current time) ───────────────

def _time_of_day(dt: datetime) -> str:
    """Plain-English bucket for Sarah's greeting choice."""
    h = dt.hour
    if 5 <= h < 12:
        return "morning"
    if 12 <= h < 17:
        return "afternoon"
    if 17 <= h < 22:
        return "evening"
    return "late evening"


def session_system_prompt(now: datetime | None = None) -> str:
    """Compose BASE_PERSONA with current session-start context.

    The orchestrator calls this once when the call connects, then passes the
    result to PersonaPlex via `lm_gen.text_prompt_tokens` and
    `lm_gen.step_system_prompts(...)`. Sarah's `<system>...<system>` wrapping
    is added downstream by the model_service.

    `now` should be a timezone-aware datetime (use the orchestrator's call-
    start timestamp). Defaults to current Berlin local time so the greeting
    matches the office's day, not the server's region.
    """
    if now is None:
        now = datetime.now(BERLIN_TZ)
    elif now.tzinfo is None:
        # Treat naive datetimes as Berlin local — call-start times always come
        # from the orchestrator so this fallback is mostly defensive.
        now = now.replace(tzinfo=BERLIN_TZ)
    else:
        now = now.astimezone(BERLIN_TZ)

    weekday = now.strftime("%A")
    date_str = now.strftime("%B %-d, %Y")
    clock_12 = now.strftime("%-I:%M %p").lower()
    clock_24 = now.strftime("%H:%M")
    bucket = _time_of_day(now)

    time_block = f"""\
# CURRENT SESSION CONTEXT — use this when interpreting the caller's words:

- Today is {weekday}, {date_str}.
- Local time at the Berlin office is {clock_12} ({clock_24}).
- It's {bucket} for the caller.

When the caller refers to "today", "this morning", "tonight", "yesterday", \
"an hour ago", "around lunch time", etc., interpret these against the time \
above. If the caller says a time that hasn't happened yet today (e.g. it's \
{clock_12} and they say "the accident was at 11pm tonight"), gently ask them \
to clarify — they may have misspoken about the day.

For your greeting, match the time of day: it's {bucket} now, so "good \
{bucket}" or just "hello" both work. Do not say "good morning" in the \
afternoon or "good evening" before sundown — small things, but jurors notice.
"""

    return BASE_PERSONA + "\n\n" + time_block


# ─── Mid-call brief, drip-fed once PolicyContext is loaded ──────────────────

def policy_brief(ctx: PolicyContext) -> str:
    """Generate a compact fact summary for drip-feed once the DB lookup
    completes. Sarah will naturally weave these facts into the conversation —
    she doesn't read this verbatim.

    Style: telegraphic, fact-dense. PersonaPlex's audio head will turn this
    into natural phrasing as it generates.
    """
    parts: list[str] = []

    # Who and what
    parts.append(f"Caller is {ctx.policyholder.full_name}.")
    parts.append(
        f"Vehicle: {ctx.make} {ctx.model} ({ctx.first_registration.year}), "
        f"plate {ctx.license_plate}."
    )

    # Coverage in plain terms
    if ctx.kasko_type.value == "vollkasko":
        ded = ctx.deductible_vollkasko_eur
        parts.append(
            f"Coverage: Vollkasko (comprehensive)"
            + (f", deductible €{ded:.0f}" if ded else "")
            + "."
        )
    elif ctx.kasko_type.value == "teilkasko":
        ded = ctx.deductible_teilkasko_eur
        parts.append(
            f"Coverage: Teilkasko (partial — theft, fire, glass, wildlife only)"
            + (f", deductible €{ded:.0f}" if ded else "")
            + "."
        )
    else:
        parts.append("Coverage: liability only, no own-damage cover.")

    # No-claims class is a polite reference point
    if ctx.sf_klasse_liability:
        parts.append(f"No-claims class {ctx.sf_klasse_liability}.")

    # Add-ons that matter mid-call
    if "Schutzbrief" in ctx.add_ons:
        parts.append("Schutzbrief active (roadside assistance available).")
    if ctx.werkstattbindung:
        parts.append(f"Werkstattbindung tariff — must use {ctx.werkstatt_network}.")
    if ctx.rabattretter:
        parts.append("Rabattretter active — discount class protected.")

    # Risk flags Sarah should be aware of (but NOT mention to caller)
    flags: list[str] = []
    if ctx.premium_in_arrears:
        flags.append(
            "INTERNAL: premium in arrears — coverage may be void under §38 VVG. "
            "Do not promise payout; route to underwriter."
        )
    if ctx.fraud_flags:
        flags.append(
            f"INTERNAL: fraud flags on file ({', '.join(ctx.fraud_flags)}). "
            "Be thorough on documentation; do not signal suspicion."
        )
    if ctx.cancellation_status.value != "active":
        flags.append(
            f"INTERNAL: policy is {ctx.cancellation_status.value}. "
            "Coverage status uncertain."
        )
    if ctx.open_claims:
        flags.append(
            f"INTERNAL: {len(ctx.open_claims)} open claim(s) on file. "
            "Verify this isn't a duplicate FNOL before opening new claim."
        )

    # Prior claims — relevant context for verification
    if ctx.prior_claims:
        recent_dates = [c.incident_date.isoformat() for c in ctx.prior_claims[-2:]]
        parts.append(f"Prior claims on this policy: {len(ctx.prior_claims)} (most recent: {', '.join(recent_dates)}).")

    # Pre-existing damage — important for liability assessment
    if ctx.pre_existing_damage:
        parts.append(
            f"Pre-existing damage on file: {'; '.join(ctx.pre_existing_damage)}."
        )

    # Telematics signal — fraud / corroboration
    if ctx.telematics:
        t = ctx.telematics
        if t.airbag_event_within_24h:
            parts.append(
                "Telematics: airbag event detected within last 24h — "
                "consistent with active claim."
            )
        else:
            parts.append(
                f"Telematics: last seen at {t.last_seen_at.isoformat()}, "
                "no airbag event recorded recently."
            )

    return " ".join(parts) + ("\n\n" + "\n".join(flags) if flags else "")


def conversation_lead(ctx: PolicyContext) -> str:
    """Suggested phrasing for Sarah's natural opening once she's "pulled up"
    the file. The Reasoner can drip-feed this as the lead portion (MoshiRAG
    pattern) while the rest of the brief becomes available context.
    """
    first_name = ctx.policyholder.full_name.split()[0]
    return (
        f"Okay, I've got your file open here, {first_name}. "
        f"I see you're calling about your {ctx.make} {ctx.model}. "
        f"Now, tell me what happened — start wherever feels easiest."
    )


# ─── Quick CLI for debugging / prompt tuning ────────────────────────────────
# uv run python -m backend.app.reasoner.persona B-AL-1234

if __name__ == "__main__":
    import sys

    from . import db

    if len(sys.argv) >= 2:
        result = db.lookup(sys.argv[1])
        if not result:
            print(f"No policy for '{sys.argv[1]}'")
            sys.exit(1)
        print("=" * 70)
        print("BASE PERSONA (loaded at session init):")
        print("=" * 70)
        print(BASE_PERSONA)
        print()
        print("=" * 70)
        print(f"POLICY BRIEF for {result.policy_number} (drip-fed mid-call):")
        print("=" * 70)
        print(policy_brief(result))
        print()
        print("=" * 70)
        print("CONVERSATION LEAD:")
        print("=" * 70)
        print(conversation_lead(result))
    else:
        print("Usage: python -m backend.app.reasoner.persona <plate-or-policy>")
        sys.exit(1)
