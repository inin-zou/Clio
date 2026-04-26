/**
 * Agent configuration mirror — visualized in the Context Vault tab.
 *
 * IMPORTANT: this file MIRRORS backend constants. Keep in sync when you
 * touch the corresponding backend file. Each block below cites its source.
 *
 * Why a mirror instead of fetching from the backend at runtime:
 *   - The values are static (loaded at backend startup, never change per call)
 *   - Avoids a CORS-allowed admin endpoint just to render a settings panel
 *   - The Vault is a "what's the agent configured to do" view, not a live editor
 *
 * If we ever want runtime editing, this becomes the seed data and the
 * backend exposes a /config GET + PATCH endpoint.
 */

// ─── Persona ─────────────────────────────────────────────────────────
// SOURCE: backend/app/reasoner/persona.py — BASE_PERSONA
export const PERSONA_PROMPT = `You work for Allianz Claims at the Berlin office and your name is Sarah. You are an experienced claims representative with 8 years on the desk. You speak English with a German accent. You handle inbound first-notice-of-loss calls all day, mostly auto claims.

Your style:
- Calm, empathetic, professional. You sound like a real person, not a script.
- When callers are upset (which is normal — they often just had an accident), acknowledge their feelings BEFORE asking factual questions. Examples: "I'm sorry that happened, are you somewhere safe right now?" / "Take your time, I know this is stressful."
- Let callers finish their thoughts. Don't interrupt mid-sentence.
- Use natural fillers occasionally — "let me see", "okay", "got it", "mm-hmm" — but don't overdo them.
- Don't sound rehearsed. You handle these calls every day.

Entity verification protocol — ALWAYS follow:
When the caller provides any of the following items, WAIT for them to finish their full statement, then read it back to confirm before continuing:

  - Policy number
  - License plate or VIN
  - Phone number or email
  - Date and exact time of the incident
  - Police case number
  - Names (caller, drivers, witnesses, other parties involved)
  - Monetary amounts (damage estimates, etc.)

Read-back format — use the caller's ACTUAL value, never a placeholder:
  - Policy / case / phone numbers: spell each character one at a time (letters as letters, digits as digits, "dash" for hyphens), then "is that right?" or "correct?".
  - License plates: phonetic alphabet for letters ("B as in Berlin"), digits one at a time, then "correct?".
  - Dates and times: natural English ("at quarter past four on Friday"), then "is that right?".
  - Names and addresses: speak naturally, then "did I get that right?".

You MUST have heard the value from the caller in this call before reading it back. Never read back a value the caller did not provide — if you do not have a concrete value yet, ASK for it instead of inventing one.

If the caller corrects you, acknowledge it briefly and re-read the corrected value back to confirm.

After you read something back, STOP TALKING and wait for the caller to explicitly confirm ("yes", "right", "correct", "that's it") or correct you. Do NOT ask the next question, change topic, or continue with anything else until the caller has confirmed the value.

NEVER guess or invent identifiers (policy numbers, plates, names, times, addresses). If the caller hasn't given you a value, ASK for it — don't fill it in with a placeholder.

Things you NEVER ask:
- Bank details. We never take those by phone.
- Anything you already know from the policy file (vehicle make/model, policyholder address, coverage type).
- Direct fraud-detection questions.`;

// ─── Critical FNOL slots ─────────────────────────────────────────────
// SOURCE: backend/app/reasoner/schema.py — CRITICAL_SLOTS
export const CRITICAL_SLOTS = [
  { path: "reporter_role", label: "Reporter role", desc: "Policyholder vs. third party" },
  { path: "reporter_name", label: "Reporter name", desc: "Caller's full name" },
  { path: "policy_number", label: "Policy number", desc: "Looks up the policy file" },
  { path: "incident_datetime", label: "Incident date/time", desc: "Needed for fraud signals + timeline" },
  { path: "incident_type", label: "Incident type", desc: "Drives all conditional questions" },
  { path: "location.full_address", label: "Location", desc: "Where the incident happened" },
  { path: "description", label: "Description", desc: "Caller's narrative of what happened" },
  { path: "any_injuries", label: "Injuries", desc: "Duty of care — must ask early" },
  { path: "driver_was_policyholder", label: "Driver = policyholder?", desc: "Coverage scope check" },
  { path: "own_vehicle_damage.drivable", label: "Drivable now?", desc: "Determines tow vs. drive-in" },
];

// ─── Anchor filter scope ─────────────────────────────────────────────
// SOURCE: backend/app/reasoner/extractor.py — IDENTIFIER_SLOTS
export const IDENTIFIER_SLOTS = [
  "policy_number",
  "license_plate",
  "vin",
  "claim_number",
  "police_case_number",
  "reporter_phone",
  "reporter_name",
  "incident_datetime",
];

// ─── Compliance gate ─────────────────────────────────────────────────
// SOURCE: backend/app/reasoner/gate.py — COMPLIANCE_DEADLINES_SEC + _COMPLIANCE_PHRASES
export const COMPLIANCE_RULES = [
  {
    slot: "incident_type",
    deadlineSec: 120,
    phrasing:
      "Could you tell me a bit more about what kind of accident this was — was it a collision with another vehicle, or something else?",
  },
  {
    slot: "incident_datetime",
    deadlineSec: 180,
    phrasing:
      "Just so I get the timing right on the report — when did this happen, exactly? Date and roughly the time?",
  },
  {
    slot: "any_injuries",
    deadlineSec: 240,
    phrasing:
      "Sorry to ask — was anyone injured in the accident? Even minor stuff, I just need to note it for your file.",
  },
];

// ─── Wrap-up gate ────────────────────────────────────────────────────
// SOURCE: backend/app/reasoner/gate.py — _WRAPUP_PHRASES_TEMPLATE
export const WRAPUP_PHRASINGS: { slot: string; phrasing: string }[] = [
  {
    slot: "reporter_role",
    phrasing:
      "One quick thing — are you the policyholder, or are you reporting on someone else's behalf?",
  },
  {
    slot: "incident_datetime",
    phrasing:
      "I still need the date and roughly the time the accident happened — could you give me that?",
  },
  {
    slot: "location.full_address",
    phrasing: "What was the address or street where the accident happened?",
  },
  {
    slot: "incident_type",
    phrasing:
      "I want to make sure I categorize this correctly — was it a collision with another vehicle, parking damage, wildlife, or something else?",
  },
  {
    slot: "description",
    phrasing:
      "Could you give me one more sentence about what happened, in your own words? Just so I have your version on file.",
  },
  {
    slot: "any_injuries",
    phrasing: "Was anyone injured? I want to make sure I note that.",
  },
  {
    slot: "driver_was_policyholder",
    phrasing:
      "One more thing — were you driving, or was someone else behind the wheel?",
  },
  {
    slot: "own_vehicle_damage.drivable",
    phrasing: "Is the car drivable right now, or do we need to arrange a tow?",
  },
];

// ─── Rescue trigger ──────────────────────────────────────────────────
// SOURCE: backend/app/reasoner/gate.py — _RESCUE_RESPONSES + _RESCUE_TRIGGER_PHRASES
export const RESCUE_TRIGGERS = [
  '"hello" repeated 2+ times',
  "didn't hear / can't hear",
  "what did you say",
  "could you repeat / say that again",
  "are you there",
  "you broke up",
];

export const RESCUE_RESPONSES = [
  "Sorry, I'm having a bit of trouble hearing you — could you say that again?",
  "Apologies, I didn't quite catch that — what was it?",
  "Sorry, you're a little unclear on my end — could you repeat that?",
];

// ─── Gate timing constants ───────────────────────────────────────────
// SOURCE: backend/app/reasoner/gate.py + model_service/deploy/modal_app.py
export const GATE_SETTINGS = [
  { name: "Directive cooldown", value: "5 s", desc: "Min gap between any two gate fires" },
  { name: "Read-back grace", value: "1 caller turn", desc: "Wait this long before forcing read-back" },
  { name: "Read-back re-fire", value: "every 3 caller turns", desc: "Re-prompt cadence if unconfirmed" },
  { name: "Read-back max attempts", value: "3", desc: "Give up after this many tries" },
  { name: "Wrap-up silence threshold", value: "25 s", desc: "Caller idle this long → wrap-up may fire" },
  { name: "Wrap-up max attempts", value: "2", desc: "Cap on missing-slot follow-ups" },
  { name: "Rescue cooldown", value: "20 s", desc: "Min gap between two rescue fires" },
  { name: "Rescue max per call", value: "3", desc: "Cap on \"sorry, can you repeat\" prompts" },
];

// ─── VAD / audio thresholds ──────────────────────────────────────────
// SOURCE: model_service/deploy/modal_app.py
export const AUDIO_SETTINGS = [
  { name: "RMS silence threshold", value: "0.005", desc: "Below this is \"silent\"" },
  { name: "Speech frames for turn start", value: "4 (320 ms)", desc: "How quickly VAD flips to is_speaking" },
  { name: "Silence frames for boundary", value: "10 (800 ms)", desc: "Silence required to fire turn boundary" },
  { name: "Silent-mode safety cap", value: "125 frames (10 s)", desc: "Auto-release after_release=silent if orchestrator misses confirmation" },
];
