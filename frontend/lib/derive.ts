/**
 * Tier-1 derivations from the existing Supabase shape.
 *
 * The design's UI expects fields like `caller_name`, `incident_type` label,
 * `detail`, `elapsed`, `fnol` (completion %), and a `code` ("CALL_01").
 * Rather than adding columns server-side, we compute these client-side
 * from the calls.fnol jsonb + calls.started_at/ended_at we already have.
 */

import type { Call, Message } from "@/lib/types";

// ─── Critical slot paths ─────────────────────────────────────────────
// Mirrors backend/app/reasoner/schema.py:CRITICAL_SLOTS. Keep in sync.
const CRITICAL_SLOTS = [
  "reporter_role",
  "reporter_name",
  "policy_number",
  "incident_datetime",
  "incident_type",
  "location.full_address",
  "description",
  "any_injuries",
  "driver_was_policyholder",
  "own_vehicle_damage.drivable",
];

// ─── Walk a dotted slot path on the FNOL jsonb ───────────────────────
function getByPath(obj: unknown, path: string): unknown {
  let cur: unknown = obj;
  for (const part of path.split(".")) {
    if (cur && typeof cur === "object" && part in (cur as object)) {
      cur = (cur as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }
  return cur;
}

function isFilled(v: unknown): boolean {
  if (v === null || v === undefined) return false;
  if (typeof v === "string") return v.trim().length > 0;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v as object).length > 0;
  return true;
}

// ─── Public helpers ──────────────────────────────────────────────────

export function callerName(c: Call): string | null {
  if (!c.fnol) return null;
  const v = getByPath(c.fnol, "reporter_name");
  return typeof v === "string" && v ? v : null;
}

export function callerDisplayName(c: Call): string {
  return callerName(c) ?? c.caller_phone ?? "Caller";
}

const INCIDENT_TYPE_LABELS: Record<string, string> = {
  collision: "Vehicle collision",
  stationary: "Parked-car damage",
  parking: "Parking incident",
  wildlife: "Wildlife collision",
  animal: "Animal incident",
  property_only: "Property damage",
  personal_injury: "Personal injury",
};

export function incidentTypeLabel(c: Call): string {
  if (!c.fnol) return "Inbound claim";
  const t = getByPath(c.fnol, "incident_type");
  if (typeof t !== "string") return "Inbound claim";
  return INCIDENT_TYPE_LABELS[t] ?? t;
}

export function locationDetail(c: Call): string {
  if (!c.fnol) return "—";
  const addr = getByPath(c.fnol, "location.full_address");
  if (typeof addr === "string" && addr) return addr;
  const country = getByPath(c.fnol, "location.country");
  if (typeof country === "string" && country) return country;
  return "Location pending";
}

/**
 * "3:48 elapsed" / "2:14 (ended)" — formatted from started_at/ended_at.
 * Pass `now` so the parent can re-render every second without recomputing
 * Date.now() inside this util.
 */
export function elapsedLabel(c: Call, now: number): string {
  const start = Date.parse(c.started_at);
  const end = c.ended_at ? Date.parse(c.ended_at) : now;
  const sec = Math.max(0, Math.floor((end - start) / 1000));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  const fmt = `${m}:${String(s).padStart(2, "0")}`;
  return c.ended_at ? `${fmt} (ended)` : `${fmt} elapsed`;
}

/**
 * 0-100 percentage of CRITICAL_SLOTS that have a non-null/non-empty value.
 */
export function fnolPercent(c: Call): number {
  if (!c.fnol) return 0;
  let filled = 0;
  for (const path of CRITICAL_SLOTS) {
    if (isFilled(getByPath(c.fnol, path))) filled += 1;
  }
  return Math.round((filled / CRITICAL_SLOTS.length) * 100);
}

/**
 * "CALL_03" — derived from chronological position in the visible list.
 * We render the nth call as `CALL_${n+1}` zero-padded.
 */
export function callCode(_call: Call, idxFromOldest: number): string {
  return `CALL_${String(idxFromOldest + 1).padStart(2, "0")}`;
}

/**
 * UI-side status label for the design's three states:
 *   live    — call in progress (no ended_at)
 *   review  — call ended, considered for follow-up (ended_at set)
 *   ended   — fallback
 *
 * Whisper / fraud states aren't wired yet — they'd need a real status
 * column on the calls table.
 */
export function statusOf(c: Call): { kind: "live" | "review" | "ended"; label: string } {
  if (!c.ended_at) return { kind: "live", label: "LIVE" };
  return { kind: "review", label: "REVIEW" };
}

/**
 * The 10 fields the design's "Claim draft" pane renders. Each pulls from
 * a slot path; missing → null (renders as "Pending").
 *
 * `slotPath` and `kind` are used by the inline editor to write back into
 * the calls.fnol jsonb. `kind` drives input parsing on save:
 *   - string:   raw text
 *   - boolean:  "yes"/"y"/"true" → true, "no"/"n"/"false" → false
 *   - datetime: ISO string passed through
 */
export type FnolFieldKind = "string" | "boolean" | "datetime";

export type FnolField = {
  label: string;
  value: string | null;
  required: boolean;
  slotPath: string;
  kind: FnolFieldKind;
};

export function fnolDraftFields(c: Call): FnolField[] {
  const get = (path: string): string | null => {
    if (!c.fnol) return null;
    const v = getByPath(c.fnol, path);
    if (v === null || v === undefined || v === "") return null;
    if (typeof v === "string") return v;
    if (typeof v === "boolean") return v ? "Yes" : "No";
    if (typeof v === "number") return String(v);
    if (Array.isArray(v)) return v.length ? v.join(", ") : null;
    return JSON.stringify(v);
  };
  const incidentLabel = incidentTypeLabel(c);
  return [
    { label: "Caller",        value: callerName(c),                      required: true,  slotPath: "reporter_name",         kind: "string"   },
    { label: "Policy",        value: get("policy_number"),               required: true,  slotPath: "policy_number",         kind: "string"   },
    { label: "Incident",      value: incidentLabel === "Inbound claim" ? null : incidentLabel, required: true, slotPath: "incident_type", kind: "string" },
    { label: "Time",          value: get("incident_datetime"),           required: true,  slotPath: "incident_datetime",     kind: "datetime" },
    { label: "Location",      value: get("location.full_address"),       required: true,  slotPath: "location.full_address", kind: "string"   },
    { label: "Vehicle plate", value: get("license_plate"),               required: true,  slotPath: "license_plate",         kind: "string"   },
    { label: "Damage",        value: get("description"),                 required: true,  slotPath: "description",           kind: "string"   },
    { label: "Third party",   value: get("other_party_involved"),        required: true,  slotPath: "other_party_involved",  kind: "string"   },
    { label: "Injuries",      value: get("any_injuries"),                required: true,  slotPath: "any_injuries",          kind: "boolean"  },
    { label: "Police report", value: get("police_case_number"),          required: false, slotPath: "police_case_number",    kind: "string"   },
  ];
}

/** Set a value at a dotted path on a jsonb-shaped object. Returns a new
 *  object (does not mutate). Creates intermediate objects as needed.
 *  Used by the inline editor to patch calls.fnol. */
export function setByPath<T extends Record<string, unknown>>(
  obj: T,
  path: string,
  value: unknown,
): T {
  const parts = path.split(".");
  // Shallow-clone the spine so React sees a new reference at every level
  const root: Record<string, unknown> = { ...(obj || {}) };
  let cur: Record<string, unknown> = root;
  for (let i = 0; i < parts.length - 1; i++) {
    const k = parts[i];
    const child = cur[k];
    cur[k] = (typeof child === "object" && child !== null && !Array.isArray(child))
      ? { ...(child as Record<string, unknown>) }
      : {};
    cur = cur[k] as Record<string, unknown>;
  }
  cur[parts[parts.length - 1]] = value;
  return root as T;
}

/** Parse user-typed input back into the right primitive based on field kind.
 *  Returns null if the input is empty (clears the slot). */
export function parseFnolInput(kind: FnolFieldKind, raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  if (kind === "boolean") {
    const lower = trimmed.toLowerCase();
    if (["yes", "y", "true", "1"].includes(lower)) return true;
    if (["no", "n", "false", "0"].includes(lower)) return false;
    return trimmed;  // unknown — store as-is so we don't silently drop info
  }
  // datetime + string both pass through; backend has no strict validation
  // on the calls.fnol jsonb so this is safe.
  return trimmed;
}

// ─── Transcript bubble grouping ──────────────────────────────────────
// Streaming transcripts arrive as fragments — "Got" / "it." / "When did" —
// because the agent_text_buf flushes on phrase boundaries. We want each
// "turn" to be ONE bubble: a continuous run of speech from one role,
// grouped together regardless of brief interruptions from the other.
//
// Critical case: Sarah is mid-utterance ("Of course, happy to help...")
// when the caller briefly interjects. The audio mute kicks in, "Of" gets
// flushed, the caller's "Hi" arrives, then Sarah resumes with "course,
// happy to help...". Without cross-role lookback, this becomes 3 bubbles
// instead of the natural 2 (one Sarah utterance + one caller interjection).
//
// Rule: for each new message, look back to the last SAME-ROLE bubble,
// skipping intervening other-role bubbles. If the gap to that bubble's
// last fragment is below SAME_TURN_GAP_MS, merge in. Otherwise, new
// bubble. We deliberately do NOT split on terminal punctuation — agent
// replies naturally contain multiple sentences and the user perceives
// them as one turn.
//
// Visual ordering: bubbles render in CREATION order (by their first
// fragment's timestamp). When a later agent fragment merges into an
// earlier agent bubble, that bubble extends in time but stays in its
// original visual position. This means a continuous Sarah utterance
// appears as ONE bubble even if the caller spoke during the middle.

export type TranscriptBubble = {
  id: number;            // first message id (stable React key)
  role: "agent" | "caller";
  source: string | null;
  text: string;          // joined fragments
  firstTimestamp: string;
  lastTimestamp: string;
};

// Threshold for "is this part of the same turn?" — generous enough that
// brief caller interjections (1-3s) don't break a continuous Sarah
// utterance, but tight enough that a real back-and-forth (Sarah, caller
// for 5s+, Sarah responding) creates two separate Sarah bubbles.
const SAME_TURN_GAP_MS = 4000;

export function groupTranscriptBubbles(messages: Message[]): TranscriptBubble[] {
  const bubbles: TranscriptBubble[] = [];
  for (const m of messages) {
    const text = m.text.trim();
    if (!text) continue;

    // Find the most recent same-role bubble (may not be the very last).
    let mergeTarget: TranscriptBubble | null = null;
    for (let i = bubbles.length - 1; i >= 0; i--) {
      if (bubbles[i].role === m.role) {
        mergeTarget = bubbles[i];
        break;
      }
    }
    const gapMs = mergeTarget
      ? Date.parse(m.timestamp) - Date.parse(mergeTarget.lastTimestamp)
      : Infinity;

    if (mergeTarget && gapMs < SAME_TURN_GAP_MS) {
      // Continuation of an in-flight turn — join with a space, but don't
      // double-space if the new fragment starts with punctuation.
      const sep = /^[,.!?;:。！？，；：]/.test(text) ? "" : " ";
      mergeTarget.text = (mergeTarget.text + sep + text)
        .replace(/\s+/g, " ")
        .trim();
      mergeTarget.lastTimestamp = m.timestamp;
    } else {
      bubbles.push({
        id: m.id,
        role: m.role,
        source: m.source ?? null,
        text,
        firstTimestamp: m.timestamp,
        lastTimestamp: m.timestamp,
      });
    }
  }
  return bubbles;
}
