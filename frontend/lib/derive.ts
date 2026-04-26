/**
 * Tier-1 derivations from the existing Supabase shape.
 *
 * The design's UI expects fields like `caller_name`, `incident_type` label,
 * `detail`, `elapsed`, `fnol` (completion %), and a `code` ("CALL_01").
 * Rather than adding columns server-side, we compute these client-side
 * from the calls.fnol jsonb + calls.started_at/ended_at we already have.
 */

import type { Call } from "@/lib/types";

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
 */
export type FnolField = { label: string; value: string | null; required: boolean };

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

  return [
    { label: "Caller",           value: callerName(c),                   required: true },
    { label: "Policy",           value: get("policy_number"),            required: true },
    { label: "Incident",         value: incidentTypeLabel(c) === "Inbound claim" ? null : incidentTypeLabel(c), required: true },
    { label: "Time",             value: get("incident_datetime"),        required: true },
    { label: "Location",         value: get("location.full_address"),    required: true },
    { label: "Vehicle plate",    value: get("license_plate"),            required: true },
    { label: "Damage",           value: get("description"),              required: true },
    { label: "Third party",      value: get("other_party_involved"),     required: true },
    { label: "Injuries",         value: get("any_injuries"),             required: true },
    { label: "Police report",    value: get("police_case_number"),       required: false },
  ];
}
