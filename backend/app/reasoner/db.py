"""
Mock policy database lookup.

Design choices for hackathon (see .claude/docs/architecture.md and
fnol-schema.md):

  - Single JSON file (`data/mock_policies.json`), loaded once at startup.
  - Module-level dict indexes for O(1) exact lookup by plate or policy number.
  - No vector search: license plates and policy numbers are exact-match keys.
    Vector search would be both slower and semantically wrong.
  - Plate / policy number normalization (uppercase, strip separators) so the
    LLM extractor's output ("B-AL-1234", "b al 1234", "BAL1234") all map to
    the same entry.
  - Fuzzy fallback by caller name + vehicle make for the case where the
    caller doesn't have their plate / policy number to hand.

Lookup latency: < 1 microsecond per call (Python dict.get on a normalized key).
The DB lookup is never the bottleneck — ASR + slot extractor LLM dominate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from .schema import PolicyContext

# data/mock_policies.json relative to project root.
# This file is at backend/app/reasoner/db.py → project root is 3 levels up.
DEFAULT_DB_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "mock_policies.json"
)

# ─── Module-level cache (loaded once) ────────────────────────────────────────
_DB: dict | None = None
_PLATE_INDEX: dict[str, str] = {}    # normalized plate    → policy_number
_POLICY_INDEX: dict[str, dict] = {}  # normalized policy # → raw policy dict


# ─── Normalization ───────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^A-Z0-9]")


def normalize(key: str) -> str:
    """Uppercase, strip all non-alphanumeric. 'B-AL-1234' → 'BAL1234'."""
    if not key:
        return ""
    return _NON_ALNUM.sub("", key.upper())


# ─── Loading ─────────────────────────────────────────────────────────────────

def load(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Load mock DB into module-level indexes. Idempotent.

    Raises FileNotFoundError if db_path doesn't exist (fail loud at startup).
    """
    global _DB
    if _DB is not None:
        return

    with db_path.open() as f:
        _DB = json.load(f)

    for plate, policy_id in _DB.get("by_license_plate", {}).items():
        _PLATE_INDEX[normalize(plate)] = policy_id

    for policy_id, policy in _DB.get("policies", {}).items():
        _POLICY_INDEX[normalize(policy_id)] = policy


def _ensure_loaded() -> None:
    if _DB is None:
        load()


def reload() -> None:
    """Force re-read from disk. Useful in development/tests."""
    global _DB
    _DB = None
    _PLATE_INDEX.clear()
    _POLICY_INDEX.clear()
    load()


# ─── Lookups ─────────────────────────────────────────────────────────────────

def get_by_plate(plate: str) -> PolicyContext | None:
    """Look up a policy by license plate. Returns None if not found."""
    _ensure_loaded()
    policy_id = _PLATE_INDEX.get(normalize(plate))
    if not policy_id:
        return None
    raw = _POLICY_INDEX.get(normalize(policy_id))
    return PolicyContext.model_validate(raw) if raw else None


def get_by_policy_number(policy_number: str) -> PolicyContext | None:
    """Look up a policy by its policy number. Returns None if not found."""
    _ensure_loaded()
    raw = _POLICY_INDEX.get(normalize(policy_number))
    return PolicyContext.model_validate(raw) if raw else None


def lookup(key: str) -> PolicyContext | None:
    """Auto-detect whether `key` is a plate or a policy number and look up.

    Convention (from mock_policies.json): policy numbers start with 'POL'.
    Anything else is treated as a plate first, then policy number as fallback.
    """
    _ensure_loaded()
    norm = normalize(key)
    if not norm:
        return None

    # Heuristic: 'POL...' is a policy number; otherwise try plate first.
    if norm.startswith("POL"):
        return get_by_policy_number(key) or get_by_plate(key)
    return get_by_plate(key) or get_by_policy_number(key)


def fuzzy_search_by_caller(
    full_name: str | None = None,
    vehicle_make: str | None = None,
    limit: int = 5,
) -> list[PolicyContext]:
    """Fallback: caller doesn't remember plate or policy number.

    All provided fields must match (AND, not OR). Name match is sub-string
    on lowercased policyholder name; vehicle make is case-insensitive equality.

    Hackathon scale — iterates all policies. Adequate up to ~10k policies.
    """
    _ensure_loaded()

    name_tokens = (full_name or "").lower().split()
    make_lc = (vehicle_make or "").lower()

    results: list[PolicyContext] = []
    for raw in _POLICY_INDEX.values():
        if name_tokens:
            holder = raw.get("policyholder", {}).get("full_name", "").lower()
            if not all(tok in holder for tok in name_tokens):
                continue
        if make_lc:
            if raw.get("make", "").lower() != make_lc:
                continue
        results.append(PolicyContext.model_validate(raw))
        if len(results) >= limit:
            break

    return results


def all_policies() -> Iterator[PolicyContext]:
    """Iterate every policy. Useful for tests and eval scripts."""
    _ensure_loaded()
    for raw in _POLICY_INDEX.values():
        yield PolicyContext.model_validate(raw)


# ─── CLI for quick debugging ─────────────────────────────────────────────────
# Usage:
#   uv run python -m backend.app.reasoner.db B-AL-1234
#   uv run python -m backend.app.reasoner.db POL-2024-001

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.app.reasoner.db <plate-or-policy-or-name>")
        print("Examples:")
        print("  python -m backend.app.reasoner.db B-AL-1234")
        print("  python -m backend.app.reasoner.db POL-2024-001")
        print("  python -m backend.app.reasoner.db --name 'Anna Schmidt'")
        sys.exit(1)

    if sys.argv[1] == "--name" and len(sys.argv) >= 3:
        results = fuzzy_search_by_caller(full_name=sys.argv[2])
        if not results:
            print(f"No policies match name '{sys.argv[2]}'")
            sys.exit(1)
        for policy in results:
            print(f"{policy.policy_number} — {policy.policyholder.full_name} — {policy.make} {policy.model}")
    else:
        key = sys.argv[1]
        result = lookup(key)
        if result:
            print(f"Match: {result.policy_number}")
            print(f"  Policyholder: {result.policyholder.full_name}")
            print(f"  Vehicle:      {result.make} {result.model} ({result.license_plate})")
            print(f"  Coverage:     {result.kasko_type.value} (deductible "
                  f"€{result.deductible_vollkasko_eur or result.deductible_teilkasko_eur or 'n/a'})")
            if result.fraud_flags:
                print(f"  ⚠ Fraud flags: {', '.join(result.fraud_flags)}")
            if result.premium_in_arrears:
                print(f"  ⚠ Premium in arrears (§38 VVG coverage issue)")
            if result.prior_claims:
                print(f"  Prior claims: {len(result.prior_claims)}")
        else:
            print(f"No policy found for '{key}'")
            sys.exit(1)
