# FNOL Schema Design

**Source:** Inca-provided spec (claim example + policy info doc)
**Code:** [`backend/app/reasoner/schema.py`](../../backend/app/reasoner/schema.py), [`backend/app/reasoner/taxonomy.py`](../../backend/app/reasoner/taxonomy.py)
**Mock data:** [`data/mock_policies.json`](../../data/mock_policies.json)

## What this is

The single source of truth for what Sarah collects during a call and what she already knows from the insurance database. Everything the Reasoner does — slot extraction, intervention triggers, wrap-up confirmation, end-of-call documentation — flows from this schema.

## Two-layer model (the key design choice)

Inca's spec is explicit:

> *"we can also query the insurance database to get a lot of information. We should ask customers for info we already got."*

So the Reasoner operates on **two data sources**:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: PolicyContext (loaded from DB at call start)        │
│   • Contract identity, coverage scope                        │
│   • Vehicle, driver scope, geographic coverage               │
│   • Policyholder, claims history, fraud flags                │
│   • Telematics, prior damage records                         │
│                                                              │
│   Sarah NEVER asks the customer for any of this.             │
└──────────────────────────────────────────────────────────────┘

                            ↓ flows into Sarah's context

┌──────────────────────────────────────────────────────────────┐
│ Layer 2: ClaimReport (captured during the call)              │
│   • Authentication / identification                          │
│   • Accident circumstances                                   │
│   • Driver, other party, witnesses                           │
│   • Police, injuries, property damage                        │
│   • Liability indicators, disclosure                         │
│   • Settlement preferences                                   │
│                                                              │
│   This is what Sarah's job is to fill in.                    │
└──────────────────────────────────────────────────────────────┘
```

## Authentication-first conversational flow

Every call starts with the same goal: get a key (license plate or policy number) to load `PolicyContext`. Once loaded, Sarah can refer to known facts naturally:

> "Okay Frau Schmidt, I see you're calling about your VW Golf — you have Vollkasko with us. Now tell me what happened…"

This requires:
1. First 1–2 turns ask: license plate OR policy number
2. Look up in mock DB ([`data/mock_policies.json`](../../data/mock_policies.json))
3. Inject the loaded `PolicyContext` summary into Sarah's text monologue stream as a "lead" filler ("let me pull up your policy") + then drip-feed key facts that Sarah should reference

This is the **MoshiRAG lead/body pattern** applied at runtime without training (see [`moshirag-analysis.md`](moshirag-analysis.md)).

## Slot tiers (drives intervention gate logic)

Not all slots are equal. Tiers from `taxonomy.py`:

| Tier | When to enforce | Examples |
|---|---|---|
| **CRITICAL** | Must capture before wrap-up; trigger nudge if missing late | `incident_datetime`, `incident_type`, `location.full_address`, `description`, `any_injuries`, `own_vehicle_damage.drivable`, `reporter_role`, `driver_was_policyholder` |
| **EXPECTED** | Should capture; wrap-up confirmation can fill last gaps | `weather`, `road_conditions`, `police_on_scene`, `preferred_communication` |
| **CONDITIONAL** | Only relevant for specific incident types or branches | `other_party.*` only for collision; `injuries` only if `any_injuries=True`; `driver_info` only if `driver_was_policyholder=False` |
| **PASSIVE** | Never asked directly; inferred from conversation or DB | `fraud_signals.*`, `report_delay_hours`, `parties_known_to_each_other` |

## Conditional tree (incident_type drives downstream slots)

```
incident_type
├── collision           → other_party (full), liability, european_accident_statement
├── stationary          → other_party (limited), hit_and_run check
├── parking             → hit_and_run, police_case_number (if hit_and_run)
├── wildlife            → no other_party, no liability indicators
├── animal              → animal owner contact (if domestic), no liability
├── property_only       → property owner, no other vehicle
└── personal_injury     → injuries (full block), police_on_scene (always)
```

`schema.py:CONDITIONAL_RULES` codifies this. The intervention gate consults it before deciding whether a missing slot is a problem.

## Passive extraction: the fraud signals trap

This part requires care. From Inca's spec:

> *"Fraud signals (partly inferable from other answers, but worth asking explicitly)"*

We do **not** ask explicit fraud questions. Reasons:
- "Do you and the other driver know each other?" — instant AI tell
- "Have you filed similar claims?" — accusatory, ruins rapport
- "Is your vehicle currently for sale?" — invasive non sequitur

Instead the Reasoner extracts these signals **passively** from existing data:

| Fraud signal | Extraction source |
|---|---|
| `report_delay_hours` | `incident_datetime` vs `now` — automatic |
| `parties_known_to_each_other` | NLP on `description` ("my friend Klaus", "my brother") |
| `similar_claims_in_24m` | Cross-reference `PolicyContext.prior_claims` |
| `vehicle_listed_for_sale` | DB lookup (not implemented in mock) |
| `damage_vs_residual_value_ratio` | `own_vehicle_damage` estimate vs `current_sum_insured_eur` |
| `inconsistencies` | Slot extractor flags contradictions in transcript |

These are computed asynchronously and never block the conversation.

## Reporter role determines flow

> *"Who is reporting? — the reporter type changes the entire downstream workflow"*

Codified in `taxonomy.py:ReporterRole`. Decision tree:

```
reporter_role
├── policyholder        → standard FNOL flow (this is what we optimize for)
├── driver              → full FNOL but verify driver in fahrerkreis
├── claimant            → DIFFERENT FLOW: third-party liability claim, not FNOL
├── repair_shop         → expedited, ask for Reparaturauftrag, verify Werkstattbindung
├── lawyer              → formal channel, no detailed Q&A, route to claims handler
└── broker              → semi-formal, broker IDs the policyholder
```

For MVP2 we should at minimum **detect** the role but only fully support `policyholder`/`driver`. The other branches gracefully escalate ("let me transfer you to a human handler").

## Mock policy DB

Five sample policies in `data/mock_policies.json` covering varied scenarios:

| Plate | Profile |
|---|---|
| `B-AL-1234` | Anna Schmidt, VW Golf, Vollkasko, clean record, SF 14 |
| `M-XK-7788` | Krüger Logistik, BMW 320i, Teilkasko, **commercial use, premium in arrears, prior claim** |
| `HH-CR-9001` | Thomas Becker, Audi Q5, Vollkasko Premium, full negligence waiver, two named drivers |
| `B-DR-4242` | Daniel Roth, Peugeot 208, Teilkasko, **frequency review fraud flag, two prior claims** |
| `K-EV-2025` | Sarah Wagner, Tesla Model Y, EV-specific tariff, **telematics required active** |

These cover the main edge cases our intervention gate has to handle:
- Premium in arrears → coverage issues under §38 VVG
- Fraud flags → caller is already on a watch list
- Telematics required → can cross-validate report against device data
- Open claims → potential duplicate FNOL detection

## What's NOT in the schema (deferred)

| Field | Reason |
|---|---|
| Bank details | Should never appear in chat per Inca spec |
| Telematics trip data | Mock DB has snapshot only; full trip data is post-hackathon |
| HIS (Hinweis- und Informationssystem) live integration | Use the mock `his_hits` field |
| Connected-car OEM event data | Same — use mock telematics only |
| Multi-language support | English-only confirmed; persona prompt handles cultural context |

## How the schema is consumed

**At call start** ([`backend/app/reasoner/state.py`](../../backend/app/reasoner/state.py) — to be written):
1. Initialize `FNOLSession(call_id, started_at)`
2. After authentication slot is filled, query mock DB → load `PolicyContext`
3. Push policy summary to model_service over control WS for Sarah to reference

**During the call** ([`backend/app/reasoner/extractor.py`](../../backend/app/reasoner/extractor.py) — to be written):
1. Each user turn → run slot extractor LLM with `ClaimReport` schema as JSON-schema constraint
2. Update `session.report` with extracted values + confidence
3. Recompute passive signals from updated state

**Intervention gate** ([`backend/app/reasoner/gate.py`](../../backend/app/reasoner/gate.py) — to be written):
1. Read `CRITICAL_SLOTS` + check which are filled
2. If wrap-up signals detected and any CRITICAL slot missing → trigger nudge
3. Use `CONDITIONAL_RULES` to determine if missing slots actually apply
4. Drip-feed the nudge text into Sarah's text monologue stream

**End of call** (output for Inca scoring):
1. `session.to_summary_dict()` → JSON file with policy + report + transcript
2. This is the "high-quality call documentation" Inca scores us on
