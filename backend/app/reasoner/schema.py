"""
FNOL (First Notice of Loss) schema for Clio.

Two-layer model per Inca's spec ("we can also query the insurance database…
we should ask customers for info we already got"):

1. PolicyContext — loaded from the (mocked) policy DB at the start of every call.
   Sarah does NOT ask the customer for any of these fields; they are pre-known.

2. ClaimReport — what Sarah captures DURING the call. The Reasoner's job is to
   fill this in through natural conversation, using PolicyContext as background
   knowledge that Sarah can refer to without asking.

The two layers reconcile at wrap-up: any conditional/expected slot still empty
is checked against PolicyContext (e.g. "driver_in_scope" cross-checks
ClaimReport.driver against PolicyContext.fahrerkreis rules).

Verification is handled via the read-back protocol (see fnol-schema.md). Sarah
verbally repeats critical entities back to the caller; ReadbackEvent records
the outcome (confirmed / corrected / unclear). This is the source of truth for
slot confidence — not algorithmic confidence scoring.
"""

from __future__ import annotations
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from .taxonomy import (
    CancellationStatus,
    CommunicationChannel,
    FahrerKreis,
    IncidentType,
    KaskoType,
    ReporterRole,
    UseType,
)


# ─── Layer 1: pre-loaded PolicyContext ──────────────────────────────────────
# Everything Sarah already knows about the caller's policy and history.
# Nothing in here is asked of the customer.

class PolicyholderInfo(BaseModel):
    full_name: str
    dob: date
    address: str
    phone: str | None = None
    email: str | None = None
    preferred_language: str = "en"
    preferred_channel: CommunicationChannel = CommunicationChannel.EMAIL
    vulnerable_flags: list[str] = Field(default_factory=list)
    legal_representative: str | None = None
    consent_marketing: bool = False
    consent_telematics: bool = False
    consent_data_sharing: bool = False


class PriorClaim(BaseModel):
    claim_number: str
    incident_date: date
    incident_type: IncidentType
    settlement_amount_eur: float | None = None
    fault_quota: float | None = None  # 0.0 = not at fault, 1.0 = fully at fault
    status: str  # "settled", "open", "denied", etc.


class DriverDeclaration(BaseModel):
    full_name: str
    dob: date
    license_held_since: date
    is_occasional_surcharge: bool = False


class TelematicsSnapshot(BaseModel):
    """Last known telematics state. Massive validation lever for fraud."""
    last_seen_at: datetime
    last_known_location: tuple[float, float] | None = None  # (lat, lon)
    last_known_speed_kmh: float | None = None
    airbag_event_within_24h: bool = False


class PolicyContext(BaseModel):
    """Loaded from the (mocked) insurance DB by license_plate or policy_number."""

    # ─── Contract identity ────────────────────────────────────────────────
    policy_number: str
    product_name: str  # e.g. "KFZ Komfort"
    tariff_version: str  # AKB version
    underwriting_insurer: str  # relevant for white-label / fronting
    broker_id: str | None = None
    effective_date: date
    renewal_date: date
    cancellation_status: CancellationStatus = CancellationStatus.ACTIVE
    premium_in_arrears: bool = False  # affects coverage under §38 VVG

    # ─── Insured vehicle ──────────────────────────────────────────────────
    license_plate: str
    vin: str
    make: str
    model: str
    first_registration: date
    engine_power_kw: int
    fuel_type: str
    vehicle_value_at_inception_eur: float
    current_sum_insured_eur: float
    use_type: UseType = UseType.PRIVATE
    annual_mileage_band: str  # e.g. "10000-15000"
    garage_address: str  # often != policyholder address
    modifications_declared: list[str] = Field(default_factory=list)

    # ─── Coverage scope ───────────────────────────────────────────────────
    liability_sum_eur: int = 100_000_000
    kasko_type: KaskoType = KaskoType.NONE
    deductible_vollkasko_eur: float | None = None
    deductible_teilkasko_eur: float | None = None
    add_ons: list[str] = Field(default_factory=list)
    # e.g. ["GAP", "Schutzbrief", "Auslandsschadenschutz", "Fahrerschutz"]
    sf_klasse_liability: str | None = None
    sf_klasse_vollkasko: str | None = None
    rabattretter: bool = False
    rabattschutz: bool = False
    werkstattbindung: bool = False
    werkstatt_network: str | None = None

    # ─── Driver scope ─────────────────────────────────────────────────────
    fahrerkreis: FahrerKreis = FahrerKreis.NAMED
    age_restriction_min: int | None = None
    declared_drivers: list[DriverDeclaration] = Field(default_factory=list)

    # ─── Geographic & temporal ────────────────────────────────────────────
    country_coverage: list[str] = Field(default_factory=lambda: ["DE", "EU"])
    saisonkennzeichen: tuple[int, int] | None = None  # (start_month, end_month)

    # ─── Exclusions ───────────────────────────────────────────────────────
    race_exclusion: bool = True
    gross_negligence_waiver: str = "standard"  # "full", "partial", "standard"
    telematics_required_active: bool = False

    # ─── Customer ─────────────────────────────────────────────────────────
    policyholder: PolicyholderInfo

    # ─── Claims & risk ────────────────────────────────────────────────────
    prior_claims: list[PriorClaim] = Field(default_factory=list)
    open_claims: list[PriorClaim] = Field(default_factory=list)
    fraud_flags: list[str] = Field(default_factory=list)
    his_hits: list[str] = Field(default_factory=list)
    pre_existing_damage: list[str] = Field(default_factory=list)

    # ─── Vehicle / telematics ─────────────────────────────────────────────
    last_known_mileage: int | None = None
    telematics: TelematicsSnapshot | None = None
    last_hu_au: date | None = None  # last vehicle inspection


# ─── Layer 2: ClaimReport — captured during the call ─────────────────────────

class LocationDetail(BaseModel):
    full_address: str | None = None
    country: str = "DE"
    road_type: str | None = None  # "highway", "rural", "urban", "parking_lot"
    lat: float | None = None
    lon: float | None = None


class DriverInfo(BaseModel):
    """If the policyholder was NOT driving."""
    full_name: str
    dob: date | None = None
    relationship_to_policyholder: str | None = None  # "spouse", "child", "friend"
    license_valid: bool | None = None
    license_class: str | None = None


class OtherParty(BaseModel):
    name: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    license_plate: str | None = None
    vehicle_make_model: str | None = None
    insurer_name: str | None = None
    insurer_policy_number: str | None = None
    registered_owner_differs: bool | None = None
    fault_admission: bool | None = None  # admitted fault at scene?
    has_already_filed_claim: bool | None = None


class Witness(BaseModel):
    name: str
    contact: str | None = None
    statement_summary: str | None = None


class Person(BaseModel):
    name: str | None = None
    relation: str | None = None  # "passenger_own", "passenger_other", "pedestrian"


class Injury(BaseModel):
    person_name: str | None = None
    person_role: str  # "policyholder", "driver", "passenger_own", "passenger_other", "pedestrian", "cyclist"
    injury_type: str  # free-form
    treatment: str | None = None  # "outpatient", "inpatient", "none"
    hospital: str | None = None
    health_insurer: str | None = None  # for subrogation
    expected_time_off_work_days: int | None = None


class VehicleDamage(BaseModel):
    description: str | None = None
    visible_areas: list[str] = Field(default_factory=list)
    photos_available: bool = False
    drivable: bool | None = None
    current_location: str | None = None
    towed_to: str | None = None


class PropertyDamage(BaseModel):
    description: str
    owner: str | None = None  # could be municipality, business, individual
    estimated_value_eur: float | None = None


class LawyerInfo(BaseModel):
    name: str
    firm: str | None = None
    phone: str | None = None
    email: str | None = None


class FraudSignals(BaseModel):
    """Inferred passively, never asked. The Reasoner sets these flags as the
    conversation unfolds, not by direct questions."""
    report_delay_hours: float | None = None
    parties_known_to_each_other: bool | None = None
    similar_claims_in_24m: int | None = None
    vehicle_listed_for_sale: bool | None = None
    damage_vs_residual_value_ratio: float | None = None
    inconsistencies: list[str] = Field(default_factory=list)


class ClaimReport(BaseModel):
    """What Sarah captures during the call. Initially mostly None."""

    # ─── Authentication / identification (asked first) ────────────────────
    policy_number: str | None = None
    license_plate: str | None = None  # alternative key for policy lookup

    reporter_role: ReporterRole | None = None
    reporter_name: str | None = None
    reporter_phone: str | None = None

    # ─── Accident circumstances ───────────────────────────────────────────
    incident_datetime: datetime | None = None
    location: LocationDetail = Field(default_factory=LocationDetail)
    weather: str | None = None
    visibility: str | None = None
    road_conditions: str | None = None
    incident_type: IncidentType | None = None
    description: str | None = None  # caller's own words
    direction_of_travel: str | None = None
    estimated_speed_kmh: int | None = None
    right_of_way_situation: str | None = None
    has_sketch_or_photos: bool | None = None
    european_accident_statement_signed: bool | None = None

    # ─── Driver ───────────────────────────────────────────────────────────
    driver_was_policyholder: bool | None = None
    driver_info: DriverInfo | None = None
    alcohol_drugs: bool | None = None
    breathalyzer_done: bool | None = None
    health_limitations: bool | None = None
    hit_and_run: bool | None = None

    # ─── Other party (conditional on incident_type) ───────────────────────
    other_party: OtherParty | None = None

    # ─── Police & authorities ─────────────────────────────────────────────
    police_on_scene: bool | None = None
    police_station: str | None = None
    police_case_number: str | None = None
    fines_or_proceedings: bool | None = None

    # ─── Witnesses ────────────────────────────────────────────────────────
    independent_witnesses: list[Witness] = Field(default_factory=list)
    passengers_own: list[Person] = Field(default_factory=list)
    passengers_other: list[Person] = Field(default_factory=list)

    # ─── Personal injuries ────────────────────────────────────────────────
    any_injuries: bool | None = None
    injuries: list[Injury] = Field(default_factory=list)

    # ─── Property damage ──────────────────────────────────────────────────
    own_vehicle_damage: VehicleDamage = Field(default_factory=VehicleDamage)
    other_vehicle_damage: VehicleDamage | None = None
    third_party_property: list[PropertyDamage] = Field(default_factory=list)
    pre_existing_damage_in_area: bool | None = None

    # ─── Liability indicators ─────────────────────────────────────────────
    fault_admission_at_scene: bool | None = None
    written_admission: bool | None = None

    # ─── Disclosure / coverage exclusion checks ───────────────────────────
    use_outside_contract: bool | None = None  # commercial use on private policy?
    racing_event: bool | None = None
    unregistered_modifications: bool | None = None
    inspection_current: bool | None = None
    use_outside_geo_coverage: bool | None = None
    late_report_reason: str | None = None

    # ─── Settlement preferences ───────────────────────────────────────────
    repair_shop_steering: bool | None = None
    rental_car_requested: bool | None = None
    lawyer_involved: LawyerInfo | None = None
    preferred_communication: CommunicationChannel | None = None

    # ─── Inferred passively ───────────────────────────────────────────────
    fraud_signals: FraudSignals = Field(default_factory=FraudSignals)


# ─── Slot tier registry ───────────────────────────────────────────────────────
# Maps schema field names → tier. Used by intervention gate to decide:
#   - CRITICAL: must capture before wrap-up; trigger nudge if missing late in call
#   - EXPECTED: should capture; wrap-up confirmation can fill gaps
#   - CONDITIONAL: only relevant for certain incident_type values
#   - PASSIVE: never ask directly; inferred or extracted from existing data

CRITICAL_SLOTS: list[str] = [
    "reporter_role",
    "incident_datetime",
    "location.full_address",
    "incident_type",
    "description",
    "any_injuries",
    "driver_was_policyholder",
    "own_vehicle_damage.drivable",
]

EXPECTED_SLOTS: list[str] = [
    "weather",
    "road_conditions",
    "police_on_scene",
    "own_vehicle_damage.description",
    "preferred_communication",
]

# Conditional slots: keyed by the predicate field, only collected if condition met
CONDITIONAL_RULES: dict[str, dict[str, list[str]]] = {
    "incident_type": {
        IncidentType.COLLISION: [
            "other_party.name",
            "other_party.license_plate",
            "other_party.insurer_name",
            "fault_admission_at_scene",
            "european_accident_statement_signed",
        ],
        IncidentType.PARKING: [
            "hit_and_run",
            "police_case_number",
        ],
        IncidentType.WILDLIFE: [
            # mostly already covered by critical; less to ask
        ],
        IncidentType.PERSONAL_INJURY: [
            "injuries",
            "police_on_scene",
        ],
    },
    "any_injuries": {
        True: ["injuries"],  # if yes, capture details
    },
    "driver_was_policyholder": {
        False: ["driver_info"],  # if no, capture replacement driver
    },
}

PASSIVE_SLOTS: list[str] = [
    "fraud_signals.report_delay_hours",
    "fraud_signals.parties_known_to_each_other",
    "fraud_signals.similar_claims_in_24m",
    "fraud_signals.inconsistencies",
]

# Slots whose VALUES must be verbally read back to the caller for confirmation.
# This is the entity-verification protocol (see persona.py BASE_PERSONA).
# Distinct from CRITICAL_SLOTS — those are about call completeness; these are
# about transcription accuracy of identifier-shaped values.
ENTITY_VERIFICATION_SLOTS: list[str] = [
    # Authentication identifiers
    "policy_number",
    "license_plate",
    # Caller / contact details
    "reporter_phone",
    # Time-sensitive
    "incident_datetime",
    # Numeric identifiers from third parties
    "police_case_number",
    "other_party.license_plate",
    "other_party.insurer_policy_number",
    "other_party.phone",
    "other_party.email",
]


# ─── Conversation tracking ───────────────────────────────────────────────────

class TranscriptTurn(BaseModel):
    """One utterance in the call. Both PersonaPlex's text monologue stream and
    the parallel Scribe v2 ASR feed into here; the extractor sees both sources.
    """
    role: Literal["caller", "agent"]
    text: str
    timestamp: datetime
    source: Literal["personaplex", "scribe"] = "personaplex"


class ReadbackEvent(BaseModel):
    """Records one execution of the entity verification read-back protocol.

    Flow:
      1. Caller provides an entity ("my policy is POL-2024-001")
      2. Sarah waits for them to finish, then reads back ("so that's POL dash
         2024 dash 001, is that right?")
      3. Caller responds (confirmed / corrected / unclear)
      4. We record this event; final_value is what's written to ClaimReport.

    The ReadbackEvent IS the slot-confidence signal — once a critical entity
    is "confirmed" via this protocol, downstream Reasoner logic trusts it
    without further verification (no extra ASR cross-checking required).
    """
    slot_path: str  # dotted path: "policy_number", "other_party.license_plate"
    proposed_value: str  # what Sarah said back
    caller_response: Literal["confirmed", "corrected", "unclear"]
    final_value: str  # corrected version if caller said "no, it's…"
    timestamp: datetime
    attempt: int = 1  # 1st, 2nd, 3rd attempt at confirming this slot


# ─── Combined session state ──────────────────────────────────────────────────

class FNOLSession(BaseModel):
    """The Reasoner's full state for one call.

    PolicyContext is loaded once at the start (after authentication).
    ClaimReport is filled in as the conversation progresses.
    Transcript and readbacks accumulate throughout.
    """
    call_id: str
    started_at: datetime
    policy: PolicyContext | None = None  # None until authenticated
    report: ClaimReport = Field(default_factory=ClaimReport)
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    readbacks: list[ReadbackEvent] = Field(default_factory=list)

    def authenticated(self) -> bool:
        return self.policy is not None

    def confirmed_slots(self) -> set[str]:
        """Slot paths that have been verbally confirmed by the caller."""
        return {
            r.slot_path
            for r in self.readbacks
            if r.caller_response == "confirmed"
        }

    def needs_readback(self, slot_path: str) -> bool:
        """True if this critical-entity slot has a value but no confirmation yet."""
        return slot_path in CRITICAL_SLOTS and slot_path not in self.confirmed_slots()

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize for end-of-call output (the 'documentation' Inca scores)."""
        return {
            "call_id": self.call_id,
            "started_at": self.started_at.isoformat(),
            "policy": self.policy.model_dump(mode="json") if self.policy else None,
            "report": self.report.model_dump(mode="json"),
            "transcript": [t.model_dump(mode="json") for t in self.transcript],
            "readbacks": [r.model_dump(mode="json") for r in self.readbacks],
        }
