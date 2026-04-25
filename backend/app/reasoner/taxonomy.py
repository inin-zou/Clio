"""
Controlled vocabulary for the FNOL Reasoner.

Single source of truth for enums used by the schema, slot extractor LLM prompts,
and intervention gate logic. Values follow the Inca-provided spec.
"""

from enum import StrEnum


class ReporterRole(StrEnum):
    """Who is calling. Changes downstream workflow significantly."""
    POLICYHOLDER = "policyholder"
    DRIVER = "driver"  # not the policyholder
    CLAIMANT = "claimant"  # other party in the accident, calling our insurer
    REPAIR_SHOP = "repair_shop"
    LAWYER = "lawyer"
    BROKER = "broker"


class IncidentType(StrEnum):
    """High-level taxonomy of loss event. Drives conditional slot logic."""
    COLLISION = "collision"  # moving traffic collision with another vehicle
    STATIONARY = "stationary"  # damage to stationary vehicle (someone hit ours)
    PARKING = "parking"  # parking lot damage, often hit-and-run
    WILDLIFE = "wildlife"  # deer, boar, etc.
    ANIMAL = "animal"  # domestic animal (dog ran out)
    PROPERTY_ONLY = "property_only"  # damage to property without third party
    PERSONAL_INJURY = "personal_injury"  # injuries primary, vehicle secondary


class KaskoType(StrEnum):
    VOLLKASKO = "vollkasko"  # comprehensive
    TEILKASKO = "teilkasko"  # partial (theft, fire, glass, wildlife)
    NONE = "none"


class UseType(StrEnum):
    PRIVATE = "private"
    COMMERCIAL = "commercial"
    COMPANY_CAR = "company_car"
    RENTAL = "rental"
    DRIVING_SCHOOL = "driving_school"
    TAXI = "taxi"


class FahrerKreis(StrEnum):
    """Who's allowed to drive under the policy."""
    NAMED = "named"  # specific listed drivers only
    OPEN = "open"  # anyone with valid license
    OPEN_WITH_AGE_RESTRICTION = "open_with_age_restriction"


class CommunicationChannel(StrEnum):
    EMAIL = "email"
    MAIL = "mail"
    PORTAL = "portal"
    PHONE = "phone"


class CancellationStatus(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    LAPSED = "lapsed"  # premium not paid


class SlotTier(StrEnum):
    """How important is this slot for declaring the FNOL complete?

    Used by the intervention gate to decide whether to nudge or wrap up.
    """
    CRITICAL = "critical"  # must capture before call ends
    EXPECTED = "expected"  # should capture; call can end without if needed
    CONDITIONAL = "conditional"  # only relevant for specific incident types
    PASSIVE = "passive"  # never asked directly; inferred from conversation
