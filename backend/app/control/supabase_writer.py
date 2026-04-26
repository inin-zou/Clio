"""Thin write-only Supabase client used by the orchestrator.

We write three table types from the backend:
  - calls    one row per call (created on session_ready, updated on close)
  - messages one row per transcript turn (Scribe + PersonaPlex)
  - events   one row per slot-update / gate-firing / readback / lifecycle

The Next.js UI subscribes to all three via Supabase Realtime using the anon
key. This decouples the UI from any in-process state on the backend, so a
container restart or scale event doesn't lose the live view.

Failures here are best-effort: every write is wrapped so a Supabase outage
can't crash a live call. The orchestrator continues to publish to the
in-memory EVENT_BUS too, which keeps the legacy SSE /events/* endpoints
working as a fallback.
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("clio.control.supabase")


# Lazy module-level client. Constructed on first use so importing this
# module never crashes when SUPABASE_URL/KEY aren't set (e.g. local tests).
_client = None
_client_init_attempted = False


def _get_client():
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.warning(
            "supabase: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing — "
            "live persistence disabled (UI will only see in-process SSE events)"
        )
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        logger.info("supabase: client initialised against %s", url)
    except Exception as e:
        logger.exception("supabase: client init failed: %s", e)
        _client = None
    return _client


# ─── Writes ──────────────────────────────────────────────────────────────


def insert_call(call_id: str, *, caller_phone: str | None = None) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table("calls").upsert({
            "id": call_id,
            "caller_phone": caller_phone,
            "started_at": datetime.now(UTC).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning("supabase insert calls failed: %s", e)


def update_call_ended(
    call_id: str,
    *,
    fnol: dict | None = None,
    reason_ended: str | None = None,
    policy_number: str | None = None,
) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        update: dict[str, Any] = {
            "ended_at": datetime.now(UTC).isoformat(),
        }
        if fnol is not None:
            update["fnol"] = fnol
        if reason_ended is not None:
            update["reason_ended"] = reason_ended
        if policy_number is not None:
            update["policy_number"] = policy_number
        client.table("calls").update(update).eq("id", call_id).execute()
    except Exception as e:
        logger.warning("supabase update calls failed: %s", e)


def insert_message(
    call_id: str,
    *,
    role: str,
    text: str,
    source: str | None = None,
    timestamp: datetime | None = None,
) -> None:
    if not text.strip():
        return
    client = _get_client()
    if client is None:
        return
    try:
        client.table("messages").insert({
            "call_id": call_id,
            "role": role,
            "source": source,
            "text": text,
            "timestamp": (timestamp or datetime.now(UTC)).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning("supabase insert messages failed: %s", e)


def insert_event(
    call_id: str,
    *,
    type: str,
    payload: dict,
    timestamp: datetime | None = None,
) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.table("events").insert({
            "call_id": call_id,
            "type": type,
            "payload": payload,
            "timestamp": (timestamp or datetime.now(UTC)).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning("supabase insert events failed: %s", e)
