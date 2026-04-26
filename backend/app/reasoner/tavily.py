"""
Tavily fact-checker — UNWIRED in this version.

This module is a self-contained tool script. It provides async fact-check
helpers for cross-referencing caller statements against external sources
during an inbound FNOL call. Nothing in the rest of the codebase imports
or calls it yet — that's intentional, deferred to a future version.

Why it exists
-------------
PersonaPlex is a generative model. It has no real-time knowledge of
weather conditions, news, or whether a given address exists. The slot
extractor blindly trusts what the caller says. For fraud signals and
demo polish, we want Sarah to *corroborate*: when the caller says "icy
roads in Berlin yesterday morning", check whether Tavily's web search
agrees. When it doesn't, surface as a fraud signal.

The architecture-decision.md and moshirag-analysis.md docs sketch this
"reasoner triggers external lookup, drip-feeds the answer into Sarah's
text monologue" pattern as a future enhancement.

Three primitive checks
----------------------
- weather_at(location, when) → corroborate weather conditions
- verify_location(address)   → confirm the address exists / is plausible
- news_check(location, when) → search for traffic/incident reports

Each returns a `FactCheckResult` with a brief summary plus an
`inconsistency_signal` boolean the gate can use to populate
`fraud_signals.inconsistencies`.

How to wire (next version)
--------------------------
1. Add to pyproject.toml dependencies:
       "tavily-python>=0.5"
2. Add to .env / .env.example:
       TAVILY_API_KEY=tvly-...
3. Create Modal secret:
       modal secret create clio-tavily TAVILY_API_KEY=tvly-...
4. Mount the secret in modal_backend.py's secret list.
5. In gate.py, add a fact_check trigger between drive and wrap-up:
       if (
           not memory.fact_checked
           and report.incident_datetime
           and report.location.full_address
       ):
           memory.fact_checked = True
           asyncio.create_task(
               _run_fact_check(session, orchestrator)
           )
6. In orchestrator, persist the result as a `fact_check` event row +
   optionally drip-feed a one-liner ("(Internal: weather was rainy.)")
   into Sarah's stream so she can reference it.

Until that wiring exists, the only entry point is `python -m
backend.app.reasoner.tavily` — see `if __name__ == "__main__"` below
for a smoke test that hits the live API (requires TAVILY_API_KEY env).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

# tavily-python is NOT in pyproject.toml yet (this module is unwired).
# Importing it lazily inside the client class so the rest of the
# backend keeps importing cleanly without the dependency installed.
# When wiring this up, add `tavily-python>=0.5` to pyproject.toml.


# ─── Result schema ──────────────────────────────────────────────────

FactCheckKind = Literal["weather", "location", "news"]


class FactCheckResult(BaseModel):
    """One fact-check pass. Designed to be JSON-serializable for
    the events table and for drip-feed context lines.
    """

    kind: FactCheckKind = Field(description="Which kind of check this is.")
    query: str = Field(description="The exact query sent to Tavily.")
    summary: str = Field(
        description="One-line human-readable summary of what we found, "
        "intended for both UI display and drip-feed into Sarah's stream. "
        "Kept short (<120 chars) so it doesn't overflow the agent_text_buf."
    )
    inconsistency_signal: bool = Field(
        default=False,
        description="True if Tavily's findings contradict what the caller "
        "stated (used to populate fraud_signals.inconsistencies). The gate "
        "decides what to do with it — surface it in the UI, escalate, etc.",
    )
    answer: str | None = Field(
        default=None,
        description="Tavily's optional synthesized answer (when "
        "include_answer was set). Useful for richer summaries.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Top-3 source URLs that grounded the result. Operator "
        "can click through to verify if needed.",
    )


# ─── The fact-checker ───────────────────────────────────────────────


class TavilyFactChecker:
    """Async wrapper around Tavily's search API, scoped to insurance-claim
    fact-check use cases. Holds one AsyncTavilyClient for the call's
    lifetime. Per Tavily docs, the async client uses httpx under the hood
    and is safe for concurrent searches.

    Construction reads TAVILY_API_KEY from the environment if not passed
    explicitly. If the key is missing, every method raises before doing
    network IO so misconfiguration fails loudly during dev.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import so the rest of the backend doesn't blow up on
        # `from . import tavily` when tavily-python isn't installed yet.
        try:
            from tavily import AsyncTavilyClient
        except ImportError as e:
            raise RuntimeError(
                "tavily-python is not installed. To enable Tavily fact-"
                "checking, add `tavily-python>=0.5` to pyproject.toml and "
                "run `uv sync`. See module docstring for the full wiring "
                "checklist."
            ) from e

        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError(
                "TAVILY_API_KEY not set. Either pass api_key= explicitly "
                "or export TAVILY_API_KEY in your environment / Modal secret."
            )
        self._client = AsyncTavilyClient(api_key=key)

    async def aclose(self) -> None:
        """Release the underlying httpx client. Safe to call multiple
        times. Pair with `async with` if you want auto-cleanup."""
        try:
            await self._client.close()
        except Exception:
            pass

    async def __aenter__(self) -> "TavilyFactChecker":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.aclose()

    # ── weather_at ───────────────────────────────────────────────────

    async def weather_at(
        self,
        location: str,
        when: datetime,
        caller_described_as: str | None = None,
    ) -> FactCheckResult:
        """Corroborate weather conditions at a specific location and time.

        `caller_described_as` is what the caller said about the weather
        ("icy roads", "heavy rain") so we can flag inconsistencies. If
        omitted, just returns a summary without a fraud signal.

        Tavily's general search isn't a weather API — it returns web
        results that *mention* weather. For demo this is fine; for
        production we'd swap to OpenWeather or similar deterministic API.
        """
        when_local = when.astimezone() if when.tzinfo else when
        date_str = when_local.strftime("%Y-%m-%d")
        # Round to nearest hour for the query — exact minute is overkill
        hour_str = when_local.strftime("%H:00")
        query = f"weather in {location} on {date_str} around {hour_str}"

        response = await self._client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=3,
            include_answer="basic",
            timeout=10,
        )

        answer = response.get("answer") or ""
        sources = [r["url"] for r in response.get("results", [])[:3]]
        summary = answer.strip() if answer else (
            response.get("results", [{}])[0].get("content", "")[:120]
        )
        summary = summary[:120].strip() if summary else "no clear weather data found"

        # Heuristic inconsistency check: very rough — looks for opposing
        # weather words. Production would parse properly or use a weather
        # API directly. Acceptable for demo where the signal is "Tavily
        # said dry but caller said icy" → operator sees the chip and decides.
        inconsistency = False
        if caller_described_as:
            inconsistency = _weather_terms_disagree(
                caller_described_as.lower(), summary.lower()
            )

        return FactCheckResult(
            kind="weather",
            query=query,
            summary=summary,
            inconsistency_signal=inconsistency,
            answer=answer or None,
            sources=sources,
        )

    # ── verify_location ──────────────────────────────────────────────

    async def verify_location(self, address: str) -> FactCheckResult:
        """Quick existence check on a free-text address. Useful when the
        caller mumbles a place name we've never heard of — Tavily either
        returns rich results (real place) or thin/empty results (fabricated
        or extremely obscure).
        """
        query = f'"{address}"'  # quoted to bias toward exact match
        response = await self._client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=3,
            include_answer=False,
            exact_match=True,
            timeout=10,
        )
        results = response.get("results", [])
        sources = [r["url"] for r in results[:3]]

        if not results:
            summary = f"no clear references to '{address}' found"
            inconsistency = True  # absence is suspicious
        else:
            top = results[0]
            summary = (
                f"location plausible — top result: "
                f"{top.get('title', '')[:60]}"
            )
            inconsistency = False

        return FactCheckResult(
            kind="location",
            query=query,
            summary=summary,
            inconsistency_signal=inconsistency,
            answer=None,
            sources=sources,
        )

    # ── news_check ───────────────────────────────────────────────────

    async def news_check(
        self,
        location: str,
        when: datetime,
        keywords: str = "traffic accident",
    ) -> FactCheckResult:
        """Search news for traffic/incident reports near the given
        location and date. If a major multi-car accident actually
        happened, news outlets often cover it — corroboration. Absence
        doesn't mean fabrication (most claims aren't newsworthy), so
        we DON'T set inconsistency_signal here.
        """
        when_local = when.astimezone() if when.tzinfo else when
        # ±1 day window — accident news may be reported the next day
        start = (when_local - timedelta(days=1)).strftime("%Y-%m-%d")
        end = (when_local + timedelta(days=1)).strftime("%Y-%m-%d")

        query = f"{keywords} {location} {when_local.strftime('%Y-%m-%d')}"
        response = await self._client.search(
            query=query,
            search_depth="basic",
            topic="news",
            max_results=3,
            include_answer="basic",
            start_date=start,
            end_date=end,
            timeout=10,
        )

        results = response.get("results", [])
        answer = response.get("answer") or ""
        sources = [r["url"] for r in results[:3]]

        if results:
            summary = (
                answer.strip()[:120]
                if answer
                else f"{len(results)} news result(s); top: {results[0].get('title', '')[:60]}"
            )
        else:
            summary = "no related news coverage found in the date window"

        return FactCheckResult(
            kind="news",
            query=query,
            summary=summary,
            inconsistency_signal=False,  # absence here is normal
            answer=answer or None,
            sources=sources,
        )


# ─── Heuristics ────────────────────────────────────────────────────


_WET_TERMS = {"rain", "rainy", "wet", "shower", "drizzle", "storm", "snow", "snowy", "ice", "icy", "sleet", "hail"}
_DRY_TERMS = {"clear", "sunny", "dry", "fair", "cloudless"}


def _weather_terms_disagree(caller_text: str, found_text: str) -> bool:
    """Cheap textual contradiction check — caller mentioned wet/icy
    conditions but Tavily's summary mentions dry/clear (or vice versa).
    False positives are acceptable; this is a flag for the operator,
    not an automatic decision.
    """
    caller_wet = any(t in caller_text for t in _WET_TERMS)
    caller_dry = any(t in caller_text for t in _DRY_TERMS)
    found_wet = any(t in found_text for t in _WET_TERMS)
    found_dry = any(t in found_text for t in _DRY_TERMS)
    if caller_wet and found_dry and not found_wet:
        return True
    if caller_dry and found_wet and not found_dry:
        return True
    return False


# ─── Smoke test ─────────────────────────────────────────────────────
# Run with:  uv run python -m backend.app.reasoner.tavily
# Requires TAVILY_API_KEY in the environment. Hits the live API.

if __name__ == "__main__":
    import asyncio

    async def _smoke() -> None:
        # Hardcoded scenario: caller says they slid on icy roads at
        # Delta Campus in Berlin yesterday morning. We check weather,
        # verify the location, and look for matching news.
        when = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0) - timedelta(days=1)
        location = "Delta Campus, Berlin"
        caller_weather_claim = "icy roads, lost control"

        async with TavilyFactChecker() as checker:
            print("=" * 64)
            print(f"weather_at({location!r}, {when.isoformat()})")
            print(f"  caller said: {caller_weather_claim!r}")
            print("=" * 64)
            r = await checker.weather_at(location, when, caller_weather_claim)
            print(f"  query:        {r.query}")
            print(f"  summary:      {r.summary}")
            print(f"  inconsistency: {r.inconsistency_signal}")
            print(f"  sources:      {r.sources}")
            print()

            print("=" * 64)
            print(f"verify_location({location!r})")
            print("=" * 64)
            r = await checker.verify_location(location)
            print(f"  query:        {r.query}")
            print(f"  summary:      {r.summary}")
            print(f"  inconsistency: {r.inconsistency_signal}")
            print(f"  sources:      {r.sources}")
            print()

            print("=" * 64)
            print(f"news_check({location!r}, {when.date().isoformat()})")
            print("=" * 64)
            r = await checker.news_check(location, when)
            print(f"  query:        {r.query}")
            print(f"  summary:      {r.summary}")
            print(f"  sources:      {r.sources}")

    asyncio.run(_smoke())
