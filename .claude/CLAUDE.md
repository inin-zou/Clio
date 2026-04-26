# Clio — Project Context for Claude Code

Read this first in every session. Detailed analysis docs live in `.claude/docs/`; this file is the always-loaded orientation.

## What this is

Voice agent for inbound insurance claim calls, built for the [Inca hackathon](https://www.get-inca.com/en). The agent must pass jurors' "human or AI" vote (>50% human), capture full FNOL documentation in a 5-minute call, and stay coherent under highway noise / dialect variation.

The agent is **Sarah** — Allianz Berlin claims rep, 8 years on the desk. English only.

## Where we are right now (TL;DR)

**End-to-end calls work.** Twilio inbound → LiveKit SIP → Modal GPU PersonaPlex → server-side audio gating → caller hears Sarah → Scribe transcribes caller → backend orchestrator runs Haiku slot extractor → gate fires read-backs → Supabase persists everything → Next.js UI shows it live.

Most recent verified call: ~3 minutes, 5 critical FNOL slots captured, two read-backs auto-confirmed, no OOM, no talkover, natural close.

The system is **demo-able**. Iteration now is on rough edges: occasional Sarah hallucination, GPU memory plateau under back-to-back calls, multi-call concurrency.

If you're new to this codebase, **read `.claude/gotchas.md` before changing anything in `model_service/deploy/modal_app.py` or `backend/app/reasoner/`**. It's the war-stories doc — every section is a failure mode we hit and the actual fix.

## Stack (locked)

| Layer | Choice |
|---|---|
| Voice model | **PersonaPlex 7B** off-the-shelf — Moshi-derived, full-duplex, public `LMGen.step(text_token=...)` injection API |
| Telephony ingress | Twilio DID (US, +12183048451) + SIP → LiveKit Cloud |
| Real-time media | LiveKit Cloud (WebRTC) — fixed-room dispatch (`clio-active`) for single-call demos |
| **GPU runtime** | **Modal A100 40GB** — PersonaPlex + Mimi (×2) co-located in ONE container (app: `personaplex-clio`) |
| **CPU backend** | **Modal `@asgi_app`** — FastAPI + orchestrator + Twilio webhook + Supabase writer (app: `clio-backend`) |
| ASR (caller transcript) | **ElevenLabs Scribe v2 Realtime** — wired up; produces clean caller transcripts that feed the slot extractor |
| Slot extractor | Anthropic Claude Haiku 4.5 (tool-use API, structured output) |
| Persistent state / live UI feed | **Supabase** — `calls`, `messages`, `events` tables + Realtime channels |
| Frontend | Next.js 15 + Tailwind, deployed to Vercel — subscribes to Supabase Realtime, no direct backend coupling |

## The architectural fact most likely to be re-derived wrong

**Audio NEVER traverses the backend ↔ model_service WebSocket.** It goes:

```
Caller phone → Twilio → LiveKit Cloud → WebRTC → Modal A100 container
                                                  ├─ Mimi codec (×2)
                                                  ├─ PersonaPlex 7B
                                                  └─ energy-VAD + drip queue
                                                  (in-process Python)
```

Backend (Reasoner) talks to Modal only via a JSON control-plane WS — directives in, transcripts/lifecycle out. Backend is **never on the audio path**.

Why: a backend-relay design adds 60-200ms per audio frame round-trip, eliminating PersonaPlex's 80ms inference advantage. See `.claude/docs/architecture.md`.

## What's done vs in flight

### Verified live ✅

| Component | Notes |
|---|---|
| Inbound call → Sarah responds → caller transcribed → FNOL filled → UI updates | Multiple successful end-to-end calls, latest ~3min with 5 critical slots captured |
| `model_service/deploy/modal_app.py` (`personaplex-clio` app) | PersonaPlex 7B, Mimi×2, energy-VAD, drip queue, server-side audio mute on caller speech, Scribe streaming, snapshot/restore for sub-second per-call ready |
| `model_service/deploy/modal_backend.py` (`clio-backend` app) | Modal CPU `@asgi_app` shipping the FastAPI orchestrator + Twilio webhook |
| `backend/app/reasoner/` | FNOL state, persona, drip directives, gate (3 triggers + cooldown + multi-slot readback queue) |
| `backend/app/control/` | Orchestrator, SSE/EventBus (legacy), Supabase writer, registry |
| `backend/app/telephony/` | `twilio_webhook.py`, `livekit_sip.py` (mints room + agent JWT, fixed-room mode) |
| `frontend/` | Next.js 15 App Router, Clio.zip-based 3-pane glass design, Supabase Realtime subscription, bg.png + favicon |
| `db/supabase_schema.sql` | 3 tables (calls / messages / events), realtime publication, anon-read RLS |
| Modal secrets | `hf-token`, `clio-livekit` (incl. `LIVEKIT_SIP_URI`), `clio-anthropic`, `clio-elevenlabs`, `clio-twilio`, `clio-backend-cfg` (incl. `SUPABASE_*`) |
| LiveKit SIP | Inbound trunk + Direct dispatch rule (`roomName: clio-active`) configured |
| Twilio | DID `+12183048451` pointed at `https://dreamonzouk--clio-backend-fastapi-app.modal.run/twilio/voice` + `/twilio/status` |

### Known rough edges (working but not perfect) ⚠️

| Issue | Status | Notes |
|---|---|---|
| GPU memory leak ~1.75GB/min during calls | Mitigated by `TORCHDYNAMO_DISABLE=1` env var; calls comfortably go 5+ min | CUDA-graph pinning; `_recreate_inference_stack` at end-of-call rebuilds wrappers; for absolute reset, `modal app stop personaplex-clio` between demos |
| Sarah occasionally hallucinates plates / policy numbers | Mitigated by persona "NEVER guess identifiers" rule + extractor strict-anchor rule | Won't be 100% — some calls still drift |
| Multi-concurrent calls | Not supported | Fixed-room dispatch + single-container backend = one call at a time |
| Whisper / Take over UI buttons | Disabled | TIER 3 features; would need operator-controlled SpeakDirective endpoint + LiveKit room handover |
| Voice waveform visualization | Decorative SVG sines, not real audio levels | TIER 4 polish |

### Deferred / not-blocking 🚧

| Component | Notes |
|---|---|
| Rescue clip pipeline (pre-recorded "you're breaking up" WAVs) | ~50 lines + recording session; not blocking demo |
| Frontend transcript entity highlighting (colored chips) | Either frontend regex over slot values or backend attaches entities to messages at insert time |
| Recent-claims status state machine (FRAUD / ESCALATED / PUSHED) | Backend post-call classification rules + new `status` column |
| Tools (synthetic dialogue gen, eval runner) | Not on critical path |

## Repo layout

```
Clio/
├── pyproject.toml              uv project for backend/
├── uv.lock
├── .env                        gitignored; real secrets
├── .env.example                committed; documents required keys (incl. SUPABASE_*)
├── README.md                   architecture, latency story, deploy steps
│
├── backend/
│   ├── app/
│   │   ├── reasoner/           FNOL state machine (CPU)
│   │   │   ├── schema.py       Pydantic — PolicyContext, ClaimReport, FNOLSession
│   │   │   ├── taxonomy.py     enums (IncidentType, ReporterRole, KaskoType, ...)
│   │   │   ├── db.py           mock_policies.json lookup
│   │   │   ├── persona.py      Sarah BASE_PERSONA + session_system_prompt(now=)
│   │   │   ├── extractor.py    Anthropic Haiku tool-use slot extractor (strict ID anchor rule)
│   │   │   ├── state.py        Session lifecycle + auth + readback merge logic
│   │   │   ├── drip.py         control-plane directive types + read-back rendering (datetime-aware)
│   │   │   └── gate.py         3 intervention triggers + cooldown + multi-attempt readbacks
│   │   ├── control/
│   │   │   ├── messages.py     WS wire types (re-exports drip directives)
│   │   │   ├── orchestrator.py CallOrchestrator — bridges WS ↔ reasoner.Session, multi-slot pending readback queue, Supabase writes
│   │   │   ├── server.py       FastAPI app: /control/{call_id} WS, /twilio/*, /events/*, /ui (legacy SSE)
│   │   │   ├── registry.py     PendingCall registry (extracted to break circular import with telephony)
│   │   │   ├── eventbus.py     in-process pub/sub (legacy SSE; Supabase is the durable channel)
│   │   │   ├── sse.py          legacy SSE endpoints + standalone HTML UI
│   │   │   └── supabase_writer.py  insert calls/messages/events; fail-soft if env missing
│   │   └── telephony/
│   │       ├── twilio_webhook.py  POST /twilio/voice + /twilio/status
│   │       └── livekit_sip.py     mint room name + agent JWT (fixed-room mode)
│   └── tests/
│       ├── conftest.py
│       ├── test_control_orchestrator.py
│       └── test_twilio_webhook.py   (12 tests pass total)
│
├── model_service/              SEPARATE uv project (heavy GPU deps)
│   ├── pyproject.toml
│   ├── server/                 mock-talker stack for protocol dev (legacy, still passes its 5 tests)
│   └── deploy/
│       ├── modal_app.py        GPU app — PersonaPlex + Mimi×2 + VAD + drip + Scribe + snapshot/restore
│       │                       + persona prompt (kept in sync with backend persona.py BASE_PERSONA)
│       └── modal_backend.py    CPU app — wraps backend FastAPI as @modal.asgi_app
│
├── frontend/                   Next.js 15 App Router (Vercel-ready)
│   ├── app/
│   │   ├── layout.tsx          loads Inter + JetBrains_Mono via next/font
│   │   ├── page.tsx            3-pane glass dashboard, Supabase Realtime subscription
│   │   ├── globals.css         Clio.zip design tokens + .glass + .voiceline + transcript styles
│   │   └── favicon.ico
│   ├── lib/
│   │   ├── supabase.ts         lazy-init anon client
│   │   ├── types.ts            Call, Message, EventRow
│   │   └── derive.ts           TIER 1 derivations: callerName, fnolPercent, elapsedLabel, ...
│   ├── public/bg.png           painterly background from Clio.zip
│   ├── package.json            next 15.5, react 19, @supabase/supabase-js
│   └── README.md
│
├── db/
│   └── supabase_schema.sql     calls / messages / events tables + RLS + realtime publication
│
├── data/
│   ├── mock_policies.json      5 sample insurance policies
│   └── sessions/               (gitignored) end-of-call JSON dumps
│
└── .claude/
    ├── CLAUDE.md               this file
    ├── gotchas.md              ★ failure-mode war stories — READ BEFORE CHANGING MODEL CODE
    ├── docs/
    │   ├── architecture.md     runtime shape, latency budget, deployment
    │   ├── architecture-decision.md  why we chose this path
    │   ├── roadmap.md          three-MVP plan + decision log
    │   ├── fnol-schema.md      Inca's spec mapped to our two-layer schema
    │   ├── livekit-sip-setup.md  one-time `lk sip` commands
    │   ├── moshirag-analysis.md
    │   └── aspirin-analysis.md
    ├── settings.json
    └── settings.local.json
```

## Conventions

### Python
- **`uv run` for everything**: `uv run python -m ...`, `uv run pytest`, `uv run ruff check`.
- **Pydantic v2** schemas; `model_dump(mode="json")` when serializing for WS / disk.
- **`async`** for I/O. Slot extractor is async; per-frame loop is async.
- **Types**: `T | None`, `list[T]`, `dict[str, T]` (3.10+ syntax).
- **No comments unless WHY is non-obvious.** Don't narrate WHAT the code does.

### Modules
- Two separate uv projects: `Clio/` (backend) and `Clio/model_service/`. Don't try to share imports across the boundary — JSON over WS is the contract.
- `backend/app/control/messages.py` re-exports `reasoner/drip.py` directive types. Single source of truth for the wire format.
- `BAKED_PERSONA` in `model_service/deploy/modal_app.py` MUST stay in sync with `BASE_PERSONA` in `backend/app/reasoner/persona.py`. The Modal one is what gets baked into the snapshot at container start.

### Testing
- Each `backend/app/reasoner/*.py` has a `__main__` smoke test runnable via `uv run python -m backend.app.reasoner.{module}`.
- `uv run pytest backend/tests/` — 12 tests pass.
- Tests that DON'T hit Anthropic API monkeypatch `extractor.SlotExtractor.extract`. Save real-API calls for explicit live tests.

### Git / Entire
- **Entire is configured.** Every `git push` captures the Claude Code session. View with `entire explain <sha>`.
- **No PR workflow** — single contributor, push to `main` directly.
- **Commit messages**: explain what changed and why. Use HEREDOC for multi-paragraph messages.

### Modal
- **Two apps**: `personaplex-clio` (GPU) and `clio-backend` (CPU `@asgi_app`).
- **Deploy GPU** (always-warm A100, ~$26/day): `CLIO_DEMO_MODE=1 modal deploy model_service/deploy/modal_app.py`
- **Deploy backend**: `modal deploy model_service/deploy/modal_backend.py`
- **`modal deploy` does NOT swap the warm container automatically.** Use `modal app stop personaplex-clio` first to force fresh-container restart with new code.
- **Stop billing when not demoing**: `modal app stop personaplex-clio`. Backend CPU is cheap, leave running.
- **Workspace**: `dreamonzouk`.
- **Secrets**: `hf-token`, `clio-livekit`, `clio-anthropic`, `clio-elevenlabs`, `clio-twilio`, `clio-backend-cfg`.
- **HF cache volume**: `hf-cache` — persistent. Do NOT delete.
- **Don't use `max_containers=N`** — not a valid `@app.function` parameter in this Modal version.

### Frontend (Next.js)
- Run from `frontend/`: `npm run dev` (port 3000) or `npm run build` for prod check.
- After major page.tsx changes, **`rm -rf .next` before `npm run dev`** — stale webpack chunks cause `__webpack_modules__[moduleId] is not a function`.
- Reads Supabase via `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` (auto-generated from parent `.env` into `frontend/.env.local`).
- All TIER 1 UI derivations (caller name, FNOL %, elapsed time, incident-type label) live in `frontend/lib/derive.ts`. Add new ones there, not in `page.tsx`.

## Decisions deliberately avoided

These look tempting but are intentionally NOT in scope. Don't propose them without reason:

- **No model training** in MVP. ASPIRin RL fine-tune is a stretch goal in roadmap.md.
- **No fork of PersonaPlex.** Public `LMGen.step(text_token=...)` API + drip-feed pattern is sufficient.
- **No multi-persona.** One Sarah only.
- **No active inject-on-every-turn.** Reasoner is a passive observer; intervenes only via 3 gate triggers (read-back / compliance / wrap-up).
- **No backend on audio path.** See "the architectural fact" above.
- **No Pipecat unless turn detection becomes a problem.** LiveKit Agents is sufficient.
- **No real RAG (vector DB retrieval).** Policy data is structured (`PolicyContext`), not text-corpus. The drip-feed mechanism IS RAG-shaped but we use it for gate-driven interventions, not retrieved knowledge.

## Persona / read-back protocol (the human-passing trick)

Sarah's prompt enforces an **entity-verification read-back**: when the caller says a policy number, plate, date, name, or amount, Sarah waits for them to finish, then reads it back ("so that's POL dash 2024 dash 001, is that right?"). Then she **STOPS TALKING** and waits for explicit confirmation (`"yes" / "correct" / "yeah"`) before moving on.

The full prompt is in `backend/app/reasoner/persona.py:BASE_PERSONA` (also baked into `BAKED_PERSONA` in `modal_app.py`).

Read-back rendering knows about field types — datetimes render as natural English ("5 a.m. on April 26th"), IDs spell out ("P O L dash 2 0 2 4 dash 0 0 1"), names speak naturally. See `drip.render_readback`.

## Latency budget at a glance

```
Caller phone ─→ Twilio                    ~100-150ms
Twilio ─→ LiveKit Cloud (SIP)             ~10-50ms
LiveKit Cloud ─→ Modal A100 (WebRTC)      ~30-80ms
PersonaPlex inference                     ~80ms
Mimi encode/decode                        ~10ms
                                          ─────
                                  Total:  ~220-320ms one-way
                                          ~450-650ms round-trip
```

Below the 800ms "feels human" production threshold. Cold-start setup ~70s on A100; mitigated by `CLIO_DEMO_MODE=1` (`min_containers=1`) + persona snapshot/restore (per-call ready in ~1-2s instead of ~30s).

See README.md "How we got latency under 800ms" for the full breakdown of what we did to hit it.

## Mental model: control plane is JSON, audio plane is WebRTC, persistence is Supabase

- **Audio plane**: caller → LiveKit → Modal direct (WebRTC). PersonaPlex inference happens in-process inside the same Modal container. Audio frames never see backend.
- **Control plane**: backend ↔ Modal over WebSocket. JSON only.
  - Backend → Modal: `SpeakDirective`, `SilenceDirective`, `ReleaseDirective`, `RescueClipDirective`, `LoadPolicyContextDirective`, `SessionStart/End`
  - Modal → Backend: `TranscriptTurn` (source: `personaplex` or `scribe`), `CallerTurnBoundary`, `ReadbackOutcome`, `SessionReady/Closed`
- **Persistent + UI feed**: Supabase. Backend writes `calls`, `messages`, `events` rows. Frontend subscribes via Realtime. No backend ↔ frontend coupling beyond the database.

## When you're stuck

1. **Bug in audio / model behavior?** Read `.claude/gotchas.md` first. 14 sections of "we hit this, here's the fix."
2. **Architecture question?** `.claude/docs/architecture.md` and `.claude/docs/architecture-decision.md`.
3. **What's the priority?** `.claude/docs/roadmap.md`.
4. **What slot does X go in?** `.claude/docs/fnol-schema.md` and `backend/app/reasoner/schema.py`.
5. **Wire format?** `backend/app/reasoner/drip.py` (the directive types) and `backend/app/control/messages.py` (the inbound types).

If proposing a change that touches the audio path or the WS protocol, **double-check the "audio doesn't go over WS" rule** before suggesting it. If proposing a change to model-tuning constants (VAD thresholds, gate cooldown, persona prompt rules), **check `.claude/gotchas.md` first** — most knobs have a "we tried this, here's what broke" entry.
