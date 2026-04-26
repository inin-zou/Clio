# Clio — Project Context for Claude Code

Read this first in every session. Detailed analysis docs live in `.claude/docs/`; this file is the always-loaded orientation.

## What this is

Voice agent for inbound insurance claim calls, built for the [Inca hackathon](https://www.get-inca.com/en). The agent must pass jurors' "human or AI" vote (>50% human), capture full FNOL documentation in a 5-minute call, and stay coherent under highway noise / dialect variation.

The agent is **Sarah** — Allianz Berlin claims rep, 8 years on the desk. English only.

## Where we are right now (TL;DR)

The architecture is fully wired up in code but the **first end-to-end phone call has never been made**. The remaining critical path:

1. Write the **Twilio inbound webhook** in `backend/app/telephony/` (mints LiveKit room+token, registers PendingCall, spawns Modal `process_call.spawn(...)`, returns TwiML).
2. Expose backend on a public URL (ngrok for dev, Modal CPU / Render for prod) so Modal can WS-connect back.
3. Configure Twilio Console's "A Call Comes In" webhook to that URL.
4. Make a real call — debug whatever blows up.

Everything below the architecture-decision and reasoner layer is solid; the unknown is whether the LiveKit/PersonaPlex/audio plumbing in `model_service/deploy/modal_app.py:process_call()` works in practice. We've verified the imports, syntax, and `setup()`; the per-frame loop hasn't seen real audio.

## Stack (locked)

| Layer | Choice |
|---|---|
| Voice model (Talker) | **PersonaPlex 7B** off-the-shelf — Moshi-derived, full-duplex, public `LMGen.step(text_token=...)` injection API |
| Telephony ingress | Twilio DID (EU) + SIP trunk → LiveKit Cloud |
| Real-time media | LiveKit Cloud (WebRTC, EU edge) |
| **GPU runtime** | **Modal A100 40GB** — LiveKit Agent + PersonaPlex co-located in ONE Python process |
| ASR (verification) | ElevenLabs Scribe v2 Realtime — always-on, feeds extractor context, used for entity verification |
| Slot extractor | Anthropic Claude Haiku 4.5 (tool-use API, structured output) |
| Backend (Reasoner) | Python 3.11+, CPU-only, runs anywhere — communicates with Modal over control-plane WS |
| Frontend | Next.js 15 + Tailwind + shadcn/ui (monitoring UI; not on critical path) |

## The architectural fact most likely to be re-derived wrong

**Audio NEVER traverses the backend ↔ model_service WebSocket.** It goes:

```
Caller phone → Twilio → LiveKit Cloud → WebRTC → Modal A100 container
                                                  ├─ LiveKit Agent (Python)
                                                  └─ PersonaPlex 7B
                                                  (in-process call)
```

Backend (Reasoner) talks to Modal only via a JSON control-plane WS — directives in, transcripts/lifecycle out. Backend is **never on the audio path**.

Why: a backend-relay design adds 60-200ms per audio frame round-trip, eliminating PersonaPlex's 80ms inference advantage. See `.claude/docs/architecture.md` for the full latency analysis.

## What's done vs in flight

### Verified live ✅
| Component | Notes |
|---|---|
| `.claude/docs/` (architecture, roadmap, FNOL schema, MoshiRAG/ASPIRin analyses) | source of truth for design |
| `backend/app/reasoner/` (db, schema, persona, extractor, state, drip, gate, taxonomy) | each module has `__main__` smoke tests |
| `backend/app/control/` (orchestrator + WS server) | 6 unit tests pass; FastAPI server boots cleanly |
| `model_service/server/` (mock-talker stack for protocol dev) | 5 tests pass |
| `model_service/deploy/modal_app.py` — image build + `setup()` | PersonaPlex 7B + 18 voice prompts (NATF/NATM/VARF/VARM) load in 56s on A100 |
| Modal deployed app | `personaplex-clio` on `dreamonzouk` workspace; secrets attached |
| Modal secrets | `hf-token`, `clio-livekit`, `clio-anthropic`, `clio-elevenlabs` |
| `.env` populated locally per `.env.example` | all 9 keys present |
| Live extractor end-to-end test | Anthropic Haiku 4.5 returns valid Pydantic-validated slot updates from a sample FNOL transcript |

### Code written but NOT YET tested with real audio ⚠️
The whole inference loop in `process_call()` has been deployed to Modal (compiles + setup runs) but has never been exercised with actual caller audio. Possible bugs hiding:
- `lm_gen.load_voice_prompt_embeddings(.pt path)` — exact API shape unconfirmed
- `lm_gen.text_prompt_tokens = list[int]` vs tensor format
- LiveKit Agent joining a SIP-trunk-sourced room (vs WebRTC participant)
- LiveKit's actual frame size delivery (we buffer for non-1920 chunks but assumption is untested)
- VAD threshold (0.005 RMS, 800ms silence) under real PSTN audio levels
- Forced `text_token` tensor dtype (`long` chosen; not validated)
- Backend ↔ Modal WebSocket connectivity from inside Modal container

### 🚧 Not yet written (blocks first end-to-end test)
| Component | Status |
|---|---|
| **Backend Twilio webhook handler** (`/twilio/voice`, `/twilio/status`) | ✅ written (`backend/app/telephony/twilio_webhook.py`), 6 unit tests pass |
| **LiveKit SIP inbound trunk + dispatch rule** | ⏳ run commands in `.claude/docs/livekit-sip-setup.md`. Required to populate `LIVEKIT_SIP_URI` in `.env` |
| **Backend deployed to public URL** (so Modal can reach it): ngrok for dev, Modal CPU / Render for prod | ⏳ ~30min. Set `BACKEND_PUBLIC_WS_URL=wss://<host>` in `.env` |
| **Twilio Console webhook URL** pointed at backend `/twilio/voice` + `/twilio/status` | ⏳ 5min |
| **First end-to-end test call** — debug whatever blows up | ⏳ unknown |
| **Modal `CLIO_DEMO_MODE=1` redeploy** before judging | ⏳ 1 command |

### 🚧 Not yet written (demo polish, not blocking)
| Component | Notes |
|---|---|
| **ReadbackOutcome** detection — pattern-match caller's response after a Sarah read-back into confirmed/corrected/unclear | ~80 lines in `process_call`'s post-readback logic |
| **Rescue clip pipeline** — pre-record 6-10 PersonaPlex-voice WAVs, server-side audio gating during playback | ~50 lines + recording session |
| **Twilio TwiML `<Say>` preamble** — "Thank you for calling Allianz, please hold" while Modal warms up | included with Twilio webhook handler |
| **Frontend monitoring UI** — Next.js orb + transcript stream + FNOL state | demo can run from terminal logs alone |
| **Tools** (synthetic dialogue gen, eval runner) | not on critical path |
| **Scribe v2 ASR side-channel** (Step 4b) — entity verification backchannel; deferred because read-back protocol covers entity recall already | re-evaluate after first real call |

## Repo layout

```
Clio/
├── pyproject.toml              uv project for backend/
├── uv.lock
├── .env                        gitignored; real secrets
├── .env.example                committed; documents required keys
├── .mcp.json                   Context7 MCP for library docs
├── .entire/                    Entire CLI (auto AI-session capture on git push)
│
├── backend/
│   ├── app/
│   │   ├── reasoner/           FNOL state machine (CPU)
│   │   │   ├── schema.py       Pydantic — PolicyContext, ClaimReport, FNOLSession
│   │   │   ├── taxonomy.py     enums (IncidentType, ReporterRole, KaskoType, ...)
│   │   │   ├── db.py           mock_policies.json lookup, < 5μs
│   │   │   ├── persona.py      Sarah BASE_PERSONA + session_system_prompt(now=)
│   │   │   ├── extractor.py    Anthropic Haiku tool-use slot extractor
│   │   │   ├── state.py        Session lifecycle + auth + readback merge logic
│   │   │   ├── drip.py         control-plane directive types (Pydantic)
│   │   │   └── gate.py         3 intervention triggers (readback, compliance, wrap-up)
│   │   ├── control/
│   │   │   ├── messages.py     WS wire types (re-exports drip directives)
│   │   │   ├── orchestrator.py CallOrchestrator — bridges WS ↔ reasoner.Session
│   │   │   └── server.py       FastAPI app with /control/{call_id} WS endpoint
│   │   └── telephony/          (NOT YET WRITTEN — Twilio webhook handler goes here)
│   └── tests/
│       ├── conftest.py         loads .env so SlotExtractor can init
│       └── test_control_orchestrator.py  6 tests pass
│
├── model_service/              SEPARATE uv project (heavy GPU deps)
│   ├── pyproject.toml
│   ├── server/                 mock-talker stack (local dev / protocol tests)
│   │   ├── protocol.py
│   │   ├── talker.py           Talker Protocol + MockTalker + PersonaPlexTalker stub
│   │   ├── session.py          per-call orchestrator with drip-feed
│   │   └── main.py             websockets server entry
│   ├── tests/
│   │   └── test_mock_session.py  5 tests, all pass
│   └── deploy/
│       └── modal_app.py        Modal class — A100, image, setup() (verified live),
│                               process_call() with steps 1-3 + 4a + 5 + 6 wired in:
│                                  - control WS handshake
│                                  - LiveKit room join + agent track publish
│                                  - persona priming (voice + system prompt)
│                                  - per-frame Mimi.encode → LMGen.step → Mimi.decode
│                                  - DripState consumed for forced text_token
│                                  - background WS receiver for ReasonerDirective updates
│                                  - energy-VAD turn boundary signaling
│
├── frontend/                   Next.js (not started)
├── data/
│   ├── mock_policies.json      5 sample insurance policies
│   └── sessions/               (gitignored) end-of-call JSON dumps
├── tools/                      utility scripts (dialogue gen, eval) — not started
└── .claude/
    ├── CLAUDE.md               this file
    ├── docs/                   detailed analysis (read these for depth)
    │   ├── architecture.md     runtime shape, latency budget, deployment
    │   ├── roadmap.md          three-MVP plan + decision log
    │   ├── architecture-decision.md  why we chose this path
    │   ├── fnol-schema.md      Inca's spec mapped to our two-layer schema
    │   ├── moshirag-analysis.md      what we took conceptually
    │   └── aspirin-analysis.md       what we took conceptually
    └── settings.json           project Claude Code config
```

## Conventions

### Python
- **`uv run` for everything**: `uv run python -m ...`, `uv run pytest`, `uv run ruff check`.
- **Pydantic v2** schemas; `model_dump(mode="json")` when serializing for WS / disk.
- **`async`** for I/O (Anthropic, WebSocket, LiveKit). Slot extractor is async; per-frame loop is async.
- **Types**: `T | None`, `list[T]`, `dict[str, T]` (3.10+ syntax).
- **No comments unless WHY is non-obvious.** Don't narrate WHAT the code does.

### Modules
- Two separate uv projects: `Clio/` (backend) and `Clio/model_service/`. Don't try to share imports across the boundary — JSON over WS is the contract.
- `backend/app/control/messages.py` re-exports `reasoner/drip.py` directive types. Single source of truth.
- `model_service/server/protocol.py` mirrors the JSON shape but is independent (no cross-package imports). If `drip.py` changes, update both.

### Testing
- Each `backend/app/reasoner/*.py` has a `__main__` smoke test runnable via `uv run python -m backend.app.reasoner.{module}`.
- Pytest tests in `backend/tests/` and `model_service/tests/`.
- `backend/tests/conftest.py` loads `.env` so SlotExtractor can init.
- Tests that DON'T hit Anthropic API monkeypatch `extractor.SlotExtractor.extract`. Save real-API calls for explicit live tests.

### Git / Entire
- **Entire is configured** (`brew install --cask entire` + `entire enable` already done). Every `git push` automatically captures the Claude Code session and links it to the commit. View with `entire explain <sha>`.
- **No PR workflow** — single contributor, push to `main` directly.
- **Commit messages**: explain what changed and why. Reference doc files when relevant. Use HEREDOC for multi-paragraph messages.

### Modal
- **One deploy command**: `modal deploy model_service/deploy/modal_app.py` (dev) or `CLIO_DEMO_MODE=1 modal deploy ...` (always-warm A100).
- **Stop billing when not demoing**: `modal app stop personaplex-clio`.
- **App name**: `personaplex-clio`. Workspace: `dreamonzouk`.
- **Secrets are referenced by name**: `hf-token`, `clio-livekit`, `clio-anthropic`, `clio-elevenlabs`.
- **HF cache volume**: `hf-cache` — persistent. PersonaPlex 7B weights already populated; do NOT delete.

## Decisions deliberately avoided

These look tempting but are intentionally NOT in scope. Don't propose them without reason:

- **No model training** in MVP1/MVP2. ASPIRin RL fine-tune is a stretch goal in roadmap.md.
- **No fork of PersonaPlex.** Public `LMGen.step(text_token=...)` API is sufficient.
- **No multi-persona.** One Sarah only.
- **No active inject-on-every-turn.** Reasoner is a passive observer; intervenes only via 3 gate triggers.
- **No primary independent ASR.** PersonaPlex's text monologue stream is the primary transcript; Scribe v2 is verification backchannel.
- **No backend on audio path.** See "the architectural fact" above.
- **No Pipecat unless turn detection becomes a problem.** LiveKit Agents is sufficient.

## Persona / read-back protocol (the human-passing trick)

Sarah's prompt enforces an **entity-verification read-back**: when the caller says a policy number, plate, date, name, or amount, Sarah waits for them to finish, then reads it back ("so that's POL dash 2024 dash 001, is that right?"). This:
- sounds human (real claims reps do this every time)
- buys natural latency budget
- catches transcription errors at the source (caller is the ground truth)

The full prompt is in `backend/app/reasoner/persona.py`. The `session_system_prompt(now=...)` adds Berlin-time-of-day context so Sarah greets correctly ("good morning" vs "good evening") and resolves "this morning" / "an hour ago" references properly.

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

Below the 800ms "feels human" production threshold. Cold-start setup measured at 59s on A100; mitigated by `CLIO_DEMO_MODE=1` (`min_containers=1`) during demo windows.

## Mental model: control plane is JSON, audio plane is WebRTC

- **Audio plane**: caller → LiveKit → Modal direct. PersonaPlex inference happens in-process inside the same Modal container. Audio frames never see backend.
- **Control plane**: backend ↔ Modal over WebSocket. JSON only. Carries:
  - Backend → Modal: `SpeakDirective`, `SilenceDirective`, `LoadPolicyContextDirective`, `RescueClipDirective`, `SessionStart/End`
  - Modal → Backend: `TranscriptTurn`, `CallerTurnBoundary`, `ReadbackOutcome`, `SessionReady/Closed`

Latency-tolerant; 100-200ms WS RTT is fine because directives are async and Modal reads from a local cache per-frame.

## When you're stuck on a design choice

Read in this order:
1. `.claude/docs/architecture.md` — what runs where
2. `.claude/docs/roadmap.md` — what's the priority + decision log
3. `.claude/docs/architecture-decision.md` — why we chose this path
4. `.claude/docs/fnol-schema.md` — what the Reasoner is trying to capture
5. The Pydantic schemas (`backend/app/reasoner/schema.py`) — the data contract

If proposing a change that touches the audio path or the WS protocol, **double-check the "audio doesn't go over WS" rule** before suggesting it.
