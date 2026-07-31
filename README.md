# Clio

Voice agent for inbound insurance claim calls. Built for the [2nd Big Berlin Hack](https://www.get-inca.com/en) — the agent must convince jurors they're talking to a human while collecting complete FNOL documentation.

## Why full-duplex, not ASR → LLM → TTS

Clio runs on **PersonaPlex 7B** (NVIDIA's fork of Kyutai's [Moshi](https://kyutai.org/Moshi.pdf)) — an end-to-end full-duplex speech LLM that consumes caller audio and emits agent audio **simultaneously**, frame-by-frame at 12.5Hz. No separate ASR / LLM / TTS stages. The model's text head and audio head run in lock-step every 80ms, so the agent can listen and speak at the same time the way humans actually do.

Same architectural bet as ByteDance's [**Seeduplex**](https://seed.bytedance.com/en/seeduplex) (announced **April 9, 2026**, deployed on Doubao at scale weeks before this hackathon) — that the way to feel human is to **remove turn-taking entirely**, not to optimize gaps in a half-duplex pipeline.

### Latency: 450-650ms round-trip vs 1.5-3.5s for half-duplex

| Stack | Round-trip | Why |
|---|---|---|
| **Half-duplex** (ASR → LLM → TTS, e.g. Pipecat / Vapi / classic Twilio voice) | **1.5-3.5 s** | Sequential: pause detection (500-1000ms) → ASR (200-500ms) → LLM generate (500-1500ms) → TTS synth (200-800ms). Each stage waits for the previous one to finish. |
| **Full-duplex** (Clio / PersonaPlex / Seeduplex / Moshi) | **~450-650 ms** | Single per-frame inference (~80ms), audio and text heads run together, no turn-taking gap. We measured this on the Modal A100 path — see [How we got latency under 800ms](#how-we-got-latency-under-800ms). |

Inca's "feels human" threshold is around 800ms round-trip. **A half-duplex stack would not clear that bar regardless of how fast each individual model is** — the sequential floor is too high. Full-duplex eliminates the floor.

### Tool calling / RAG: shaped by MoshiRAG + ASPIRin

The intervention gate + drip-feed pattern (`backend/app/reasoner/`) draws on two recent full-duplex papers (both 2026, both code-unreleased):

- [**MoshiRAG**](https://arxiv.org/abs/2604.12928) (Kyutai, Apr 2026) — answers *"how do you push external knowledge into a full-duplex model mid-utterance without making it pause?"* Their solution: a `⟨ret⟩` trigger token + async retrieval + lead-portion filler ("hmm, let me think...") that buys time while retrieval lands. Clio uses the same three-segment shape (lead → body → tail) but simpler: templated text injection via the drip queue instead of learned embedding retrieval. See [`.claude/docs/moshirag-analysis.md`](.claude/docs/moshirag-analysis.md).

- [**ASPIRin**](https://arxiv.org/abs/2604.10065) (NTU + NVIDIA, Apr 2026) — answers *"how do you RL-train a full-duplex model to learn when to speak without breaking what it says?"* Their insight: project the action space onto a binary {speak, silent} decision, RL only on that, leave token-content distribution untouched. We don't fine-tune (no RL in MVP), but the principle informs the architecture: **timing and intervention decisions live in the Python gate, not in the model's prompt**, so we never destabilize PersonaPlex's natural conversational fluency. See [`.claude/docs/aspirin-analysis.md`](.claude/docs/aspirin-analysis.md).

The Tavily fact-checker module (`backend/app/reasoner/tavily.py`) implements the MoshiRAG-shaped retrieval primitive — built but currently unwired for latency reasons (see "Optional: Tavily fact-checker" below).

See [`.claude/docs/architecture-decision.md`](.claude/docs/architecture-decision.md) for the full design rationale.

## High-level architecture

```
   Caller phone
        │
        ▼
   Twilio DID ───► /twilio/voice webhook (backend, Modal CPU)
        │                    │
        │                    └─ mints LiveKit room + agent token
        │                       spawns Modal GPU process_call
        │                       returns TwiML <Dial><Sip>
        ▼
   LiveKit Cloud SIP ◄── audio (WebRTC) ──► Modal GPU container
                                                │
                                                ├─ Mimi codec (×2)
                                                ├─ PersonaPlex 7B
                                                └─ energy-VAD + drip queue
                                                │
                          control-plane WebSocket (JSON, no audio)
                                                │
                                                ▼
                              Modal CPU container (backend FastAPI)
                                ├─ orchestrator + slot extractor (Haiku)
                                ├─ intervention gate
                                ├─ Twilio webhook
                                └─ Supabase writer ──► Supabase
                                                          │
                                                          └─► Realtime
                                                              channel
                                                                 │
                                                                 ▼
                                                          Next.js UI
                                                          (Vercel)
```

- **`model_service/`** — PersonaPlex 7B + Mimi codec inference + drip-feed token injection. Deployed to Modal **A100 80GB** (sized for the documented ~1.75GB/min CUDA-graph leak — peak per-call usage is ~33GB, which leaves comfortable headroom).
- **`backend/`** — FastAPI orchestrator (FNOL state, slot extractor, intervention gate, structural anchor filter, audio-rescue trigger), Twilio webhook, Supabase writer. Deployed to Modal CPU.
- **`frontend/`** — Next.js 15 dashboard (dark glass design). Three views:
  - **`/` Operations** — live transcript bubbles (caller left, agent right), editable claim draft (click to edit any FNOL field, writes back via Supabase RLS update policy), Send follow-up button.
  - **`/` Context Vault tab** — read-only mirror of Sarah's persona prompt, critical slots, compliance / wrap-up phrasings, gate timings, VAD thresholds.
  - **`/architecture`** — standalone tech-overview page: three-planes diagram, latency-budget bar chart vs half-duplex, stack comparison table, measured benchmarks. Not linked from the main nav (direct URL only).
  Subscribes to Supabase Realtime — no direct backend coupling. Deployable to Vercel.
- **`db/`** — Supabase schema (`supabase_schema.sql`). Apply once via Supabase SQL Editor.
- **`data/`** — Mock policy DB + per-call session JSON dumps (gitignored).
- **`.claude/`** — Project context for future Claude Code sessions. Includes:
  - `CLAUDE.md` — always-loaded orientation
  - `gotchas.md` — model-tuning failure modes (read before changing model code)
  - `docs/` — architecture decision, FNOL schema, MoshiRAG/ASPIRin analysis, LiveKit SIP setup

## Why this structure

- **`model_service/` is isolated** because GPU deployment has different infra. Talks to `backend/` over a JSON-only WebSocket. Audio NEVER traverses that WebSocket — it stays on the WebRTC plane (Twilio → LiveKit → Modal GPU direct).
- **`backend/` is CPU-only** so it can run on Modal CPU (`@modal.asgi_app`), Render, ngrok-from-laptop, or anywhere else.
- **`frontend/` reads Supabase directly**, not the backend. Decouples UI from backend container lifecycle. Realtime postgres_changes channel handles live updates.
- **PersonaPlex is NOT vendored** — `pip install` from NVIDIA's repo at image build time.
- **No `shared/` schema package** — backend owns the FNOL Pydantic schema; frontend reads jsonb from Supabase.

## First-time setup

```bash
# Entire — automatic Claude Code session capture linked to git commits
brew tap entireio/tap && brew install --cask entire
entire enable    # interactive: select "Claude Code" when prompted

# Environment variables — copy and fill in
cp .env.example .env
# .env is gitignored. See .env.example for the full list and what each
# is for. Highlights:
#   ELEVENLABS_API_KEY            Scribe v2 Realtime ASR
#   HF_TOKEN                      Download PersonaPlex weights
#   ANTHROPIC_API_KEY             Slot extractor (Haiku)
#   LIVEKIT_API_KEY/_SECRET/_URL  LiveKit Cloud
#   LIVEKIT_SIP_URI               Inbound SIP host (from LK dashboard)
#   TWILIO_ACCOUNT_SID/_AUTH_TOKEN/_PHONE_NUMBER
#   MODAL_TOKEN_ID/_SECRET        Modal CLI auth
#   BACKEND_PUBLIC_WS_URL         Backend's public URL (set after first deploy)
#   SUPABASE_URL                  Supabase project URL
#   SUPABASE_ANON_KEY             Browser-safe key for the UI
#   SUPABASE_SERVICE_ROLE_KEY     Backend-only, in Modal secret
```

### Supabase schema (one-time, before first call)

1. Create a Supabase project at https://supabase.com/dashboard
2. SQL Editor → paste + run the contents of `db/supabase_schema.sql`
3. Settings → API → copy URL + anon + service-role keys into `.env`

### LiveKit SIP setup (one-time, before first call)

See `.claude/docs/livekit-sip-setup.md` for the `lk sip inbound-trunk create`
+ `dispatch-rule create` commands. After it's done, paste the project's
SIP URI into `LIVEKIT_SIP_URI` in `.env`.

## Deploy (production-ish)

Three things to deploy (in order):

```bash
# 1. GPU model service (always-warm A100 80GB, ~$50/day)
CLIO_DEMO_MODE=1 modal deploy model_service/deploy/modal_app.py

# 2. CPU backend (FastAPI control plane + Twilio webhook + Supabase writer)
modal deploy model_service/deploy/modal_backend.py
# → prints a https://<...>.modal.run URL
# → put that URL (as wss://...) into clio-backend-cfg secret + redeploy
# → point Twilio Console "A Call Comes In" at <URL>/twilio/voice

# 3. Frontend monitoring UI
cd frontend && npx vercel --prod
# → set NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY in Vercel
```

Modal secrets you need (set once via `modal secret create`):
- `hf-token`            (HF_TOKEN)
- `clio-livekit`        (LIVEKIT_API_KEY/_SECRET/_URL/_SIP_URI)
- `clio-anthropic`      (ANTHROPIC_API_KEY)
- `clio-elevenlabs`     (ELEVENLABS_API_KEY)
- `clio-twilio`         (TWILIO_ACCOUNT_SID/_AUTH_TOKEN/_PHONE_NUMBER + TWILIO_SKIP_SIGNATURE_VALIDATION=1 for dev)
- `clio-backend-cfg`    (BACKEND_PUBLIC_WS_URL + MODAL_APP_NAME + MODAL_CLS_NAME + SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY)

## Running locally (dev)

```bash
# Backend FastAPI control plane (no GPU needed):
uv run uvicorn backend.app.control.server:app --reload --port 8000

# Frontend monitoring UI:
cd frontend && npm run dev      # → http://localhost:3000

# Tests (backend only — model_service has its own pyproject):
uv run pytest backend/tests/

# Quick smoke tests for individual reasoner modules:
uv run python -m backend.app.reasoner.persona B-AL-1234
uv run python -m backend.app.reasoner.gate
uv run python -m backend.app.reasoner.drip
```

For full end-to-end (real inbound call), the GPU container has to run on
Modal — local M-series Macs can't run PersonaPlex 7B inference.

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11+ (`uv`, FastAPI, livekit-agents, Pydantic v2) |
| Model service | Python (PyTorch, moshi/personaplex), deployed to Modal |
| Frontend | Next.js 15 + Tailwind + Supabase Realtime, deployed to Vercel |
| Telephony | Twilio DID + SIP → LiveKit Cloud |
| ASR (always-on, feeds LLM context) | ElevenLabs Scribe v2 Realtime (~150ms) |
| LLM (Reasoner slot extractor) | Anthropic Claude Haiku 4.5 |
| Persistent state / live UI feed | Supabase (Postgres + Realtime) |

## Model architecture

Two models, one container, one passive supervisor.

```
                ┌──────────────── Modal A100 container ────────────────┐
caller audio    │                                                       │
─── 24kHz ────▶ │  Mimi codec (encode) ─── codes ───▶ PersonaPlex 7B   │
                │                                       │   ▲           │
                │                              text+audio   │           │
                │                                       │   │ forced    │
                │                                       │   │ text_token│
                │  other Mimi (decode) ◄── codes ──────┘   │ injection │
                │           │                              │           │
                └───────────┼──────────────────────────────┼───────────┘
   agent audio              │                              │
─── 24kHz ◄─────────────────┘                              │
                                                           │
                              control-plane WebSocket (JSON, no audio)
                                                           │
                ┌──── Modal CPU container ─────────────────┼───────────┐
                │                                          ▼           │
                │   FastAPI orchestrator                                │
                │     ├─ slot extractor (Haiku, on caller turn)         │
                │     ├─ intervention gate (4 trigger rules)            │
                │     └─ FNOL state machine                             │
                │                                                       │
                │   Twilio webhook + LiveKit room/token mint            │
                │   Supabase writer (transcripts, events, fnol)         │
                └───────────────────────────────────────────────────────┘
```

### Three components on the GPU

- **Mimi codec** — neural audio codec, 24kHz ↔ 8 streaming token codebooks.
  Two instances: one for caller audio (encode), one for Sarah's audio
  (decode). Sharing one Mimi for both leaks GPU memory because the
  internal streaming-conv `prev_x` buffer grows unbounded.

- **PersonaPlex 7B** — NVIDIA's fork of Kyutai's Moshi. Full-duplex
  (consumes caller audio + emits agent audio simultaneously). Two heads:
  text and audio. We can force the text head to specific tokens via
  `lm_gen.step(text_token=...)`. The audio head responds to whatever
  text is being emitted.

- **Energy-VAD** — RMS-threshold voice activity detector with hysteresis.
  Fires `caller turn boundary` event when caller is silent for 800ms
  after speaking. Used both to (a) trigger backend's slot extractor and
  (b) gate Sarah's audio output so she stops talking when caller speaks.

### Sarah's "behavior layer" lives entirely in three places

1. **Persona prompt** (~3500 chars). Loaded into PersonaPlex at container
   start via `step_system_prompts`. Defines Sarah's identity, tone,
   read-back protocol, turn-taking rules, "don't guess identifiers" rule,
   small-talk handling.

2. **Drip-feed forced tokens.** When the backend Reasoner decides Sarah
   needs to say something specific (forced read-back, compliance prompt,
   wrap-up question), it tokenizes the text and pushes it into a per-call
   queue. The per-frame inference loop pulls one token per frame and
   forces it into `lm_gen.step(text_token=forced_id)`.

3. **Server-side audio gating.** When VAD detects the caller is speaking
   AND no drip directive is in flight, we zero out Sarah's PCM frame
   before publishing to LiveKit. This is what physically prevents her
   from talking over the caller — forcing the text PAD token alone
   doesn't silence the audio head.

### Reasoner is a passive observer with four intervention triggers

Default behavior: Sarah free-samples, Reasoner watches. Four gate
triggers can override that, in priority order:

| Trigger | When | Action |
|---|---|---|
| **Audio rescue** | Caller signals they can't hear ("Hello? Hello?", "what did you say", "could you repeat") | Force Sarah to acknowledge: "Sorry, I'm having trouble hearing you, could you say that again?" Cooldown 20s, max 3 per call |
| **Pending read-back** | Entity slot filled but unconfirmed for >1 caller turn | Force Sarah to read it back, then hold silence for confirmation. Re-fires every 3 caller turns up to 3 attempts |
| **Compliance deadline** | Critical slot (`incident_type` / `incident_datetime` / `any_injuries`) still empty after its per-slot deadline (120-240s) | Force Sarah to ask the missing question with a templated phrasing |
| **Wrap-up gate** | Caller signals end-of-call (or 25s of silence) and critical slots are still empty | Iterate through the missing slots (max 2 attempts) — drops "Before I let you go" prefix to avoid sounding like Sarah is closing |

All gate texts are **hand-templated** (in `gate.py` and `drip.py`), not
LLM-generated. Predictable latency, no inflight calls during voice loop.

### Anti-hallucination: structural anchor filter on the data plane

Sarah is generative. She occasionally free-samples a confident-sounding
identifier ("On June 15.", "POL-2024-001") that the caller never said.
Rather than try to make the persona prompt strict enough to prevent this
(brittle), Clio drops the offending values **after extraction, before
they reach the FNOL row**:

`backend/app/reasoner/extractor.py:filter_caller_anchored()` runs
between Haiku's slot updates and `apply_updates()`. For identifier-shaped
slots (policy_number, license_plate, vin, claim_number, police_case_number,
reporter_phone, reporter_name, incident_datetime), it requires the value's
`source_quote` (or the value itself, normalized) to literally appear in
a recent caller turn. If not, the update is dropped with a logged
warning. Caller may still hear Sarah say a hallucinated value out loud
(audio cosmetic, not data-corrupting), but the persisted FNOL only
contains values the caller actually spoke.

### Optional: Tavily fact-checker (built, deactivated for latency)

`backend/app/reasoner/tavily.py` is a self-contained async wrapper around
Tavily's web search API, with three primitive checks:
- `weather_at(location, when)` — corroborates the caller's weather claim
- `verify_location(address)` — confirms the address exists / is plausible
- `news_check(location, when)` — looks for traffic/incident news in a ±1 day window

Each returns a structured `FactCheckResult` with an `inconsistency_signal`
flag the gate can use to populate `fraud_signals.inconsistencies`. The
intent: when the caller says "icy roads" but Tavily reports clear and
dry, surface that as a fraud chip in the operator UI.

**Why it's currently unwired:** even running async off the hot loop,
each Tavily lookup adds 1-2s of round-trip and 80-200ms of result-processing
overhead. The drip-feed pattern would inject "(Internal: weather was X)"
into Sarah's text stream — fine in isolation, but each one consumes
agent_text_buf flush capacity and competes with gate-driven directives
for the drip queue. We chose to keep the voice loop strictly local
(Mimi + PersonaPlex + persona-cached state) and defer external lookups
to a follow-up version.

The full wiring checklist is in the module's top-level docstring (six
numbered steps: dep, env, secret, mount, gate trigger, orchestrator
routing). Smoke test from the CLI with
`TAVILY_API_KEY=tvly-... uv run python -m backend.app.reasoner.tavily`.

## How we got latency under 800ms

Inca's "feels human" threshold is around 800ms round-trip (caller speaks
→ caller hears Sarah's response). We measured ~450-650ms in practice.
Here's what each leg costs and what we optimized.

### One-way latency budget

```
Caller phone ─→ Twilio              100-150 ms   (fixed, telco)
Twilio ─→ LiveKit Cloud SIP          10-50 ms   (fixed, EU edge)
LiveKit Cloud ─→ Modal A100 WebRTC   30-80 ms   (region-pinned)
Mimi.encode (1 frame)                  ~5 ms
PersonaPlex inference (1 frame)        ~80 ms   ← the dominant cost
Mimi.decode (1 frame)                  ~5 ms
Modal ─→ LiveKit ─→ Twilio ─→ caller 50-100 ms

ROUND-TRIP TOTAL                  ~450-650 ms
```

### What we did to hit that

1. **Co-located PersonaPlex + LiveKit Agent in one Modal A100 container.**
   The original sketch had a separate "model service" the LiveKit Agent
   talked to over its own WebSocket. We collapsed it to in-process Python
   calls. Removed 60-200ms of network round-trip per audio frame —
   wholesale eliminating PersonaPlex's 80ms inference advantage.

2. **Audio NEVER traverses the backend ↔ Modal control WebSocket.**
   Caller audio goes Twilio → LiveKit → Modal direct via WebRTC. Backend
   only sends/receives JSON directives + transcripts. If audio went
   through backend, we'd add 30-100ms each way.

3. **Persona snapshot/restore.** Initial `step_system_prompts` to seed
   Sarah's persona into the KV cache takes ~30s. Doing that per-call
   makes the first second of conversation dead air. Instead we run it
   ONCE in `@modal.enter`, save the post-prime streaming state to
   safetensors, and restore it per-call in ~1-2s. See "gotchas.md" for
   why `copy.deepcopy` doesn't work and how to get the round-trip right.

4. **`min_containers=1` warm container.** With `CLIO_DEMO_MODE=1`, one
   GPU container stays warm. First call avoids the 60-75s cold start
   that would otherwise OOM the demo.

5. **Slot extractor (Anthropic Haiku) runs OFF the conversational hot
   loop.** It triggers on caller turn boundary — i.e., AFTER the caller
   stops talking. Its 1-2s latency is hidden in the natural pause
   between turns, not added to Sarah's response time.

6. **Gate decisions are rule-based, not LLM-based.** Each trigger
   (read-back / compliance / wrap-up) has hand-templated phrasing in
   `_compliance_phrasing` / `_wrapup_phrasing` / `render_readback`. Zero
   inflight calls during the voice loop = zero added latency from the
   intervention path.

7. **Backend on Modal CPU `@asgi_app`** — same Modal infra as the GPU
   container, so the control-plane WebSocket between them is
   intra-datacenter. Few-ms RTT for the JSON directives.

8. **`TORCHDYNAMO_DISABLE=1`** — disables CUDA graph capture inside
   moshi. Slightly slower per-frame inference (~5-10ms), but it stops a
   memory leak that OOMs ~3 minutes into a call. Net win for demo
   reliability.

### Latency breakdown by what the caller perceives

| Phase | Time | What's happening |
|---|---|---|
| Cold-start to "Hello, this is Sarah" | <2s with warm container, ~75s without | Container join + persona snapshot restore + caller's TwiML preamble window |
| Caller speaks → Sarah starts replying | 450-650ms | One-way through model + return path |
| Caller pause → Sarah picks up | <1s | EPAD wake-up token after VAD detects 800ms of silence |
| Caller mentions a slot → backend captures it | 1-2s after caller turn ends | Haiku extractor runs in background; Sarah keeps talking unaffected |
| Gate decides to intervene → Sarah complies | <300ms after directive sent | Backend WS → Modal → drip queue → next frame's `text_token` |

For deeper detail on any of these, see `.claude/docs/architecture.md`
and `.claude/gotchas.md`.
