# Clio

Voice agent for inbound insurance claim calls. Built for the [Inca hackathon](https://www.get-inca.com/en) — the agent must convince jurors they're talking to a human while collecting complete FNOL documentation.

See [`.claude/docs/architecture-decision.md`](.claude/docs/architecture-decision.md) for the full design rationale.

## High-level architecture

```
Twilio SIP ──audio──► model_service (GPU)         backend (CPU)
                       │  PersonaPlex 7B    ◄─────►  Reasoner
                       │  injection loop      WS      FNOL state
                       │                              telephony glue
                       └──audio───► Twilio
                                                       │
                                                       ▼
                                                   frontend
                                                   (monitoring)
```

- **`model_service/`** — PersonaPlex 7B inference + Reasoner-driven `text_token` injection. GPU. Deployed to Modal.
- **`backend/`** — Reasoner (FNOL state, slot extractor, intervention gate) + Twilio webhooks. CPU.
- **`frontend/`** — Next.js monitoring UI: live transcript, FNOL state visualization, call history.
- **`data/`** — Synthetic FNOL dialogues for evaluation and prompt-tuning. No model training data.
- **`tools/`** — Utility scripts: dialogue generator, eval runner, voice cloning.
- **`.claude/docs/`** — Technical analysis docs (MoshiRAG, ASPIRin, architecture decision).

## Why this structure

- **`model_service/` is isolated** because GPU deployment has different infra than the rest. It runs as its own service, talks to `backend/` over WebSocket. Nothing imports across this boundary.
- **`backend/` is CPU-only** so it can run anywhere (Vercel, Render, Modal CPU, localhost).
- **PersonaPlex is NOT vendored** — installed via `pip` in `model_service/requirements.txt`. Repo stays lean.
- **No `shared/` package** — backend defines the FNOL schema; frontend fetches it at runtime via API.

## First-time setup

```bash
# Entire — automatic Claude Code session capture linked to git commits
brew tap entireio/tap && brew install --cask entire
entire enable    # interactive: select "Claude Code" when prompted

# Environment variables — copy and fill in
cp .env.example .env
# .env is gitignored. Required keys: ELEVENLABS_API_KEY, ANTHROPIC_API_KEY,
# LIVEKIT_API_KEY, LIVEKIT_API_SECRET, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
# MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN
```

## Running services (per service, native tooling)

```bash
# Model service (PersonaPlex inference, GPU required)
cd model_service && uv run python -m server.main

# Backend (LiveKit Agent + Reasoner, CPU)
cd backend && uv run python -m app.telephony.livekit_agent dev

# Frontend (Next.js monitoring UI)
cd frontend && pnpm dev
```

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

### Reasoner is a passive observer with three intervention triggers

Default behavior: Sarah free-samples, Reasoner watches. Three gate
triggers can override that:

| Trigger | When | Action |
|---|---|---|
| **Pending read-back** | Entity slot filled but unconfirmed for >1 caller turn | Force Sarah to read it back, then hold silence for confirmation |
| **Compliance deadline** | Critical slot still empty after 120-180s into call | Force Sarah to ask the missing question with a templated phrasing |
| **Wrap-up gate** | Caller signals end-of-call but critical slots missing | Force one last question before letting Sarah close |

All gate texts are **hand-templated** (in `gate.py` and `drip.py`), not
LLM-generated. Predictable latency, no inflight calls during voice loop.

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
