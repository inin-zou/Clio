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
| Frontend | Next.js 15 + Tailwind + shadcn/ui, deployed to Vercel |
| Telephony | Twilio DID + SIP → LiveKit Cloud |
| ASR (always-on, feeds LLM context) | ElevenLabs Scribe v2 Realtime (150ms latency) |
| LLM (Reasoner slot extractor) | Anthropic Claude Haiku |
