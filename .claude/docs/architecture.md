# Clio System Architecture

**Status:** Decided
**Last updated:** 2026-04-25
**Companion docs:** [`architecture-decision.md`](architecture-decision.md), [`moshirag-analysis.md`](moshirag-analysis.md), [`aspirin-analysis.md`](aspirin-analysis.md)

This doc describes the runtime system shape: telephony, services, data flow, latency budget, and deployment topology. Read this if you want to understand how a juror's phone call becomes audio coming back through Sarah.

## Design intent

Inbound phone calls from jurors → real PSTN → real-time conversation with Sarah (PersonaPlex 7B) → structured FNOL output at end of call. The agent must pass the >50% human-vote bar.

The system is intentionally *boring* on infrastructure (LiveKit + Twilio + Modal) and *novel* only on the part that matters: Reasoner-driven `text_token` injection on PersonaPlex.

## Locked-in stack decisions

| Layer | Choice | Why |
|---|---|---|
| PSTN ingress | **Twilio** (DID + SIP trunk) | Reliable, jurors call a real number, hackathon can pre-provision |
| Real-time media | **LiveKit Cloud** | First-class SIP trunk integration, WebRTC infra, Agents framework with Python SDK |
| Agent runtime | **LiveKit Agent (Python) co-located with PersonaPlex on Modal A100** | Audio never crosses backend; one network hop from LiveKit Cloud directly into the Modal container. Reasoner stays separate over control-plane WS. |
| Voice model | **PersonaPlex 7B** off-the-shelf | Public injection API on `LMGen.step(text_token=...)`, no fine-tuning needed |
| Model hosting | **Modal** (GPU) | Existing team experience, fast cold start, auto-scaling |
| **Verification ASR** | **AssemblyAI Universal-3 Pro** (backchannel only) | Has the only published entity-recall benchmark (16.7% missed entity rate vs Deepgram Nova-3 25.2%). NOT primary transcript — see "Verification ASR backchannel" below |
| Optional: VAD/turn detection | **Pipecat Smart Turn v2** | Semantic endpointing better than raw VAD; only add if LiveKit's built-in turn detection isn't good enough |
| Frontend | **Next.js** | Monitoring UI: live transcript, FNOL slot state, call history |

**LiveKit is mandatory. Pipecat is optional.** See "Pipecat: add or skip?" below.

## Service boundaries (what runs where)

**Critical design choice:** the LiveKit Agent and PersonaPlex 7B are **co-located on the same Modal A100 container**. Audio never traverses the model_service ↔ backend boundary — it goes LiveKit Cloud → Modal directly via WebRTC, then in-process Python calls into PersonaPlex. The backend Reasoner runs separately (CPU anywhere) and communicates only via a control-plane WebSocket carrying JSON directives (latency-tolerant).

This avoids a 100-200ms WebSocket round-trip per audio frame that would have wiped out PersonaPlex's 80ms inference advantage.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Caller phone (juror)                                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ PSTN
┌────────────────────────────────▼────────────────────────────────────────┐
│ Twilio                                                                  │
│  • DID number (e.g. +49 30 XXXX XXXX)                                   │
│  • SIP trunk → LiveKit                                                  │
│  • Optional: TwiML "<Say>thank you for calling Allianz, please hold"    │
│    while Modal warms up (only matters if keep_warm=0)                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ SIP
┌────────────────────────────────▼────────────────────────────────────────┐
│ LiveKit Cloud                                                           │
│  • Creates room per call                                                │
│  • WebRTC media routing (Opus 24kHz)                                    │
│  • Dispatches the call to our Agent worker (registered process)         │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ WebRTC (in-bound: caller audio)
                                 │ WebRTC (out-bound: agent audio)
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│ model_service/  (Modal A100 container — GPU + LiveKit Agent in ONE      │
│                  Python process; both audio and inference here)         │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  LiveKit Agent worker (livekit-agents SDK)                        │  │
│  │   • Registers with LiveKit Cloud at boot, accepts dispatches      │  │
│  │   • Subscribes to caller audio track per dispatched room          │  │
│  │   • Publishes Sarah's audio track back to the room                │  │
│  │   • Runs ElevenLabs Scribe v2 stream in parallel for entity       │  │
│  │     verification (always-on ASR; feeds backend extractor context) │  │
│  └─────────────────────┬─────────────────────────────────────────────┘  │
│                        │ in-process Python call (μs latency)            │
│  ┌─────────────────────▼─────────────────────────────────────────────┐  │
│  │  PersonaPlex 7B (loaded once in @modal.enter)                     │  │
│  │   • LMGen.step(input_tokens, text_token=forced) per 80ms frame    │  │
│  │   • Persona priming at session start: text_prompt + voice_prompt  │  │
│  │   • forced text_token is read from a per-call directive cache     │  │
│  │     populated asynchronously by the control-plane WS              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ Control plane WebSocket (JSON only)
                         │   • backend pushes ReasonerDirectives
                         │   • Modal pushes transcript turns + status
                         │   • latency-tolerant: 50-200ms RTT is fine
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ backend/  (CPU-only — runs on Render / Vercel / Modal CPU / localhost)  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  app/control/server.py — control-plane WS server (or client)      │  │
│  │   • Receives transcripts (PersonaPlex monologue + Scribe ASR)     │  │
│  │   • Pushes directives back to Modal                               │  │
│  └─────────────────────┬─────────────────────────────────────────────┘  │
│                        │                                                │
│  ┌─────────────────────▼─────────────────────────────────────────────┐  │
│  │  app/reasoner/                                                    │  │
│  │   • state.py — FNOL slot tracker (Pydantic)                       │  │
│  │   • extractor.py — async slot extractor (Anthropic Haiku)         │  │
│  │   • gate.py — intervention triggers (wrap-up, drift, compliance)  │  │
│  │   • drip.py — text token sequencer (12.5 Hz cadence)              │  │
│  │   • db.py / mock_policies.json — policy lookup                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  app/telephony/twilio_handler.py                                  │  │
│  │   • Receives Twilio inbound webhooks                              │  │
│  │   • Generates LiveKit room + agent token                          │  │
│  │   • Triggers Modal's PersonaPlexAgent.process_call.spawn(...)     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │ frontend/  (Next.js — monitoring)   │
         │                                     │
         │  • Live transcript stream           │
         │  • FNOL slot state visualization    │
         │  • Call history + recordings        │
         │  • Agent status (idle / on call)    │
         │                                     │
         │  Reads via backend HTTP/SSE         │
         └─────────────────────────────────────┘
```

## Control plane WebSocket (backend ⇄ Modal)

**Only JSON messages travel this channel — no audio.** Audio stays inside Modal (caller audio in via LiveKit WebRTC, processed by PersonaPlex, agent audio out via LiveKit WebRTC). The control plane carries directives both ways.

### Backend → Modal (Reasoner directives)

```json
{"type": "speak", "seq": 12, "text": "Before I let you go...", "after_release": "resume"}
{"type": "silent", "seq": 13, "duration_frames": 25}
{"type": "release", "seq": 14}
{"type": "rescue_clip", "seq": 15, "clip_id": "breaking_up_1"}
{"type": "load_policy_context", "seq": 16, "policy_brief": "Caller is Anna Schmidt..."}
```

See `backend/app/reasoner/drip.py` for the full schema. Each directive has a monotonic `seq` so Modal can order/dedupe.

### Modal → Backend (transcripts + status)

```json
{"type": "transcript", "seq": 100, "role": "agent", "token_id": 234, "text": " yes",
                       "source": "personaplex"}
{"type": "transcript", "seq": 101, "role": "caller", "text": "POL-2024-001",
                       "source": "scribe"}
{"type": "session_ready", "call_id": "abc-123"}
{"type": "session_closed", "call_id": "abc-123", "reason": "caller hung up"}
```

Backend's Reasoner aggregates the transcript stream (from both sources) and runs slot extraction asynchronously per user-turn boundary, writing back directives when the gate fires.

### Why this is fast enough

Audio is the latency-critical path; it never goes over this WS. The control plane handles thousands of small JSON messages per call, each ~100 bytes, and a 100-200ms RTT to push a directive doesn't degrade the user experience because:
- Sarah's speech generation is fed by the latest cached directive at frame time (read locally on Modal)
- Reasoner-driven nudges are inherently slow (the gate fires once every 5-30s)
- Transcript pushes are fire-and-forget — backend can be momentarily slow without blocking audio

If we ever need to colocate the Reasoner with PersonaPlex (e.g. to eliminate the WS entirely), `app/reasoner/` is pure Python and can be imported into the Modal container directly. Documented as a fallback, not currently needed.

## Verification ASR backchannel

**Mental model first** (this is easy to get wrong):

> PersonaPlex's text monologue stream is the **primary transcript**. Slot extraction runs on it. The independent ASR is a **verification backchannel** — it never replaces PersonaPlex; it cross-checks high-stakes entities (policy numbers, license plates, dates, amounts) when slot extractor confidence is low.

### Why verification, not replacement

PersonaPlex outputs both audio AND a synchronized text monologue stream. For most conversational content ("I had a car accident on the A100, around 8pm"), that transcript is good enough — slot extractor pulls "incident_type", "location", "incident_datetime" without issue. The transcript was designed for the model's own conditioning, not as production-grade ASR, but it's serviceable for the bulk of the conversation.

The problem is narrow: PersonaPlex is Moshi-derived, and Moshi-class models have only middling digit/entity accuracy. When a caller says "P-O-L dash 2-0-2-4 dash 0-0-1", PersonaPlex's transcript may corrupt one or two characters. For free-form prose that's fine; for an identifier we'll write into the FNOL JSON, it's a real failure.

Solution: **run a second ASR in parallel, only consult its output when the slot extractor flags low confidence on entity-critical slots.**

### Architecture

```
User audio (24kHz PCM)
       │
       ├──► PersonaPlex ───► audio out (Sarah speaks)
       │       │
       │       └──► text monologue stream ──┐
       │                                    │
       │                                    ▼
       │                            ┌─────────────────┐
       │                            │ Slot Extractor  │  PRIMARY path
       │                            │ (Anthropic      │
       │                            │  Haiku, async)  │
       │                            └────────┬────────┘
       │                                     │ extracts e.g. policy_number = "POL-2024-001"
       │                                     │ confidence = 0.78
       │                                     ▼
       │                            ┌─────────────────────────────┐
       │                            │ Reasoner state              │
       │                            │  IF entity-critical AND     │
       │                            │     confidence < 0.9:       │
       │                            │  THEN consult ↓ backchannel │
       │                            └────────┬────────────────────┘
       │                                     │
       └──► AssemblyAI Universal-3 ──────────┴──► (asked only when needed)
              (streaming, parallel)               same audio segment's transcript
                                                          │
                                                          ▼
                                              ┌────────────────────┐
                                              │ Verification check │
                                              │  Compare PP and    │
                                              │  AAI transcripts   │
                                              │  for the slot      │
                                              └─────────┬──────────┘
                                                        │
                                          ┌─────────────┴────────────┐
                                          │                          │
                                       agree                       differ
                                          │                          │
                                          ▼                          ▼
                                   confidence ↑                trigger Sarah
                                   continue                    read-back via
                                                               drip-feed
```

### When verification fires

Three triggers, narrow by design:

1. **Slot extractor flagged low confidence** on a CRITICAL or EXPECTED entity-bearing slot (`policy_number`, `license_plate`, `phone`, `incident_datetime`, `police_case_number`, monetary amounts). Threshold: confidence < 0.9.
2. **User correction phrase detected** ("actually", "I meant", "no, it's…"). Re-extract on the fresh segment, compare with backchannel.
3. **Wrap-up confirmation** — at end-of-call, run verification across all critical entity slots even if confidence was OK in real-time. Catches anything PersonaPlex and the extractor both missed.

**95% of the call**, the verification ASR transcript is generated but **never consumed**. It's there for when we need it.

### Why this lowers our ASR requirements

When ASR is the verification backchannel rather than the primary transcript:

| Requirement | Primary ASR (we ruled this out) | Verification backchannel (what we picked) |
|---|---|---|
| Streaming latency | <300ms (real-time conversational gating) | 500ms-1s OK (only consulted at confidence checks / wrap-up) |
| General WER | Critical | Less important — we're cross-checking, not transcribing |
| **Entity recall** | Critical | **Critical** ← the only metric that really matters |
| Cost | High (always-on) | Low (small fraction of audio actually examined) |

### Why AssemblyAI Universal-3 Pro for this role

The deciding factor: **AssemblyAI is the only ASR with a published entity-specific benchmark.**

| Model | Missed entity rate (names/emails/phones/CC) | Notes |
|---|---|---|
| **AssemblyAI Universal-3 Pro** | **16.7%** | Best published number; verified against multi-vendor test set |
| Deepgram Nova-3 | 25.2% | Same benchmark, AssemblyAI's testing |
| Microsoft Azure | 25.1% | Same benchmark |
| OpenAI GPT-4o Transcribe | 23.3% | Same benchmark |
| ElevenLabs Scribe v2 Realtime | (not benchmarked on entities) | Best general WER (2.3%) and lowest latency (150ms), but no public entity data |

We considered Scribe v2 (faster, better general WER) but for the verification role its latency advantage is wasted (we only consult it at confidence-check moments, not on every frame). AssemblyAI's published entity advantage matches the metric we actually care about.

If our own eval harness shows Scribe v2 is comparable on entity recall in our specific test calls, we can swap. AssemblyAI is the safer default given the published evidence.

### Cost ballpark

- AssemblyAI Universal-3 Pro Streaming: $0.21/h
- Hackathon demo: ~10 hours of test calls total
- Total cost: ~$2

If Inca's "we cover premium credits" includes ASR, this is free. Either way, it's not a budget concern.

### LiveKit Agents wiring

```python
from livekit.plugins import assemblyai

agent = VoiceAssistant(
    stt=assemblyai.STT(
        api_key=ASSEMBLYAI_API_KEY,
        sample_rate=24000,  # match PersonaPlex
        language="en",
    ),
    # PersonaPlex is NOT plugged into LiveKit's TTS slot —
    # it has its own bidirectional audio bridge to model_service.
)
```

The LiveKit Agent receives PersonaPlex's text monologue stream over the control plane WS and feeds it to the slot extractor. The AssemblyAI transcript is parallel; the Reasoner pulls it only when verification triggers fire.

## Latency budget (the analysis you asked for)

### Baseline: PersonaPlex on its own

Per the Moshi paper, theoretical 160ms, practical ~200ms model latency. With LiveKit Agent + PersonaPlex co-located on Modal, audio takes one network hop into Modal and one back out:

| Hop | Time |
|---|---|
| PersonaPlex inference (per frame) | ~80ms |
| Mimi encode + decode | ~10ms |
| **Sub-total: model + codec (in-process Python)** | **~90ms** |
| Twilio ↔ LiveKit Cloud ↔ Modal (single WebRTC hop, EU regions) | ~30–80ms one-way |
| PSTN: handset ↔ Twilio | ~100–150ms one-way |
| **Total mouth-to-ear (one direction)** | **~220–320ms** |

Well below the 500ms "feels human" threshold. The co-location decision saved us ~80–150ms vs the original two-hop design (where backend would have been the LiveKit Agent and forwarded audio to model_service over a separate WS).

Cold-start measured on Modal A100 (verified 2026-04-25):
- Container boot: ~20s
- PersonaPlex setup (Mimi + LM 7B + LMGen + warmup): **59s** measured
- **Total cold start: ~80s** — masked by `keep_warm=1` during demo, or by Twilio `<Say>` preamble for dev

### Adding our injection: how much latency does it add?

**Per-frame additions inside the per-call loop:**

1. Read latest Reasoner directive from local in-process cache: **<0.1ms**
2. Build forced text_token tensor (or use cached one): **<0.1ms**
3. Branch in `LMGen.prepare_step_input()` to write the cache: **<0.1ms** (single tensor index op)

**Total per-frame overhead: <1ms**, well under the 80ms frame budget. **The injection does not meaningfully add latency.**

### What CAN add latency (avoid these)

| Anti-pattern | Cost | Fix |
|---|---|---|
| Audio path crossing the model_service ↔ backend boundary | +60–200ms per round-trip | Co-locate LiveKit Agent + PersonaPlex on Modal — already done |
| Synchronously calling slot extractor LLM in the per-frame loop | +200–1000ms per frame, system collapses | Extractor runs async on user-turn boundary; the per-frame loop reads cached state |
| Per-frame RPC from Modal to backend ("what should I inject?") | +20–100ms per frame depending on geo | Backend *pushes* directives; Modal reads from local cache only |
| Deploying backend in a different region than Modal | +80–150ms control-plane RTT | Backend region-aligned with Modal (both EU). Less critical than audio path but still good hygiene |
| Burst-inject text tokens (>20 chars at once) | Audio head degenerates → token repetition | Drip-feed at 12.5 Hz cadence (VAOS journal lesson) |
| Cold-starting on the first juror call | ~80s of dead air | `keep_warm=1` during demo period (`CLIO_DEMO_MODE=1 modal deploy ...`); Twilio `<Say>` preamble as belt-and-braces |

### Honest summary

**With co-location and async control, our software adds ~0ms to the audio path.** All real latency lives in: PersonaPlex itself (~90ms, fixed), the single LiveKit Cloud → Modal WebRTC hop (~30–80ms, geo-dependent), and PSTN handset latency (~100–150ms each way, fixed). Our code controls none of these meaningfully.

The latency risk is **architectural** (don't double-hop audio through backend, don't run extractor synchronously) not **algorithmic** (the math of our injection is cheap).

## Pipecat: add or skip?

Pipecat is **not required** for this architecture. LiveKit Agents already provides:
- VAD (Silero)
- Audio capture/playback
- Turn detection (basic)
- Function calling helpers

Pipecat would add:
- **Smart Turn v2** semantic endpointing (better at "is the user actually done?")
- Pipecat Flows (state machine helpers — but our Reasoner already does this)
- Plugin ecosystem for STT/LLM/TTS (we don't use these — PersonaPlex is monolithic)

**Decision rule:**
- **Default: LiveKit only.** Build the Agent directly with `livekit-agents` Python SDK.
- **Add Pipecat only if** turn detection accuracy is hurting human-pass rate during testing. Smart Turn v2 wraps cleanly around LiveKit Agents (Daily/Pipecat published the integration pattern).

This deferral keeps the dependency graph minimal until we have evidence Pipecat helps.

## Deployment topology

### Hackathon-day (single-region, simple)

| Service | Where | Why |
|---|---|---|
| Twilio DID | EU region | minimize PSTN hop |
| LiveKit | LiveKit Cloud (EU) | matches Twilio region |
| **model_service (LiveKit Agent + PersonaPlex co-located)** | **Modal A100, EU region** | Audio path stays inside this single container — no extra hops |
| backend (Reasoner + Twilio webhook handler + control WS) | Render / Vercel / Modal CPU | CPU-only, communicates with model_service via JSON over WS |
| frontend | Vercel | not latency-critical |

Audio path is exactly one hop: LiveKit Cloud (EU) → Modal A100 (EU). 5–20ms intra-region.

### Dev / local

```bash
# Terminal 1: model_service (mock-talker, no GPU needed) — protocol tests + drip-feed dev
cd model_service && uv run python -m server.main

# Terminal 2: backend control plane (Reasoner WS server + Twilio webhook handler)
cd backend && uv run python -m app.control.server

# Terminal 3: frontend (monitoring UI)
cd frontend && pnpm dev
```

For real-PersonaPlex end-to-end testing, deploy to Modal:

```bash
modal deploy model_service/deploy/modal_app.py     # dev: keep_warm=0
CLIO_DEMO_MODE=1 modal deploy model_service/deploy/modal_app.py   # demo: always-warm
modal app stop personaplex-clio                    # stop billing when done
```

For end-to-end test without a real phone: LiveKit Agents' built-in dev console connects via web mic.

## Persona (Sarah) configuration

System prompt template (lives in `backend/app/reasoner/persona.py`):

```python
SARAH_PERSONA = """
You work for Allianz Claims Berlin and your name is Sarah. You are an experienced claims representative with 8 years on the desk.

When callers are upset or shaken (which is normal — they often just had an accident), acknowledge their feelings before asking factual questions. You are calm, empathetic, and professional.

Information: [policy DB lookups will be drip-fed here mid-call]
"""

VOICE_PROMPT = "NATF1.pt"  # try NATF2 as alternative
```

This is the only persona configuration. It's loaded once into PersonaPlex's KV cache via `lm_gen.step_system_prompts()` at session start.

## Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| LiveKit room creation fails | Twilio fallback to a hold message + emergency phone number |
| model_service GPU OOM | Modal auto-scaling kicks new instance; current call dropped with apology clip |
| Slot extractor LLM API rate limit | Local fallback regex extractor for critical slots (policy_number, name) |
| PersonaPlex audio degenerates (rare burst-injection bug) | Inject rescue clip + reset KV cache |
| Caller in heavy noise (highway scenario from Inca brief) | Rescue clip "you're breaking up" — actually a feature, not a bug |
| Backend ↔ model_service WebSocket dropped mid-call | Reconnect with session ID; LMGen state can resume from cache |

## Open questions to resolve before code-write

1. **LiveKit Agent or LiveKit + Pipecat?** Default LiveKit-only; revisit after first end-to-end test. ✅ Decided LiveKit-only.
2. **Custom voice for Sarah, or NATF1 off the shelf?** Test NATF1, NATF2, NATM2 quickly; pick best in <1 hour.
3. **Where does the FNOL state schema live as the source of truth?** Backend `app/reasoner/schema.py` (Pydantic). Frontend imports types via codegen or defines TS types manually.
4. **Twilio DID provisioning** — need an EU number with SIP trunk routing to LiveKit. ✅ Done.
5. **Modal cold-start time for PersonaPlex** — measured 59s for setup, ~80s total with container boot. Demo uses `min_containers=1` (CLIO_DEMO_MODE=1). Twilio `<Say>` preamble as belt-and-braces.
6. **GPU choice** — A100 40GB (~$26/day keep-warm) for VRAM headroom over A10G. ✅ Validated PersonaPlex 7B loads cleanly on A100.

## What this architecture deliberately does NOT do

- **No model training.** All capability comes from off-the-shelf PersonaPlex + persona prompt + Reasoner logic.
- **No active inject-on-every-turn.** Reasoner is a passive observer; intervenes only at wrap-up, drift, or compliance deadline.
- **No multi-persona.** One Sarah. Multi-persona is a v2 problem.
- **No custom audio model.** PersonaPlex's prosody is what we ship.
- **No fork of PersonaPlex.** Public `LMGen.step(text_token=...)` API is sufficient.

## References

- [`architecture-decision.md`](architecture-decision.md) — why we chose this path over training, MoshiRAG, ASPIRin
- [`moshirag-analysis.md`](moshirag-analysis.md) — concepts borrowed (lead/body/tail structure, three-LLM data gen)
- [`aspirin-analysis.md`](aspirin-analysis.md) — concepts borrowed (timing/content separation)
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — Python SDK for headless agent participants
- [LiveKit SIP integration](https://docs.livekit.io/sip/) — Twilio trunk → LiveKit room
- [Pipecat Smart Turn v2](https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/) — semantic endpointing if needed
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex) — the model + injection API
- [Modal docs](https://modal.com/docs/guide) — GPU deployment
