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
| Agent runtime | **LiveKit Agent (Python)** | Headless participant joins each room, owns the audio bridge + Reasoner |
| Voice model | **PersonaPlex 7B** off-the-shelf | Public injection API on `LMGen.step(text_token=...)`, no fine-tuning needed |
| Model hosting | **Modal** (GPU) | Existing team experience, fast cold start, auto-scaling |
| **Verification ASR** | **AssemblyAI Universal-3 Pro** (backchannel only) | Has the only published entity-recall benchmark (16.7% missed entity rate vs Deepgram Nova-3 25.2%). NOT primary transcript — see "Verification ASR backchannel" below |
| Optional: VAD/turn detection | **Pipecat Smart Turn v2** | Semantic endpointing better than raw VAD; only add if LiveKit's built-in turn detection isn't good enough |
| Frontend | **Next.js** | Monitoring UI: live transcript, FNOL slot state, call history |

**LiveKit is mandatory. Pipecat is optional.** See "Pipecat: add or skip?" below.

## Service boundaries (what runs where)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Caller phone (juror)                                                    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ PSTN
┌────────────────────────────────▼────────────────────────────────────────┐
│ Twilio                                                                  │
│  • DID number (e.g. +49 30 XXXX XXXX)                                   │
│  • SIP trunk → LiveKit                                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ SIP
┌────────────────────────────────▼────────────────────────────────────────┐
│ LiveKit Cloud                                                           │
│  • Creates room per call                                                │
│  • WebRTC media routing                                                 │
│  • Notifies our Agent dispatch service                                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ WebRTC (Opus, 24kHz)
┌────────────────────────────────▼────────────────────────────────────────┐
│ backend/  (LiveKit Agent process — CPU, runs on Modal CPU or Render)    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  app/telephony/livekit_agent.py                                   │  │
│  │   • Joins LiveKit room as headless participant                    │  │
│  │   • Receives caller audio frames (24kHz PCM)                      │  │
│  │   • Forwards audio → model_service WebSocket                      │  │
│  │   • Plays response audio frames back into the room                │  │
│  └─────────────────────┬─────────────────────────────────────────────┘  │
│                        │                                                │
│  ┌─────────────────────▼─────────────────────────────────────────────┐  │
│  │  app/reasoner/                                                    │  │
│  │   • state.py — FNOL slot tracker (Pydantic)                       │  │
│  │   • extractor.py — async slot extractor (gpt-4o-mini / Haiku)     │  │
│  │   • gate.py — intervention triggers (wrap-up, drift, compliance)  │  │
│  │   • drip.py — text token sequencer (12.5 Hz cadence)              │  │
│  └─────────────────────┬─────────────────────────────────────────────┘  │
│                        │                                                │
│   Pushes Reasoner state via WS (control plane, async, non-blocking)     │
│                        │                                                │
└────────────────────────┼────────────────────────────────────────────────┘
                         │
       ┌─────────────────┴─────────────────┐
       │                                   │
       │ Audio plane (WebSocket, PCM)      │ Control plane (WebSocket, JSON)
       │   - bidirectional 24kHz PCM       │   - Reasoner pushes:
       │   - tight latency budget          │     {action: "speak"|"silent",
       │                                   │      drip: "..." | null}
       │                                   │   - latency-tolerant
       ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ model_service/  (Python, GPU — Modal H100/A100)                         │
│                                                                         │
│  • PersonaPlex 7B loaded once, kept warm via streaming_forever()        │
│  • Persona priming at session start: text_prompt + voice_prompt         │
│  • Per-frame loop (every 80ms = 12.5 Hz):                               │
│      1. Read latest Reasoner directive from local cache                 │
│      2. Decide forced text_token: None | EPAD | PAD | drip_token_id     │
│      3. tokens = lm_gen.step(input_tokens=user_audio,                   │
│                              text_token=forced_tensor)                  │
│      4. Decode audio frame (~80ms PCM)                                  │
│      5. Push frame back over audio WS                                   │
│  • Pre-recorded rescue clips for "you're breaking up" fallback          │
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

## Two-channel WebSocket protocol (backend ⇄ model_service)

We use **two separate WebSocket connections** between backend and model_service:

### Audio plane

- Bidirectional 24kHz mono PCM
- Frame size: 80ms (1920 samples) — matches Moshi's frame rate
- Tight latency: every ms here is felt by the caller
- Backend writes user audio chunks; model_service writes agent response chunks

### Control plane

- JSON messages, lower frequency
- **Reasoner → model_service** (push only, frequent):
  ```json
  {"type": "directive", "action": "drip", "tokens": [123, 456, 789]}
  {"type": "directive", "action": "epad"}
  {"type": "directive", "action": "pad"}
  {"type": "directive", "action": "release"}
  {"type": "rescue_clip", "clip_id": "breaking_up_1"}
  ```
- **model_service → backend** (push, every Moshi frame):
  ```json
  {"type": "text_token", "token_id": 234, "decoded": " yes"}
  {"type": "audio_event", "kind": "started_speaking"}
  ```

Why two channels: audio plane has hard real-time constraints; control plane is async and allows the Reasoner to update state without blocking audio. Mixing them in one connection causes head-of-line blocking under load.

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

Per the Moshi paper, theoretical 160ms, practical ~200ms model latency. Add codec, round-trip:

| Hop | Time |
|---|---|
| PersonaPlex inference (per frame) | ~80ms |
| Mimi encode + decode | ~10ms |
| **Sub-total: model + codec** | **~90ms** |
| Add network round-trip (Twilio ↔ LiveKit ↔ backend ↔ model_service, all in EU) | ~80–150ms |
| Speaker → mic → PSTN → Twilio | ~100–150ms |
| **Total mouth-to-ear** | **~270–390ms** |

This is below the 500ms "feels human" threshold. Good baseline.

### Adding our injection: how much latency does it add?

**Per-frame additions inside the model_service loop:**

1. Read latest Reasoner directive from local cache (in-process dict): **<0.1ms**
2. Build forced text_token tensor (or use cached one): **<0.1ms**
3. Branch in `LMGen.prepare_step_input()` to write the cache: **<0.1ms** (single tensor index op)

**Total per-frame overhead: <1ms**, well under the 80ms frame budget. **The injection does not meaningfully add latency.**

### What CAN add latency (avoid these)

| Anti-pattern | Cost | Fix |
|---|---|---|
| Synchronously call slot extractor LLM in the per-frame loop | +200–1000ms per frame, system collapses | Run extractor async on user-turn boundary, push state to model_service over control WS |
| Make per-frame RPC from model_service to backend ("what should I inject?") | +20–100ms per frame depending on geo | Reasoner *pushes* directives; model_service reads from local cache |
| Deploy backend in US, model_service in EU (or vice versa) | +80–150ms per audio round-trip | Co-locate in same region (EU for Berlin hackathon) |
| Single WebSocket carrying both audio + control | Head-of-line blocking under contention | Two separate WebSockets (described above) |
| Burst-inject text tokens (>20 chars at once) | Audio head degenerates → token repetition | Drip-feed at 12.5 Hz cadence (VAOS journal lesson) |

### Honest summary

**If we design the WebSocket protocol right, the injection adds ~0ms.** All real latency lives in: PersonaPlex itself (~90ms, fixed), telephony round-trip (~80–150ms, geo-dependent), and PSTN handset latency (~100–150ms, fixed). Our code controls none of these meaningfully.

The latency risk is **architectural** (don't make per-frame RPC, don't run extractor synchronously) not **algorithmic** (the math of our injection is cheap).

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
| backend (LiveKit Agent) | Modal CPU (EU) or Render | same region as LiveKit |
| model_service | Modal GPU (EU, H100 or A100) | same region as backend |
| frontend | Vercel | not latency-critical |

All audio-path components in EU (Frankfurt or Dublin). Total intra-region latency: 5–20ms per hop.

### Dev / local

```bash
# Terminal 1: model service (needs GPU; or use Modal serve --watch)
cd model_service && uv run python -m server.main

# Terminal 2: backend (LiveKit Agent connects to LiveKit Cloud dev project)
cd backend && uv run python -m app.telephony.livekit_agent dev

# Terminal 3: frontend
cd frontend && pnpm dev
```

For end-to-end test without a real phone: LiveKit's web SDK — talk into the browser, it joins as a room participant.

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

1. **LiveKit Agent or LiveKit + Pipecat?** Default LiveKit-only; revisit after first end-to-end test.
2. **Custom voice for Sarah, or NATF1 off the shelf?** Test NATF1, NATF2, NATM2 quickly; pick best in <1 hour.
3. **Where does the FNOL state schema live as the source of truth?** Backend `app/reasoner/schema.py` (Pydantic). Frontend imports types via codegen or defines TS types manually.
4. **Twilio DID provisioning** — need an EU number with SIP trunk routing to LiveKit. Do this on day 1.
5. **Modal cold-start time for PersonaPlex** — must be <30s or we lose calls. Use `keep_warm=True` and a heartbeat.

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
