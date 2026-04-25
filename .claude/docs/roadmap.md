# Clio Roadmap

**Last updated:** 2026-04-25
**Companion:** [`architecture.md`](architecture.md), [`architecture-decision.md`](architecture-decision.md)

Three MVPs, in dependency order. Each has a sharp definition of done and the smallest scope that proves the next layer can be built on top.

---

## MVP 1 — PersonaPlex with Reasoner-driven injection (offline)

**Goal:** Prove the injection mechanism works end-to-end on a single machine, without telephony in the picture.

### Definition of done

A juror plays a recorded WAV of "I just had a car accident, my name is..." → our patched PersonaPlex responds in audio, with Sarah persona, and the **Reasoner forced Sarah to ask a specific FNOL slot question** (e.g., "before we continue, can I get your policy number?") at a triggered moment. Output WAV + transcript JSON show the injection took effect.

### Scope (smallest possible)

- Vendor PersonaPlex install (`pip install` from NVIDIA repo URL)
- Boot `model_service/server/main.py` locally on a GPU machine
- Implement minimal Reasoner: just a hardcoded "after 5s, force Sarah to say `'before we continue, can I get your policy number?'`"
- Drip-feed implementation: takes a string, encodes via `text_tokenizer.encode()`, yields one token per frame
- Run `offline.py`-style flow: input WAV → patched inference loop → output WAV
- Verify output text JSON contains the forced tokens at the expected frame indices

### Success criteria

- Output audio has Sarah's voice asking the forced question
- No audio degeneration (no token repetition collapse)
- Forced tokens appear in transcript JSON at the right time
- Audio quality subjectively comparable to vanilla PersonaPlex

### Out of scope

- LiveKit, Twilio, telephony — none of it
- Real Reasoner intelligence (just a stub)
- Frontend
- Slot extractor LLM calls

### Estimated effort

**~1 day.** Most time goes to: vendoring PersonaPlex, getting GPU env right, writing the per-frame loop with `text_token` injection, debugging drip-feed cadence.

### Risks

| Risk | Mitigation |
|---|---|
| `LMGen.step(text_token=...)` API doesn't behave as documented | Read `lm.py:776-892` carefully; have working `offline.py` baseline first to compare against |
| Drip-feed cadence wrong → audio degenerates | Strict 1 token / 80ms frame, enforce in code with assertions |
| Persona priming + injection interact unexpectedly | Test without persona first (vanilla Moshi behavior), then add Sarah persona |

---

## MVP 2 — Inbound phone call with full Reasoner

**Goal:** A juror calls a real phone number → talks to Sarah → call ends with structured FNOL JSON. The end-to-end demo for the hackathon judging.

### Definition of done

A juror dials a Twilio DID number from any phone → LiveKit routes to our Agent → Sarah answers, has a multi-turn FNOL conversation → call ends → backend writes complete FNOL state + transcript to disk. ≥80% of required slots are correctly captured in a 5-min test call.

### Scope

- Provision Twilio DID + SIP trunk to LiveKit
- LiveKit Agents Python SDK: `backend/app/telephony/livekit_agent.py`
- Audio bridge between LiveKit room and `model_service` WebSocket
- Real Reasoner:
  - `state.py` — FNOL Pydantic schema + slot tracker
  - `extractor.py` — async LLM call (gpt-4o-mini or Haiku) per user turn
  - `gate.py` — three intervention triggers (wrap-up / drift / compliance)
  - `drip.py` — token sequencer
- Two-channel WebSocket protocol (audio + control) between backend and model_service
- Sarah persona prompt + voice (NATF1 default)
- Call recording + transcript persistence
- Smoke-test call from a real phone

### Success criteria

- Call can be initiated and survives 5+ minutes
- ≥80% of required FNOL slots captured in test calls
- ≥50% of testers vote "human" in blind test (the actual Inca metric)
- End-to-end mouth-to-ear latency <500ms (per [`architecture.md`](architecture.md) latency budget)
- Slots captured even if asked out-of-order; wrap-up confirmation covers gaps

### Out of scope (deferred to post-MVP)

- Multi-persona switching
- Frontend monitoring UI (can be CLI logs for the demo)
- Voice cloning custom Sarah voice
- ASPIRin fine-tuning
- Production-grade error handling (just enough to survive a call)

### Estimated effort

**~2-3 days** after MVP1 is working. Major components:
- Day A: Twilio DID provisioned, LiveKit dev project set up, Agent skeleton joins a room
- Day B: Audio bridge to model_service working over WebSocket; Reasoner state + extractor running
- Day C: Intervention gate tuned; persona prompt iterated; smoke test from real phones

### Risks

| Risk | Mitigation |
|---|---|
| LiveKit ↔ Twilio SIP setup takes longer than expected | Start day 1, even before MVP1 done. Twilio EU number provisioning has lead time |
| Audio bridge has latency spikes under real WebRTC conditions | Test early with LiveKit web client before going to PSTN; measure p99 latency |
| Slot extractor is too slow / unreliable | Use Haiku (250-400ms typical); cache per-turn results; fall back to regex for critical slots |
| Modal cold-start drops first call | Use `keep_warm=True` + heartbeat; preload model on Modal startup hook |
| Sarah's voice doesn't sound like a Berlin claims rep | A/B test NATF1/2/3 with hackathon teammates within hour 1 of MVP1 |

---

## MVP 3 — ASPIRin fine-tune on PersonaPlex (post-hackathon stretch)

**Goal:** Improve robustness of timing behavior (turn-taking, backchannel, interruption handling) by RL fine-tuning PersonaPlex with ASPIRin's binary action space projection. **Not needed for hackathon submission.**

### Why this is here

ASPIRin's evaluation showed standard Moshi can be improved on Full-Duplex-Bench by RL training timing decisions. If our MVP2 demo shows juror feedback like "the agent felt slightly off in turn-taking," ASPIRin training is the principled way to fix that. Otherwise it's not on the critical path.

### Definition of done

A LoRA checkpoint on PersonaPlex weights such that, swapped in via `lm_gen.load_lora()`, it produces measurably better Full-Duplex-Bench scores (lower TOR on pause/backchannel/turn-taking) with no measurable regression in semantic naturalness or persona consistency.

### Scope

- Reproduce ASPIRin's training setup against PersonaPlex weights (instead of base Moshi)
- Source ~40-50 hours of dual-channel conversational speech with ASR timestamps
- Implement Action Space Projection on PersonaPlex's tokenizer
- GRPO RL with paper's reward functions (interruption + response, product)
- LoRA rank 256, train temporal embeddings fully
- Evaluate on Full-Duplex-Bench
- A/B test against baseline PersonaPlex in our Sarah persona setting

### Estimated effort

**~3-5 days of focused work** after MVP2, assuming we have:
- Modal H100 access (~$2-4/hr × ~12-24h training = ~$50-100 compute)
- Dual-channel training audio (the bottleneck — see "Open question" below)

---

## ⚠ The concern: can we ASPIRin-finetune PersonaPlex?

This is the one piece I want to address head-on because you flagged it.

### Short answer: yes, technically feasible. Architecturally compatible. Practical gotchas exist.

### Architectural compatibility (✅ all green)

| Requirement | PersonaPlex status |
|---|---|
| Same base architecture as Moshi | ✅ "PersonaPlex is based on the Moshi architecture and weights" — paper, README |
| 12.5Hz frame rate | ✅ Identical, uses same Mimi codec |
| EPAD/PAD/BOS/EOS tokens | ✅ Same vocab, same IDs (0/3/1/2 confirmed in `offline.py:293`) |
| LoRA-targetable linear layers | ✅ Same Transformer structure as Moshi; ASPIRin's "all linear layers" target works identically |
| Temporal Transformer embeddings | ✅ Exists in PersonaPlex, ASPIRin trains these fully |
| Binary `V_pad` vs `V_non-pad` partition | ✅ Same tokenizer, partition definition transfers directly |

### Practical gotchas (⚠ to plan around)

**1. Persona conditioning interacts with RL training.**
PersonaPlex's persona lives in KV cache after `step_system_prompts()`. ASPIRin RL was trained on base Moshi without persona priming. Two failure modes:
- **(a)** Train ASPIRin RL with no persona prompt → RL distribution doesn't match deployed Sarah-persona distribution → timing improvements may not transfer
- **(b)** Train ASPIRin RL with Sarah-only persona → timing only learned for Sarah, model overfits

**Fix:** train across **multiple personas** (rotate through PersonaPlex's 16 voice prompts × 4-5 role prompts). The timing learning should generalize because it's at the binary `{speak, silent}` level, not at the content level.

**2. Training data is the real bottleneck.**
ASPIRin used 43h "in-house" dual-channel speech with ASR timestamps. Our options:
- **Fisher English Corpus** (LDC2004T19) — paid, but PersonaPlex was already trained on a subset (1,217h). $$$, takes time to license.
- **Synthetic dual-channel** — generate using two PersonaPlex instances chatting (Character Voice Booth pattern) + parakeet ASR for timestamps. Free, fast, but lower quality than real human conversational dynamics.
- **Mix** — small Fisher subset + larger synthetic. Best quality/cost trade-off.

For MVP3, **synthetic-only is probably enough**. Generate ~50h of dual-channel synthetic dialogue with parakeet timestamps. Only fall back to Fisher if synthetic timing distribution isn't realistic enough.

**3. Reward signal needs voice activity ground truth.**
ASPIRin computes overlap (`R_int`) and latency (`R_re`) from ASR-derived voice activity timestamps. We need:
- ASR running on user channel (parakeet-tdt-0.6b-v3, same as ASPIRin)
- ASR running on agent channel (or use `agent_text_token != PAD` as proxy)
- Reward computation script

This is straightforward but adds infrastructure. Plan to write this in MVP3 prep.

**4. PersonaPlex is post-trained from Moshi.**
ASPIRin assumes you start from base Moshi. Starting from PersonaPlex means the temporal embeddings already encode "be a customer service rep" priors. RL on top will shift these. Two outcomes possible:
- **Optimistic:** RL only nudges `{PAD, non-PAD}` distribution; persona behavior preserved.
- **Pessimistic:** Temporal embedding updates leak into persona behavior, Sarah becomes slightly different post-RL.

**Mitigation:** train with very small KL penalty `β` (paper used 0.001) and validate persona consistency after every checkpoint. If persona drifts, freeze temporal embeddings and use LoRA only.

**5. Compute is fine, not a blocker.**
- ASPIRin paper: 8× V100, batch=1/GPU, 3 epochs, 43h data → roughly 24-72h training
- Our setup: 1× H100 on Modal ≈ 4× V100 throughput → ~6-18h on H100
- Cost: ~$15-50 compute. Trivial.

The bottleneck is **data and validation**, not compute.

### Recommendation for MVP3 sequencing

1. **Get MVP2 working first.** Don't start MVP3 prep until we know the demo works without it.
2. **Measure first.** Run our MVP2 system through Full-Duplex-Bench (or at least its turn-taking subset) to know our baseline TOR. If TOR is already <0.5, ASPIRin's gain (~0.36) isn't worth it.
3. **Synthetic data first.** Generate ~50h dual-channel synthetic dialogue with two PersonaPlex instances. Parakeet ASR for timestamps. Validate distribution looks like real conversation.
4. **LoRA-only ablation first.** Train with LoRA rank 256, no temporal embedding updates. If this gives any gain, expand to full ASPIRin recipe.
5. **A/B test against vanilla.** Same Sarah persona, same test calls, blind comparison.

If ASPIRin LoRA on PersonaPlex doesn't help (possible if PersonaPlex post-training already encoded better timing than base Moshi), we ship MVP2 as-is. No regret.

---

## Cross-cutting backlog (post-MVP)

These are deferred to keep MVPs sharp. Track separately as we encounter need:

- Custom Sarah voice (record WAV → encode to `.pt` voice prompt)
- Frontend monitoring UI (Next.js orb + transcript + FNOL state visualization)
- Multi-persona support (handler can switch personas mid-call for handoffs)
- MoshiRAG-style policy DB integration (when Sarah needs to look up actual coverage)
- Compliance audit trail (call recording + decision log for every Reasoner intervention)
- Multi-language support (Voxtral-Realtime would be the candidate Talker for German/French)
- ASPIRin training data pipeline as reusable infrastructure
- Production observability (latency p99 dashboards, drop-call recovery)

---

## Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-25 | Lock LiveKit + Twilio for telephony | First-class SIP integration, hackathon-day reliability |
| 2026-04-25 | Pipecat optional, add only if Smart Turn v2 needed | LiveKit Agents covers VAD + turn detection out of box |
| 2026-04-25 | No model training in MVP1/MVP2 | Public injection API + Reasoner sufficient; training is MVP3 stretch |
| 2026-04-25 | ASPIRin compat: feasible, not blocked architecturally | See "concern" section above |
| 2026-04-25 | Sarah persona only, no multi-persona | Simplifies prompt iteration; multi-persona is v2 |
| 2026-04-25 | Synthetic data > Fisher for MVP3 training | Cost, time, no licensing friction |
| 2026-04-25 | English-only confirmed; PersonaPlex stays as Talker | Inca jurors will speak English (German not required) |
| 2026-04-25 | ASR is verification backchannel, NOT primary transcript | PersonaPlex's text monologue is primary; AssemblyAI consulted only when entity confidence is low or at wrap-up. Lowers latency requirements; aligns ASR choice with entity recall (the metric we actually care about) rather than general WER |
| 2026-04-25 | AssemblyAI Universal-3 Pro over Scribe v2 for verification role | AssemblyAI has the only published entity-recall benchmark (16.7% missed entity rate). Scribe v2's lower latency advantage is irrelevant for backchannel use |
