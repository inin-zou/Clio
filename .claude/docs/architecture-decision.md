# Architecture Decision: Reasoner-Driven Text Injection on PersonaPlex

**Date:** 2026-04-25
**Project:** Clio (Inca Hackathon — voice agent for inbound insurance claim calls)
**Decision:** Build a Reasoner–Talker system using PersonaPlex 7B as the Talker, with Reasoner-driven `text_token` injection via the public `LMGen.step()` API. No model training required.

## The decision in one paragraph

We will use **PersonaPlex 7B off-the-shelf** as the audio-side Talker (Sarah, claims rep). All conversational intelligence — FNOL slot tracking, Empathy timing, when-to-speak — lives in an external **Reasoner** written in Python. The Reasoner observes the live transcript via Moshi's text monologue stream and intervenes by writing `text_token` arguments into `lm_gen.step()`, which is already a public API on PersonaPlex's inference loop. No fork is required. No training is required. The architecture aligns with the *passive checklist* dialog management pattern: the Reasoner is mostly a silent observer, only forcing tokens at three specific intervention triggers.

## Background: what we evaluated and ruled out

### Option A — End-to-end training a custom voice-to-voice model (Voxtral-Realtime / Moshi fine-tune)

**Rejected.** Voxtral pretraining is multi-week, multi-GPU. Even adaptation requires multi-day training runs. Per the Voxtral paper and J-Moshi (4 months of work to adapt Moshi to Japanese), end-to-end training is not hackathon-scale.

### Option B — MoshiRAG-style trained ⟨ret⟩ injection

**Rejected for hackathon, valuable conceptually.** MoshiRAG provides the gold-standard mechanism for async knowledge injection into a duplex Talker — see `moshirag-analysis.md`. But:
- 1.9M dialogues + 100k training updates required
- Code/weights not yet public (paper from 2026-04-14)
- We don't actually need force tool-call behavior for our use case

We take three things from MoshiRAG conceptually: the lead/body/tail response structure, the three-LLM data construction pattern, and the dropout-style robustness — all applied at prompt/system level, not via training.

### Option C — ASPIRin-style RL for timing

**Rejected for hackathon, validates our design.** ASPIRin proves that timing must be decoupled from content to avoid reward hacking — see `aspirin-analysis.md`. Our Reasoner–Talker split *is* this decoupling at the system level. We get ASPIRin's behavioral benefit without its 8× V100 + GRPO infrastructure cost.

### Option D — Cascaded STT + text LLM + TTS via Pipecat/LiveKit

**Considered as fallback.** Production-proven, but loses the prosodic naturalness of full-duplex. We will keep Pipecat handy as a backup if PersonaPlex stability blocks us, but the duplex feel is a key human-pass advantage we want to preserve.

### Option E — PersonaPlex with Reasoner-driven text injection

**Selected.** See below.

## Why this works: PersonaPlex's public injection API

The critical finding from reading [github.com/NVIDIA/personaplex](https://github.com/NVIDIA/personaplex):

`moshi/moshi/models/lm.py:815` — the `LMGen.step()` method already accepts a `text_token` parameter:

```python
@torch.no_grad()
def step(self,
         input_tokens: torch.Tensor=None,
         moshi_tokens: torch.Tensor=None,
         text_token: torch.Tensor=None,        # ← public injection point
         return_embeddings: bool=False):
```

Internal implementation (`lm.py:776-779`) writes the forced token directly into the state cache and marks it as teacher-forced:

```python
if text_token is not None:
    write_position = (state.offset + lm_model.delays[0]) % CT
    state.cache[:, 0, write_position] = text_token
    state.provided[:, 0, write_position] = True
```

Then in `process_transformer_output()` (`lm.py:892`), the forced token replaces the sampled one:

```python
next_text_token = torch.where(
    provided_[:, 0, 0],         # if teacher-forced
    target_[:, 0, 0],           # use forced token
    sampled_text_token          # else sample normally
)
```

The depth Transformer (audio head) is then conditioned on `next_text_token`, so **the audio output faithfully follows whatever text we forced**. This is the EPAD/PAD + drip-feed injection we need, and it's already first-class API.

### Special token IDs (from `offline.py:293`)

```python
text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
EPAD = 0    # force start of speech
BOS  = 1    # beginning of sentence
EOS  = 2    # end of sentence
PAD  = 3    # silence
```

Other tokens via `text_tokenizer.encode(text)`.

## System architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ Telephony layer                                                    │
│   Twilio SIP trunk → server.py WebSocket (or LiveKit)              │
│   24kHz mono in / out                                              │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   PersonaPlex 7B    │
                │   (Talker = Sarah)  │
                │                     │
                │   Voice prompt:     │ ← NATF1 or NATF2 (.pt file)
                │   Text prompt:      │ ← "You work for Allianz Claims..."
                │                     │
                │   per-frame loop:   │
                │   lm_gen.step(      │
                │     input_tokens=…, │ ← live mic audio
                │     text_token=??   │ ←──── injection point
                │   )                 │
                └─────────┬───────────┘
                          │
                ┌─────────▼───────────────────────┐
                │     Reasoner (Python)           │
                │                                 │
                │  ┌───────────────────────────┐  │
                │  │ Slot Extractor (LLM call) │  │ ← runs every user turn
                │  │  transcript → {slots}     │  │   ~200ms latency
                │  └───────────────────────────┘  │
                │  ┌───────────────────────────┐  │
                │  │ FNOL Stack                │  │
                │  │  policy_num: ✗            │  │
                │  │  incident_dt: ✓           │  │
                │  │  injuries: ✗ critical     │  │
                │  │  ...                      │  │
                │  └───────────────────────────┘  │
                │  ┌───────────────────────────┐  │
                │  │ Intervention Gate         │  │
                │  │  most of the time: NO-OP  │  │
                │  │  3 trigger conditions:    │  │
                │  │    1. wrap-up gate        │  │
                │  │    2. drift detection     │  │
                │  │    3. compliance deadline │  │
                │  └───────────────────────────┘  │
                │             ↓                   │
                │     next_token_decision         │
                │       None | EPAD | PAD |       │
                │       drip-feed token id        │
                └─────────────────────────────────┘
```

## The Reasoner's three intervention modes

```python
def reasoner_decide(state, transcript, user_audio_active) -> Optional[int]:
    """
    Returns a text_token id to force, or None to let PersonaPlex sample freely.
    Called once per Moshi frame (every 80ms).
    """
    # Default: no intervention. PersonaPlex (Sarah) speaks freely.
    if not state.intervention_pending():
        return None

    if state.mode == "WRAP_UP":
        # Stack non-empty + caller signaling end of call.
        # Force a confirmation question that covers missing slots.
        return state.drip.next_token()  # drips "Before I let you go..."

    if state.mode == "NUDGE":
        # Caller has been off-topic >30s, critical slot still missing.
        # Force a gentle redirect.
        return state.drip.next_token()  # drips "Sorry to interrupt — ..."

    if state.mode == "COMPLIANCE":
        # Hard deadline: e.g. injuries question must be asked by 4 min mark.
        return state.drip.next_token()  # drips "I just need to confirm..."

    return None
```

95% of the time `reasoner_decide()` returns `None` — Sarah talks freely with her PersonaPlex-trained naturalness intact. The Reasoner only takes the wheel when the FNOL state genuinely demands it.

## The actual code patch (~30 lines)

Modify `personaplex/moshi/moshi/offline.py` (or `server.py` for WebSocket use) — replace the per-frame loop at lines 267–295:

```python
# Original loop iterates user_encoded chunks and calls lm_gen.step
for user_encoded in lm_encode_from_sphn(...):
    for c in range(user_encoded.shape[-1]):
        step_in = user_encoded[:, :, c:c+1]

        # ★ NEW: Reasoner decides whether to inject
        forced_text_id = reasoner.next_token(
            generated_text_tokens=generated_text_tokens,
            user_audio_active=detect_voice_activity(step_in),
        )
        forced_tensor = (
            torch.tensor([forced_text_id], device=device)
            if forced_text_id is not None else None
        )

        tokens = lm_gen.step(step_in, text_token=forced_tensor)
        if tokens is None:
            continue

        pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
        generated_frames.append(pcm)

        # Feed transcript back to Reasoner for slot extraction
        text_token_id = tokens[0, 0, 0].item()
        generated_text_tokens.append(text_token_id)
        reasoner.observe(text_token_id, role="agent")
        # User-side ASR (separate path) calls reasoner.observe(..., role="user")
```

## Persona priming (no code changes — use existing API)

```python
text_prompt = (
    "You work for Allianz Claims Berlin and your name is Sarah. "
    "You are an experienced claims representative. Be empathetic and calm. "
    "When callers are upset, acknowledge their feelings before asking questions. "
    "..."
)
voice_prompt = "NATF1.pt"  # try NATF2 as an alternative

# Standard PersonaPlex priming flow:
lm_gen.text_prompt_tokens = text_tokenizer.encode(
    wrap_with_system_tags(text_prompt)
)
lm_gen.load_voice_prompt(voice_prompt_path)
lm_gen.step_system_prompts(mimi)  # baked into KV cache
```

## Build sequence

| Step | Output | Effort |
|---|---|---|
| 1. Vendor PersonaPlex into `Clio/personaplex/` | Working `python -m moshi.offline` baseline | < 1h |
| 2. Choose voice (test NATF1, NATF2, NATM2 with Sarah persona prompt) | Audio sample, decide on voice | < 1h |
| 3. Three-LLM text dialogue generator | 50 valid FNOL transcripts in `Clio/data/dialogues.jsonl` | half day |
| 4. Slot extractor + FNOL state machine | `reasoner/state.py`, `reasoner/extractor.py` | half day |
| 5. Intervention gate | `reasoner/gate.py` with three triggers | 2-3h |
| 6. Drip-feed injection patch on `offline.py` | `~30 lines diff applied | 1h |
| 7. Wire to telephony (Twilio + PersonaPlex `server.py`) | First end-to-end call | half day |
| 8. Iterate persona/prosody/timing on real test calls | Tuned demo | continuous |

Roughly **2 days of focused work** for a working end-to-end demo.

## What we are explicitly NOT doing

- **No training.** Not Voxtral-Realtime, not LoRA on Moshi, not MoshiRAG-style projection, not ASPIRin RL. All deferred post-hackathon.
- **No fork of PersonaPlex.** The injection API is public; we vendor it as a dependency.
- **No multi-persona.** One Sarah, period. Multi-persona is a v2 problem.
- **No active inject-on-every-turn (MoshiRAG-style).** Passive checklist with rare interventions.
- **No custom telephony stack.** Twilio SIP + PersonaPlex's existing WebSocket server (or Pipecat fallback).
- **No primary independent ASR.** PersonaPlex's text monologue is the primary transcript for slot extraction. We add a parallel ASR (AssemblyAI Universal-3 Pro) **only as a verification backchannel** consulted when entity confidence is low or at wrap-up — see [`architecture.md`](architecture.md#verification-asr-backchannel).

## Risks and unknowns

| Risk | Probability | Mitigation |
|---|---|---|
| PersonaPlex inference loop too tightly coupled to allow `text_token` injection per frame in server mode (only `offline.py` confirmed) | Medium | If `server.py` doesn't expose the same API cleanly, vendor `offline.py` as the inference path and bridge to telephony manually |
| Drip-feed text causes audio degeneration when injected too fast (VAOS journal documented this) | High | Strict 12.5Hz cadence, ~20 chars/frame max. Do not burst-inject. |
| Sarah's voice (NATF1/NATF2) doesn't match the "Berlin claims rep" mental image jurors expect | Medium | A/B test the 4 NATF voices early. Fallback: record a custom voice prompt WAV and use `lm_gen.load_voice_prompt(wav_path)` |
| Force-injected tokens cause prosody artifacts on rare words (brand names like "Allianz") | Low–Medium | Phonetic respelling fallback ("ah-lee-ahnts") for any term that drifts in early tests |
| Slot extractor fails on noisy / accented transcripts | Medium | Use Smart Turn v2 + Whisper for clean transcripts before passing to extractor |
| Latency budget exceeded (target: <1s response, <1s no overlap) | Medium | Reasoner runs async; LLM extractor on small/fast model (Haiku or gpt-4o-mini) |

## Definition of done (demo-ready)

A juror can call a Twilio number, role-play a claimant after a car accident, and have a 3–5 minute conversation with Sarah where:

1. Sarah opens with empathy ("Are you somewhere safe right now?") before asking factual questions
2. Sarah captures all required FNOL slots through natural conversation, not interrogation
3. Sarah handles thinking pauses without interrupting
4. Sarah handles caller barge-in gracefully
5. At wrap-up, Sarah confirms captured details ("Just to confirm what I have...")
6. End-of-call: structured FNOL JSON written + complete call transcript
7. ≥50% of jurors vote "human" in blind test

## References

- `moshirag-analysis.md` — what we took from MoshiRAG conceptually
- `aspirin-analysis.md` — what we took from ASPIRin conceptually
- [PersonaPlex paper (arXiv:2602.06053)](https://arxiv.org/abs/2602.06053)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [PersonaPlex weights (HuggingFace)](https://huggingface.co/nvidia/personaplex-7b-v1)
- [Moshi paper (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [VAOS Voice Bridge debugging journal](https://gist.github.com/jmanhype/5aefd67d9e67b37a8b408abdab39b6d3)
- [Can Speech LLMs Think while Listening? (arXiv:2510.07497)](https://arxiv.org/abs/2510.07497) — formal treatment of EPAD/PAD steering
