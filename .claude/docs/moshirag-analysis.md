# MoshiRAG Analysis

**Paper:** [MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models (arXiv:2604.12928)](https://arxiv.org/abs/2604.12928)
**Authors:** Chung-Ming Chien (TTIC), Manu Orsini (Kyutai), Eugene Kharitonov (Kyutai), Neil Zeghidour (Kyutai), Karen Livescu (TTIC), Alexandre Défossez (Kyutai)
**Status:** v1 = 2026-04-14; v2 = 2026-04-17. **Code/weights NOT released** as of 2026-04-25.

## TL;DR

MoshiRAG addresses one specific question: how do you push *external knowledge* into a full-duplex speech LLM that's already mid-utterance, without making it pause? The answer is a `⟨ret⟩` trigger token in Moshi's text monologue stream + asynchronous retrieval + additive embedding injection into the temporal Transformer. Generation never stalls — the model fills the retrieval window with conversational filler ("hmm, let me think...") then transitions into the grounded body of the response when the retrieval lands.

## Mechanism

### Three-segment response structure

A knowledge-intensive response has a built-in three-part shape:

```
[lead portion]              [body portion]              [tail portion]
no external knowledge       reference-grounded          optional wrap-up
filler, hedging             actual content
   ↑
   ⟨ret⟩ inserted here
   via TTS forced alignment
```

Lead portion = filler that buys time while retrieval runs. Body portion = where injected knowledge takes effect.

### Training data construction (three-LLM role-play)

Three parallel LLMs generate transcripts:

- **user LLM** — asks knowledge-intensive questions
- **Moshi LLM** — agent side, sees only conversation history, NOT the reference
- **reference LLM** — provides ground-truth knowledge

Topics drawn from:
- 307k from Natural Questions training split
- 90k from HotpotQA
- 76k from TriviaQA
- 5.5k LLM-generated expert-domain topics
- → ~479.5k topics

Three prompt variants (v1 basic, v2 user challenges, v3 small talk) yield **~1.9M conversation instances / 47,770 hours of audio**.

`⟨ret⟩` token placed *before the first text token of the lead portion* via TTS forced alignment.

### Injection mechanism (the math)

Reference text → `ARC-Encoder` (4× compression, frozen) → projected via 1-layer trainable linear → summed into temporal Transformer input over `l` time steps:

$$
h'_i = \begin{cases} h_i + h^{\text{ref}}_{i - (i_{\text{ret}} + d/f_r)} & \text{if } i_{\text{ret}} + d/f_r < i \le i_{\text{ret}} + d/f_r + l \\ h_i & \text{otherwise} \end{cases}
$$

Where:
- `h^ref_i = proj(emb^ref_i)` — 1-layer trainable linear projection
- `f_r = 12.5 Hz` — Moshi frame rate
- `d` = actual retrieval latency
- Injection happens **only in the temporal Transformer**, not the depth Transformer
- Injection persists for `l` frames (compressed reference length)

Design choices:
- **Additive, not concat** — preserves sequence length, keeps streaming
- **Dropout 0.2** — 20% of training, the reference is replaced with a learnable `h_dropout` vector. Teaches graceful degradation when retrieval fails.
- **Reference encoder frozen, Moshi + projection trained jointly**

### Retrieval pipeline

- **NOT a fixed indexed corpus.** References are generated on the fly:
  - LLM-based: Gemma 3 27B reads conversation context, generates concise factual reference
  - Search-based: Tavily web search API
- Conversation context = ASR transcript of user + Moshi's own text output, aggregated and sent to the retrieval backend
- Target retrieval delay: **≤ 2 seconds**

### Inference flow (the key engineering trick)

```
t=0      User asks knowledge-intensive question
t=0.3s   Moshi samples ⟨ret⟩ in text stream
         ↓ async dispatch
         ┌────────────────────────────────────┐
         │ Background:                        │
         │  - collect ASR + Moshi transcripts │
         │  - call Gemma/Tavily               │
         │  - ARC-Encoder → 4× compress       │
         │  - duration ≤ 2s                   │
         └────────────────────────────────────┘
                    ↓
t=0.3-2s  Moshi continues talking — outputs lead portion
         "Hmm, let me check that for you..."
         (no external knowledge needed; just filler)
                    ↓
t≈2s     Retrieval result returns; injection window begins
         h'_i = h_i + h^ref_i for next l frames
                    ↓
t=2s+    Moshi outputs body portion, grounded in reference
```

**Generation never stalls.** If retrieval fails, dropout training lets the model fall back to its prior.

## Evaluation

### Knowledge accuracy (massive gains)

| Dataset | Vanilla Moshi | MoshiRAG | GPT-4o Audio |
|---|---|---|---|
| LlamaQ | 62.3% | **83.0%** | 88.4% |
| WebQ | 26.6% | **71.5%** | 81.0% |
| TriviaQA | 22.8% | **73.7%** | 90.6% |
| HaluEval | 10.5% | **42.0%** | 68.7% |

WebQ went from 26.6% → 71.5%, **approaching GPT-4o Audio (81%)** despite being only a 7B end-to-end speech model.

### Full-Duplex-Bench (turn-taking)

- Turn-taking latency: **0.18 s** vs vanilla Moshi 0.27 s
- Pause TOR (synthetic): 0.32 vs Moshi 0.99 (lower = better)

Even timing improved as a side effect — training data with conversational filler taught the model to use filler to mask retrieval latency.

### Keyword delay

Definition: time from response onset to when the key answer-content first appears.
- MoshiRAG: ~3.1s
- Vanilla Moshi: ~2.1s
- GPT-4o Audio: ~5.5s

### Training hyperparameters

- 100k updates
- LR 2e-6
- Batch size 32
- 7B Moshi base
- 1B ASR model

## Relevance to our project

### What we take from MoshiRAG

1. **Three-segment response structure (lead → body → tail)** — directly applicable to our Reasoner-driven UX. When the agent needs to look up a slot value or pull policy data, lead with filler ("okay let me confirm one thing"), then deliver the grounded body. No training needed to use this pattern at the prompt level.
2. **Three-LLM data construction (user / agent / reference)** — exactly the pattern for our text-first synthetic FNOL dialogue generation. The reference LLM in our case provides the policy DB / FNOL ground truth.
3. **Dropout-style robustness** — at inference, design the agent to gracefully fall back when retrieval fails ("let me just confirm this with the team and call you right back").

### What we DO NOT take from MoshiRAG

- We don't need force tool-call behavior. Our architecture uses a **passive checklist** (see `architecture-decision.md`) where the Reasoner observes and only intervenes at wrap-up or drift, not via mid-stream `⟨ret⟩` injection.
- We don't need to train. MoshiRAG requires 1.9M dialogues + 100k updates + a custom projection layer. Replicating this is a multi-week engineering project, not hackathon-scale.
- The trained projection layer + ARC-Encoder give cleaner *embedding-space* injection, but we can approximate the *behavior* using text-stream drip-feed (no training).

### What we approximate

| MoshiRAG (trained) | Our approximation (no training) |
|---|---|
| `⟨ret⟩` token learned via supervision | Reasoner externally decides when to nudge |
| ARC-Encoder + linear projection → embedding addition | Drip-feed retrieved text on the text monologue stream |
| Lead/body/tail learned end-to-end | Pre-recorded filler clips + Reasoner-driven body |
| Async retrieval with 2s target | Async retrieval; agent says "let me pull that up" while waiting |

The principled MoshiRAG path is the *right* engineering answer if you have weeks. We get ~70% of the UX benefit with ~5% of the training cost.

## Open questions / risks

- **Weights are not public yet.** 4 of 6 authors are Kyutai; Kyutai's track record on open-sourcing (Moshi, Hibiki, Mimi, Kyutai TTS/STT all open) suggests a release within 1–3 months is plausible but not guaranteed.
- **Cross-domain generalization** — MoshiRAG was trained on QA-style queries (NQ, HotpotQA, TriviaQA). Insurance claim flow is structurally different (multi-turn slot filling, not Q&A). If we ever did try to fine-tune on this approach, we'd need to construct claim-specific training data (~5k dialogues, much smaller than NQ-scale).
