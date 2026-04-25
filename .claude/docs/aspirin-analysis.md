# ASPIRin Analysis

**Paper:** [ASPIRin: Action Space Projection for Interactivity-Optimized Reinforcement Learning in Full-Duplex Speech Language Models (arXiv:2604.10065)](https://arxiv.org/abs/2604.10065)
**Authors:** Chi-Yuan Hsiao, Ke-Han Lu, Yu-Kuan Fu, Guan-Ting Lin, Hsiao-Tsung Hung, Hung-yi Lee
**Affiliations:** NTU + ASUS Open Cloud Infrastructure + NVIDIA AI Tech Center
**Status:** Code/weights NOT released as of 2026-04-25.

## TL;DR

ASPIRin solves a different question than MoshiRAG: how do you RL-train a full-duplex speech LLM to learn *when to speak* (vs stay silent) without destroying *what it says*? Standard token-level RL (e.g., GRPO) on timing rewards causes catastrophic reward hacking — the model learns to produce repetitive, incoherent text to game the latency metric. ASPIRin's solution: project the entire vocabulary onto a binary {active, silent} state, run GRPO on the **binary policy only**, leave the underlying token distribution untouched.

## The problem ASPIRin is fixing

If you take base Moshi and apply standard GRPO with rewards like "low overlap with user speech, low response latency," the model converges on a degenerate strategy:

- Repeat short tokens to "always be speaking" (gives perfect response latency)
- Or talk over the user (low latency, ignores interruption penalty)

The paper's diagnosis: **fine-grained token policy cannot simultaneously optimize timing and semantics with the same gradient.** The two objectives conflict, the model reward-hacks.

ASPIRin's insight: **separate the action space.** Make the RL signal flow through a *binary* speak/silent decision only; the token-content distribution is left alone.

## Mechanism

### Vocabulary partition

The vocabulary `V` is partitioned into:
- `V_pad` = padding tokens (= silence)
- `V_non-pad` = everything else (= active speech)

Binary state: `s_t = 𝕀(y_t ∈ V_non-pad) ∈ {0, 1}`.

The paper does not enumerate additional special tokens for the partition — it's effectively `[PAD]` vs everything else.

### Action Space Projection (the math)

Given raw token logits `z_θ(v | x_<t, s_<t)`, compute a binary state policy:

**Step 1 — sum logits within each class:**

$$
z'_\theta(s_t \mid x_{<t}, s_{<t}) = \sum_{v \in \mathcal{V}_{s_t}} z_\theta(v \mid x_{<t}, s_{<t})
$$

**Step 2 — softmax over the two state logits:**

$$
\pi'_\theta(s_t \mid \cdot) = \frac{\exp(z'_\theta(s_t \mid \cdot))}{\sum_{s \in \{0,1\}} \exp(z'_\theta(s \mid \cdot))}
$$

This is **sum-then-softmax** (not log-sum-exp — a surprising choice).

After the binary state is decided, the specific token is sampled normally from the original logits (the paper is silent on the exact protocol but the architecture only makes sense if token sampling is unmodified).

### Reward functions

Two scalar rewards, both hard-threshold indicators over the rollout:

$$
R_{\text{int}} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(o_k \le 1.0\,\text{s})
$$

$o_k$ = overlap (in seconds) between the k-th model utterance and any user utterance. ≤1s = pass.

$$
R_{\text{re}} = \frac{1}{K} \sum_{k=1}^K \mathbb{I}(l_k \le 1.0\,\text{s})
$$

$l_k$ = latency from user utterance end to model utterance start. ≤1s = pass.

**Total reward = product (NOT sum):**

$$
R_{\text{total}} = R_{\text{int}} \times R_{\text{re}}
$$

The product means you cannot trade interruption for latency. Both must clear the threshold.

ASR timestamps for user voice activity from `nvidia/parakeet-tdt-0.6b-v3`.

### GRPO objective (with projection)

Standard GRPO modified to use the binary policy:

$$
\mathcal{L}_{\text{ASPIRin}}(\theta) = -\frac{1}{\sum |s_i|} \sum_{i=1}^G \sum_{t=1}^{|s_i|} \left[ \frac{\pi'_\theta(s_{i,t} \mid \cdot)}{\pi'_{\theta_{\text{old}}}(s_{i,t} \mid \cdot)} \hat{A}_{i,t} - \beta\, \mathbb{D}_{\text{KL}}\!\left[\pi'_\theta \,\|\, \pi'_{\text{ref}}\right] \right]
$$

- `Â_i,t = (R_total,i − μ_R) / σ_R` — group-relative advantage normalization
- `β = 0.001` — KL penalty coefficient
- KL computed on the binary policy, NOT on the token policy

Critical: gradients flow through `π'_θ` only. The cluster-internal token distribution gets zero RL signal → semantics preserved.

### Training setup

| Item | Value |
|---|---|
| Data | 43h in-house dual-channel conversational speech |
| Clips | ~1,300 × 2-min |
| Group size G | 2 |
| Epochs | 3 |
| Optimizer | AdamW |
| LR | 1e-5 |
| Hardware | 8× V100, batch=1/GPU |
| LoRA rank | 256 (all linear layers) |
| Fully trained | Temporal Transformer embeddings |
| Frozen | Everything else |

ASR preprocessing: discard examples with <50% active speech.

## Evaluation

### Full-Duplex-Bench (TOR = Take-Over Rate, lower is more natural)

| Dimension | Moshi base | Standard GRPO | **ASPIRin** |
|---|---|---|---|
| Pause Handling TOR | 0.467 | 0.642 (worse) | **0.482** |
| Backchanneling TOR | 0.495 | 0.704 (worse) | **0.486** |
| Turn-Taking TOR | 0.436 | 0.709 (worse) | **0.364** |
| User Interrupt Latency | 0.265s | 0.153s | 0.273s |
| User Interrupt GPT-4o score | 3.894 | 3.247 | **3.734** |

**Standard GRPO is uniformly worse than the base model** — that's the cost of reward hacking. ASPIRin is the only RL approach that helps without degrading.

### Repetition (the smoking gun for reward hacking)

| Metric | Standard GRPO | ASPIRin | Reduction |
|---|---|---|---|
| 2-gram seq-rep-n | 0.117 | **0.054** | −54% |
| 3-gram seq-rep-n | 0.072 | **0.029** | −60% |

ASPIRin cuts repetition more than half compared to standard GRPO, confirming the projection prevents the model from gaming the reward via repetitive output.

## Why it works (the conceptual point)

> *"mapping to a binary 'speak or not' decision concentrates learning on timing alone. The model thus discovers that silence can be rewarding."*

The token-level RL formulation has a hidden assumption: that any improvement in "when to speak" must be expressible as a change in "what tokens to emit." For a duplex model, that's wrong — silence is a first-class action, separable from content. Separating them in the RL formulation lets the model learn timing without distorting content.

This is a *meta*-architectural choice: change the action space, keep the model.

## Relevance to our project

### What we take from ASPIRin

1. **Conceptual separation of timing from content.** Our passive checklist architecture already does this at the system level — the Reasoner controls *when* to nudge, the Talker controls *what* the nudge sounds like. ASPIRin validates this design split is principled.
2. **Reward design** — the product structure (`R_int × R_re`) is directly applicable to evaluating our system end-to-end. We can use the same metrics to measure if our agent is winning the human-pass test:
   - Overlap with user > 1s = AI tell
   - Response latency > 1s = AI tell
3. **Why we don't need to train.** ASPIRin requires GRPO infrastructure, ASR-timestamped rollouts, 8× V100, and 43h labeled data. None of that fits a 48h hackathon. But the *behaviors* ASPIRin trains into the model — appropriate timing — we can approximate with:
   - Smart Turn v2 (Pipecat) for endpointing
   - Force EPAD/PAD on Moshi's text monologue stream (already a public API: `LMGen.step(text_token=...)`)
   - Reasoner-driven decision logic for when to speak

### What we DO NOT take from ASPIRin

- We don't train. The RL pipeline cost is multi-day even with ideal infrastructure.
- We don't apply Action Space Projection at inference. The projection is a *training-time* construct; at inference, ASPIRin samples normally.

### How we get ASPIRin's behavioral effects without training

| ASPIRin (trained) | Our approximation (inference-only) |
|---|---|
| Binary timing policy learned via RL | Reasoner emits `{SPEAK, LISTEN, NUDGE}` action per frame |
| `R_int` interruption penalty | Smart Turn v2 + VAD ensures we don't speak during user audio |
| `R_re` response latency | EPAD injection forces speak within 80ms when Reasoner says SPEAK |
| Reward hacking prevented by projection | We never train on token-level rewards, so no hacking surface exists |

Effectively: ASPIRin teaches the model to internalize the timing logic. We externalize the same logic into the Reasoner. Same behavior, different software boundary.

## Open questions / risks

- **No public weights.** Hung-yi Lee's NTU group historically open-sources speech tooling (SUPERB benchmark, etc.); plausible release within months but not committed.
- **The "what comes after the binary decision" gap.** The paper doesn't fully specify how token sampling works once the binary state is set to "active." If reproducing, this is the implementation detail to reverse-engineer.
- **Could ASPIRin compose with our externalized Reasoner?** Yes, in principle — train ASPIRin once for baseline timing, layer Reasoner-driven EPAD overrides on top for hard-rule cases (wrap-up, compliance). Out of scope for hackathon, sensible direction post-hackathon.
