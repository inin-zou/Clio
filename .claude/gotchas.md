# Model-tuning gotchas

Hard-won lessons from getting PersonaPlex 7B + the gate to feel like a
human claims rep on phone audio. Read this before changing anything in
`model_service/deploy/modal_app.py` or `backend/app/reasoner/`.

If you're hitting one of the symptoms below, jump straight to that
section.

---

## Audio: Sarah talks over the caller

**Symptom:** Caller starts speaking, Sarah keeps talking. Caller gets
frustrated and says "are you listening to me?"

**Root cause:** PersonaPlex is duplex — **text head and audio head are
separate**. Forcing the text token to PAD ("don't say a new word") does
NOT silence the audio head. Audio keeps streaming continuation samples.

**Fix (the real one):** server-side audio gating in `process_call`. When
the energy-VAD says caller is speaking AND no backend directive is in
flight, **zero out the agent PCM frame** before publishing to LiveKit.

```python
if vad.is_speaking and not drip.queue and drip.silent_frames_remaining == 0:
    agent_pcm_np = np.zeros_like(agent_pcm_np)
```

Forcing PAD on the text token alone is necessary but not sufficient.

**Tradeoff knob:** `SPEECH_FRAMES_FOR_TURN_START` decides how quickly
VAD flips `is_speaking` True.

| Value | Latency | Behavior |
|---|---|---|
| 2 (160ms) | Fast | Sarah pauses for short caller acks. **Current setting.** |
| 4 (320ms) | Slower | Brief acks ignored, Sarah talks over them. **Don't.** |

---

## Audio: dead air after caller finishes

**Symptom:** Caller stops speaking. Sarah doesn't respond for 5+ seconds.
Caller has to say something else to "wake her up".

**Root cause:** Sarah's text head got stuck in PAD-sampling. Without an
explicit nudge, she keeps emitting PAD tokens.

**Fix:** inject `EPAD` into the drip queue when VAD detects caller turn
boundary. EPAD = id 0, the explicit "start speaking" token.

```python
if vad.update(chunk):  # turn boundary fired
    if not drip.queue and drip.silent_frames_remaining == 0 \
       and not agent_text_buf.strip():       # ← critical guard
        drip.queue.append(EPAD_TOKEN_ID)
```

**The third guard matters.** Without `agent_text_buf.strip()` check,
EPAD fires even when Sarah just finished a sentence and was about to
naturally pause — making her too eager and stuttering into the next
phrase.

---

## Model output: Sarah hallucinates plates / policy numbers / times

**Symptom:** Caller gives no identifying info. Sarah says "Got it,
POL-2024-001" or "Did you say B as in Berlin, A as in Apple?".

**Root cause:** PersonaPlex is a generative model. With no real input
to anchor on, the text head free-samples claim-context tokens that
match its training distribution.

**Fix (persona prompt — the only reliable knob):** explicit "don't
guess" rule. Lives in `BAKED_PERSONA` in `modal_app.py` and mirrored
in `backend/app/reasoner/persona.py:BASE_PERSONA`.

```
- NEVER guess or invent identifiers (policy numbers, plate numbers,
  names, times, addresses). If the caller hasn't given you a value,
  ASK for it — don't fill it in with a placeholder. Reading back a
  fabricated value is worse than asking again.
```

The two persona files MUST stay in sync. The Modal one bakes into the
snapshot at container start; the backend one is informational (not
loaded into the model directly, but used for documentation).

---

## Model output: Sarah ignores small talk

**Symptom:** Caller says "Hey, how are you?". Sarah responds with
"What is the policy number?" — robotic and rude.

**Fix:** persona rule:

```
- If the caller says something off-topic ("how are you", small talk),
  briefly acknowledge it like a human would, then gently redirect:
  "I'm doing well, thanks. Now, can you tell me what happened?"
```

---

## GPU memory: leak ~1.75GB/min, OOMs at ~3min into call

**Symptom:** Container OOMs mid-call with `torch.OutOfMemoryError`.
GPU usage climbs ~1.75GB per minute even though the model + KV cache
should plateau at ~20GB.

**Root cause #1: shared Mimi for encode + decode.** PersonaPlex's
`offline.py` creates TWO Mimi instances — one for caller-encoder,
one for agent-decoder. Sharing one Mimi for both causes the streaming
convolution's `prev_x` buffer to grow unbounded across frames.

**Fix:** `self.mimi.encode(...)` for caller audio, `self.other_mimi.decode(...)`
for agent audio. Always.

**Root cause #2: CUDA Graphs pin tensor memory.** `_LMGenState`
contains `CUDAGraphed` objects (graphed_main, graphed_embeddings,
graphed_depth) that capture pointer-level GPU memory. PyTorch's
`empty_cache()` cannot release graph-pinned memory. `reset_streaming()`
doesn't either, because the graphs reference the underlying buffers.

**Fix #1 (lighter, recommended):** disable dynamo via image env.

```python
.env({
    ...,
    "TORCHDYNAMO_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
```

This stops graph capture entirely. Per-frame inference is ~10% slower
but stays under the 80ms/frame budget. **This is the single biggest
fix for back-to-back call survival.**

**Fix #2 (heavier, fallback):** rebuild `lm_gen` + mimi at end of every
call (`_recreate_inference_stack`). Drops the wrappers + their CUDA
graphs, keeps the model weights. Adds ~5-10s between calls. See the
method in `modal_app.py`.

**Diagnostic env vars / log lines:**
```
[setup] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   ← confirm propagation
[setup] GPU after warmup: 17.8GB used / 42.4GB              ← baseline
call XXX: post-call rebuild in 1.8s — GPU NN.NGB used        ← cleanup actual
```

---

## Persona priming: 30s per call kills the demo

**Symptom:** First inbound call lasts 8 seconds because caller hangs up
during 30s of dead air after the TwiML preamble. Logs show
`persona primed in 33.2s`.

**Root cause:** `step_system_prompts` runs the entire persona text
(~3500 chars, ~800 tokens) through the model token-by-token to seed the
KV cache with Sarah's context. ~30-40s on A100.

**Fix:** snapshot/restore. Run priming ONCE in `setup()`, save the
post-prime streaming state to `/tmp` via `save_streaming_state` (moshi
provides safetensors + JSON), then per-call restore via
`load_streaming_state` + `set_streaming_state_inplace`. Drops per-call
priming from 30s to ~1-2s.

```python
# setup() — once per container:
self.lm_gen.load_voice_prompt_embeddings(...)
self.lm_gen.text_prompt_tokens = tokenizer.encode(BAKED_PERSONA)
self.lm_gen.step_system_prompts(self.mimi)         # 30s
self.lm_gen.save_streaming_state(path, meta_path)  # 2s, on disk

# process_call() — per call:
self.lm_gen.reset_streaming()                      # critical, see below
flat = load_streaming_state(path, meta_path, device="cpu")
self.lm_gen.set_streaming_state_inplace(flat)      # 1-2s
```

### Critical detail #1: `reset_streaming` BEFORE the restore

`set_streaming_state_inplace` iterates the LIVE module's fields and
expects each to match the type pattern (Tensor vs None) of the snapshot.
Some fields are lazy-init'd None and only become Tensor after first
inference. **Always call `reset_streaming()` first** so live and
snapshot match the same baseline.

Without this you get: `KeyError: Expected to find a streaming state for
.cache.`

### Critical detail #2: load to CPU, copy to GPU per-tensor

`load_streaming_state(path, meta, device="cuda")` allocates the entire
flat dict's worth of tensors directly on GPU at once — ~1GB spike that
OOMs at call start. Use `device="cpu"`, then `set_streaming_state_inplace`
moves each tensor to the live tensor's device via `.to(value.device)`
during the in-place copy. No spike.

### Critical detail #3: don't deepcopy

`copy.deepcopy(lm_gen.get_streaming_state())` fails with
`TypeError: cannot pickle 'CUDAGraph' object`. The disk safetensors
round-trip is the only path that works.

### Critical detail #4: warmup BEFORE snapshot

`step_system_prompts` populates some streaming state fields lazily.
The snapshot must be taken AFTER a warmup pass populates them, otherwise
the field pattern doesn't match what `set_streaming_state_inplace`
expects on the next restore. Setup runs `_build_inference_stack(warmup_iters=4)`
before priming + snapshotting.

---

## Drip-feed: forced text token format

**Symptom:** `lm_gen.step(text_token=...)` errors or produces garbage.

**Fix:** must be `torch.tensor([id], device="cuda", dtype=torch.long)`.
Not `int32`. Not on CPU. Not as a Python int.

```python
forced_tensor = torch.tensor([forced_id], device="cuda", dtype=torch.long)
tokens = self.lm_gen.step(input_tokens=user_codes[:, :, c:c+1],
                          text_token=forced_tensor)
```

When constructing a SpeakDirective response, **always lead with `EPAD`**
(id 0) before the content tokens. Without EPAD the model often stays
stuck in PAD sampling and the drip just gets discarded.

```python
def force_speak(self, token_ids: list[int], ...) -> None:
    self.queue.clear()
    self.queue.append(EPAD_TOKEN_ID)        # ← critical
    self.queue.extend(token_ids)
```

---

## Gate: stuttering when two directives fire back-to-back

**Symptom:** Sarah says
`"Okay so that's X Y Z 1 2 3, is that right? Okay so that's A B C D 2 2,
is that right?"` — two readbacks crammed together, no breathing room.

**Root cause:** Gate fires readback for slot A, then immediately fires
readback for slot B (both are unconfirmed). Sarah's drip queue has
both texts queued back-to-back; she plays them with no gap.

**Fix:** `DIRECTIVE_COOLDOWN_SEC = 5` in `gate.py`. The gate refuses
to return a directive within 5s of the previous one. Lets Sarah's
prior utterance play out + gives the caller time to respond.

Implementation: `_stamp(directive, now)` records `_last_fired_at`;
`decide()` checks `now - _last_fired_at < cooldown` first.

---

## Readback: "Yeah" attributed to wrong slot

**Symptom:** Gate fires readback for `policy_number`, then `license_plate`.
Caller says "Yeah" — meant to confirm policy. Gets attributed to plate.
Gate keeps re-firing policy readback because it never sees confirmation.

**Root cause (old):** `self._pending_readback: dict | None` was a single
slot. The second gate-fire clobbered it.

**Fix:** FIFO queue `self._pending_readbacks: list[dict]`. A single
"yeah/correct" pops the OLDEST. A "both/all/correct, both of them"
drains the entire queue.

```python
if confirm_all:
    to_resolve = list(self._pending_readbacks)
    self._pending_readbacks = []
else:
    to_resolve = [self._pending_readbacks.pop(0)]
```

---

## Readback rendering: ISO datetime byte-spelled

**Symptom:** Sarah says
`"Okay so that's 2 0 2 6 dash 0 4 dash 2 6 0 5 0 0 0 0 plus 0 0 0 0,
is that right?"` — unintelligible.

**Root cause:** `render_readback("incident_datetime", value)` was running
`_spell_for_voice` on the raw ISO string.

**Fix:** dispatch in `render_readback` based on slot label.

```python
def render_readback(slot_label: str, value: str) -> str:
    if _is_datetime_slot(slot_label):     # _datetime, _at, _date, _time
        return f"... that's {_render_datetime(value)}, is that right?"
    if _is_id_slot(slot_label):           # plate, policy_number, VIN, ...
        return f"... that's {_spell_for_voice(value)}, is that right?"
    return f"... that's {value}, is that right?"
```

`_render_datetime("2026-04-26T05:00:00+00:00")` → `"7 a.m. on April 26th"`
(Berlin local).

---

## Slot extraction: garbage values from cross-talk

**Symptom:** Caller says "What? 43." (they're confused by Sarah's read-
back). Extractor assigns `license_plate = "43"`.

**Fix:** rule 10 in extractor system prompt. Identifier slots
(policy_number, license_plate, vin, *_phone, *_number) require a
clear caller statement that explicitly assigns the value
("my plate is X", "no it's X"). Reject bare digits drifting through
the transcript.

---

## Scribe ASR: livekit plugin needs explicit http session

**Symptom:** `RuntimeError: Attempted to use an http session outside of
a job context` when initializing `elevenlabs.STT()`.

**Root cause:** `livekit-plugins-elevenlabs` tries to grab a session
from `livekit.agents.utils.http_context`, which only exists inside an
agent-worker run. We invoke STT directly, not as a livekit-agents worker.

**Fix:** create our own `aiohttp.ClientSession()` and pass as
`http_session=`. Close it in the call's finally block.

```python
import aiohttp
from livekit.plugins import elevenlabs

scribe_http_session = aiohttp.ClientSession()
scribe = elevenlabs.STT(
    api_key=os.environ["ELEVENLABS_API_KEY"],
    model_id="scribe_v2_realtime",
    sample_rate=16000,
    language_code="en",
    http_session=scribe_http_session,
)
# ... at end of call:
await scribe_http_session.close()
```

Also: do NOT pass both `model_id` and `use_realtime=True` — the plugin
warns and ignores `use_realtime` when model_id is set.

Cleanup order matters: cancel the consumer task FIRST, then close the
stream, then close the http session. Reversed produces "asyncgen
already running" and "Unclosed client session" log spam.

---

## Modal: fast iteration recipe

```bash
# Fix → redeploy → test cycle:
modal app stop personaplex-clio        # force fresh container
CLIO_DEMO_MODE=1 modal deploy model_service/deploy/modal_app.py
# wait for "[setup] PersonaPlex ready" in dashboard
# dial the number
```

`modal deploy` alone does NOT swap the warm container to new code —
`min_containers=1` keeps the old one alive serving requests. `app stop`
forces a fresh start.

`max_containers=N` is **not a valid `@app.function` parameter** in this
Modal version. Don't try to set it. If you need to enforce single
container for in-memory state coherence, you'll have to find another way
(or move state to a shared store).

---

## Modal: logger.info is silently dropped

**Symptom:** You added `logger.info("call %s: ...", call_id)` calls but
they don't appear in `modal app logs`.

**Root cause:** Default Python logging level is `WARNING`. INFO is
filtered out.

**Fix:** at module top of `modal_app.py`:

```python
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    force=True,  # override any prior config from imported libs
)
```

The `force=True` matters — moshi/livekit set their own basicConfig
during import.

---

## Python: local `import time` shadows the module name

**Symptom:** `NameError: name 'time' is not defined` inside a function
that previously used `time.perf_counter()`.

**Root cause:** Python's scoping. If `setup()` does `import time` near
its top, `time` becomes a local name in **every code path of setup**,
including before the import line. Python flags any reference to a
function-local name before its assignment as undefined.

**Fix:** import `time` at module top (it's stdlib, free). Avoid local
imports of stdlib modules unless you have a reason.

This bit us once when adding `_recreate_inference_stack` — it referenced
`time.perf_counter()` but didn't `import time` itself, and the module-
level import was shadowed by setup's local one.

---

## Twilio + LiveKit SIP: dispatch rule must match

**Symptom:** Caller dials, hears the TwiML preamble, then dead silence.
Modal logs show `joined LiveKit room` and `SessionReady` but never
`caller audio subscribed`.

**Root cause:** Modal joined room `clio-<call_id>`. Caller's SIP call
landed in a DIFFERENT LiveKit room because the dispatch rule routes
inbound calls based on its own logic, not the SIP URI user-part.

**Fix:** LiveKit dispatch rule must be `Direct` type with
`roomName: clio-active` (a fixed room everyone shares). Backend then
mints `clio-active` as the room name (set
`CLIO_USE_FIXED_ROOM=1`, default). Modal joins the same fixed room.
Single concurrent call only — fine for hackathon.

If you ever support multi-call concurrency, you need either:
- An "Individual" dispatch rule that uses the SIP URI user-part as the
  room suffix (exact LiveKit syntax varies by version)
- Or LiveKit webhooks: when a SIP call arrives, LiveKit POSTs the room
  name it created, and backend then spawns Modal targeting that room
