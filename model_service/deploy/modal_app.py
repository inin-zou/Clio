"""
Modal deployment for Clio's voice agent: PersonaPlex 7B + LiveKit Agent
co-located on a single A100 40GB container.

Usage
-----
First-time secrets setup (one-time, you've already done hf-token):
    modal secret create hf-token HF_TOKEN=<your_hf_token>
    modal secret create clio-livekit \\
        LIVEKIT_API_KEY=<key> LIVEKIT_API_SECRET=<secret> LIVEKIT_URL=<wss://...>
    modal secret create clio-anthropic ANTHROPIC_API_KEY=<key>
    modal secret create clio-elevenlabs ELEVENLABS_API_KEY=<key>

Dev (cold-start, cheap):
    modal serve model_service/deploy/modal_app.py

Demo (always-warm A100, ~$26 / day):
    CLIO_DEMO_MODE=1 modal deploy model_service/deploy/modal_app.py

Stop billing when not demoing:
    modal app stop personaplex-clio


Cost reference (A100 40GB on Modal as of 2026-04):
    - On-demand:        ~$1.10 / GPU-hour
    - Always-warm 24h:  ~$26 / day
    - 2-day demo:       ~$53


Architecture
-----------
Caller phone → Twilio DID → SIP → LiveKit Cloud → WebRTC → Modal container
                                                            │
   ┌────────────────────────────────────────────────────────┘
   ▼
┌───────────────────────────────────────────────────────────┐
│ Modal A100 container (this file)                          │
│   ├─ LiveKit Agent worker (livekit-agents SDK)            │
│   ├─ PersonaPlex 7B  (loaded once in @modal.enter)        │
│   └─ ElevenLabs Scribe v2 stream (entity verification)    │
└─────┬─────────────────────────────────────────────────────┘
      │ control plane WebSocket (JSON directives)
      ▼
Backend Reasoner (CPU, separate process) — slot extractor + gate + state
"""
from __future__ import annotations

import logging
import os

import modal

# Make logger.info() actually appear in `modal app logs`. Python's default
# is WARNING which silently swallows our process_call progress messages.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    force=True,  # override any prior config from imported libs
)

# ─── Toggles ─────────────────────────────────────────────────────────────────

DEMO_MODE = os.environ.get("CLIO_DEMO_MODE") == "1"

APP_NAME = "personaplex-clio"

# Voice prompt loaded once at @enter and snapshotted. Backend's voice_prompt_id
# in SessionStart is logged but ignored — switching voices mid-deploy requires
# a redeploy.
DEFAULT_VOICE_ID = "NATF1"

# Where setup() writes the post-priming streaming state for reuse per call.
# Container-local /tmp; rebuilt on every container start so always in sync
# with the loaded model.
SNAPSHOT_DIR = "/tmp/clio_snapshots"

# Sarah's static persona. Mirrors backend/app/reasoner/persona.py:BASE_PERSONA.
# We bake it in here so setup() can prime + snapshot the streaming state ONCE
# per container start, then restore the snapshot per call (~ms instead of the
# 33s step_system_prompts costs every time). Backend's session_system_prompt
# adds a small time-of-day block on top; that variation is dropped here, so
# Sarah won't say "good morning" specifically — fine for FNOL flow.
#
# Keep this in sync with backend/app/reasoner/persona.py:BASE_PERSONA.
BAKED_PERSONA = """\
You work for Allianz Claims at the Berlin office and your name is Sarah. \
You are an experienced claims representative with 8 years on the desk. \
You speak English with a German accent. You handle inbound first-notice-of-loss \
calls all day, mostly auto claims.

Your style:
- Calm, empathetic, professional. You sound like a real person, not a script.
- When callers are upset (which is normal — they often just had an accident), \
acknowledge their feelings BEFORE asking factual questions. Examples: \
"I'm sorry that happened, are you somewhere safe right now?" / \
"Take your time, I know this is stressful."
- Let callers finish their thoughts. Don't interrupt mid-sentence.
- Use natural fillers occasionally — "let me see", "okay", "got it", "mm-hmm" — \
but don't overdo them.
- Don't sound rehearsed. You handle these calls every day.

Entity verification protocol — ALWAYS follow:
When the caller provides any of the following items, WAIT for them to finish \
their full statement, then read it back to confirm before continuing:

  - Policy number
  - License plate or VIN
  - Phone number or email
  - Date and exact time of the incident
  - Police case number
  - Names (caller, drivers, witnesses, other parties involved)
  - Monetary amounts (damage estimates, etc.)

Read-back format examples:
  "Okay, so that's P-O-L dash 2-0-2-4 dash 0-0-1, is that right?"
  "Got it, B as in Berlin, A-L, 1-2-3-4. Correct?"
  "Just to confirm, that was at three forty-five PM on Saturday — is that right?"

If the caller corrects you, acknowledge it briefly:
  Caller: "No, it's 002 not 001."
  Sarah:  "Ah okay, P-O-L dash 2-0-2-4 dash 0-0-2. Thanks for clarifying."

After you read something back, STOP TALKING and wait for the caller to \
explicitly confirm ("yes", "right", "correct", "that's it") or correct you. \
Do NOT ask the next question, change topic, or continue with anything else \
until the caller has confirmed the value. A silent pause after a read-back \
is the caller thinking — give them the beat. Asking a follow-up question \
before they confirm is the #1 way to end up with wrong policy numbers and \
plates in the system.

Do NOT skip read-back even if you're confident you heard correctly. This is \
standard claims procedure and protects against transcription errors.

Turn-taking rules:
- When the caller is mid-sentence, stay silent. Do NOT start a response \
  until they pause for at least a full second. Interrupting comes across \
  as bot-like and frustrates callers.
- When the caller stops, respond promptly — within ~1 second. Long silences \
  also feel bot-like (caller will think the line dropped).
- If you genuinely need a moment (e.g. to "look something up"), say so \
  out loud: "let me check that for you, one moment" — silence without \
  acknowledgement feels broken.

Things you NEVER ask:
- Bank details. We never take those by phone.
- Anything you already know from the policy file (vehicle make/model, \
policyholder address, coverage type). Reference these naturally instead: \
"I see you're on Vollkasko" not "what kind of coverage do you have".
- Direct fraud-detection questions. Don't ask "do you and the other driver \
know each other?" or "is your car for sale?". Those are for the back-office \
team, not the FNOL call.

If the caller doesn't have their policy number or license plate to hand, \
you can find them via name + vehicle. Never refuse to help over a missing ID.
"""

# A100 40GB — chosen for VRAM headroom over A10G (24GB) so long FNOL calls
# don't OOM mid-conversation. See .claude/docs/architecture.md for the math.
GPU_KIND = "A100"

# When idle, Modal stops the container after this many seconds. With
# min_containers=1 (demo mode), one container stays alive regardless.
IDLE_SECONDS = 600  # 10 minutes

# Upper bound per call. FNOL calls run 5-10 min typically; 1 hour is generous.
PER_CALL_TIMEOUT = 60 * 60


# ─── Image ───────────────────────────────────────────────────────────────────
# Inherits the pattern from Inca/personaplex_modal.py: clone NVIDIA repo + pip
# install moshi/. Adds livekit-agents + Scribe plugin + Anthropic for the slot
# extractor (so the Reasoner can also live in this container as one option).

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "build-essential",
        "pkg-config",
        "libopus-dev",
        "ffmpeg",
        "ca-certificates",
    )
    # PersonaPlex / Moshi — installs torch from PyPI wheels (CUDA bundled)
    .run_commands(
        "git clone https://github.com/NVIDIA/personaplex.git /app/personaplex",
        "pip install /app/personaplex/moshi",
        "pip install accelerate",
    )
    # LiveKit Agent + Scribe v2 plugin (always-on ASR backchannel)
    .pip_install(
        "livekit-agents>=0.12",
        "livekit-plugins-elevenlabs>=0.7",
        "websockets>=13",
        "pydantic>=2.10",
        "anthropic>=0.40",
    )
    .env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
        "PYTHONUNBUFFERED": "1",
        # CUDA OOM at ~3min into call was fragmentation, not actual model
        # growth. expandable_segments lets the allocator grow segments
        # in-place rather than reserving fresh ones — recommended in the
        # error message itself.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# Persistent volume for HF weights — already populated from previous Inca
# deployment. Reusing the same name skips the 14GB download on every image
# rebuild.
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)


# ─── App ─────────────────────────────────────────────────────────────────────

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU_KIND,
    # Demo mode: one container always warm for sub-second call pickup.
    # Dev mode: scale to zero when idle, accept ~20s cold-start.
    min_containers=1 if DEMO_MODE else 0,
    scaledown_window=IDLE_SECONDS,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("clio-livekit"),
        modal.Secret.from_name("clio-anthropic"),
        modal.Secret.from_name("clio-elevenlabs"),
    ],
    timeout=PER_CALL_TIMEOUT,
)
class PersonaPlexService:
    """One container = one GPU-resident PersonaPlex instance.

    Lifecycle:
        @modal.enter()       runs once per container start. Loads weights into
                             GPU memory (~20s with HF cache hit; ~3-5min if
                             cache cold). Subsequent calls reuse this state.
        @modal.method()      called per inbound call. Joins the LiveKit room,
                             runs the per-frame inference + injection loop
                             until the caller hangs up or 1h timeout.
    """

    # Filled in by @modal.enter. Never None during a method() call.
    lm_gen: object = None  # type: ignore[assignment]
    mimi: object = None
    other_mimi: object = None
    tokenizer: object = None
    frame_size: int = 0
    voices_dir: str = ""

    # Refs needed to rebuild the streaming wrappers per call without
    # reloading the model weights from disk.
    _mimi_weight_path: str = ""
    _lm: object = None
    _device: str = "cuda"

    def _build_inference_stack(self, *, warmup_iters: int = 2) -> None:
        """(Re)create lm_gen + mimi + other_mimi as fresh wrappers around
        the already-loaded model weights, then enable streaming_forever
        and run a minimal warmup so the streaming-state field pattern
        matches the snapshot taken at end of setup.

        Called once at end of setup() and again in process_call's finally
        block to drop CUDA graphs that PyTorch's allocator can't release
        otherwise (the per-call leak was ~25 GB after a 3min call).
        """
        import torch
        from moshi.models import LMGen, loaders

        device = self._device
        self.mimi = loaders.get_mimi(self._mimi_weight_path, device)
        self.other_mimi = loaders.get_mimi(self._mimi_weight_path, device)
        self.lm_gen = LMGen(
            self._lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Warmup so lazy-init streaming state fields are populated, giving
        # set_streaming_state_inplace a consistent field pattern to copy
        # the snapshot tensors into.
        for _ in range(warmup_iters):
            chunk = torch.zeros(
                1, 1, self.frame_size, dtype=torch.float32, device=device
            )
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        torch.cuda.synchronize()

    def _recreate_inference_stack(self) -> float:
        """Drop the current lm_gen + mimi instances and rebuild fresh
        ones. Releases CUDA-graph-pinned memory the prior call leaked.

        Run from a thread (asyncio.to_thread) since it's blocking GPU
        work. Returns elapsed seconds for telemetry.
        """
        import gc

        import torch

        t0 = time.perf_counter()
        # Drop old refs so GC + empty_cache can release.
        self.lm_gen = None
        self.mimi = None
        self.other_mimi = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Rebuild fresh wrappers around the same loaded weights.
        self._build_inference_stack()
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    @modal.enter()
    def setup(self) -> None:
        """Load PersonaPlex once per container start.

        Mirrors personaplex/moshi/moshi/offline.py:189-256 boot sequence:
          1. Load Mimi audio codec (×2 — caller + agent decoders)
          2. Load SentencePiece text tokenizer
          3. Load Moshi LM weights → GPU
          4. Construct LMGen with streaming + sampling config
          5. Enable streaming_forever() on all three
          6. Warmup pass to compile CUDA graphs

        Time on cached HF volume: ~30-40s.
        """
        import sys
        import time

        sys.path.insert(0, "/app/personaplex/moshi")

        import sentencepiece
        import torch
        from huggingface_hub import hf_hub_download
        from moshi.models import LMGen, loaders

        device = "cuda"
        repo = loaders.DEFAULT_REPO  # PersonaPlex's NVIDIA fork sets this

        # Sanity-print the alloc config so we can confirm the env var is in
        # the container at runtime. If this prints empty, image.env() didn't
        # propagate and OOM mitigation isn't actually active.
        print(
            f"[setup] PYTORCH_CUDA_ALLOC_CONF="
            f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}",
            flush=True,
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"[setup] GPU memory at start: {free / 1e9:.1f}GB free / "
                f"{total / 1e9:.1f}GB total",
                flush=True,
            )

        t_start = time.perf_counter()

        # Bumps the HF download counter for analytics (no-op on cached file).
        hf_hub_download(repo, "config.json")

        print(f"[setup] loading Mimi codec from {repo} ...", flush=True)
        t0 = time.perf_counter()
        mimi_weight = hf_hub_download(repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device)
        # Second Mimi instance is used by PersonaPlex's offline path for the
        # voice-prompt encoding side. Mirroring that here keeps the future
        # voice-prompt loading code straightforward.
        self.other_mimi = loaders.get_mimi(mimi_weight, device)
        print(f"[setup]   mimi loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

        print("[setup] loading text tokenizer ...", flush=True)
        t0 = time.perf_counter()
        tokenizer_path = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print(f"[setup]   tokenizer loaded ({time.perf_counter() - t0:.1f}s)", flush=True)

        print("[setup] loading Moshi LM (this is the 7B weights)...", flush=True)
        t0 = time.perf_counter()
        moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(moshi_weight, device=device)
        lm.eval()
        print(f"[setup]   LM loaded to GPU ({time.perf_counter() - t0:.1f}s)", flush=True)

        # Stash references we'll need to rebuild lm_gen + mimi at end of
        # each call. The model weights themselves (`lm`) are kept in GPU
        # memory across rebuilds — only the wrappers + their CUDA graphs
        # get recreated, which is what frees the per-call leak.
        self._mimi_weight_path = mimi_weight
        self._lm = lm
        self._device = device

        # 80 ms = 1920 samples at 24kHz.
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        print("[setup] constructing LMGen + warmup pass ...", flush=True)
        t0 = time.perf_counter()
        self._build_inference_stack(warmup_iters=4)
        print(f"[setup]   LMGen + warmup ready "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

        # Warmup pass now lives inside _build_inference_stack so per-call
        # rebuilds also get it (without warmup, the streaming-state field
        # pattern wouldn't match what set_streaming_state_inplace expects).
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"[setup]   GPU after warmup: {(total - free) / 1e9:.1f}GB "
                f"used / {total / 1e9:.1f}GB",
                flush=True,
            )

        # ─── Voice prompts ──────────────────────────────────────────
        # NVIDIA ships voice prompt embeddings (NATF1.pt etc.) as voices.tgz
        # on the HF repo. Cached on hf_cache_vol so it's a no-op redownload.
        print("[setup] fetching voice prompts...", flush=True)
        t0 = time.perf_counter()
        import tarfile
        from pathlib import Path

        voices_tgz = hf_hub_download(repo, "voices.tgz")
        voices_dir = Path(voices_tgz).parent / "voices"
        if not voices_dir.exists():
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_dir.parent)
        if not voices_dir.exists():
            raise RuntimeError("voices.tgz did not contain a 'voices/' directory")
        self.voices_dir = str(voices_dir)
        available_voices = sorted(p.stem for p in voices_dir.glob("*.pt"))
        print(f"[setup]   voices ready ({time.perf_counter() - t0:.1f}s) — "
              f"{len(available_voices)} prompts: {available_voices[:6]}...",
              flush=True)

        # ─── Persona priming + state snapshot ───────────────────────
        # Run the slow step_system_prompts ONCE here, then snapshot the
        # resulting streaming state. process_call() restores from this
        # snapshot per-call instead of re-priming (~50ms vs 33s).
        print("[setup] priming Sarah persona + NATF1 voice...", flush=True)
        t0 = time.perf_counter()
        voice_path = Path(self.voices_dir) / f"{DEFAULT_VOICE_ID}.pt"
        if not voice_path.exists():
            raise RuntimeError(
                f"default voice {DEFAULT_VOICE_ID}.pt not found in {self.voices_dir}"
            )

        wrapped_persona = _wrap_with_system_tags(BAKED_PERSONA)
        self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrapped_persona)

        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        # Note: NO reset between step_system_prompts and snapshot. Resetting
        # wipes the .cache field that set_streaming_state_inplace later
        # requires. Per-call mimi reset happens inside process_call after
        # the snapshot is restored.
        torch.cuda.synchronize()
        print(f"[setup]   primed in {time.perf_counter() - t0:.1f}s", flush=True)

        print("[setup] saving post-prime streaming state to /tmp...", flush=True)
        t0 = time.perf_counter()
        # Use moshi's save_streaming_state (safetensors + json) instead of
        # in-memory deepcopy. The dataclass _LMGenState contains CUDAGraph
        # objects that can't be pickled, AND deepcopying the data tensors
        # would orphan the graphs (which point at the originals). The on-
        # disk safetensors path is the only API moshi supports for state
        # snapshots, and `set_streaming_state_inplace` copies tensors INTO
        # the live buffers via `.copy_()`, leaving CUDA graphs intact.
        # `os` is imported at module level; importing it locally here would
        # make `os` function-scoped everywhere in setup() and shadow the
        # earlier `os.environ.get(...)` call.
        import shutil
        shutil.rmtree(SNAPSHOT_DIR, ignore_errors=True)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        for name, module in [
            ("lm_gen", self.lm_gen),
            ("mimi", self.mimi),
            ("other_mimi", self.other_mimi),
        ]:
            module.save_streaming_state(
                f"{SNAPSHOT_DIR}/{name}.safetensors",
                f"{SNAPSHOT_DIR}/{name}.json",
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"[setup]   snapshot saved ({time.perf_counter() - t0:.1f}s) "
                f"— GPU {(total - free) / 1e9:.1f}GB used / {total / 1e9:.1f}GB",
                flush=True,
            )
        else:
            print(f"[setup]   snapshot saved ({time.perf_counter() - t0:.1f}s)", flush=True)

        print(
            f"[setup] PersonaPlex ready — total {time.perf_counter() - t_start:.1f}s",
            flush=True,
        )

    # No instance attrs needed — snapshots live on disk in SNAPSHOT_DIR.

    @modal.method()
    async def process_call(
        self,
        call_id: str,
        livekit_room: str,
        livekit_agent_token: str,
        backend_control_url: str,
    ) -> dict:
        """Handle one inbound call. Long-running (call duration up to 1h).

        Step 4a implementation: control WS + LiveKit room + Sarah persona
        priming + per-frame PersonaPlex inference (no Reasoner-driven
        injection yet — that's Step 5; no VAD turn-boundary detection —
        Step 6). Caller audio flows through Mimi.encode → LMGen.step →
        Mimi.decode → LiveKit; agent text tokens are pushed to backend
        over the control WS.

        Args:
            call_id: WS path will be `{backend_control_url}/control/{call_id}`.
            livekit_room: LiveKit room name (logged; agent_token authorises join).
            livekit_agent_token: signed JWT for joining the room as agent.
            backend_control_url: backend base URL, e.g. wss://abc.ngrok.app.

        Returns:
            Summary dict — frame_count + ok flag.
        """
        import asyncio
        import json
        import logging
        import os
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import numpy as np
        import torch
        import websockets
        from livekit import rtc

        logger = logging.getLogger("personaplex.process_call")
        logger.info("call %s: process_call started", call_id)

        livekit_url = os.environ.get("LIVEKIT_URL")
        if not livekit_url:
            return {"ok": False, "error": "LIVEKIT_URL not set"}

        if self.lm_gen is None:
            return {"ok": False, "error": "PersonaPlex not loaded (setup() failed?)"}

        # Reclaim cached but unallocated GPU memory from any previous call
        # on this warm container. Combined with expandable_segments env var,
        # this stops cumulative fragmentation across calls from OOMing.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        ws_url = f"{backend_control_url.rstrip('/')}/control/{call_id}"
        t_call_started = time.perf_counter()
        frame_count = 0
        room: rtc.Room | None = None

        try:
            async with websockets.connect(ws_url, max_size=2**20) as ws:
                logger.info("call %s: control WS connected", call_id)

                # ─── 1. Receive SessionStart ──────────────────────────
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                session_start = json.loads(raw)
                if session_start.get("type") != "session_start":
                    return {
                        "ok": False,
                        "error": f"expected session_start, got {session_start.get('type')}",
                    }
                system_prompt = session_start["system_prompt"]
                voice_prompt_id = session_start["voice_prompt_id"]
                logger.info(
                    "call %s: SessionStart (voice=%s, prompt=%d chars)",
                    call_id, voice_prompt_id, len(system_prompt),
                )

                # ─── 2. Restore the pre-primed Sarah persona snapshot ──
                # setup() runs step_system_prompts ONCE per container start
                # (~33s) and snapshots the resulting state. We restore that
                # snapshot per call (~50ms). Backend's system_prompt field
                # is logged for visibility but not enforced — to change
                # Sarah's persona, edit BAKED_PERSONA + redeploy.
                if voice_prompt_id != DEFAULT_VOICE_ID:
                    logger.warning(
                        "call %s: backend asked for voice %s but container "
                        "is primed with %s — using primed voice",
                        call_id, voice_prompt_id, DEFAULT_VOICE_ID,
                    )
                t0 = time.perf_counter()
                # Flip to True (then redeploy) to A/B test the slow per-call
                # priming path. If Sarah sounds correct here but garbled with
                # the snapshot path, the snapshot is broken.
                if False:  # set True to force slow re-prime each call
                    # SLOW PATH (debug only) — re-prime per call. Use this
                    # to A/B test whether snapshot/restore is corrupting
                    # state. If Sarah sounds correct here but garbled with
                    # the snapshot path, the snapshot is broken.
                    voice_path = Path(self.voices_dir) / f"{DEFAULT_VOICE_ID}.pt"
                    self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
                    self.lm_gen.text_prompt_tokens = self.tokenizer.encode(
                        _wrap_with_system_tags(BAKED_PERSONA)
                    )
                    self.mimi.reset_streaming()
                    self.other_mimi.reset_streaming()
                    self.lm_gen.reset_streaming()
                    self.lm_gen.step_system_prompts(self.mimi)
                    self.mimi.reset_streaming()
                    logger.info(
                        "call %s: persona re-primed (slow path) in %.1fs",
                        call_id, time.perf_counter() - t0,
                    )
                else:
                    # Load the disk snapshots into flat dicts and call
                    # set_streaming_state_inplace, which copies tensors INTO
                    # the live module buffers via .copy_() — leaving the
                    # CUDA graphs (which point at those buffers) intact.
                    #
                    # Critical: reset_streaming() FIRST so the live module
                    # has the same None/Tensor pattern as the snapshot.
                    # Without this we hit `value.copy_(None.to(...))` errors
                    # for fields that moshi lazy-allocates between calls.
                    from moshi.modules.streaming import load_streaming_state
                    self.lm_gen.reset_streaming()
                    self.mimi.reset_streaming()
                    self.other_mimi.reset_streaming()
                    for name, module in [
                        ("lm_gen", self.lm_gen),
                        ("mimi", self.mimi),
                        ("other_mimi", self.other_mimi),
                    ]:
                        # Load to CPU then let set_streaming_state_inplace
                        # do the .to(value.device) per-tensor copy. Loading
                        # the whole flat dict directly to CUDA spikes
                        # allocation by ~1GB and OOMs at call start.
                        flat = load_streaming_state(
                            f"{SNAPSHOT_DIR}/{name}.safetensors",
                            f"{SNAPSHOT_DIR}/{name}.json",
                            device="cpu",
                        )
                        module.set_streaming_state_inplace(flat)
                        del flat
                    torch.cuda.empty_cache()
                    # Wipe mimi's encoder state so the priming-time audio
                    # frames don't bleed into the caller's first encode.
                    self.mimi.reset_streaming()
                    logger.info(
                        "call %s: persona restored from snapshot in %.3fs",
                        call_id, time.perf_counter() - t0,
                    )

                # ─── 3. Join LiveKit room ──────────────────────────────
                room = rtc.Room()
                caller_track: rtc.RemoteAudioTrack | None = None
                track_event = asyncio.Event()
                caller_left = asyncio.Event()

                @room.on("participant_disconnected")
                def _on_participant_disconnected(participant):
                    # SIP participants from Twilio show up with identity like
                    # `sip_+4917684071300`. When they leave (caller hangs up),
                    # tear down the room so the audio_stream loop ends and
                    # process_call can finish + send session_closed.
                    if participant.identity.startswith("sip_"):
                        logger.info(
                            "call %s: caller disconnected (identity=%s)",
                            call_id, participant.identity,
                        )
                        caller_left.set()
                        asyncio.create_task(room.disconnect())

                @room.on("track_subscribed")
                def _on_track(track, publication, participant):
                    nonlocal caller_track
                    if track.kind == rtc.TrackKind.KIND_AUDIO and caller_track is None:
                        caller_track = track
                        track_event.set()
                        logger.info(
                            "call %s: caller audio subscribed (id=%s)",
                            call_id, participant.identity,
                        )

                await room.connect(livekit_url, livekit_agent_token)
                logger.info("call %s: joined LiveKit room", call_id)

                audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
                agent_track = rtc.LocalAudioTrack.create_audio_track(
                    "sarah-agent-audio", audio_source
                )
                await room.local_participant.publish_track(agent_track)

                cold_start = time.perf_counter() - t_call_started
                await ws.send(json.dumps({
                    "type": "session_ready",
                    "call_id": call_id,
                    "voice_prompt_id": voice_prompt_id,
                    "cold_start_seconds": cold_start,
                }))
                logger.info("call %s: SessionReady (%.2fs)", call_id, cold_start)

                try:
                    await asyncio.wait_for(track_event.wait(), timeout=30)
                except asyncio.TimeoutError:
                    return {"ok": False, "error": "no caller audio track within 30s"}

                # ─── 3b. Open Scribe v2 streaming STT for caller transcript ─
                # PersonaPlex's text head emits Sarah's monologue, NOT a
                # transcript of the caller. Without an external ASR, the
                # backend slot extractor sees only Sarah's words and can't
                # capture FNOL data. Scribe streams realtime text per caller
                # utterance, sent over the control WS as `source: "scribe"`.
                scribe_stream = None
                scribe_consume_task = None

                # Loud env-var diagnostic. The Modal `clio-elevenlabs` secret
                # SHOULD set ELEVENLABS_API_KEY but if someone created it as
                # ELEVEN_API_KEY (matching LiveKit plugin docs) we fall back.
                scribe_key = (
                    os.environ.get("ELEVENLABS_API_KEY")
                    or os.environ.get("ELEVEN_API_KEY")
                )
                env_keys_present = sorted(
                    k for k in os.environ
                    if "ELEVEN" in k.upper() or "SCRIBE" in k.upper()
                )
                logger.info(
                    "call %s: scribe init — key_set=%s, env_keys=%s",
                    call_id, bool(scribe_key), env_keys_present,
                )

                # Owned by this call; closed in finally.
                scribe_http_session = None
                if scribe_key:
                    try:
                        import aiohttp
                        from livekit.agents import stt as lk_stt
                        from livekit.plugins import elevenlabs as eleven

                        # The plugin tries to grab a session from
                        # `livekit.agents.utils.http_context` which only
                        # exists inside an agent-worker run. We're calling
                        # the STT class directly, so we must own the session.
                        scribe_http_session = aiohttp.ClientSession()

                        scribe = eleven.STT(
                            api_key=scribe_key,
                            model_id="scribe_v2_realtime",
                            sample_rate=16000,
                            language_code="en",
                            http_session=scribe_http_session,
                        )
                        scribe_stream = scribe.stream()

                        async def _consume_scribe():
                            try:
                                async for ev in scribe_stream:
                                    if ev.type != lk_stt.SpeechEventType.FINAL_TRANSCRIPT:
                                        continue
                                    if not ev.alternatives:
                                        continue
                                    text = ev.alternatives[0].text or ""
                                    if not text.strip():
                                        continue
                                    try:
                                        await ws.send(json.dumps({
                                            "type": "transcript",
                                            "call_id": call_id,
                                            "role": "caller",
                                            "text": text,
                                            "source": "scribe",
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "is_final": True,
                                        }))
                                        logger.info(
                                            "call %s: scribe final: %r",
                                            call_id, text[:80],
                                        )
                                    except websockets.exceptions.ConnectionClosed:
                                        return
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                # ElevenLabs sometimes closes the upstream
                                # WS at end of call with status 1000 (normal
                                # closure). The plugin re-raises it as an
                                # APIStatusError. We don't care — the call
                                # is ending anyway.
                                msg = str(e).lower()
                                if "closed" in msg or "1000" in msg:
                                    logger.debug(
                                        "call %s: scribe consumer ended: %s",
                                        call_id, e,
                                    )
                                else:
                                    logger.exception(
                                        "call %s: scribe consumer crashed: %s",
                                        call_id, e,
                                    )

                        scribe_consume_task = asyncio.create_task(_consume_scribe())
                        logger.info("call %s: Scribe STT stream opened", call_id)
                    except Exception as e:
                        logger.exception(
                            "call %s: failed to start Scribe (continuing without): %s",
                            call_id, e,
                        )
                        scribe_stream = None
                else:
                    logger.warning(
                        "call %s: ELEVENLABS_API_KEY not set — no caller ASR",
                        call_id,
                    )

                # ─── 4-6. Per-frame inference loop + directives + VAD ──
                # Drip state — mutated by the directive receiver task,
                # consumed by the inference loop one frame at a time.
                drip = DripState()
                vad = TurnBoundaryDetector()
                directive_count = 0
                turn_boundary_count = 0

                async def receive_directives() -> None:
                    """Background task: parse incoming WS messages and
                    update drip state. Runs concurrent with the audio loop."""
                    nonlocal directive_count
                    try:
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            mtype = msg.get("type")
                            if mtype == "speak":
                                # Tokenize the text content (no <system> tags
                                # — those are init-time only). Drip-feed
                                # leads with EPAD inside DripState.force_speak.
                                content_tokens = self.tokenizer.encode(msg["text"])
                                drip.force_speak(
                                    token_ids=content_tokens,
                                    after_release=msg.get("after_release", "resume"),
                                )
                                directive_count += 1
                                logger.info(
                                    "call %s: speak directive — %d tokens (reason=%s)",
                                    call_id, len(content_tokens),
                                    msg.get("reason", ""),
                                )
                            elif mtype == "silent":
                                drip.force_silence(msg["duration_frames"])
                                directive_count += 1
                                logger.info(
                                    "call %s: silent directive — %d frames",
                                    call_id, msg["duration_frames"],
                                )
                            elif mtype == "release":
                                drip.force_release()
                                directive_count += 1
                                logger.info("call %s: release directive", call_id)
                            elif mtype == "load_policy_context":
                                # Modal doesn't act on this directly — backend
                                # uses subsequent SpeakDirectives to surface
                                # the brief into Sarah's monologue stream.
                                logger.info(
                                    "call %s: policy_context loaded (%d chars)",
                                    call_id, len(msg.get("policy_brief", "")),
                                )
                            elif mtype == "rescue_clip":
                                # TODO: server-side audio mute + WAV stream.
                                logger.warning(
                                    "call %s: rescue_clip not yet implemented (clip=%s)",
                                    call_id, msg.get("clip_id"),
                                )
                            elif mtype == "session_end":
                                logger.info("call %s: backend requested end", call_id)
                                return
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("call %s: directive WS closed", call_id)

                directive_task = asyncio.create_task(receive_directives())

                logger.info("call %s: starting PersonaPlex inference loop", call_id)
                audio_stream = rtc.AudioStream(
                    caller_track, sample_rate=24000, num_channels=1
                )
                buffer = np.empty(0, dtype=np.float32)

                # Agent transcript batching. PersonaPlex's text head emits
                # one token per 80ms frame. We accumulate and flush at
                # natural boundaries (sentence end > comma > word boundary
                # if buffer is long enough) to keep the UI/log readable.
                # Hard timeout fallback prevents indefinite hold on a
                # super-long sentence.
                AGENT_TEXT_HARD_TIMEOUT = 2.5  # seconds
                AGENT_TEXT_PHRASE_MIN = 30      # min chars to flush on a comma
                AGENT_TEXT_WORD_MIN = 60        # min chars to flush on word
                AGENT_TEXT_MAX_BUFFER = 120     # absolute hard flush
                SKIP_TEXT_TOKENS = {
                    PAD_TOKEN_ID, EPAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID,
                }
                agent_text_buf = ""
                agent_last_flush = time.perf_counter()

                async for event in audio_stream:
                    if caller_left.is_set():
                        # Defensive — in case audio_stream keeps yielding
                        # after the SIP participant left (it shouldn't, but
                        # belt-and-suspenders so we don't sit on a dead call).
                        logger.info("call %s: ending loop (caller left)", call_id)
                        break

                    # Fan out the caller frame to Scribe ASR in parallel
                    # with PersonaPlex's encode/decode. Scribe does its
                    # own buffering and emits FINAL_TRANSCRIPT events when
                    # the caller pauses.
                    if scribe_stream is not None:
                        try:
                            scribe_stream.push_frame(event.frame)
                        except Exception as e:
                            # Don't let an STT hiccup take down the call,
                            # but log the FIRST occurrence so we know the
                            # plugin contract is wrong (silent push_frame
                            # failures are how Scribe stops working without
                            # any clue in logs).
                            if scribe_stream is not None:
                                logger.warning(
                                    "call %s: scribe push_frame failed (%s); disabling",
                                    call_id, e,
                                )
                                scribe_stream = None

                    incoming = _audioframe_to_float32(event.frame)
                    buffer = np.concatenate([buffer, incoming])

                    # Drain whole 80ms chunks; tail of <1920 samples waits.
                    while buffer.shape[0] >= self.frame_size:
                        chunk = buffer[: self.frame_size]
                        buffer = buffer[self.frame_size :]

                        # VAD: detect caller turn boundary before encoding.
                        # If True, the caller just finished speaking — push
                        # event to backend so the slot extractor runs, AND
                        # nudge Sarah out of any PAD lock with EPAD so she
                        # actually responds (without this she sometimes
                        # stays silent until the caller speaks again).
                        if vad.update(chunk):
                            turn_boundary_count += 1
                            try:
                                await ws.send(json.dumps({
                                    "type": "caller_turn_boundary",
                                    "call_id": call_id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }))
                                logger.info(
                                    "call %s: caller turn boundary #%d",
                                    call_id, turn_boundary_count,
                                )
                            except websockets.exceptions.ConnectionClosed:
                                pass
                            # Wake-up nudge — only if no backend directive
                            # is already mid-flight, so we don't clobber a
                            # planned interruption from the gate.
                            if (
                                not drip.queue
                                and drip.silent_frames_remaining == 0
                            ):
                                drip.queue.append(EPAD_TOKEN_ID)

                        # Mimi.encode wants [B, channels, T] float32 on cuda
                        chunk_t = torch.from_numpy(chunk).to(
                            device="cuda", dtype=torch.float32
                        ).reshape(1, 1, -1)
                        user_codes = self.mimi.encode(chunk_t)

                        # LMGen.step — feed forced text_token from drip state
                        for c in range(user_codes.shape[-1]):
                            forced_id = drip.next_forced_token()
                            # Talkover suppression: if no directive is active
                            # AND VAD says the caller is currently speaking,
                            # force PAD so Sarah waits for them to finish.
                            # The model's natural duplex behavior is unreliable
                            # on phone-band audio (different from training
                            # distribution).
                            if forced_id is None and vad.is_speaking:
                                forced_id = PAD_TOKEN_ID
                            forced_tensor = (
                                torch.tensor([forced_id], device="cuda", dtype=torch.long)
                                if forced_id is not None
                                else None
                            )
                            tokens = self.lm_gen.step(
                                input_tokens=user_codes[:, :, c : c + 1],
                                text_token=forced_tensor,
                            )
                            if tokens is None:
                                continue

                            # Decode agent audio. Critical: use other_mimi
                            # (the decoder twin), NOT self.mimi (which is
                            # the encoder side processing caller audio).
                            # Sharing one Mimi for both encode + decode
                            # causes the streaming convolution's prev_x
                            # buffer to grow unbounded → OOM at ~3min.
                            agent_pcm_t = self.other_mimi.decode(tokens[:, 1:9])
                            agent_pcm_np = agent_pcm_t.detach().cpu().numpy()[0, 0]

                            # Push to LiveKit as int16 PCM
                            agent_int16 = _float32_to_int16(agent_pcm_np)
                            out_frame = rtc.AudioFrame(
                                data=agent_int16.tobytes(),
                                sample_rate=24000,
                                num_channels=1,
                                samples_per_channel=len(agent_int16),
                            )
                            await audio_source.capture_frame(out_frame)

                            # Accumulate agent text tokens into a buffer;
                            # flush at natural boundaries so the UI/log
                            # shows complete phrases instead of mid-word
                            # fragments. Skip control tokens entirely.
                            text_token_id = int(tokens[0, 0, 0].item())
                            if text_token_id not in SKIP_TEXT_TOKENS:
                                piece = _decode_text_token(
                                    self.tokenizer, text_token_id
                                )
                                if piece:
                                    agent_text_buf += piece

                            flush_text = _maybe_flush_agent_buf(
                                agent_text_buf,
                                age=time.perf_counter() - agent_last_flush,
                            )
                            if flush_text is not None:
                                # Whatever wasn't flushed (partial trailing
                                # word) stays for the next iteration.
                                agent_text_buf = agent_text_buf[len(flush_text):]
                                agent_last_flush = time.perf_counter()
                                cleaned = flush_text.strip()
                                if cleaned:
                                    try:
                                        await ws.send(json.dumps({
                                            "type": "transcript",
                                            "call_id": call_id,
                                            "role": "agent",
                                            "text": cleaned,
                                            "source": "personaplex",
                                            "timestamp": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                        }))
                                    except websockets.exceptions.ConnectionClosed:
                                        logger.warning(
                                            "call %s: backend WS closed mid-call",
                                            call_id,
                                        )
                                        break

                            frame_count += 1
                            if frame_count == 1:
                                logger.info(
                                    "call %s: first PersonaPlex frame produced",
                                    call_id,
                                )

                # Cancel the directive receiver before tearing down
                directive_task.cancel()
                try:
                    await directive_task
                except asyncio.CancelledError:
                    pass

                # Close Scribe STT stack. Order matters:
                #   1. Cancel the consumer task (stops iterating stream)
                #   2. Close the stream (signal end-of-input)
                #   3. Close the http session (release WS to ElevenLabs)
                # Reversed order produces "RuntimeError: aclose(): generator
                # is already running" + "Unclosed client session" warnings.
                if scribe_consume_task is not None:
                    scribe_consume_task.cancel()
                    try:
                        await scribe_consume_task
                    except (asyncio.CancelledError, Exception):
                        pass
                if scribe_stream is not None:
                    try:
                        # end_input lets the plugin flush any pending audio
                        # before we close, avoiding the "connection closed
                        # unexpectedly" log spam.
                        if hasattr(scribe_stream, "end_input"):
                            scribe_stream.end_input()
                        await scribe_stream.aclose()
                    except Exception:
                        pass
                if scribe_http_session is not None:
                    try:
                        await scribe_http_session.close()
                    except Exception:
                        pass

                logger.info(
                    "call %s: inference loop ended after %d frames (%.1fs)",
                    call_id, frame_count, time.perf_counter() - t_call_started,
                )

                await ws.send(json.dumps({
                    "type": "session_closed",
                    "call_id": call_id,
                    "reason": "audio stream ended",
                    "duration_seconds": time.perf_counter() - t_call_started,
                }))

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("call %s: control WS closed: %s", call_id, e)
        except Exception as e:
            logger.exception("call %s: process_call crashed: %s", call_id, e)
            return {"ok": False, "error": str(e), "frame_count": frame_count}
        finally:
            if room is not None:
                try:
                    await room.disconnect()
                except Exception:
                    pass
            # End-of-call memory reclaim. reset_streaming + empty_cache
            # didn't actually free anything because CUDAGraphed objects in
            # _LMGenState pin their captured tensors' memory addresses.
            # The only path that reliably releases is to drop the entire
            # lm_gen + mimi instances and rebuild fresh wrappers around
            # the still-loaded model weights. The next call's start-of-
            # process_call snapshot restore puts the new instances back
            # into post-priming state.
            try:
                elapsed = await asyncio.to_thread(self._recreate_inference_stack)
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    logger.info(
                        "call %s: post-call rebuild in %.1fs — GPU %.1fGB used / %.1fGB",
                        call_id, elapsed,
                        (total - free) / 1e9, total / 1e9,
                    )
            except Exception as e:
                logger.warning("call %s: post-call rebuild failed: %s",
                               call_id, e)

        return {
            "ok": True,
            "call_id": call_id,
            "frame_count": frame_count,
            "duration_seconds": time.perf_counter() - t_call_started,
        }

    @modal.method()
    async def health(self) -> dict:
        """Lightweight health check. Backend can hit this every minute to
        verify the warm container is responsive."""
        return {
            "ok": True,
            "model_loaded": self.lm_gen is not None,
            "demo_mode": DEMO_MODE,
        }

    @modal.method()
    async def dump_state_shape(self) -> dict:
        """Compare what get_streaming_state() returns vs the dataclass fields
        set_streaming_state_inplace expects. The mismatch tells us which
        fields we need to inject as None into our snapshot."""
        from dataclasses import fields

        def _walk_state(obj, path=""):
            """Recursively walk a streaming state object and its submodules."""
            out = {"path": path or "<root>", "type": type(obj).__name__}
            if hasattr(obj, "_streaming_state") and obj._streaming_state is not None:
                ss = obj._streaming_state
                expected = [f.name for f in fields(ss)] if hasattr(ss, "__dataclass_fields__") else []
                live = {}
                for f_name in expected:
                    try:
                        v = getattr(ss, f_name)
                        live[f_name] = type(v).__name__ + (
                            f"{tuple(v.shape)}" if hasattr(v, "shape") else ""
                        )
                    except Exception as e:
                        live[f_name] = f"<error: {e}>"
                got = obj.get_streaming_state()
                out["state_class"] = type(ss).__name__
                out["expected_fields"] = expected
                out["actual_keys"] = list(got.keys())
                out["live_values"] = live
                out["missing_from_snapshot"] = [
                    f for f in expected if f not in got
                ]
            return out

        # Run priming so state is in the same shape we'd snapshot
        from pathlib import Path

        voice_path = Path(self.voices_dir) / f"{DEFAULT_VOICE_ID}.pt"
        wrapped = _wrap_with_system_tags(BAKED_PERSONA)
        self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrapped)
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)

        return {
            "lm_gen": _walk_state(self.lm_gen, "lm_gen"),
            "mimi": _walk_state(self.mimi, "mimi"),
            "other_mimi": _walk_state(self.other_mimi, "other_mimi"),
        }

    @modal.method()
    async def introspect_streaming_state(self) -> dict:
        """Inspect the moshi/PersonaPlex objects to find their streaming-
        state API. We need this to snapshot post-priming state in setup()
        and restore it per-call (replacing the 33s step_system_prompts
        priming with a sub-second restore).

        Returns a structured report listing relevant attributes/methods
        on lm_gen, mimi, and the underlying lm.transformer."""
        import inspect

        def _kind(obj) -> str:
            t = type(obj).__name__
            if hasattr(obj, "shape"):
                return f"{t}{tuple(obj.shape)}"
            if hasattr(obj, "__len__"):
                try:
                    return f"{t}(len={len(obj)})"
                except Exception:
                    pass
            return t

        def _scan(obj, label: str, extra_keywords=()) -> dict:
            keywords = (
                "stream", "state", "cache", "kv",
                "prompt", "reset", "clone", "copy", "snapshot",
                *extra_keywords,
            )
            attrs = []
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                if not any(k in name.lower() for k in keywords):
                    continue
                try:
                    val = getattr(obj, name)
                except Exception as e:
                    attrs.append({"name": name, "error": str(e)[:80]})
                    continue
                entry: dict = {"name": name}
                if callable(val):
                    try:
                        entry["signature"] = str(inspect.signature(val))
                    except Exception:
                        entry["signature"] = "?"
                    entry["kind"] = "method"
                else:
                    entry["kind"] = _kind(val)
                attrs.append(entry)
            return {
                "label": label,
                "type": type(obj).__name__,
                "module": type(obj).__module__,
                "attrs": attrs,
            }

        # The transformer (deep inside lm) is where KV cache lives in moshi.
        # Try to find it via common paths.
        lm = getattr(self.lm_gen, "lm_model", None) or getattr(
            self.lm_gen, "lm", None
        )
        transformer = None
        if lm is not None:
            transformer = (
                getattr(lm, "transformer", None)
                or getattr(lm, "depformer", None)
                or getattr(lm, "_transformer", None)
            )

        report = {
            "lm_gen": _scan(self.lm_gen, "lm_gen"),
            "mimi":   _scan(self.mimi, "mimi"),
            "lm":     _scan(lm, "lm") if lm is not None else None,
            "transformer": (
                _scan(transformer, "transformer", extra_keywords=("offset",))
                if transformer is not None else None
            ),
        }
        return report


# ─── Per-frame helpers ───────────────────────────────────────────────────────
# These run inside the Modal container, so they can assume torch/numpy are
# available. Kept at module level so process_call() stays readable.


def _clone_streaming_state(state):
    """Deep-copy a moshi streaming-state structure.

    `get_streaming_state()` returns `{module_name: _StateClassInstance}`
    where each value is a dataclass with nested state (tensors, custom
    cache classes, etc). `set_streaming_state()` ASSIGNS the dataclass
    to `module._streaming_state`, so later inference mutates our snapshot
    unless we deep-clone before each restore.

    We use `copy.deepcopy` rather than a hand-rolled walker so we don't
    silently share-by-reference any custom classes that aren't dataclasses
    (which is what the earlier hand-rolled helper did, producing garbled
    output after the first call).
    """
    import copy

    return copy.deepcopy(state)


def _wrap_with_system_tags(text: str) -> str:
    """PersonaPlex expects system prompt wrapped in <system>...<system>.
    Mirrors personaplex/offline.py:wrap_with_system_tags."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _audioframe_to_float32(frame) -> "numpy.ndarray":
    """LiveKit AudioFrame (int16) → float32 numpy in [-1, 1]."""
    import numpy as np

    int16 = np.frombuffer(frame.data, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


def _float32_to_int16(arr) -> "numpy.ndarray":
    """float32 PCM in [-1, 1] → int16 numpy. Clamps before quantizing."""
    import numpy as np

    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _maybe_flush_agent_buf(buf: str, age: float) -> str | None:
    """Decide what (if anything) of the agent transcript buffer to flush.

    Returns the prefix to flush (caller pops it from buf), or None to wait.

    Flush priority:
      1. Sentence end (`.`, `?`, `!`)            — flush UP TO and including
      2. Comma + buffer >= AGENT_TEXT_PHRASE_MIN — flush up to and including
      3. Word boundary (space) + buf >= AGENT_TEXT_WORD_MIN — flush up to space
      4. Hard buffer cap (AGENT_TEXT_MAX_BUFFER) — flush whatever's there
      5. Hard age cap (AGENT_TEXT_HARD_TIMEOUT)  — flush whatever's there
    """
    if not buf:
        return None
    AGENT_TEXT_HARD_TIMEOUT = 2.5
    AGENT_TEXT_PHRASE_MIN = 30
    AGENT_TEXT_WORD_MIN = 60
    AGENT_TEXT_MAX_BUFFER = 120

    # 1. Sentence-final punctuation
    for p in (".", "?", "!"):
        i = buf.rfind(p)
        if i >= 0:
            return buf[: i + 1]
    # 2. Comma at phrase length
    if len(buf) >= AGENT_TEXT_PHRASE_MIN:
        i = buf.rfind(",")
        if i >= 0:
            return buf[: i + 1]
    # 3. Word boundary at long-buffer length
    if len(buf) >= AGENT_TEXT_WORD_MIN:
        i = buf.rfind(" ")
        if i > 0:
            return buf[:i]
    # 4. Hard cap
    if len(buf) >= AGENT_TEXT_MAX_BUFFER:
        return buf
    # 5. Timeout
    if age >= AGENT_TEXT_HARD_TIMEOUT:
        return buf
    return None


def _decode_text_token(tokenizer, token_id: int) -> str:
    """Map a text token id to its display string. Special-cases the four
    control tokens (EPAD/BOS/EOS/PAD) per personaplex/offline.py:293."""
    SPECIAL = {0: "EPAD", 1: "BOS", 2: "EOS", 3: "PAD"}
    if token_id in SPECIAL:
        return SPECIAL[token_id]
    piece = tokenizer.id_to_piece(token_id)
    return piece.replace("▁", " ")  # SentencePiece word-boundary marker


# Special token ids (mirror personaplex/offline.py:293)
EPAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 3


# ─── Caller turn boundary detection ─────────────────────────────────────────
# Energy-based VAD with hysteresis. Fires when the caller has been silent
# for `SILENCE_FRAMES_FOR_BOUNDARY` consecutive frames AFTER having spoken.
# Backend's orchestrator uses this to trigger the slot extractor.
#
# Tuning: 12.5 Hz frame rate × 10 frames = 800ms of silence before we
# consider a turn complete. Real claims callers pause-think for 1-3s, so
# 800ms is a tight but reasonable boundary. Bump SILENCE_FRAMES_FOR_BOUNDARY
# if extractor fires too eagerly mid-thought.
RMS_SILENCE_THRESHOLD = 0.005       # below this is "silent"
SILENCE_FRAMES_FOR_BOUNDARY = 10    # 10 × 80ms = 800ms
SPEECH_FRAMES_FOR_TURN_START = 2    # debounce — 160ms above threshold to count


class TurnBoundaryDetector:
    """Tracks caller speech state and fires a turn-boundary callback when
    the caller transitions from speaking → silent for the threshold duration."""

    def __init__(self) -> None:
        self.consecutive_silent = 0
        self.consecutive_speech = 0
        self.is_speaking = False
        self.turn_in_progress = False  # caller said anything since last boundary?

    def update(self, chunk_float32) -> bool:
        """Feed one 80ms chunk of caller PCM. Returns True if a turn boundary
        just fired (caller stopped speaking)."""
        import numpy as np

        rms = float(np.sqrt(np.mean(chunk_float32 ** 2)))

        if rms > RMS_SILENCE_THRESHOLD:
            # Caller is producing audio above noise floor
            self.consecutive_silent = 0
            self.consecutive_speech += 1
            if self.consecutive_speech >= SPEECH_FRAMES_FOR_TURN_START:
                self.is_speaking = True
                self.turn_in_progress = True
            return False

        # Below threshold = silence
        self.consecutive_speech = 0
        self.consecutive_silent += 1

        if (
            self.is_speaking
            and self.turn_in_progress
            and self.consecutive_silent >= SILENCE_FRAMES_FOR_BOUNDARY
        ):
            self.is_speaking = False
            self.turn_in_progress = False
            return True
        return False


class DripState:
    """Per-call queue of forced text tokens for the per-frame loop to consume.

    The control-plane WS receiver mutates this state as ReasonerDirectives
    arrive. The per-frame loop calls next_forced_token() each frame and
    passes the result (or None for free sampling) to LMGen.step.

    Concurrent access: backend directives arrive on a separate asyncio task
    from the inference loop, but Python's GIL plus the simple deque ops
    make this safe without explicit locking. If we ever multi-thread, wrap
    with asyncio.Lock.
    """

    def __init__(self) -> None:
        from collections import deque

        self.queue: "deque[int]" = deque()
        self.after_release: str = "resume"
        self.silent_frames_remaining: int = 0

    def force_speak(self, token_ids: list[int], after_release: str = "resume") -> None:
        """SpeakDirective handler. Always lead with EPAD to break out of
        any active PAD state."""
        self.queue.clear()
        self.queue.append(EPAD_TOKEN_ID)
        self.queue.extend(token_ids)
        self.after_release = after_release
        self.silent_frames_remaining = 0

    def force_silence(self, frames: int) -> None:
        """SilenceDirective handler. Holds PAD for N frames."""
        self.queue.clear()
        self.silent_frames_remaining = max(0, int(frames))

    def force_release(self) -> None:
        """ReleaseDirective handler. Cancel any active override."""
        self.queue.clear()
        self.silent_frames_remaining = 0
        self.after_release = "resume"

    def next_forced_token(self) -> int | None:
        """One frame ticks here. Returns the token to force, or None to
        let LMGen sample freely."""
        if self.silent_frames_remaining > 0:
            self.silent_frames_remaining -= 1
            return PAD_TOKEN_ID
        if self.queue:
            return self.queue.popleft()
        if self.after_release == "silent":
            return PAD_TOKEN_ID
        return None


# ─── CLI helpers ─────────────────────────────────────────────────────────────
# Run locally with `modal run model_service/deploy/modal_app.py::warmup_test`
# to verify the container boots cleanly before pointing real traffic at it.

@app.local_entrypoint()
def warmup_test() -> None:
    """Smoke test: instantiate the service, call health()."""
    svc = PersonaPlexService()
    result = svc.health.remote()
    print(f"health: {result}")


@app.local_entrypoint()
def introspect_streaming_state() -> None:
    """Print the streaming-state API surface of lm_gen / mimi / transformer.

    Use the output to figure out how to snapshot post-priming state in
    setup() and restore it per-call.

    Usage:
        modal run model_service/deploy/modal_app.py::introspect_streaming_state
    """
    import json

    svc = PersonaPlexService()
    report = svc.introspect_streaming_state.remote()
    print(json.dumps(report, indent=2, default=str))


@app.local_entrypoint()
def cold_start_bench() -> None:
    """Measure cold-start time end-to-end. Forces a fresh container by
    stopping any running ones first via `modal app stop` separately."""
    import time

    svc = PersonaPlexService()
    t0 = time.perf_counter()
    result = svc.health.remote()
    elapsed = time.perf_counter() - t0
    print(f"cold-start health: {result}")
    print(f"end-to-end time: {elapsed:.1f}s "
          f"(includes Modal control-plane RTT, container boot, @enter, RPC)")
