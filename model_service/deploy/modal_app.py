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

import os

import modal

# ─── Toggles ─────────────────────────────────────────────────────────────────

DEMO_MODE = os.environ.get("CLIO_DEMO_MODE") == "1"

APP_NAME = "personaplex-clio"

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

        # 80 ms = 1920 samples at 24kHz.
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)

        print("[setup] constructing LMGen ...", flush=True)
        t0 = time.perf_counter()
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),  # 0.5s spacer
            sample_rate=self.mimi.sample_rate,
            device=device,
            frame_rate=self.mimi.frame_rate,
            # Match the offline.py CLI defaults — these are reasonable for FNOL voice
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        # Critical: tells the model + codec to maintain streaming KV cache /
        # state across step() calls. Without this each step starts cold.
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        print(f"[setup]   LMGen ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        print("[setup] warmup pass (compiles CUDA graphs)...", flush=True)
        t0 = time.perf_counter()
        for _ in range(4):
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
        print(f"[setup]   warmup done ({time.perf_counter() - t0:.1f}s)", flush=True)

        print(
            f"[setup] PersonaPlex ready — total {time.perf_counter() - t_start:.1f}s",
            flush=True,
        )

    @modal.method()
    async def join_call(
        self,
        livekit_room: str,
        livekit_token: str,
        system_prompt: str,
        voice_prompt_id: str = "NATF1",
        backend_control_url: str = "",
    ) -> dict:
        """Handle one inbound call.

        Backend invokes this when Twilio dispatches a call into a LiveKit
        room. The container joins that room as the agent participant and
        runs Sarah for the call's duration.

        Args:
            livekit_room: LiveKit room name (one per call).
            livekit_token: LiveKit access token for the agent participant.
            system_prompt: BASE_PERSONA + time block from
                backend.app.reasoner.persona.session_system_prompt(now=...).
            voice_prompt_id: Which Sarah voice to use ("NATF1"/"NATF2"/etc.).
            backend_control_url: WebSocket URL where backend's Reasoner is
                listening for ReasonerDirectives + transcript events.

        Returns:
            Summary dict with call_id, duration, frame counts. Backend
            persists this for post-call analysis.
        """
        # TODO(clio): implement the actual agent loop:
        #   1. Connect to LiveKit room (livekit-agents SDK).
        #   2. Open backend control WS for directives + transcript push.
        #   3. Subscribe to caller audio track; for each 80ms frame:
        #        a. Read latest forced_text_token from the directive cache.
        #        b. tokens = self.lm_gen.step(input_tokens=..., text_token=...)
        #        c. Decode agent audio with Mimi; publish to LiveKit room.
        #        d. Push agent text token to backend over control WS.
        #   4. Run ElevenLabs Scribe v2 stream in parallel on caller audio
        #      → push transcript turns to backend (entity verification).
        #   5. Loop until LiveKit room disconnect.
        return {
            "ok": False,
            "error": "join_call not yet implemented — see TODO in modal_app.py",
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
