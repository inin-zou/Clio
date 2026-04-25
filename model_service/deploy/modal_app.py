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
    async def process_call(
        self,
        call_id: str,
        livekit_room: str,
        livekit_agent_token: str,
        backend_control_url: str,
    ) -> dict:
        """Handle one inbound call. Long-running (call duration up to 1h).

        Step 1-3 implementation: control-plane WS handshake + LiveKit room
        join + audio echo (caller audio passed straight back to validate
        the wiring). PersonaPlex inference and Reasoner-driven injection
        come in subsequent steps.

        Args:
            call_id: matches the path the WS will be served at on backend
                (`{backend_control_url}/control/{call_id}`).
            livekit_room: LiveKit room name (one per call). Logged but not
                used directly; the agent_token is what authorizes the join.
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

        import websockets
        from livekit import rtc

        logger = logging.getLogger("personaplex.process_call")
        logger.info("call %s: process_call started", call_id)

        livekit_url = os.environ.get("LIVEKIT_URL")
        if not livekit_url:
            logger.error("LIVEKIT_URL not set in Modal secrets")
            return {"ok": False, "error": "LIVEKIT_URL not set"}

        ws_url = f"{backend_control_url.rstrip('/')}/control/{call_id}"
        t_call_started = time.perf_counter()
        frame_count = 0
        room: rtc.Room | None = None

        try:
            async with websockets.connect(ws_url, max_size=2**20) as ws:
                logger.info("call %s: control WS connected to %s", call_id, ws_url)

                # ─── Step 1: receive SessionStart from backend ─────────
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    return {"ok": False, "error": "timeout waiting for SessionStart"}

                session_start = json.loads(raw)
                if session_start.get("type") != "session_start":
                    return {
                        "ok": False,
                        "error": f"expected session_start, got {session_start.get('type')}",
                    }
                system_prompt = session_start["system_prompt"]
                voice_prompt_id = session_start["voice_prompt_id"]
                logger.info(
                    "call %s: SessionStart received (voice=%s, prompt %d chars)",
                    call_id, voice_prompt_id, len(system_prompt),
                )

                # ─── Step 2: join LiveKit room ─────────────────────────
                room = rtc.Room()
                caller_audio_track: rtc.RemoteAudioTrack | None = None
                track_subscribed = asyncio.Event()

                @room.on("track_subscribed")
                def _on_track(
                    track: rtc.Track,
                    publication: rtc.RemoteTrackPublication,
                    participant: rtc.RemoteParticipant,
                ) -> None:
                    nonlocal caller_audio_track
                    if track.kind == rtc.TrackKind.KIND_AUDIO and caller_audio_track is None:
                        caller_audio_track = track
                        track_subscribed.set()
                        logger.info(
                            "call %s: subscribed to caller audio (participant=%s)",
                            call_id, participant.identity,
                        )

                @room.on("disconnected")
                def _on_disconnect() -> None:
                    logger.info("call %s: room disconnected", call_id)

                await room.connect(livekit_url, livekit_agent_token)
                logger.info("call %s: joined LiveKit room %s as agent", call_id, livekit_room)

                # ─── Step 3: publish our agent audio track ─────────────
                # 24kHz mono — matches Mimi codec output. PersonaPlex frames
                # written to this source will reach the caller via WebRTC.
                audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
                agent_track = rtc.LocalAudioTrack.create_audio_track(
                    "sarah-agent-audio", audio_source
                )
                await room.local_participant.publish_track(agent_track)
                logger.info("call %s: published agent audio track", call_id)

                # Backend can now safely start sending directives.
                cold_start = time.perf_counter() - t_call_started
                await ws.send(json.dumps({
                    "type": "session_ready",
                    "call_id": call_id,
                    "voice_prompt_id": voice_prompt_id,
                    "cold_start_seconds": cold_start,
                }))
                logger.info("call %s: SessionReady sent (%.2fs)", call_id, cold_start)

                # Wait for caller audio (Twilio SIP trunk publishes it after
                # the call connects). 30s grace.
                try:
                    await asyncio.wait_for(track_subscribed.wait(), timeout=30)
                except asyncio.TimeoutError:
                    logger.error("call %s: timed out waiting for caller audio track", call_id)
                    return {"ok": False, "error": "no caller audio track within 30s"}

                # ─── Step 1-3 validation: echo loop ────────────────────
                # For now we just pass caller audio back to the room. Once
                # this works end-to-end, Step 4a swaps the echo for
                # Mimi.encode → lm_gen.step → Mimi.decode.
                logger.info("call %s: starting echo loop (Step 1-3 validation)", call_id)
                audio_stream = rtc.AudioStream(
                    caller_audio_track, sample_rate=24000, num_channels=1
                )

                async for event in audio_stream:
                    await audio_source.capture_frame(event.frame)
                    frame_count += 1
                    if frame_count == 1:
                        logger.info(
                            "call %s: first audio frame received "
                            "(samples=%d, rate=%dHz)",
                            call_id, event.frame.samples_per_channel,
                            event.frame.sample_rate,
                        )
                    if frame_count % 250 == 0:  # ~20s at 12.5Hz
                        elapsed = time.perf_counter() - t_call_started
                        logger.debug(
                            "call %s: echoed %d frames (%.1fs elapsed)",
                            call_id, frame_count, elapsed,
                        )

                logger.info("call %s: audio stream ended after %d frames",
                            call_id, frame_count)

                # ─── Cleanup ──────────────────────────────────────────
                await ws.send(json.dumps({
                    "type": "session_closed",
                    "call_id": call_id,
                    "reason": "audio stream ended",
                    "duration_seconds": time.perf_counter() - t_call_started,
                }))

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning("call %s: control WS closed unexpectedly: %s", call_id, e)
        except Exception as e:
            logger.exception("call %s: process_call crashed: %s", call_id, e)
            return {
                "ok": False,
                "error": str(e),
                "frame_count": frame_count,
            }
        finally:
            if room is not None:
                try:
                    await room.disconnect()
                except Exception:
                    pass

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
