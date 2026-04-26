"""Modal CPU deployment for the Clio backend FastAPI app.

Hosts every endpoint that's *not* on the audio path:
  - POST /twilio/voice     ← Twilio webhook on inbound call
  - POST /twilio/status    ← Twilio call lifecycle callback
  - WS   /control/{call_id} ← Modal GPU container connects here
  - GET  /health, /calls   ← monitoring

Why Modal CPU instead of ngrok-from-laptop:
  - Stable URL (no ngrok URL rotation)
  - Survives laptop sleep / mobile demos
  - Same workspace as personaplex-clio: secrets + auth shared
  - Cheap: ~$0.10/day always-warm vs ~$26/day for the A100

Usage
-----
First deploy (URL is unknown until first deploy completes):

    modal deploy model_service/deploy/modal_backend.py

Modal prints the deployed URL, e.g.
    https://dreamonzouk--clio-backend-fastapi-app.modal.run

Save that URL into the clio-backend-cfg secret (so the backend tells the
GPU container where to WS-back to itself):

    modal secret create clio-backend-cfg \\
        BACKEND_PUBLIC_WS_URL=wss://dreamonzouk--clio-backend-fastapi-app.modal.run \\
        --force

Then redeploy so the new secret is mounted:

    modal deploy model_service/deploy/modal_backend.py

After redeploy, point Twilio Console at:
    https://<URL>/twilio/voice    (Voice webhook)
    https://<URL>/twilio/status   (Status callback)

Stop billing when not demoing:

    modal app stop clio-backend
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "clio-backend"

# Repo root — only resolvable when this file is imported locally during
# `modal deploy`. Inside the container the file is flattened to
# /root/modal_backend.py, but the image is already built so we don't need
# it then. None at runtime is the intended state.
_resolved = Path(__file__).resolve()
REPO_ROOT = _resolved.parents[2] if len(_resolved.parents) >= 3 else None


# ─── Image ───────────────────────────────────────────────────────────────────
# Install deps via pyproject.toml so the deployed runtime mirrors local.
# Skips the GPU-only model_service deps entirely.

if REPO_ROOT is not None:
    # Local context (during `modal deploy`): build the real image with
    # local source + data mounted in.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install_from_pyproject(str(REPO_ROOT / "pyproject.toml"))
        .env({
            "PYTHONUNBUFFERED": "1",
            "CLIO_SESSION_OUTPUT_DIR": "/root/data/sessions",
        })
        # add_local_* MUST come last in the image build — these get layered
        # on at container start (not bake time) so local-file changes don't
        # trigger an image rebuild.
        .add_local_python_source("backend")
        .add_local_dir(str(REPO_ROOT / "data"), "/root/data")
    )
else:
    # Container context: the image is already built and Modal won't
    # rebuild it from this stub. We just need a value so the @app.function
    # decorator below doesn't NameError.
    image = modal.Image.debian_slim(python_version="3.12")


app = modal.App(APP_NAME, image=image)


# Volume for end-of-call session JSON dumps. Persists across container
# restarts so we can grep through past calls during debugging.
sessions_vol = modal.Volume.from_name("clio-sessions", create_if_missing=True)


@app.function(
    secrets=[
        # Reused from personaplex-clio. Must contain LIVEKIT_API_KEY,
        # LIVEKIT_API_SECRET, LIVEKIT_URL, plus the new LIVEKIT_SIP_URI.
        modal.Secret.from_name("clio-livekit"),
        # Reused. ANTHROPIC_API_KEY for the slot extractor.
        modal.Secret.from_name("clio-anthropic"),
        # NEW. TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER.
        modal.Secret.from_name("clio-twilio"),
        # NEW. BACKEND_PUBLIC_WS_URL (the URL of THIS deployment as wss://).
        # Set after first deploy; redeploy to inject.
        modal.Secret.from_name("clio-backend-cfg"),
    ],
    # Always-warm so Twilio's webhook doesn't pay cold start (Twilio gives
    # ~15s before timing out). Also keeps the WS endpoint hot for the GPU
    # container's connect-back.
    min_containers=1,
    # Per-call WebSocket lives for the duration of the call. 1h matches the
    # GPU container's timeout in modal_app.py.
    timeout=60 * 60,
    cpu=1.0,
    memory=1024,
    volumes={"/root/data/sessions": sessions_vol},
)
@modal.asgi_app()
def fastapi_app():
    """The FastAPI app exported by backend.app.control.server.

    Imported lazily inside the function so image build doesn't try to
    import backend (which needs runtime env vars to construct clients)."""
    from backend.app.control.server import app as fastapi
    return fastapi


# ─── CLI helpers ─────────────────────────────────────────────────────────────


@app.local_entrypoint()
def print_url() -> None:
    """Print the URLs of the deployed (non-ephemeral) app.

    Usage: `modal run model_service/deploy/modal_backend.py::print_url`

    Note: `modal run` itself creates an ephemeral instance with a `-dev`
    URL suffix that's destroyed when this entrypoint exits. We look up
    the persistent deploy via Function.from_name so the printed URLs
    point at the long-lived deployment, not this temporary one.
    """
    try:
        deployed = modal.Function.from_name(APP_NAME, "fastapi_app")
        url = deployed.get_web_url()
    except modal.exception.NotFoundError:
        print(f"No deployed app named '{APP_NAME}' found. Run "
              f"`modal deploy model_service/deploy/modal_backend.py` first.")
        return

    print(f"backend URL : {url}")
    print(f"twilio voice: {url}/twilio/voice")
    print(f"twilio stat : {url}/twilio/status")
    print(f"ws control  : {url.replace('https://', 'wss://')}/control/<call_id>")
    print()
    print("Set as Modal secret so the backend tells GPU containers to "
          "connect here:")
    ws = url.replace("https://", "wss://")
    print(f"  modal secret create clio-backend-cfg "
          f"BACKEND_PUBLIC_WS_URL={ws} --force")
    print("  modal deploy model_service/deploy/modal_backend.py  # redeploy")
