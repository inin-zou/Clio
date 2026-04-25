# Clio model_service

PersonaPlex 7B inference + Reasoner-driven `text_token` injection. Runs on Modal GPU in production; runs locally with a `MockTalker` stub for protocol development without GPU.

## Architecture

```
Backend (LiveKit Agent)
    ↕ WebSocket  ws://host:port/session
                 single multiplexed channel for MVP1:
                 audio frames + control directives + transcript + status
    ↕
Server (this package)
    ↓ uses
Talker interface  (server/talker.py)
    ├─ MockTalker          local dev — no torch, no GPU
    └─ PersonaPlexTalker   Modal-only, requires [modal] extra (torch + moshi)
```

The `Session` (`server/session.py`) owns one Talker instance per connection plus a drip-feed queue. Inbound `SpeakDirective`s tokenize into queued `text_token` IDs that the per-frame loop force-injects via `Talker.step(forced_text_token_id=…)`.

## Running locally (MockTalker, no GPU)

```bash
# from this directory
uv sync
uv run python -m server.main          # ws://127.0.0.1:8765
```

## Tests

```bash
uv run pytest                          # tests/test_mock_session.py
```

## Deploying to Modal (PersonaPlexTalker)

Not yet wired up. See `deploy/modal_app.py` (TODO) and `talker.PersonaPlexTalker.init_session` for the implementation plan. The Modal image installs `[modal]` extra (torch + moshi from NVIDIA's PersonaPlex repo).

## Wire format

Single multiplexed WebSocket. All messages are JSON discriminated unions on `type`. See `server/protocol.py` for the full schema. Audio frames are base64-encoded 16-bit PCM at 24kHz mono, one frame = 80ms = 1920 samples.

## Why this layout

- **`server/protocol.py`** — single source of truth for the WS protocol. Mirrors backend's `drip.py` directive types but is independent (no cross-package import).
- **`server/talker.py`** — abstract interface; PersonaPlex inference is plug-replaceable.
- **`server/session.py`** — per-call orchestration; talker-implementation-agnostic. Drip-feed lives here.
- **`server/main.py`** — websockets server entry point.

The split lets backend/livekit_agent.py be developed end-to-end against MockTalker without waiting on GPU deployment.
