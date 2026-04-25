"""
WebSocket entry point for model_service.

  uv run python -m server.main           # uses MockTalker (local dev)
  uv run python -m server.main --real    # uses PersonaPlexTalker (Modal only)

Single multiplexed WS at ws://host:port/session. Both audio frames (base64 PCM
inside JSON) and control directives travel the same connection. We'll split
into two-channel architecture (architecture.md) once head-of-line blocking
becomes measurable — for MVP1 single-channel is simpler and adequate.

Wire format: every JSON message is a discriminated union (`type` field) per
protocol.py. Unrecognized types return ServerError(code="unknown_type").
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError
import websockets

from .protocol import ClientMessage, ServerError
from .session import Session
from .talker import MockTalker, Talker

if TYPE_CHECKING:
    from websockets.legacy.server import WebSocketServerProtocol

logger = logging.getLogger("clio.model_service")

# Pydantic 2 union dispatcher. Built once at import; reused per message.
_CLIENT_MSG = TypeAdapter(ClientMessage)


def _make_talker(use_real: bool) -> Talker:
    if use_real:
        from .talker import PersonaPlexTalker
        return PersonaPlexTalker()
    return MockTalker()


async def _handle_connection(
    websocket: WebSocketServerProtocol,
    use_real_talker: bool,
) -> None:
    talker = _make_talker(use_real_talker)
    session = Session(talker=talker)
    logger.info("client connected, talker=%s", type(talker).__name__)

    sender_task = asyncio.create_task(_sender(websocket, session))
    try:
        async for raw in websocket:
            try:
                payload = json.loads(raw)
                message = _CLIENT_MSG.validate_python(payload)
            except (json.JSONDecodeError, ValidationError) as e:
                err = ServerError(code="invalid_message", message=str(e))
                await websocket.send(err.model_dump_json())
                continue
            await session.handle(message)
    except websockets.ConnectionClosed:
        logger.info("client disconnected")
    finally:
        await session.close(reason="connection closed")
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass


async def _sender(websocket: WebSocketServerProtocol, session: Session) -> None:
    """Drain session.outgoing() back to the WebSocket."""
    try:
        async for msg in session.outgoing():
            try:
                await websocket.send(msg.model_dump_json())
            except websockets.ConnectionClosed:
                return
    except asyncio.CancelledError:
        pass


async def main(host: str, port: int, use_real_talker: bool) -> None:
    logger.info("starting model_service on ws://%s:%d (real=%s)",
                host, port, use_real_talker)

    async def handler(ws: WebSocketServerProtocol) -> None:
        await _handle_connection(ws, use_real_talker)

    async with websockets.serve(handler, host, port, max_size=2**20):
        await asyncio.Future()  # run forever


def cli() -> None:
    parser = argparse.ArgumentParser(description="Clio model_service WS server.")
    parser.add_argument("--host", default=os.environ.get("MODEL_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MODEL_SERVICE_PORT", "8765")))
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use PersonaPlexTalker (requires GPU + moshi installed). Default: MockTalker.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    asyncio.run(main(args.host, args.port, args.real))


if __name__ == "__main__":
    cli()
