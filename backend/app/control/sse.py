"""SSE router for streaming live call events to a browser UI.

Endpoints
---------
GET  /events/active     SSE stream of events from ALL currently-active calls
GET  /events/{call_id}  SSE stream scoped to one call_id
GET  /ui                Single-page monitoring UI (subscribes to /events/active)

Events
------
Each SSE message is a JSON object. Common fields:
  - type: "transcript" | "session_ready" | "session_end" |
          "slot_updates" | "gate_directive" | "readback"
  - call_id: present on /events/active so the UI can group multi-call streams
  - timestamp: ISO 8601

Specific shapes are produced by orchestrator.py — keep them in sync.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .eventbus import ANY_CALL, EVENT_BUS

logger = logging.getLogger("clio.control.sse")

router = APIRouter(tags=["ui"])


# ─── SSE helpers ─────────────────────────────────────────────────────────────


async def _sse_stream(request: Request, call_id: str) -> AsyncIterator[bytes]:
    """Generic SSE generator for a given subscription target."""
    queue = EVENT_BUS.subscribe(call_id)
    try:
        # Initial hello so the client confirms the stream is open immediately,
        # not after the first real event (which could be minutes away on a
        # quiet call).
        yield b'event: hello\ndata: {}\n\n'
        while True:
            if await request.is_disconnected():
                return
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Heartbeat to keep proxies (Modal, Cloudflare, etc.) from
                # closing an idle connection.
                yield b': keepalive\n\n'
                continue
            payload = json.dumps(event, default=str)
            yield f"data: {payload}\n\n".encode()
    finally:
        EVENT_BUS.unsubscribe(call_id, queue)


@router.get("/events/active")
async def events_active(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _sse_stream(request, ANY_CALL),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


@router.get("/events/{call_id}")
async def events_call(call_id: str, request: Request) -> StreamingResponse:
    return StreamingResponse(
        _sse_stream(request, call_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── UI page ─────────────────────────────────────────────────────────────────


_UI_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Clio — live call monitor</title>
<style>
  :root {
    --bg: #0b0d12;
    --panel: #141821;
    --ink: #e6e9ef;
    --muted: #8b93a3;
    --accent: #5cc8ff;
    --caller: #ffd479;
    --agent: #a0e7a0;
    --slot: #c89cff;
    --gate: #ff9c8a;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%;
    background: var(--bg); color: var(--ink);
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
  }
  body { display: grid; grid-template-columns: 1fr 380px; height: 100vh; }
  #transcript {
    overflow-y: auto; padding: 16px 24px;
    border-right: 1px solid #222630;
  }
  #sidebar {
    overflow-y: auto; padding: 16px 20px; background: var(--panel);
  }
  h1 { font-size: 13px; letter-spacing: .12em; text-transform: uppercase;
       color: var(--muted); margin: 0 0 12px; }
  .row {
    margin: 6px 0; padding: 8px 12px; border-radius: 6px;
    background: rgba(255,255,255,.03); display: grid;
    grid-template-columns: 56px 1fr 100px; gap: 12px;
  }
  .row .who { font-weight: 600; font-size: 12px; padding-top: 1px; }
  .row .who.caller { color: var(--caller); }
  .row .who.agent  { color: var(--agent); }
  .row .src        { color: var(--muted); font-size: 11px; padding-top: 2px;
                     text-align: right; }
  .row .text       { line-height: 1.45; }
  .meta {
    margin: 6px 0; padding: 6px 12px; border-radius: 6px;
    color: var(--muted); font-size: 12px; font-style: italic;
  }
  .meta.slot { color: var(--slot); }
  .meta.gate { color: var(--gate); }
  .meta.lifecycle { color: var(--accent); }
  .meta.readback { color: #ffd9a8; }
  #status {
    padding: 8px 14px; border-radius: 6px; background: #1a1f2c;
    color: var(--muted); font-size: 12px; margin-bottom: 16px;
  }
  #status.connected { color: var(--agent); }
  #status.error { color: var(--gate); }
  .slot-card {
    padding: 8px 10px; margin: 4px 0; border-radius: 6px;
    background: rgba(200,156,255,.08); border-left: 3px solid var(--slot);
  }
  .slot-path { font-size: 11px; color: var(--muted); }
  .slot-value { font-weight: 600; word-break: break-word; }
  pre { white-space: pre-wrap; word-break: break-word;
        background: #1a1f2c; padding: 10px; border-radius: 6px;
        font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
  <main id="transcript">
    <h1>Live transcript</h1>
    <div id="rows"></div>
  </main>
  <aside id="sidebar">
    <div id="status">connecting…</div>
    <h1>FNOL state</h1>
    <div id="slots"><div class="meta">waiting for first extraction…</div></div>
    <h1 style="margin-top:24px">Recent events</h1>
    <pre id="raw"></pre>
  </aside>

<script>
const rowsEl = document.getElementById("rows");
const slotsEl = document.getElementById("slots");
const rawEl = document.getElementById("raw");
const statusEl = document.getElementById("status");

const slots = new Map();           // slot_path -> value
const recent = [];                 // last N raw events for the debug pane

function addTranscript(ev) {
  const row = document.createElement("div");
  row.className = "row";
  const time = new Date(ev.timestamp).toLocaleTimeString();
  row.innerHTML = `
    <div class="who ${ev.role}">${ev.role.toUpperCase()}</div>
    <div class="text"></div>
    <div class="src">${ev.source} · ${time}</div>
  `;
  row.querySelector(".text").textContent = ev.text;
  rowsEl.appendChild(row);
  rowsEl.parentElement.scrollTop = rowsEl.parentElement.scrollHeight;
}

function addMeta(cls, text) {
  const row = document.createElement("div");
  row.className = `meta ${cls}`;
  row.textContent = text;
  rowsEl.appendChild(row);
  rowsEl.parentElement.scrollTop = rowsEl.parentElement.scrollHeight;
}

function renderSlots() {
  if (slots.size === 0) {
    slotsEl.innerHTML = '<div class="meta">waiting for first extraction…</div>';
    return;
  }
  slotsEl.innerHTML = "";
  for (const [path, value] of slots.entries()) {
    const card = document.createElement("div");
    card.className = "slot-card";
    card.innerHTML = `
      <div class="slot-path">${path}</div>
      <div class="slot-value"></div>
    `;
    card.querySelector(".slot-value").textContent =
      typeof value === "object" ? JSON.stringify(value) : String(value);
    slotsEl.appendChild(card);
  }
}

function pushRaw(ev) {
  recent.unshift(ev);
  if (recent.length > 12) recent.pop();
  rawEl.textContent = recent
    .map((e) => `${e.type.padEnd(14)} ${(e.timestamp || "").slice(11, 19)}`)
    .join("\\n");
}

function handleEvent(ev) {
  pushRaw(ev);
  switch (ev.type) {
    case "transcript":
      addTranscript(ev);
      break;
    case "session_ready":
      addMeta("lifecycle",
        `▶ session ready (cold start ${ev.cold_start_seconds.toFixed(2)}s, voice=${ev.voice_prompt_id})`);
      break;
    case "session_end":
      addMeta("lifecycle", `■ session end — ${ev.reason || "—"}`);
      break;
    case "slot_updates":
      for (const u of (ev.applied || [])) {
        slots.set(u.slot_path, u.value);
        addMeta("slot", `📋 ${u.slot_path} = ${typeof u.value === 'object' ? JSON.stringify(u.value) : u.value}`);
      }
      renderSlots();
      break;
    case "gate_directive":
      addMeta("gate",
        `🛎  gate fired (${ev.directive_type}): ${ev.reason}` +
        (ev.text ? ` — Sarah will say: "${ev.text}"` : ""));
      break;
    case "readback":
      addMeta("readback",
        `↩ readback ${ev.slot_path} → ${ev.caller_response} (${ev.final_value})`);
      break;
  }
}

function connect() {
  statusEl.textContent = "connecting…";
  statusEl.className = "";
  const es = new EventSource("/events/active");
  es.addEventListener("hello", () => {
    statusEl.textContent = "● connected — waiting for events";
    statusEl.className = "connected";
  });
  es.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data);
      handleEvent(ev);
    } catch (e) {
      console.error("bad event", msg.data, e);
    }
  };
  es.onerror = () => {
    statusEl.textContent = "● disconnected — retrying…";
    statusEl.className = "error";
    es.close();
    setTimeout(connect, 2000);
  };
}

connect();
</script>
</body>
</html>
"""


@router.get("/ui", response_class=HTMLResponse)
async def ui_page() -> HTMLResponse:
    return HTMLResponse(content=_UI_HTML)
