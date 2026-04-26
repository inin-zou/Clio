"use client";

import { useEffect, useMemo, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { Call, EventRow, Message } from "@/lib/types";

const RECENT_CALL_LIMIT = 10;

export default function Page() {
  const [calls, setCalls] = useState<Call[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  // ─── Initial load: recent calls ───────────────────────────────────────
  useEffect(() => {
    const sb = supabase();
    let cancelled = false;

    (async () => {
      const { data, error } = await sb
        .from("calls")
        .select("*")
        .order("started_at", { ascending: false })
        .limit(RECENT_CALL_LIMIT);
      if (cancelled) return;
      if (error) {
        setError(error.message);
        return;
      }
      setCalls((data as Call[]) ?? []);
      // Auto-pick the most recent active call (no ended_at) or the latest.
      const active = (data as Call[] | null)?.find((c) => !c.ended_at);
      if (!selectedId) setSelectedId((active ?? data?.[0])?.id ?? null);
    })();

    // Subscribe to new/updated calls so the sidebar stays current.
    const callsCh = sb
      .channel("calls-list")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "calls" },
        (payload) => {
          setCalls((prev) => {
            const next = [...prev];
            if (payload.eventType === "INSERT") {
              next.unshift(payload.new as Call);
            } else if (payload.eventType === "UPDATE") {
              const i = next.findIndex(
                (c) => c.id === (payload.new as Call).id,
              );
              if (i >= 0) next[i] = payload.new as Call;
            }
            return next.slice(0, RECENT_CALL_LIMIT);
          });
          // If a brand-new call appears and nothing is selected yet, jump to it.
          if (payload.eventType === "INSERT" && !selectedId) {
            setSelectedId((payload.new as Call).id);
          }
        },
      )
      .subscribe();

    return () => {
      cancelled = true;
      sb.removeChannel(callsCh);
    };
  }, [selectedId]);

  // ─── Per-call: load history + subscribe to realtime ───────────────────
  useEffect(() => {
    if (!selectedId) return;
    const sb = supabase();
    let cancelled = false;

    (async () => {
      const [m, e] = await Promise.all([
        sb
          .from("messages")
          .select("*")
          .eq("call_id", selectedId)
          .order("timestamp", { ascending: true })
          .limit(500),
        sb
          .from("events")
          .select("*")
          .eq("call_id", selectedId)
          .order("timestamp", { ascending: true })
          .limit(500),
      ]);
      if (cancelled) return;
      setMessages((m.data as Message[]) ?? []);
      setEvents((e.data as EventRow[]) ?? []);
    })();

    const ch = sb
      .channel(`call-${selectedId}`)
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "messages",
          filter: `call_id=eq.${selectedId}`,
        },
        (payload) => {
          setMessages((prev) => [...prev, payload.new as Message]);
        },
      )
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "events",
          filter: `call_id=eq.${selectedId}`,
        },
        (payload) => {
          setEvents((prev) => [...prev, payload.new as EventRow]);
        },
      )
      .subscribe();

    return () => {
      cancelled = true;
      sb.removeChannel(ch);
    };
  }, [selectedId]);

  const selectedCall = useMemo(
    () => calls.find((c) => c.id === selectedId) ?? null,
    [calls, selectedId],
  );

  // Merge messages + events into one chronological feed for the main pane.
  const feed = useMemo(() => {
    const items: Array<
      | { kind: "message"; row: Message }
      | { kind: "event"; row: EventRow }
    > = [
      ...messages.map((row) => ({ kind: "message" as const, row })),
      ...events.map((row) => ({ kind: "event" as const, row })),
    ];
    items.sort((a, b) =>
      a.row.timestamp < b.row.timestamp
        ? -1
        : a.row.timestamp > b.row.timestamp
          ? 1
          : 0,
    );
    return items;
  }, [messages, events]);

  // Latest known value per slot path (from slot_update events).
  const slots = useMemo(() => {
    const out = new Map<string, unknown>();
    for (const ev of events) {
      if (ev.type !== "slot_update") continue;
      const applied = (ev.payload as { applied?: Array<{ slot_path: string; value: unknown }> })
        .applied ?? [];
      for (const u of applied) {
        if (u.value !== null && u.value !== undefined) {
          out.set(u.slot_path, u.value);
        }
      }
    }
    return out;
  }, [events]);

  return (
    <main className="grid grid-cols-[260px_1fr_340px] h-screen">
      {/* ─── Left: recent calls ───────────────────────────── */}
      <aside className="border-r border-[#222630] overflow-y-auto p-3">
        <h2 className="text-xs uppercase tracking-widest text-muted mb-2 px-2">
          Recent calls
        </h2>
        {error && (
          <div className="text-xs text-gate p-2">{error}</div>
        )}
        {calls.length === 0 && (
          <div className="text-xs text-muted p-2">no calls yet</div>
        )}
        {calls.map((c) => {
          const active = !c.ended_at;
          const selected = c.id === selectedId;
          return (
            <button
              key={c.id}
              onClick={() => setSelectedId(c.id)}
              className={`w-full text-left px-3 py-2 rounded mb-1 text-sm transition ${
                selected
                  ? "bg-panel text-ink"
                  : "hover:bg-panel/60 text-muted"
              }`}
            >
              <div className="flex items-center gap-2">
                <span
                  className={`inline-block w-2 h-2 rounded-full ${
                    active ? "bg-agent animate-pulse" : "bg-muted/50"
                  }`}
                />
                <span className="font-mono text-xs truncate flex-1">
                  {c.id}
                </span>
              </div>
              <div className="text-[10px] text-muted mt-1">
                {new Date(c.started_at).toLocaleTimeString()}
                {c.policy_number && ` · ${c.policy_number}`}
              </div>
            </button>
          );
        })}
      </aside>

      {/* ─── Middle: live feed ────────────────────────────── */}
      <section className="overflow-y-auto p-6">
        {!selectedCall && (
          <div className="text-muted text-sm">
            Select a call from the left to view its live transcript.
          </div>
        )}
        {selectedCall && (
          <>
            <h1 className="text-xs uppercase tracking-widest text-muted mb-1">
              Live transcript
            </h1>
            <p className="text-xs text-muted mb-4 font-mono">
              {selectedCall.id}
              {selectedCall.ended_at
                ? " · ended"
                : " · live"}
            </p>
            <div className="space-y-2">
              {feed.length === 0 && (
                <div className="text-muted italic text-sm">
                  Waiting for first event…
                </div>
              )}
              {feed.map((item) =>
                item.kind === "message" ? (
                  <MessageRow key={`m-${item.row.id}`} m={item.row} />
                ) : (
                  <EventRowView key={`e-${item.row.id}`} e={item.row} />
                ),
              )}
            </div>
          </>
        )}
      </section>

      {/* ─── Right: FNOL state ────────────────────────────── */}
      <aside className="border-l border-[#222630] overflow-y-auto p-4 bg-panel">
        <h2 className="text-xs uppercase tracking-widest text-muted mb-3">
          FNOL state
        </h2>
        {slots.size === 0 && (
          <div className="text-xs text-muted italic">
            no slots filled yet
          </div>
        )}
        {Array.from(slots.entries()).map(([path, value]) => (
          <div
            key={path}
            className="mb-2 p-2 rounded bg-slot/10 border-l-2 border-slot"
          >
            <div className="text-[10px] text-muted">{path}</div>
            <div className="text-sm break-words">
              {typeof value === "object"
                ? JSON.stringify(value)
                : String(value)}
            </div>
          </div>
        ))}
        {selectedCall?.fnol && (
          <details className="mt-6">
            <summary className="text-xs text-muted cursor-pointer mb-2">
              full FNOL JSON
            </summary>
            <pre className="text-[10px] overflow-x-auto bg-bg p-2 rounded">
              {JSON.stringify(selectedCall.fnol, null, 2)}
            </pre>
          </details>
        )}
      </aside>
    </main>
  );
}

function MessageRow({ m }: { m: Message }) {
  const colorClass = m.role === "caller" ? "text-caller" : "text-agent";
  return (
    <div className="grid grid-cols-[64px_1fr_100px] gap-3 py-1.5 px-3 rounded hover:bg-panel/40">
      <div className={`text-xs font-semibold ${colorClass}`}>
        {m.role.toUpperCase()}
      </div>
      <div className="text-sm leading-relaxed">{m.text}</div>
      <div className="text-[10px] text-muted text-right">
        {m.source ?? ""} · {new Date(m.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}

function EventRowView({ e }: { e: EventRow }) {
  let label = "";
  let cls = "text-muted";
  switch (e.type) {
    case "slot_update": {
      const applied = (e.payload as { applied?: Array<{ slot_path: string; value: unknown }> })
        .applied ?? [];
      label = `📋 ${applied
        .map(
          (u) =>
            `${u.slot_path} = ${typeof u.value === "object" ? JSON.stringify(u.value) : u.value}`,
        )
        .join(", ")}`;
      cls = "text-slot";
      break;
    }
    case "gate_fired": {
      const p = e.payload as { reason?: string; text?: string };
      label = `🛎  gate: ${p.reason ?? ""}${p.text ? ` — "${p.text}"` : ""}`;
      cls = "text-gate";
      break;
    }
    case "readback": {
      const p = e.payload as {
        slot_path?: string;
        caller_response?: string;
        final_value?: string;
      };
      label = `↩ readback ${p.slot_path} → ${p.caller_response} (${p.final_value})`;
      cls = "text-[#ffd9a8]";
      break;
    }
    default:
      label = `${e.type}: ${JSON.stringify(e.payload)}`;
  }
  return (
    <div className={`text-xs italic px-3 py-1 ${cls}`}>{label}</div>
  );
}
