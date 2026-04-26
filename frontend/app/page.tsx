"use client";

import { useEffect, useMemo, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { Call, EventRow, Message } from "@/lib/types";
import {
  callCode,
  callerDisplayName,
  elapsedLabel,
  fnolDraftFields,
  fnolPercent,
  groupTranscriptBubbles,
  incidentTypeLabel,
  locationDetail,
  statusOf,
  type TranscriptBubble,
} from "@/lib/derive";
import {
  AUDIO_SETTINGS,
  COMPLIANCE_RULES,
  CRITICAL_SLOTS,
  GATE_SETTINGS,
  IDENTIFIER_SLOTS,
  PERSONA_PROMPT,
  RESCUE_RESPONSES,
  RESCUE_TRIGGERS,
  WRAPUP_PHRASINGS,
} from "@/lib/agent-config";

const RECENT_CALL_LIMIT = 12;

export default function Page() {
  const [tab, setTab] = useState<"ops" | "vault">("ops");
  const [calls, setCalls] = useState<Call[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Tick every second so elapsed times update.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // ─── Initial load: recent calls + realtime sub ─────────────────────
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
      const list = (data as Call[]) ?? [];
      setCalls(list);
      const active = list.find((c) => !c.ended_at);
      setSelectedId((s) => s ?? (active ?? list[0])?.id ?? null);
    })();

    const ch = sb
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
          if (payload.eventType === "INSERT") {
            const id = (payload.new as Call).id;
            setSelectedId((s) => s ?? id);
          }
        },
      )
      .subscribe();

    return () => {
      cancelled = true;
      sb.removeChannel(ch);
    };
  }, []);

  // ─── Per-call: history + realtime ──────────────────────────────────
  useEffect(() => {
    if (!selectedId) return;
    const sb = supabase();
    let cancelled = false;
    (async () => {
      const [m, e] = await Promise.all([
        sb.from("messages").select("*").eq("call_id", selectedId)
          .order("timestamp", { ascending: true }).limit(500),
        sb.from("events").select("*").eq("call_id", selectedId)
          .order("timestamp", { ascending: true }).limit(500),
      ]);
      if (cancelled) return;
      setMessages((m.data as Message[]) ?? []);
      setEvents((e.data as EventRow[]) ?? []);
    })();

    const ch = sb
      .channel(`call-${selectedId}`)
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "messages",
          filter: `call_id=eq.${selectedId}` },
        (payload) => setMessages((p) => [...p, payload.new as Message]),
      )
      .on(
        "postgres_changes",
        { event: "INSERT", schema: "public", table: "events",
          filter: `call_id=eq.${selectedId}` },
        (payload) => setEvents((p) => [...p, payload.new as EventRow]),
      )
      .subscribe();

    return () => {
      cancelled = true;
      sb.removeChannel(ch);
    };
  }, [selectedId]);

  const callsOldFirst = useMemo(() => [...calls].reverse(), [calls]);
  const callIndexById = useMemo(() => {
    const m = new Map<string, number>();
    callsOldFirst.forEach((c, i) => m.set(c.id, i));
    return m;
  }, [callsOldFirst]);

  // Active calls = currently-live only (ended_at IS NULL). When nothing is
  // live, the panel shows its empty state — no borrowing from history, so
  // the "Active" label and the LIVE/REVIEW badges always tell the truth.
  const activeCalls = useMemo(() => {
    return calls.filter((c) => !c.ended_at).slice(0, 3);
  }, [calls]);

  const recentClaims = useMemo(() => {
    return calls.filter((c) => c.ended_at).slice(0, 6);
  }, [calls]);

  const selectedCall = useMemo(
    () => calls.find((c) => c.id === selectedId) ?? null,
    [calls, selectedId],
  );

  // KPI: average call duration over completed calls in the visible list.
  const avgFnolLabel = useMemo(() => {
    const completed = calls.filter((c) => c.ended_at);
    if (completed.length === 0) return "—";
    const totalSec = completed.reduce((acc, c) => {
      const start = Date.parse(c.started_at);
      const end = c.ended_at ? Date.parse(c.ended_at) : start;
      return acc + Math.max(0, (end - start) / 1000);
    }, 0);
    const avg = totalSec / completed.length;
    const m = Math.floor(avg / 60);
    const s = Math.round(avg % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }, [calls]);

  return (
    <>
      <div className="env-bg" />
      <TopToolbar tab={tab} setTab={setTab} />

      <main
        style={{
          position: "relative",
          zIndex: 1,
          maxWidth: 1480,
          margin: "0 auto",
          padding: "108px 40px 72px",
        }}
      >
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-end",
            marginBottom: 56,
            gap: 32,
          }}
        >
          <div style={{ maxWidth: 640 }}>
            <h1 className="display" style={{ fontSize: 60, marginBottom: 18 }}>
              Inbound claims,
              <br />
              drafted as they speak.
            </h1>
            <p
              style={{
                color: "var(--text-secondary)",
                fontSize: 16,
                lineHeight: 1.55,
                margin: 0,
                letterSpacing: "-0.005em",
                maxWidth: 540,
              }}
            >
              Live transcript, structured claim. Every word the caller says
              becomes part of the FNOL — in real time.
            </p>
          </div>
          {error && (
            <div
              style={{
                color: "var(--status-error)",
                fontSize: 12,
                fontFamily: "var(--font-mono)",
              }}
            >
              {error}
            </div>
          )}
        </header>

        {tab === "ops" && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "260px 1fr 280px",
            gap: 32,
            alignItems: "start",
          }}
        >
          {/* ─── LEFT: active calls ─── */}
          <aside
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 4,
              paddingTop: 8,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                padding: "0 16px 12px",
              }}
            >
              <h3 className="section-title">Active</h3>
              <span className="meta-label">
                {activeCalls.length} of {calls.length}
              </span>
            </div>
            {activeCalls.length === 0 && (
              <div
                style={{
                  padding: "12px 16px",
                  color: "var(--text-tertiary)",
                  fontSize: 12,
                }}
              >
                No calls yet. Dial the number.
              </div>
            )}
            {activeCalls.map((c) => (
              <CallRowSidebar
                key={c.id}
                call={c}
                code={callCode(c, callIndexById.get(c.id) ?? 0)}
                selected={c.id === selectedId}
                onClick={() => setSelectedId(c.id)}
                now={now}
              />
            ))}
          </aside>

          {/* ─── MIDDLE: hero call ─── */}
          {selectedCall ? (
            <HeroCall
              call={selectedCall}
              code={callCode(selectedCall, callIndexById.get(selectedCall.id) ?? 0)}
              messages={messages}
              now={now}
            />
          ) : (
            <section
              className="glass"
              style={{
                padding: 60,
                color: "var(--text-tertiary)",
                fontSize: 14,
              }}
            >
              Select a call to view its live transcript.
            </section>
          )}

          {/* ─── RIGHT: recent + KPIs ─── */}
          <aside
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 36,
              paddingTop: 8,
            }}
          >
            <div>
              <h3 className="section-title" style={{ marginBottom: 10 }}>
                Recent claims
              </h3>
              <div>
                {recentClaims.length === 0 && (
                  <div
                    style={{
                      color: "var(--text-tertiary)",
                      fontSize: 12,
                      padding: "8px 14px",
                    }}
                  >
                    No completed calls yet.
                  </div>
                )}
                {recentClaims.map((c) => (
                  <ClaimRow
                    key={c.id}
                    call={c}
                    onClick={() => setSelectedId(c.id)}
                  />
                ))}
              </div>
            </div>
            <div>
              <h3 className="section-title" style={{ marginBottom: 14 }}>
                This window
              </h3>
              <div style={{ display: "flex", gap: 28 }}>
                <div className="kpi">
                  <span className="kpi-label">Avg FNOL</span>
                  <span className="kpi-value">{avgFnolLabel}</span>
                </div>
                <div className="hairline-y" />
                <div className="kpi">
                  <span className="kpi-label">Total calls</span>
                  <span className="kpi-value">{calls.length}</span>
                </div>
              </div>
            </div>
          </aside>
        </div>
        )}

        {tab === "vault" && <ContextVault />}
      </main>
    </>
  );
}

// ───────────────────────────────────────────────────────────────────
// Subcomponents
// ───────────────────────────────────────────────────────────────────

function TopToolbar({
  tab,
  setTab,
}: {
  tab: "ops" | "vault";
  setTab: (t: "ops" | "vault") => void;
}) {
  return (
    <header className="top-toolbar">
      <span className="toolbar-brand">clio</span>
      <div className="toolbar-segments">
        {(
          [
            { id: "ops", label: "Operations" },
            { id: "vault", label: "Context Vault" },
          ] as const
        ).map((t) => (
          <button
            key={t.id}
            className={`toolbar-seg ${tab === t.id ? "is-active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <span className="toolbar-meta">v1.0 · beta</span>
    </header>
  );
}

function CallRowSidebar({
  call,
  code,
  selected,
  onClick,
  now,
}: {
  call: Call;
  code: string;
  selected: boolean;
  onClick: () => void;
  now: number;
}) {
  const status = statusOf(call);
  const dotClass =
    status.kind === "live"
      ? "dot-live"
      : status.kind === "review"
        ? "dot-review"
        : "dot-ended";
  return (
    <div
      className={`call-row ${selected ? "is-selected" : ""}`}
      onClick={onClick}
    >
      <div className="call-row-top">
        <span className={`dot ${dotClass}`} />
        <span className="status-text">
          {status.label} · {elapsedLabel(call, now)}
        </span>
      </div>
      <div className="call-row-name">{callerDisplayName(call)}</div>
      <div className="call-row-meta">
        {incidentTypeLabel(call)} · {locationDetail(call)} · FNOL{" "}
        {fnolPercent(call)}%
      </div>
      <div
        className="meta-label"
        style={{ marginTop: 6, fontSize: 9 }}
      >
        {code}
      </div>
    </div>
  );
}

function ClaimRow({ call, onClick }: { call: Call; onClick: () => void }) {
  const fraudFlags = (call.fnol as Record<string, unknown> | null)?.[
    "fraud_flags"
  ] as unknown[] | undefined;
  const flagged = Array.isArray(fraudFlags) && fraudFlags.length > 0;
  const cls = flagged ? "is-flagged" : "";
  const status = flagged ? "FLAGGED" : "REVIEW";
  return (
    <div className="claim-row" onClick={onClick} style={{ cursor: "pointer" }}>
      <span>
        <span className="claim-row-name">{callerDisplayName(call)}</span>
        <span className="claim-row-type"> · {incidentTypeLabel(call)}</span>
      </span>
      <span className={`claim-row-status ${cls}`}>{status}</span>
    </div>
  );
}

function VoiceRibbon({
  kind,
  amp = 12,
  freq = 0.012,
}: {
  kind: "agent" | "caller";
  amp?: number;
  freq?: number;
}) {
  const w = 1600;
  const h = 96;
  const phase = kind === "agent" ? 0 : Math.PI / 2;
  const points: string[] = [];
  for (let x = 0; x <= w; x += 6) {
    const y =
      h / 2 +
      Math.sin(x * freq + phase) * amp +
      Math.sin(x * freq * 0.5) * (amp * 0.35);
    points.push(`${x},${y.toFixed(1)}`);
  }
  return (
    <div className={`voiceline-ribbon ${kind} active`}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <path d={`M ${points.join(" L ")}`} strokeWidth="1" />
      </svg>
    </div>
  );
}

function HeroCall({
  call,
  code,
  messages,
  now,
}: {
  call: Call;
  code: string;
  messages: Message[];
  now: number;
}) {
  const [showAll, setShowAll] = useState(false);
  const fields = fnolDraftFields(call);
  const visible = showAll ? fields : fields.slice(0, 5);
  const filledCount = fields.filter((f) => f.value).length;
  const status = statusOf(call);

  return (
    <section
      className="glass"
      style={{
        padding: 40,
        display: "flex",
        flexDirection: "column",
        gap: 36,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 32,
        }}
      >
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 18,
            }}
          >
            <span
              className={`dot ${status.kind === "live" ? "dot-live" : "dot-ended"}`}
            />
            <span
              className={`status-text ${status.kind === "live" ? "status-text-live" : ""}`}
            >
              {status.label} · {elapsedLabel(call, now)}
            </span>
            <span className="meta-label" style={{ marginLeft: 6 }}>
              {code}
            </span>
          </div>
          <h2
            className="display"
            style={{ fontSize: 44, marginBottom: 8 }}
          >
            {callerDisplayName(call)}
          </h2>
          <p
            style={{
              color: "var(--text-secondary)",
              fontSize: 15,
              margin: 0,
              letterSpacing: "-0.005em",
            }}
          >
            {incidentTypeLabel(call)} — {locationDetail(call)}
          </p>
        </div>
        {status.kind === "live" && (
          <div style={{ display: "flex", gap: 10, flexShrink: 0 }}>
            <button className="btn btn-primary" disabled>Take over →</button>
          </div>
        )}
      </div>

      <div
        className="voiceline"
        style={{
          height: 88,
          background: "rgba(255,248,240,0.02)",
          border: "1px solid var(--hairline)",
          position: "relative",
        }}
      >
        <VoiceRibbon kind="agent" />
        <VoiceRibbon kind="caller" amp={9} freq={0.018} />
        <div
          style={{
            position: "absolute",
            inset: "auto 16px 10px 16px",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span className="meta-label">Agent</span>
          <span className="meta-label">Caller</span>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.4fr 1fr",
          gap: 40,
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginBottom: 14,
            }}
          >
            <h3 className="section-title">Live transcript</h3>
            <span className="meta-label">{messages.length} turns</span>
          </div>
          <div
            className="transcript-stream"
            style={{ maxHeight: 360, overflow: "auto" }}
          >
            {messages.length === 0 && (
              <div
                style={{
                  color: "var(--text-tertiary)",
                  fontSize: 13,
                  fontStyle: "italic",
                  padding: "12px 0",
                }}
              >
                Waiting for first turn…
              </div>
            )}
            {groupTranscriptBubbles(messages).map((b) => (
              <TranscriptBubbleView
                key={b.id}
                b={b}
                startedAt={call.started_at}
              />
            ))}
          </div>
        </div>

        <div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginBottom: 14,
            }}
          >
            <h3 className="section-title">Claim draft</h3>
            <span className="meta-label">
              {filledCount} of {fields.length} captured · {fnolPercent(call)}%
            </span>
          </div>
          <div>
            {visible.map((f, i) => (
              <div key={i} className="fnol-row">
                <span className="fnol-label">{f.label}</span>
                <span
                  className={`fnol-value ${f.value ? "" : "is-pending"}`}
                >
                  {f.value || "Pending"}
                </span>
              </div>
            ))}
          </div>
          <button
            className="show-more"
            onClick={() => setShowAll((s) => !s)}
          >
            {showAll ? "Show less" : "Show full draft  →"}
          </button>
        </div>
      </div>
    </section>
  );
}

function TranscriptBubbleView({
  b,
  startedAt,
}: {
  b: TranscriptBubble;
  startedAt: string;
}) {
  const offsetSec = Math.max(
    0,
    Math.floor((Date.parse(b.firstTimestamp) - Date.parse(startedAt)) / 1000),
  );
  const mm = Math.floor(offsetSec / 60);
  const ss = String(offsetSec % 60).padStart(2, "0");
  const isCaller = b.role === "caller";
  return (
    <div className={`bubble-row ${isCaller ? "is-caller" : "is-agent"}`}>
      <div className={`bubble ${isCaller ? "is-caller" : "is-agent"}`}>
        <div className="bubble-meta">
          <span>{isCaller ? "Caller" : "Agent"}</span>
          <span style={{ color: "var(--text-quaternary)" }}>
            {`${mm}:${ss}`}
          </span>
          {b.source && <span>{b.source}</span>}
        </div>
        <div className="bubble-text">{b.text}</div>
      </div>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────
// Context Vault — Sarah's persona, slot checklist, gate rules.
// Read-only mirror of backend constants. See lib/agent-config.ts.
// ───────────────────────────────────────────────────────────────────

function ContextVault() {
  return (
    <div className="vault">
      {/* Persona prompt — the WHO */}
      <VaultCard
        title="Persona prompt"
        subtitle="Loaded into PersonaPlex once at container start. Defines Sarah's identity, tone, and read-back protocol."
      >
        <pre className="vault-pre">{PERSONA_PROMPT}</pre>
      </VaultCard>

      {/* Two-column row: critical slots + identifier slots */}
      <div className="vault-row">
        <VaultCard
          title="Critical FNOL slots"
          subtitle="The 10 fields Sarah must collect before the call can close cleanly."
        >
          <div className="vault-list">
            {CRITICAL_SLOTS.map((s) => (
              <div key={s.path} className="vault-item">
                <div className="vault-item-main">
                  <code className="vault-code">{s.path}</code>
                  <span className="vault-item-label">{s.label}</span>
                </div>
                <span className="vault-item-desc">{s.desc}</span>
              </div>
            ))}
          </div>
        </VaultCard>

        <VaultCard
          title="Anchor filter scope"
          subtitle="Updates for these slots are dropped unless the value appears in caller speech. Protects the FNOL from Sarah's hallucinations."
        >
          <div className="vault-chips">
            {IDENTIFIER_SLOTS.map((s) => (
              <code key={s} className="vault-chip">
                {s}
              </code>
            ))}
          </div>
        </VaultCard>
      </div>

      {/* Compliance — what Sarah asks when slots are missing */}
      <VaultCard
        title="Compliance triggers"
        subtitle="If a slot is still empty after its deadline, the gate forces Sarah to ask. Hand-templated phrasings, not LLM-generated."
      >
        <div className="vault-rules">
          {COMPLIANCE_RULES.map((r) => (
            <div key={r.slot} className="vault-rule">
              <div className="vault-rule-head">
                <code className="vault-code">{r.slot}</code>
                <span className="vault-rule-deadline">
                  {r.deadlineSec}s deadline
                </span>
              </div>
              <p className="vault-rule-text">"{r.phrasing}"</p>
            </div>
          ))}
        </div>
      </VaultCard>

      {/* Wrap-up — what Sarah asks when the caller wants to leave */}
      <VaultCard
        title="Wrap-up prompts"
        subtitle="If the caller signals close while critical slots are still empty, the gate iterates through these (max 2 attempts)."
      >
        <div className="vault-rules">
          {WRAPUP_PHRASINGS.map((w) => (
            <div key={w.slot} className="vault-rule">
              <div className="vault-rule-head">
                <code className="vault-code">{w.slot}</code>
              </div>
              <p className="vault-rule-text">"{w.phrasing}"</p>
            </div>
          ))}
        </div>
      </VaultCard>

      {/* Rescue — what Sarah says when caller can't hear her */}
      <VaultCard
        title="Audio rescue"
        subtitle="When the caller signals they can't hear Sarah, she acknowledges instead of plowing ahead."
      >
        <div className="vault-row">
          <div style={{ flex: 1, minWidth: 0 }}>
            <h4 className="vault-subhead">Trigger phrases</h4>
            <ul className="vault-bullets">
              {RESCUE_TRIGGERS.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h4 className="vault-subhead">Sarah's response (rotates)</h4>
            <ul className="vault-bullets">
              {RESCUE_RESPONSES.map((r) => (
                <li key={r}>"{r}"</li>
              ))}
            </ul>
          </div>
        </div>
      </VaultCard>

      {/* Gate + audio knobs */}
      <div className="vault-row">
        <VaultCard
          title="Gate timing"
          subtitle="Cooldowns and limits that shape the cadence of Sarah's interventions."
        >
          <SettingsTable rows={GATE_SETTINGS} />
        </VaultCard>

        <VaultCard
          title="VAD / audio thresholds"
          subtitle="Voice-activity tunables baked into the per-frame inference loop."
        >
          <SettingsTable rows={AUDIO_SETTINGS} />
        </VaultCard>
      </div>

      <p className="vault-footnote">
        Read-only mirror. Edit{" "}
        <code>backend/app/reasoner/persona.py</code>,{" "}
        <code>gate.py</code>, or{" "}
        <code>model_service/deploy/modal_app.py</code> and redeploy to change.
      </p>
    </div>
  );
}

function VaultCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="glass vault-card">
      <header className="vault-card-head">
        <h3 className="vault-card-title">{title}</h3>
        {subtitle && <p className="vault-card-subtitle">{subtitle}</p>}
      </header>
      <div>{children}</div>
    </section>
  );
}

function SettingsTable({
  rows,
}: {
  rows: { name: string; value: string; desc: string }[];
}) {
  return (
    <div className="vault-table">
      {rows.map((r) => (
        <div key={r.name} className="vault-table-row">
          <span className="vault-table-name">{r.name}</span>
          <code className="vault-table-value">{r.value}</code>
          <span className="vault-table-desc">{r.desc}</span>
        </div>
      ))}
    </div>
  );
}
