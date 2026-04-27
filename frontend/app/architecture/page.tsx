/**
 * Architecture / benchmarks page — standalone, NOT linked from the main
 * Operations or Context Vault tabs. Accessible only via direct URL
 * (e.g. /architecture). Built so it can be screen-recorded for the
 * launch film or shared as a one-off explainer.
 *
 * Visual language matches the rest of the app: glass panels, green
 * accent (rgba(50,215,75)), Inter + JetBrains Mono. Re-uses .glass and
 * .vault-* classes from globals.css plus a few page-local additions
 * for the latency chart.
 */

"use client";

export default function ArchitecturePage() {
  return (
    <>
      <div className="env-bg" />
      <main
        style={{
          position: "relative",
          zIndex: 1,
          maxWidth: 1240,
          margin: "0 auto",
          padding: "80px 40px 96px",
        }}
      >
        {/* ─── Hero ─── */}
        <header style={{ marginBottom: 64 }}>
          <span className="meta-label" style={{ marginBottom: 18, display: "inline-block" }}>
            CLIO · TECHNICAL OVERVIEW
          </span>
          <h1
            className="display"
            style={{ fontSize: 56, marginBottom: 22, lineHeight: 1.05 }}
          >
            Full-duplex,
            <br />
            sub-second by design.
          </h1>
          <p
            style={{
              fontSize: 16,
              color: "var(--text-secondary)",
              lineHeight: 1.6,
              maxWidth: 620,
              margin: 0,
            }}
          >
            Clio is built on PersonaPlex 7B — an end-to-end speech LLM that
            consumes caller audio and emits agent audio simultaneously,
            frame-by-frame at 12.5Hz. No ASR. No TTS. No turn-taking gap.
            Same architectural family as ByteDance&rsquo;s{" "}
            <a
              href="https://seed.bytedance.com/en/seeduplex"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "var(--text-primary)", textDecoration: "underline" }}
            >
              Seeduplex
            </a>{" "}
            (April 2026), Kyutai&rsquo;s Moshi, and a handful of recent
            full-duplex papers.
          </p>
        </header>

        {/* ─── The three planes ─── */}
        <section className="glass arch-card">
          <header className="arch-card-head">
            <h2 className="arch-card-title">Three planes, deliberately decoupled</h2>
            <p className="arch-card-subtitle">
              Audio never traverses the backend. Persistence never blocks
              the voice loop. Each plane runs at its own latency budget.
            </p>
          </header>
          <div className="arch-planes">
            <PlaneRow
              tier="AUDIO PLANE"
              tierColor="green"
              latency="~450-650 ms RTT"
              flow="Caller phone → Twilio → LiveKit Cloud (WebRTC) → Modal A100 80GB"
              detail="PersonaPlex 7B + Mimi codec ×2 + energy-VAD + drip queue, all in-process. Audio never leaves the GPU container until it returns to the caller. Per-frame loop runs at 80ms."
            />
            <PlaneRow
              tier="CONTROL PLANE"
              tierColor="white"
              latency="async, off the hot loop"
              flow="Modal ⇄ Backend (FastAPI on Modal CPU) over JSON WebSocket"
              detail="TranscriptTurn, CallerTurnBoundary, ReadbackOutcome flow up. SpeakDirective, SilenceDirective, ReleaseDirective flow down. Slot extractor (Haiku) and intervention gate run here, hidden in the natural pause between turns."
            />
            <PlaneRow
              tier="PERSISTENCE PLANE"
              tierColor="grey"
              latency="real-time but non-blocking"
              flow="Backend → Supabase Postgres → Realtime channel → Next.js UI"
              detail="Calls, messages, events. Frontend reads via the anon key + read-only RLS. No direct backend ↔ frontend coupling — both ends just talk to the database."
            />
          </div>
        </section>

        {/* ─── Latency budget ─── */}
        <section className="glass arch-card" style={{ marginTop: 28 }}>
          <header className="arch-card-head">
            <h2 className="arch-card-title">Latency budget — full-duplex vs half-duplex</h2>
            <p className="arch-card-subtitle">
              Inca&rsquo;s &ldquo;feels human&rdquo; threshold is around 800ms round-trip. A
              half-duplex stack cannot clear that bar — the sequential floor
              is structurally too high.
            </p>
          </header>

          <div className="arch-bars">
            <LatencyBar
              label="Half-duplex stack"
              sublabel="Pipecat / Vapi / classic Twilio voice"
              segments={[
                { name: "Caller pause detection", ms: 750, color: "#5d5d5d" },
                { name: "ASR (speech → text)", ms: 350, color: "#7a7a7a" },
                { name: "LLM generate", ms: 1000, color: "#9a9a9a" },
                { name: "TTS synth + first chunk", ms: 500, color: "#b8b8b8" },
              ]}
              total="2.6 s typical · up to 3.5 s"
              isOver800={true}
            />

            <LatencyBar
              label="Full-duplex (Clio)"
              sublabel="PersonaPlex on Modal A100 80GB"
              segments={[
                { name: "Telco + WebRTC ingress", ms: 130, color: "#1a4a35" },
                { name: "Mimi encode", ms: 5, color: "#216044" },
                { name: "PersonaPlex inference (per frame)", ms: 80, color: "#2a8054" },
                { name: "Mimi decode", ms: 5, color: "#32a868" },
                { name: "Egress back to caller", ms: 330, color: "#52c47e" },
              ]}
              total="~550 ms measured · 450-650 ms band"
              isOver800={false}
            />
          </div>

          <p className="arch-bar-footnote">
            The PersonaPlex inference itself is only ~80 ms — the bulk of
            full-duplex round-trip is unavoidable telco + WebRTC transit.
            Half-duplex pays the same transit AND adds three sequential
            inference stages plus a turn-taking gap.
          </p>
        </section>

        {/* ─── Stack comparison ─── */}
        <section className="glass arch-card" style={{ marginTop: 28 }}>
          <header className="arch-card-head">
            <h2 className="arch-card-title">Stack comparison</h2>
            <p className="arch-card-subtitle">
              The architectural choices that determine whether a voice agent
              feels like a person or a phone tree.
            </p>
          </header>
          <div className="arch-table">
            <div className="arch-table-row arch-table-head">
              <span>Capability</span>
              <span>Half-duplex (typical SaaS)</span>
              <span>Full-duplex (Clio)</span>
            </div>
            {COMPARISON.map((row) => (
              <div key={row.capability} className="arch-table-row">
                <span className="arch-table-cap">{row.capability}</span>
                <span className="arch-table-half">{row.half}</span>
                <span className="arch-table-full">{row.full}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ─── Benchmarks (cards row) ─── */}
        <section style={{ marginTop: 28 }}>
          <h2
            className="arch-section-title"
            style={{ marginBottom: 16 }}
          >
            Measured on production
          </h2>
          <div className="arch-bench-grid">
            {BENCHMARKS.map((b) => (
              <div key={b.label} className="glass arch-bench">
                <span className="arch-bench-label">{b.label}</span>
                <span className="arch-bench-value">{b.value}</span>
                <span className="arch-bench-context">{b.context}</span>
              </div>
            ))}
          </div>
        </section>

        {/* ─── Why this matters ─── */}
        <section className="glass arch-card" style={{ marginTop: 28 }}>
          <header className="arch-card-head">
            <h2 className="arch-card-title">Why this matters</h2>
          </header>
          <div className="arch-why">
            <div className="arch-why-item">
              <span className="arch-why-num">01</span>
              <div>
                <h4 className="arch-why-h">No turn-taking gap</h4>
                <p className="arch-why-p">
                  PersonaPlex listens and speaks at the same time — the way
                  humans actually do. The agent can interject mid-utterance,
                  acknowledge while the caller is still speaking, and back
                  off when interrupted. No detection-then-decide-then-respond
                  pipeline.
                </p>
              </div>
            </div>
            <div className="arch-why-item">
              <span className="arch-why-num">02</span>
              <div>
                <h4 className="arch-why-h">Single inference path</h4>
                <p className="arch-why-p">
                  One model produces both text and audio per frame. Half-duplex
                  stacks compose three different models, each with its own
                  latency budget, error mode, and failure surface. Fewer parts
                  means lower latency, lower cost, and fewer ways to sound
                  like a robot.
                </p>
              </div>
            </div>
            <div className="arch-why-item">
              <span className="arch-why-num">03</span>
              <div>
                <h4 className="arch-why-h">Reasoner stays out of the voice loop</h4>
                <p className="arch-why-p">
                  Slot extraction, fact-checking, and gate decisions all run
                  on the backend asynchronously, hidden in the natural
                  caller-pause window. Sarah&rsquo;s next response never waits on
                  Haiku, Tavily, or the database.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ─── Source links ─── */}
        <footer style={{ marginTop: 56, textAlign: "center" }}>
          <p
            style={{
              fontSize: 11,
              color: "var(--text-tertiary)",
              fontFamily: "var(--font-mono), monospace",
              letterSpacing: "0.04em",
              margin: 0,
            }}
          >
            CLIO · BUILT ON PERSONAPLEX · INSPIRED BY MOSHI · MOSHIRAG · ASPIRIN · SEEDUPLEX
          </p>
          <p
            style={{
              fontSize: 11,
              color: "var(--text-quaternary)",
              fontFamily: "var(--font-mono), monospace",
              letterSpacing: "0.04em",
              margin: "8px 0 0",
            }}
          >
            Big Berlin Hack 2026 · v1.0 · beta
          </p>
        </footer>
      </main>
    </>
  );
}

// ─── Components ─────────────────────────────────────────────────────

function PlaneRow({
  tier,
  tierColor,
  latency,
  flow,
  detail,
}: {
  tier: string;
  tierColor: "green" | "white" | "grey";
  latency: string;
  flow: string;
  detail: string;
}) {
  return (
    <div className={`arch-plane arch-plane-${tierColor}`}>
      <div className="arch-plane-head">
        <span className="arch-plane-tier">{tier}</span>
        <span className="arch-plane-latency">{latency}</span>
      </div>
      <code className="arch-plane-flow">{flow}</code>
      <p className="arch-plane-detail">{detail}</p>
    </div>
  );
}

function LatencyBar({
  label,
  sublabel,
  segments,
  total,
  isOver800,
}: {
  label: string;
  sublabel: string;
  segments: { name: string; ms: number; color: string }[];
  total: string;
  isOver800: boolean;
}) {
  const totalMs = segments.reduce((acc, s) => acc + s.ms, 0);
  // Scale: max width corresponds to 3500ms (so half-duplex fills most, full-duplex shows ~17%)
  const SCALE_MAX = 3500;
  return (
    <div className="arch-bar-row">
      <div className="arch-bar-meta">
        <div>
          <div className="arch-bar-label">{label}</div>
          <div className="arch-bar-sublabel">{sublabel}</div>
        </div>
        <div className={`arch-bar-total ${isOver800 ? "is-bad" : "is-good"}`}>
          {total}
        </div>
      </div>
      <div className="arch-bar-track">
        {segments.map((s, i) => (
          <div
            key={i}
            className="arch-bar-seg"
            style={{
              width: `${(s.ms / SCALE_MAX) * 100}%`,
              background: s.color,
            }}
            title={`${s.name}: ${s.ms} ms`}
          >
            {s.ms >= 200 && (
              <span className="arch-bar-seg-label">{s.ms}ms</span>
            )}
          </div>
        ))}
        {/* 800ms threshold marker */}
        <div
          className="arch-bar-threshold"
          style={{ left: `${(800 / SCALE_MAX) * 100}%` }}
          title="800ms — Inca's 'feels human' threshold"
        >
          <span>800ms</span>
        </div>
      </div>
      <div className="arch-bar-legend">
        {segments.map((s, i) => (
          <div key={i} className="arch-bar-legend-item">
            <span
              className="arch-bar-legend-swatch"
              style={{ background: s.color }}
            />
            <span>
              {s.name} <span className="arch-bar-legend-ms">({s.ms}ms)</span>
            </span>
          </div>
        ))}
      </div>
      <div className="arch-bar-totalrow">
        Sum: <code>{totalMs}ms</code> one-way (round-trip is ~2× minus shared egress)
      </div>
    </div>
  );
}

// ─── Data ───────────────────────────────────────────────────────────

const COMPARISON = [
  {
    capability: "Round-trip latency",
    half: "1.5 – 3.5 s",
    full: "450 – 650 ms",
  },
  {
    capability: "Turn-taking",
    half: "Explicit silence detection (500-1000ms gap)",
    full: "None — continuous duplex",
  },
  {
    capability: "Listen while speaking",
    half: "No (one direction at a time)",
    full: "Yes (text + audio heads run in lock-step)",
  },
  {
    capability: "Pipeline stages",
    half: "ASR + LLM + TTS (3+ models, sequential)",
    full: "Single end-to-end model",
  },
  {
    capability: "Inference rate",
    half: "Per-utterance (after caller stops)",
    full: "Per-frame at 12.5Hz (every 80ms)",
  },
  {
    capability: "Interruption handling",
    half: "Reactive, often 1-2s late",
    full: "Frame-accurate (server-side audio mute)",
  },
  {
    capability: "Backpressure surface",
    half: "Each stage can stall the chain",
    full: "Single forward pass — no chain",
  },
  {
    capability: "Sounds like a robot at",
    half: "First sentence (the gap gives it away)",
    full: "Rarely — gap is sub-perceptual",
  },
];

const BENCHMARKS = [
  {
    label: "Per-frame inference",
    value: "~80 ms",
    context: "PersonaPlex 7B on A100 80GB at 12.5Hz",
  },
  {
    label: "Round-trip RTT",
    value: "450-650 ms",
    context: "Caller speaks → caller hears Sarah",
  },
  {
    label: "Cold start (first call)",
    value: "60-75 s",
    context: "Container join + persona snapshot rebuild",
  },
  {
    label: "Per-call ready time",
    value: "~50 ms",
    context: "Snapshot restore (vs 33s full re-prime)",
  },
  {
    label: "Slot extractor latency",
    value: "1-2 s",
    context: "Hidden in caller-pause window — async",
  },
  {
    label: "GPU memory peak",
    value: "33 GB / 80 GB",
    context: "A100 80GB · 41% utilization · 47GB headroom",
  },
];
