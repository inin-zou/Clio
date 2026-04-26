-- Clio — Supabase schema for live conversation persistence + UI realtime.
--
-- Apply once per Supabase project via Dashboard → SQL Editor → New query.
-- Idempotent: safe to re-run if you tweak something (drops + recreates).
--
-- Backend writes via the service-role key (bypasses RLS).
-- Browser/Next.js UI subscribes via the anon key + read-only RLS policies.

-- ─── Tables ──────────────────────────────────────────────────────────────

-- One row per call. Key is the call_id minted by the Twilio webhook.
create table if not exists calls (
  id              text primary key,
  started_at      timestamptz default now(),
  ended_at        timestamptz,
  caller_phone    text,
  policy_number   text,
  fnol            jsonb,
  reason_ended    text
);

-- One row per transcript turn — both Scribe (caller) and PersonaPlex (agent).
create table if not exists messages (
  id        bigserial primary key,
  call_id   text not null references calls(id) on delete cascade,
  role      text not null,            -- 'caller' | 'agent'
  source    text,                     -- 'scribe' | 'personaplex'
  text      text not null,
  timestamp timestamptz default now()
);

-- One row per non-message event — slot fills, gate firings, readbacks,
-- session lifecycle. payload is the full event JSON for forward compat.
create table if not exists events (
  id        bigserial primary key,
  call_id   text not null references calls(id) on delete cascade,
  type      text not null,            -- 'slot_update' | 'gate_fired' | 'readback' | 'lifecycle'
  payload   jsonb not null,
  timestamp timestamptz default now()
);

-- ─── Indexes ─────────────────────────────────────────────────────────────

create index if not exists messages_call_id_ts on messages (call_id, timestamp);
create index if not exists events_call_id_ts on events (call_id, timestamp);
create index if not exists calls_started_at on calls (started_at desc);

-- ─── Realtime ────────────────────────────────────────────────────────────
-- Subscribers (the Next.js UI) get postgres_changes events for these tables.

alter publication supabase_realtime add table calls;
alter publication supabase_realtime add table messages;
alter publication supabase_realtime add table events;

-- ─── Row-level security ──────────────────────────────────────────────────
-- For the hackathon: read-only public access. Anyone with the anon key can
-- read every call's transcript. WARNING: this exposes call data publicly.
-- Replace with auth-scoped policies before any non-demo use.

alter table calls    enable row level security;
alter table messages enable row level security;
alter table events   enable row level security;

drop policy if exists "anon read calls"    on calls;
drop policy if exists "anon read messages" on messages;
drop policy if exists "anon read events"   on events;

create policy "anon read calls"    on calls    for select using (true);
create policy "anon read messages" on messages for select using (true);
create policy "anon read events"   on events   for select using (true);
