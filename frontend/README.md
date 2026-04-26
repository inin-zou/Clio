# Clio frontend — live call monitor

Next.js 15 App Router + Supabase Realtime. Three-pane layout:
- **Left:** recent calls (live status dot)
- **Middle:** chronological feed of caller/agent transcripts + slot updates + gate firings + readbacks
- **Right:** running FNOL state (slot path → value) + final FNOL JSON when the call ends

The backend writes every event to Supabase (`calls`, `messages`, `events`).
This page subscribes via the Supabase Realtime client — no SSE, no
backend coupling beyond the database.

## Local dev

```bash
cd frontend
cp .env.local.example .env.local
# fill in NEXT_PUBLIC_SUPABASE_URL + NEXT_PUBLIC_SUPABASE_ANON_KEY
# (same values as the parent .env's SUPABASE_URL + SUPABASE_ANON_KEY)

pnpm install   # or npm / bun
pnpm dev       # → http://localhost:3000
```

## Deploy to Vercel

```bash
cd frontend
vercel          # first run links the project
vercel --prod   # promote
```

Add the two `NEXT_PUBLIC_SUPABASE_*` vars in the Vercel project settings
(or via `vercel env`).

## Schema dependency

Apply `db/supabase_schema.sql` (in repo root) once via the Supabase
dashboard's SQL Editor. The page expects:
- `calls(id, started_at, ended_at, caller_phone, policy_number, fnol, reason_ended)`
- `messages(id, call_id, role, source, text, timestamp)`
- `events(id, call_id, type, payload, timestamp)`

All three tables must be in the `supabase_realtime` publication (the
schema script handles this). Anon key needs `select` policy on each.
