"use client";

import { createClient, SupabaseClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

let _client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (!url || !anonKey) {
    throw new Error(
      "Supabase env not set. Define NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local",
    );
  }
  if (!_client) {
    _client = createClient(url, anonKey, {
      realtime: { params: { eventsPerSecond: 20 } },
    });
  }
  return _client;
}
