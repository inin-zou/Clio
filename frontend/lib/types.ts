export type Call = {
  id: string;
  started_at: string;
  ended_at: string | null;
  caller_phone: string | null;
  policy_number: string | null;
  fnol: Record<string, unknown> | null;
  reason_ended: string | null;
};

export type Message = {
  id: number;
  call_id: string;
  role: "caller" | "agent";
  source: "scribe" | "personaplex" | null;
  text: string;
  timestamp: string;
};

export type EventRow = {
  id: number;
  call_id: string;
  type: "slot_update" | "gate_fired" | "readback" | "lifecycle";
  payload: Record<string, unknown>;
  timestamp: string;
};
