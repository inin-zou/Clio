# LiveKit Cloud SIP setup (one-time)

The Twilio webhook returns TwiML that `<Dial><Sip>`s into a SIP URI of the
form `sip:clio-<call_id>@<LIVEKIT_SIP_URI>`. For LiveKit Cloud to accept
that and route the caller into a room named `clio-<call_id>`, you need:

1. An **inbound SIP trunk** (whitelists Twilio).
2. A **dispatch rule** matching the SIP URI user-part → room name.

Run these commands once per LiveKit project (the trunk + rule persist).

## Prereqs

```bash
# LiveKit CLI
brew install livekit-cli

# Auth via the same project keys backend uses
export LIVEKIT_URL=$(grep ^LIVEKIT_URL .env | cut -d= -f2)
export LIVEKIT_API_KEY=$(grep ^LIVEKIT_API_KEY .env | cut -d= -f2)
export LIVEKIT_API_SECRET=$(grep ^LIVEKIT_API_SECRET .env | cut -d= -f2)

# Twilio info we'll need
export TWILIO_NUMBER=$(grep ^TWILIO_PHONE_NUMBER .env | cut -d= -f2)
```

## 1. Create the inbound SIP trunk

```bash
cat > /tmp/clio-trunk.json <<EOF
{
  "name": "clio-twilio-inbound",
  "numbers": ["${TWILIO_NUMBER}"]
}
EOF

lk sip inbound-trunk create /tmp/clio-trunk.json
```

Note the `sip_trunk_id` (e.g. `ST_xxxxxx`) and the **SIP URI** that
LiveKit prints — it'll look like `<project-subdomain>.sip.livekit.cloud`.
Put that value in `.env` as `LIVEKIT_SIP_URI`.

## 2. Create the dispatch rule

`type: individual` with `roomPrefix: clio-` makes LiveKit create
(or reuse) a room named `clio-<sip-user-part>` for each inbound call.
The Twilio webhook spawns Modal to join that exact room name.

```bash
cat > /tmp/clio-dispatch.json <<EOF
{
  "name": "clio-room-per-call",
  "trunk_ids": ["<paste sip_trunk_id from step 1>"],
  "rule": {
    "dispatchRuleIndividual": {
      "roomPrefix": "clio-"
    }
  }
}
EOF

lk sip dispatch-rule create /tmp/clio-dispatch.json
```

## 3. Verify

```bash
lk sip inbound-trunk list
lk sip dispatch-rule list
```

## Twilio side

In the Twilio Console → Phone Numbers → your DID → Voice & Fax:

- **A Call Comes In** → Webhook → `https://<BACKEND_PUBLIC_HOST>/twilio/voice` (POST)
- **Call Status Changes** → `https://<BACKEND_PUBLIC_HOST>/twilio/status` (POST)

For local dev with ngrok:

```bash
ngrok http 8000
# copy the https URL → use that as <BACKEND_PUBLIC_HOST>
# also set BACKEND_PUBLIC_WS_URL=wss://<host>  in .env
```

If you leave `TWILIO_SKIP_SIGNATURE_VALIDATION=1` for dev (the
default in `.env.example`), Twilio's signed URL doesn't have to match
exactly. For prod, unset that and make sure Twilio's webhook URL exactly
matches what the backend sees in `request.url`.
