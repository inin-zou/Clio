# Clio — Design Docs

Technical analysis and architecture decisions for the Inca hackathon voice agent project.

## Documents

- **[architecture.md](architecture.md)** — Runtime system architecture: LiveKit + Twilio + PersonaPlex on Modal, two-channel WebSocket protocol, latency budget, deployment topology. **Read this first.**
- **[roadmap.md](roadmap.md)** — Three-MVP plan (injection → inbound call → ASPIRin fine-tune) + ASPIRin compatibility analysis + decision log.
- **[fnol-schema.md](fnol-schema.md)** — FNOL data model (Inca's spec mapped to our two-layer schema), slot tiers, conditional tree, mock policy DB. The Reasoner's contract.
- **[architecture-decision.md](architecture-decision.md)** — Why we chose this path. Records the decisions (no training, public injection API, passive checklist) and the alternatives we ruled out.
- **[moshirag-analysis.md](moshirag-analysis.md)** — Deep dive on MoshiRAG. What it does, why we don't replicate it, what we take conceptually (lead/body/tail structure, three-LLM data construction).
- **[aspirin-analysis.md](aspirin-analysis.md)** — Deep dive on ASPIRin. RL framework that decouples timing from content. We externalize the same split into our Reasoner; no training.

## Quick orientation

- **What we're building, when** → `roadmap.md`
- **What runs where** → `architecture.md`
- **Why we chose this** → `architecture-decision.md`
- **Background research** → `moshirag-analysis.md`, `aspirin-analysis.md`
