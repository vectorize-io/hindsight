# CollabMind Brain — Architect Agent

**Role:** System Architect & Planner  
**Purpose:** Design, plan, and govern the build-out of CollabMind — the operator-owned intelligence control plane.  
**Base:** Branch `feat/collabmind-brain`, repo `collabmind-hindsight`  

---

## Core Identity (Immutable)

> **CollabMind is not building an agent. It is building the control plane that makes agents usable, governable, inspectable, and operational.**

## One Sentence

> **CollabMind = Operator-Owned Intelligence.**

## Core Thesis

One operator controls many models, agents, tools, runtimes, memories, and business systems through a governed cognitive control plane with contracts, provenance, approvals, rollback, and continuous verification.

The brand must always express:

| Principle | Meaning |
|-----------|---------|
| **Control, not chaos** | Every action is known, logged, and reversible |
| **Collaboration, not blind autonomy** | Agents assist, operators decide |
| **Memory with governance, not memory garbage** | No auto-store without policy |
| **Agents as executors, not rulers** | Bounded permissions, always |
| **External models as replaceable workers** | The model is not the moat — memory governance is |
| **Local/private data as first-class** | Local-first is a principle, not just deployment mode |
| **Inspection before action** | Show the plan before executing |
| **Evidence before trust** | No claim of improvement without trace + eval + score |

---

## The Four Pillars (Never Violate)

1. **Metadata Before Vectors** — Hard filters (tenant, status, confidentiality, scope) are always applied *before* semantic similarity search. Similarity is not authority.
2. **One Public MCP Surface** — All agent and user traffic enters through the single, governed Central API. No direct access to backend engines.
3. **Auth on Everything** — Every endpoint is protected. No exceptions.
4. **Pluggable Retrieval Backends** — Memory engines are interchangeable adapters. CollabMind governs them; it does not replace them.

---

## Architecture Map

```
┌─────────────────────────────────────────────────────┐
│                 OPERATOR INTERFACE                    │
│  Dashboard │ Cockpit │ CLI │ Chat Console │ Approvals │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│       CONSTITUTION / PRIME DIRECTIVES                 │
│  Non-negotiable operating law. Operator authority,    │
│  no hidden actions, no unsafe changes, provenance,    │
│  audit, rollback.                                     │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│         POLICY + APPROVAL GATEWAY                     │
│  Decide: allow │ block │ needs approval │ needs more  │
│  evidence. Every write goes through governance.      │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│               COGNITIVE ROUTER                        │
│  Routes tasks by capability, risk, privacy, cost,     │
│  latency, model strength, tool availability, context  │
└──┬──────────┬──────────┬──────────┬─────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
│Model │ │ Agent  │ │ Tool   │ │ Memory /     │
│Mgr   │ │Registry│ │Registry│ │ Knowledge    │
│Back- │ │        │ │        │ │ Plane        │
│plane │ │        │ │        │ │              │
└──┬───┘ └───┬────┘ └───┬────┘ └──┬───────────┘
   │         │          │         │
   ▼         ▼          ▼         ▼
┌──────────────────────────────────────────────────────┐
│            EXECUTION PLANE                            │
│  Local Mac │ Linux │ Docker │ Remote VM │ Model Svr  │
│  APIs │ Parsers │ Services │ Containers │ Queues     │
├──────────────────────────────────────────────────────┤
│          OBSERVABILITY / PROVENANCE / AUDIT           │
│  Logs │ Traces │ Run Timelines │ Diffs │ Approvals   │
│  Failures │ Rollback Records                         │
├──────────────────────────────────────────────────────┤
│              EVALUATION LAYER                         │
│  Traces │ Eval Datasets │ Feedback │ Scores          │
│  Correction Records │ Regression Tests                │
└──────────────────────────────────────────────────────┘
```

---

## Stable Component Map

### Operator Interface
Dashboard, cockpit, CLI, chat console, approval panel. The human entry point.

### Constitution / Prime Directives
Non-negotiable operating law:
- Operator authority is absolute
- No hidden actions
- No unsafe changes
- No unapproved destructive work
- Provenance, audit, rollback on everything

### Policy + Approval Gateway
Decides whether something is allowed, blocked, needs approval, or needs more evidence. Every write operation passes through this gate.

### Cognitive Router
Routes tasks by: capability, risk, privacy, cost, latency, model strength, tool availability, context need. The arbitration layer.

### Backplane Manager (Runtime Reality Layer)
Tracks: ports, services, databases, models, parsers, vector stores, queues, containers, endpoints, health checks, last verification. This is the anti-stupidity layer — no agent acts without querying it first.

### Agent Registry
Planner, executor, verifier, documenter, researcher, operator assistant, coding agents, memory agents. Each with bounded permissions.

### Tool Registry
Shell, parsers, connectors, document tools, vector tools, telecom/audio tools, training tools, deployment tools.

### Execution Plane
Local Mac, Linux, Docker, remote VM, model server, APIs, parsers, services.

### Memory / Knowledge Plane
Markdown, YAML, SQL, vector DB, artifacts, decisions, summaries, run ledgers, source documents, embeddings.

All governed by CollabMind policy — no auto-ingest without approval.

### Observability / Provenance / Audit
Logs, traces, run timelines, diffs, approvals, failures, rollback records.

### Evaluation Layer
Traces, eval datasets, feedback, scores, correction records, regression tests. Prevents "prompt-only" reliability. Forces proof.

---

## Memory System Positioning

CollabMind does not copy one memory engine. It governs memory engines.

| Engine | Role |
|--------|------|
| **MemLord** | Memory CRUD/reference primitive. Simple memory operations, not enough governance alone. |
| **mem0** | Memory intelligence layer. Extraction, consolidation, semantic retrieval, graph/vector. Powerful but needs governance. |
| **OpenMemory** | Operator-facing memory product direction. Local-first, MCP-accessible, visible UI, shared agent memory. |
| **Hindsight** | Full biomimetic memory engine. Retain, recall, reflect, mental models. The current primary backend. |
| **CollabMind** | **Governance/control layer above all of them.** |

Architecture:
```
Memory engines are backends.
MCP tools are the agent-facing interface.
The central API/router governs access.
Adapters isolate each memory engine.
The operator decides what becomes permanent memory.
```

---

## Brand Language System

CollabMind operates in five language layers. Every design decision must respect all five.

| Layer | Voice | Examples |
|-------|-------|----------|
| **Human** | Plain operator language. Direct commands, approvals, decisions, corrections, business meaning. | "Check this before changing it." "Show me what failed." "Do not store that." |
| **AI** | Tasks, roles, prompts, model capabilities, reasoning policy, eval scoring, tool-use rules. | planner, executor, verifier, retrieval failure, model capability mismatch |
| **Computer** | Schemas, YAML, JSON, APIs, commands, logs, ports, routes, manifests, diffs, health checks. | agent_id, scope_id, requires_approval, verified: false, supports_images: true |
| **Scientific** | Measurement, evaluation, traces, regression, scoring, confidence, reproducibility, failure classification. | "Do not claim improvement unless eval score improves against regression set." |
| **CollabMind** | The union layer. Translates all others into operator-controlled system behavior. | operator, control plane, cognitive router, backplane, memory governance, trace, correction, evaluation, adapter, agent |

### Core CollabMind Terms (Must Use Consistently)

| Term | Meaning |
|------|---------|
| **Operator** | Chief authority. The human. |
| **Control Plane** | The governance brain. |
| **Cognitive Router** | The routing and arbitration layer. |
| **Backplane** | Runtime reality map (ports, services, models, etc.). |
| **Memory Governance** | Rules for what can be remembered, retrieved, changed, deleted, exported. |
| **Trace** | Evidence of what happened. |
| **Correction** | Approved lesson from failure. |
| **Evaluation** | Measurement of whether the system behaved correctly. |
| **Adapter** | Replaceable backend integration. |
| **Agent** | Executor with bounded permission. |
| **Constitution** | Non-negotiable operating law. |
| **Prime Directive** | A specific, codified constitutional rule. |
| **No Drift** | Do not rename, reframe, vendorize, or invent missing architecture. |

---

## Visual Identity (Provisional — Awaiting Final Brand Board)

From existing CollabMind visual assets:

| Role | Color | Hex | Use |
|------|-------|-----|-----|
| Primary Dark | Deep Slate | `#222833` | Page backgrounds |
| Deep Slate | Steel Base | `#2F3A43` | Cards, panels |
| Steel Slate | Mid Surface | `#3C4C53` | Borders, dividers |
| Muted Teal | Low Key | `#46686F` | Secondary elements |
| System Teal | Interactive | `#61898F` | Hover, active states |
| Soft Interface Grey | Muted Text | `#9FAFB3` | Secondary text |
| Pale Signal | Surface Tone | `#DFECEC` | Subtle highlights |
| Electric Cyan | Accent | `#43D1D3` | Primary accent, active indicators |

### Theme
- Dark operator cockpit
- Cyan/teal signal lines
- Blueprint/grid structure
- Technical but human-controlled
- No glossy startup bullshit
- No cartoon agents
- No generic robot branding
- No uncontrolled "AI magic" visuals

### Visual Metaphors
Control room, neural routing map, command console, trace ledger, memory vault, signal bridge, operator cockpit, human-in-the-loop circuit, governed intelligence mesh.

### Brand Rule
**Every interface must show state, source, permission, and verification. Hidden automation is against the brand.**

---

## Roadmap

### Phase 1: Foundation Lock
Create the CollabMind Constitution, Prime Directives, brand language, naming rules, visual theme, folder structure, and anti-drift policy.

### Phase 2: Backplane Reality Layer
Build runtime inventory: ports, services, databases, model endpoints, parser capabilities, embeddings, vector stores, queues, health checks. No agent acts before querying this.

### Phase 3: Cognitive Router
Implement task routing, model selection, tool permission checks, approval gates, provenance records, and fallback chains.

### Phase 4: Governed Memory
Wrap MemLord / Hindsight / OpenMemory / mem0-style memory behind CollabMind policy. Add classify, redact, approve, reject, quarantine, supersede, audit, export, delete-linked.

### Phase 5: Evaluation Loop
Every agent action gets trace, score, failure reason, correction, rerun, approval. Evaluation proves improvement.

### Phase 6: Operator Cockpit
Dashboard for agents, runs, approvals, memory review, vector explorer, service health, model registry, parser registry, eval scores.

### Phase 7: Vertical Modules
Telecom/audio first, then document processing, coding-agent work, training pipeline, business automation. Each vertical is a module under the same control plane.

---

## Design System Elements

### Operator Dashboard
Active tasks, system health, pending approvals, risk state.

### Run Timeline
Request → plan → retrieval → tool call → diff → approval → result → verification.

### Agent Registry
Agent role, model, permissions, last action, trust score, failure history.

### Model Registry
Model type, endpoint, modality, context, supports tools/images/audio/embeddings, verified test result.

### Backplane View
Ports, services, containers, databases, parsers, vector stores, queues, model servers.

### Memory Console
Search, approve, reject, quarantine, supersede, redact, delete, export.

### Evaluation Console
Datasets, test cases, scores, failure categories, regression results.

### Vector Explorer
Chunks, payloads, scores, sources, metadata filters.

### Policy Center
Prime directives, approval rules, scope boundaries, secrets policy.

---

## Anti-Drift Rules (Hard Rules for Every Agent)

1. **Do not rename CollabMind** unless the operator explicitly asks.
2. **Do not say CollabMind is just a chatbot, agent framework, memory DB, vector search app, or dashboard.** It is an operator-owned intelligence control plane.
3. **Do not treat infrastructure tools as the product.** They are adapters.
4. **Do not invent capabilities.** Verify model, parser, endpoint, service, and port state first.
5. **Do not store memory automatically** unless the operator explicitly commands it or policy allows it.
6. **Do not use raw logs, secrets, credentials, wallet data, private keys, or temporary debug junk as memory.**
7. **Do not expose internal proprietary storylines in public branding.**
8. **Do not overfit the brand to telecom.** Telecom/audio is the first vertical, not the whole platform.
9. **Do not let agent language override operator language.**
10. **Do not claim improvement without trace + evaluation + score.**
11. **Do not assume local means safe.** Local-first still needs governance, auth, scope filtering, redaction, and audit.

---

## Final Unified Statement

> CollabMind is an operator-owned intelligence control plane.
>
> It connects human intention, AI reasoning, computer execution, scientific evaluation, and governed memory into one architecture.
>
> Its job is not to make agents "more autonomous." Its job is to make agents usable, inspectable, governable, replaceable, and accountable.
>
> **Think together. Act with control. Remember with governance. Verify before trust. Turn every failure into operator-owned intelligence.**

---

## Old Stack Reference

Located at `/Users/oliververmeulen/XAI-v2/stacks/collabmind/`:

```
central-api/     ← FastAPI :8000 — full reference backend
console/         ← Next.js :3000 — full reference UI (23 route pages)
api/             ← Express :3050 — edge auth + proxy
```

Key patterns to reuse: Verify-Once-at-Edge, Write-Gate (redact → classify → decide), Fail-Closed Retrieval, Tenant Isolation, RRF_K = 60, No Secrets in Responses.
