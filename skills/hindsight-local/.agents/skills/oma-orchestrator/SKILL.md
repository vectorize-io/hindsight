---
name: oma-orchestrator
description: Automated multi-agent orchestrator that spawns CLI subagents in parallel, coordinates via MCP Memory, and monitors progress. Use for orchestration, parallel execution, and automated multi-agent workflows.
---

# Orchestrator - Automated Multi-Agent Coordinator

## Scheduling

### Goal
Automatically orchestrate multi-agent execution with task decomposition, native/fallback dispatch, memory coordination, progress monitoring, verification, QA cross-review, retry, and result collection.

### Intent signature
- User asks to orchestrate, run in parallel, automate multi-agent execution, or coordinate full-stack work end to end.
- Task requires multiple specialist agents and a persistent review/remediation loop.

### When to use
- Complex feature requires multiple specialized agents working in parallel
- User wants automated execution without manually spawning agents
- Full-stack implementation spanning backend, frontend, mobile, and QA
- User says "run it automatically", "run in parallel", or similar automation requests

### When NOT to use
- Simple single-domain task -> use the specific agent directly
- User wants step-by-step manual control -> use oma-coordination
- Quick bug fixes or minor changes

### Expected inputs
- Complex feature or workflow request
- Project config, model/vendor routing, agent types, task constraints, and workspace/session needs
- Acceptance criteria and verification expectations

### Expected outputs
- Orchestrator session state, task board, progress files, result files, and final summary
- Specialist agent outputs after mechanical checks, automated verify, and QA cross-review
- Review history and retry/remediation status when loops fail

### Dependencies
- `.agents/oma-config.yaml`, `.codex/agents/*.toml`, `.gemini/agents/*.md`, or fallback `oma agent:spawn`
- Memory provider config, subagent prompt template, scripts, task templates, verify script, and session metrics

### Control-flow features
- Branches by vendor/native dispatch availability, priority tiers, agent completion/failure, verification status, QA verdict, retry limits, and clarification debt
- Spawns processes/agents and reads/writes memory/result files
- Blocks termination until persistent workflows complete

## Structural Flow

### Entry
1. Resolve agent vendor routing and runtime dispatch path.
2. Decompose request into priority-tiered tasks.
3. For each task, classify into one or more `domain_tags` by matching against the `Intent signature` block of each installed `.agents/skills/oma-*/SKILL.md`. Tasks that match no domain confidently inherit the union of their parent feature's tags.
4. Build a per-task `exposed_skill_set` = skills whose name is in `domain_tags`. If `|exposed_skill_set| < 2` after classification, fall back to the full installed set (flat exposure) and record `exposure_fallback: true` in the task board.
5. Create session memory and task board with `exposed_skill_set` and `exposure_fallback` per task.

### Scenes
1. **PREPARE**: Plan, setup session ID, and initialize memory files.
2. **ACT**: Spawn agents by priority tier within parallelism limits.
3. **VERIFY**: Run self-check, `oma verify`, and QA cross-review loop.
4. **RECOVER**: Retry failed agents with review history when limits allow.
5. **FINALIZE**: Collect result files, compile summary, and clean progress files.

### Transitions
- If native dispatch is available for current runtime/vendor, use it.
- If vendors differ or native path is unavailable, use fallback spawn.
- If verify or QA fails, feed feedback back to the implementation agent.
- If review loop limits are exceeded, report review history and quality warning.
- If a task's `exposed_skill_set` excludes a skill that a recovered failure indicates was needed, re-classify the task and re-dispatch with the expanded set rather than retrying against the original narrow set.

### Failure and recovery
- Retry failed agents up to configured limits.
- Re-spawn with review history when review loop is exhausted.
- Pause or request re-specification when clarification debt thresholds are exceeded.

### Exit
- Success: all tasks complete, verify/review pass, and results are summarized.
- Partial success: failed agents, exhausted review loops, or clarification debt are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read config and task context | `READ` | oma config, routing, request |
| Classify task into domain tags | `INFER` | task text vs each skill's `Intent signature` |
| Compute exposed skill set | `SELECT` | intersection of domain tags and installed skills |
| Select dispatch path | `SELECT` | Native vs fallback |
| Write session state | `WRITE` | task board and memory files |
| Spawn agents | `CALL_TOOL` | native CLI or `oma agent:spawn` |
| Poll progress | `READ` | progress/result files |
| Run verification | `CALL_TOOL` | `oma verify`, tests, QA |
| Update retry state | `UPDATE_STATE` | loop counters and CD metrics |
| Report final result | `NOTIFY` | compiled summary |

### Tools and instruments
- Native CLI subagent dispatch, fallback spawn scripts, memory tools, verify script, QA agent
- Session metrics, prompt templates, task templates

### Canonical command path
```bash
oma agent:spawn <agent-type> "<task>" <session-id> -w <workspace>
oma verify <agent-type> --workspace <workspace> --json
```

When native runtime dispatch is available, prefer the runtime-specific native path listed in this skill before falling back to `oma agent:spawn`.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | Session, task-board, progress, result, config files |
| `PROCESS` | Agent CLI processes and verify scripts |
| `MEMORY` | Session state and clarification debt |
| `CODEBASE` | Workspaces owned by spawned agents |

### Preconditions
- Task is decomposable into specialist agent work.
- Runtime/vendor dispatch path or fallback exists.

### Effects and side effects
- Spawns agents and writes session/progress/result artifacts.
- May cause code changes through specialist agents.
- May trigger iterative review and retries.

### Guardrails
1. Orchestrate per-agent dispatch from the project configuration before spawning any agent.
2. If `target_vendor === current_runtime_vendor` and the runtime has a verified native path, use native dispatch.
3. Otherwise fall back to `oma agent:spawn`.
4. Never exceed the configured parallelism or retry limits.
5. Keep session state, task-board state, progress files, and result files aligned throughout the run.
6. Domain gating must be soft: prefer a narrower `exposed_skill_set`, but fall back to flat exposure when classification confidence is low rather than starving a task of a required specialist.

Current native executor paths:
- Claude Code: `claude --agent <agent>`
- Codex CLI: `codex exec "@agent ..."` using `.codex/agents/*.toml`
- Gemini CLI: `gemini -p "@agent ..."` using `.gemini/agents/*.md`

Vendor-specific execution protocols are injected automatically for fallback CLI runs.

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| MAX_PARALLEL | 3 | Max concurrent subagents |
| MAX_RETRIES | 2 | Retry attempts per failed task |
| POLL_INTERVAL | 30s | Status check interval |
| MAX_TURNS (impl) | 20 | Turn limit for backend/frontend/mobile |
| MAX_TURNS (review) | 15 | Turn limit for qa/debug |
| MAX_TURNS (plan) | 10 | Turn limit for pm |

### Memory Configuration

Memory provider and tool names are configurable via `mcp.json`:
```json
{
  "memoryConfig": {
    "provider": "serena",
    "basePath": ".serena/memories",
    "tools": {
      "read": "read_memory",
      "write": "write_memory",
      "edit": "edit_memory"
    }
  }
}
```

### Workflow Phases

**PHASE 1 - Plan**: Analyze request -> decompose tasks -> generate session ID
**PHASE 1.5 - Domain gate**: For each task, intersect `Intent signature` matches across installed skills to derive `exposed_skill_set`. Record `exposure_fallback: true` when the intersection is too small to be useful and the flat library is used instead.
**PHASE 2 - Setup**: Use memory write tool to create `orchestrator-session.md` + `task-board.md` (include `exposed_skill_set` per task)
**PHASE 3 - Execute**: Spawn agents by priority tier (never exceed MAX_PARALLEL); inject only `exposed_skill_set` into each subagent's available specialist list
**PHASE 4 - Monitor**: Poll every POLL_INTERVAL; handle completed/failed/crashed agents
**PHASE 4.5 - Verify**: Run `oma verify {agent-type}` per completed agent
**PHASE 5 - Collect**: Read all `result-{agent}-{sessionId}.md`, compile summary, cleanup progress files

See `resources/subagent-prompt-template.md` for prompt construction.
See `resources/memory-schema.md` for memory file formats.

### Memory File Ownership

| File | Owner | Others |
|------|-------|--------|
| `orchestrator-session.md` | orchestrator | read-only |
| `task-board.md` | orchestrator | read-only |
| `progress-{agent}[-{sessionId}].md` | that agent | orchestrator reads |
| `result-{agent}[-{sessionId}].md` | that agent | orchestrator reads |

### Agent-to-Agent Review Loop (PHASE 4.5)

After each agent completes, enter an iterative review loop, not a single-pass verification.

### Loop Flow

```
Agent completes work
    ↓
[1] Mechanical Self-Check: lint, type-check, tests, diff scope
    ↓
[2] Verify: Run `oma verify {agent-type} --workspace {workspace}`
    ↓ FAIL → Agent receives feedback, fixes, back to [1]
    ↓ PASS
[3] Cross-Review: QA agent reviews the changes
    ↓ FAIL → Agent receives review feedback, fixes, back to [1]
    ↓ PASS
Accept result
```

### Step Details

**[1] Mechanical Self-Check** (formerly "Self-Review"):
Before requesting external review, the implementation agent must:
- Run lint, type-check, and tests in the workspace
- Verify only planned files were modified (diff scope check)
- Fix any mechanical failures (compile errors, test failures)

**Quality judgment is NOT performed in this step.**
Design quality, architecture alignment, and acceptance criteria satisfaction
are evaluated exclusively in [3] Cross-Review by the QA agent.
Reason: Self-evaluation bias causes agents to consistently overrate their own output
(ref: Anthropic harness design research).

**[2] Automated Verify**:
```bash
oma verify {agent-type} --workspace {workspace} --json
```
- **PASS (exit 0)**: Proceed to cross-review
- **FAIL (exit 1)**: Feed verify output back to the agent as correction context

**[3] Cross-Review**: Spawn QA agent to review the changes:
- QA agent reads the diff, runs checks, evaluates against acceptance criteria
- If `docs/CODE-REVIEW.md` exists, QA agent uses it as the review checklist
- QA agent outputs: PASS (with optional nits) or FAIL (with specific issues)
- On FAIL: issues are fed back to the implementation agent for fixing

### Loop Limits

| Counter | Max | On Exceeded |
|---------|-----|-------------|
| Self-check + fix cycles | 3 | Escalate to cross-review regardless |
| Cross-review rejections | 2 | Report to user with review history |
| Total loop iterations | 5 | Force-complete with quality warning |

### Review Feedback Format

When feeding review results back to the implementation agent:
```
## Review Feedback (iteration {n}/{max})
**Reviewer**: {self / verify / qa-agent}
**Verdict**: FAIL
**Issues**:
1. {specific issue with file and line reference}
2. {specific issue}
**Fix instruction**: {what to change}
```

This replaces single-pass verification. Most "nitpicking" should happen agent-to-agent.
Human review is reserved for final approval, not catching lint errors.

### Retry Logic (after review loop exhaustion)
- 1st retry: Re-spawn agent with full review history as context
- 2nd retry: Re-spawn with "Try a different approach" + review history
- Final failure: Report to user with complete review trail, ask whether to continue or abort

### Clarification Debt (CD) Monitoring

Track user corrections during session execution. See `../_shared/core/session-metrics.md` for full protocol.

### Event Classification
When user sends feedback during session:
- **clarify** (+10): User answering agent's question
- **correct** (+25): User correcting agent's misunderstanding
- **redo** (+40): User rejecting work, requesting restart

### Threshold Actions
| CD Score | Action |
|----------|--------|
| CD >= 50 | **RCA Required**: QA agent must add entry to `lessons-learned.md` |
| CD >= 80 | **Session Pause**: Request user to re-specify requirements |
| `redo` >= 2 | **Scope Lock**: Request explicit allowlist confirmation before continuing |

### Recording
After each user correction event:
```
[EDIT]("session-metrics.md", append event to Events table)
```

At session end, if CD >= 50:
1. Include CD summary in final report
2. Trigger QA agent RCA generation
3. Update `lessons-learned.md` with prevention measures



## References
- Prompt template: `resources/subagent-prompt-template.md`
- Memory schema: `resources/memory-schema.md`
- Config: `config/cli-config.yaml`
- Scripts: `scripts/spawn-agent.sh`, `scripts/parallel-run.sh`, `scripts/verify.sh`
- Task templates: `templates/`
- Skill-to-agent mapping: `../_shared/core/skill-routing.md`
- Verification: `scripts/verify.sh <agent-type>`
- Session metrics: `../_shared/core/session-metrics.md`
- API contract template (SSOT): `../_shared/core/api-contracts/template.md`; read generated contracts from `.agents/results/api-contracts/` (run artifact) or `docs/plans/contracts/` (durable spec)
- Context loading: `../_shared/core/context-loading.md`
- Difficulty guide: `../_shared/core/difficulty-guide.md`
- Reasoning templates: `../_shared/core/reasoning-templates.md`
- Clarification protocol: `../_shared/core/clarification-protocol.md`
- Context budget: `../_shared/core/context-budget.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
