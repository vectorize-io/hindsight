# Lessons Learned

A cumulative lesson repository across sessions. All agents reference this file at execution start.
QA Agent and Orchestrator add new lessons after session completion.

---

## Skill → Domain Mapping

| Skill | Primary Section | Secondary Section |
|-------|-----------------|-------------------|
| backend-agent | Backend | Cross-Domain |
| frontend-agent | Frontend | Cross-Domain |
| mobile-agent | Mobile | Cross-Domain |
| debug-agent | Debug | Relevant domain (Backend/Frontend/Mobile) |
| qa-agent | QA / Security | Relevant domain being reviewed |
| pm-agent | Cross-Domain | All sections (for planning awareness) |
| orchestrator | Cross-Domain | All sections (for coordination) |

---

## How to Use

### Reading (All Agents)
- At Complex task start: Read your domain section to prevent repeating mistakes
- Medium tasks: Reference if related items exist
- Simple tasks: Can skip

### Writing (QA Agent, Orchestrator)
Add in the following format after session completion:
```markdown
### {YYYY-MM-DD}: {agent-type} - {one-line summary}
- **Problem**: {what went wrong}
- **Cause**: {why it happened}
- **Solution**: {how it was fixed}
- **Prevention**: {how to prevent in the future}
```

---

## Backend Lessons

> This section is referenced by backend-agent, debug-agent (for backend bugs).

### Initial Lessons (Recorded at Project Setup)
- **Use SQLAlchemy 2.0 style only**: Use `select()` instead of `query()`. Legacy style causes warnings.
- **Always review after Alembic autogenerate**: Auto-generated migrations may have missing indexes or incorrect types.
- **FastAPI Depends chain**: Calling other Depends inside a dependency function can cause ordering issues. Verify with tests.
- **async/await consistency**: Don't mix sync/async in a single router. Unify everything as async.

---

## Frontend Lessons

> This section is referenced by frontend-agent, debug-agent (for frontend bugs).

### Initial Lessons
- **Next.js App Router**: `useSearchParams()` must be used inside a `<Suspense>` boundary. Otherwise, build error.
- **shadcn/ui components**: Import path is `@/components/ui/button`, not `shadcn/ui`.
- **TanStack Query v5**: First argument of `useQuery` is object form `{ queryKey, queryFn }`. v4's `useQuery(key, fn)` form doesn't work.
- **Tailwind dark mode**: `dark:` prefix requires `darkMode: 'class'` setting to work.

---

## Mobile Lessons

> This section is referenced by mobile-agent, debug-agent (for mobile bugs).

### Initial Lessons
- **Riverpod 2.4+ code generation**: When using `@riverpod` annotation, `build_runner` execution required. Run `dart run build_runner build` before building.
- **GoRouter redirect**: Returning current path in redirect function causes infinite loop. Must return `null` to indicate no redirect.
- **Flutter 3.19+ Material 3**: `useMaterial3: true` is the default. M3 applies even without explicit setting in ThemeData.
- **Network on iOS simulator**: Use `127.0.0.1` instead of localhost. Or `10.0.2.2` for Android.

---

## QA / Security Lessons

> This section is referenced by qa-agent.

### Initial Lessons
- **Rate limiting verification method**: Send continuous requests with `curl` to verify 429 response. Code review alone is insufficient.
- **CORS wildcard**: `*` is OK for development environment, but must restrict to specific domains in production build.
- **npm audit vs safety**: Frontend uses `npm audit`, backend (Python) uses `pip-audit` or `safety check`.

---

## Debug Lessons

> This section is referenced by debug-agent.

### Initial Lessons
- **React hydration error**: Caused by code with different server/client values like `Date.now()`, `Math.random()`, `window.innerWidth`. Wrap with `useEffect` + `useState`.
- **N+1 query detection**: Setting `echo=True` in SQLAlchemy logs all queries. If same pattern query repeats, it's N+1.
- **State loss after Flutter hot reload**: initState of StatefulWidget doesn't re-execute on hot reload. State initialization logic should go in didChangeDependencies.

---

## QA Evaluation Lessons

> Referenced by qa-agent. Tracks patterns where QA judgment failed or succeeded.
> Unlike other sections (which track implementation mistakes), this section tracks
> the evaluator's own blind spots and strengths.

### Initial Lessons
- **Runtime verification is not optional**: Static code review cannot detect stubbed features, display-only implementations, or broken user flows. Always execute Step 2.5 (Runtime Verification) for Medium/Complex tasks.
- **Self-evaluation bias exists**: Implementation agents overrate their own output. Cross-review by a separate QA agent is the only reliable quality judgment. Mechanical self-checks (lint/test/build) are fine; quality judgment is not.
- **Severity calibration matters**: Auth/data-loss issues are always CRITICAL regardless of how small the code change appears. Do not downgrade severity based on diff size.
- **false_negatives are the most costly error**: A bug that ships costs more than a false_positive that wastes 5 minutes of impl agent time. Bias toward caution.

---

## Cross-Domain Lessons

> Referenced by all agents.

### Initial Lessons
- **API contract mismatch**: Parsing fails when backend uses `snake_case` but frontend expects `camelCase`. Casing must be specified in contract.
- **Timezone issues**: Backend stores in UTC, frontend displays in local timezone. Unify on ISO 8601 format.
- **Auth token passing**: Watch for mistakes where backend expects `Authorization: Bearer {token}` but frontend sends `token` header.

---

## Lesson Addition Protocol

### Automatic RCA Trigger (MANDATORY)

RCA (Root Cause Analysis) entry is **required** when:

| Trigger | Responsible Agent | Deadline |
|---------|-------------------|----------|
| Session CD score >= 50 | QA Agent | Before session close |
| Verification failure (verify.sh exit 1) | Debug Agent or failing agent | Before retry |
| Same error type occurs 2+ times in session | Orchestrator | Immediate |
| User explicitly requests "don't do this again" | Current agent | Before next action |

**RCA is not optional.** If threshold is met, lesson MUST be added before session completion.

### RCA Entry Format (Required Fields)

```markdown
### {YYYY-MM-DD}: {agent-type} - {one-line summary}
- **Problem**: {what went wrong - be specific}
- **Root Cause**: {why it happened - go deeper than surface}
- **Fix Applied**: {how it was resolved this time}
- **Prevention**: {process/prompt change to prevent recurrence}
- **CD Impact**: {clarify/correct/redo count if applicable}
```

### When QA Agent Adds
When finding recurring issues during review:
1. Add lesson to the relevant domain section
2. Format: `### {date}: {one-line summary}` + problem/cause/solution/prevention
3. MCP memory tool: `[EDIT]("lessons-learned.md", additional content)`

### When Orchestrator Adds
When there are failed tasks at session end:
1. Analyze failure cause
2. Add lesson to the relevant domain section
3. Prevent same mistakes in next session

### Auto-Generation from Experiment Ledger

At session end, the Orchestrator extracts discarded experiments with **delta <= -5** from the Experiment Ledger (see `experiment-ledger.md`) and generates lesson candidates.

Auto-generated lessons use the RCA Entry Format above, with these additions:
- **Root Cause** field specifies which quality dimension regressed and why
- Append `(Source: Experiment Ledger #{N}, Session {session_id})` to the summary line
- Append to the relevant domain section (based on agent type)

Only the Orchestrator performs this at session end, after all agents have completed and the ledger is finalized.

### When Lessons Become Too Many (50+)
- Move old lessons (6+ months) to archive
- Delete lessons invalidated by framework version upgrades
- This cleanup is performed manually (agents should not delete arbitrarily)
