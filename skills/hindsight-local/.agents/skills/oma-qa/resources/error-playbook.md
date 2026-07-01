# QA Agent - Error Recovery Playbook

When you encounter a failure during review, follow these recovery steps.
Do NOT stop or ask for help until you have exhausted the playbook.

---

## Automated Tool Fails to Run

**Symptoms**: `npm audit`, `bandit`, `lighthouse` command errors

1. Check: is the tool installed? Note missing tool in result
2. Check: are you in the correct directory?
3. If `npm audit`: try `npm audit --production` to skip devDependencies
4. If `bandit`: check Python path; may need `python -m bandit`
5. If `lighthouse`: requires a running server; note if server not available
6. **If tool unavailable**: Fall back to manual review, record `tool_unavailable: ["tool_name"]` in result

---

## False Positive Suspected

**Symptoms**: Finding looks like a vulnerability but might be safe

1. Trace the data flow: does user input actually reach the dangerous operation?
2. Check: is there validation/sanitization upstream?
3. Check: is the framework handling this automatically? (e.g., ORM prevents SQL injection)
4. If uncertain: mark severity as `MEDIUM` with note "verify manually"
5. **NEVER do this**: Mark as CRITICAL without certainty (false alarms erode trust)

---

## Cannot Access Source Code

**Symptoms**: Serena `find_symbol` returns nothing, file not found

1. Check: correct file path? Use `search_for_pattern` with broader terms
2. Check: is the code in a different directory or monorepo?
3. Use `get_symbols_overview` on parent directories to find the structure
4. If truly inaccessible: review what you CAN access and note gaps in report

---

## Performance Metrics Unavailable

**Symptoms**: Can't run Lighthouse, no APM data, no load test results

1. Check if dev server is running for Lighthouse
2. If no server: review code statically for performance anti-patterns:
   - N+1 queries (loops with DB calls)
   - Missing pagination
   - Large bundle imports
   - No code splitting
3. Report findings with `static_analysis_only: true` flag
4. Recommend specific metrics to measure when environment is available

---

## Scope Too Large

**Symptoms**: Full audit requested but codebase has 100+ files

1. Prioritize: auth/security-critical files first
2. Use pattern search to find high-risk areas:
   - `search_for_pattern("password|secret|token|api_key")`
   - `search_for_pattern("execute|eval|innerHTML")`
3. Review critical paths: auth flow, payment, data mutation
4. Note in report: `scope_coverage: "critical paths only, full audit requires more"`

---

## Rate Limit / Quota Error

**Symptoms**: `429`, `RESOURCE_EXHAUSTED`, `rate limit exceeded`

1. **Stop immediately**: do not make additional API calls
2. Save current work to `progress-{agent-id}[-{sessionId}].md`
3. Record Status: `quota_exceeded` in `result-{agent-id}[-{sessionId}].md`
4. Specify remaining tasks

---

## Serena Memory Unavailable

1. Retry once
2. If 2 consecutive failures: use local file `/tmp/progress-{agent-id}[-{sessionId}].md`
3. Add `memory_fallback: true` flag to result

---

## General Principles

- **False positive prevention**: If finding is uncertain, lower severity and mark "verify manually"
- **Blocked**: If no progress after 5 turns, save current state, `Status: blocked`
- **No code modification**: QA only reports; delegate code changes to the appropriate agent
