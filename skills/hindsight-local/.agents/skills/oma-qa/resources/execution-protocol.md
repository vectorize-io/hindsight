# QA Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Quick security + quality check | **Medium**: Full 4 steps | **Complex**: Full + prioritized scope
2. **Check lessons**: read QA section in `../../_shared/core/lessons-learned.md`
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: security/auth concerns, existing code conflict potential?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Budget context**: follow `../../_shared/core/context-budget.md` (prioritize high-risk files)
5. **After review**: add recurring issues to `../../_shared/core/lessons-learned.md`
6. **If standards matter**: read `resources/iso-quality.md`

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Scope
- Identify what to review: new feature, full audit, or specific concern
- List all files/modules to inspect
- Determine review depth: quick check vs. comprehensive audit
- Use Serena to map the codebase:
  - `get_symbols_overview("src/")`: Understand structure
  - `search_for_pattern("password.*=.*[\"']")`: Find hardcoded secrets
  - `search_for_pattern("execute.*\\$\\{")`: Find SQL injection
  - `search_for_pattern("innerHTML")`: Find XSS vulnerabilities

## Step 2: Audit
Review in this priority order:
1. **Security** (CRITICAL): OWASP Top 10, auth, injection, data protection
2. **Performance**: API latency, N+1 queries, bundle size, Core Web Vitals
3. **Accessibility**: WCAG 2.2 AA, keyboard nav, screen reader, contrast
4. **Code Quality**: test coverage, complexity, architecture adherence

When applicable, map findings and gaps to:
- **ISO/IEC 25010** quality characteristics
- **ISO/IEC 29119** test planning, design, traceability, and exit criteria

Use `resources/checklist.md` (renamed qa-checklist) as the comprehensive review guide.

## Step 2.5: Runtime Verification

Static code review misses entire categories of bugs: display-only features,
stubbed functionality, broken user flows, and edge cases that only surface
at runtime. This step requires interacting with the running application.

### When to Execute
- Web app with UI: ALWAYS for Medium/Complex tasks
- API-only: ALWAYS (curl/httpie verification)
- Simple tasks (single file, no UI): SKIP

### Execution by App Type

#### Web Applications (Chrome DevTools MCP: Isolated Mode)

1. Start the application (`bun run dev`, `uv run manage.py runserver`, etc.)
2. Open the app in an **isolated browser context** to avoid contaminating the user's session:
   ```
   new_page(url: "http://localhost:PORT", isolatedContext: "qa-test")
   ```
   - Pages in the same `isolatedContext` share cookies/storage
   - Pages in different contexts are fully isolated
   - Always use `isolatedContext: "qa-test"` for QA verification
3. Navigate and inspect:
   ```
   navigate_page(url)            → navigate within the isolated context (SPA routes, sub-pages)
   take_snapshot()               → a11y tree snapshot with uid-tagged elements (prefer over screenshots)
   ```
4. Interact with modified features:
   ```
   click(uid)                    → click buttons/links
   fill(uid, value)              → fill input/textarea/select fields
   fill_form(elements)           → fill multiple fields at once
   type_text(text, submitKey)    → type into focused input
   press_key(key)                → keyboard shortcuts, Enter, Tab
   hover(uid)                    → hover for tooltips, dropdown menus, state changes
   drag(from_uid, to_uid)        → drag-and-drop
   ```
5. Verify functional correctness:
   - `list_console_messages(types: ["error", "warn"])` → detect JS errors
   - `list_network_requests()` → verify API calls fired and returned expected status
   - `get_network_request(reqid)` → inspect request/response bodies
   - `evaluate_script(function)` → check DOM state, global variables, computed styles
   - `wait_for(text)` → confirm expected content appears after interaction
6. Visual and accessibility verification:
   - `take_screenshot()` → visual output matches acceptance criteria
   - `lighthouse_audit(mode: "snapshot")` → accessibility, SEO, best practices scores
   - `emulate(viewport: "375x812x3,mobile,touch")` → responsive/mobile verification
7. Performance profiling (when performance criteria exist):
   - `performance_start_trace(reload: true)` → capture Core Web Vitals
   - `performance_stop_trace()` → analyze LCP, INP, CLS
   - `take_memory_snapshot(filePath)` → detect memory leaks
8. Cleanup:
   ```
   close_page(pageId)  → close isolated test pages after verification
   ```
9. **Fallback** (no Chrome DevTools MCP available):
   - Use curl/httpie to hit rendered endpoints
   - Verify HTTP status codes and response bodies
   - Check redirects, auth flows, and error pages

#### API Endpoints
1. Start the server
2. Execute acceptance criteria as actual HTTP requests:
   ```bash
   # Example: verify auth flow end-to-end
   curl -s -X POST localhost:8000/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","password":"secure123"}'

   # Verify rate limiting actually triggers
   for i in $(seq 1 20); do
     curl -s -o /dev/null -w "%{http_code}" localhost:8000/api/auth/login
   done
   ```
3. Verify database state after operations (query directly or via API)

#### Mobile Applications
1. If emulator/simulator available: launch and interact
2. If not: verify via API layer + widget test execution

### Recording Results

Append to the QA report under a new section:

```markdown
## Runtime Verification Results

| Feature | Method | Expected | Actual | Status |
|---------|--------|----------|--------|--------|
| User registration | curl POST /api/auth/register | 201 + user created | 201 + user in DB | PASS |
| Rate limiting | 20x rapid POST /api/auth/login | 429 after threshold | 429 after 10 req | PASS |
| Empty state UI | navigate_page + take_snapshot | Empty state message | Blank white page | FAIL |
| Button handler | click(uid) + list_network_requests | POST /api/save fired | No request fired | FAIL |
| Console errors | list_console_messages(["error"]) | 0 errors | TypeError at app.js:42 | FAIL |
```

### Stubbed Feature Detection

Specifically check for these patterns that static review cannot catch:
- Buttons/forms that render but have no backend handler → `click(uid)` + `list_network_requests()` to verify
- Features that display placeholder data instead of real data → `evaluate_script()` to check data binding
- Interactive elements that don't respond to user input → `click(uid)` + `take_snapshot()` to compare before/after
- Audio/video/file upload controls with no actual processing → `upload_file(uid, filePath)` + verify response

---

## Evaluator Posture: SKEPTICAL by default

Apply this posture when making verdict decisions in Step 3 and Step 4:
- Assume code has bugs until you prove otherwise with concrete evidence.
- "Probably works" or "should be fine" is NOT valid evidence. Run it, show output, prove it.
- If a feature exists in the plan but you cannot verify it works at runtime: FAIL.
- Never downgrade a real bug to "non-critical" just to pass a gate.
- Use WARNING when all remaining issues are MEDIUM or lower and none block deployment.
- When in doubt between PASS and FAIL, choose FAIL and explain why.

## Step 3: Report
Generate structured report with:
- Overall status:
  - PASS: no CRITICAL, no HIGH, and no MEDIUM issues
  - WARNING: no CRITICAL and no HIGH, but MEDIUM issues exist
  - FAIL: any CRITICAL or HIGH issue found
- Findings grouped by severity (CRITICAL > HIGH > MEDIUM > LOW)
- Each finding: file:line, description, remediation code
- Performance metrics vs. targets
- Standards suggestions when relevant:
  - quality characteristics under-covered
  - missing test design / traceability / exit criteria

## Step 4: Verify
- Run `resources/self-check.md` to verify your own review quality
- Ensure no false positives (each finding is real and reproducible)
- Confirm remediation suggestions are correct and complete
- Run `../../_shared/core/common-checklist.md` for general quality

## On Error
See `resources/error-playbook.md` for recovery steps.
