# Frontend Agent - Error Recovery Playbook

When you encounter a failure, find the matching scenario and follow the recovery steps.
Do NOT stop or ask for help until you have exhausted the playbook.

---

## False Positive: `proxy.ts` flagged as dead code or `middleware.ts` demanded

**Symptoms**: Reviewer claims `src/proxy.ts` won't be picked up, demands rename to `src/middleware.ts`, or flags the auth gate as not wired.

1. **STOP. This is a fatal self-error.** Next.js 16+ renamed `middleware.ts` → `proxy.ts`.
2. The canonical request-proxy / auth-gate file is `proxy.ts` (root or `src/`) exporting a `proxy` function.
3. `middleware.ts` is BANNED in this project, not "deprecated", BANNED. Touch it and you die.
4. Verify by checking `package.json` for `next@16+`; `proxy.ts` is the only correct answer.
5. Reference: https://nextjs.org/docs/messages/middleware-to-proxy
6. **Retract the finding.** Do NOT raise this as critical / security / medium / low at any severity.

---

## TypeScript Compilation Error

**Symptoms**: `TS2322`, `TS2345`, `Type X is not assignable to type Y`

1. Read the error: which file, which line, which types conflict
2. Check: is the interface/type definition correct?
3. Check: is the API response type matching the expected shape?
4. If API mismatch: update the type to match actual response (don't cast with `as any`)
5. If generic issue: use explicit type parameter `<Type>` instead of inference
6. **NEVER do this**: `@ts-ignore`, `as any` (hides type issues without resolving them)

---

## Build Error

**Symptoms**: `next build` fails, `Module not found`, `SyntaxError`

1. Read the full error: which module, which file
2. If missing dependency: note in result as "requires `npm install X`"; do NOT install yourself
3. If import path wrong: use `search_for_pattern("export.*ComponentName")` to find actual path
4. If dynamic import issue: ensure component is client-side (`'use client'`)
5. Re-run build after fix to confirm

---

## Test Failure

**Symptoms**: `vitest` FAILED, `expect(X).toBe(Y)` assertion errors

1. Read the error: expected vs received, which test file
2. `find_symbol("ComponentName")` to check current implementation
3. Determine: test outdated or implementation wrong?
   - Test expects old behavior → update test
   - Component bug → fix component
4. Re-run the specific test: `npx vitest run path/to/test.ts`
5. **After 3 failures**: Try a different approach. Record in progress

---

## Hydration Mismatch (Next.js)

**Symptoms**: `Hydration failed`, `Text content does not match server-rendered HTML`

1. Find the component that renders differently on server vs client
2. Common causes:
   - `Date.now()` or `Math.random()` in render
   - Browser-only APIs (`window`, `localStorage`) without `useEffect`
   - Conditional rendering based on client-only state
3. Fix: wrap client-only code in `useEffect` + state, or use `'use client'`
4. If third-party component: wrap with `dynamic(() => import(...), { ssr: false })`

---

## API Integration Error

**Symptoms**: `Network Error`, `CORS`, `401 Unauthorized`, wrong data shape

1. **CORS**: Check backend CORS config; is frontend origin allowed?
2. **401**: Check token; is it in the header? is it expired?
3. **Wrong data**: Log `response.data` and compare with expected type
4. **Network Error**: Is the backend running? Correct port?
5. If backend isn't your responsibility: document the expected API contract in result

---

## Styling / Layout Broken

**Symptoms**: Component renders but looks wrong, responsive breakpoint fails

1. Check Tailwind classes: typo? wrong breakpoint prefix?
2. Check parent container: is it blocking layout? (`overflow-hidden`, fixed width)
3. Test at specific breakpoints: 320px, 768px, 1024px, 1440px
4. Use browser DevTools to inspect computed styles
5. If dark mode issue: check `dark:` variants applied

---

## Rate Limit / Quota Error (Gemini API)

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

- **After 3 failures**: If same approach fails 3 times, must try a different method
- **Blocked**: If no progress after 5 turns, save current state and record `Status: blocked`
- **Out of scope**: If you find backend issues, only record in result; do not modify directly
