# Search Agent - Error Playbook

## docs Route Errors

### Context7 library not found
**Symptom**: `resolve-library-id` returns no match
**Recovery**:
1. Try alternative library name spellings (e.g., "nextjs" vs "next.js")
2. If still not found: auto-fallback to `web` route
3. Inform user: "Could not find official docs via Context7. Falling back to web search."

### Context7 returns empty/irrelevant docs
**Symptom**: `query-docs` returns content that doesn't match the query topic
**Recovery**:
1. Try broader or narrower topic parameter
2. Fall back to `web` route with the original query
3. Present web results with note about docs miss

## web Route Errors

### Native search returns empty
**Symptom**: Runtime search tool returns 0 results
**Recovery**:
1. Simplify query (remove qualifiers, keep core terms)
2. Retry with simplified query
3. If still empty: run `oma search fetch <url>` on candidate URLs

### Native search blocked (402/403)
**Symptom**: Target site returns access denied
**Recovery**:
1. Run `oma search fetch <url>`; this auto-escalates api → probe → impersonate → browser
2. Each strategy tries progressively more aggressive access methods
3. If all strategies fail: report "Unable to access this source"; rerun
   with `--include-archive` to try caches (AMP / archive.today / Wayback)

### `oma search fetch` all strategies fail
**Symptom**: Non-zero exit code after api/probe/impersonate/browser exhausted.
**Recovery**:
1. Read the `attempts` array in JSON output: strategies, `elapsedMs`,
   HTTP status, detected `signals`.
2. Rerun with `--include-archive` for cached fallbacks.
3. `paywall` signal → content gated; report auth requirement.
4. `js-essential` + browser failed → site blocks headless Chrome;
   suggest manual fetch or alternative source.
5. Exit code 6 (timeout) → rerun with `--timeout 30` or larger.

### Browser strategy cannot find Chrome
**Symptom**: `"Chrome/Chromium not found ... or set OMA_CHROME_PATH."`
**Recovery**:
1. `oma search doctor` to see detection state.
2. Install Chrome / Edge / Brave / Chromium, or set `OMA_CHROME_PATH`.
3. Or `--skip browser` to rely on api/probe/impersonate only.

### curl_cffi not installed (impersonate strategy)
**Symptom**: `"curl_cffi is not installed. Run: pip install curl_cffi"`.
**Recovery**: `pip install curl_cffi`, or `--skip impersonate`.

## code Route Errors

### gh search code rate limit
**Symptom**: HTTP 403 or "rate limit exceeded" from GitHub API
**Recovery**:
1. Inform user: "GitHub API rate limit reached (30 req/min). Wait briefly or narrow your query."
2. Suggest adding language/repo qualifiers to reduce result set
3. Do NOT retry automatically in a loop

### glab api authentication missing
**Symptom**: `glab` not installed or not authenticated
**Recovery**:
1. Check if `glab` is available: `which glab`
2. If not installed: "GitLab CLI is not installed. Using GitHub search as default."
3. If not authenticated: "GitLab CLI requires authentication. Run `glab auth login` first."
4. Fall back to `gh search code` with notice

### gh/glab returns 0 results
**Symptom**: No code matches found
**Recovery**:
1. Suggest broader query terms
2. Suggest removing language filter if one was applied
3. Offer to try `web` route instead: "No code results found. Search web for examples?"

## Trust Scoring Errors

### Domain not in registry
**Symptom**: URL domain has no entry in trust-registry.md
**Recovery**: Label as `unknown` with score `—`. Do NOT exclude from results.

### --strict mode returns 0 results
**Symptom**: All results filtered out by trust score threshold
**Recovery**:
1. Inform user: "No results meet the strict trust threshold (verified+)."
2. Suggest: "Rerun with `--wide` to see all results with trust labels."

## Intent Classification Errors

### Misclassified intent
**Symptom**: User reports results are from wrong source type
**Recovery**:
1. Suggest explicit flag: "Try `--docs` or `--web` to specify the search type."
2. Results footer always shows: "Mode: {mode} ({auto|flag})" for transparency

### docs mode returns 0, user wanted docs
**Symptom**: `docs` classified correctly but Context7 has no match
**Recovery**:
1. Auto-add `web` search in parallel
2. Merge results, clearly label which came from web vs docs
3. Inform: "Official docs not found in Context7. Showing web results."

## Multi-Vendor Errors

### Runtime has no web search tool
**Symptom**: Current vendor doesn't expose a search tool
**Recovery**:
1. Skip native search, go directly to `oma search fetch <url>`
2. The probe strategy (Jina Reader + curl variants) works without vendor-specific tools

### Tool name mismatch across vendors
**Symptom**: Expected tool name doesn't exist in current runtime
**Recovery**: Follow vendor-detection protocol to identify correct tool names.
Execution protocol references generic capabilities, not specific tool names.
