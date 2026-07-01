# Search Agent - Execution Protocol

## Step 0: Parse Query

1. **Extract flags** from the query string:
   - `--docs`, `--code`, `--web`: force specific route
   - `--strict`: only verified+ sources (trust score >= 0.85)
   - `--wide`: all sources with trust labels
   - `--gitlab`: force `glab api` for code route
2. **Classify intent** using `resources/intent-rules.md`:
   - If flag is present: skip classification, use flag
   - If no flag: apply keyword pattern matching
   - If ambiguous: use `web` + `docs` parallel (default fallback)
3. **Log** selected mode and route(s) for transparency

## Step 1: Route Dispatch

### docs route
1. Call Context7 `resolve-library-id` with the library/framework name
2. If resolved: call `query-docs` with the library ID and query topic
3. If NOT resolved: fall back to `web` route with notice to user

### web route
1. Use runtime native search tool (WebSearch, Google Search, Bing, etc.)
2. If native search fails or returns blocked/empty:
   - Enter the bypass fallback (see below)
3. Collect top results with URLs

#### Bypass Fallback (on native search failure)
Delegate to the native CLI: `oma search fetch <url>`. The pipeline
auto-escalates through four strategies and stops on the first success:

- **api**: platform-specific handlers (Twitter syndication, Reddit JSON,
  HN Firebase, arXiv Atom, SE v2.3, Bluesky AT Protocol, Mastodon, Wikipedia,
  CrossRef, OpenLibrary, Lobste.rs, dev.to, V2EX, npm, PyPI, Naver blog/finance).
- **probe**: parallel Jina Reader + WebFetch + curl UA variants (first-wins).
- **impersonate**: Python `curl_cffi` subprocess (safari→chrome→firefox;
  Korean hosts prefer safari). Auto-skips remaining TLS targets on
  JS-essential markers.
- **browser**: `puppeteer-core` + system Chrome via CDP. No MCP runtime
  dependency. Install Chrome or set `OMA_CHROME_PATH`.

Sidecar: add `--include-archive` to try AMP → archive.today → Wayback
Machine when all primary strategies fail. Archive hits tag `provenance`
so consumers can deprioritize cached content.

Flags: `--only <list>`, `--skip <list>`, `--timeout <sec>`, `--locale <v>`,
`--pretty`. Exit codes: 0=ok, 2=blocked, 3=not-found, 4=invalid-input,
5=auth-required, 6=timeout, 1=error.

### code route
1. Detect platform from user-provided URL or context:
   - URL contains `github.com` -> use `gh search code "query"`
   - URL contains `gitlab.com` -> use `glab api "/search?scope=blobs&search=query"`
   - No URL (keyword only) -> use `gh search code` (default, largest OSS coverage)
   - `--gitlab` flag -> force `glab api`
2. Parse CLI output into structured results
3. Include repo name, file path, and match context

### local route
Delegate entirely to Serena MCP:
- `find_symbol` for named code entities
- `search_for_pattern` for arbitrary text patterns
- `get_symbols_overview` for structural exploration

Do NOT duplicate Serena's functionality. Simply pass through.

## Step 2: Collect Results

1. Gather results from all dispatched routes
2. Normalize into uniform format:
   ```
   { source: "docs|web|code|local", title, url, domain, snippet }
   ```
3. Deduplicate by URL

## Step 3: Trust Scoring

1. For each result with a URL, extract the domain
2. Resolve trust score (see `resources/trust-registry.md`):
   a. Check Serena memory cache (`trust-registry-cache`)
   b. If cache miss: apply heuristic pattern matching (domain patterns → category + score)
   c. If heuristic returns unknown: call Tranco API (`siterank.redirect2.me`) for validation
3. Attach trust level, tags, and score:
   - Resolved: use level and score
   - Unresolved: label as `unknown` with score `—`
4. Write newly resolved scores to Serena memory cache
5. If `--strict` mode: filter out results below `verified` (< 0.85)
   - If 0 results remain: suggest `--wide` rerun
6. Sort by: relevance first, trust score as tiebreaker

## Step 4: Present Results

Format output as:
```
Query: "{query}"
Mode: {mode} ({auto|flag})

{ROUTE} #{n}  {title} — {source}     [{tags} {stars} {score}]
```

Example:
```
Query: "Next.js middleware authentication"
Mode: docs + web (auto)

DOCS #1  Next.js Middleware — next.js docs     [verified,lang-docs 0.95]
DOCS #2  Authentication — next-auth.js docs    [verified,lib-docs 0.90]
WEB  #3  Middleware Auth Guide — vercel.com     [verified,vendor 0.90]
WEB  #4  Next.js Auth Tutorial — dev.to        [external,blog 0.35]
CODE #5  middleware.ts — vercel/next.js         [github]
```

## On Error
See `resources/error-playbook.md` for recovery steps.
