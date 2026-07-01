# Output LAWs — oma-market

These 8 LAWs govern every brief emitted by `oma market render`. The `render` CLI runs a self-check before writing the result file; violations are auto-corrected when possible, otherwise rendering fails. Use `--no-self-check` only for debug.

LAWs apply to every QUERY_TYPE except where an explicit COMPARISON exception is noted.

---

## LAW 1 — No `Sources:` block at the end

Do not append `Sources:`, `References:`, `Further reading:`, `Citations:`, or any bulleted list of publication names / handles / URLs after the closing line of the brief. The engine footer is the only visible citation list. The saved raw JSON sidecar (when emitted) is the durable record.

**Self-check**: scan the last 15 lines for `^Sources:`, `^References:`, `^Further reading:`, `^Citations:` (case-insensitive) followed by a bulleted list. If found, strip.

---

## LAW 2 — No invented title line

For QUERY_TYPE `pain`, `trend`, `discovery`: the first line of the body (after the badge and one blank line) is the prose label `What we learned:` on its own line. Not a title, not a header, not `# {Topic}`.

For QUERY_TYPE `competitor` (or `--vs <entity>` set): use the COMPARISON template — first line is `# {A} vs {B}: 시장 신호` (Korean) or `# {A} vs {B}: Market Signal` (English).

**Self-check**: detect intent from output frontmatter or topic. If pain/trend/discovery and first body line != `What we learned:`, regenerate or insert.

---

## LAW 3 — No em-dash or en-dash

Use ` - ` (hyphen with spaces) instead of `—` or `–`. Em-dashes are the strongest AI-output tell. Apply to body, citation lines, footer.

**Exception**: quoted source text that literally used an em-dash. In quoted snippets, preserve the source character.

**Self-check**: regex `/[–—]/` outside of `> ` blockquotes. Replace ` — ` with ` - `, `—` with `-`.

---

## LAW 4 — No `##` headers in body

For QUERY_TYPE pain/trend/discovery, body has no `##` or `###` headers. Structure: bold-lead-in paragraphs + a prose label `KEY PATTERNS from the research:` + numbered list. Framework sections (`## SWOT`, `## Porter's 5 Forces`, `## PESTEL`) are allowed when frameworks are active.

For QUERY_TYPE competitor (COMPARISON): the following `##` headers are required and allowed:
- `## Quick Verdict`
- `## {Entity A}`, `## {Entity B}` (one per compared entity)
- `## Head-to-Head`
- `## The Bottom Line`

Any other `##` is forbidden.

**Self-check**: scan body lines for `^#{2,3}\s`. If header text is not in the allowed framework or COMPARISON list, flag for regenerate.

---

## LAW 5 — Engine footer pass-through

The render command emits a footer block bounded by `<!-- ENGINE FOOTER -->` and `<!-- END ENGINE FOOTER -->`. It contains: sources_used, sources_failed coverage, cluster count, item count, p50/p95 latency, cache hit/miss, total cost (if paid sources used).

Render must emit this verbatim. Do not paraphrase, recompute, reorder, or replace with a synthesized `## Notable Stats`.

---

## LAW 6 — No raw evidence dump

Cluster JSON details (`### N. (score, items, sources: ...)`, `- Uncertainty: single-source`, `- Uncertainty: thin-evidence`) are internal to the cluster output JSON. Markdown body must paraphrase as bold-lead-in paragraphs. If the body contains the literal pattern `### \d+. .* \(score \d+,`, that's a LAW 6 violation.

**Self-check**: pattern `/^### \d+\. .* \(score /m` — fail.

---

## LAW 7 — Inline citations as `[name](url)`

Every cited handle, subreddit, channel, publication is wrapped `[name](url)` at first mention in the body. Examples:
- `per [@octocat](https://github.com/octocat)`
- `[r/programming](https://reddit.com/r/programming) shipped a thread saying ...`
- NOT: `per https://github.com/octocat/repo` (raw URL)
- NOT: `per @octocat` (plain handle when URL available)
- NOT: `per [Rolling Stone]()` (broken empty link)

**Plain-text fallback**: when the raw data genuinely has no URL, use plain text — never emit broken empty link.

**Self-check**: find bare URLs (`/https?:\/\/[^\s)]+/` outside backticks and outside `[..](..)`) — flag for rewrite. Empty `\[[^\]]+\]\(\s*\)` — flag.

---

## LAW 8 — Badge first line

Line 1 of every brief is the badge, exactly:

```
🔎 oma-market v{ver} · synced {YYYY-MM-DD}
```

Where `{ver}` is the cli package version (read from `cli/package.json` `version`) and `{YYYY-MM-DD}` is today's UTC date. Line 2 is blank. Line 3 begins the body.

**Self-check**: line 1 must match `/^🔎 oma-market v[\d.]+ · synced \d{4}-\d{2}-\d{2}$/`. If missing, prepend.

---

## Self-check execution order (render.ts)

1. LAW 8 (badge) — prepend if missing.
2. LAW 2 (body opener) — verify body starts with `What we learned:` (or COMPARISON title).
3. LAW 4 (forbidden headers) — flag/regenerate.
4. LAW 6 (evidence dump) — flag/regenerate.
5. LAW 7 (citation format) — rewrite raw URLs and empty links.
6. LAW 3 (em-dash) — replace.
7. LAW 1 (trailing Sources block) — strip.
8. LAW 5 (footer) — verify present.

LAWs 8, 7, 3, 1 auto-correct. LAWs 2, 4, 6, 5 are violations that fail render with exit 1 (use `--no-self-check` to bypass for debug).

---

## Why these LAWs exist

Adapted from `last30days-skill` v3.0.x regression history: every LAW corresponds to a documented failure mode where a reasoning model produced output that the user later flagged as "AI slop", invented titles, missing citations, or wrong structure. The self-check enforces them at machine speed, not at user-review speed.
