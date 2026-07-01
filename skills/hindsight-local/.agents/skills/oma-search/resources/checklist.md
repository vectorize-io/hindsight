# Search Agent - Verification Checklist

Run this checklist before presenting results to the user.

## Intent Classification
- [ ] Query was classified into a valid mode (docs/web/code/local)
- [ ] Override flags were respected if provided
- [ ] Classification reasoning is logged in output header ("Mode: X (auto|flag)")

## Route Execution
- [ ] Correct route(s) dispatched based on classified intent
- [ ] docs route: Context7 library resolved, or fallback triggered with notice
- [ ] web route: native search attempted first, `oma search fetch` only on failure
- [ ] code route: platform detected from URL or default to gh
- [ ] local route: delegated to Serena without duplication

## Result Quality
- [ ] Results are relevant to the original query
- [ ] No duplicate URLs in final output
- [ ] Results are sorted by relevance, trust score as tiebreaker
- [ ] Each result has: source label, title, domain, trust info

## Trust Scoring
- [ ] Every non-local result has a trust label attached
- [ ] Unknown domains labeled as `—`, not excluded
- [ ] `--strict` mode correctly filters below 0.85
- [ ] `--strict` with 0 results shows rerun suggestion

## Error Handling
- [ ] Failures reported transparently (no silent drops)
- [ ] Fallback chain followed (docs->web, web->`oma search fetch`)
- [ ] Rate limits reported without retry loops

## Output Format
- [ ] Header shows query and mode
- [ ] Results numbered with route prefix (DOCS/WEB/CODE)
- [ ] Trust tags and scores shown in brackets
- [ ] Filtered count shown in --strict mode
