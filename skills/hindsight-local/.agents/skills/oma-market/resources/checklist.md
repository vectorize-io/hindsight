# Pre-submit checklist for oma-market — all items must pass before writing the brief to results/.

## Self-Check Items

- [ ] detect-trap exit code is 0 or 2 with user reframe (never silently swallowed)
- [ ] harvest `sources_used` has >= 2 entries, or warn coverage in brief footer
- [ ] All cluster representatives have URL or plain-text fallback (no bare signal IDs)
- [ ] No `—` (em-dash) or `–` (en-dash) characters in output body (use ` - ` instead)
- [ ] No `## ` headers in output body outside framework sections (SWOT, Porter's 5F, PESTEL)
- [ ] No `Sources:` / `References:` / `Further reading:` block at end of brief
- [ ] Badge present on line 1 in format `🔎 oma-market v{ver} · synced {YYYY-MM-DD}`
- [ ] Inline `[name](url)` citation count > 0 (at least one cited source in body)

## How to Use

Run this checklist in Step 7 of `execution-protocol.md` before writing the output file.

For each failing item, revise the render output and re-check before proceeding. Do NOT write the brief if any item fails.

If `sources_used` has only 1 entry (coverage warning case), write the brief but append a warning line in the engine footer:

```
⚠ Low coverage: only 1 source returned signals. Consider --window 90d or check env keys.
```
