# Pre-Delivery Checklist — oma-slide

Run this gate after Phase 5 (Review) and before Phase 6 delivery (bundle/export).
Every item maps to a SKILL.md guardrail or the validator contract in `fixed-stage.md`.

## Gate (must pass)

- [ ] `oma slide validate --dir "$DECK_DIR" --format json` passes, or the surfaced diff was explicitly approved by the user after 3 auto-fix iterations
- [ ] Every slide root carries `data-om-validate` (validator contract)
- [ ] All slides authored at the fixed 1920×1080 stage in px units (no vw/vh/% layout)
- [ ] All `<img src>` / `<video src>` point to local `./assets/<file>` only — no remote URLs
- [ ] `meta.json` is current: `{ title, order[], style, density, speakerNotes }` matches the actual `slide-NN.html` set

## Accessibility and motion

- [ ] All CSS animations wrapped in `@media (prefers-reduced-motion: no-preference)`
- [ ] Nav controls have visible focus states (`.deck-nav button:focus-visible`)
- [ ] CJK text present → Pretendard font included on those slides

## Delivery artifacts

- [ ] `oma slide bundle --dir "$DECK_DIR"` produced `out/deck.html` and it opens in a browser
- [ ] If `./assets/` contains video: user was warned the bundle is not fully self-contained
- [ ] PPTX export (if requested) labeled **experimental** in user-facing output
- [ ] Canva export (if performed): design URL included in the delivery summary

## Delivery summary (Phase 6c)

- [ ] Reported: workdir path, slide file list, `out/deck.html` path, export paths, validate status
- [ ] Reported: any `TODO(oma-deferred)` items (e.g., unresolved image generation placeholders)
