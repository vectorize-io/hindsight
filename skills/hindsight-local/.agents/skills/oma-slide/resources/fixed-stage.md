# Fixed-Stage Rules — oma-slide

> The canonical reference for the 1920×1080 stage model, embedding instructions, px-authoring
> conventions, and the CJK → Pretendard rule.

## 1. The 1920×1080 Invariant

All slides are authored at exactly **1920 × 1080 px**. This is non-negotiable:

- The validator (`oma slide validate`) renders at 1920×1080 and checks geometry at that size.
- The exporter (`oma slide pdf|png`) captures at 1920×1080 before any post-processing.
- PPTX export converts 1920×1080 px to 720×405 pt (÷ 2.667) — pixel authoring only, no pt in HTML.
- Do **not** author in percentages, `vw/vh`, or responsive units that reflow the layout.
  Fixed `px` values only inside `.slide`.

```
Canvas size:  1920 px wide  ×  1080 px tall
Aspect ratio: 16:9
PPTX output:  720 pt wide   ×  405 pt tall   (px ÷ 2.667)
```

## 2. Stage Scaling — How It Works

`deck-stage.js` computes:

```js
const scale = Math.min(viewportWidth / 1920, viewportHeight / 1080);
```

Then positions the `.deck-stage` element so it is centred (letterbox / pillarbox):

```js
offsetLeft = (viewportWidth  - 1920 * scale) / 2
offsetTop  = (viewportHeight - 1080 * scale) / 2

stageEl.style.transform = `scale(${scale})`;
stageEl.style.left      = `${offsetLeft}px`;
stageEl.style.top       = `${offsetTop}px`;
```

`transform-origin` is `top left` (set in `viewport-base.css`). This keeps the maths
simple and avoids a secondary translate.

The viewport background (`#000` default) fills the letterbox/pillarbox bars. You can change
the background color of `.deck-viewport` in a theme without breaking the scale maths.

## 3. Embedding into a Deck

Paste or link the two shared assets into every deck's `<head>`. The `oma slide new` command
copies them into the workdir automatically; `oma slide bundle` inlines them into the single-file
output.

### Option A — External files (development / per-slide workdir)

```html
<head>
  <link rel="stylesheet" href="./viewport-base.css" />
</head>
<body>
  <deck-stage>
    <div class="deck-viewport">
      <div class="deck-stage">
        <section class="slide" id="slide-01">
          <!-- 1920×1080 content here -->
        </section>
        <section class="slide" id="slide-02">
          <!-- ... -->
        </section>
      </div>
    </div>
  </deck-stage>

  <!-- Optional: speaker notes (JSON keyed by 0-based slide index) -->
  <script type="application/json" id="speaker-notes">
    { "0": "Opening remarks...", "1": "Second slide notes..." }
  </script>

  <script src="./deck-stage.js"></script>
</body>
```

### Option B — Inlined (single-file bundle — `out/deck.html`)

`oma slide bundle` inlines both files; the structure is the same but the
`<link>` is replaced by `<style>...</style>` and `<script src>` becomes `<script>...</script>`.

### Optional: nav controls + slide counter

`deck-stage.js` looks for existing `.deck-nav` and `.deck-counter` elements.
If none are present the viewer still works (keyboard/touch/wheel only).
For `viewer.html`, the CLI injects:

```html
<nav class="deck-nav" aria-label="Slide navigation">
  <button id="btn-prev" aria-label="Previous slide">&#8592;</button>
  <button id="btn-next" aria-label="Next slide">&#8594;</button>
</nav>
<div class="deck-counter" role="status" aria-live="polite"></div>
```

## 4. px-Authoring Conventions

| Element | Convention |
|---|---|
| Slide root | `position: absolute; inset: 0; width: 1920px; height: 1080px;` |
| Safe zones | Left/right margin ≥ 80px; top/bottom margin ≥ 60px |
| Body text | 28–36 px minimum; heading 64–120 px |
| Icon / decorative image | explicit `width`/`height` in px |
| Background gradients | allowed (CSS); rasterized to PNG at PPTX export |
| Clipping / overflow | `overflow: hidden` on `.slide` prevents bleed-out |
| Animations | `transform` + `opacity` only; wrap in `@media (prefers-reduced-motion: no-preference)` |

### 8-px Grid

Align all spacing, padding, and element positions to multiples of 8 px. This keeps designs
crisp at all scale factors and maps cleanly to pt at PPTX export.

### Z-index layers

```
0     background layer   (solid color, image, gradient)
10    atmosphere layer   (subtle overlay, texture)
100   content layer      (text, charts, icons)
200   highlight layer    (callouts, badges)
1000  overlay layer      (modal-style content, reserved)
```

## 5. Slide Visibility — Always Use .active / .visible

**Never** toggle slides with `display:none` or `display:block`. Some layout classes from design
themes may override that and reveal all slides at once.

Use the CSS class protocol defined in `viewport-base.css`:

| Class | Meaning |
|---|---|
| _(no class)_ | Hidden: `visibility:hidden; opacity:0; pointer-events:none` |
| `.active` | Fully visible: `visibility:visible; opacity:1; pointer-events:auto` |
| `.visible` | Fading out: `visibility:visible; opacity:0; pointer-events:none` |

`deck-stage.js` manages these classes automatically. Slide authors do not need to set them.

## 6. Validator Contract — data-om-validate

`deck-stage.js` automatically annotates each `.slide` with:

```html
data-screen-label="Slide N / M"
data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
```

`oma slide validate` (puppeteer-core) reads these attributes to:

1. Locate each slide in the rendered DOM.
2. Know which checks to run (overflow / overlap / size checks).
3. Report findings as `{ code, message, slide, selector?, rect? }`.

**Do not remove or override `data-om-validate`** in authored slide HTML. If a slide intentionally
clips text (e.g., a decorative element), annotate the clipping element with `data-om-no-check`
to suppress false-positive overflow findings.

## 7. CJK Content → Pretendard Required

Any deck whose content contains Korean (한국어), Japanese (日本語), or Chinese (中文) characters
**must** include Pretendard Variable as the primary font.

```html
<!-- In <head> — Pretendard via CDN (fallback to system CJK stack) -->
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
/>
```

CSS font-family for CJK decks:

```css
:root {
  --font-body: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
    "Apple SD Gothic Neo", "Noto Sans KR", "Noto Sans JP", "Noto Sans SC", sans-serif;
  --font-display: "Pretendard Variable", Pretendard, sans-serif;
}
```

For latin-only decks, distinctive display fonts are encouraged (anti-"AI slop"):
system-font-stack restriction applies only to the **body** font fallback, not to
the chosen heading/display typeface.

## 8. prefers-reduced-motion Support

Wrap all CSS animations and transitions in slide content using:

```css
@media (prefers-reduced-motion: no-preference) {
  .animated-element {
    animation: slideIn 0.4s ease;
  }
}
```

Or equivalently, use the `.motion-safe` utility class provided by `viewport-base.css`:

```css
/* viewport-base.css disables animation-duration + transition-duration
   for .motion-safe * when prefers-reduced-motion: reduce */
```

`deck-stage.js` cross-fade transitions are also disabled when the user prefers reduced motion
(the slide visibility swap is instant).

## 9. Print / Save-as-PDF

Trigger with `Ctrl+P` / `Cmd+P` in the browser, or use `oma slide pdf --mode print`.

`deck-stage.js` removes the CSS transform before the print layout renders, so the browser
sees the true 1920×1080 px layout. `viewport-base.css` `@media print` rules:

- Remove `.deck-viewport` overflow/fixed positioning.
- Remove transform from `.deck-stage`.
- Make all `.slide` elements visible with `break-after: page`.
- Hide `.deck-nav` and `.deck-counter`.

Result: one clean 1920×1080 slide per printed page.

## 10. Presenter View (postMessage API)

When `viewer.html` opens the deck inside an `<iframe>`, `deck-stage.js` posts on every
slide change:

```js
window.parent.postMessage(
  { type: "slideIndexChanged", index: N, total: M, note: "speaker note text" },
  "*"
);
```

The presenter view reads speaker notes from
`<script type="application/json" id="speaker-notes">` (a JSON object keyed by 0-based index).

The parent frame can navigate the iframe by posting:

```js
iframeEl.contentWindow.postMessage({ type: "navigateTo", index: N }, "*");
```
