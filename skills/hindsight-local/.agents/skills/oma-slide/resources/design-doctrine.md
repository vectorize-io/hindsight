# Design Doctrine — oma-slide

> Anti-"AI slop" aesthetics, font rules, content-density modes, and accessibility requirements.
> Read this document before writing any slide HTML. It defines the aesthetic contract for all
> oma-slide generated decks.

---

## 1. The Anti-"AI Slop" Commitment

Most AI-generated presentations are immediately recognizable: Inter or Roboto at arbitrary sizes,
purple-to-blue linear gradients, white cards dropped on a gradient background, icon rows that
feel copied from a template screenshot. These choices read as unowned — the visual equivalent of
filler text.

Every oma-slide deck must make a **committed aesthetic choice** and follow it through. That means:

- A **distinctive typographic voice** — not the path of least resistance.
- A **committed palette** — 2–4 colors with semantic roles, not "looks fine on screen."
- An **atmospheric intention** — the deck should feel like something, not just contain information.

The test: could this slide have come from a specific studio, designer, or publication? If yes,
it is doing its job.

---

## 2. Typography Rules

### 2a. Forbidden Defaults

Do **not** use the following as the primary or display typeface:

| Typeface | Why forbidden |
|---|---|
| Inter | Ubiquitous SaaS default; zero personality at display sizes |
| Roboto | Google Material default; reads as a framework artifact |
| Arial / Helvetica (unmodified) | Corporate filler; visually neutral to the point of invisibility |
| Open Sans | Overused in educational/government templates |
| Lato | Same problem as Open Sans |

Using any of the above as a **body/fallback** in a font stack is acceptable. Using them as
the **display/headline typeface** produces AI-slop output.

### 2b. Distinctive Latin Display Fonts (encouraged)

Choose from distinctive, well-crafted typefaces. Examples (all free via Google Fonts or equivalent CDN):

| Typeface | Character | Good for |
|---|---|---|
| Playfair Display | Classical serif elegance | Editorial, advisory, literary |
| Cormorant Garamond | Literary high-contrast serif | Scholarly, quiet luxury |
| Fraunces | Optical-size serif with personality | Warm editorial, brand work |
| Bricolage Grotesque | Variable grotesk with rhythm | Contemporary, design-led |
| Bebas Neue | Condensed caps impact | Bold poster, headlines-only |
| Shrikhand | Bold decorative display | Loud editorial, manifesto |
| Space Grotesk | Technical yet warm | Tech, SaaS with voice |
| Syne | Geometric with irregular flow | Indie, experimental |
| Alfa Slab One | Slab-serif punch | Activist, campaign energy |
| DM Serif Display | Transitional, refined | Professional with warmth |

This is not an exhaustive list — use judgment. A face should be chosen because it serves the
deck's mood, not because it is available.

### 2c. CJK Decks — Pretendard Required

Any deck whose content includes **Korean (한국어), Japanese (日本語), or Chinese (中文)** characters
**must** include Pretendard Variable as the primary font. This is non-negotiable.

```html
<!-- In <head> -->
<link rel="preconnect" href="https://cdn.jsdelivr.net" />
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
/>
```

```css
:root {
  --font-body: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont,
    "Apple SD Gothic Neo", "Noto Sans KR", "Noto Sans JP", "Noto Sans SC", sans-serif;
  --font-display: "Pretendard Variable", Pretendard, sans-serif;
}
```

Rationale: the system CJK font stack varies wildly across OS versions and produces
inconsistent rendering at large display sizes. Pretendard is modern, variable, and has
a dynamic subset CDN making it practical for web delivery.

For bilingual EN/CJK decks: pair Pretendard with a compatible latin display face (e.g.,
Bricolage Grotesque or a Grotesk that harmonizes with Pretendard's proportions).

### 2d. Font Loading

Load display fonts via `<link rel="preconnect">` + `<link rel="stylesheet">` in the `<head>`.
The validator awaits `document.fonts.ready` before measuring geometry — so font choices directly
affect overflow detection. Choose fonts early in the generation phase, not as an afterthought.

---

## 3. Color Palette Rules

### 3a. Anti-Patterns to Avoid

| Pattern | Why it fails |
|---|---|
| Purple-to-blue linear gradient background | The single most common AI-generated visual cliché |
| Gradient orbs/blobs floating on white | Overused in SaaS landing-page design, reads as template |
| Full-rainbow accent colors | Signals no palette decision was made |
| Pure `#ffffff` background with no atmospheric treatment | Clinically neutral; no sense of material or surface |
| Bright neon on black + one more neon (no restraint) | Every cyberpunk AI deck; needs commitment to be distinctive |

### 3b. Committed Palette Structure

A good palette has **semantic roles**, not just colors:

```
background   — the canvas; sets material and atmosphere
text-primary — main readable content; must pass WCAG AA on background
text-muted   — supporting text, captions; must pass WCAG AA
accent       — maximum one primary accent; the deck's signature color
accent-alt   — optional secondary accent; only if the design concept requires it
```

Name colors descriptively: `"Ink Navy (#1a2332)"` not `"dark blue"`. The name carries
the palette intent.

### 3c. WCAG AA Minimum

All text must meet **WCAG 2.1 AA** contrast ratios:
- Normal text (< 18 pt / < 14 pt bold): 4.5:1 contrast ratio
- Large text (≥ 18 pt / ≥ 14 pt bold): 3:1 contrast ratio

At 1920×1080, "normal" text corresponds to roughly ≤ 24 px; "large" text to ≥ 24 px.

Use a contrast checker before finalizing a palette. If a design direction cannot meet AA, revise
the palette — do not compromise accessibility for aesthetics.

---

## 4. Atmospheric Intention

Atmosphere is the quality that makes a deck feel like it was designed rather than assembled.
It comes from:

- **Background treatment**: solid color, gradient with direction and purpose, subtle texture,
  or photographic.
- **Whitespace**: deliberate emptiness communicates confidence. Cramped slides communicate anxiety.
- **Typographic hierarchy**: 2–3 sizes maximum per slide, each with a clear role.
- **Layer logic** (from `fixed-stage.md`):
  - z-index 0 — background
  - z-index 10 — atmosphere (texture, overlay)
  - z-index 100 — content
  - z-index 200 — highlights/callouts

The wildcard preview in Phase 2 is the appropriate place to introduce an unexpected atmospheric
direction. The safe preset and bold template choices should stay within recognizable territory.

---

## 5. Content-Density Modes

The `density` field in `meta.json` governs how slides are laid out and how much content they carry.

### 5a. Speaker-Led (Sparse)

The presenter is the primary content delivery vehicle. Slides are visual anchors.

| Element | Guidance |
|---|---|
| Headline | 1 short statement per slide; 64–120 px; ≤ 10 words |
| Body text | 0–3 lines maximum; 32–48 px; no prose paragraphs |
| Bullets | Avoid. Use one concept per slide instead. |
| Visuals | Large, occupying ≥ 40% of the canvas |
| Slide count | Allow 1 slide per 1–2 minutes of talk time |
| Speaker notes | Extensive notes in `meta.json.speakerNotes[]` for the presenter |

### 5b. Reading-First (Dense)

The deck is a self-contained document. Readers consume it without a presenter.

| Element | Guidance |
|---|---|
| Headline | Clear, self-explanatory; 48–72 px |
| Body text | Full sentences allowed; 28–36 px; max 8 lines per slide |
| Bullets | Acceptable; ≤ 6 items; each bullet ≤ 20 words |
| Visuals | Supporting; labeled with captions |
| Slide count | Allow as many slides as the content requires |
| Speaker notes | Optional; deck must stand alone without them |

### 5c. Mixed Density

Some decks mix modes (e.g., opener slides are sparse, data slides are dense). Apply the relevant
mode rules per slide and record the overall dominant mode in `meta.json.density`.

---

## 6. Slide Count Guidance

| Purpose | Sparse (speaker-led) | Dense (reading-first) |
|---|---|---|
| Pitch / investor (5 min) | 6–10 slides | 8–14 slides |
| Internal report (15 min) | 12–18 slides | 18–30 slides |
| Conference talk (30 min) | 20–35 slides | 30–50 slides |
| Executive briefing (10 min) | 8–12 slides | 12–20 slides |
| Product launch (standalone) | 10–16 slides | 16–28 slides |

These are guidelines, not hard limits. Content scope governs slide count; do not pad or compress
slides to hit a number.

---

## 7. What a Good Slide Looks Like

A well-crafted slide satisfies all of these:

1. **One idea** — a viewer can state the slide's point in one sentence.
2. **Scannable hierarchy** — the largest element is the most important.
3. **Comfortable empty space** — the canvas breathes; content does not fill edge to edge.
4. **Consistent with the deck** — the color, font, and grid choices are the same as every other slide.
5. **Readable at scale** — body text ≥ 28 px; the smallest legible text at 1920×1080 passes WCAG AA.
6. **Accessible in motion** — any animations are wrapped in `@media (prefers-reduced-motion: no-preference)`.

---

## 8. What Makes a Deck Feel Distinctive (not just acceptable)

The difference between "it looks fine" and "this is well designed" is usually one of:

- **Typographic commitment**: the display typeface is unmistakably itself at every size.
- **Palette ownership**: the colors have a name and a reason; they could not have been chosen randomly.
- **Compositional tension**: not every element is centered and balanced — some slides use asymmetry, large negative space, or a single dominant element to create visual interest.
- **Atmospheric consistency**: the first slide and the last slide feel like they come from the same world.
- **Motion that earns its place**: if there are animations, they reinforce the content rhythm — a fade-up for an incoming point, not a spin because CSS supports it.

---

## 9. Accessibility Requirements (WCAG AA + Reduced Motion)

### 9a. prefers-reduced-motion

**All CSS animations and transitions in generated slides must be wrapped:**

```css
@media (prefers-reduced-motion: no-preference) {
  .animated-element {
    animation: fadeUp 0.5s ease forwards;
    transition: opacity 0.3s ease;
  }
}
```

Use `transform` + `opacity` only for animations. Avoid `width`, `height`, `margin`, `padding`
transitions (these cause layout reflows and are expensive at 1920×1080).

`deck-stage.js` cross-fade transitions are also disabled when `prefers-reduced-motion: reduce`
is set by the system.

### 9b. Focus States

Navigation controls (`.deck-nav button`) must have visible `:focus-visible` styles. The viewer
is keyboard-navigable; sighted keyboard users must see which element has focus.

### 9c. Fixed Stage Tradeoff

The fixed 1920×1080 letterbox model is an accepted, conscious limitation:
- On small mobile screens the stage scales down (text shrinks proportionally).
- Screen readers receive the raw HTML structure, not the scaled layout.
- Zoom interactions in browsers may distort the scaled stage.

These are known tradeoffs of the stage model — do not attempt to work around them in generated
HTML. Document this limitation when relevant to the user.
