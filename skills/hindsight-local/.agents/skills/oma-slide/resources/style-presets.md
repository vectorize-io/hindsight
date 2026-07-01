# Style Presets — oma-slide

> 12 curated, offline, self-contained presets vendored for zero-network reliability.
> These are the always-safe fallback core. Use them directly or as the basis for the
> "safe preset" preview in Phase 2 style discovery.

**Attribution:** These presets are adapted from `zarazhangrui/frontend-slides`
(MIT License — Copyright zarazhangrui). Adapted for the oma-slide fixed-stage model
(1920×1080 px canvas). See attribution note at end of this file.

---

## How to Choose

| Preset slug | Mood | Scheme | Best for |
|---|---|---|---|
| `ink-press` | Editorial, authoritative | Light | Journalism, publishing, advisory |
| `night-signal` | Institutional, measured | Dark | Investor decks, board, legal |
| `studio-electric` | High-contrast, design-led | Dark | Design studios, keynotes |
| `parchment-serif` | Quiet, scholarly, warm | Light | Research, white papers, academic |
| `cobalt-clean` | Modern professional, calm | Light | B2B SaaS, consulting, reports |
| `forest-quiet` | Organic, warm, considered | Dark | Sustainability, wellness, nature |
| `neobrutalist-block` | Bold, graphic, confident | Light | Agencies, indie SaaS, brand |
| `warm-monochrome` | Restrained, literary, archival | Light | Research synthesis, policy briefs |
| `coral-editorial` | Warm, graphic, magazine | Dark | Fashion, lifestyle, manifestos |
| `pastel-pop` | Friendly, approachable, indie | Light | Creators, indie launches, community |
| `midnight-scholar` | Scholarly, literary, quiet | Dark | Research, advisory, longform |
| `clean-tech` | Precise, modern, technical | Light | Developer talks, SaaS, data |

---

## Preset Definitions

### `ink-press`

**Mood:** Editorial, authoritative, classic
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Aged Newsprint | `#f5f0e8` |
| text-primary | Press Black | `#1a1a1a` |
| text-muted | Stone Gray | `#6b6b6b` |
| accent | Vermilion Red | `#d4380d` |

**Font pairing:**
- Display: `Playfair Display` (Google Fonts — weight 700, 900)
- Body: `DM Sans` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #f5f0e8;
  --color-text: #1a1a1a;
  --color-muted: #6b6b6b;
  --color-accent: #d4380d;
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Journalism, publishing, thought leadership, policy reports, advisory decks.
**Avoid:** Playful consumer brands, high-tech product demos.

---

### `night-signal`

**Mood:** Institutional, trustworthy, measured
**Scheme:** Dark

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Deep Navy | `#0d1b2a` |
| text-primary | Bone White | `#f2ede4` |
| text-muted | Slate Silver | `#9ba8b5` |
| accent | Muted Gold | `#c9a84c` |

**Font pairing:**
- Display: `DM Serif Display` (Google Fonts)
- Body: `DM Sans` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #0d1b2a;
  --color-text: #f2ede4;
  --color-muted: #9ba8b5;
  --color-accent: #c9a84c;
  --font-display: 'DM Serif Display', Georgia, serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Investor decks, board presentations, legal/policy briefs, consulting deliverables.
**Avoid:** Playful or consumer-facing content.

---

### `studio-electric`

**Mood:** High-contrast, design-led, bold
**Scheme:** Dark

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Studio Black | `#0a0a0a` |
| text-primary | Electric Yellow | `#f5e642` |
| text-muted | Warm Gray | `#888888` |
| accent | Electric Yellow | `#f5e642` |

**Font pairing:**
- Display: `Space Grotesk` (Google Fonts — weight 700)
- Body: `Space Grotesk` (Google Fonts — weight 400)

```html
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #0a0a0a;
  --color-text: #f5e642;
  --color-muted: #888888;
  --color-accent: #f5e642;
  --font-display: 'Space Grotesk', system-ui, sans-serif;
  --font-body: 'Space Grotesk', system-ui, sans-serif;
}
```

**When to use:** Design studios, agency credentials, creative reviews, brand showcases, tech keynotes.
**Avoid:** Warm, friendly, or institutionally conservative contexts.

---

### `parchment-serif`

**Mood:** Quiet, scholarly, warm
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Warm Parchment | `#faf6ef` |
| text-primary | Ink Brown | `#2d2015` |
| text-muted | Dusty Umber | `#8a7560` |
| accent | Sage Green | `#5a7a5c` |

**Font pairing:**
- Display: `Cormorant Garamond` (Google Fonts — weight 600, 700)
- Body: `Jost` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,600;0,700;1,600&family=Jost:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #faf6ef;
  --color-text: #2d2015;
  --color-muted: #8a7560;
  --color-accent: #5a7a5c;
  --font-display: 'Cormorant Garamond', Georgia, serif;
  --font-body: 'Jost', system-ui, sans-serif;
}
```

**When to use:** Research synthesis, white papers, academic presentations, literary and arts decks.
**Avoid:** Tech demos, urgent business pitches.

---

### `cobalt-clean`

**Mood:** Modern professional, trustworthy, calm
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Cream Paper | `#fafaf7` |
| text-primary | Charcoal | `#1c1c1e` |
| text-muted | Neutral Gray | `#6e6e73` |
| accent | Electric Cobalt | `#0047ff` |

**Font pairing:**
- Display: `Bricolage Grotesque` (Google Fonts — weight 700, 800)
- Body: `DM Sans` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@700;800&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #fafaf7;
  --color-text: #1c1c1e;
  --color-muted: #6e6e73;
  --color-accent: #0047ff;
  --font-display: 'Bricolage Grotesque', system-ui, sans-serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** B2B SaaS, consulting, advisory updates, internal reports, product reviews.
**Avoid:** Highly creative or expressive contexts where "professional" reads as limiting.

---

### `forest-quiet`

**Mood:** Organic, warm, considered
**Scheme:** Dark (mixed)

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Deep Forest Green | `#1e3a2f` |
| text-primary | Cream Linen | `#f2ede3` |
| text-muted | Sage Mist | `#8aad99` |
| accent | Rust Clay | `#c4622d` |

**Font pairing:**
- Display: `Fraunces` (Google Fonts — weight 700, 900)
- Body: `DM Sans` (Google Fonts — weight 400)

```html
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,700;0,9..144,900;1,9..144,700&family=DM+Sans:wght@400&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #1e3a2f;
  --color-text: #f2ede3;
  --color-muted: #8aad99;
  --color-accent: #c4622d;
  --font-display: 'Fraunces', Georgia, serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Sustainability, wellness, environmental, nature, organic brands, advisory work.
**Avoid:** Fast-paced tech or consumer brands needing energy and urgency.

---

### `neobrutalist-block`

**Mood:** Bold, graphic, confident, design-led
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Off-White | `#f0ede8` |
| text-primary | Ink Black | `#111111` |
| text-muted | Mid Gray | `#555555` |
| accent | Pastel Neon Yellow | `#e8f542` |
| accent-alt | Hot Pink | `#ff3c78` |

**Font pairing:**
- Display: `Syne` (Google Fonts — weight 700, 800)
- Body: `DM Sans` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #f0ede8;
  --color-text: #111111;
  --color-muted: #555555;
  --color-accent: #e8f542;
  --color-accent-alt: #ff3c78;
  --font-display: 'Syne', system-ui, sans-serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Agency credentials, design-led pitches, indie SaaS, brand redesigns, creative reviews.
**Avoid:** Regulated disclosures, traditional institutional contexts.

---

### `warm-monochrome`

**Mood:** Restrained, literary, archival, minimal
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Ivory Ledger | `#f7f4ee` |
| text-primary | Near Black | `#191918` |
| text-muted | Gray Stone | `#757570` |
| accent | Near Black (no color) | `#191918` |

**Font pairing:**
- Display: `Playfair Display` (Google Fonts — weight 700, 900)
- Body: `Jost` (Google Fonts — weight 400)

```html
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Jost:wght@400&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #f7f4ee;
  --color-text: #191918;
  --color-muted: #757570;
  --color-accent: #191918;
  --font-display: 'Playfair Display', Georgia, serif;
  --font-body: 'Jost', system-ui, sans-serif;
}
```

**When to use:** User research synthesis, white papers, longform reports, academic/policy briefs.
**Avoid:** Decks requiring visual personality, color-led storytelling, or brand identity.

---

### `coral-editorial`

**Mood:** Warm, graphic, magazine-cover
**Scheme:** Dark

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Near Black | `#141414` |
| text-primary | Bone Cream | `#f0ebe2` |
| text-muted | Warm Gray | `#9e9991` |
| accent | Coral Flame | `#e8603a` |

**Font pairing:**
- Display: `Bebas Neue` (Google Fonts)
- Body: `DM Sans` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #141414;
  --color-text: #f0ebe2;
  --color-muted: #9e9991;
  --color-accent: #e8603a;
  --font-display: 'Bebas Neue', Impact, sans-serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Fashion, lifestyle, fitness, F&B, agency credentials, creator portfolios, brand manifestos.
**Avoid:** Quiet or institutional contexts where the coral+dark palette reads as too aggressive.

---

### `pastel-pop`

**Mood:** Friendly, approachable, indie, warm
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Sun Peach | `#fdebd8` |
| text-primary | Deep Brown | `#2b1d0e` |
| text-muted | Warm Taupe | `#8a7060` |
| accent | Coral Pink | `#f4745b` |
| accent-alt | Mint Green | `#72c5a0` |

**Font pairing:**
- Display: `Syne` (Google Fonts — weight 700)
- Body: `DM Sans` (Google Fonts — weight 400)

```html
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&family=DM+Sans:wght@400&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #fdebd8;
  --color-text: #2b1d0e;
  --color-muted: #8a7060;
  --color-accent: #f4745b;
  --color-accent-alt: #72c5a0;
  --font-display: 'Syne', system-ui, sans-serif;
  --font-body: 'DM Sans', system-ui, sans-serif;
}
```

**When to use:** Creator portfolios, indie product launches, lifestyle brands, small-business pitches, community decks.
**Avoid:** Formal institutional contexts where warmth reads as a liability.

---

### `midnight-scholar`

**Mood:** Scholarly, literary, quiet, intellectual
**Scheme:** Dark

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Midnight Navy | `#10182b` |
| text-primary | Warm Ivory | `#f0e9d8` |
| text-muted | Dusty Periwinkle | `#8a95b5` |
| accent | Dusty Teal | `#5ba8a0` |

**Font pairing:**
- Display: `Cormorant Garamond` (Google Fonts — weight 700, italic 700)
- Body: `Jost` (Google Fonts — weight 400)

```html
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,700;1,700&family=Jost:wght@400&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #10182b;
  --color-text: #f0e9d8;
  --color-muted: #8a95b5;
  --color-accent: #5ba8a0;
  --font-display: 'Cormorant Garamond', Georgia, serif;
  --font-body: 'Jost', system-ui, sans-serif;
}
```

**When to use:** Research synthesis, white papers, academic presentations, advisory deliverables, founder reflections.
**Avoid:** High-energy, playful, or consumer-focused contexts.

---

### `clean-tech`

**Mood:** Precise, modern, technical, neutral
**Scheme:** Light

**Palette:**

| Role | Name | Hex |
|---|---|---|
| background | Soft White | `#f8f9fc` |
| text-primary | Dark Ink | `#0f1117` |
| text-muted | Cool Gray | `#5c6370` |
| accent | Signal Blue | `#2563eb` |

**Font pairing:**
- Display: `Space Grotesk` (Google Fonts — weight 700)
- Body: `Space Grotesk` (Google Fonts — weight 400, 500)

```html
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap" rel="stylesheet" />
```

```css
:root {
  --color-bg: #f8f9fc;
  --color-text: #0f1117;
  --color-muted: #5c6370;
  --color-accent: #2563eb;
  --font-display: 'Space Grotesk', system-ui, sans-serif;
  --font-body: 'Space Grotesk', system-ui, sans-serif;
}
```

**When to use:** Developer talks, SaaS product demos, technical reports, engineering team presentations.
**Avoid:** Creative, editorial, or emotionally expressive contexts where "clean" reads as cold.

---

## Preset Selection Quick Guide

| Deck purpose | Recommended presets |
|---|---|
| Investor / board / formal | `night-signal`, `cobalt-clean`, `warm-monochrome` |
| B2B SaaS / consulting | `cobalt-clean`, `clean-tech`, `ink-press` |
| Tech talk / developer | `studio-electric`, `clean-tech`, `night-signal` |
| Research / academic | `parchment-serif`, `warm-monochrome`, `midnight-scholar` |
| Brand / editorial | `ink-press`, `coral-editorial`, `neobrutalist-block` |
| Creative / design agency | `studio-electric`, `neobrutalist-block`, `coral-editorial` |
| Wellness / sustainability | `forest-quiet`, `pastel-pop`, `parchment-serif` |
| Creator / indie | `pastel-pop`, `neobrutalist-block`, `coral-editorial` |
| Literary / cultural | `midnight-scholar`, `parchment-serif`, `warm-monochrome` |

---

## Using a Preset in meta.json

Record the chosen preset slug in `meta.json`:

```json
{ "style": "night-signal" }
```

The oma-slide skill reads `meta.json.style` and applies the preset's CSS variables as a `<style>`
block at the top of each slide `<head>`. Presets are a starting point — the skill customizes
them per slide content and may extend with slide-specific overrides.

---

## Attribution

These presets are adapted from the `zarazhangrui/frontend-slides` project,
available at https://github.com/zarazhangrui/frontend-slides, released under the
**MIT License**.

> MIT License — Copyright (c) zarazhangrui
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify, merge,
> publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
> to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.

Palettes and font pairings have been adapted for the oma-slide 1920×1080 fixed-stage model
and extended with additional original presets.
