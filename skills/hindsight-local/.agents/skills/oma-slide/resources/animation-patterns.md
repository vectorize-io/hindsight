# Animation Patterns — oma-slide

> Effect-to-feeling guide, CSS entrance patterns, background effects, and reduced-motion guards.
> All patterns are CSS-only and paste-able into slide `<style>` blocks.

## Ground Rules

1. Use `transform` + `opacity` only — these compose on the GPU without layout reflow.
2. Every animated element **must** be wrapped in `@media (prefers-reduced-motion: no-preference)`.
3. Keep durations short: **150ms** for micro-interactions, **200–500ms** for entrance transitions.
4. Do not combine more than two animation properties on a single element.
5. Use `animation-fill-mode: both` so elements start in their pre-animated state (no flash).
6. Stagger sibling elements with `animation-delay` increments of 60–80ms.

---

## 1. Effect-to-Feeling Guide

Choose an animation category that matches the deck's intended emotional register.

| Feeling | Category | When to use |
|---|---|---|
| Dramatic / cinematic | scale-in + fade, blur-reveal | Opening slides, key reveal moments, bold poster-style decks |
| Techy / precise | clip-reveal, horizontal wipe | Data slides, product demos, developer talks |
| Playful / energetic | bounce-up, scale-pop, color-swap | Consumer brands, indie launches, creative reviews |
| Corporate / polished | fade-up (slow), subtle slide-right | B2B pitches, consulting deliverables, investor decks |
| Calm / editorial | fade-only (very slow), typewriter | White papers, research synthesis, literary editorial |
| Minimal / restrained | opacity-only, no motion | Monochrome decks, dense reading-first slides |

---

## 2. Entrance Patterns

### 2a. Fade-Up (default, versatile)

Content rises from 20 px below while fading in. The safest, most readable entrance.

```css
@keyframes fade-up {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: no-preference) {
  .enter-fade-up {
    animation: fade-up 0.4s cubic-bezier(0.22, 0.61, 0.36, 1) both;
  }

  /* Stagger children */
  .enter-fade-up:nth-child(1) { animation-delay: 0ms; }
  .enter-fade-up:nth-child(2) { animation-delay: 70ms; }
  .enter-fade-up:nth-child(3) { animation-delay: 140ms; }
  .enter-fade-up:nth-child(4) { animation-delay: 210ms; }
}
```

### 2b. Scale-In (dramatic, bold openers)

Element scales from 92% to 100% while fading in. Use for single focal elements, never for
body text lists.

```css
@keyframes scale-in {
  from { opacity: 0; transform: scale(0.92); }
  to   { opacity: 1; transform: scale(1); }
}

@media (prefers-reduced-motion: no-preference) {
  .enter-scale-in {
    animation: scale-in 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  }
}
```

### 2c. Blur-In (cinematic, editorial)

Content materializes from a soft blur. Effective for headline-only slides, atmosphere-first
designs, and transitions between major sections.

```css
@keyframes blur-in {
  from { opacity: 0; filter: blur(8px); transform: scale(1.02); }
  to   { opacity: 1; filter: blur(0);   transform: scale(1); }
}

@media (prefers-reduced-motion: no-preference) {
  .enter-blur-in {
    animation: blur-in 0.55s cubic-bezier(0.25, 0.46, 0.45, 0.94) both;
  }
}
```

### 2d. Slide-Right (horizontal reveal, techy/data)

Content enters from the left. Use for sequential reveals on data slides or step-by-step
process flows.

```css
@keyframes slide-right {
  from { opacity: 0; transform: translateX(-32px); }
  to   { opacity: 1; transform: translateX(0); }
}

@media (prefers-reduced-motion: no-preference) {
  .enter-slide-right {
    animation: slide-right 0.35s cubic-bezier(0.22, 0.61, 0.36, 1) both;
  }
}
```

### 2e. Scale-Pop (playful, emphatic)

A slight overshoot spring — use for badges, numbers, icons, and callout elements.
Do NOT use for body text (legibility suffers during overshoot).

```css
@keyframes scale-pop {
  0%   { opacity: 0; transform: scale(0.8); }
  70%  { opacity: 1; transform: scale(1.05); }
  100% { transform: scale(1); }
}

@media (prefers-reduced-motion: no-preference) {
  .enter-scale-pop {
    animation: scale-pop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
  }
}
```

### 2f. Fade-Only (calm, restrained)

Pure opacity transition. For editorial and reading-first decks where motion should be invisible.

```css
@media (prefers-reduced-motion: no-preference) {
  .enter-fade {
    animation: fade-in 0.6s ease both;
  }
}

@keyframes fade-in {
  from { opacity: 0; }
  to   { opacity: 1; }
}
```

---

## 3. Background Effects

Background effects run continuously on the z-index 0 / 10 layers. They should be subtle —
the content layer at z-index 100 must always be the primary focus.

### 3a. Gradient Pulse (atmosphere, dark decks)

Slow, looping radial gradient shift. Use on dark-scheme slides for depth without distraction.

```css
@keyframes gradient-pulse {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@media (prefers-reduced-motion: no-preference) {
  .bg-gradient-pulse {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    background-size: 300% 300%;
    animation: gradient-pulse 12s ease infinite;
  }
}

/* Without motion preference: show static midpoint */
@media (prefers-reduced-motion: reduce) {
  .bg-gradient-pulse {
    background: #16213e;
  }
}
```

### 3b. Noise Overlay (tactile, print-like)

A pseudo-random grain overlay using SVG turbulence. Adds warmth to solid-color backgrounds.
Apply on the atmosphere layer (z-index 10), with `pointer-events: none`.

```css
.bg-noise-overlay {
  position: absolute;
  inset: 0;
  z-index: 10;
  pointer-events: none;
  opacity: 0.04;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
  background-repeat: repeat;
  background-size: 200px 200px;
}
```

### 3c. Scan Line (retro, techy)

Horizontal scan-line texture. For 8-bit / retro-tech / CRT aesthetics.

```css
.bg-scanlines {
  position: absolute;
  inset: 0;
  z-index: 10;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0, 0, 0, 0.12) 2px,
    rgba(0, 0, 0, 0.12) 4px
  );
}
```

### 3d. Vignette (cinematic, edge darkening)

Darkens the slide edges, drawing attention to the center. Use on photo-background slides.

```css
.bg-vignette {
  position: absolute;
  inset: 0;
  z-index: 10;
  pointer-events: none;
  background: radial-gradient(
    ellipse at center,
    transparent 50%,
    rgba(0, 0, 0, 0.45) 100%
  );
}
```

---

## 4. Slide Transition Hint

`deck-stage.js` manages slide visibility (`.active` / `.visible` classes). The crossfade timing
is controlled by a CSS custom property:

```css
/* In viewport-base.css or a per-deck <style> block */
:root {
  --slide-transition-duration: 300ms;  /* default */
}
```

Reduce to `0ms` for instant cuts (editorial, data-heavy decks). Increase to `500ms` for
cinematic transitions (title sequences, dramatic openers).

Under `prefers-reduced-motion: reduce`, `deck-stage.js` overrides this to `0ms` regardless
of the set value — no need to conditionally set it.

---

## 5. Reduced-Motion Guard Reference

Always wrap CSS animations in this media query:

```css
@media (prefers-reduced-motion: no-preference) {
  /* Animation / transition rules here */
}
```

For properties that should have a fallback static state:

```css
/* Default (no motion): static state */
.hero-headline { opacity: 1; transform: none; }

/* Motion allowed: animated entry */
@media (prefers-reduced-motion: no-preference) {
  .hero-headline {
    opacity: 0;
    transform: translateY(24px);
    animation: fade-up 0.5s 0.1s cubic-bezier(0.22, 0.61, 0.36, 1) forwards;
  }
}
```

This pattern ensures content is always visible regardless of user preference — the animation
is an enhancement, not a requirement for content visibility.

---

## 6. Timing Reference

| Use case | Duration | Easing |
|---|---|---|
| Micro-interaction (hover, focus) | 150ms | `ease` |
| Entrance — default | 350–450ms | `cubic-bezier(0.22, 0.61, 0.36, 1)` |
| Entrance — dramatic | 450–600ms | `cubic-bezier(0.25, 0.46, 0.45, 0.94)` |
| Entrance — spring/pop | 400ms | `cubic-bezier(0.34, 1.56, 0.64, 1)` |
| Background pulse (continuous) | 10–15s | `ease infinite` |
| Slide crossfade | 200–400ms | `ease` |
| Reduced-motion override | 0ms | — |
