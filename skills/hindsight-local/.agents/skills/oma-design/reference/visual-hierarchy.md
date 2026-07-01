# Visual Hierarchy Reference

## Why This Matters

Other reference files in this directory explain **how** to implement
design (pixel values, code snippets, library APIs). This file explains
**why** certain spatial, color, and temporal choices create effective
visual hierarchy. Use these 7 principles during Phase 4 PROPOSE to
reason about direction trade-offs, and during Phase 6 AUDIT to evaluate
whether the generated DESIGN.md produces a clear information hierarchy.

The principles below draw from Gestalt psychology, graphic design
pedagogy, and interaction design practice.

---

## 1. Alignment

**Items that are aligned are perceived as related.**

When elements share an edge or axis, the viewer groups them into a
coherent unit. Misalignment introduces visual noise and forces the
viewer to work harder to find structure.

- **Phase 4**: when proposing layout approaches, call out which
  elements share an alignment axis and why (e.g., "left-aligned heading
  and body text create a strong reading anchor").
- **Phase 5**: DESIGN.md Section 5 (Layout Principles) should specify
  alignment rules for key surfaces. Cards should share top edges,
  headings should align with body containers.

Practical checks:
- All cards in a grid share their top edge and internal padding origin.
- Navigation items are baseline-aligned with the logo.
- CTA buttons in a hero align horizontally with the subtitle's left
  edge (on left-aligned layouts).

---

## 2. Color

**Color differences establish visual weight and semantic priority.**

Higher saturation, warmer hues, and greater luminance contrast pull
attention. Muted, desaturated, or low-contrast elements recede. The
viewer reads the highest-contrast element first.

- **Phase 4**: explain how each proposed palette encodes hierarchy
  (e.g., "CTA green is the only high-saturation color on a desaturated
  surface; it draws the eye instantly").
- **Phase 5**: DESIGN.md Section 2 (Color Palette & Roles) must map
  every color to a hierarchy level (primary action > heading > body >
  metadata > decorative).

Relationship to `color-and-contrast.md`: that file covers palette
construction, WCAG ratios, and dark theme mechanics. This principle
covers **why** you assign a specific color to a specific role: the
hierarchy intent behind the assignment.

---

## 3. Contrast

**Juxtaposition of unlike elements heightens both.**

Contrast is not just a WCAG ratio; it is a deliberate design tool.
Warm vs cool, heavy vs light, dense vs sparse, serif vs sans-serif.
When two notably different things sit close together, both become more
visible and more memorable.

- **Phase 4**: use contrast intentionally: pair a bold serif display
  heading with a light sans-serif body, or a bright CTA on a muted
  surface. Explain the contrast pairing and its effect.
- **Phase 6**: audit for unintentional uniformity. If every component
  has the same visual weight, nothing stands out. The hierarchy is flat.

Types of contrast to leverage:
| Type | Example | Effect |
|---|---|---|
| Chromatic | Saturated CTA on desaturated surface | Eye is drawn to action |
| Typographic | Serif heading + sans body | Creates editorial tension |
| Scale | 96px display + 14px body | Extreme size ratio signals importance |
| Density | Packed data table + airy hero | Shift in pace guides reading |
| Motion | Single animated element in static context | Attention magnet |

---

## 4. Proximity

**Elements placed near each other are perceived as a group.**

This is the fundamental rationale behind spacing systems. A 16px gap
between a heading and its body says "these belong together"; a 64px gap
between two sections says "these are separate topics."

- **Phase 5**: DESIGN.md Section 5 (Layout Principles) must define at
  least two proximity tiers: intra-group spacing (tight, 8-16px) and
  inter-group spacing (loose, 48-96px). The ratio between them is more
  important than the absolute values.
- **Phase 6**: check that spacing does not accidentally merge unrelated
  elements or split related ones.

Relationship to `spatial-design.md`: that file provides the spacing
scale (4px → 128px) and grid system. This principle explains **when**
to pick which value: tight proximity for grouped elements, wide
proximity for section separation.

---

## 5. Size

**Larger elements are perceived as more important.**

Size is the most direct hierarchy signal. A 96px display heading
dominates a 14px caption. A 48x48 icon outweighs a 16x16 one.
Designers use size ratios (not absolute values) to establish the
relative importance of elements.

- **Phase 4**: when proposing type scales, explain the ratio between
  the largest and smallest visible elements. A 6:1 ratio (96px vs 16px)
  produces a dramatic hierarchy; a 2:1 ratio (32px vs 16px) produces a
  flatter, more editorial feel.
- **Phase 5**: DESIGN.md Section 3 (Typography Rules) table should
  make the size progression explicit: Display > H1 > H2 > Body > Small.
  The jump between adjacent levels should feel deliberate.

Accessibility note: size as hierarchy tool must coexist with
accessibility minimums. Body text ≥ 16px on mobile is a floor, not a
ceiling. Touch targets ≥ 44x44pt regardless of visual importance.

---

## 6. Texture

**Texture adds meaning beyond color and shape.**

Texture is an underused design tool. Most AI-generated designs default
to flat surfaces differentiated only by color. Deliberate texture
(noise, grain, paper, fabric, dither, stipple) creates visual richness
and can communicate material quality, surface hierarchy, and interactive
affordance.

> "Texture can be used to create highlights, instead of relying on
> color." (Jessica Cardona)

**Active uses of texture:**
- **Surface differentiation**: add subtle noise to a background to
  distinguish it from a card surface that is flat; this communicates
  depth without using shadow.
- **Brand warmth**: a grain overlay on hero photography creates a
  film/analog feel that humanizes a digital interface.
- **Interactive affordance**: a stippled or fabric texture on a
  draggable surface hints at tactile interaction.
- **Section separation**: alternating textured (noise) and flat
  (smooth) section backgrounds creates rhythm without color changes.

**This is NOT anti-pattern avoidance.** The `anti-patterns.md` line
"prefer texture over gradient" is a defensive rule. This principle is
offensive: texture is a first-class design choice, not a fallback when
gradients are banned.

- **Phase 4**: when proposing aesthetic directions, include a texture
  strategy ("hero noise at 3-5% opacity", "card surfaces are flat,
  page canvas has subtle grain"). Explain what the texture communicates.
- **Phase 5**: DESIGN.md Section 1 (Visual Theme & Atmosphere) should
  describe the texture language. Section 4 (Component Stylings) should
  note which surfaces are textured and which are flat.

---

## 7. Time

**Digital interfaces unfold over time; use that dimension.**

Unlike print, screens can change, react, and reveal. Time is a design
material. How information appears, sequences, and transforms is a
hierarchy signal equal to size, color, and position.

**Temporal UX patterns:**

- **Progressive disclosure**: show essential information first, reveal
  detail on demand. Reduces cognitive load. The viewer processes the
  hierarchy the designer intended, not a wall of simultaneous data.
- **Reveal sequence**: hero heading appears → then subtitle → then CTA.
  The stagger (100-200ms between elements) tells the viewer what to
  read first. This is temporal hierarchy, controlled by animation delay.
- **State transitions**: a button morphing from "Submit" to a loading
  spinner to a success checkmark communicates system status across time.
  Each state has its own visual hierarchy within the same space.
- **Scroll-driven narrative**: sections that animate into view as the
  user scrolls create a story arc. The order of appearance IS the
  hierarchy. What appears first matters most.
- **Micro-interactions**: a hover-triggered tooltip or a focus-ring
  glow creates a momentary hierarchy shift, "pay attention to this
  right now."

**This is broader than animation.** `motion-design.md` covers the
mechanics (GSAP timelines, motion.div, spring physics). This principle
covers the design intent: **why** you sequence elements this way, and
what hierarchy the temporal ordering communicates.

- **Phase 4**: when proposing motion strategies, describe the temporal
  hierarchy: what the viewer sees first, second, third. Not just
  "fade-in on scroll" but "headline arrives first to anchor the
  message, supporting stats appear 200ms later to reinforce it."
- **Phase 5**: DESIGN.md Section 9 (Agent Prompt Guide) component
  prompts should include timing relationships, not just static layout
  ("headline appears first at 0ms, subtitle at 200ms, CTA at 400ms").

---

## Using These Principles Together

No single principle creates hierarchy alone. Effective design layers
multiple signals:

```
A hero CTA button that is:
  - Large (Size: 48px tall, 18px text)
  - High-contrast (Color: saturated green on dark surface)
  - Isolated (Proximity: surrounded by 32px+ whitespace)
  - Aligned with the headline (Alignment: shares left edge)
  - Textured differently from the surface (Texture: solid fill vs
    noise background)
  - Appears last in the reveal sequence (Time: headline → subtitle → CTA)

…is unmistakably the primary action. Remove any one signal and it
still works. Add all six and the hierarchy is bulletproof.
```

When auditing a DESIGN.md in Phase 6, check whether key elements
(primary CTA, hero heading, navigation) use at least 3 of the 7
principles to establish their hierarchy position. If an element relies
on only one signal (e.g., just color), it is fragile and may fail for
users with color blindness or reduced motion preferences.
