# DESIGN.md Specification

## Purpose

DESIGN.md is the single source of truth for a project's visual language.
It is both human-readable AND machine-readable. It is portable across tools,
teams, and AI agents. When a coding agent reads DESIGN.md, it should be able
to generate code that matches the design system without additional guidance.

The 9-section structure below is aligned with the community
[getdesign](https://getdesign.md) schema so that templates fetched by
`resources/getdesign-fetcher.md` drop in as Phase 2 seeds without shape
mismatch. Downstream agents (`oma-frontend`, `frontend-engineer`) are
expected to read Section 9 ("Agent Prompt Guide") verbatim as
copy-paste instructions.

## 9-Section Structure

### 1. Visual Theme & Atmosphere

Evocative description of mood, density, and aesthetic philosophy.
This section is about intent and feeling, NOT technical specs.

Example:
> A sophisticated dark interface with gallery-like spaciousness.
> Clean geometric forms float on deep backgrounds with subtle glass effects.
> The atmosphere is confident and minimal; every element earns its place.

### 2. Color Palette & Roles

Each color entry must include:
- Descriptive Name + Hex Code + Functional Role

Format: `Descriptive Name (#hexcode): functional role`

Example:
```
- Deep Space Black (#0a0a0a): primary background, page canvas
- Warm Ivory (#f5f0eb): primary text on dark surfaces
- Signal Green (#22c55e): CTAs, success states, active indicators
- Soft Mist (rgba(255,255,255,0.1)): borders, dividers, glass surfaces
- Ember Red (#ef4444): error states, destructive actions
- Steel Gray (#6b7280): secondary text, placeholders, disabled states
```

Include shade scales (100-900) for primary and neutral colors if needed.
Organize into named subsections when the palette is large: Primary,
Accent, Interactive, Neutral Scale, Surface & Borders, Shadow Colors.

### 3. Typography Rules

Specify:
- Font families (with fallback stacks)
- Hierarchy as a **table** (mandatory format, not prose):

| Role | Font | Size | Weight | Line Height | Letter Spacing | Features | Notes |
|---|---|---|---|---|---|---|---|
| Display | Instrument Serif | clamp(48px, 8vw, 96px) | 400 italic | 0.9 | -0.04em | none | Hero only |
| H1 | Pretendard Variable | 40px | 700 | 1.2 | -0.02em | none | Section headings |
| Body | Pretendard Variable | 16px | 400 | 1.6 | 0 | ss01 | ≥16px on mobile |
| Code | JetBrains Mono | 13px | 500 | 1.5 | 0 | tnum | Data tables |

- Principles (letter-spacing rules for headings vs body, CJK line-height
  1.7-1.8, font-feature-settings where applicable).

> **Seed rule**: when a vendor template is merged via `getdesign-fetcher.md`,
> its Typography section is NOT adopted. Font selection always follows
> SKILL.md Rule #2 (system stack default) and Rule #3 (Pretendard Variable
> for CJK projects). Vendor type scale and tracking values may inform
> defaults but are not authoritative.

### 4. Component Stylings

Concrete specifications for core components. Each subsection must give
exact measurements, transition durations, and accessibility notes.

- **Buttons**: sizes, variants (primary/secondary/ghost), border-radius,
  padding, hover/active/focus states
- **Cards & Containers**: background, border, shadow, padding, border-radius
- **Badges & Pills**: background, text color, padding, radius, font size
- **Inputs & Forms**: height, border, focus states, error states,
  placeholder styling
- **Navigation**: height, layout, mobile behavior, glass effects
- **Decorative Elements**: dividers, icons, ornaments

### 5. Layout Principles

- Spacing System (base unit, full scale, typically 4px/8px grid)
- Grid & Container (12-column, responsive gutters, max content widths)
- Whitespace Philosophy (breathing room per section)
- Border Radius Scale (`sm`, `md`, `lg`, `xl`, `full`)

### 6. Depth & Elevation

- Shadow scale (elevation levels 0-5)
- Z-index scale (layers: base, dropdown, overlay, modal, toast, tooltip)
- Glass-surface rules (when blur is allowed and at what intensity)
- Light source convention (top-down vs multi-directional)

### 7. Do's and Don'ts

Project-specific rules that encode visual discipline. Pull from
`resources/anti-patterns.md` plus any brand-specific constraints from
`.design-context.md`. Write as bullet pairs:

```
- DO: Use solid brand purple (#5e6ad2) for primary CTAs.
- DON'T: Use purple gradients or purple-to-blue fades.
- DO: Glass surfaces only on navigation and accent badges.
- DON'T: Apply backdrop-blur to content sections or cards.
```

### 8. Responsive Behavior

- **Breakpoints** with pixel values (mobile 320-639, md 768+, lg 1024+, xl 1280+)
- **Touch targets**: minimum 44x44pt on mobile
- **Collapsing strategy**: what stacks, what hides, what resizes, what
  swaps (e.g., horizontal nav → hamburger, grid → single column)
- **Image behavior**: fluid width, aspect-ratio locks, art direction
- **Safe areas**: iOS notch, keyboard avoidance

### 9. Agent Prompt Guide

**This section is mandatory.** Downstream coding agents read this verbatim.
Keep it copy-paste ready, no prose padding.

#### Quick Color Reference

List every critical color with its role, for instant lookup:
```
- Primary CTA: Signal Green (#22c55e)
- CTA Hover: Green Dark (#16a34a)
- Background: Deep Space Black (#0a0a0a)
- Heading text: Warm Ivory (#f5f0eb)
- Body text: Soft Mist (rgba(255,255,255,0.72))
- Border: Soft Mist (rgba(255,255,255,0.1))
- Link: Signal Green (#22c55e)
- Success: Signal Green (#22c55e)
- Error: Ember Red (#ef4444)
- Surface: Near Black (#111111)
```

#### Example Component Prompts

Provide 4-6 ready-to-paste prompts covering hero, card, button, nav,
form, and footer. Each prompt must embed the exact values needed to
reproduce the component without referring back to other sections:

```
- "Build a hero section on #0a0a0a background. Display headline in
   Instrument Serif italic at clamp(48px,8vw,96px), weight 400,
   line-height 0.9, color #f5f0eb. Subtitle in Pretendard Variable
   18px weight 400, color rgba(255,255,255,0.72). Primary CTA: #22c55e
   background, #0a0a0a text, 12px radius, 14px 24px padding,
   transition background 150ms. Secondary CTA: transparent background,
   1px rgba(255,255,255,0.2) border, #f5f0eb text, same padding."
- "Build a card: #111111 background, 1px rgba(255,255,255,0.08) border,
   16px radius, 24px padding. Title in Pretendard Variable 20px weight
   600, color #f5f0eb. Body in Pretendard Variable 15px weight 400,
   color rgba(255,255,255,0.64), line-height 1.6."
```

#### Iteration Guide

5-8 numbered rules that capture the system's DNA, the things a
downstream agent must respect when iterating on the design:

```
1. Heading color is always #f5f0eb (warm ivory), never pure white.
2. Body opacity is 0.72 on dark surfaces; drop to 0.56 for metadata.
3. Border-radius ladder: 8px (inputs) → 12px (buttons) → 16px (cards)
   → 24px (modals). Never pill shapes except badges.
4. Glass surfaces only on navigation and badges, never content cards.
5. Motion: 150ms for hovers, 250ms for transitions, 400ms for entrances.
6. Mobile body size is 16px minimum; never shrink below.
7. Never stack more than two cards inside a single section.
```

## Writing Guidelines

- **Be Descriptive**: "Ocean-deep Cerulean (#0077B6)" not "blue"
- **Be Functional**: always explain what each element is *used for*
- **Be Precise**: include hex codes and pixel values in parentheses after descriptions
- **Be Consistent**: same terminology throughout the document
- **Be Portable**: no framework-specific syntax in descriptions (utility
  class names belong inside Section 9 prompts only)
- **English only**: DESIGN.md is a technical artifact. Prose sections
  stay in English even when the project's response language is localized.
  This mirrors the i18n rule that config keys, file paths, and machine
  contracts stay English.

## Extraction Pipeline (from existing designs)

When extracting DESIGN.md from an existing site, Stitch project, or
getdesign vendor template:

1. **Retrieval**: fetch source (HTML/CSS, Stitch screen data, or
   vendor template via `resources/getdesign-fetcher.md`)
2. **Extraction**: identify fonts, colors, spacing, component patterns
3. **Translation**: convert raw CSS values to semantic descriptions
4. **Synthesis**: organize into the 9 sections above
5. **Alignment**: verify consistency, resolve conflicts, fill gaps,
   apply the Seed rule in Section 3 (override vendor typography with
   project language strategy)
