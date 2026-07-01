# Color & Contrast Reference

## Color System Design

Define 5-7 core colors with semantic roles. Each color entry must include:
- Descriptive Name + Hex Code + Functional Role

Example:
```
- Deep Space Black (#0a0a0a): primary background, page canvas
- Warm Ivory (#f5f0eb): primary text on dark surfaces
- Signal Green (#22c55e): CTAs, success states, active indicators
- Soft Mist (rgba(255,255,255,0.1)): borders, dividers, glass surfaces
- Ember Red (#ef4444): error states, destructive actions
```

Generate 100-900 shade scales for primary and neutral colors.
Include functional colors: success (green), warning (amber), error (red), info (blue).

## Color Psychology Quick Reference

| Color    | Association              | Best For                     |
|----------|--------------------------|------------------------------|
| Blue     | Trust, stability         | Finance, enterprise, health  |
| Green    | Growth, nature           | Sustainability, fintech      |
| Purple   | Premium, creative        | Luxury, creative tools       |
| Red      | Urgency, energy          | Food, entertainment, sales   |
| Orange   | Warmth, friendliness     | Community, education         |
| Black    | Sophistication, power    | Luxury, fashion, premium SaaS|
| White    | Clean, minimal           | Healthcare, productivity     |
| Teal     | Balance, clarity         | Wellness, communication      |

## WCAG 2.2 Contrast Requirements

| Level | Normal Text | Large Text (18px bold / 24px+) | UI Components |
|-------|-------------|-------------------------------|---------------|
| AA    | 4.5:1       | 3:1                           | 3:1           |
| AAA   | 7:1         | 4.5:1                         | 3:1           |

Focus indicators: 3:1 contrast against adjacent colors.

Tools: use `oklch()` or `hsl()` for programmatic shade generation.
Test with color blindness simulators (protanopia, deuteranopia, tritanopia).

## Dark Theme Patterns

### Background Layers
- Canvas: pure black (#000) or near-black (hsl(260 87% 3%))
- Surface: slightly lighter (hsl(240 6% 9%))
- Elevated: bg-white/[0.03] to bg-white/[0.06]

### Text on Dark
- Primary text: warm off-white (#f5f0eb or #e8e4df), NOT pure white (#fff)
- Secondary text: 60% opacity (text-white/60)
- Tertiary/muted: 40% opacity (text-white/40)

### Borders & Dividers
- Use white at 10-20% opacity: `border-white/10`, `border-white/20`
- Never use gray hex values on dark backgrounds; use alpha transparency

### Elevation
- Convey depth through opacity, not shadow
- `bg-white/[0.01]` → `bg-white/[0.03]` → `bg-white/[0.05]` (increasing elevation)
- Reserve box-shadow for glass effects only

### Glass Effects
```css
backdrop-filter: blur(4px);
background: rgba(255, 255, 255, 0.01);
box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1);
```

## Anti-Patterns
- DON'T: Purple-to-blue linear gradient backgrounds (strongest AI slop signal)
- DON'T: Purple-to-pink gradient text
- DON'T: Rainbow/multi-stop gradient borders
- DON'T: Gradient orbs/blobs as hero decoration ("AI SaaS look")
- DON'T: Mesh gradient backgrounds as primary visual
- DON'T: Gradient + glassmorphism + blur combo (triple AI slop)
- DON'T: Gray text on colored backgrounds without checking contrast
- DON'T: Pure white (#fff) on pure black (#000); too harsh, causes eye strain
- DON'T: Rely on color alone to convey meaning (accessibility)
- DO: Use solid colors or subtle single-hue gradients
- DO: Derive gradients from brand colors with clear purpose
- DO: Prefer texture (noise, grain, dither) over gradient for visual interest
- DO: Use gradients only for functional purposes (fade overlays, depth cues)
- DO: Name colors semantically ("Deep Ocean Navy #1a2332" not "dark blue")
- DO: Test with color blindness simulators before finalizing palette
