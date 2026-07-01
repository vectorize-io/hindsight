---
description: Design system, typography, color, accessibility (WCAG 2.2),
  responsive-first layout, and motion design standards
---


# Design Standards

## Core Rules

1. System font stack as default (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`). Add custom fonts only with project justification.
2. If the service supports CJK languages: prioritize CJK-ready fonts (Pretendard Variable > Noto Sans CJK > system-ui fallback).
3. Name colors semantically with hex values: "Deep Ocean Navy (#1a2332)" not "dark blue".
4. Recommend components with install commands (shadcn CLI).
5. ALL output must be responsive-first (mobile layout as default, enhance upward).
6. WCAG AA minimum for all designs. Respect `prefers-reduced-motion`.
7. Present 2-3 design directions and get user confirmation before generating.

## Anti-Patterns to Avoid

- **Typography**: Don't default to custom Google Fonts; body text minimum 16px on mobile; test CJK at every size
- **Color**: Avoid purple-to-blue gradient backgrounds; avoid gradient orbs/blobs; use solid colors or subtle single-hue gradients
- **Layout**: 8px grid; works at 375px minimum; avoid nested cards inside nested cards
- **Motion**: transform + opacity only for 60fps; 150ms micro-interactions, 200-500ms transitions; respect prefers-reduced-motion
- **Components**: Use shadcn/ui for base; all interactive elements must have visible focus states
