---
name: oma-design
description: >
  AI design specialist skill with DESIGN.md management, anti-pattern enforcement,
  optional Stitch MCP integration, and component library guidance.
  Covers typography, color systems, motion design (motion/react, GSAP, Three.js),
  responsive-first layouts, and accessibility (WCAG 2.2).
---

# oma-design

## Scheduling

### Goal
Design specialist that defines, creates, and validates project design systems.
DESIGN.md is the central artifact; all design work revolves around it.

### Intent signature
- User asks for design system, `DESIGN.md`, visual direction, typography, color, motion, accessibility, anti-pattern review, or component guidance.
- User needs design decisions before frontend implementation or wants UI quality audited from a design perspective.

### When to use
- Defining or revising a project design system
- Creating or auditing `DESIGN.md`
- Selecting typography, color, layout, motion, or component direction
- Reviewing UI work for responsive behavior, accessibility, and visual quality
- Using optional vendor inspiration from Stitch MCP or getdesign

### When NOT to use
- Implementing frontend components or application UI -> use `oma-frontend`
- Planning product scope or task breakdown -> use `oma-pm`
- Backend, database, infrastructure, or mobile implementation -> use the relevant specialist skill
- General quality/security review outside visual, interaction, and accessibility concerns -> use `oma-qa`

### Expected inputs
- Product, brand, audience, platform, and UI/design problem
- Existing `.design-context.md`, `DESIGN.md`, screenshots, references, or component constraints
- Accessibility, responsive, language, and implementation constraints

### Expected outputs
- Design direction, revised `DESIGN.md`, audit findings, component guidance, or handoff notes
- Responsive-first, WCAG-aware design recommendations
- Optional vendor seed attribution when getdesign is used

```yaml
outputs:
  - name: design-doc
    description: Updated DESIGN.md when the run materially advances the design system
    artifact: "DESIGN.md"
    required: false
  - name: design-context
    description: Refreshed .design-context.md snapshot when a discovery pass runs
    artifact: ".design-context.md"
    required: false
```

### Dependencies
- `.design-context.md` and `DESIGN.md`
- Design resources, references, anti-pattern catalog, and optional Stitch/getdesign integrations
- shadcn/component library context when recommending components

### Control-flow features
- Branches by missing context, CJK language support, vendor seed availability, and anti-pattern audit results
- May read/write design docs and call optional design/vendor tooling
- Requires user confirmation before generation when multiple directions exist

## Structural Flow

### Entry
1. Check `.design-context.md`; if missing, run setup before design work.
2. Identify target audience, platform, content language, and design artifact.
3. Decide whether vendor inspiration or Stitch integration is relevant.

### Scenes
1. **PREPARE**: Load design context and constraints.
2. **ACQUIRE**: Extract existing design signals, references, and anti-pattern risks.
3. **REASON**: Propose directions, typography, color, layout, motion, and accessibility choices.
4. **ACT**: Generate or revise `DESIGN.md` and related guidance.
5. **VERIFY**: Audit responsive behavior, WCAG, Nielsen heuristics, and AI-slop patterns.
6. **FINALIZE**: Handoff design decisions and attribution where required.

### Transitions
- If `.design-context.md` is missing, create it before continuing.
- If CJK support is needed, prioritize CJK-ready fonts.
- If vendor seed fetch fails, choose retry, continue without seed, or abort.
- If anti-patterns appear, surface alternatives before finalizing.

### Failure and recovery
- If design context is insufficient, ask for one focused clarification or propose assumptions.
- If vendor inspiration is unavailable, continue with local design synthesis.
- If accessibility checks fail, revise before handoff.

### Exit
- Success: design artifact is project-specific, responsive-first, accessible, and audit-ready.
- Partial success: missing context, vendor failure, or open design decision is explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Read design context | `READ` | `.design-context.md`, `DESIGN.md`, references |
| Select design direction | `SELECT` | 2-3 directions and recommended option |
| Infer visual system | `INFER` | Typography, color, layout, motion |
| Call optional tooling | `CALL_TOOL` | Stitch/getdesign/shadcn when relevant |
| Write design artifact | `WRITE` | `DESIGN.md` or audit output |
| Validate design quality | `VALIDATE` | Checklist, WCAG, anti-patterns |
| Report handoff | `NOTIFY` | Final design summary |

### Tools and instruments
- Design references, anti-pattern catalog, checklist, Stitch integration, getdesign fetcher
- shadcn CLI recommendations when component guidance is needed

### Canonical workflow path
```text
1. Check `.design-context.md`; create it if missing.
2. Produce 2-3 design directions and get confirmation.
3. Generate or revise `DESIGN.md`, then run the design checklist.
```

Optional vendor seed discovery:
```bash
bunx getdesign@latest list
```

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `LOCAL_FS` | `.design-context.md`, `DESIGN.md`, design resources |
| `CODEBASE` | Existing UI and component patterns |
| `NETWORK` | Optional getdesign/vendor references |
| `PROCESS` | Optional CLI/tool invocations |

### Preconditions
- Target design problem and artifact are identifiable.
- Design context exists or setup can create it.

### Effects and side effects
- May create or modify `DESIGN.md` and design context artifacts.
- May fetch vendor seed material and append MIT attribution.
- Does not implement frontend code directly.

### Guardrails
1. Check `.design-context.md` before any design work. If missing, run Phase 1 (Setup) to create it.
2. System font stack as default (`system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`). Add custom fonts only with project justification.
3. If the service supports CJK languages (ko/ja/zh): prioritize CJK-ready fonts (Pretendard Variable > Noto Sans CJK > system-ui fallback). If latin-only: choose fonts appropriate for the target audience.
4. Enforce anti-patterns strictly; reject AI slop. See `resources/anti-patterns.md`.
5. Name colors semantically with hex values: "Deep Ocean Navy (#1a2332)" not "dark blue".
6. Recommend components with install commands (shadcn CLI).
7. ALL output must be responsive-first (mobile layout as default, enhance upward).
8. WCAG AA minimum for all designs. Respect `prefers-reduced-motion`.
9. Stitch MCP is optional; all phases work without it.
10. Present 2-3 design directions and get user confirmation before generating.

### Anti-Pattern Quick Reference

### Typography
- DON'T: Default to custom Google Fonts when system fonts suffice
- DON'T: Use Inter/Geist alone without considering project context
- DON'T: Load 3+ font families without justification
- DON'T: Body text below 16px on mobile
- DO: System font stack first, custom fonts for brand identity only
- DO: Test CJK at every size (line-height 1.7-1.8)

### Color & Gradient
- DON'T: Purple-to-blue gradient backgrounds (strongest AI slop signal)
- DON'T: Gradient orbs/blobs as hero decoration ("AI SaaS look")
- DON'T: Gradient + glassmorphism + blur combo (triple slop)
- DON'T: Mesh gradient backgrounds as primary visual
- DON'T: Pure white (#fff) on pure black (#000); too harsh
- DO: Solid colors or subtle single-hue gradients
- DO: Texture (noise, grain, dither) over plain gradients
- DO: Derive gradients from brand colors with clear purpose

### Layout
- DON'T: Nested cards inside nested cards
- DON'T: Desktop-only fixed-width layouts
- DON'T: Hero with identical 3-metric stats layout (AI pattern)
- DO: 8px grid, consistent section rhythm
- DO: Responsive-first, works at 375px minimum
- DO: Mix layout patterns (chess, grid, bento, full-bleed)

### Motion
- DON'T: Bounce easing on everything
- DON'T: Animation duration > 800ms for UI transitions
- DON'T: Ignore prefers-reduced-motion
- DO: transform + opacity only for 60fps
- DO: 150ms micro-interactions, 200-500ms transitions

### Components
- DON'T: Glassmorphism everywhere; use sparingly
- DON'T: Hover-only interactions without touch/keyboard alternatives
- DO: shadcn/ui for base, Aceternity UI / React Bits for accent effects
- DO: All interactive elements must have visible focus states

### Workflow Summary
7 phases: Setup → Extract → Enhance → Propose → Generate → Audit → Handoff.
See `resources/execution-protocol.md` for full detail.

### Vendor Inspiration (getdesign)

Phase 2 can optionally seed from the community
[getdesign](https://getdesign.md) catalog
([VoltAgent/awesome-design-md](https://github.com/VoltAgent/awesome-design-md),
MIT). Trigger it by listing a supported vendor domain in the
`## Reference Sites` section of your `.design-context.md`:

```markdown
## Reference Sites
- [linear.app](https://linear.app): clean dark UI, minimal, professional
- [stripe.com](https://stripe.com): strong hierarchy, purposeful animation
```

Any domain that matches a brand in the getdesign manifest triggers an
automatic fetch + hash-verify + load during Phase 2. No new fields, no
extra config. Full vendor list: see `bunx getdesign@latest list`
(telemetry is always disabled by oma-design).

**Seed, not final.** oma-design treats vendor templates as inspiration
and synthesizes a project-specific DESIGN.md around them. Importantly:
- **Typography is never adopted from the seed.** Rule #2 (system font
  stack default) and Rule #3 (Pretendard Variable / Noto Sans CJK for
  ko/ja/zh) always win over the vendor's latin-only fonts.
- **Anti-patterns are pre-audited** before synthesis. If a vendor uses
  heavy glassmorphism or purple gradients, Phase 4 will surface the
  choice explicitly rather than copy the pattern silently.
- **Offline is fine.** If the fetch fails, you get a 3-option dialog
  (retry / continue without seed / abort). Default: continue.

Attribution is appended to the generated `DESIGN.md` in Phase 7 as a
required MIT compliance footer. Full fetcher rules, matching algorithm,
injection defenses, and multi-vendor merge policy live in
`resources/getdesign-fetcher.md`.

### Resources
- `resources/execution-protocol.md`: 7-phase workflow
- `resources/anti-patterns.md`: Full DO/DON'T catalog
- `resources/checklist.md`: Audit checklist (Responsive + WCAG + Nielsen + Slop)
- `resources/design-md-spec.md`: DESIGN.md generation guide (9 sections)
- `resources/design-tokens.md`: CSS/Tailwind/shadcn export templates
- `resources/prompt-enhancement.md`: Vague request to detailed spec
- `resources/stitch-integration.md`: Stitch MCP tool mapping (optional)
- `resources/getdesign-fetcher.md`: Vendor seed fetch, hash verify, seed rules
- `resources/error-playbook.md`: Design error recovery

## References
- `reference/visual-hierarchy.md`: 7 hierarchy principles (Alignment, Color, Contrast, Proximity, Size, Texture, Time)
- `reference/typography.md`: Font selection, type scale, CJK
- `reference/color-and-contrast.md`: Color psychology, WCAG contrast
- `reference/spatial-design.md`: 8px grid, breakpoints, spacing
- `reference/motion-design.md`: motion/react, GSAP, Three.js, ogl, Temporal UX
- `reference/responsive-design.md`: Mobile-first, theme system
- `reference/component-patterns.md`: shadcn/Aceternity/React Bits catalog
- `reference/accessibility.md`: WCAG 2.2, ARIA, focus, reduced-motion
- `reference/shader-and-3d.md`: WebGL, R3F, ogl, performance

### Examples
- `examples/design-context-example.md`: .design-context.md example
- `examples/landing-page-prompt.md`: Detailed landing page prompt
