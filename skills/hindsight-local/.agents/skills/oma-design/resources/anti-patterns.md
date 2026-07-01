# Anti-Patterns: AI Design Slop Detection

> "If you showed this interface to someone and said 'AI made this,'
>  would they believe you immediately? If yes, that's the problem."

## Typography
- DON'T: Default to custom Google Fonts when system fonts suffice
- DON'T: Use Inter or Geist alone without considering project context
- DON'T: Load 3+ font families without clear justification
- DON'T: Set body text below 16px on mobile
- DON'T: Use light font-weight (300) for body on dark backgrounds without testing contrast
- DON'T: Apply identical letter-spacing to headings and body
- DON'T: Use latin-only fonts when the service targets CJK users
- DO: Start with system font stack, add custom fonts only for brand identity
- DO: For CJK services, add Pretendard or Noto Sans CJK explicitly
- DO: Use modular type scale with clamp() for fluid sizing
- DO: Test CJK characters at every size (line-height 1.7-1.8)
- DO: Use font-display: swap to prevent FOIT
- DO: Subset fonts to needed character ranges for performance

## Color & Gradient
- DON'T: Purple-to-blue linear gradient backgrounds (strongest AI slop signal)
- DON'T: Purple-to-pink gradient text
- DON'T: Rainbow/multi-stop gradient borders
- DON'T: Gradient orbs/blobs as hero decoration ("AI SaaS look")
- DON'T: Mesh gradient backgrounds as primary visual
- DON'T: Gradient + glassmorphism + blur combo (triple AI slop)
- DON'T: Gray text on colored backgrounds without checking contrast
- DON'T: Pure white (#fff) on pure black (#000); too harsh, causes eye strain
- DON'T: Rely on color alone to convey meaning (accessibility violation)
- DO: Use solid colors or subtle single-hue gradients
- DO: Derive gradients from brand colors with clear functional purpose
- DO: Prefer texture (noise, grain, dither) over gradient for visual interest
- DO: Use gradients only for functional purposes (fade overlays, depth cues)
- DO: Name colors semantically ("Deep Ocean Navy #1a2332" not "dark blue")
- DO: Test with color blindness simulators before finalizing palette

## Layout & Space
- DON'T: Nested cards inside nested cards
- DON'T: Mix spacing values outside the 8px grid scale
- DON'T: Hero sections with identical 3-metric stats layout (AI pattern)
- DON'T: Generate fixed-width desktop-only layouts
- DON'T: Use padding less than 16px on mobile containers
- DON'T: Force identical card heights with arbitrary min-height
- DON'T: Use the same card grid layout for every section
- DO: Consistent section rhythm (same vertical padding across similar sections)
- DO: Responsive-first: mobile layout as default, enhance upward
- DO: Every section must work at 375px width minimum
- DO: Mix layout patterns within a page (chess, grid, bento, full-bleed)
- DO: Use gap instead of margins for grid/flex children

## Motion & Animation
- DON'T: Bounce easing on everything (strongest motion AI slop signal)
- DON'T: Animation duration > 800ms for UI transitions
- DON'T: Animate layout-triggering properties (width, height, top, left)
- DON'T: Auto-play animations that cannot be paused (a11y violation)
- DON'T: More than 2-3 animated elements visible simultaneously
- DON'T: Use will-change on everything; consumes GPU memory
- DON'T: Use linear easing for UI elements; looks robotic
- DO: Animate only transform and opacity for 60fps
- DO: 150ms for micro-interactions, 200-500ms for transitions
- DO: Always honor prefers-reduced-motion media query
- DO: Use Intersection Observer to trigger animations only when visible
- DO: Pause off-screen Canvas/WebGL renderers

## Components
- DON'T: Glassmorphism on every element; reserve for badges, nav pills, accent cards
- DON'T: Icon + Title + Description card grid as the only layout pattern
- DON'T: Hover-only interactions without touch/keyboard alternatives
- DON'T: Identical card heights forced with arbitrary min-height
- DO: Mix section patterns within a page (chess + grid + stats + testimonials)
- DO: Choose component libraries intentionally (shadcn for base, Aceternity/React Bits for accents)
- DO: All interactive elements must have visible focus states
- DO: Include install commands when recommending components
