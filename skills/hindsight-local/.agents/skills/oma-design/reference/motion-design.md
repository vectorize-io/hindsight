# Motion Design Reference

## Animation Libraries & When to Use

> **`framer-motion` is BANNED.** It is the legacy package name for `motion`. Always import from `motion/react`, never from `framer-motion`. (Package install/manager rules live in the frontend rule.)

| Library         | Import                      | Best For                              |
|-----------------|-----------------------------|---------------------------------------|
| motion          | `from "motion/react"`       | Declarative, springs, layout, gestures|
| GSAP            | `from "gsap"`               | Timelines, ScrollTrigger, SplitText   |
| Three.js / R3F  | `from "@react-three/fiber"` | 3D scenes, physics (Rapier)           |
| ogl             | `from "ogl"`                | Lightweight WebGL shaders             |
| Lenis           | `from "lenis/react"`        | Smooth scroll                         |

### Decision Guide
- Simple entrance/exit → **motion**
- Scroll-linked transforms → **motion** (`useScroll` + `useTransform`)
- Complex timelines with sequencing → **GSAP** (TimelineMax, ScrollTrigger)
- Character/word splitting → **GSAP** (SplitText plugin)
- 3D objects or physics → **Three.js + R3F**
- GPU shader effects (backgrounds) → **ogl** (lighter than Three.js)
- Smooth page scrolling → **Lenis**

## Motion Principles

### Why Animate

Motion is not decoration; it is information. Every animation must
answer one of these questions for the viewer:

- **Where did this come from?** (origin: slide-in from source direction)
- **What changed?** (state: loading → success → done)
- **What should I look at?** (attention: pulse, scale, glow)
- **How are these related?** (connection: staggered entrance = group)
- **What happens next?** (sequence: progressive reveal = narrative)

If the animation does not answer any of these, remove it.

### Timing Guidelines

| Context | Duration | Easing | Why |
|---|---|---|---|
| Micro-interaction (hover, focus) | 100-150ms | ease-out | Instant feedback, never sluggish |
| UI transition (tab, panel) | 200-300ms | ease-out or spring | Visible but not blocking |
| Entrance animation | 300-500ms | spring or ease-out | Enough to read the motion path |
| Page transition | 400-600ms | ease-in-out | Narrative pacing between views |
| Stagger between siblings | 50-100ms | (inherit parent) | Creates reading order |

### Natural Feel

- Use spring physics or ease-out, never linear for UI elements.
  Linear motion looks robotic; springs feel physical.
- Overshoot (spring damping < 1) is fine for playful UIs; critically
  damped springs suit professional UIs.

### Restraint

- Maximum 2-3 animated elements visible simultaneously.
- If everything moves, nothing stands out; motion hierarchy collapses.
- The MOST important element should be the ONLY thing moving at the
  moment it needs attention.

## Temporal UX: Motion as Hierarchy

Animation duration and sequence are hierarchy signals equal to size,
color, and position. See `reference/visual-hierarchy.md` (Principle 7:
Time) for the theoretical framework. This section covers practical
application.

### Progressive Disclosure

Show essential information first, reveal detail on demand.

```tsx
// Phase 1: critical content visible immediately (no animation)
// Phase 2: supporting content fades in after 300ms
// Phase 3: detailed metadata available on hover/click

<motion.div initial={{ opacity: 1 }}>
  <h1>Primary message</h1>
</motion.div>
<motion.div
  initial={{ opacity: 0, y: 12 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ delay: 0.3 }}
>
  <p>Supporting context</p>
</motion.div>
```

Design rule: if a user sees the page for only 2 seconds, they should
grasp the primary message. Animation delay controls what registers
first.

### Reveal Sequence (Stagger as Narrative)

The order of appearance IS the hierarchy. Don't stagger for decoration;
stagger to tell a story:

```
Hero section reveal sequence:
  0ms     Badge ("Launching v2.0")      ← context first
  150ms   Headline                       ← primary message
  300ms   Subtitle                       ← supporting detail
  450ms   CTA buttons                    ← action last
```

Why this order: the badge primes the context ("this is new"), the
headline delivers the core message, the subtitle elaborates, and the
CTA appears only after the user understands what they're acting on.

Reversing this (CTA first, headline last) would be disorienting: the
user sees an action before understanding the context.

### State Machines (Temporal Transitions)

A single element can occupy multiple hierarchy positions over time:

```
Button states:
  Idle     → low hierarchy (neutral surface, awaits interaction)
  Hover    → elevated (subtle glow, cursor change)
  Active   → dominant (depressed, color shift)
  Loading  → dominant (spinner replaces label, blocks interaction)
  Success  → confirmatory (green check, then fades to idle)
```

Each state shift communicates system status. Design the transition
between states, not just the states themselves.

### Scroll-Driven Narrative

Sections that animate into view as the user scrolls create a story arc.
The viewport is a stage; scrolling is the user turning pages.

Design rules:
- Each section should have one primary entrance, not 5 elements
  animating independently. Use stagger within the section but treat
  the section as a narrative beat.
- Scroll-linked parallax (e.g., `useScroll` + `useTransform`) should
  reinforce depth, not just look cool. Background layers move slower
  than foreground; this is physically accurate parallax.
- Never hijack scroll (scroll-jacking). The user's scroll input should
  always produce proportional visual movement.

## Common Patterns

### Entrance Animation (motion/react)
```tsx
<motion.div
  initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
  animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
  transition={{ duration: 0.35, delay: index * 0.1 }}
/>
```

### Scroll-Driven Transform (motion/react)
```tsx
const { scrollYProgress } = useScroll()
const opacity = useTransform(scrollYProgress, [0, 0.5], [0, 1])
const y = useTransform(scrollYProgress, [0, 1], [100, 0])
```

### Staggered Children (motion/react)
```tsx
<motion.div
  variants={{
    show: { transition: { staggerChildren: 0.1 } }
  }}
  initial="hidden"
  animate="show"
>
  {items.map(item => (
    <motion.div
      key={item.id}
      variants={{
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 }
      }}
    />
  ))}
</motion.div>
```

### GSAP ScrollTrigger
```tsx
useGSAP(() => {
  gsap.from(el.current, {
    scrollTrigger: { trigger: el.current, start: "top 80%" },
    y: 50,
    opacity: 0,
    duration: 0.8,
    ease: "power2.out"
  })
})
```

### GSAP SplitText
```tsx
const split = new SplitText(el.current, { type: "chars,words" })
gsap.from(split.chars, {
  opacity: 0,
  y: 20,
  stagger: 0.03,
  duration: 0.5
})
```

### Reduced Motion
```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

```tsx
// motion/react
const prefersReducedMotion = useReducedMotion()
<motion.div
  animate={prefersReducedMotion ? {} : { y: [0, -10, 0] }}
/>
```

## Anti-Patterns
- DON'T: Bounce easing on everything (strongest motion AI slop signal)
- DON'T: Animation duration > 800ms for UI transitions
- DON'T: Animate layout-triggering properties (width, height, top, left)
- DON'T: Auto-play animations that can't be paused (a11y violation)
- DON'T: More than 2-3 animated elements visible simultaneously
- DON'T: Use `will-change` on everything; it consumes GPU memory
- DON'T: Use `linear` easing for UI elements; looks robotic
- DO: Animate only `transform` and `opacity` for 60fps
- DO: Always honor `prefers-reduced-motion` media query
- DO: Use `Intersection Observer` to trigger animations only when visible
- DO: Pause off-screen Canvas/WebGL with Intersection Observer
- DO: Remove `will-change` after animation completes
