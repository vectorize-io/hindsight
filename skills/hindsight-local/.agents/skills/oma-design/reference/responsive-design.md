# Responsive Design Reference

## Mobile-First Approach

All designs start with mobile layout as the default. Enhance upward with breakpoints.
Never produce desktop-only layouts.

- Content priority: most important content first in DOM order
- Touch-first interaction patterns
- Progressive enhancement: add complexity at larger breakpoints

## Theme System

### Light / Dark / System

```css
/* Light (default) */
:root {
  --background: 0 0% 100%;
  --foreground: 0 0% 3.9%;
}

/* Dark */
.dark {
  --background: 0 0% 3.9%;
  --foreground: 0 0% 98%;
}

/* System preference */
@media (prefers-color-scheme: dark) {
  :root:not(.light) {
    --background: 0 0% 3.9%;
    --foreground: 0 0% 98%;
  }
}
```

### Dark Theme Token Pattern
- **Surface layers**: bg-background → bg-card → bg-muted (increasing lightness)
- **Text layers**: text-foreground → text-muted-foreground → text-foreground/50
- **Borders**: use semantic `border-border`, not `border-gray-xxx`
- **Elevation**: opacity-based (bg-white/[0.01] → /[0.05]), not shadow-based

## Responsive Typography

- Use `clamp()` for fluid sizing between breakpoints
- Reduce heading sizes by 1-2 steps on mobile
- Increase `line-height` for CJK on all sizes (1.7-1.8)
- Minimum body text: 16px on mobile (never smaller)

```css
/* Fluid heading */
font-size: clamp(2rem, 1rem + 4vw, 5.5rem);

/* Fluid body */
font-size: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);
```

## Responsive Layout Patterns

| Pattern          | Mobile (default)  | Tablet (md)    | Desktop (lg+)     |
|------------------|-------------------|----------------|--------------------|
| Feature grid     | grid-cols-1       | grid-cols-2    | grid-cols-3 or 4   |
| Chess section    | stacked (col)     | stacked (col)  | side-by-side (row) |
| Stats            | grid-cols-2       | grid-cols-2    | grid-cols-4        |
| Testimonials     | grid-cols-1       | grid-cols-2    | grid-cols-3        |
| Nav              | hamburger/drawer  | hamburger      | full horizontal    |
| Hero buttons     | flex-col w-full   | flex-row       | flex-row           |
| Footer           | stacked           | grid-cols-2    | grid-cols-4        |

## Responsive Images & Media

- Use `srcset` and `sizes` for responsive images
- Video: provide poster fallback, consider hiding on mobile if heavy
- Background effects (shaders, Canvas): simplify or disable on mobile
- `object-fit: cover` for hero images/videos
- Lazy load below-the-fold images with `loading="lazy"`

## Responsive Spacing

| Element          | Mobile         | Desktop         |
|------------------|----------------|-----------------|
| Section padding  | py-16 px-4     | py-32 px-24     |
| Card padding     | p-4            | p-6 or p-8      |
| Grid gap         | gap-4          | gap-6           |
| Container        | px-4           | max-w-7xl mx-auto px-8 |

## Responsive Navigation Patterns

### Mobile (< 768px)
- Hamburger menu with slide-in drawer or full-screen overlay
- Bottom tab bar for app-like experiences
- Sticky header (slim, 48-56px height)

### Desktop (>= 1024px)
- Horizontal nav bar with glass pill container
- Floating navbar with scroll-based show/hide
- Sidebar for dashboard layouts

## Anti-Patterns
- DON'T: Generate fixed-width desktop-only layouts
- DON'T: Use hover-only interactions (no touch equivalent)
- DON'T: Hide critical content on mobile with `hidden md:block`
- DON'T: Use absolute pixel widths for containers
- DON'T: Ignore landscape orientation on mobile
- DO: Test at 375px width as minimum (iPhone SE)
- DO: Test at 320px for edge cases
- DO: Provide touch alternatives for all hover interactions
- DO: Use `min-h-[44px]` for all tappable elements on mobile
