# Accessibility Reference (WCAG 2.2)

## Contrast Requirements

| Level | Normal Text (< 18px bold / < 24px) | Large Text (>= 18px bold / >= 24px) | UI Components |
|-------|-------------------------------------|--------------------------------------|---------------|
| AA    | 4.5:1                               | 3:1                                  | 3:1           |
| AAA   | 7:1                                 | 4.5:1                                | 3:1           |

Focus indicators must have >= 3:1 contrast against adjacent colors.

### Testing Tools
- Browser DevTools: Accessibility panel → contrast ratio
- axe DevTools browser extension
- Lighthouse accessibility audit
- Color blindness simulation (Chrome DevTools → Rendering → Emulate vision deficiencies)

## Motion & Animation

### prefers-reduced-motion (MANDATORY)

All animations must respect this media query:

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

In motion/react:
```tsx
import { useReducedMotion } from "motion/react"

const prefersReduced = useReducedMotion()
// Disable or simplify animations when true
```

### Rules
- Never auto-play animations that can't be paused
- Provide a mechanism to pause all animations (e.g., a toggle)
- Flashing content: never flash more than 3 times per second
- Parallax scrolling must be disableable

## Semantic HTML

### Landmarks
Every page must use these semantic elements:
- `<header>`: site header with navigation
- `<nav>`: primary navigation
- `<main>`: main content (one per page)
- `<footer>`: site footer
- `<section>`: thematic grouping with heading
- `<article>`: self-contained content

### Headings
- Sequential hierarchy: h1 → h2 → h3 (never skip levels)
- One `<h1>` per page
- Headings must describe the section content

### Images
- Informative images: descriptive `alt` text
- Decorative images: `alt=""` (empty alt)
- Complex images: `alt` + `aria-describedby` for long description
- SVG icons: `role="img"` + `aria-label` or `aria-labelledby` with `<title>`

### Links & Buttons
- Links navigate: `<a href="...">`
- Buttons act: `<button>` or `<button type="submit">`
- Never use `<div onClick>` for interactive elements
- Link text must be descriptive (not "click here")

## ARIA Essentials

### Common Patterns
- **Icon-only buttons**: `aria-label="Close menu"`
- **Toggle buttons**: `aria-expanded="true/false"`
- **Modals**: `aria-modal="true"` + `role="dialog"` + `aria-labelledby`
- **Live updates**: `aria-live="polite"` for status messages
- **Loading states**: `aria-busy="true"` on the updating region
- **Navigation menus**: `aria-current="page"` on active link

### Rules
- Don't use ARIA when native HTML semantics suffice
- `aria-label` overrides visible text; use only when no visible label exists
- Test with screen readers (VoiceOver on Mac, NVDA on Windows)

## Focus Management

### Visible Focus Ring
```css
:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Remove default outline only when using :focus-visible */
:focus:not(:focus-visible) {
  outline: none;
}
```

### Focus Rules
- All interactive elements must be keyboard-focusable
- Focus order must follow visual/logical order
- Focus trap in modals and dialogs (Tab cycles within, Escape closes)
- Skip-to-content link as the first focusable element on every page:
```html
<a href="#main-content" class="sr-only focus:not-sr-only">
  Skip to main content
</a>
```

### After Navigation
- When opening a modal: focus the first interactive element or the close button
- When closing a modal: return focus to the trigger element
- After route change in SPA: focus the main heading or announce the new page

## Keyboard Navigation

All functionality must be operable via keyboard:
- `Tab` / `Shift+Tab`: navigate between focusable elements
- `Enter` / `Space`: activate buttons and links
- `Escape`: close modals, dropdowns, popovers
- `Arrow keys`: navigate within components (tabs, menus, sliders)
- Custom keyboard shortcuts: document them, make them discoverable

## Screen Reader Considerations

- Use `sr-only` (Tailwind) for visually hidden but screen-reader-accessible text
- Announce dynamic content changes with `aria-live` regions
- Avoid using CSS `content` for meaningful text
- Test that all information conveyed visually is also available to screen readers
