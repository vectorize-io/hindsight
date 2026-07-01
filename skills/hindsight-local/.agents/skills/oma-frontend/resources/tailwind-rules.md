# Tailwind CSS Rules

> Basic Tailwind syntax (spacing scale, flex/grid, breakpoints) is assumed knowledge. This file only covers project-specific rules and non-obvious patterns. See `snippets.md` for ready-made component snippets.

## Core Rules

1. **No inline styles**: Tailwind classes only
2. **Mobile-first**: base classes target mobile, scale up with `sm:` `md:` `lg:`
3. **No hardcoded colors**: always use semantic tokens (see below)
4. **Dark mode aware**: every color token must work under `dark:` variant
5. **Arbitrary values only with justification**: `w-[789px]` needs a design reason; otherwise use the scale

## Semantic Color Tokens

Map all colors to design tokens in `packages/design-tokens`. Never use raw colors like `bg-white` or `text-gray-900`.

```tsx
// Good
<div className="bg-background text-foreground border-border">
<button className="bg-primary text-primary-foreground">

// Bad
<div className="bg-white text-black border-gray-300">
```

Available tokens:
- `background` / `foreground`
- `primary` / `primary-foreground`
- `secondary` / `secondary-foreground`
- `muted` / `muted-foreground`
- `accent` / `accent-foreground`
- `destructive` / `destructive-foreground`
- `border` / `input` / `ring`

## `cn()` Utility

Use `cn()` (clsx + tailwind-merge) whenever classes are conditional or need to be overridable from a prop:

```tsx
import { cn } from "@/lib/utils";

<div className={cn(
  "base-class",
  condition && "conditional-class",
  { "variant-a": variant === "a" },
  className, // prop override must come last
)} />
```

## Focus States (A11y)

Every interactive element needs a visible focus state:

```tsx
className="focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
```

Screen-reader-only text: `<span className="sr-only">...</span>`.

## Tailwind v4 Theme

Tailwind v4 uses CSS-based configuration. `tailwind.config.js` is deprecated. Add custom tokens via `@theme` in `app/globals.css`:

```css
@import "tailwindcss";

@theme {
  --spacing-18: 4.5rem;
  --color-brand-50: #...;
  --color-brand-900: #...;
}
```

## Anti-Patterns

- **`@apply` in CSS files**: defeats utility-first intent. Use classes in JSX, or extract a component.
- **Hardcoded hex/rgb**: always go through the token system.
- **Desktop-first breakpoints**: `grid-cols-3 sm:grid-cols-1` reverses the mobile-first flow.
- **Nested overrides**: if you find yourself writing `lg:!bg-primary`, the component API is wrong; add a variant instead.

## Formatting

Follow whatever class-sorting convention the project's formatter enforces (Prettier + `prettier-plugin-tailwindcss`, Biome's `useSortedClasses`, or similar). Do not hand-reorder against the formatter.
