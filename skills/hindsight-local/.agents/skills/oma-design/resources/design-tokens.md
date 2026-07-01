# Design Token Export Templates

## CSS Custom Properties

```css
:root {
  /* Colors */
  --color-background: #0a0a0a;
  --color-foreground: #f5f0eb;
  --color-primary: #22c55e;
  --color-primary-foreground: #0a0a0a;
  --color-secondary: #1a1a2e;
  --color-muted: #6b7280;
  --color-muted-foreground: #9ca3af;
  --color-border: rgba(255, 255, 255, 0.1);
  --color-error: #ef4444;
  --color-warning: #f59e0b;
  --color-success: #22c55e;
  --color-info: #3b82f6;

  /* Typography */
  --font-body: system-ui, -apple-system, BlinkMacSystemFont,
               'Segoe UI', Roboto, sans-serif;
  --font-heading: 'Instrument Serif', Georgia, serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Spacing (8px grid) */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-6: 24px;
  --space-8: 32px;
  --space-12: 48px;
  --space-16: 64px;
  --space-24: 96px;
  --space-32: 128px;

  /* Border Radius */
  --radius-sm: 0.375rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-xl: 1rem;
  --radius-2xl: 1.5rem;
  --radius-3xl: 2rem;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.15);
  --shadow-glass: inset 0 1px 1px rgba(255, 255, 255, 0.1);

  /* Transitions */
  --duration-fast: 150ms;
  --duration-normal: 300ms;
  --duration-slow: 500ms;
  --easing-default: cubic-bezier(0.4, 0, 0.2, 1);
  --easing-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

## Tailwind Config Extension

```ts
// tailwind.config.ts
import type { Config } from "tailwindcss"

export default {
  theme: {
    extend: {
      fontFamily: {
        body: ["var(--font-body)"],
        heading: ["var(--font-heading)"],
        mono: ["var(--font-mono)"],
      },
      colors: {
        background: "var(--color-background)",
        foreground: "var(--color-foreground)",
        primary: {
          DEFAULT: "var(--color-primary)",
          foreground: "var(--color-primary-foreground)",
        },
        secondary: "var(--color-secondary)",
        muted: {
          DEFAULT: "var(--color-muted)",
          foreground: "var(--color-muted-foreground)",
        },
        border: "var(--color-border)",
        error: "var(--color-error)",
        warning: "var(--color-warning)",
        success: "var(--color-success)",
        info: "var(--color-info)",
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
        "2xl": "var(--radius-2xl)",
        "3xl": "var(--radius-3xl)",
      },
      transitionDuration: {
        fast: "var(--duration-fast)",
        normal: "var(--duration-normal)",
        slow: "var(--duration-slow)",
      },
    },
  },
} satisfies Config
```

## shadcn/ui Theme Variables

When using shadcn/ui, map tokens to its expected HSL format in `globals.css`:

```css
@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 0 0% 3.9%;
    --primary: 142 71% 45%;
    --primary-foreground: 0 0% 3.9%;
    --secondary: 240 10% 10%;
    --muted: 240 5% 46%;
    --muted-foreground: 240 5% 65%;
    --border: 0 0% 100% / 0.1;
    --ring: 142 71% 45%;
    --radius: 0.75rem;
  }

  .dark {
    --background: 0 0% 3.9%;
    --foreground: 30 10% 95%;
    --primary: 142 71% 45%;
    --primary-foreground: 0 0% 3.9%;
    --secondary: 240 10% 10%;
    --muted: 240 5% 46%;
    --muted-foreground: 240 5% 65%;
    --border: 0 0% 100% / 0.1;
    --ring: 142 71% 45%;
  }
}
```

## Usage Notes

- These are **templates**; replace values with project-specific colors, fonts, and spacing
- Always derive tokens from the DESIGN.md specification
- CSS custom properties are the source of truth; Tailwind config references them via `var()`
- For CJK projects, update `--font-body` to include Pretendard or Noto Sans CJK
- shadcn/ui expects HSL values without the `hsl()` wrapper (Tailwind adds it)
