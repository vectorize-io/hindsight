# Spatial Design Reference

## Base Unit: 8px Grid

All spacing derives from an 8px base unit (4px for fine adjustments).

### Spacing Scale
| Token | Value | Use |
|-------|-------|-----|
| 1     | 4px   | Fine adjustments, icon padding |
| 2     | 8px   | Tight spacing, inline gaps |
| 3     | 12px  | Small component padding |
| 4     | 16px  | Default padding, gap between related items |
| 6     | 24px  | Section sub-spacing |
| 8     | 32px  | Component separation |
| 12    | 48px  | Section internal padding |
| 16    | 64px  | Section vertical padding (mobile) |
| 24    | 96px  | Section vertical padding (desktop) |
| 32    | 128px | Hero section padding |

## Grid System

- 12-column grid for page layouts
- Responsive gutters: 16px (mobile) → 24px (tablet) → 32px (desktop)
- Max content width: 1280px (xl) or 1024px (lg) for text-heavy content
- Always center content with `mx-auto`

## Responsive Breakpoints

| Name | Width  | Typical Use          | Columns |
|------|--------|----------------------|---------|
| sm   | 640px  | Large phones         | 1-2     |
| md   | 768px  | Tablets              | 2-3     |
| lg   | 1024px | Small laptops        | 3-4     |
| xl   | 1280px | Desktops             | 4+      |
| 2xl  | 1536px | Large displays       | 4+      |

## Section Spacing Patterns

| Section Type   | Mobile (py)      | Desktop (py)     |
|----------------|------------------|------------------|
| Hero           | py-24 (96px)     | py-40 (160px)    |
| Content        | py-16 (64px)     | py-32 (128px)    |
| Stats/CTA      | py-16 (64px)     | py-24 (96px)     |
| Footer         | py-8 (32px)      | py-12 (48px)     |

Horizontal padding: `px-4` (mobile) → `px-6 md:px-16 lg:px-24`

### Within Sections
- Between heading and content: space-y-4 (16px)
- Between content blocks: space-y-8 (32px) or space-y-12 (48px)
- Card grid gap: gap-4 (mobile) → gap-6 (desktop)

## Touch Targets (Mobile)

- Minimum tappable area: **44x44pt** (Apple HIG) / **48x48dp** (Material)
- Spacing between adjacent targets: minimum 8px
- Safe area: respect notch, home indicator, and rounded corners
- Thumb-friendly zones: place primary actions in bottom half of screen

## Content Width & Readability

- Optimal line length: 45-75 characters
- For full-width sections: constrain text with `max-w-md` or `max-w-lg`
- For centered hero text: `max-w-4xl` or `max-w-5xl`

## Anti-Patterns
- DON'T: Mix spacing values outside the 8px scale
- DON'T: Use padding less than 16px on mobile containers
- DON'T: Force identical card heights with arbitrary min-height
- DON'T: Ignore horizontal overflow on mobile
- DO: Maintain consistent section rhythm (same py across similar sections)
- DO: Test at 375px viewport width as minimum mobile target
- DO: Use `gap` instead of margins for grid/flex children
