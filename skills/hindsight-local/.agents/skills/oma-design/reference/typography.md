# Typography Reference

## Default: System Font Stack

Unless the project has specific brand typography requirements, prefer the system font stack for optimal performance and native feel:

```css
--font-body: system-ui, -apple-system, BlinkMacSystemFont,
             'Segoe UI', Roboto, sans-serif;
```

Only add custom fonts when:
- Brand guidelines require specific typefaces
- Display/heading fonts for visual identity (e.g., Instrument Serif for hero text)
- The service targets CJK users → add Pretendard or Noto Sans CJK

## Font Selection by Language Support

### CJK Services (ko/ja/zh)
```css
--font-body: 'Pretendard Variable', 'Pretendard',
             'Noto Sans KR', system-ui, sans-serif;
--font-heading: 'Instrument Serif', 'Pretendard', serif;
```

### Latin-Only
```css
--font-body: system-ui, -apple-system, sans-serif;
--font-heading: 'Instrument Serif', Georgia, serif;
```

### Multilingual
```css
--font-body: 'Noto Sans', system-ui, sans-serif;
```
Noto Sans covers 1000+ languages, best for international products.

## Font Pairing Patterns
- **Display + Body**: serif heading + sans-serif body (classic contrast)
- **Mono accent**: JetBrains Mono, Fira Code (for code/data-heavy UIs)
- **Variable fonts preferred**: fewer HTTP requests, finer weight/width control

## Type Scale

Use a modular scale for consistent hierarchy:
- **Major Third (1.250)**: subtle, professional
- **Perfect Fourth (1.333)**: clear hierarchy, good for marketing

### Fluid Typography with clamp()
```css
/* Body */
font-size: clamp(1rem, 0.9rem + 0.5vw, 1.125rem);

/* H1 Display */
font-size: clamp(2rem, 1rem + 4vw, 5.5rem);

/* H2 Section */
font-size: clamp(1.5rem, 1rem + 2.5vw, 3.5rem);
```

### Heading Hierarchy
- Maximum 4 heading levels visible in a single view
- Display/hero headings: tight leading (0.9-1.1), tight tracking (tracking-tight)
- Section headings: leading 1.1-1.2
- Body text: leading 1.5 (latin), 1.7-1.8 (CJK)

### Letter Spacing
- Large headings (> 3rem): tighten (tracking-tight or -0.02em)
- Small caps / labels: slightly wide (+0.05em)
- Body: normal (0)

## Anti-Patterns
- DON'T: Default to Google Fonts when system fonts suffice
- DON'T: Use more than 2-3 font families per project
- DON'T: Set body text below 16px on mobile
- DON'T: Use light font-weight (300) for body on dark backgrounds without testing contrast
- DON'T: Apply identical letter-spacing to headings and body
- DON'T: Use latin-only fonts when the service targets CJK users
- DO: Test CJK characters at every size; they need more line-height than latin
- DO: Use `font-display: swap` for custom fonts to prevent FOIT
- DO: Subset fonts to only needed character ranges for performance
