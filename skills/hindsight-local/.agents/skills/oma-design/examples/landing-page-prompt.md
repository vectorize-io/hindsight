# Landing Page Design Prompt: Example

This is an example of the level of detail Phase 3 (Enhance) should produce.
Based on motionsites.ai-level specifications.

---

## Project Context
- Dark premium SaaS landing page
- React + Vite + TypeScript + Tailwind CSS + shadcn/ui
- Pure black background throughout
- "Liquid glass" morphism effect for UI chrome

## Design System

### Fonts
- Heading: Instrument Serif (italic) for display headings only
- Body: system-ui stack (or Pretendard for CJK)

### CSS Variables
```css
:root {
  --background: 0 0% 3.9%;
  --foreground: 30 10% 95%;
  --primary: 142 71% 45%;
  --primary-foreground: 0 0% 3.9%;
  --border: 0 0% 100% / 0.1;
  --radius: 0.75rem;
}
```

### Liquid Glass Utility
```css
.liquid-glass {
  background: rgba(255, 255, 255, 0.01);
  backdrop-filter: blur(4px);
  box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1);
  position: relative;
  overflow: hidden;
}
/* ::before pseudo-element for gradient border mask */
```

---

## Section Specifications

### HERO (full viewport)
- **Layout**: centered, min-h-screen, flex column
- **Background**: video (mp4, autoplay loop muted) with gradient overlay to black at bottom
- **Badge**: liquid-glass rounded-full pill with "New" tag + announcement text
- **Heading**: BlurText component (motion/react), word-by-word blur-to-clear animation
  - text-6xl md:text-7xl lg:text-[5.5rem] font-heading italic
  - leading-[0.8] tracking-[-4px]
- **Subtext**: motion.p, fade-in with blur at 0.8s delay
  - text-white/60 font-body font-light max-w-md
- **CTAs**: motion.div at 1.1s delay
  - Primary: liquid-glass-strong rounded-full + ArrowUpRight icon
  - Secondary: text-only "Watch the Film" + Play icon
- **Responsive**: heading clamp(2rem, 5.5rem), CTAs stack on mobile

### PARTNERS BAR
- **Layout**: centered column, below hero
- **Badge**: liquid-glass rounded-full labeled "Trusted by the teams behind"
- **Names**: horizontal row, text-2xl md:text-3xl font-heading italic text-white, gap-12
- **Companies**: Stripe, Vercel, Linear, Notion, Figma
- **Responsive**: reduce gap, text-xl on mobile, wrap if needed

### HOW IT WORKS (video background)
- **Layout**: full-width, min-h-[700px], py-32 px-6 md:px-16 lg:px-24
- **Background**: HLS video (hls.js), absolute cover, z-0
  - Top + bottom fade gradients (200px each, black ↔ transparent)
- **Content** (z-10, centered):
  - Badge: liquid-glass rounded-full labeled "How It Works"
  - Heading: "You dream it. We ship it."
  - Subtext: description paragraph
  - Button: liquid-glass-strong rounded-full + ArrowUpRight

### FEATURES CHESS (alternating rows)
- **Layout**: py-24 px-6 md:px-16 lg:px-24
- **Header**: badge "Capabilities", heading "Pro features. Zero complexity."
- **Row 1** (text left, image right):
  - H3 + paragraph + CTA button
  - Image in liquid-glass rounded-2xl container
- **Row 2** (image left, text right; lg:flex-row-reverse):
  - Same structure, reversed layout
- **Responsive**: stack vertically, image above text on mobile

### FEATURES GRID (4 columns)
- **Layout**: grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6
- **Each card**: liquid-glass rounded-2xl p-6
  - Icon: liquid-glass-strong rounded-full w-10 h-10
  - Title: text-lg font-heading italic text-white
  - Description: text-white/60 font-body font-light text-sm
- **Responsive**: 1 col → 2 col → 4 col

### STATS (video background)
- **Background**: HLS video, desaturated (filter: saturate(0))
  - Top + bottom black fades (200px)
- **Content**: liquid-glass rounded-3xl p-12 md:p-16
  - Grid: grid-cols-2 lg:grid-cols-4 gap-8 text-center
  - Values: text-4xl md:text-5xl lg:text-6xl font-heading italic
  - Labels: text-white/60 font-body font-light text-sm
- **Responsive**: 2x2 on mobile, 4-col on desktop

### TESTIMONIALS
- **Layout**: 3-column grid
- **Each card**: liquid-glass rounded-2xl p-8
  - Quote: text-white/80 font-body font-light text-sm italic
  - Name: text-white font-body font-medium text-sm
  - Role: text-white/50 font-body font-light text-xs
- **Responsive**: 1 col stack on mobile

### CTA + FOOTER
- **Background**: HLS video + top/bottom black fades
- **CTA Content** (z-10, centered):
  - Heading: text-5xl md:text-6xl lg:text-7xl
  - Subtext: description
  - Buttons: liquid-glass-strong + bg-white text-black
- **Footer**: mt-32 pt-8 border-t border-white/10
  - Links: Privacy, Terms, Contact (text-white/40 text-xs)
  - Copyright: (c) 2026 Studio

---

## Dependencies
- hls.js (HLS video streaming)
- motion (animation; import from "motion/react")
- lucide-react (icons)
- tailwindcss-animate

## Key Patterns
- All badges: liquid-glass rounded-full px-3.5 py-1 text-xs font-medium
- All headings: text-4xl md:text-5xl lg:text-6xl font-heading italic tracking-tight leading-[0.9]
- All video fades: 200px height gradient overlays
- Page wrapper: bg-black overflow-visible
