# Component Patterns Reference

## Recommended Libraries

### shadcn/ui (Foundation)
- Install: `npx shadcn@latest init`
- Use for: buttons, inputs, dialogs, dropdowns, tabs, cards, forms
- Customizable via CSS variables in `globals.css`
- Code ownership model: components live in your project

### Aceternity UI (Premium Effects)
- Install: `npx shadcn@latest add @aceternity/<component>`
- Dark/premium aesthetic by default
- Key components by category:

| Category | Components |
|----------|-----------|
| Backgrounds | Aurora, Sparkles, Background Beams, Spotlight, Wavy Background, Meteors, Vortex |
| Cards | 3D Card, Wobble Card, Evervault Card, Card Stack, Focus Cards, Glare Card |
| Scroll | Hero Parallax, Macbook Scroll, Sticky Scroll Reveal, Container Scroll |
| Text | Text Generate Effect, Typewriter, Flip Words, Hero Highlight, Colourful Text |
| Navigation | Floating Navbar, Floating Dock (macOS-style), Tabs, Sidebar |
| Input | Placeholders And Vanish Input, File Upload |
| Overlay | Animated Modal, Animated Tooltip, Link Preview |
| Layout | Bento Grid, Layout Grid |
| 3D | 3D Globe, 3D Pin, 3D Marquee |

### React Bits (Statement Pieces)
- Install: `npx shadcn@latest add https://reactbits.dev/r/<Name>-TS-TW`
- Shader-heavy, physics-based, high visual impact
- Key components by category:

| Category | Components |
|----------|-----------|
| Text (23) | BlurText, DecryptedText, GlitchText, SplitText, GradientText, ShinyText, ScrollFloat, TextPressure, VariableProximity |
| Backgrounds (40) | Aurora, LiquidChrome, LiquidEther, Silk, Hyperspeed, Iridescence, Galaxy, Ballpit, Waves, Plasma, Lightning |
| Cursor (8+) | BlobCursor, SplashCursor, MagnetLines, PixelTrail, GhostCursor, ImageTrail |
| Glass | FluidGlass, GlassSurface, GlassIcons |
| Cards | TiltedCard, SpotlightCard, DecayCard, PixelCard, ReflectiveCard, BounceCards |
| 3D | Lanyard (Rapier physics), CircularGallery, DomeGallery, ModelViewer |
| Navigation | Dock, FlowingMenu, GooeyNav, PillNav, BubbleMenu |

## Section Patterns (Landing Pages)

### Hero Section
- **Background**: video (mp4/HLS), animated shader (Aurora/Particles), or solid + gradient
- **Content stack**: badge pill → display heading → subtext → CTA buttons
- **Badge**: glass pill with announcement text
- **Heading**: display font, large (clamp 2rem-5.5rem), tight leading (0.9)
- **Subtext**: muted color, max-w-md, font-light
- **CTAs**: primary (solid) + secondary (glass outline or text-only)
- **Animation**: BlurText word reveal, motion.p fade-in with delay

### Features Section
- **Chess layout**: alternating text/image rows (text left → image right, then reverse)
- **Grid layout**: 3-4 column cards with icon + title + description
- **Bento layout**: asymmetric grid with mixed content sizes
- **Each card**: glass surface, icon in circle, heading, description, optional CTA

### Social Proof
- **Stats bar**: 3-4 metrics in glass card, large display numbers + small labels
- **Testimonials**: 3-column grid, glass cards, italic quote + avatar + name/role
- **Logo bar**: infinite marquee scroll of partner logos/names
- **Partner names**: display font, italic, muted or white text

### CTA / Footer
- **Background**: video with gradient overlay, or gradient
- **CTA**: centered large heading + subtext + 2 buttons (primary + secondary)
- **Footer**: multi-column links grid (4 columns) + copyright bar
- **Footer border**: border-t border-white/10

## Glass Effect (Liquid Glass CSS)

### Subtle Variant
```css
.liquid-glass {
  background: rgba(255, 255, 255, 0.01);
  background-blend-mode: luminosity;
  backdrop-filter: blur(4px);
  border: none;
  box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.1);
  position: relative;
  overflow: hidden;
}
/* Gradient border via ::before with mask-composite */
.liquid-glass::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1.4px;
  background: linear-gradient(180deg,
    rgba(255,255,255,0.45) 0%, rgba(255,255,255,0.15) 20%,
    rgba(255,255,255,0) 40%, rgba(255,255,255,0) 60%,
    rgba(255,255,255,0.15) 80%, rgba(255,255,255,0.45) 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
}
```

### Strong Variant
Same structure but with:
- `backdrop-filter: blur(50px)`
- Stronger box-shadow: `4px 4px 4px rgba(0,0,0,0.05), inset 0 1px 1px rgba(255,255,255,0.15)`
- Higher gradient opacity (0.5 / 0.2)

### Usage
- **Badges/pills**: liquid-glass + rounded-full
- **Navigation**: liquid-glass + rounded-full (pill nav)
- **Cards**: liquid-glass + rounded-2xl or rounded-3xl
- **Buttons**: liquid-glass-strong + rounded-full

## HLS Video Background Pattern

```tsx
import Hls from "hls.js"

useEffect(() => {
  const video = videoRef.current
  if (!video) return
  if (video.canPlayType("application/vnd.apple.mpegurl")) {
    video.src = hlsSrc // Safari native
  } else if (Hls.isSupported()) {
    const hls = new Hls()
    hls.loadSource(hlsSrc)
    hls.attachMedia(video)
    return () => hls.destroy()
  }
}, [hlsSrc])
```

With top/bottom fade gradients (200px height):
```
linear-gradient(to bottom, black, transparent) for top fade
linear-gradient(to top, black, transparent) for bottom fade
```

## Anti-Patterns
- DON'T: Use glassmorphism on every element; reserve for badges, nav pills, accent cards
- DON'T: Use the same card layout for every section (all 3-col grids)
- DON'T: Force identical card heights with arbitrary min-height
- DON'T: Rely on hover-only interactions without touch/keyboard alternatives
- DO: Mix section patterns within a page (chess + grid + stats + testimonials)
- DO: Choose component libraries intentionally based on the effect needed
- DO: Always include install commands when recommending components
