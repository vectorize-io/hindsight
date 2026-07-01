# Prompt Enhancement Protocol

## Purpose

Transform vague user requests into detailed, section-by-section design
specifications that produce high-quality, specific output.

## When to Activate

Activate when the user request is vague:
- Less than 3 sentences
- No section details specified
- Generic terms like "make a landing page" or "design a website"

Do NOT activate when the user provides detailed specs (section layouts, specific components, color choices).

## Process

### 1. Input Analysis

Identify what the user explicitly specified vs what needs inference:

| Specified | Needs Inference |
|-----------|-----------------|
| "dark landing page" | which sections? |
| "for AI product" | what components? |
| "premium feel" | which animation strategy? |
| (nothing about mobile) | responsive approach? |

### 2. Section-by-Section Enhancement

For each section, specify:
1. **Layout**: structure, columns, alignment, max-width
2. **Background**: solid / gradient / video / animated shader
3. **Typography**: heading font + size + weight, body style
4. **Components**: specific elements (badges, cards, buttons, forms)
5. **Motion**: entrance animation, scroll behavior, hover effects
6. **Responsive**: how it changes on mobile vs desktop

### 3. Enhancement Template

```
SECTION: [Name]
- Layout: [structure description]
- Background: [background treatment]
- Content:
  - [element 1]: [description with styling notes]
  - [element 2]: [description with styling notes]
- Motion: [animation approach]
- Responsive: [mobile behavior]
```

## Example Enhancement

### User Input
> "Make a landing page for my SaaS product"

### Enhanced Output

```
SECTION: Hero
- Layout: centered, full viewport height, max-w-5xl
- Background: subtle animated gradient (brand colors) or solid dark
- Content:
  - Badge: glass pill with announcement text ("New: Feature X launched")
  - Heading: display font, clamp(2rem, 5.5rem), tight leading (0.9)
  - Subtext: body font, muted color, max-w-md, font-light
  - CTA: primary button (solid) + secondary button (glass outline)
- Motion: heading BlurText word reveal, subtext fade-in at 0.8s delay
- Responsive: heading scales down 2 steps, buttons stack vertically

SECTION: Social Proof
- Layout: horizontal bar below hero, centered
- Background: transparent (same as hero)
- Content:
  - Badge: glass pill "Trusted by teams at..."
  - Logos/names: infinite marquee scroll, 5-8 partner names
- Motion: continuous marquee, pauses on hover
- Responsive: reduce gap, smaller text

SECTION: Features (Chess Layout)
- Layout: alternating text/image rows, py-24 px-6 md:px-16 lg:px-24
- Background: solid dark
- Content per row:
  - H3 heading + paragraph + CTA button
  - Image/video/GIF in glass rounded container
  - Row 1: text left, image right
  - Row 2: image left, text right (lg:flex-row-reverse)
- Motion: fade-in on scroll intersection
- Responsive: stack vertically on mobile (image above text)

SECTION: Features (Grid)
- Layout: 3-4 column card grid, gap-6
- Content per card:
  - Icon in glass circle (w-10 h-10)
  - Title: heading font, text-lg
  - Description: body font, muted, text-sm
- Motion: staggered entrance on scroll
- Responsive: 1 col mobile → 2 col tablet → 3-4 col desktop

SECTION: Stats
- Layout: glass card, 4-column grid, centered
- Background: desaturated video or solid with texture
- Content:
  - 3-4 large display numbers + small labels
  - Values: display font, text-5xl
  - Labels: body font, muted, text-sm
- Motion: number count-up on scroll intersection
- Responsive: 2x2 grid on mobile

SECTION: Testimonials
- Layout: 3-column grid
- Content per card:
  - Glass surface card, p-8
  - Quote: italic, muted white (text-white/80)
  - Avatar + Name + Role
- Motion: none or subtle fade-in
- Responsive: 1 col stack on mobile

SECTION: CTA
- Layout: centered, py-32
- Background: gradient overlay or video
- Content:
  - Large heading: display font, text-6xl
  - Subtext: body font, muted
  - Two buttons: primary (solid) + secondary (glass)
- Responsive: heading scales down, buttons stack

SECTION: Footer
- Layout: multi-column links grid + copyright bar
- Content:
  - 4 columns: Product, Company, Resources, Legal
  - Bottom bar: copyright + links (text-white/40 text-xs)
  - Border-top: border-white/10
- Responsive: stack into 2 columns on mobile, 1 on small mobile
```

## Post-Enhancement

After presenting the enhanced prompt:
1. Ask the user for confirmation or adjustments
2. Apply feedback
3. Proceed to Phase 4 (Propose) with the refined specification
