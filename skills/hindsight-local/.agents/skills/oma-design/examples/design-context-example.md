# .design-context.md: Example

This is an example of what `.design-context.md` looks like after Phase 1 (Setup).
The file lives in the project root and captures project-specific design decisions.

---

## Project
- **Name**: Apex Revenue Platform
- **Type**: B2B SaaS landing page + dashboard
- **Languages**: English (primary), Korean (secondary)

## Target Audience
- **Role**: Sales leaders, revenue ops managers, growth teams
- **Tech level**: Moderate; comfortable with dashboards, not developers
- **Age range**: 28-45
- **Context**: Evaluating tools during work hours, often on laptop

## Brand Personality
- Professional but not corporate
- Confident and forward-looking
- Data-driven, precise language
- Avoids jargon and buzzwords

## Aesthetic Direction
- **Theme**: Dark premium (Apple-inspired)
- **Mood**: Confident, spacious, sophisticated
- **Surface treatment**: Glass morphism for UI chrome, solid for content areas
- **Visual accents**: Subtle green (#22c55e) for CTAs and data highlights
- **Texture**: Prefer noise/grain over gradients for visual interest

## Typography
- **Body**: Pretendard Variable (CJK support for Korean) + system-ui fallback
- **Headings**: Instrument Serif italic for hero/display headings only
- **Mono**: JetBrains Mono for data tables and code snippets
- **Justification**: CJK support required for Korean localization

## Color Direction
- **Background**: Deep near-black (#0a0a0a)
- **Text**: Warm off-white (#f5f0eb), not pure white
- **Primary accent**: Signal Green (#22c55e) for CTAs, success states
- **Avoid**: Purple gradients, rainbow effects, mesh gradients
- **Borders**: White at 10% opacity (rgba(255,255,255,0.1))

## Accessibility
- **Level**: WCAG AA minimum
- **Motion**: prefers-reduced-motion support required
- **Contrast**: 4.5:1 for normal text, 3:1 for large text
- **Touch targets**: 44x44pt minimum on mobile

## Reference Sites
- [linear.app](https://linear.app): clean dark UI, minimal, professional
- [vercel.com](https://vercel.com): developer-premium aesthetic, great typography
- [stripe.com](https://stripe.com): strong hierarchy, purposeful animation

> **Note**: every domain in this section is automatically matched
> against the `getdesign` vendor catalog during Phase 1. All three
> entries above resolve to community templates (`linear.app`, `vercel`,
> `stripe`) and will be fetched as Phase 2 seeds. To opt a URL out of
> vendor matching, use a domain that is not in the catalog (e.g., an
> internal design reference or a custom portfolio URL). See
> `resources/getdesign-fetcher.md` for the matching algorithm and the
> Seed Application Rules. Notably, Typography is never adopted from
> the vendor seed, so the Pretendard Variable choice in this file will
> still win on the Korean-localized project above.

## Component Preferences
- **Base**: shadcn/ui for all foundational components
- **Effects**: Aceternity UI for hero parallax, React Bits for text animations
- **Animation**: motion/react for transitions, GSAP for scroll-triggered reveals
