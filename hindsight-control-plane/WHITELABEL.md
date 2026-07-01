# Whitelabel Configuration Guide

This document explains how to configure and deploy the Hindsight Control Plane as a whitelabeled application for your brand.

## Overview

The control plane supports full whitelabel customization through environment variables. You can configure:

- **Brand Identity**: Product name, tagline, and messaging
- **Visual Assets**: Logo, favicon, and Open Graph images
- **Color Palette**: Primary, secondary, and accent colors
- **External Links**: GitHub, documentation, support, and website URLs
- **Metadata**: Page titles, descriptions, and social sharing content

## Quick Start

### 1. Configure Environment Variables

Copy the appropriate example file to `.env.local`:

```bash
# For default Hindsight branding
cp .env.example .env.local

# For CollabMinds branding
cp .env.collabminds .env.local
```

Edit `.env.local` with your configuration values.

### 2. Add Brand Assets

Place your brand assets in the `/public` directory:

```
/public
  ├── your-logo.png           # Main logo (recommended: 200x50px)
  ├── your-favicon.png         # Favicon (recommended: 32x32px or 64x64px)
  └── your-og-image.png        # Open Graph image (recommended: 1200x630px)
```

### 3. Update Environment Variables

Point to your assets in `.env.local`:

```env
NEXT_PUBLIC_BRAND_LOGO_URL=/your-logo.png
NEXT_PUBLIC_BRAND_FAVICON_URL=/your-favicon.png
NEXT_PUBLIC_BRAND_OG_IMAGE=/your-og-image.png
```

### 4. Build and Deploy

```bash
npm run build
npm start
```

## Configuration Reference

### Brand Identity

```env
# Product name displayed throughout the UI
NEXT_PUBLIC_BRAND_NAME=YourBrand

# Optional tagline displayed in headers
NEXT_PUBLIC_BRAND_TAGLINE=Your Custom Tagline
```

### Visual Assets

```env
# Logo displayed in the navigation header
# Can be a path relative to /public or an absolute URL
NEXT_PUBLIC_BRAND_LOGO_URL=/logo.png

# Favicon displayed in the browser tab
NEXT_PUBLIC_BRAND_FAVICON_URL=/favicon.png
```

### Color Palette

Define your brand colors using hex codes (with or without `#`):

```env
# Primary color - used for buttons, links, key UI elements
NEXT_PUBLIC_BRAND_PRIMARY_COLOR=#0074d9

# Secondary color - used in gradients and complementary elements
NEXT_PUBLIC_BRAND_SECONDARY_COLOR=#009296

# Accent color - used for calls-to-action and highlights
NEXT_PUBLIC_BRAND_ACCENT_COLOR=#f59e0b
```

**Color Guidelines:**
- Use hex codes with 6 characters (#RRGGBB)
- Ensure sufficient contrast for accessibility (WCAG AA minimum)
- Primary color should be your main brand color
- Secondary color complements primary (often used in gradients)
- Accent color should stand out for important actions

### External Links

Configure links that appear in the UI:

```env
# GitHub repository link (shown in navigation)
NEXT_PUBLIC_BRAND_GITHUB_URL=https://github.com/your-org/your-repo

# Documentation site
NEXT_PUBLIC_BRAND_DOCS_URL=https://docs.yourbrand.com

# Support/help center
NEXT_PUBLIC_BRAND_SUPPORT_URL=https://support.yourbrand.com

# Main website
NEXT_PUBLIC_BRAND_WEBSITE_URL=https://yourbrand.com
```

**Note:** If a link is not configured (empty or omitted), the corresponding UI element will be hidden.

### SEO & Social Metadata

```env
# Page title (appears in browser tab and search results)
NEXT_PUBLIC_BRAND_META_TITLE=YourBrand - Control Plane

# Meta description (appears in search results)
NEXT_PUBLIC_BRAND_META_DESCRIPTION=Manage your YourBrand infrastructure

# Open Graph image for social sharing
NEXT_PUBLIC_BRAND_OG_IMAGE=/og-image.png
```

## How It Works

### 1. Configuration Loading

The `src/lib/whitelabel-config.ts` file exports a `whitelabelConfig` object that reads from environment variables at build time (for server-side rendering) and runtime (for client-side code).

### 2. Color System

Brand colors are centralized in two places:

- **Static Colors**: `src/lib/brand-colors.ts` exports color constants used in components
- **CSS Variables**: `src/components/brand-style-injector.tsx` injects colors as CSS custom properties

This two-tier approach ensures:
- React components can use typed color constants
- CSS and Tailwind classes can reference dynamic variables
- Colors are consistent across the entire application

### 3. Component Integration

Components import and use whitelabel configuration:

```tsx
import { whitelabelConfig } from '@/lib/whitelabel-config';
import { BRAND_COLORS } from '@/lib/brand-colors';

// Use in JSX
<h1>{whitelabelConfig.brandName}</h1>

// Use in styles
<div style={{ color: BRAND_COLORS.primary }} />
```

## Deployment Examples

### CollabMinds Configuration

The included `.env.collabminds` file demonstrates a complete whitelabel setup:

- **Brand**: CollabMinds
- **Theme**: Cyan (#39d4d4) / Purple (#9d7bf0) / Green (#3fd07a)
- **Style**: Technical operator aesthetic

### Default Hindsight Configuration

The `.env.example` file contains the default Hindsight branding:

- **Brand**: Hindsight
- **Theme**: Blue (#0074d9) / Teal (#009296) / Amber (#f59e0b)
- **Style**: Professional SaaS aesthetic

## Testing Your Configuration

1. **Local Development**: Run `npm run dev` and verify:
   - Brand name appears in navigation
   - Logo displays correctly
   - Colors match your brand palette
   - External links work (if configured)

2. **Build Verification**: Run `npm run build` and check:
   - No build errors related to missing environment variables
   - Static assets are included in the build output

3. **Production Deployment**:
   - Set environment variables in your hosting platform
   - Verify metadata in browser inspector and social media preview tools
   - Test all configured links

## Troubleshooting

### Logo not displaying
- Verify the file exists in `/public`
- Check the path in `NEXT_PUBLIC_BRAND_LOGO_URL`
- Ensure the image format is supported (PNG, JPEG, SVG)

### Colors not updating
- Clear browser cache and rebuild
- Verify hex codes are valid (6 characters, with or without #)
- Check browser console for CSS errors

### Links not appearing
- Ensure the environment variable is set
- Links are conditionally rendered - empty values hide the UI element
- Rebuild after changing environment variables

### Build errors
- Check that all `NEXT_PUBLIC_*` variables are set at build time
- Verify TypeScript errors in `src/lib/whitelabel-config.ts`
- Ensure all imported modules exist

## Advanced Customization

### Custom Fonts

Fonts are configured in `src/app/globals.css`. To use custom fonts:

1. Add font imports at the top of `globals.css`
2. Update CSS variables:
   ```css
   :root {
     --font-sans: 'YourFont', sans-serif;
     --font-heading: 'YourHeadingFont', sans-serif;
     --font-mono: 'YourMonoFont', monospace;
   }
   ```

### Additional Theme Variables

To add more whitelabel configuration:

1. Add environment variables to `.env.example`
2. Update `WhitelabelConfig` interface in `src/lib/whitelabel-config.ts`
3. Add getters in `getWhitelabelConfig()`
4. Use in components via `whitelabelConfig.yourNewProp`

### Internationalization (i18n)

Product names and brand-specific text in i18n files are currently hardcoded. To make them dynamic:

1. Use `whitelabelConfig.brandName` in components instead of translation keys
2. Or create dynamic translation interpolation with brand name as a variable

## Support

For questions or issues with whitelabel configuration:

1. Check this documentation first
2. Review example configurations (`.env.example`, `.env.collabminds`)
3. Inspect `src/lib/whitelabel-config.ts` for available options
4. Open an issue in the repository

## License

The whitelabel configuration system is part of the Hindsight Control Plane project. See LICENSE for details.
