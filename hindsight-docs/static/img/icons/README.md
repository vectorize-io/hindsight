# Sidebar Icons

This directory contains SVG icons for the sidebar navigation items.

## Required Icons

### Client Icons
- `python.svg` - Python logo (download from https://www.python.org/community/logos/)
- `nodejs.svg` - Node.js logo (download from https://nodejs.org/en/about/branding)
- `terminal.svg` - Terminal/CLI icon
- `package.svg` - Package/box icon for Embedded SDK

### Integration Icons
- `mcp.svg` - MCP Server icon
- `litellm.svg` - LiteLLM logo (download from https://github.com/BerriAI/litellm)
- `openclaw.svg` - OpenClaw logo
- `vercel.svg` - Vercel triangle logo (download from https://vercel.com/design/brands)
- `skills.svg` - Skills/star icon

## Specifications

- **Format**: SVG (preferred) or PNG
- **Size**: 16x16px or larger (will be scaled to 14x14px)
- **Style**: Monochrome or simple colors work best
- **Color**: Icons should work on both light and dark backgrounds

## Alternative: Using Remote URLs

Instead of local files, you can use remote URLs directly in the CSS:

```css
a.menu__link[href*="/sdks/python"]::before {
  background-image: url('https://cdn.jsdelivr.net/npm/simple-icons@v10/icons/python.svg');
}
```

Popular icon CDNs:
- Simple Icons: https://simpleicons.org/
- cdnjs: https://cdnjs.com/
- jsDelivr: https://www.jsdelivr.com/

## Creating Icons

If you need to create custom icons, use tools like:
- Figma (https://figma.com)
- Inkscape (https://inkscape.org)
- SVGOMG for optimization (https://jakearchibald.github.io/svgomg/)
