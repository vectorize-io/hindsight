# Design Error Playbook

## Contrast Failure

**Symptom**: Text hard to read, WCAG contrast check fails.

**Fix**:
1. Check contrast ratio with browser DevTools or axe
2. For dark themes: use warm off-white (#f5f0eb) instead of pure white (#fff)
3. Increase text opacity (text-white/60 → text-white/80)
4. Darken or lighten background to increase contrast
5. For colored backgrounds: use dark text or add a dark overlay

**Prevention**: Always define text colors relative to backgrounds in DESIGN.md.

---

## AI Slop Detected

**Symptom**: Design looks generic, fails the "AI made this" test.

**Fix**:
1. Check anti-patterns.md to identify which patterns triggered
2. Replace purple gradients with brand-derived colors
3. Add texture (noise/grain) instead of plain gradients
4. Vary layout patterns across sections (not all card grids)
5. Inject project-specific design context from .design-context.md
6. Replace Inter/Geist with system fonts or project-specific fonts
7. Remove gratuitous glassmorphism; reserve for 1-2 accent elements

**Prevention**: Always run Phase 1 (Setup) to establish project context before designing.

---

## Responsive Breakage

**Symptom**: Horizontal scroll, overlapping elements, tiny touch targets on mobile.

**Fix**:
1. Check at 375px viewport width (iPhone SE)
2. Stack columns: grid-cols-1 on mobile, grid-cols-2+ on md/lg
3. Increase touch targets to min 44x44pt (min-h-[44px] min-w-[44px])
4. Use clamp() for fluid typography instead of fixed sizes
5. Hide decorative elements on mobile if they cause overflow
6. Switch from side-by-side to stacked layout on mobile
7. Check for fixed-width elements: replace px values with percentages or max-w

**Prevention**: Always start with mobile layout, enhance upward.

---

## Animation Jank

**Symptom**: Janky scrolling, low FPS, dropped frames, laggy interactions.

**Fix**:
1. Only animate `transform` and `opacity`; never width, height, top, left
2. Reduce concurrent animations to max 2-3 visible simultaneously
3. Pause off-screen Canvas/WebGL with Intersection Observer
4. Add `will-change: transform` sparingly and remove after animation completes
5. Reduce particle count on mobile (50-75% fewer)
6. Provide static fallback for prefers-reduced-motion
7. Reduce canvas resolution: `renderer.setPixelRatio(Math.min(devicePixelRatio, 2))`

**Prevention**: Test on actual mobile devices, not just browser responsive mode.

---

## Stitch MCP Connection Failure

**Symptom**: MCP tools not responding, authentication errors.

**Fix**:
1. Run `npx @_davideast/stitch-mcp doctor --verbose`
2. Check API key validity or OAuth token expiry
3. Verify API is enabled: `gcloud beta services mcp enable stitch.googleapis.com`
4. For OAuth: re-authenticate with `npx @_davideast/stitch-mcp init`
5. For API key: verify STITCH_API_KEY environment variable is set

**Fallback**: Proceed without Stitch; all workflow phases work standalone.

---

## Design System Inconsistency

**Symptom**: Colors, spacing, or typography don't match DESIGN.md.

**Fix**:
1. Re-read DESIGN.md and compare against generated code
2. Search for hardcoded hex values: `grep -r "#[0-9a-fA-F]\{6\}" src/`
3. Replace hardcoded values with CSS custom properties or Tailwind theme tokens
4. Check for arbitrary Tailwind values (e.g., `text-[13px]`); replace with scale values
5. Verify spacing uses 8px grid multiples only

**Prevention**: Generate CSS custom properties from DESIGN.md before writing component code.

---

## Theme Switching Issues

**Symptom**: Elements look wrong in dark/light mode, missing tokens.

**Fix**:
1. Verify all color tokens have both :root and .dark variants
2. Check for hardcoded colors that bypass the theme system
3. Ensure borders use alpha transparency (border-white/10) not gray hex values
4. Test both themes end-to-end; toggle and check every section
5. Add system preference detection: prefers-color-scheme media query

**Prevention**: Define complete light and dark token sets in DESIGN.md before implementing.
