# Execution Protocol: 7-Phase Design Workflow

## Phase 1: SETUP (Context Gathering)

Check for `.design-context.md` in the project root.

If it does not exist:
1. Scan codebase for existing design signals:
   - `package.json`: font packages, UI libraries, CSS framework
   - Tailwind config: existing theme, colors, fonts
   - Existing CSS/SCSS: design tokens, custom properties
   - `DESIGN.md`: if already present, use as starting point
2. Ask the user (one question at a time, prefer multiple-choice):
   - What languages does the service support? (determines font strategy)
   - Who is the target audience? (B2B/B2C, age range, tech level)
   - What is the brand personality? (professional / casual / premium / playful)
   - What aesthetic direction? (dark premium / clean minimal / colorful / brutalist / other)
   - Any reference sites or designs to draw from?
   - Accessibility requirements? (WCAG AA / AAA / none specified)
3. Save answers to `.design-context.md`

**MUST complete before proceeding. Never skip context gathering.**

### Vendor Inspiration Detection
After `.design-context.md` exists (either newly created or already present):
1. Parse the `## Reference Sites` section (if any) and extract bare
   domains (`linear.app`, `stripe.com`, …).
2. Load `resources/getdesign-fetcher.md` and run its Matching Algorithm
   against the live `getdesign@latest` manifest.
3. Hold the resolved brand list in memory for Phase 2. No disk write.
4. If zero matches, continue silently; vendor inspiration is optional.

### Stitch MCP Check (Optional)
If the user wants to use Stitch for design extraction or generation:
1. Check if Stitch MCP is available: look for stitch-related tools
2. If not available, offer setup:
   - Load `resources/stitch-integration.md` for client-specific setup instructions
   - Ask which client they use (Claude Code / Cursor / VS Code / Gemini CLI / Codex)
   - Provide the matching setup command
3. If user declines: proceed without Stitch (all phases work standalone)

---

## Phase 2: EXTRACT (Optional)

Phase 2 runs through branches in priority order. Use the first branch
that has data; subsequent branches are fallbacks.

### Branch A: Stitch MCP (highest priority when available)
1. `list_projects` → find the relevant project
2. `get_project` → extract `designTheme` (colors, fonts, roundness)
3. `list_screens` → enumerate all screens
4. `get_screen_code` → download HTML/CSS for analysis
5. Extract design tokens → synthesize into DESIGN.md draft

### Branch B: getdesign Vendor Seed
Triggered when Phase 1 resolved at least one vendor brand from the
`## Reference Sites` section.

1. For each resolved brand: follow `resources/getdesign-fetcher.md`
   steps. Fetch via `bunx getdesign@latest add <brand> --out <tmp>`
   with `GETDESIGN_DISABLE_TELEMETRY=1`, verify the SHA256 against the
   manifest `templateHash`, then load the file into context with the
   prompt-injection framing described in the fetcher doc.
2. Run an immediate **pre-audit** of each seed against
   `resources/anti-patterns.md`. Record any anti-pattern violations
   (glassmorphism density, purple gradients, nested cards, etc.) to
   surface in Phase 4 PROPOSE.
3. Delete all temp files once context is loaded.
4. Apply the Seed Application Rules from the fetcher doc: adopt
   color/spacing/components/depth/responsive; reject typography;
   rewrite theme/do-don't/agent-prompt-guide in Phase 5.

### Branch C: Reference URL (no Stitch, no vendor seed)
1. If the user supplied a URL that did not match any getdesign brand:
   fetch and analyze HTML/CSS directly.
2. Extract: font families, color values, spacing patterns, component
   structures.
3. Translate raw values into the 9-section DESIGN.md format
   (`resources/design-md-spec.md`).

### Branch D: No Reference
Skip to Phase 3.

---

## Phase 3: ENHANCE (Prompt Augmentation)

**Skip Phase 3 entirely if Phase 2 Branch B (getdesign vendor seed) was
triggered.** A vendor seed already carries section-by-section detail,
so further prompt enhancement would duplicate work. Jump straight to
Phase 4 PROPOSE.

Otherwise, if the user request is vague (< 3 sentences, no section
details):

1. Load `resources/prompt-enhancement.md`
2. Transform the request into a section-by-section specification:
   - For each section specify: layout, background, typography, components, motion, responsive behavior
3. Present the enhanced prompt to the user for confirmation
4. Adjust based on feedback

If the request is already detailed: skip to Phase 4.

---

## Phase 4: PROPOSE (Multi-Concept)

### Default: No vendor seed
Present 2-3 distinct design directions. Each direction must include:

1. **Color palette**: 5-7 colors with semantic names and functional roles
2. **Typography pairing**: system fonts or custom fonts with justification
3. **Layout approach**: chess / grid / bento / full-bleed / mixed
4. **Motion strategy**: scroll-driven / hover-based / entrance-only / minimal
5. **Component recommendations**: which libraries (shadcn base + Aceternity / React Bits accents)
6. **Visual mood**: one-sentence description of the feel

Present as a comparison table with pros/cons for each direction.

### Vendor seed override (Branch B was triggered in Phase 2)
Replace the "2-3 distinct directions" rule with a **3-variation
formula** anchored on the seed:

- **A (Faithful)**: stay as close to the vendor template as possible;
  apply only the mandatory Seed Application Rules (typography override
  for CJK, anti-pattern removal flagged in Phase 2 pre-audit).
- **B (Hybrid)**: blend the seed's color/spacing/components with the
  project's brand tone from `.design-context.md`. This is usually the
  best default for production work.
- **C (Loose inspiration)**: keep only the seed's structural patterns
  (rhythm, density, component philosophy) and rebuild the visual layer
  from the project brand.

For each variation, include the pre-audit violations from Phase 2 with
a note on whether that variation keeps or removes them. Users must
consciously choose to keep any anti-patterns.

### Multi-vendor merge
If two or more vendors matched in Phase 1, do not auto-blend. Present
the dimension-level selection dialog from
`resources/getdesign-fetcher.md` ("Multi-Vendor Merge Policy") before
presenting variations A/B/C.

**MUST get user confirmation on the chosen direction before proceeding.**

---

## Phase 5: GENERATE

Based on the chosen direction:

1. Write `DESIGN.md` following `resources/design-md-spec.md`
   (9 sections, including the mandatory Section 9 "Agent Prompt
   Guide" with Quick Color Reference, Example Component Prompts, and
   Iteration Guide).
2. If a vendor seed fed Phase 2, apply the Seed Application Rules from
   `resources/getdesign-fetcher.md`: adopt color/spacing/components/
   depth/responsive from the seed; rewrite typography, visual theme,
   do-don't, and agent prompt guide from scratch using project
   context.
3. Output design tokens in applicable formats:
   - CSS Custom Properties (`:root { --color-primary: ... }`)
   - Tailwind config extensions (`theme.extend.colors`)
   - shadcn/ui theme variables (if shadcn is in use)
4. Generate component code if requested by the user

### Responsive-First Rule
ALL generated designs MUST be responsive by default. Never produce desktop-only layouts.

Minimum breakpoints to address:
- Mobile (default): 320px-639px
- Tablet (md): 768px+
- Desktop (lg): 1024px+

Every section must specify:
- Mobile layout (stacked, single column)
- Desktop layout (grid, side-by-side)
- Touch targets >= 44x44pt on mobile

---

## Phase 6: AUDIT

The final audit below runs against the synthesized DESIGN.md. Note
that Phase 2 Branch B already ran a **pre-audit** on any vendor seeds
before synthesis. Those findings should appear as decisions in the
final DESIGN.md Section 7 (Do's and Don'ts) rather than as violations
here.

Load `resources/checklist.md` and run all checks in order:

1. **Responsive** (MANDATORY, run first)
   - All sections render at 375px width
   - No horizontal scroll
   - Touch targets >= 44x44pt
   - Navigation collapses appropriately

2. **WCAG 2.2 Accessibility**
   - Text contrast >= 4.5:1 AA
   - Focus indicators visible
   - prefers-reduced-motion respected
   - Semantic HTML landmarks

3. **Nielsen's 10 Heuristics**
   - Visibility of system status
   - Consistency and standards
   - Aesthetic and minimalist design
   - (full list in checklist.md)

4. **AI Slop Check**
   - No purple gradient backgrounds
   - No Inter-only typography
   - No triple-nested cards
   - No bouncing animations everywhere
   - Passes the "AI made this" test

5. **Design System Consistency**
   - All colors from defined palette
   - All spacing from 8px grid scale
   - Typography uses defined scale

Fix violations automatically where possible, or report to user with recommendations.

---

## Phase 7: HANDOFF

1. Save `DESIGN.md` to the project root.
2. If Phase 2 Branch B fired, append the **License Attribution** block
   from `resources/getdesign-fetcher.md` as the final section of
   `DESIGN.md`. This is mandatory for MIT compliance.
3. Update `.design-context.md` if new decisions were made.
4. Write design token files if not already written.
5. Ensure temp seed files from Phase 2 Branch B have been deleted.
6. Inform the user:
   > "Design complete. DESIGN.md has been created.
   >  To implement, delegate to oma-frontend or run /orchestrate."
