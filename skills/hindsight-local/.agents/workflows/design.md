---
name: design
description: Design workflow that creates design systems, DESIGN.md, and design tokens with anti-pattern enforcement and accessibility checks
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip phases.** Execute from Phase 1 in order.
- **Do NOT write implementation code.** This workflow produces DESIGN.md, design tokens, and design guidance, not application code.
- **You MUST use MCP tools throughout the workflow.**
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `search_for_pattern`) to analyze the existing codebase.
  - Use memory tools (write/edit) to record design results.
  - Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`)
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`
  - Do NOT use raw file reads or grep as substitutes.

---

> **Vendor note:** This workflow executes inline (no subagent spawning). All vendors use their native code analysis and file tools.

---

## Phase 1: SETUP (Context Gathering)

Read `.design-context.md` in the project root.

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

Then parse the `## Reference Sites` section (if any) and resolve each
domain against the live `getdesign@latest` manifest. See
`.agents/skills/oma-design/resources/getdesign-fetcher.md`. Hold matched brands in memory for
Phase 2 Branch B. No vendor match = no branch activated.

**Do NOT proceed until design context is established.**

---

## Phase 2: EXTRACT (Optional)

Run branches in priority order, use the first with data:

- **Branch A (Stitch MCP)**: load `.agents/skills/oma-design/resources/stitch-integration.md` if
  Stitch MCP is available; extract designTheme + screens.
- **Branch B (getdesign Vendor Seed)**: if Phase 1 matched any vendor,
  follow `.agents/skills/oma-design/resources/getdesign-fetcher.md`. Fetch via
  `bunx getdesign@latest add <brand> --out <tmp> --force` with
  `GETDESIGN_DISABLE_TELEMETRY=1`, verify SHA256 against manifest,
  load with prompt-injection framing, run anti-pattern pre-audit,
  delete temp.
- **Branch C (Reference URL)**: fetch and analyze HTML/CSS directly.
- **Branch D (No reference)**: skip to Phase 3.

Every branch ends by feeding the 5-stage pipeline:
Retrieval → Extraction → Translation → Synthesis → Alignment.

---

## Phase 3: ENHANCE (Prompt Augmentation)

**Skip if Phase 2 Branch B fired.** Vendor seeds already supply section
detail.

Otherwise, if the user request is vague (< 3 sentences, no section
details):
- Load `.agents/skills/oma-design/resources/prompt-enhancement.md`
- Transform into section-by-section specification
- Present enhanced prompt to user for confirmation

If already detailed: skip to Phase 4.

---

## Phase 4: PROPOSE (Multi-Concept)

// turbo
Default (no vendor seed): present 2-3 distinct design directions. Each
direction includes:
- Color palette (5-7 colors with semantic names and functional roles)
- Typography pairing (system fonts default, custom only with justification)
- Layout approach (chess / grid / bento / full-bleed / mixed)
- Motion strategy (scroll-driven / hover-based / entrance-only / minimal)
- Recommended component libraries (shadcn base + Aceternity / React Bits accents)

Vendor seed present: override with the 3-variation formula
(A Faithful, B Hybrid, C Loose inspiration) and surface any
anti-patterns flagged in the Phase 2 pre-audit. Multi-vendor merges
require the dimension-level selection dialog from
`.agents/skills/oma-design/resources/getdesign-fetcher.md`.

**You MUST get user confirmation on the chosen direction before proceeding.**

---

## Phase 5: GENERATE

// turbo
Based on the chosen direction:
1. Write `DESIGN.md` following `.agents/skills/oma-design/resources/design-md-spec.md` (9 sections,
   including the mandatory Agent Prompt Guide in Section 9)
2. If a vendor seed is in play: apply Seed Application Rules from
   `.agents/skills/oma-design/resources/getdesign-fetcher.md`. Adopt color/spacing/components/
   depth/responsive; rewrite typography for CJK projects; never copy
   the seed's Agent Prompt Guide verbatim.
3. Output design tokens:
   - CSS Custom Properties
   - Tailwind config extensions
   - shadcn/ui theme variables (if applicable)
4. Generate component code if requested

### Responsive-First Rule (MANDATORY)
ALL output must be responsive by default. Never produce desktop-only layouts.
- Mobile (default): 320px-639px
- Tablet (md): 768px+
- Desktop (lg): 1024px+

---

## Phase 6: AUDIT

Load `.agents/skills/oma-design/resources/checklist.md` and run all checks in order:

1. **Responsive** (MANDATORY, run first)
2. **WCAG 2.2 Accessibility**
3. **Nielsen's 10 Heuristics**
4. **AI Slop Check** (anti-patterns.md)
5. **Design System Consistency**

Fix violations or report to user with recommendations.

---

## Phase 7: HANDOFF

1. Save `DESIGN.md` to the project root
2. If Phase 2 Branch B fired, append the License Attribution block
   from `.agents/skills/oma-design/resources/getdesign-fetcher.md` to the bottom of `DESIGN.md`
   (mandatory for MIT compliance).
3. Update `.design-context.md` if new decisions were made
4. Write design token files if not already written
5. Verify all temp seed files have been deleted
6. Inform the user:
   > "Design complete. DESIGN.md has been created. To implement, delegate to oma-frontend or run /orchestrate."
