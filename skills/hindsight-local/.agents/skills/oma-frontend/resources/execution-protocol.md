# Frontend Agent - Execution Protocol

## Step 0: Prepare
1. **Assess difficulty**: see `../../_shared/core/difficulty-guide.md`
   - **Simple**: Skip to Step 3 | **Medium**: All 4 steps | **Complex**: All steps + checkpoints
2. **Check lessons**: read your domain section in `../../_shared/core/lessons-learned.md`
3. **Clarify requirements**: follow `../../_shared/core/clarification-protocol.md`
   - Check **Uncertainty Triggers**: business logic, security/auth, existing code conflicts?
   - Determine level: LOW → proceed | MEDIUM → present options | HIGH → ask immediately
4. **Budget context**: follow `../../_shared/core/context-budget.md` (read symbols, not whole files)

**Intelligent Escalation**: When uncertain, escalate early. Don't blindly proceed.

Follow these steps in order (adjust depth by difficulty).

## Step 1: Analyze
- Read the task requirements carefully
- Read `DESIGN.md` if present at the project root; treat Section 9 (Agent Prompt Guide) as authoritative component spec
- Identify which components, pages, and hooks are needed
- Check existing code with Serena: `get_symbols_overview("src/components")`, `find_symbol("ComponentName")`
- Review existing patterns: `find_referencing_symbols("Button")` to understand usage conventions
- List assumptions; ask if unclear

## Step 2: Plan
- Decide on component structure (which are new, which extend existing)
- Define props interfaces with TypeScript
- Plan state management approach (Jotai or Zustand for client, nuqs for URL, TanStack Query for server)
- Identify API integration points (TanStack Query hooks)
- Plan responsive breakpoints and accessibility requirements

## Step 3: Implement
- Create/modify files in this order:
  1. TypeScript types/interfaces
  2. API client hooks (TanStack Query)
  3. Reusable UI components (shadcn/ui based)
  4. Feature components (compose UI + logic)
  5. Page components (route-level)
  6. Tests (unit + integration)
- Follow `resources/tailwind-rules.md` for styling

## Step 4: Verify
- Run `resources/checklist.md` items
- Run `../../_shared/core/common-checklist.md` items
- Check TypeScript strict mode: no errors
- Verify responsive design at 320px, 768px, 1024px, 1440px
- Test keyboard navigation and screen reader compatibility

## On Error
See `resources/error-playbook.md` for recovery steps.
