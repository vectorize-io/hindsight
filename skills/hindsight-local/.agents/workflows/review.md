---
name: review
description: Full QA review pipeline covering security audit (OWASP Top 10), performance analysis, accessibility check (WCAG 2.2 AA), and code quality review
disable-model-invocation: true
---

# MANDATORY RULES: VIOLATION IS FORBIDDEN

- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
- **NEVER skip steps.** Execute from Step 1 in order.
- **You MUST use MCP tools throughout the workflow.**
  - Use code analysis tools (`get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, `search_for_pattern`) for code analysis and review.
  - Use memory write tool to record review results.
  - Memory path: configurable via `memoryConfig.basePath` (default: `.serena/memories`)
  - Tool names: configurable via `memoryConfig.tools` in `.agents/mcp.json`
  - Do NOT use raw file reads or grep as substitutes.

---

## Vendor Detection

Before starting, determine your runtime environment by following `.agents/skills/_shared/core/vendor-detection.md`.
The detected vendor determines how the QA agent is spawned (Step 7).

### L1 Decision Events

Emit required L1 decisions by calling `oma state:emit` directly, as documented in `.agents/skills/_shared/runtime/event-spec.md`.

---

## Step 1: Identify Review Scope

Ask the user what to review: specific files, a feature branch, or the entire project.
If a PR or branch is provided, diff against the base branch to scope the review.

---

## Step 2: Run Automated Security Checks

// turbo
Run available security tools: `npm audit` (Node.js), `bandit` (Python), or equivalent.
Check for known vulnerabilities in dependencies. Flag any CRITICAL or HIGH findings.

---

## Step 3: Manual Security Review (OWASP Top 10)

Use MCP code analysis tools (`search_for_pattern` and `find_symbol`) to review code for:
- Injection (SQL, XSS, command)
- Broken auth, sensitive data exposure
- Broken access control, security misconfig
- Insecure deserialization
- Known vulnerable components
- Insufficient logging

---

## Step 4: Performance Analysis

Use MCP tools to check for:
- N+1 queries, missing indexes
- Unbounded pagination, memory leaks
- Unnecessary re-renders (React)
- Missing lazy loading
- Large bundle sizes, unoptimized images

---

## Step 5: Accessibility Review (WCAG 2.2 AA)

Check for:
- Semantic HTML, ARIA labels
- Keyboard navigation, color contrast
- Focus management, screen reader compatibility
- Image alt text

---

## Step 6: Code Quality Review

Use MCP code analysis tools (`get_symbols_overview` and `find_referencing_symbols`) to check for:
- Consistent naming, proper error handling
- Test coverage, TypeScript strict mode compliance
- Unused imports/variables
- Proper async/await usage
- Public API documentation

---

## Step 7: Generate QA Report

Compile all findings into a prioritized report:
- **CRITICAL**: Security breaches, data loss risks
- **HIGH**: Blocks launch
- **MEDIUM**: Fix this sprint
- **LOW**: Backlog

Each finding must include: `file:line`, description, and remediation code.
Use memory write tool to record the final report.

After severity classification is complete, emit and verify the required review decision:

```bash
oma state:emit "decision.made" '{"subject":"review.severity-classification","decision":"Use the classified finding severities for the QA report and follow-up routing.","rationale":"Findings have been reviewed and assigned CRITICAL/HIGH/MEDIUM/LOW severity with remediation context."}'
oma state:verify --workflow review --checkpoint severity-classification
```

---

## Agent Delegation: Spawn QA Agent

For large review scopes, delegate Steps 2-7 to a QA agent instead of running inline.

### If Claude Code
Use the Agent tool to spawn subagent:
- `Agent(subagent_type="qa-reviewer", prompt="Review the following files for security, performance, accessibility, and code quality issues: [file list]. Follow .agents/skills/oma-qa/SKILL.md for review standards. Report findings as: CRITICAL / HIGH / MEDIUM / LOW with file:line, description, and remediation code.", run_in_background=true)`

### If Codex CLI
Request parallel subagent execution with the review scope and standards.

### If Gemini CLI or Antigravity or CLI Fallback
```bash
oma agent:spawn qa-agent "Review files for security, performance, accessibility, and code quality. Follow .agents/skills/oma-qa/SKILL.md standards. Report as CRITICAL/HIGH/MEDIUM/LOW with file:line and remediation." session-id
```

---

## Fix-Verify Loop (with --fix option)

When user wants fixes too, execute review then fix then re-review loop:

1. Spawn QA agent (per vendor method above) to get issue list.
2. If CRITICAL/HIGH issues exist:
   - Spawn domain agent to fix issues:

### If Claude Code
     - `Agent(subagent_type="backend-engineer", prompt="Fix these issues: [issues + fix instructions]", run_in_background=true)`
     - `Agent(subagent_type="frontend-engineer", prompt="Fix these issues: [issues + fix instructions]", run_in_background=true)`

### If Codex CLI
     Request parallel subagent execution with the issues and fix instructions.

### If Gemini CLI or Antigravity or CLI Fallback
     ```bash
     oma agent:spawn backend "Fix issues: [issues]" session-id -w ./backend &
     oma agent:spawn frontend "Fix issues: [issues]" session-id -w ./frontend &
     wait
     ```

3. Re-spawn QA agent (per vendor method above) to re-review fixed code.
4. Repeat up to 3 times until no CRITICAL/HIGH issues remain.
