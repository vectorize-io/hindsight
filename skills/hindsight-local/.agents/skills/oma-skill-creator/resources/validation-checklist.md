# SSL-lite Skill Validation Checklist

Use this checklist after creating or updating a skill.

## Required Structure

- YAML frontmatter exists.
- Frontmatter includes `name`.
- Frontmatter includes a routing-grade `description`.
- The only top-level `##` headings outside code fences are exactly:
  - `## Scheduling`
  - `## Structural Flow`
  - `## Logical Operations`
  - `## References`
- The skill contains exactly one of:
  - `### Canonical command path`
  - `### Canonical workflow path`

## Scheduling Checks

- `Goal` states the capability and outcome.
- `Intent signature` contains concrete user prompt patterns or domain triggers.
- `When to use` gives positive routing cases.
- `When NOT to use` names boundaries and cross-routes to adjacent skills.
- `Expected inputs` and `Expected outputs` are explicit. If the skill produces machine-checkable artifacts, prefer the structured `outputs:` YAML block over freeform bullets so `oma verify` can run a closure check.
- `Dependencies` names tools, files, standards, APIs, or resources.
- `Control-flow features` describes branching, tool calls, writes, and clarification points.

## Cross-Skill Boundary Check

- Run `oma skills audit` (or `oma doctor`) after editing frontmatter `description`.
- Resolve any `FAIL` (≥ 75% similarity) pair by rewriting one description to highlight distinct triggers, domains, or boundaries.
- `WARN` (≥ 60%) pairs are acceptable when descriptions cover genuinely related domains; document the distinction in `When NOT to use` cross-routes.

## Structural Checks

- `Entry` states what to verify before acting.
- `Scenes` use SSL-style scene vocabulary where practical.
- `Transitions` describe condition-to-action routing.
- `Failure and recovery` covers common failures.
- `Exit` defines success, partial success, and failure.

## Logical Checks

- `Actions` map operations to SSL primitives.
- `Tools and instruments` names concrete tools, scripts, commands, APIs, or references.
- The canonical path is executable or operational enough for an agent to follow without extra context.
- `Resource scope` names affected resources such as `CODEBASE`, `LOCAL_FS`, `PROCESS`, `CREDENTIALS`, `NETWORK`, `USER_DATA`, or `MEMORY`.
- `Preconditions` are clear.
- `Effects and side effects` name writes, commands, network calls, generated artifacts, or state changes.
- `Guardrails` protect against unsafe, broad, or low-quality execution.

## Reference Checks

- `References` points only to files that exist or are intentionally planned.
- Long examples and provider-specific variants are in `resources/`, not duplicated inline.
- Reference files are one hop from `SKILL.md`; avoid deep reference chains.

## Suggested Commands

Check top-level headings and canonical path:

```bash
f=".agents/skills/{skill-name}/SKILL.md"
awk 'BEGIN{c=0} /^```/{c=!c; next} !c && /^## /{print $0}' "$f"
rg -n '^### Canonical (command|workflow) path$' "$f"
```

Check formatting whitespace:

```bash
git diff --check -- ".agents/skills/{skill-name}"
```
