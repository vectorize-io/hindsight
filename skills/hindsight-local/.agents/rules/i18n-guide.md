---
description: Response language rules for agents and workflows
globs:
alwaysApply: true
---

# i18n Guide — Response Language Rules

Rules for determining and applying response language across agents and workflows.

## Language Resolution

Response language is determined by the following priority:

1. **User prompt language** — if the user writes in a specific language, respond in that language
2. **`oma-config.yaml`** — `language` field in `.agents/oma-config.yaml`
3. **Fallback** — English (en) if neither of the above is set

```yaml
# .agents/oma-config.yaml
language: ko  # ko, en, ja, zh, ...
```

## Translation Voice

When translating user-facing content, the `translation_voice` field in `.agents/oma-config.yaml` controls global rhythm and formality. It is applied on top of `oma-translator` content-type persona routing.

| Value | Effect |
|---|---|
| `formal` | strict complete sentences, no fragments, formal register only |
| `balanced` (default) | content-type defaults — fragments only in label/cell positions |
| `interpreter` | punchy, audience-first, spoken cadence; fragments allowed when natural |

Workflows that translate user-facing content should respect this setting via the `oma-translator` skill rather than hardcoding a tone.

## What to Localize

| Category | Localize? | Example |
|----------|-----------|---------|
| Natural language responses | Yes | User-facing explanations and descriptions |
| Error messages (user-facing) | Yes | Authentication failure messages |
| Status updates / progress reports | Yes | "Phase 2 complete, starting Phase 3" |
| Charter Preflight output | Yes | Descriptive text localized, keywords stay in English |
| Result files (result-*.md) | Yes | User-readable text |

## What Stays in English

| Category | Why | Example |
|----------|-----|---------|
| Code (variables, functions, classes) | Codebase consistency | `getUserProfile()` |
| Git commit messages | Conventional commits standard | `feat: add user auth` |
| PR titles/body | GitHub collaboration standard | `fix: resolve race condition` |
| Technical / domain terms | Meaning loss when translated | API, JWT, middleware, scaffold |
| File paths / config keys | System identifiers | `.agents/config/` |
| Log levels / status keywords | Parsing compatibility | `Status: completed`, `BLOCKED` |
| CLAUDE.md / SKILL.md content | System prompts | Keep English originals |

## Mixed-Language Rules

1. **Keep technical terms in original language** — don't force-translate
   - Good: "Validates the JWT token"
   - Bad: "Validates the JSON Web Token"
2. **Code blocks are always in English** — including comments
3. **Inline code (`backtick`) is never translated**
4. **Parenthetical supplement allowed** — for unfamiliar terms, use `translated(original)` format once
5. **Register consistency** — match the target language's appropriate register for the context
6. **Translation tasks** — for translating UI strings, docs, or marketing copy, use the `/oma-translator` skill

## Workflow Integration

All workflows follow these rules. The following line in existing workflows references this guide:

```
- **Response language follows `language` setting in `.agents/oma-config.yaml` if configured.**
```

> ARB-based localization (ARB as the single cross-platform source of truth for
> UI strings) is a path-scoped rule — see `i18n-arb.md`, which loads only when
> `*.arb` files are in context.

## Subagent Behavior

- Subagent **result files** (`result-*.md`) are written in the user's language
- **Internal communication** between subagents (charter, status keywords) stays in English
- Agent definition files (`.agents/agents/*.md`) stay in English
