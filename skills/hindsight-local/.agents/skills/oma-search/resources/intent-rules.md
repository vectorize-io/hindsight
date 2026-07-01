# Intent Classification Rules

## Classification Priority

1. **Override flags**: always win, skip classification entirely
2. **Keyword pattern matching**: scan query for mode-specific keywords
3. **Signal detection**: contextual clues (library name + version, error message, etc.)
4. **Fallback**: `web` + `docs` parallel when no clear signal

## Override Flags

| Flag | Forced Mode | Description |
|------|-------------|-------------|
| `--docs` | `docs` | Official documentation only via Context7 |
| `--code` | `code` | GitHub/GitLab code search only |
| `--web` | `web` | Web search only (native + `oma search fetch` fallback) |
| `--strict` | (modifier) | Filter results to verified+ sources (score >= 0.85) |
| `--wide` | (modifier) | Show all sources with trust labels (no filtering) |
| `--gitlab` | (modifier) | Force `glab api` instead of `gh` for code route |

## Keyword Patterns

### docs mode

**Keywords:** official, docs, documentation, API ref, reference, spec, specification

**Signals:**
- Library/framework name + version mentioned (e.g., "React 19 docs")
- Specific API or method name + "how to use"

### web mode

**Keywords:** example, tutorial, how to, guide, blog, comparison, vs, use case, best practice

**Signals:**
- Opinion/comparison queries ("X vs Y", "best way to")
- Error messages or stack traces (solution search)
- Trend or news queries

### code mode

**Keywords:** implementation, pattern, repo, repository, source code, code search, codebase

**Signals:**
- Language/framework + "how is it implemented"
- Specific algorithm or pattern name
- URL containing github.com or gitlab.com

### local mode

**Keywords:** this project, here, this file, function, class, method, variable, our code

**Signals:**
- References to current codebase context
- Relative file paths mentioned
- "Where is X defined" type queries

> Multi-language activation (Korean, Japanese, etc.) is owned by `.agents/hooks/core/triggers.json §oma-search.keywords`. This file documents English mode classifiers only.

## Fallback Rule

When no clear intent is detected:
- Dispatch `web` + `docs` in parallel
- Merge results with trust scoring
- Present combined ranked output

## Ambiguity Resolution Examples

| Query | Detected Mode | Reason |
|-------|--------------|--------|
| "React useState" | `web` + `docs` (fallback) | Ambiguous: could be docs or tutorial |
| "React useState official docs" | `docs` | "official docs" keyword |
| "React useState use case" | `web` | "use case" keyword |
| "React useState implementation" | `code` | "implementation" keyword |
| "find handleAuth function" | `local` | "function" + "find" signals local |
| "Next.js vs Remix" | `web` | "vs" comparison signal |
| "OAuth PKCE github" | `code` | "github" platform signal |
