---
name: oma-search
description: Intent-based search router with trust scoring. Routes queries to optimal channels (Context7 docs, native web search, gh/glab code search, Serena local) and attaches domain trust labels. Use for search, find, lookup, reference, docs, code search, and web research.
---

# Search Agent - Intent-Based Search Router

## Scheduling

### Goal
Classify information-seeking requests, route them to the best search channel, attach trust labels, and return source-grounded results.

### Intent signature
- User asks to search, find, look up, reference docs, inspect official documentation, search GitHub/GitLab code, or gather web research.
- Another skill needs reusable search infrastructure with trust scoring.

### When to use
- Finding official library/framework documentation
- Web research for tutorials, examples, comparisons, and solutions
- Searching GitHub/GitLab code for implementation patterns
- Any query where the search channel is unclear (auto-routing)
- Other skills needing search infrastructure (shared invocation)

### When NOT to use
- Local codebase exploration only -> use Serena MCP directly
- Git history or blame analysis -> use SCM Agent
- Full architecture research -> use Architecture Agent (may invoke this skill internally)

### Expected inputs
- Query string, intent hint, or explicit flags such as `--docs`, `--code`, `--web`, `--strict`, `--wide`, `--gitlab`
- Optional required source type, recency, domain, or trust constraints

### Expected outputs
- Ranked search results with route, source, trust label, and concise relevance summary
- Fallback explanation when primary route fails
- Source links or references suitable for the calling skill

### Dependencies
- Context7 MCP for docs, runtime-native web search, `gh`/`glab` for code, Serena for local search
- `resources/intent-rules.md`, `resources/trust-registry.md`, execution protocol, examples, and checklist

### Control-flow features
- Branches by classified intent, user flags, route success/failure, and trust constraints
- May call web/docs/code/local tools
- Scores domains at domain level only

## Structural Flow

### Entry
1. Parse the query and flags.
2. Classify the search intent.
3. Select one best route unless ambiguity or flags justify more.

### Scenes
1. **PREPARE**: Parse query and classify route.
2. **ACT**: Dispatch to docs, web, code, or local search.
3. **ACQUIRE**: Collect search results and source metadata.
4. **VERIFY**: Apply trust scoring and route-specific quality checks.
5. **FINALIZE**: Present ranked results or fallback status.

### Transitions
- If `--docs`, `--code`, `--web`, `--strict`, `--wide`, or `--gitlab` is provided, flags override classifier.
- If docs route fails, fall back to web.
- If web search needs fetch escalation, use `oma search fetch` strategies.
- If query is purely local, use Serena MCP instead of web.

### Failure and recovery
- If primary route fails, fall forward to the next appropriate route.
- If trust score is weak, label it instead of hiding uncertainty.
- If no reliable results exist, report that and suggest a narrower query.

### Exit
- Success: results are routed, trust-scored, and source-grounded.
- Partial success: route failures or trust limitations are explicit.

## Logical Operations

### Actions
| Action | SSL primitive | Evidence |
|--------|---------------|----------|
| Parse query and flags | `READ` | User request |
| Classify intent | `SELECT` | Intent rules |
| Dispatch search route | `CALL_TOOL` | Docs, web, code, local tools |
| Collect results | `READ` | Search outputs |
| Score trust | `VALIDATE` | Trust registry |
| Rank and format | `INFER` | Relevance and trust |
| Report results | `NOTIFY` | Final answer |

### Tools and instruments
- Context7 docs tools
- Runtime-native web search
- `gh search code` or `glab api`
- Serena MCP for local project search

### Canonical command path
```bash
gh search code "<query>"
glab api "/search?scope=blobs&search=<query>"
```

For docs and web routes, use the runtime's available official-docs or web-search tools after classifying intent; do not duplicate routes unless the intent is ambiguous.

### Resource scope
| Scope | Resource target |
|-------|-----------------|
| `NETWORK` | Web/docs/source-code search targets |
| `CODEBASE` | Local files when local search is selected |
| `PROCESS` | `gh`, `glab`, and CLI search commands |
| `MEMORY` | Query classification, trust labels, selected results |

### Preconditions
- Query and route constraints are clear enough to classify.
- Required search tools are available or fallback is possible.

### Effects and side effects
- Performs external searches or local code searches.
- Produces ranked references that may influence downstream implementation or research.

### Guardrails
1. **Classify intent before searching**: every query goes through IntentClassifier first
2. **One query, one best route**: avoid redundant multi-route unless intent is ambiguous
3. **Trust score every result**: all non-local results get domain trust labels from the registry
4. **Flags override classifier**: user-provided flags (`--docs`, `--code`, `--web`, `--strict`, `--wide`, `--gitlab`) always take precedence
5. **Fail forward**: if primary route fails, fall back gracefully (docs->web, web->`oma search fetch` strategies)
6. **No additional MCP required**: Context7 for docs, runtime native for web, CLI for code, Serena for local
7. **Vendor-agnostic web search**: use whatever the current runtime provides (WebSearch, Google, Bing)
8. **Domain-level trust only**: do not attempt sub-path or page-level scoring

### Routes

| Route | Primary Tool | Fallback | Trigger |
|-------|-------------|----------|---------|
| `docs` | Context7 MCP (`resolve-library-id` → `query-docs`) | `web` route | Official docs, API reference |
| `web` | Runtime native search | `oma search fetch` (api/probe/impersonate/browser) | Tutorials, examples, solutions |
| `code` | `gh search code` / `glab api` | (none) | Implementation patterns, repos |
| `local` | Serena MCP (delegate) | (none) | Current project files, symbols |

### Default Workflow
1. **Parse**: Extract query, detect flags, classify intent
2. **Route**: Dispatch to the appropriate search channel(s)
3. **Collect**: Gather results from dispatched routes
4. **Score**: Attach trust labels to each result domain
5. **Present**: Format and rank results for the user

### Invocation

#### Standalone
```
/oma-search "React Server Components streaming"
/oma-search --docs "Next.js middleware"
/oma-search --code "PKCE implementation"
/oma-search --strict "JWT refresh token rotation"
```

#### Shared Infrastructure (from other skills)
Other skills reference oma-search by specifying intent and query:
1. State intent: `docs` | `web` | `code` | `local`
2. Pass query string
3. Use Trust Score in results to weigh source reliability

## References
Follow `resources/execution-protocol.md` step by step.
See `resources/examples.md` for input/output examples.
Use `resources/intent-rules.md` for intent classification reference.
Use `resources/trust-registry.md` for domain trust scoring reference.
Before submitting, run `resources/checklist.md`.
Vendor-specific execution protocols are injected automatically by `oma agent:spawn`.
Source files live under `../_shared/runtime/execution-protocols/{vendor}.md`.
- Execution steps: `resources/execution-protocol.md`
- Intent classification: `resources/intent-rules.md`
- Trust registry: `resources/trust-registry.md`
- Examples: `resources/examples.md`
- Checklist: `resources/checklist.md`
- Error recovery: `resources/error-playbook.md`
- Context loading: `../_shared/core/context-loading.md`
- Context budget: `../_shared/core/context-budget.md`
- Lessons learned: `../_shared/core/lessons-learned.md`
