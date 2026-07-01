# Search Agent - Input/Output Examples

## Example 1: docs route (auto-detected)

**Input:**
```
/oma-search "React useEffect official docs"
```

**Intent Classification:**
- Keyword "official docs" detected -> `docs` mode

**Output:**
```
Query: "React useEffect official docs"
Mode: docs (auto)

DOCS #1  useEffect — react.dev                  [verified,lang-docs 0.95]
DOCS #2  Synchronizing with Effects — react.dev  [verified,lang-docs 0.95]
DOCS #3  useEffect API Reference — react.dev     [verified,lang-docs 0.95]
```

## Example 2: web route (auto-detected)

**Input:**
```
/oma-search "Redis pub/sub use cases"
```

**Intent Classification:**
- Keyword "use cases" detected -> `web` mode

**Output:**
```
Query: "Redis pub/sub use cases"
Mode: web (auto)

WEB  #1  Redis Pub/Sub Guide — redis.io/docs     [verified,vendor-docs 0.90]
WEB  #2  Real-time Chat with Redis — dev.to      [external,blog 0.40]
WEB  #3  Redis Pub/Sub Patterns — stackoverflow   [community,qa 0.65]
WEB  #4  Redis Event-Driven Architecture — medium [external,blog 0.35]
```

## Example 3: code route (flag override)

**Input:**
```
/oma-search --code "OAuth PKCE implementation TypeScript"
```

**Intent Classification:**
- `--code` flag -> skip classification, forced `code` mode

**Output:**
```
Query: "OAuth PKCE implementation TypeScript"
Mode: code (flag)

CODE #1  src/auth/pkce.ts — panva/oauth4webapi         [github]
CODE #2  lib/pkce.ts — auth0/nextjs-auth0               [github]
CODE #3  packages/oauth/src/pkce.ts — lucia-auth/lucia   [github]
```

## Example 4: Ambiguous query (fallback parallel)

**Input:**
```
/oma-search "Next.js middleware"
```

**Intent Classification:**
- No clear keyword signal -> fallback `web` + `docs` parallel

**Output:**
```
Query: "Next.js middleware"
Mode: docs + web (auto)

DOCS #1  Middleware — nextjs.org/docs             [verified,lang-docs 0.95]
DOCS #2  next.config.js — nextjs.org/docs         [verified,lang-docs 0.95]
WEB  #3  Middleware Deep Dive — vercel.com/blog    [verified,vendor 0.90]
WEB  #4  Next.js Middleware Guide — freecodecamp   [external,tutorial 0.45]
```

## Example 5: --strict mode

**Input:**
```
/oma-search --strict "JWT refresh token rotation"
```

**Intent Classification:**
- `--strict` modifier -> filter to verified+ (>= 0.85)
- No route flag -> fallback `web` + `docs`

**Output:**
```
Query: "JWT refresh token rotation"
Mode: docs + web (auto, strict)

DOCS #1  JWT Best Practices — datatracker.ietf.org  [verified,standards 0.95]
WEB  #2  Token Rotation — auth0.com/docs             [verified,vendor-docs 0.90]
WEB  #3  Refresh Token Rotation — auth0.com/docs       [verified,vendor-docs 0.90]

Filtered: 4 results below trust threshold (0.85). Use --wide to see all.
```

## Example 6: GitLab code search

**Input:**
```
/oma-search --code --gitlab "CI pipeline caching strategy"
```

**Output:**
```
Query: "CI pipeline caching strategy"
Mode: code (flag, gitlab)

CODE #1  .gitlab-ci.yml — gitlab-org/gitlab          [gitlab]
CODE #2  ci/cache.yml — gitlab-org/omnibus-gitlab     [gitlab]
CODE #3  .gitlab-ci.yml — getsentry/sentry            [gitlab]
```

## Example 7: Shared infrastructure call (from oma-debug)

**Context:** oma-debug encounters an unfamiliar error message

**Internal call:**
```
Intent: web
Query: "TypeError: Cannot read properties of undefined (reading 'map') React"
```

**Output returned to oma-debug:**
```
Query: "TypeError: Cannot read properties of undefined (reading 'map') React"
Mode: web (explicit)

WEB  #1  Common React Error Fix — stackoverflow      [community,qa 0.65]
WEB  #2  Handling Undefined in React — react.dev      [verified,lang-docs 0.95]
WEB  #3  TypeError Debugging — dev.to                 [external,blog 0.40]
```

## Example 8: docs fallback to web

**Input:**
```
/oma-search --docs "htmx form validation"
```

**Context7 fails to resolve "htmx" library.**

**Output:**
```
Query: "htmx form validation"
Mode: docs -> web (fallback)

Could not find official docs via Context7. Showing web results.

WEB  #1  Form Validation — htmx.org                  [verified,lang-docs 0.90]
WEB  #2  htmx Validation Guide — dev.to              [external,blog 0.40]
WEB  #3  htmx Form Patterns — reddit.com/r/htmx      [community,forum 0.50]
```
