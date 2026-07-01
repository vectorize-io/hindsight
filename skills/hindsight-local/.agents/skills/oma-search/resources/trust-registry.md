# Trust Registry Reference

Domain trust scoring strategy for search results.
Uses heuristic pattern matching + Tranco domain ranking for validation.

## Trust Levels

| Level | Score Range | Label | Description |
|-------|-----------|-------|-------------|
| verified | 0.85-0.95 | **** | Official documentation, vendor sites, standards bodies |
| community | 0.50-0.70 | *** | User-generated, curated platforms |
| external | 0.20-0.49 | ** | Third-party content sources |
| unknown | (none) | (none) | Cannot determine trust level |

## Scoring Rules

1. **Domain-level only**: score applies to the entire domain, not sub-paths
2. **Unknown domains are NOT excluded**: they appear with `—` label
3. **`--strict` filter**: only shows results with score >= 0.85 (verified+)
4. **Sort tiebreaker**: when relevance is equal, higher trust score ranks first

## Score Resolution

### Step 1: Classify domain by pattern (category)

First match wins.

#### verified patterns

| Pattern | Score | Rationale |
|---------|-------|-----------|
| Domain matches Context7 library source | 0.95 | Context7 resolved = official docs |
| `docs.*` or `*.docs.*` subdomain | 0.90 | Official documentation subdomain |
| `developers.*` subdomain | 0.85 | Developer portal convention |
| Known TLD: `*.gov`, `*.edu` | 0.90 | Institutional domains |
| Known standards: `w3.org`, `tc39.es`, `ietf.org`, `owasp.org` | 0.95 | Standards bodies |
| Canonical framework/language site | 0.95 | e.g., react.dev, go.dev, doc.rust-lang.org |

#### community patterns

| Pattern | Score | Rationale |
|---------|-------|-----------|
| `github.com`, `gitlab.com` | 0.60 | Code hosting (content quality varies) |
| `stackoverflow.com`, `*.stackexchange.com` | 0.65 | Curated Q&A |
| `wikipedia.org` | 0.65 | Community-curated encyclopedia |
| Package registries: `npmjs.com`, `pypi.org`, `crates.io`, `pub.dev` | 0.60 | Package metadata |
| `reddit.com` | 0.50 | Forum discussion |

#### external patterns

| Pattern | Score | Rationale |
|---------|-------|-----------|
| Blog platforms: `medium.com`, `dev.to`, `hashnode.com`, `substack.com` | 0.35 | Unvetted individual content |
| Blog platforms (KR): `velog.io`, `tistory.com` | 0.30 | Korean blog platforms |
| Tutorial aggregators: `w3schools.com`, `geeksforgeeks.org` | 0.30 | Variable quality |
| Higher-quality tutorials: `freecodecamp.org`, `baeldung.com` | 0.45 | Editorially reviewed |

#### unknown

Everything not matching above patterns.

### Step 2: Validate with Tranco ranking (optional)

Use the Tranco siterank API to verify domain legitimacy:

```bash
curl -s "https://siterank.redirect2.me/api/rank.json?domain={domain}"
```

Response:
```json
{ "success": true, "domain": "react.dev", "rank": 12301 }
{ "success": false, "domain": "sketchy.xyz", "rank": 0 }
```

**Tranco determines popularity, NOT content quality.**
Use it as a validator, not a scorer:

| Tranco Result | Effect on Score |
|--------------|----------------|
| Ranked (any position) | Confirms domain is real and established. Keep heuristic score. |
| Not ranked + pattern matched | Domain is niche but category is clear. Keep heuristic score. |
| Not ranked + no pattern match | Genuinely unknown. Label as `unknown —`. |

**Rate limit**: free for light non-commercial use. Do NOT call for every result.
Call only when heuristic returns `unknown` and you need validation.

### Step 3: Cache in Serena memory

Cache resolved scores to avoid re-evaluation:

```
write_memory("trust-registry-cache", resolved_scores)
read_memory("trust-registry-cache")
```

Cache is project-scoped and survives skill updates.

## Lookup Algorithm

```
1. Extract domain from result URL (strip protocol, path, query)
2. Check Serena memory cache (trust-registry-cache)
3. If cache miss → apply heuristic pattern matching (Step 1)
4. If heuristic returns unknown → call Tranco API for validation (Step 2)
5. Attach [level, tags, score] to result
6. Write newly resolved scores to cache (Step 3)
```

## Data Sources

| Source | Purpose | Cost | API |
|--------|---------|------|-----|
| Heuristic patterns | Domain category classification | Free | N/A (built-in rules) |
| [Tranco List](https://tranco-list.eu/) | Domain popularity validation | Free | `siterank.redirect2.me/api/rank.json` |
| [redirect2me/siterank](https://github.com/redirect2me/siterank) | Self-hostable Tranco API | Free, OSS | Self-host option |
| Serena memory | Score caching | Free | MCP memory tools |
