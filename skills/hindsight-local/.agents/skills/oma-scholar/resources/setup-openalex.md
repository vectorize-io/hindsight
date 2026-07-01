# OpenAlex API Key Setup (Optional)

OpenAlex powers metadata enrichment during Generate mode by backfilling missing DOI, venue, and year from a vast academic catalog. **The skill works without a key**, but enrichment is skipped.

## When you need this

- You frequently generate sidecars from raw text/LaTeX where DOI is not visible
- You want DOI/venue auto-resolved from the title
- You hit anonymous rate limits ("$1/day" allowance exhausted)

## Free Tier (Generous)

Without any key, OpenAlex grants $1/day equivalent free quota:
- **Unlimited** single-entity lookups
- **10,000** list/filter calls per day
- **1,000** search calls per day
- **100** content downloads per day

For most users, anonymous use is enough.

## Get a Key (30 seconds)

1. Sign up: https://openalex.org/
2. Generate key: https://openalex.org/settings/api
3. Copy the key

## Configure

### Option A: Per-shell environment

```bash
export OPENALEX_API_KEY="your_key_here"
```

### Option B: Persisted in `~/.claude/.env` (recommended)

```bash
mkdir -p ~/.claude
echo 'OPENALEX_API_KEY=your_key_here' >> ~/.claude/.env
```

Some Claude Code setups source this file automatically; if not, source it from your shell rc:

```bash
echo '[ -f ~/.claude/.env ] && set -a && . ~/.claude/.env && set +a' >> ~/.zshrc
```

### Option C: Polite-pool email only (no key)

If you don't want to sign up but want priority, set just an email:

```bash
export OPENALEX_EMAIL="you@example.com"
```

This puts you in the "polite pool" with better latency, no key required.

## Verify

```bash
curl -s "https://api.openalex.org/works?search=attention+is+all+you+need&api_key=$OPENALEX_API_KEY" \
  | head -c 500
```

A 200 response with JSON results means the key is live.

## Pricing Beyond Free Tier

OpenAlex uses pay-as-you-go past the daily free allowance, with no monthly subscription. Most academic users never exceed the free tier. See https://openalex.org/pricing for current rates.

## Skill Behavior

When this skill needs metadata enrichment:

1. Check `OPENALEX_API_KEY` env var
2. If set -> use authenticated calls
3. If not set, check `OPENALEX_EMAIL` -> anonymous polite-pool
4. If neither, fall back to anonymous calls
5. On 403/429 -> stop enrichment, tell user to set the key, leave fields omitted (anti-fabrication)

## Privacy

- The key (or email) is sent to OpenAlex servers as a query parameter
- No source paper content is sent to OpenAlex; only title/author search strings
- OpenAlex is operated by OurResearch (a non-profit); see their privacy policy at https://openalex.org/
