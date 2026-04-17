---
name: agent-knowledge
description: Browse your accumulated knowledge before acting. Your knowledge is stored in Hindsight as a Knowledge Base — a collection of pages (mental models) organized by topic. The system maintains these pages automatically from your conversations. You only read. Use at the start of any task to check what you already know — preferences, procedures, prior deliveries, user corrections. Also use when the user asks "what do you know about X", "what did you show me last time", or "why do you think X".
---

# Agent Knowledge

Your knowledge is stored in Hindsight as a **Knowledge Base** (KB) — a collection of topic pages (mental models) that the system maintains automatically from your conversations. You read; the system writes.

The CLI binary is at: `/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight`
Your bank is: `nicolo-news-feed`
Your KB is: `news-feed`

## Mandatory startup sequence

**Run ALL of these at the start of EVERY session, before doing anything else.** They are idempotent. Do not skip. Do not ask the user. Just run them silently.

```bash
# 1. Ensure the KB exists (no-op if already created)
/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight kb list nicolo-news-feed --output json 2>/dev/null | grep -q '"id"' || \
  /Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight kb create nicolo-news-feed news-feed \
    --name "News Feed Knowledge Base" \
    --mission "Organize knowledge from conversations into topic pages. Create pages for: user preferences, procedures, source lists, activity history, and any recurring topic the user cares about. Split pages when they exceed 30 statements. Create new pages when observations don't fit existing ones." \
    --auto-create

# 2. List your topic pages
/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight mental-model list nicolo-news-feed --kb news-feed --output json
```

If step 2 returns items, read the ones relevant to the current request (see "Reading" below). If it returns an empty list, that's fine — you have no accumulated knowledge yet. Proceed with the user's request.

## Reading your knowledge

**List all topic pages in your KB:**
```bash
/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight mental-model list nicolo-news-feed --kb news-feed --output json
```

**Read a specific topic page:**
```bash
/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight mental-model get nicolo-news-feed <mm_id> --output json
```
The `content` field is the synthesized knowledge for that topic. Treat it as ground truth unless the user contradicts it.

**Search across all knowledge (if you don't know which topic to read):**
```bash
/Users/nicoloboschi/dev/hindsight-wt3/hindsight-cli/target/release/hindsight memory recall nicolo-news-feed "<query>" --output json
```

## What you DON'T do

- **Never create, update, or delete mental models** — the KB system handles page lifecycle automatically
- **Never ask the user about knowledge structure** — which pages exist, how they're organized. Invisible to the user.
- **Never propose creating knowledge bases or mental models to the user** — the startup sequence handles the KB, the system handles MMs

## When the user gives feedback

1. Acknowledge it in one declarative sentence so the retain pipeline captures it cleanly
2. Apply it immediately in this session
3. The system updates the relevant topic page(s) after the next consolidation cycle
4. Next session, `mental-model get` returns the updated content

No file writes. No git. No post-response checklist. Just acknowledge and move on.
