---
sidebar_position: 6
---

# Sanity CMS Blog Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/sanity-blog-memory)
:::


A Hindsight cookbook recipe demonstrating how to sync blog posts from **Sanity CMS** to Hindsight agent memory, enabling semantic search, temporal queries, and AI-powered content insights.

## Features

- **Blog Post Sync**: Automatically sync all blog posts from Sanity to Hindsight
- **Document-based Upsert**: Idempotent syncing with `document_id` - re-running sync updates existing content
- **Semantic Search**: Find related content using natural language queries
- **Temporal Queries**: Ask "What did I write in January 2025?"
- **Reflect for Insights**: Generate AI-powered analysis of your blog content
- **Related Content Discovery**: Power "Related Posts" features with semantic similarity

## Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │         │                 │
│   Sanity CMS    │───────▶│    Sync Script  │───────▶│   Hindsight     │
│   (Content)     │  GROQ   │   (TypeScript)  │  HTTP   │   (Memory)      │
│                 │         │                 │         │                 │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                                               │
                                                               ▼
                                                        ┌─────────────────┐
                                                        │                 │
                                                        │   Your App      │
                                                        │   - Recall      │
                                                        │   - Reflect     │
                                                        │                 │
                                                        └─────────────────┘
```

## Quick Start

### 1. Start Hindsight

Choose your preferred LLM provider:

**Option A: Using Docker Compose (Recommended)**

```bash
# Set your API key
export OPENAI_API_KEY=sk-...
# OR
export GOOGLE_API_KEY=...  # Gemini (free tier available)
# OR
export GROQ_API_KEY=...    # Groq (free tier available)

# Start Hindsight
docker compose up -d
```

**Option B: Using Docker directly**

```bash
export OPENAI_API_KEY=sk-...

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

- **API**: http://localhost:8888
- **Control Plane UI**: http://localhost:9999

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your values
nano .env
```

Required settings:
```bash
# Hindsight
HINDSIGHT_API_URL=http://localhost:8888
HINDSIGHT_BANK_ID=blog-memory

# Sanity CMS
SANITY_PROJECT_ID=your-project-id
SANITY_DATASET=production
```

### 3. Install Dependencies

```bash
npm install
```

### 4. Sync Your Blog Posts

```bash
npm run sync
```

Expected output:
```
=======================================
  Sanity -> Hindsight Blog Sync
=======================================

Setting up memory bank...
Memory bank "blog-memory" ready

Fetching posts from Sanity CMS...
Found 10 posts to sync

Syncing posts to Hindsight...
  [1/10] "Why I Chose Qwik"... done
  [2/10] "Building AI Agents"... done
  ...

=======================================
  Sync Complete
=======================================
  Synced: 10 posts
```

### 5. Query Your Content

```bash
npm run query
```

## Query Examples

### Semantic Search

Find related content using natural language:

```typescript
import { recallMemory } from './hindsight-client.js';

// Find posts about AI agents
const result = await recallMemory('AI agents and automation', {
  budget: 'mid',
  maxTokens: 2048,
});

console.log(`Found ${result.results.length} relevant posts`);
```

### Temporal Queries

Ask about content from specific time periods:

```typescript
// Posts from January 2025
const result = await recallMemory('What did I write about in January 2025?', {
  queryTimestamp: '2025-01-31T23:59:59Z',
});
```

### Reflect for Insights

Generate AI-powered analysis of your content:

```typescript
import { reflectOnMemory } from './hindsight-client.js';

// Analyze blog themes
const insights = await reflectOnMemory(
  'What are the main themes of my blog? What topics do I write about most?',
  { budget: 'high' }
);

console.log(insights.text);
```

### Related Content Discovery

Power your "Related Posts" feature:

```typescript
// Find posts similar to a specific article
const related = await recallMemory(
  'Find posts related to "Why I Chose Qwik for My Personal Website"',
  { budget: 'mid' }
);
```

## Memory Structure

Each blog post is stored with rich metadata for optimal recall:

```
# Blog Post: {title}

**Published:** {date}
**URL:** {base_url}/blog/{slug}
**Tags:** {tags}
**Reading Time:** {reading_time}

## Description
{description}

## Content
{full_content}
```

Key features:
- **document_id**: `post:{slug}` - Enables upsert on re-sync
- **context**: `blog-post` - Categorizes the memory type
- **timestamp**: Post publication date - Enables temporal queries

## Use Cases

### 1. AI-Powered Blog Search

Replace keyword search with semantic understanding:

```typescript
// Old: keyword matching
const results = posts.filter(p => p.title.includes('React'));

// New: semantic understanding
const result = await recallMemory('frontend framework tutorials');
```

### 2. Content Recommendation Engine

Generate personalized recommendations:

```typescript
const recommendations = await reflectOnMemory(
  'Based on a reader interested in "AI automation", recommend related posts'
);
```

### 3. Writing Assistant

Get topic suggestions based on your existing content:

```typescript
const suggestions = await reflectOnMemory(
  'What topics should I write about next? What gaps exist in my content?'
);
```

### 4. Content Analytics

Analyze your blog's evolution:

```typescript
const analysis = await reflectOnMemory(
  'How have my writing topics evolved over the past year?'
);
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HINDSIGHT_API_URL` | Hindsight API endpoint | `http://localhost:8888` |
| `HINDSIGHT_BANK_ID` | Memory bank identifier | `blog-memory` |
| `SANITY_PROJECT_ID` | Your Sanity project ID | (required) |
| `SANITY_DATASET` | Sanity dataset name | `production` |
| `SANITY_API_TOKEN` | Sanity API token (for private datasets) | (none) |
| `SANITY_API_VERSION` | Sanity API version | `2024-01-09` |
| `SITE_URL` | Your blog's base URL | `https://example.com` |

### Memory Bank Disposition

The memory bank is configured with disposition traits optimized for blog content:

```typescript
{
  skepticism: 2,   // Trusting - blog content is authoritative
  literalism: 4,   // Literal - exact content matters
  empathy: 3,      // Balanced
}
```

## Extending for Other CMS Platforms

This pattern can be adapted for any CMS. The key components:

### 1. CMS Client

Replace `sanity-client.ts` with your CMS:

```typescript
// contentful-client.ts
import { createClient } from 'contentful';

export async function getAllPosts(): Promise<BlogPost[]> {
  const client = createClient({...});
  const entries = await client.getEntries({ content_type: 'blogPost' });
  return entries.items.map(transformPost);
}
```

### 2. Content Transformation

Ensure your content is formatted for semantic search:

```typescript
function formatPostContent(post: BlogPost): string {
  return `# ${post.title}
  
**Published:** ${post.date}
...
${post.content}`;
}
```

### 3. Document ID Strategy

Use a consistent document ID for upsert behavior:

```typescript
await retainBlogPost(content, {
  documentId: `post:${post.slug}`,  // Unique, stable identifier
  timestamp: post.date,
});
```

## Troubleshooting

### "Connection refused" error

Make sure Hindsight is running:
```bash
docker compose up -d
curl http://localhost:8888/health
```

### "No posts found" during sync

Check your Sanity configuration:
```bash
# Verify project ID
echo $SANITY_PROJECT_ID

# Test GROQ query
npx sanity query '*[_type == "post"][0..2]{title}'
```

### Slow recall/reflect responses

This is normal for the first query as Hindsight builds embeddings. Subsequent queries are faster. Use `budget: 'low'` for faster responses at the cost of recall quality.

## Resources

- [Hindsight Documentation](https://hindsight.vectorize.io/)
- [Hindsight GitHub](https://github.com/vectorize-io/hindsight)
- [Sanity CMS Documentation](https://www.sanity.io/docs)
- [Hindsight Cookbook](https://github.com/vectorize-io/hindsight-cookbook)

## License

MIT
