---
sidebar_position: 7
---

# Stance Tracker


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/stancetracker)
:::


An AI-powered application that tracks political candidates' stances on issues over time using Hindsight memory system and web scraping.

## Features

- **Geographic Targeting**: Track stances by country, state/province, and city
- **Multi-Candidate Tracking**: Monitor multiple candidates simultaneously
- **Temporal Analysis**: Historical stance tracking with configurable time ranges
- **Automated Scraping**: Periodic content collection with configurable frequencies (hourly/daily/weekly)
- **Stance Change Detection**: Automatic detection and highlighting of position changes
- **Interactive Timeline**: Visual graph showing stance evolution with reference callouts
- **Source Attribution**: All stances linked to verified sources with excerpts

## Architecture

### Memory System (Hindsight Integration)

This app uses the Hindsight memory system from `github.com/vectorize-io/hindsight`:

1. **Banks**: Each scraper agent has its own memory bank
2. **Retain**: Stores candidate statements and web scraping results
3. **Recall**: Semantic search to retrieve relevant memories
4. **Reflect**: Generates contextual analysis using stored memories
5. **Temporal Search**: Queries memories within specific time periods

### Tech Stack

- **Frontend**: Next.js 16, React, TypeScript, TailwindCSS
- **Visualization**: Recharts for timeline graphs
- **Backend**: Next.js API routes
- **Memory**: Hindsight (from github.com/vectorize-io/hindsight)
- **Database**: JSON file storage (no database required)
- **Web Search**: Tavily API
- **LLM**: OpenAI/Anthropic/Groq (configurable)
- **Scheduling**: node-cron

## Prerequisites

1. **Hindsight API** running (from github.com/vectorize-io/hindsight)
2. **API Keys**:
   - Tavily API key (for web search)
   - LLM provider API key (OpenAI, Anthropic, or Groq)

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Hindsight API (from github.com/vectorize-io/hindsight)
HINDSIGHT_API_URL=http://localhost:8888

# Tavily API (for web search)
TAVILY_API_KEY=your_tavily_api_key_here

# LLM Provider
LLM_PROVIDER=openai  # or anthropic, groq
LLM_API_KEY=your_llm_api_key_here
LLM_MODEL=gpt-4-turbo-preview
```

### 3. Start Hindsight

Clone and run Hindsight from github.com/vectorize-io/hindsight:

```bash
# Clone and run github.com/vectorize-io/hindsight
cd /path/to/hindsight
cargo run --bin hindsight-server
```

Verify Hindsight is running at `http://localhost:8888`

### 4. Run the Application

```bash
npm run dev
```

Visit `http://localhost:3000`

## Usage

### Creating a Tracking Session

1. **Set Location**: Enter country (required), state/province, and city (optional)
2. **Choose Topic**: Specify the issue to track (e.g., "Climate Change Policy")
3. **Add Candidates**: Enter names of candidates/politicians to track
4. **Configure Time Range**: Set historical start/end dates for initial analysis
5. **Set Frequency**: Choose how often to check for updates (hourly/daily/weekly)
6. **Start Tracking**: Click "Start Tracking" to begin

### Viewing Results

- **Timeline Graph**: Shows confidence levels of each candidate's stance over time
- **Stance Changes**: Red circles on the graph indicate detected position changes
- **Click Points**: Click any point to see detailed stance information and sources
- **Source Links**: Each stance includes links to original references

### Managing Sessions

- **Pause/Resume**: Temporarily stop or restart tracking
- **Run Now**: Trigger an immediate update outside the schedule
- **Status**: View current session status and frequency

## API Endpoints

### Sessions

- `POST /api/sessions` - Create new tracking session
- `GET /api/sessions?id={id}` - Get session details
- `GET /api/sessions` - List all sessions
- `PATCH /api/sessions` - Update session status

### Stances

- `POST /api/stances` - Process candidate stance
- `GET /api/stances?sessionId={id}&candidate={name}` - Get stances

### Scheduler

- `POST /api/scheduler` - Control session scheduling
  - Actions: `start`, `stop`, `run`

## Hindsight Integration Examples

### 1. Storing Memories

```typescript
// Store web scraping results
await hindsightClient.retain(bankId, articleContent, {
  context: 'web_search_result',
  timestamp: articleDate,
  metadata: { url: articleUrl }
});
```

### 2. Semantic Search

```typescript
// Search for relevant memories
const results = await hindsightClient.recall(bankId, query, {
  budget: 'high',
  maxTokens: 8192
});
```

### 3. Temporal Filtering

```typescript
// Query memories up to a specific point in time
const results = await hindsightClient.recall(bankId, query, {
  queryTimestamp: '2024-12-01T00:00:00Z'
});
```

### 4. Contextual Analysis

```typescript
// Generate analysis using stored memories
const response = await hindsightClient.reflect(bankId,
  'What is the candidate\'s stance on this issue?',
  { budget: 'high' }
);
```

## Production Deployment

### Vercel Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables in Vercel dashboard:
# - HINDSIGHT_API_URL
# - TAVILY_API_KEY
# - LLM_PROVIDER
# - LLM_API_KEY
# - LLM_MODEL
```

**Note**: The `data/` directory for JSON storage will be ephemeral on Vercel. For production, consider using a persistent database or object storage.

## Development

### Project Structure

```
stancetracker/
├── app/
│   ├── api/          # API routes
│   ├── globals.css   # Global styles
│   ├── layout.tsx    # Root layout
│   └── page.tsx      # Main page
├── components/       # React components
├── lib/
│   ├── db/          # JSON database utilities
│   ├── hindsight-client.ts    # Hindsight API client
│   ├── llm-client.ts       # LLM provider client
│   ├── web-scraper.ts      # Tavily web scraper
│   ├── scraper-agent.ts    # Content scraper
│   ├── rag-system.ts       # Memory retrieval
│   ├── stance-extractor.ts # Stance analysis
│   ├── stance-pipeline.ts  # Main pipeline
│   └── scheduler.ts        # Job scheduling
└── types/           # TypeScript types
```

### Adding New LLM Providers

Edit `lib/llm-client.ts` and add a new method:

```typescript
private async newProviderComplete(messages, options) {
  // Implementation
}
```

## Limitations

- **Web Search**: Uses Tavily API which has rate limits
- **Source Verification**: Manual verification recommended for critical applications
- **Stance Extraction**: LLM-based, subject to model limitations
- **Storage**: JSON file storage is not suitable for high-scale production use
- **Rate Limits**: Respect API rate limits for Tavily, Hindsight, and LLM providers

## Future Enhancements

- [ ] Real-time social media monitoring
- [ ] Speech/video transcription analysis
- [ ] Multi-language support
- [ ] Sentiment analysis integration
- [ ] Comparative analysis dashboard
- [ ] Export to CSV/PDF
- [ ] Email notifications for stance changes
- [ ] Public API for third-party integrations

## License

MIT

## Support

For issues or questions, please check:
- Hindsight documentation: `github.com/vectorize-io/hindsight/README.md`
- Tavily API docs: https://tavily.com/
- Project issues: Create an issue in the repository
