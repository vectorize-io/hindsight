<div align="center">

# 🎬 ReelCraft AI

### AI-powered Reel recreation assistant — reverse-engineers any video into a complete, personalized production blueprint.

[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js)](https://nextjs.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-3.8_Flash-blue?logo=google)](https://aistudio.google.com)
[![Hindsight](https://img.shields.io/badge/Memory-Hindsight-purple)](https://hindsight.vectorize.io)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-blue?logo=typescript)](https://typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?logo=tailwindcss)](https://tailwindcss.com)

</div>

---

## 📸 Screenshots

### Home — "How did they make that Reel?"
![ReelCraft AI Home Page](public/screenshots/02-home.png)
> The hero page with the 5-step workflow and feature overview. Dark, creator-focused UI.

---

### Analyze — Upload a Reel
![Analyze Page — Video Upload](public/screenshots/01-analyze.png)
> Upload a video file or paste a social media URL. Hindsight memory indicator shows personalization is active.

---

### Create My Version
![Create My Version Page](public/screenshots/03-create.png)
> Fill in your topic, equipment, skill level, platform, and duration. The AI generates a completely original Reel plan inspired by the analyzed structure.

---

### Projects — All Analyzed Reels
![Projects Page](public/screenshots/04-projects.png)
> All previously analyzed Reels saved with their full recreation guides, timestamps, and status badges.

---

### Creative Memory — Powered by Hindsight
![Creative Memory Page](public/screenshots/05-memory.png)
> The Hindsight memory dashboard shows all stored creator preferences — skill level, equipment, editing software, style — categorized and explained.

---

### Chat — AI Content Coach
![Chat Page](public/screenshots/06-chat.png)
> Persistent AI chatbot that remembers your preferences across sessions. Quick prompts for common content creation questions.

---

## The Problem

A content creator watches a viral Reel and thinks: *"How did they actually make this?"*

Existing tools summarize videos. They describe what happens. But they don't tell you **how to make it** — what camera angle was used, how the lighting was set up, how the edit was structured, what transitions were used, what software to use, what shots to film.

---

## The Solution

ReelCraft AI does not summarize videos. It **reverse-engineers the production process**.

Upload any Reel → get a complete **23-section production blueprint**:

| Section | What you get |
|---------|-------------|
| 1 | Reel overview & creative strategy |
| 2 | Required skill level (Beginner / Intermediate / Advanced) |
| 3 | Skills to learn before starting |
| 4 | Equipment list (Must Have / Nice to Have / Professional) |
| 5 | Apps & software recommendations |
| 6 | Filming techniques with Why? explanations |
| 7 | Pre-production checklist |
| 8 | Script / voiceover structure |
| 9 | Complete shot list with filming instructions |
| 10 | Detailed filming instructions (Beginner + Pro method) |
| 11 | Lighting setup with ASCII diagram |
| 12 | Audio & music plan |
| 13 | Step-by-step editing timeline |
| 14 | App-specific guide (CapCut + Premiere Pro) |
| 15 | Color grading settings |
| 16 | Text & graphics guide |
| 17 | Transitions breakdown |
| 18 | Export settings (Instagram / TikTok / YouTube) |
| 19 | Publishing guide with hashtags & caption |
| 20 | Beginner version (phone + free apps only) |
| 21 | Professional version |
| 22 | Time & budget estimate |
| 23 | Interactive final checklist |

And using **Hindsight** as its persistent memory system, ReelCraft AI **learns your workflow** and adapts every guide to you specifically.

---

## What Makes ReelCraft AI Different

| Feature | Generic AI Tools | ReelCraft AI |
|---------|-----------------|--------------|
| Output | Summary of what happens | Production blueprint for HOW to make it |
| Personalization | None | Adapts to your skill, gear, and software via Hindsight |
| Memory | Session only | Persistent across sessions (Hindsight) |
| Shot guidance | Generic advice | Specific: distance, angle, height, focus, movement |
| Editing guide | General tips | App-specific steps for CapCut AND Premiere Pro |
| Beginner support | Assumed knowledge | Explains every technique from scratch |
| Learning mode | N/A | Teaches the WHY behind every technique |

---

## Architecture

```
reelcraft-ai/
├── app/                        # Next.js 15 App Router
│   ├── page.tsx                # Home page
│   ├── layout.tsx              # Root layout + Navbar
│   ├── api/
│   │   ├── analyze/route.ts    # Video analysis pipeline
│   │   ├── chat/route.ts       # AI chatbot with Hindsight
│   │   ├── memory/route.ts     # Hindsight memory CRUD
│   │   ├── create-version/     # Custom Reel generation
│   │   ├── projects/           # Project CRUD
│   │   └── demo/route.ts       # Demo seeding
│   └── (pages)/
│       ├── analyze/            # Upload / URL input
│       ├── projects/[id]/      # Full 23-section guide
│       ├── create/             # Create My Version
│       ├── memory/             # Creative Memory dashboard
│       ├── chat/               # AI Content Coach
│       ├── projects/           # Project list
│       └── demo/               # Demo launcher
│
├── components/
│   ├── layout/Navbar.tsx
│   ├── ui/
│   │   ├── MemoryBanner.tsx    # 🧠 Hindsight recall display
│   │   ├── SectionCard.tsx     # Collapsible guide sections
│   │   ├── VideoUploader.tsx
│   │   └── WhyButton.tsx       # "Why?" tooltip per technique
│   └── analysis/
│       ├── SceneTimeline.tsx   # Interactive scene visualization
│       ├── ShotListTable.tsx   # Shot-by-shot guide
│       ├── EquipmentGrid.tsx
│       ├── TechniqueCards.tsx
│       └── FinalChecklist.tsx  # Interactive with progress bar
│
├── lib/
│   ├── hindsight/              # Hindsight memory service
│   ├── video/                  # ffprobe metadata + frame extraction
│   ├── agent/                  # Gemini vision analysis + chat
│   └── db/                     # JSON project store
│
└── data/demo.ts                # Complete seeded demo project
```

---

## Video Analysis Pipeline

```
Video Upload / URL
        │
        ▼
  1. VALIDATE  (file type, size, URL domain)
        │
        ▼
  2. EXTRACT METADATA  (ffprobe — duration, resolution, fps, orientation)
        │
        ▼
  3. EXTRACT FRAMES  (ffmpeg — 10 evenly-spaced frames → base64 JPEG)
        │
        ▼
  4. RECALL HINDSIGHT MEMORY  (creator profile summary)
        │
        ▼
  5. GEMINI VISION ANALYSIS  (frames + metadata + creator profile → 23-section JSON)
        │
        ▼
  6. PARSE + STORE  (project DB + Hindsight insight)
        │
        ▼
  7. RETURN  (complete guide + memoryUsed + recalledPreferences)
```

---

## Hindsight Memory Architecture

### Memory Strategy — Store What Matters

ReelCraft AI does **not** store every message. It stores **meaningful creator facts**:

```
USER PROFILE          → skill level, device, editing software, platforms
CREATIVE PREFERENCES  → style, pacing, color grade, music, hook style
WORKFLOW              → apps owned, filming setup, equipment
PAST PROJECTS         → previous analyses + key insights
```

### Memory Flow

```
User sends message
        │
        ├─ Heuristic check: contains preference signals?
        │   ("I use", "my phone", "beginner", "Premiere Pro"…)
        │
        ├─ YES → Gemini extraction → retainBatch() to Hindsight
        └─ NO  → skip (no wasted API calls)

Before generating guide:
        └─ recall() → reflect() → inject into system prompt
                                 → guide adapts automatically
```

### SDK Usage

```typescript
// Store a preference
await client.retain(bankId, "Creator edits using Premiere Pro", {
  metadata: { category: "workflow", app: "reelcraft-ai" },
});

// Search memories
const result = await client.recall(bankId,
  "skill level, equipment, editing software"
);

// AI-synthesized profile
const profile = await client.reflect(bankId,
  "Summarize this creator's workflow in 3-4 sentences"
);
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 15 (App Router) |
| Language | TypeScript 5.7 |
| Styling | Tailwind CSS 3.4 |
| AI / LLM | Google Gemini 3.8 Flash (`@google/genai`) — **free tier** |
| Memory | Hindsight (`@vectorize-io/hindsight-client`) |
| Video Processing | ffmpeg + ffprobe (`fluent-ffmpeg`) |
| Icons | `lucide-react` |
| Markdown | `react-markdown` + `remark-gfm` |
| Database | JSON file store (dev) |

---

## Setup

### Prerequisites
- Node.js 18+
- ffmpeg installed ([ffmpeg.org](https://ffmpeg.org/download.html))
- Google Gemini API key — **free** at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- Hindsight API key — at [hindsight.vectorize.io](https://hindsight.vectorize.io)

### Install ffmpeg

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### Run the App

```bash
# 1. Install dependencies
npm install

# 2. Configure environment variables
cp .env.example .env.local
# Edit .env.local — add your API keys

# 3. Start
npm run dev

# Open http://localhost:3000
```

---

## Environment Variables

```env
# Google Gemini — FREE at aistudio.google.com/apikey
GEMINI_API_KEY=AIzaSy...
GEMINI_MODEL=gemini-3.8-flash

# Hindsight persistent memory
HINDSIGHT_API_KEY=hsk-...
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io

# App
NEXT_PUBLIC_APP_URL=http://localhost:3000
DEFAULT_USER_ID=reelcraft-user-default
MAX_UPLOAD_SIZE_MB=200
```

> The **demo mode** at `/demo` works with zero API keys — it loads pre-seeded data instantly.

---

## Demo Instructions (60 seconds)

1. **Open** `http://localhost:3000` → see the hero page
2. **Click "Try the Demo"** → pre-analyzed food Reel loads instantly
3. **Browse the 23-section guide** — Skills, Equipment, Shot List, Lighting, Editing, Color, Export
4. **Click "Scene Timeline" tab** → interactive scene visualization
5. **Go to Memory page** → see stored preferences: iPhone, CapCut, Beginner, fast-paced
6. **Open Chat** → type *"I also like warm color grades"* → watch Hindsight store it live
7. **Upload a new Reel** → banner shows *"🧠 Personalizing this guide using your creative memory"*

**Key message:**
> *ReelCraft AI doesn't just analyze videos. It learns how YOU create content and turns any reference Reel into a personalized production blueprint.*

---

## Example: How Memory Changes Everything

**Without Hindsight:**
```
User: Analyze this food Reel
AI:   [Generic guide — assumes professional equipment]
```

**With Hindsight:**
```
User: Analyze this food Reel
AI:   🧠 Personalizing using your creative memory…
      Recalled: iPhone filming · CapCut editing · Beginner · Fast-paced

      Shot list adapted for iPhone camera angles
      Editing steps written for CapCut specifically
      Explanations kept beginner-friendly
      Target duration: 20-30 seconds
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      BROWSER                             │
│  Home  │  Analyze  │  Guide  │  Memory  │  Chat         │
└────────────────────┬────────────────────────────────────┘
                     │ API calls
┌────────────────────▼────────────────────────────────────┐
│               NEXT.JS API ROUTES                         │
│  /analyze  /projects/[id]  /memory  /chat  /demo        │
└──────┬──────────────┬──────────────┬────────────────────┘
       │              │              │
       ▼              ▼              ▼
  VIDEO ANALYSIS   PROJECT DB   HINDSIGHT MEMORY
  (ffmpeg+Gemini)  (JSON file)  (Vectorize Cloud)
       │                              │
       └──────────────────────────────┘
              Gemini 3.8 Flash
         reads creator profile from
         Hindsight before generating
              personalized guide
```

---

## Limitations

- **URL analysis** — social platforms block direct downloads; URL mode uses metadata + context only. Upload the file for best results.
- **JSON database** — suitable for dev/demo; swap for PostgreSQL in production
- **Single user** — uses one default user ID locally; production would use real auth
- **503 retries** — Gemini free tier gets overloaded at peak times; the app auto-retries up to 4 times with backoff

---

## Future Improvements

- [ ] Streaming analysis with real-time progress
- [ ] Multi-user authentication
- [ ] PostgreSQL + Prisma
- [ ] PDF export of the recreation guide
- [ ] Community Reel blueprint library
- [ ] Mobile app (React Native)
- [ ] Direct social media OAuth

---

<div align="center">

Built for the **Hindsight Hackathon** · Powered by [Hindsight](https://hindsight.vectorize.io) persistent agent memory

</div>
