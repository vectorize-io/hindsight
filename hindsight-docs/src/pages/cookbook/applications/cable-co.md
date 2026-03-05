---
sidebar_position: 1
---

# CableConnect — AI Customer Service Copilot Demo


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/cable-co)
:::


An AI copilot that assists a customer service representative (CSR) by suggesting responses and actions for simulated customer scenarios. The CSR approves or rejects each suggestion with feedback. The copilot learns from corrections via [Hindsight](https://hindsight.vectorize.io) and stops repeating mistakes.

## Prerequisites

- Python 3.11+
- Node.js 18+
- An OpenAI API key (for GPT-4o)
- A Hindsight API key ([sign up](https://hindsight.vectorize.io))

## Quick Start

### 1. Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your credentials
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key
HINDSIGHT_API_KEY=hsk_your-hindsight-key
HINDSIGHT_API_URL=https://api.hindsight.vectorize.io
HINDSIGHT_BANK_NAME=cable-connect-demo
EOF

# Start the backend (port 8002)
./run.sh
```

### 2. Frontend

In a second terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server (port 5173)
npm run dev
```

Open http://localhost:5173 in your browser.

## Running the Demo

1. Click **Next Customer** to load the first scenario
2. The AI copilot will analyze the customer's issue and suggest a response
3. Review the suggestion in the right panel:
   - **Send to Customer** — approves the response, sends it to the customer chat
   - **Approve** — executes a system action (credit, dispatch, etc.)
   - **Reject** — type feedback explaining what was wrong
4. The copilot adjusts based on your feedback and tries again
5. Continue until the customer is satisfied, then approve the resolve action
6. Click **Next Customer** for the next scenario

### What to Watch For

The 8 scenarios include 3 **learning pairs** — the first scenario teaches the agent a rule, the second tests whether it remembers:

| Pair | Scenarios | What the Agent Learns |
|------|-----------|----------------------|
| A | 2 then 4 | Credit adjustments are capped at $25 |
| B | 3 then 8 | Run remote diagnostics before scheduling a dispatch |
| C | 5 then 6 | Retention offers require 24+ months of tenure |

With **Memory On** (the default), the copilot recalls past CSR feedback before each new customer. By the test scenario, it should handle the situation correctly without being corrected.

Toggle **Memory Off** to see how the agent behaves without learning — it will make the same mistakes every time.

### Controls

- **Mode** dropdown — Switch between Memory On and Memory Off
- **Reset** — Deletes all stored memories and starts the scenario queue over
- **Refresh Models** — Manually triggers a refresh of the agent's mental models

## Configuration

All configuration is via environment variables in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `HINDSIGHT_API_KEY` | — | Your Hindsight API key (required) |
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | Hindsight API endpoint |
| `HINDSIGHT_BANK_NAME` | `cable-connect-demo` | Name of the memory bank |
| `LLM_MODEL` | `openai/gpt-4o` | LLM model (via LiteLLM format) |
| `BACKEND_PORT` | `8002` | Backend server port |

## Project Structure

```
cable-co/
├── backend/
│   ├── run.sh                  # Start script (loads .env, runs uvicorn)
│   ├── requirements.txt
│   ├── telecom_data.py         # Accounts, plans, billing, outages, scenarios
│   ├── agent_tools.py          # 19 tools + business rule hints
│   └── app/
│       ├── main.py             # FastAPI + WebSocket
│       ├── config.py
│       └── services/
│           ├── agent_service.py    # Copilot loop with CSR approval gate
│           └── memory_service.py   # Hindsight retain/recall/mental models
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── stores/sessionStore.ts
│       ├── hooks/useWebSocket.ts
│       └── components/
│           ├── ControlBar.tsx
│           ├── CustomerChat.tsx
│           ├── CopilotChat.tsx
│           ├── KnowledgePanel.tsx
│           └── MentalModelsPanel.tsx
└── article.md                  # Detailed writeup of how agent learning works
```

## How It Works

See article.md for a detailed explanation of the agent learning architecture, including how Hindsight transforms CSR feedback into observations and mental models that improve the copilot's behavior over time.
