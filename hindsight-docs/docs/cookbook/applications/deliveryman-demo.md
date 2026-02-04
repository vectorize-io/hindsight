---
sidebar_position: 2
---

# Deliveryman Demo


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/deliveryman-demo)
:::


A delivery agent simulation that demonstrates Hindsight's long-term memory capabilities. An AI agent navigates a multi-building office complex to deliver packages, learning employee locations and optimal paths over time through mental models.

## Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Setup (Fresh Environment)

### 1. Clone Repositories

```bash
# Clone Hindsight (memory engine)
git clone https://github.com/anthropics/hindsight.git

# Clone the cookbook (contains this demo)
git clone https://github.com/anthropics/hindsight-cookbook.git
```

### 2. Start Hindsight API

```bash
cd hindsight
cp .env.example .env
```

Edit `.env` with your LLM configuration:

```bash
HINDSIGHT_API_LLM_PROVIDER=groq
HINDSIGHT_API_LLM_API_KEY=<your-groq-api-key>
HINDSIGHT_API_LLM_MODEL=openai/gpt-oss-120b
HINDSIGHT_API_HOST=0.0.0.0
HINDSIGHT_API_PORT=8888
HINDSIGHT_API_ENABLE_OBSERVATIONS=true

# Retain extraction settings (improves employee/location extraction)
HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom
HINDSIGHT_API_RETAIN_CUSTOM_INSTRUCTIONS="Delivery agent. Remember employee locations, building layout, and optimal paths."

# Embedded database storage
PG0_DATA_DIR=/tmp/hindsight-data
```

Start the API:

```bash
./scripts/dev/start-api.sh
# Runs on http://localhost:8888
```

### 3. Start Hindsight Control Plane (Optional)

The control plane provides a web UI for inspecting memory banks, facts, and mental models.

```bash
cd hindsight
./scripts/dev/start-control-plane.sh
# Runs on a dynamic port (check terminal output)
```

### 4. Start Demo Backend

```bash
cd hindsight-cookbook/deliveryman-demo/backend

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```bash
OPENAI_API_KEY=<your-openai-api-key>
GROQ_API_KEY=<your-groq-api-key>
HINDSIGHT_API_URL=http://localhost:8888
LLM_MODEL=openai/gpt-4o
```

Start the backend:

```bash
./run.sh
# Or manually:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws wsproto --reload
```

**Note:** The `--ws wsproto` flag is required for WebSocket support. Without it, connections will fail with error 1006.

### 5. Start Demo Frontend

```bash
cd hindsight-cookbook/deliveryman-demo/frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### 6. Open the Demo

Navigate to http://localhost:5173 in your browser.

## How It Works

1. The agent receives a delivery task (e.g., "Deliver Package #3954 to Victor Huang")
2. It navigates a multi-building complex with floors, elevators, and sky bridges
3. Along the way it encounters employees and learns their locations
4. After each delivery, the conversation is sent to Hindsight via the **retain** API
5. Hindsight extracts facts (employee locations, building layout) and builds **mental models**
6. On subsequent deliveries, the agent queries Hindsight to recall what it learned

## Architecture

```
Browser (5173) → Frontend (React + Phaser)
                    ↓ WebSocket
                 Backend (8000) → FastAPI + Delivery Agent
                    ↓ HTTP
                 Hindsight API (8888) → Memory Engine + PostgreSQL
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| WebSocket error 1006 | Restart backend with `--ws wsproto` flag |
| Mental models missing employees | Check `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom` is set |
| Hindsight connection refused | Verify Hindsight API is running on port 8888 |
| Frontend shows "Disconnected" | Check backend is running on port 8000 |
