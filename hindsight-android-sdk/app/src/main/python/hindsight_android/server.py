"""Stripped-down Hindsight API server for on-device Android use.

Uses SQLite backend instead of PostgreSQL. Exposes retain/recall via FastAPI
on localhost. Started from Kotlin via Chaquopy.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Models ────────────────────────────────────────────────────────


class MemoryItem(BaseModel):
    content: str
    timestamp: str | None = None
    context: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] | None = None


class RetainRequest(BaseModel):
    items: list[MemoryItem]


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class RetainResponse(BaseModel):
    success: bool
    bank_id: str
    items_count: int
    facts_extracted: int = 0
    usage: TokenUsage | None = None


class Budget(str, Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class RecallRequest(BaseModel):
    query: str
    types: list[str] | None = None
    budget: Budget = Budget.MID
    max_results: int = 20


class RecallResult(BaseModel):
    id: str
    text: str
    type: str | None = None
    entities: list[str] | None = None
    context: str | None = None
    occurred_start: str | None = None
    occurred_end: str | None = None
    metadata: dict[str, str] | None = None
    tags: list[str] | None = None
    score: float | None = None


class RecallResponse(BaseModel):
    results: list[RecallResult]


class CreateBankRequest(BaseModel):
    name: str
    mission: str = ""


class BankResponse(BaseModel):
    bank_id: str
    name: str
    mission: str


# ── Fact Extraction ───────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a memory extraction system. Extract factual information from the given text.

Rules:
- Extract ONLY meaningful facts (personal info, preferences, events, plans, expertise)
- Skip greetings, filler, process chatter
- Each fact should be a single, self-contained statement
- Classify as "world" (general knowledge) or "experience" (personal experience)
- Extract entity names mentioned
- Convert relative dates to absolute when possible

Output JSON:
{"facts": [{"text": "...", "fact_type": "world|experience", "entities": ["..."], "occurred_start": null, "occurred_end": null}]}

If no meaningful facts, return: {"facts": []}"""


async def extract_facts(
    text: str, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None
) -> tuple[list[dict], TokenUsage]:
    """Extract facts from text using LLM."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    usage = TokenUsage(
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        total_tokens=response.usage.total_tokens if response.usage else 0,
    )

    content = response.choices[0].message.content or '{"facts": []}'
    parsed = json.loads(content)
    return parsed.get("facts", []), usage


async def generate_embeddings(
    texts: list[str],
    api_key: str,
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=texts)
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [d.embedding for d in sorted_data]


# ── Server ────────────────────────────────────────────────────────

_server_thread: threading.Thread | None = None
_server_instance = None
_backend = None


def create_app(db_path: str) -> Any:
    """Create the FastAPI app with SQLite backend."""
    from fastapi import FastAPI, HTTPException

    from .sqlite_backend import Fact, SQLiteMemoryBackend

    global _backend
    _backend = SQLiteMemoryBackend(db_path)
    _backend.initialize()

    app = FastAPI(title="Hindsight Android SDK", version="0.1.0")

    def _get_config() -> dict:
        return {
            "api_key": os.environ.get("HINDSIGHT_API_LLM_API_KEY", ""),
            "model": os.environ.get("HINDSIGHT_API_LLM_MODEL", "gpt-4o-mini"),
            "base_url": os.environ.get("HINDSIGHT_API_LLM_BASE_URL"),
            "embeddings_model": os.environ.get(
                "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL", "text-embedding-3-small"
            ),
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "sqlite"}

    @app.post("/v1/default/banks", response_model=BankResponse)
    async def create_bank(req: CreateBankRequest):
        bank_id = str(uuid.uuid4())
        _backend.create_bank(bank_id, req.name, req.mission)
        return BankResponse(bank_id=bank_id, name=req.name, mission=req.mission)

    @app.post(
        "/v1/default/banks/{bank_id}/memories/retain", response_model=RetainResponse
    )
    async def retain(bank_id: str, req: RetainRequest):
        cfg = _get_config()
        if not cfg["api_key"]:
            raise HTTPException(status_code=400, detail="LLM API key not configured")

        # Ensure bank exists
        if not _backend.get_bank(bank_id):
            _backend.create_bank(bank_id, bank_id)

        total_facts = 0
        total_usage = TokenUsage()

        for item in req.items:
            # Extract facts via LLM
            raw_facts, usage = await extract_facts(
                item.content, cfg["api_key"], cfg["model"], cfg["base_url"]
            )
            total_usage.input_tokens += usage.input_tokens
            total_usage.output_tokens += usage.output_tokens
            total_usage.total_tokens += usage.total_tokens

            if not raw_facts:
                continue

            # Generate embeddings for all fact texts
            fact_texts = [f.get("text", "") for f in raw_facts]
            embeddings = await generate_embeddings(
                fact_texts, cfg["api_key"], cfg["embeddings_model"], cfg["base_url"]
            )

            # Store facts
            facts = []
            for i, raw in enumerate(raw_facts):
                facts.append(
                    Fact(
                        id=str(uuid.uuid4()),
                        bank_id=bank_id,
                        text=raw.get("text", ""),
                        fact_type=raw.get("fact_type", "world"),
                        embedding=embeddings[i] if i < len(embeddings) else None,
                        entities=raw.get("entities", []),
                        context=item.context,
                        occurred_start=raw.get("occurred_start"),
                        occurred_end=raw.get("occurred_end"),
                        metadata=item.metadata,
                        tags=item.tags or [],
                        created_at=time.time(),
                    )
                )

            _backend.store_facts(facts)
            total_facts += len(facts)

        return RetainResponse(
            success=True,
            bank_id=bank_id,
            items_count=len(req.items),
            facts_extracted=total_facts,
            usage=total_usage,
        )

    @app.post(
        "/v1/default/banks/{bank_id}/memories/recall", response_model=RecallResponse
    )
    async def recall(bank_id: str, req: RecallRequest):
        cfg = _get_config()
        if not cfg["api_key"]:
            raise HTTPException(status_code=400, detail="LLM API key not configured")

        # Generate query embedding
        query_embeddings = await generate_embeddings(
            [req.query], cfg["api_key"], cfg["embeddings_model"], cfg["base_url"]
        )

        # Search
        facts = _backend.search(
            bank_id=bank_id,
            query_embedding=query_embeddings[0],
            limit=req.max_results,
            fact_types=req.types,
        )

        results = []
        for fact in facts:
            score = None
            if fact.embedding and query_embeddings:
                from .sqlite_backend import cosine_similarity

                score = cosine_similarity(fact.embedding, query_embeddings[0])
            results.append(
                RecallResult(
                    id=fact.id,
                    text=fact.text,
                    type=fact.fact_type,
                    entities=fact.entities,
                    context=fact.context,
                    occurred_start=fact.occurred_start,
                    occurred_end=fact.occurred_end,
                    metadata=fact.metadata,
                    tags=fact.tags,
                    score=score,
                )
            )

        return RecallResponse(results=results)

    return app


def start_server(
    db_path: str, api_key: str, model: str = "gpt-4o-mini", port: int = 8741
) -> None:
    """Start the Hindsight server in a background thread.

    Called from Kotlin via Chaquopy.
    """
    import uvicorn

    global _server_thread, _server_instance

    # Set config via env vars (read by create_app)
    os.environ["HINDSIGHT_API_LLM_API_KEY"] = api_key
    os.environ["HINDSIGHT_API_LLM_MODEL"] = model

    app = create_app(db_path)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    _server_instance = uvicorn.Server(config)

    def _run():
        _server_instance.run()

    _server_thread = threading.Thread(target=_run, daemon=True)
    _server_thread.start()


def stop_server() -> None:
    """Stop the Hindsight server."""
    global _server_instance, _backend
    if _server_instance:
        _server_instance.should_exit = True
    if _backend:
        _backend.close()
        _backend = None


def is_running() -> bool:
    """Check if the server is running."""
    return _server_instance is not None and _server_instance.started
