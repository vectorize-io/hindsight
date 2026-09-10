"""An OpenAI/Cohere-compatible stub the Hindsight server talks to over real HTTP.

The whole point of standing this up as a *server* rather than monkeypatching a
provider class in-process is that nothing in ``hindsight_api`` has to change or
even know it exists. Three environment variables point the real server here:

    HINDSIGHT_API_LLM_BASE_URL                  -> /v1/chat/completions
    HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL    -> /v1/embeddings
    HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL -> /rerank

Which means the tests exercise the production provider code for real — the
OpenAI client, the JSON-repair path, the retry and rate-limit handling, the
structured-output plumbing. An in-process fake replaces all of that with
nothing, and a surprising share of this project's provider bugs live precisely
there.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .lexical import EMBEDDING_DIMENSION
from .rulebook import ChatRequest, ReceivedWebhook, Stubs
from .validation import RequestRejected, validate_chat, validate_embeddings, validate_rerank

logger = logging.getLogger(__name__)


def create_stub_app(stubs: Stubs) -> FastAPI:
    app = FastAPI(title="Hindsight system-test provider stub")

    @app.exception_handler(RequestRejected)
    async def _rejected(_request: Request, exc: RequestRejected) -> JSONResponse:
        stubs.rejected_requests.append(str(exc))
        return _provider_error(str(exc), code="invalid_request_error")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await request.json()
        validate_chat(body)

        chat_request = ChatRequest(
            model=body["model"],
            messages=body["messages"],
            tools=tuple(tool["function"]["name"] for tool in body.get("tools") or []),
        )

        reply = stubs.llm.resolve(chat_request)
        if reply is None:
            unmatched = stubs.llm.unmatched[-1]
            logger.error("No stub rule matched: %s", unmatched.suggestion())
            # 400, not 500: the server must not burn its retry budget re-asking a
            # question no rule will ever answer. A non-retryable status surfaces
            # the gap on the first call.
            return _provider_error(
                f"No stub rule matched this call. Add one:\n    {unmatched.suggestion()}",
                code="no_stub_rule",
            )

        return JSONResponse(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": reply.message, "finish_reason": reply.finish_reason, "logprobs": None}
                ],
                "usage": _usage(chat_request.all_text, str(reply.message.get("content") or "")),
            }
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        validate_embeddings(body)

        raw_input = body["input"]
        inputs = [raw_input] if isinstance(raw_input, str) else raw_input

        dimensions = body.get("dimensions", EMBEDDING_DIMENSION)
        if dimensions != EMBEDDING_DIMENSION:
            raise RequestRejected(
                f"stub embeddings only serve {EMBEDDING_DIMENSION}-dim vectors (the width of the "
                f"server's vector columns); asked for {dimensions}"
            )

        return JSONResponse(
            {
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": index, "embedding": stubs.embeddings.embed(text)}
                    for index, text in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": sum(len(text.split()) for text in inputs), "total_tokens": 0},
            }
        )

    @app.post("/webhook")
    async def webhook(request: Request) -> JSONResponse:
        """Stand in for a customer's webhook endpoint.

        Records the headers as well as the body: the signature a receiver is
        expected to verify travels in a header, and a delivery that arrives
        unsigned is indistinguishable from one anybody could forge.
        """
        stubs.webhooks.append(
            ReceivedWebhook(headers={k.lower(): v for k, v in request.headers.items()}, body=await request.json())
        )
        return JSONResponse({"received": True})

    @app.post("/rerank")
    async def rerank(request: Request) -> JSONResponse:
        body = await request.json()
        validate_rerank(body)

        query = body["query"]
        results = [
            {"index": index, "relevance_score": stubs.rerank.score(query, document)}
            for index, document in enumerate(body["documents"])
        ]
        results.sort(key=lambda item: item["relevance_score"], reverse=True)
        return JSONResponse({"id": "rerank-stub", "results": results})

    return app


def _usage(prompt: str, completion: str) -> dict[str, int]:
    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = max(1, len(completion.split()))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _provider_error(message: str, *, code: str) -> JSONResponse:
    """The error envelope OpenAI-compatible clients expect, so the SDK parses it."""
    return JSONResponse(
        status_code=400,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code, "param": None}},
    )
