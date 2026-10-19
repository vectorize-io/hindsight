"""An OpenAI/Cohere-compatible stub the Hindsight server talks to over real HTTP.

The whole point of standing this up as a *server* rather than monkeypatching a
provider class in-process is that nothing in ``hindsight_api`` has to change or
even know it exists. Three environment variables point the real server here:

    HINDSIGHT_API_LLM_BASE_URL                  -> /v1/chat/completions
    HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL    -> /v1/embeddings
    HINDSIGHT_API_RERANKER_SILICONFLOW_BASE_URL -> /rerank
    HINDSIGHT_API_RERANKER_TYPESAFE_BASE_URL     -> /v1/systemone

With ``HINDSIGHT_API_LLM_PROVIDER=anthropic`` the same base URL serves the
Anthropic Messages API (``/v1/messages``) and its Workload Identity Federation
token exchange (``/v1/oauth/token``), answered from the same rulebook.

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
from .rulebook import ChatRequest, ReceivedWebhook, Stubs, TokenExchange
from .validation import RequestRejected, validate_chat, validate_embeddings, validate_rerank, validate_systemone

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
            api_key=request.headers.get("authorization", "").removeprefix("Bearer ") or None,
            headers={k.lower(): v for k, v in request.headers.items()},
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

        # Parked after resolve, so the call is already recorded and its rule already
        # matched: from the story's side the step has happened and its operation is
        # running, only the answer is outstanding.
        held = stubs.llm.hold_for(chat_request)
        if held is not None:
            await held.park()

        if reply.error_status is not None:
            return _provider_error(
                f"stub provider error {reply.error_status}",
                code="stub_error",
                status=reply.error_status,
                retry_after_seconds=reply.retry_after_seconds,
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
                "usage": _usage(
                    chat_request.all_text,
                    str(reply.message.get("content") or ""),
                    visible_tokens=reply.visible_tokens,
                    reasoning_tokens=reply.reasoning_tokens,
                ),
            }
        )

    @app.post("/v1/oauth/token")
    async def oauth_token(request: Request) -> JSONResponse:
        """The Workload Identity Federation exchange: identity token in, access token out.

        The issued token is numbered, so a story can tell which exchange a later call's
        bearer came from. An hour's lifetime keeps the SDK from re-exchanging mid-story.
        """
        body = await request.json()
        exchange = TokenExchange(
            assertion=body["assertion"],
            federation_rule_id=body["federation_rule_id"],
            organization_id=body["organization_id"],
            service_account_id=body.get("service_account_id"),
            access_token=f"stub-wif-access-{len(stubs.token_exchanges) + 1}",
        )
        stubs.token_exchanges.append(exchange)
        return JSONResponse({"access_token": exchange.access_token, "token_type": "Bearer", "expires_in": 3600})

    @app.post("/v1/messages")
    async def messages(request: Request) -> JSONResponse:
        """The Anthropic Messages API, answered from the chat-completions rulebook.

        The request is translated into a ``ChatRequest`` (the separate ``system`` field
        becomes a leading system message, so step anchors in it still match), and the
        rule's text reply back into an Anthropic text block. Tool-use turns are not
        translated yet: no story drives reflect through this provider.
        """
        body = await request.json()
        system = body.get("system")
        prefix = [{"role": "system", "content": system}] if system else []
        authorization = request.headers.get("authorization", "")
        chat_request = ChatRequest(
            model=body["model"],
            messages=[*prefix, *body["messages"]],
            tools=tuple(tool["name"] for tool in body.get("tools") or []),
            api_key=authorization.removeprefix("Bearer ") or request.headers.get("x-api-key"),
            headers={k.lower(): v for k, v in request.headers.items()},
        )

        reply = stubs.llm.resolve(chat_request)
        if reply is None:
            unmatched = stubs.llm.unmatched[-1]
            logger.error("No stub rule matched: %s", unmatched.suggestion())
            return _anthropic_error(f"No stub rule matched this call. Add one:\n    {unmatched.suggestion()}", 400)
        if reply.error_status is not None:
            return _anthropic_error(f"stub provider error {reply.error_status}", reply.error_status)

        usage = _usage(chat_request.all_text, str(reply.message.get("content") or ""))
        return JSONResponse(
            {
                "id": "msg-stub",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": reply.message["content"]}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"]},
            }
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        body = await request.json()
        validate_embeddings(body)
        if stubs.embeddings.fail_with_status is not None:
            status = stubs.embeddings.fail_with_status
            return _provider_error(f"stub provider error {status}", code="stub_error", status=status)

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

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        """TypeSafe's typed-question endpoint.

        A ``choice`` whose options are the candidates is the ranking: every option
        gets a probability, and they sum to 1, so the stub scores each option's text
        lexically and normalises, except the ``impact`` choice, a delta refresh's gate,
        which is the rulebook's verdict on the memory as yes or no. A ``noul`` asks about one memory, and is answered
        with the rulebook's verdict for it, else its lexical relevance to the state. A
        ``score`` is the cut: its answer is a level, and the rulebook decides which
        one.
        """
        body = await request.json()
        validate_systemone(body)

        answers = {}
        state = body["state"]
        if isinstance(state, dict) and "new_fact" in state:
            # Consolidation triage: the verdict is scripted per fact (see TriageStub).
            fact_text = state["new_fact"]["text"]
            relation = stubs.triage.relation(fact_text)
            fits = stubs.triage.fits(fact_text)
            yes = {"covers_o0": relation == "restates"}
            for question_id, question in body["questions"].items():
                if question_id == "relation":
                    pick = relation
                elif question_id == "keep_as":
                    pick = "as_stated" if fits else "reworded"
                elif set(question["criteria"]) == {"yes", "no"}:
                    pick = "yes" if yes.get(question_id, False) else "no"
                else:
                    pick = next(iter(question["criteria"]))
                probabilities = {key: (0.99 if key == pick else 0.0) for key in question["criteria"]}
                answers[question_id] = {"type": "choice", "choice": pick, "probabilities": probabilities}
            return JSONResponse({"model": "stub", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})

        for question_id, question in body["questions"].items():
            if question["type"] == "noul":
                memory = question["instructions"]["memory"]
                stubs.rerank.states.append(body["state"])
                answers[question_id] = {"type": "noul", "noul": stubs.rerank.verdict(body["state"], memory)}
            elif question_id == "impact":
                # A delta refresh's gate: could this new memory change the page? The memory
                # is a line of the state, and the rulebook's verdict for it decides.
                memory = body["state"].split("New memory: ", 1)[1].split("\n", 1)[0]
                stubs.rerank.states.append(body["state"])
                keep = stubs.rerank.verdict(body["state"], memory) >= 0.5
                answers[question_id] = {
                    "type": "choice",
                    "choice": "yes" if keep else "no",
                    "probabilities": {"yes": 0.9, "no": 0.1} if keep else {"yes": 0.1, "no": 0.9},
                    "confidence": 0.9,
                }
            elif question["type"] == "choice":
                criteria = question["criteria"]
                scores = {
                    key: stubs.rerank.score(body["state"], text if isinstance(text, str) else str(text))
                    for key, text in criteria.items()
                }
                total = sum(scores.values()) or 1.0
                probabilities = {key: value / total for key, value in scores.items()}
                best = max(probabilities, key=lambda key: probabilities[key])
                answers[question_id] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": probabilities,
                    "confidence": probabilities[best],
                }
            else:
                answers[question_id] = {
                    "type": "score",
                    "score": float(stubs.rerank.cut_level),
                    "legend": dict(enumerate(question["criteria"])),
                    "confidence": 0.9,
                }
        return JSONResponse({"model": body.get("model", "stub"), "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}})

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


def _usage(
    prompt: str, completion: str, *, visible_tokens: int | None = None, reasoning_tokens: int = 0
) -> dict[str, int | dict[str, int]]:
    prompt_tokens = max(1, len(prompt.split()))
    completion_tokens = visible_tokens if visible_tokens is not None else max(1, len(completion.split()))
    usage: dict[str, int | dict[str, int]] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens + reasoning_tokens,
        "total_tokens": prompt_tokens + completion_tokens + reasoning_tokens,
    }
    if reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def _provider_error(
    message: str, *, code: str, status: int = 400, retry_after_seconds: int | None = None
) -> JSONResponse:
    """The error envelope OpenAI-compatible clients expect, so the SDK parses it."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error", "code": code, "param": None}},
        headers={"retry-after": str(retry_after_seconds)} if retry_after_seconds is not None else None,
    )


def _anthropic_error(message: str, status: int) -> JSONResponse:
    """The error envelope the Anthropic SDK parses."""
    return JSONResponse(
        status_code=status, content={"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    )
