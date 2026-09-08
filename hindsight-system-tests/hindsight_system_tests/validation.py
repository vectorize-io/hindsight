"""Reject requests a real provider would reject.

A fake you control drifts permissive, and a permissive fake hides exactly the
bugs that hurt most here: look at what has actually broken in production —
Bedrock rejecting ``response_format``, Azure 400ing on ``prompt_cache_key``,
Vertex 404ing a region. Every one of those is *the provider refusing our
request*, and a stub that cheerfully accepts anything would have shipped all of
them green.

So the stub validates. The allowlist below is the OpenAI chat-completions
surface; a field outside it means Hindsight is sending something real OpenAI
would not accept, and the stub says so with a 400 instead of answering.
"""

from __future__ import annotations

from typing import Any

# The OpenAI chat-completions request surface. Kept explicit rather than
# permissive: a new field appearing here should be a deliberate edit, made when
# someone confirms the real API accepts it.
CHAT_FIELDS = frozenset(
    {
        "model",
        "messages",
        "audio",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "modalities",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)

EMBEDDINGS_FIELDS = frozenset({"model", "input", "dimensions", "encoding_format", "user"})

RERANK_FIELDS = frozenset({"model", "query", "documents", "top_n", "return_documents", "max_chunks_per_doc"})


class RequestRejected(Exception):
    """The stub refused the request the way the real provider would."""


def validate_chat(body: dict[str, Any]) -> None:
    _reject_unknown(body, CHAT_FIELDS, "chat/completions")

    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestRejected("'model' must be a non-empty string")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestRejected("'messages' must be a non-empty array")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise RequestRejected(f"messages[{index}] must be an object with a 'role'")

    response_format = body.get("response_format")
    if response_format is not None:
        _validate_response_format(response_format)

    tools = body.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise RequestRejected("'tools' must be an array")
        for index, tool in enumerate(tools):
            if not isinstance(tool, dict) or tool.get("type") != "function" or "function" not in tool:
                raise RequestRejected(f"tools[{index}] must be {{'type': 'function', 'function': {{...}}}}")
            if not isinstance(tool["function"].get("name"), str):
                raise RequestRejected(f"tools[{index}].function.name must be a string")


def _validate_response_format(response_format: Any) -> None:
    if not isinstance(response_format, dict):
        raise RequestRejected("'response_format' must be an object")

    kind = response_format.get("type")
    if kind not in {"text", "json_object", "json_schema"}:
        raise RequestRejected(f"response_format.type {kind!r} is not one of text/json_object/json_schema")

    if kind != "json_schema":
        return

    schema_block = response_format.get("json_schema")
    if not isinstance(schema_block, dict):
        raise RequestRejected("response_format.json_schema must be an object")
    if not isinstance(schema_block.get("name"), str) or not schema_block["name"]:
        raise RequestRejected("response_format.json_schema.name must be a non-empty string")
    if not isinstance(schema_block.get("schema"), dict):
        raise RequestRejected("response_format.json_schema.schema must be an object")

    if schema_block.get("strict") is True:
        _validate_strict_schema(schema_block["schema"], path="schema")


def _validate_strict_schema(schema: dict[str, Any], *, path: str) -> None:
    """Enforce OpenAI's structured-output rules for ``strict: true``.

    These are the rules that produce a real 400 in production and are easy to
    violate by generating a schema from a Pydantic model: every object must set
    ``additionalProperties: false`` and list every property as required.
    """
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            raise RequestRejected(f"{path}: strict schemas must set additionalProperties=false")
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        missing = set(properties) - required
        if missing:
            raise RequestRejected(
                f"{path}: strict schemas must list every property as required; missing {sorted(missing)}"
            )
        for name, child in properties.items():
            if isinstance(child, dict):
                _validate_strict_schema(child, path=f"{path}.{name}")

    items = schema.get("items")
    if isinstance(items, dict):
        _validate_strict_schema(items, path=f"{path}[]")

    for keyword in ("anyOf", "oneOf", "allOf"):
        for index, child in enumerate(schema.get(keyword) or []):
            if isinstance(child, dict):
                _validate_strict_schema(child, path=f"{path}.{keyword}[{index}]")

    for name, child in (schema.get("$defs") or {}).items():
        if isinstance(child, dict):
            _validate_strict_schema(child, path=f"{path}.$defs.{name}")


def validate_embeddings(body: dict[str, Any]) -> None:
    _reject_unknown(body, EMBEDDINGS_FIELDS, "embeddings")

    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestRejected("'model' must be a non-empty string")

    value = body.get("input")
    if isinstance(value, str):
        inputs = [value]
    elif isinstance(value, list):
        inputs = value
    else:
        raise RequestRejected("'input' must be a string or an array of strings")

    if not inputs:
        raise RequestRejected("'input' must not be empty")
    for index, item in enumerate(inputs):
        if not isinstance(item, str):
            raise RequestRejected(f"input[{index}] must be a string")


def validate_rerank(body: dict[str, Any]) -> None:
    _reject_unknown(body, RERANK_FIELDS, "rerank")

    if not isinstance(body.get("query"), str) or not body["query"]:
        raise RequestRejected("'query' must be a non-empty string")

    documents = body.get("documents")
    if not isinstance(documents, list) or not documents:
        raise RequestRejected("'documents' must be a non-empty array")
    for index, document in enumerate(documents):
        if not isinstance(document, str):
            raise RequestRejected(f"documents[{index}] must be a string")


def _reject_unknown(body: dict[str, Any], allowed: frozenset[str], endpoint: str) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestRejected(
            f"{endpoint}: unrecognized request fields {unknown}. "
            "Either the real provider would reject these too, or the allowlist in "
            "hindsight_system_tests/validation.py needs a deliberate update."
        )
