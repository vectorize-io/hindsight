"""Tool calling, for every provider: a turn's call is the reply to a response schema.

No provider is ever sent ``tools`` or ``tool_choice``. The functions are described in the
system prompt, the turn asks for its call in words, and the reply is held to a JSON schema:
a forced (named) turn to that function's arguments, a free ("required") turn to "one of the
functions, with its arguments". The reply is handed back as an ordinary ``LLMToolCall``, so
callers see tool calls as before. The conversation's earlier calls and their results are
replayed as plain assistant and user turns.

Why one way, and this one: a response schema is the one thing every provider Hindsight
supports enforces, while native ``tool_choice`` is not — vLLM's gemma4 parser parses tool
calls but skips the grammar for "required" and named choices, and a small model then
answered a forced search with the stop signal (75% of reflect's forced steps on Gemma 4) or a
forced ``done`` with another tool. Small models had a long history of tool-calling quirks
per provider; a model that can follow a schema is now all reflect asks for. A schema imposed
on a reply the conversation never asked for gave degenerate arguments, hence the turn request.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any

from ..llm_interface import LLMToolChoice, LLMToolChoiceMode
from ..response_models import LLMToolCall


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    return tool.get("function", tool)


def _closed(schema: dict[str, Any]) -> dict[str, Any]:
    """The strict subset every schema-enforcing API accepts: each object closed, each property
    required, a property the tool left optional made nullable (and dropped from the reply).

    No string bounds are added: under xgrammar a bounded string admits no escape sequence,
    so a markdown page could not hold a single line break.
    """
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and isinstance(node.get("properties"), dict):
                required = set(node.get("required") or [])
                for name, prop in node["properties"].items():
                    if name not in required and isinstance(prop, dict) and isinstance(prop.get("type"), str):
                        prop["type"] = [prop["type"], "null"]
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return schema


#: Free-text notes about a call, moved after its real arguments.
_TRAILING_ARGUMENTS = ("reason",)


def _arguments_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """The tool's arguments as a closed schema, with ``reason`` written last.

    A held reply is written in property order, and a search's ``reason`` came first:
    after explaining itself, Gemma 4 wrote the query as a keyword list that never closed
    ("2FA mandatory MFA mandatory 2FA mandatory ...") in 7 of 8 replies to a first
    forced search, and 0 of 8 with the query first. The note is only for debugging.
    """
    parameters = _function(tool).get("parameters") or {"type": "object", "properties": {}}
    schema = _closed({"type": "object", **parameters})
    properties = schema.get("properties") or {}
    if any(name in properties for name in _TRAILING_ARGUMENTS):
        ordered = {k: v for k, v in properties.items() if k not in _TRAILING_ARGUMENTS}
        ordered.update({k: properties[k] for k in _TRAILING_ARGUMENTS if k in properties})
        schema["properties"] = ordered
        schema["required"] = list(ordered)
    return schema


@dataclass(frozen=True)
class ResponseSchema:
    """The JSON schema a turn's reply is held to, and the name it is sent under."""

    name: str
    schema: dict[str, Any]


def response_schema(tools: list[dict[str, Any]], tool_choice: LLMToolChoice) -> ResponseSchema:
    """The schema the reply is held to for this turn."""
    if tool_choice.mode is LLMToolChoiceMode.NAMED:
        tool = forced_tool(tools, tool_choice)
        return ResponseSchema(name=_function(tool)["name"], schema=_arguments_schema(tool))
    options = [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [_function(t)["name"]]},
                "arguments": _arguments_schema(t),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for t in tools
    ]
    return ResponseSchema(
        name="next_call",
        schema={
            "type": "object",
            "properties": {"call": {"anyOf": options}},
            "required": ["call"],
            "additionalProperties": False,
        },
    )


def forced_tool(tools: list[dict[str, Any]], tool_choice: LLMToolChoice) -> dict[str, Any]:
    name = tool_choice.selected_function_name
    matches = [t for t in tools if _function(t).get("name") == name]
    if len(matches) != 1:
        raise ValueError(
            f"Named tool_choice must reference exactly one declared tool; found {len(matches)} for {name!r}"
        )
    return matches[0]


def turn_request(tools: list[dict[str, Any]], tool_choice: LLMToolChoice) -> str:
    """The user turn that asks for this turn's call in words."""
    if tool_choice.mode is LLMToolChoiceMode.NAMED:
        function = _function(forced_tool(tools, tool_choice))
        purpose = (function.get("description") or "").split(". ")[0].strip().rstrip(".")
        return (
            f"Call `{function['name']}` now{f' ({purpose})' if purpose else ''}. "
            "Reply with only its arguments, as one JSON object."
        )
    return (
        "Decide your next step: call exactly ONE of the declared functions. Reply with one JSON object, "
        '{"call": {"name": <function name>, "arguments": {...}}}.'
    )


def calls_from_reply(content: str | None, tools: list[dict[str, Any]], tool_choice: LLMToolChoice) -> list[LLMToolCall]:
    """The call a schema-enforced reply stands for.

    Arguments that do not parse (a reply cut off at the token cap) ride as ``_raw``, as a
    native call's do, for the caller to judge. Nulls stand for arguments the tool left
    optional and the model did not set, so they are dropped.
    """
    try:
        reply = json.loads(content or "")
    except json.JSONDecodeError:
        reply = None
    if tool_choice.mode is LLMToolChoiceMode.NAMED:
        name = _function(forced_tool(tools, tool_choice))["name"]
        arguments = _unwrapped(reply, name)
    else:
        call = reply.get("call") if isinstance(reply, dict) else None
        name = call.get("name") if isinstance(call, dict) else None
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if not name:
            return []  # no call to hand back: the caller treats it as a turn without one
    if not isinstance(arguments, dict):
        arguments = {"_raw": content or ""}
    else:
        arguments = {k: v for k, v in arguments.items() if v is not None}
    return [LLMToolCall(id=f"call_{name}_{uuid.uuid4().hex[:12]}", name=name, arguments=arguments)]


def _unwrapped(reply: Any, name: str) -> Any:
    """A forced turn's arguments, from the two wrappers a model put around them.

    Where a provider does not enforce ``additionalProperties: false`` (Gemini's schema
    support ignores it), a model answered a forced ``done`` as ``{"done": {...}}``, and
    the page arrived with no ``markdown`` field; another wrote the free-turn shape
    ``{"call": {"name": ..., "arguments": {...}}}``. Either one names this very function,
    so it has one reading.
    """
    if not isinstance(reply, dict) or len(reply) != 1:
        return reply
    ((key, value),) = reply.items()
    if key == name and isinstance(value, dict):
        return value
    if (
        key == "call"
        and isinstance(value, dict)
        and value.get("name") == name
        and isinstance(value.get("arguments"), dict)
    ):
        return value["arguments"]
    return reply


class SchemaFormat:
    """A raw JSON schema passed where providers take a Pydantic response model.

    Providers read a response model through ``model_json_schema`` (via
    ``structured_output``), ``__name__`` and ``model_validate``; this answers all three, so
    every provider's structured-output path carries the schema unchanged. Built per turn by
    ``schema_format``.
    """

    schema: dict[str, Any]

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return copy.deepcopy(cls.schema)

    @classmethod
    def model_validate(cls, value: Any) -> Any:
        return value


def schema_format(held: ResponseSchema) -> type[SchemaFormat]:
    return type(held.name, (SchemaFormat,), {"schema": held.schema})


def _declarations(tools: list[dict[str, Any]]) -> str:
    functions = [{k: v for k, v in _function(t).items() if k in ("name", "description", "parameters")} for t in tools]
    return "## Functions\n\nYou act by calling these functions:\n\n" + json.dumps(
        functions, ensure_ascii=False, indent=1
    )


def as_text_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as plain role/content turns, with the functions declared in the system prompt.

    An earlier call is the assistant's reply as it was written (``{"call": {...}}``); its result
    is a user turn naming the function. Turns are not merged, so the prefix of a longer
    conversation renders to the prefix of its text.
    """
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function", call)
                names[call.get("id", "")] = function["name"]
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                calls.append(
                    json.dumps({"call": {"name": function["name"], "arguments": arguments}}, ensure_ascii=False)
                )
            text = "\n".join(calls)
            if message.get("content"):
                text = f"{message['content']}\n{text}"
            out.append({"role": "assistant", "content": text})
        elif role == "tool":
            name = message.get("name") or names.get(message.get("tool_call_id", ""), "the function")
            out.append({"role": "user", "content": f"Result of `{name}`:\n{message.get('content') or ''}"})
        else:
            out.append({k: v for k, v in message.items() if k in ("role", "content")})
    declarations = _declarations(tools)
    if out and out[0].get("role") == "system":
        out[0] = {"role": "system", "content": f"{out[0].get('content') or ''}\n\n{declarations}"}
    else:
        out.insert(0, {"role": "system", "content": declarations})
    return out
