"""Tool calling through a response schema instead of ``tool_choice`` (HINDSIGHT_API_LLM_TOOLS_VIA_SCHEMA).

The model never makes a native tool call. The tools stay declared, so it reads their
names, descriptions and parameters as usual, but ``tool_choice`` is "none" and every
turn's reply is held to a JSON schema: a forced (named) turn to that tool's arguments,
a free ("required") turn to "one of the tools, with its arguments". The reply is then
handed back as the call, so callers see ordinary tool calls.

Why: a server can parse tool calls without enforcing ``tool_choice`` -- vLLM's gemma4
parser skips the grammar for "required" and named choices -- and a small model then
answers a forced search with the stop signal (75% of reflect's forced steps on Gemma 4)
or a forced ``done`` with another tool. A response schema is what such servers do
enforce. A turn is also asked for in words: a schema imposed on a reply the
conversation never asked for gave degenerate arguments.

Shared by the providers that implement it (OpenAI-compatible, Codex): the schema and
the reply are provider-neutral, only the transport differs.
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
        arguments = reply
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
