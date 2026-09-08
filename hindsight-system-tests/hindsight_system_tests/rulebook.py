"""What the stub answers, and how a test says so.

Two rules shape everything here.

**Match on content, not on the request body.** The obvious way to build a fake
LLM is to key responses off the exact request — the wiremock default. That fails
here within a week: Hindsight's prompts change constantly, and body-matched
stubs would turn every prompt tweak into thirty red tests. So a rule matches on
what the call is *about*: the pipeline step it belongs to (see ``steps.py``,
which owns the one anchor phrase per step) and substrings of the prompt that the
test itself supplied, and therefore controls.

**A call nobody scripted is a test failure, not a default.** The existing
``MockLLM`` synthesizes plausible facts from the input when it doesn't recognize
a call, which is exactly why tests using it pass without proving anything. Here
an unmatched call returns HTTP 400 and records a ready-to-paste rule, so writing
a new system test is mechanical: run it, paste the suggested rule, run again.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ChatRequest:
    """One ``/v1/chat/completions`` call, in the terms a rule matches on."""

    model: str
    messages: list[dict[str, Any]]
    tools: tuple[str, ...] = ()
    """Names of the tools offered on this call, when it is a tool-using turn."""

    @property
    def user_text(self) -> str:
        return "\n".join(_text_of(m.get("content")) for m in self.messages if m.get("role") == "user")

    @property
    def all_text(self) -> str:
        return "\n".join(_text_of(m.get("content")) for m in self.messages)


def _text_of(content: Any) -> str:
    """The text of a message, whether it is a string or OpenAI content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


@dataclass(frozen=True)
class StubbedReply:
    """One assistant turn the stub will answer with.

    A named type rather than a ``(message, finish_reason)`` pair: the two travel
    together everywhere, and a bare tuple makes every call site re-document which
    half is which.
    """

    message: dict[str, Any]
    """The OpenAI assistant message, passed into the response verbatim."""

    finish_reason: str


@dataclass(frozen=True)
class ChatRule:
    contains: tuple[str, ...]
    respond: Callable[[ChatRequest], StubbedReply]
    requires_tools: bool = False
    """When set, match only a turn that offers at least one tool.

    The reflect loop's search turns and its answering turn share a system prompt,
    so a substring cannot tell them apart — but only the search turns carry
    tools. Without this, a rule meant for the ladder also swallows the turn that
    was supposed to write the answer.
    """

    tool: str | None = None
    """When set, match only a turn that offers this tool.

    The reflect loop sends the *same* system prompt on every turn and varies only
    the tools it offers, so the tool list is the one thing that distinguishes one
    step of the loop from the next. Prompt substrings cannot.
    """

    def matches(self, request: ChatRequest) -> bool:
        if self.requires_tools and not request.tools:
            return False
        if self.tool is not None and self.tool not in request.tools:
            return False
        return all(needle in request.all_text for needle in self.contains)


class RuleBuilder:
    """The ``when`` half of a rule; one of the ``returns*`` methods completes it."""

    def __init__(self, stub: LLMStub, *, contains: tuple[str, ...], tool: str | None = None) -> None:
        self._stub = stub
        self._contains = contains
        self._tool = tool

    def returns(self, payload: BaseModel) -> LLMStub:
        """Answer with ``payload`` serialized as the assistant message content.

        This is the structured-output path: the server asked for JSON matching a
        schema, so the content *is* a JSON document. Taking a model rather than a
        dict means a mistyped field fails here, in the test, instead of surfacing
        several layers away as "the LLM returned unusable facts".
        """
        body = payload.model_dump_json()
        return self._register(lambda _request: _assistant_message(body))

    def returns_text(self, text: str) -> LLMStub:
        return self._register(lambda _request: _assistant_message(text))

    def answers_with(self, build: Callable[[ChatRequest], BaseModel]) -> LLMStub:
        """Compute the reply from the request that triggered it.

        Needed where the answer must contain identifiers the server minted at run
        time and the test cannot know in advance — consolidation is the case:
        an observation has to cite the `source_fact_ids` of facts created moments
        earlier, which arrive in the prompt.

        Still deterministic: the same prompt yields the same reply. Prefer
        ``returns`` wherever a literal payload will do, since a rule that computes
        its answer can hide a wrong one.
        """
        return self._register(lambda request: _assistant_message(build(request).model_dump_json()))

    def returns_tool_call(self, tool_name: str, **arguments: Any) -> LLMStub:
        """Answer by calling one named tool with specific arguments.

        Needed where the arguments matter — `done` carries the reflect answer in
        its own argument, so calling it with the wrong shape ends the loop with
        "the done tool returned no answer".
        """

        def respond(_request: ChatRequest) -> StubbedReply:
            return _tool_call(tool_name, arguments)

        return self._register(respond, requires_tools=True)

    def calls_the_offered_tool(self, **arguments: Any) -> LLMStub:
        """Answer a forced-tool turn by calling whichever tool it was offered.

        The reflect prelude walks a fixed ladder of searches, offering one tool per
        turn and refusing to advance until it is called. A story about what reflect
        *concludes* does not care which rung it is on, only that the ladder is
        climbed — so this drives every search turn with one rule instead of one
        rule per tool, and keeps working when a rung is added or renamed.
        """

        def respond(request: ChatRequest) -> StubbedReply:
            return _tool_call(request.tools[0], arguments)

        return self._register(respond, requires_tools=True)

    def _register(self, respond: Callable[[ChatRequest], StubbedReply], *, requires_tools: bool = False) -> LLMStub:
        self._stub.add_rule(
            ChatRule(contains=self._contains, respond=respond, tool=self._tool, requires_tools=requires_tools)
        )
        return self._stub


def _tool_call(tool_name: str, arguments: dict[str, Any]) -> StubbedReply:
    return StubbedReply(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{tool_name}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        finish_reason="tool_calls",
    )


def _assistant_message(content: str) -> StubbedReply:
    return StubbedReply(message={"role": "assistant", "content": content}, finish_reason="stop")


@dataclass(frozen=True)
class UnmatchedCall:
    """A call no rule answered, kept so the fixture can fail the test with it."""

    tools: tuple[str, ...]
    """Tools offered on the turn, which is what distinguishes one reflect step
    from the next when the prompt is identical."""

    prompt: str
    """The whole conversation, system message included.

    Not just the user turn: for several steps — a mental-model refresh sends the
    bare source query as its user message — the only thing identifying the caller
    is in the system prompt, so a report that omitted it would name no anchor and
    show nothing useful.
    """

    def suggestion(self) -> str:
        """A rule the author can paste into the test, near enough to be a starting point."""
        from .steps import STEP_ANCHORS

        step = next((name for name, anchor in STEP_ANCHORS.items() if anchor in self.prompt), None)
        offered = f', tool="{self.tools[0]}"' if self.tools else ""
        if step:
            return f'llm.on_step("{step}"{offered}, contains=[...]).returns(...)'
        return (
            "llm.on_chat(contains=[...]).returns(...)  "
            "# no known step anchor matched — add one to steps.py. Prompt: "
            f"{self.prompt[:600]!r}"
        )


class LLMStub:
    """The rulebook for ``/v1/chat/completions``."""

    def __init__(self) -> None:
        self._rules: list[ChatRule] = []
        self.unmatched: list[UnmatchedCall] = []
        self.calls: list[ChatRequest] = []
        """Every call the stub answered, in order.

        For the stories whose subject is what the server *sent*: that a directive
        reached the reflect prompt, that a disposition changed it. Prompt assembly
        is deterministic even though the model's reading of it is not, so it can be
        asserted directly rather than judged.
        """
        self._install_builtins()

    def _install_builtins(self) -> None:
        """Answer the calls that are startup plumbing rather than test subject matter.

        ``verify_connection`` pings the provider before the server reports healthy.
        Making every test declare a rule for it would be noise with no signal.
        """
        self.on_step("connection_probe").returns_text("ok")

    def on_step(self, step: str, *, contains: str | list[str] | None = None, tool: str | None = None) -> RuleBuilder:
        """Declare what the model says at a named pipeline step.

        The preferred form. ``step`` resolves through ``steps.py``, so the prompt
        text this depends on is written down in one place instead of in every test.
        """
        from .steps import anchor_for

        return self.on_chat(contains=[anchor_for(step), *_as_tuple(contains)], tool=tool)

    def on_chat(self, *, contains: str | list[str] | None = None, tool: str | None = None) -> RuleBuilder:
        """Match on prompt substrings, and optionally on the tool a turn offers."""
        return RuleBuilder(self, contains=_as_tuple(contains), tool=tool)

    def add_rule(self, rule: ChatRule) -> None:
        self._rules.append(rule)

    def resolve(self, request: ChatRequest) -> StubbedReply | None:
        """The reply to answer with, or ``None`` when nothing matched."""
        self.calls.append(request)
        for rule in self._rules:
            if rule.matches(request):
                return rule.respond(request)

        self.unmatched.append(UnmatchedCall(tools=request.tools, prompt=request.all_text))
        return None

    def prompts_for(self, step: str) -> list[str]:
        """Every prompt sent to one named step, for asserting what was assembled."""
        from .steps import anchor_for

        anchor = anchor_for(step)
        return [call.all_text for call in self.calls if anchor in call.all_text]

    def reset(self) -> None:
        self._rules.clear()
        self.unmatched.clear()
        self.calls.clear()
        self._install_builtins()


class EmbeddingStub:
    """Deterministic embeddings — see ``lexical.py`` for why word overlap suffices."""

    def embed(self, text: str) -> list[float]:
        from .lexical import lexical_embedding

        return lexical_embedding(text)


class RerankStub:
    """Relevance from the same lexical model, so ranking stays self-consistent."""

    def score(self, query: str, document: str) -> float:
        from .lexical import lexical_relevance

        return lexical_relevance(query, document)


@dataclass(frozen=True)
class ReceivedWebhook:
    """One delivery the stub accepted, kept so a test can assert on it."""

    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class Stubs:
    """The three backends a test configures, handed to it as one object."""

    llm: LLMStub = field(default_factory=LLMStub)
    embeddings: EmbeddingStub = field(default_factory=EmbeddingStub)
    rerank: RerankStub = field(default_factory=RerankStub)
    rejected_requests: list[str] = field(default_factory=list)
    """Requests the stub refused as malformed — see ``validation.py``."""

    webhooks: list[ReceivedWebhook] = field(default_factory=list)
    """Webhook deliveries the stub received, in arrival order.

    The stub is the only receiver a hermetic test can offer, so it doubles as the
    customer endpoint: without somewhere for a delivery to land, "the webhook
    fired" can only be read off the server's own delivery log, which proves it
    tried rather than that anything arrived.
    """

    def reset(self) -> None:
        """Between tests. The embedding and rerank stubs are pure, so only the
        rulebook and the rejection log carry state worth clearing."""
        self.llm.reset()
        self.rejected_requests.clear()
        self.webhooks.clear()


def _as_tuple(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)
