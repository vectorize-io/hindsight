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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class ChatRequest:
    """One ``/v1/chat/completions`` call, in the terms a rule matches on."""

    model: str
    messages: list[dict[str, Any]]

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

    def matches(self, request: ChatRequest) -> bool:
        haystack = request.all_text
        return all(needle in haystack for needle in self.contains)


class RuleBuilder:
    """The ``when`` half of a rule; one of the ``returns*`` methods completes it."""

    def __init__(self, stub: LLMStub, *, contains: tuple[str, ...]) -> None:
        self._stub = stub
        self._contains = contains

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

    def _register(self, respond: Callable[[ChatRequest], StubbedReply]) -> LLMStub:
        self._stub.add_rule(ChatRule(contains=self._contains, respond=respond))
        return self._stub


def _assistant_message(content: str) -> StubbedReply:
    return StubbedReply(message={"role": "assistant", "content": content}, finish_reason="stop")


@dataclass(frozen=True)
class UnmatchedCall:
    """A call no rule answered, kept so the fixture can fail the test with it."""

    excerpt: str

    def suggestion(self) -> str:
        """A rule the author can paste into the test, near enough to be a starting point."""
        from .steps import STEP_ANCHORS

        step = next((name for name, anchor in STEP_ANCHORS.items() if anchor in self.excerpt), None)
        if step:
            return f'llm.on_step("{step}", contains=[...]).returns(...)'
        return (
            "llm.on_chat(contains=[...]).returns(...)  "
            "# no known step anchor matched — add one to steps.py. Prompt: "
            f"{self.excerpt[:400]!r}"
        )


class LLMStub:
    """The rulebook for ``/v1/chat/completions``."""

    def __init__(self) -> None:
        self._rules: list[ChatRule] = []
        self.unmatched: list[UnmatchedCall] = []
        self._install_builtins()

    def _install_builtins(self) -> None:
        """Answer the calls that are startup plumbing rather than test subject matter.

        ``verify_connection`` pings the provider before the server reports healthy.
        Making every test declare a rule for it would be noise with no signal.
        """
        self.on_step("connection_probe").returns_text("ok")

    def on_step(self, step: str, *, contains: str | list[str] | None = None) -> RuleBuilder:
        """Declare what the model says at a named pipeline step.

        The preferred form. ``step`` resolves through ``steps.py``, so the prompt
        text this depends on is written down in one place instead of in every test.
        """
        from .steps import anchor_for

        return self.on_chat(contains=[anchor_for(step), *_as_tuple(contains)])

    def on_chat(self, *, contains: str | list[str] | None = None) -> RuleBuilder:
        """Match on prompt substrings alone, for a call with no named step yet."""
        return RuleBuilder(self, contains=_as_tuple(contains))

    def add_rule(self, rule: ChatRule) -> None:
        self._rules.append(rule)

    def resolve(self, request: ChatRequest) -> StubbedReply | None:
        """The reply to answer with, or ``None`` when nothing matched."""
        for rule in self._rules:
            if rule.matches(request):
                return rule.respond(request)

        self.unmatched.append(UnmatchedCall(excerpt=request.user_text or request.all_text))
        return None

    def reset(self) -> None:
        self._rules.clear()
        self.unmatched.clear()
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


@dataclass
class Stubs:
    """The three backends a test configures, handed to it as one object."""

    llm: LLMStub = field(default_factory=LLMStub)
    embeddings: EmbeddingStub = field(default_factory=EmbeddingStub)
    rerank: RerankStub = field(default_factory=RerankStub)
    rejected_requests: list[str] = field(default_factory=list)
    """Requests the stub refused as malformed — see ``validation.py``."""

    def reset(self) -> None:
        """Between tests. The embedding and rerank stubs are pure, so only the
        rulebook and the rejection log carry state worth clearing."""
        self.llm.reset()
        self.rejected_requests.clear()


def _as_tuple(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)
