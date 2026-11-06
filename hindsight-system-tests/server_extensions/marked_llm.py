"""A custom LLM provider, loaded by class path (``marked_llm:MarkedOpenAILLM``).

It is a built-in provider with one detail changed — the shape the docs recommend for
an OpenAI-compatible backend that is almost, but not quite, OpenAI: subclass
``OpenAICompatibleLLM``, tell it which wire dialect it speaks, and override what
differs. What differs here is a header naming this class on every request, so a story
can see at the provider that the call came through it.
"""

from typing import Any

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

MARK_HEADER = "x-marked-llm"


class MarkedOpenAILLM(OpenAICompatibleLLM):
    def __init__(self, **kwargs: Any) -> None:
        # The configured provider is this class's path; the parent needs the dialect it speaks.
        kwargs["provider"] = "openai"
        kwargs["default_headers"] = {**(kwargs.get("default_headers") or {}), MARK_HEADER: "MarkedOpenAILLM"}
        super().__init__(**kwargs)
