# LLM Wrapper Refactoring Plan v2

## Current State
- `llm_wrapper.py`: 2000+ lines, monolithic class with provider-specific private methods
- `LLMInterface`: Abstract base class exists but not used
- `providers/codex_llm.py`: Exists but incomplete and not integrated

## Goal
Clean provider abstraction where each provider is a separate class implementing `LLMInterface`.

## Architecture

```
hindsight_api/engine/
├── llm_interface.py              (✓ exists - abstract base class)
├── llm_wrapper.py                (refactor to thin facade + factory)
└── providers/
    ├── __init__.py               (export all providers)
    ├── openai_compatible_llm.py  (OpenAI, Groq, Ollama, LMStudio)
    ├── anthropic_llm.py          (Anthropic)
    ├── gemini_llm.py             (Gemini, VertexAI)
    ├── codex_llm.py              (✓ exists - update)
    ├── claude_code_llm.py        (Claude Code)
    └── mock_llm.py               (Mock for testing)
```

## Implementation Steps

### 1. Create Factory Function (in llm_wrapper.py)
```python
def create_llm_provider(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str,
    **kwargs
) -> LLMInterface:
    """Factory to instantiate the correct provider."""
    if provider == "openai-codex":
        from .providers import CodexLLM
        return CodexLLM(...)
    elif provider == "claude-code":
        from .providers import ClaudeCodeLLM
        return ClaudeCodeLLM(...)
    # ... etc
```

### 2. Provider Classes Structure

Each provider implements:
```python
class ProviderLLM(LLMInterface):
    async def verify_connection(self) -> None
    async def call(self, messages, ...) -> Any
    async def call_with_tools(self, messages, tools, ...) -> LLMToolCallResult
    async def cleanup(self) -> None
```

### 3. Backwards Compatibility

Keep `LLMProvider` as a facade:
```python
class LLMProvider:
    def __init__(self, ...):
        self._provider = create_llm_provider(...)

    async def call(self, ...):
        return await self._provider.call(...)

    @classmethod
    def for_memory(cls) -> "LLMProvider":
        # Keep existing factory methods
```

## Benefits
1. Each provider is ~200-400 lines (manageable)
2. Easy to add new providers (no changes to existing code)
3. Easy to test individual providers
4. Clear separation of concerns
5. Maintains backwards compatibility

## Migration Strategy
1. Extract one provider at a time
2. Test after each extraction
3. Keep old code commented until all tests pass
4. Remove old code in final commit
