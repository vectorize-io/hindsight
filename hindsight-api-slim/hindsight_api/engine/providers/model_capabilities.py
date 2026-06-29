"""Shared provider model capability helpers."""


def supports_openai_compatible_reasoning(model: str) -> bool:
    """Return True for OpenAI-compatible reasoning model names."""
    model_lower = (model or "").lower()
    if "deepseek" in model_lower:
        # DeepSeek v4-flash is the non-thinking route. Treating every
        # DeepSeek model as reasoning injects unsupported reasoning params.
        return any(x in model_lower for x in ["v4-pro", "reasoner", "r1", "thinking"])
    return any(x in model_lower for x in ["gpt-5", "o1", "o3"])
