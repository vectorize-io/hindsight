"""
LiteLLM Router LLM provider — ordered fallback across multiple deployments.

Wraps ``litellm.Router`` with a chain of deployments. Requests are tried against
the primary deployment first; on transient errors (rate-limit, timeout, 5xx) the
Router falls back to subsequent deployments in declared order. Chain entries are
modelled as distinct LiteLLM ``model_name`` groups so that the ``fallbacks`` map
expresses the linear chain explicitly — this gives deterministic fallback order
rather than the load-balancing behaviour you get from same-group deployments.

Auth-style errors (401/403) are not retried across deployments, since fallback
on a misconfigured key would just propagate the failure.

The retry/parse/metrics loop is shared with ``LiteLLMLLM`` via inheritance: this
class only overrides the small surface that differs (the completion fn, the
deployment kwargs, and the deployment-name resolution for metrics).

References:
    - LiteLLM Router: https://docs.litellm.ai/docs/routing
    - Router fallbacks: https://docs.litellm.ai/docs/routing#fallbacks

Chain entry shape (matches ``HINDSIGHT_API_LLM_LITELLMROUTER_CHAIN``)::

    [
      {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."},
      {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "sk-ant-..."}
    ]

Entries support arbitrary extra keys: anything not in
``{provider, model, api_key, base_url, litellm_params}`` is forwarded to the
LiteLLM Router deployment top-level (e.g. ``rpm``, ``tpm``, ``weight``,
``model_info``). Anything inside a ``litellm_params`` sub-object is merged into
the inner ``litellm_params`` dict (e.g. per-deployment ``temperature``,
``extra_headers``).
"""

import logging
from typing import Any

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

logger = logging.getLogger(__name__)


# Hindsight provider name → LiteLLM model prefix.
# Providers not listed are treated as OpenAI-compatible: the configured base_url
# carries the routing and the model is sent under the "openai/" prefix.
_LITELLM_PROVIDER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "vertexai": "vertex_ai",
    "groq": "groq",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "bedrock": "bedrock",
    "ollama": "ollama_chat",
    "litellm": None,  # caller already provided a fully-qualified model string
}

# Primary group name. All chain entries fall back to subsequent indices.
_PRIMARY_GROUP = "hindsight-chain-0"


def _build_litellm_model(provider: str, model: str) -> str:
    """Translate a Hindsight (provider, model) pair into a LiteLLM model string."""
    if "/" in model:
        return model  # caller pre-qualified the model, respect it
    prefix = _LITELLM_PROVIDER_PREFIX.get(provider.lower(), "openai")
    if prefix is None:
        return model
    return f"{prefix}/{model}"


def _build_model_list(chain: list[dict[str, Any]], timeout: float) -> list[dict[str, Any]]:
    """
    Translate a Hindsight chain into a LiteLLM Router ``model_list``.

    Known Hindsight keys (``provider``, ``model``, ``api_key``, ``base_url``) are
    mapped onto LiteLLM's ``litellm_params``. Anything inside a ``litellm_params``
    sub-object is merged into the inner dict (later wins). Any other top-level
    key is forwarded verbatim to the deployment record (``rpm``, ``tpm``,
    ``weight``, ``model_info`` …).
    """
    model_list: list[dict[str, Any]] = []
    for i, raw_entry in enumerate(chain):
        entry = dict(raw_entry)  # don't mutate caller data
        provider = entry.pop("provider")
        model = entry.pop("model")
        api_key = entry.pop("api_key", None)
        base_url = entry.pop("base_url", None)
        nested_params = dict(entry.pop("litellm_params", {}) or {})

        litellm_params: dict[str, Any] = {
            "model": _build_litellm_model(provider, model),
            "timeout": timeout,
        }
        if api_key:
            litellm_params["api_key"] = api_key
        if base_url:
            litellm_params["api_base"] = base_url
        # Caller-supplied litellm_params win over our defaults.
        litellm_params.update(nested_params)

        deployment: dict[str, Any] = {
            "model_name": f"hindsight-chain-{i}",
            "litellm_params": litellm_params,
        }
        # Any remaining keys are Router-level deployment fields (rpm, tpm, weight, ...).
        deployment.update(entry)
        model_list.append(deployment)
    return model_list


def _build_fallbacks(chain_length: int) -> list[dict[str, list[str]]]:
    """Map the primary group to every subsequent chain index in declared order."""
    if chain_length <= 1:
        return []
    return [{_PRIMARY_GROUP: [f"hindsight-chain-{i}" for i in range(1, chain_length)]}]


class LiteLLMRouterLLM(LiteLLMLLM):
    """
    LLM provider backed by ``litellm.Router`` with ordered fallback.

    Each entry in the chain becomes a distinct LiteLLM model_group; the primary
    group is wired to fall back through the remaining groups in declared order.
    Requests are always issued against the primary group — Router internally
    re-issues against the fallback groups when the primary fails.

    Inherits the retry/parse/metrics loop from ``LiteLLMLLM`` and only overrides
    the small surface that differs.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        chain: list[dict[str, Any]],
        reasoning_effort: str = "low",
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            **kwargs,
        )
        if not chain:
            raise ValueError("LiteLLMRouterLLM requires a non-empty chain")
        self.chain = chain

        from litellm import Router

        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)

        model_list = _build_model_list(chain, timeout=timeout)
        fallbacks = _build_fallbacks(len(chain))
        self._router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=0,  # outer loop in call()/call_with_tools() owns retries
            allowed_fails=1,  # cooldown a deployment after a single failure within a window
            cooldown_time=60,
        )

        chain_summary = ", ".join(f"{e['provider']}/{e['model']}" for e in chain)
        logger.info(f"LiteLLM Router initialized with {len(chain)} deployment(s): [{chain_summary}]")

    # ── overrides for the shared retry/parse loop ───────────────────────────

    @property
    def _stage_label(self) -> str:
        return "litellmrouter"

    async def _acompletion(self, **kwargs: Any) -> Any:
        return await self._router.acompletion(**kwargs)

    def _resolve_completion_model(self, response: Any) -> str:
        hidden = getattr(response, "_hidden_params", None) or {}
        return hidden.get("model") or self.model

    def _build_common_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        # Always issue against the primary group; Router handles deployment selection.
        kwargs: dict[str, Any] = {
            "model": _PRIMARY_GROUP,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    async def verify_connection(self) -> None:
        from hindsight_api.engine.llm_interface import OutputTooLongError

        try:
            await self.call(
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=50,
                temperature=0.0,
                scope="verification",
                max_retries=0,
            )
            logger.info("LiteLLM Router connection verified successfully")
        except OutputTooLongError:
            logger.info("LiteLLM Router connection verified successfully (response truncated)")
        except Exception as e:
            logger.error(f"LiteLLM Router connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify LiteLLM Router connection: {e}") from e
