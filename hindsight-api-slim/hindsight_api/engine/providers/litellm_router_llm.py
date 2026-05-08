"""
LiteLLM Router LLM provider — pure pass-through to ``litellm.Router``.

The full configuration object is forwarded verbatim. We do not translate model
names, infer fallbacks, or impose defaults: whatever the user puts in the
``HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG`` env var becomes ``Router(**config)``.
The only Hindsight-imposed convention is that requests are issued against the
``model_name`` of the first entry in ``model_list``; chain ordering, fallbacks,
load-balancing, retries and cooldowns are LiteLLM Router's responsibility.

See https://docs.litellm.ai/docs/routing for the supported keys (``model_list``,
``fallbacks``, ``context_window_fallbacks``, ``num_retries``, ``cooldown_time``,
``routing_strategy``, ``allowed_fails``, …).

The retry/parse/metrics loop is shared with ``LiteLLMLLM`` via inheritance: this
class only overrides the small surface that differs (the completion fn, the
deployment kwargs, and the deployment-name resolution for metrics).

Example ``HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG``::

    {
      "model_list": [
        {"model_name": "primary",  "litellm_params": {"model": "openai/gpt-4o-mini",       "api_key": "sk-..."}},
        {"model_name": "fallback", "litellm_params": {"model": "anthropic/claude-sonnet-4", "api_key": "sk-ant-..."}}
      ],
      "fallbacks": [{"primary": ["fallback"]}],
      "num_retries": 0,
      "cooldown_time": 60
    }
"""

import logging
from typing import Any

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

logger = logging.getLogger(__name__)


class LiteLLMRouterLLM(LiteLLMLLM):
    """
    LLM provider backed by ``litellm.Router``.

    The full Router config is supplied by the caller. We pass it verbatim to
    ``Router(**config)`` and route requests against the first ``model_list``
    entry's ``model_name``. Inherits the retry/parse/metrics loop from
    ``LiteLLMLLM``; only the completion fn and the call kwargs differ.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        config: dict[str, Any],
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
        if not isinstance(config, dict) or not config.get("model_list"):
            raise ValueError("LiteLLMRouterLLM requires a config dict with a non-empty 'model_list'")
        first = config["model_list"][0]
        primary_model_name = first.get("model_name") if isinstance(first, dict) else None
        if not primary_model_name:
            raise ValueError("LiteLLMRouterLLM: model_list[0] must have a 'model_name'")

        self.config = config
        self._primary_model_name = primary_model_name

        from litellm import Router

        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        self._router = Router(**config)

        logger.info(
            f"LiteLLM Router initialized: {len(config['model_list'])} deployment(s), "
            f"primary model_name={primary_model_name!r}"
        )

    # ── overrides for the shared retry/parse loop ───────────────────────────

    @property
    def _stage_label(self) -> str:
        return "litellmrouter"

    async def _acompletion(self, **kwargs: Any) -> Any:
        return await self._router.acompletion(**kwargs)

    def _resolve_completion_model(self, response: Any) -> str:
        hidden = getattr(response, "_hidden_params", None) or {}
        return hidden.get("model") or self._primary_model_name

    def _build_common_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        # Always issue against the primary group; Router handles deployment selection,
        # cross-group fallbacks, retries, cooldowns — whatever the user configured.
        kwargs: dict[str, Any] = {
            "model": self._primary_model_name,
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
