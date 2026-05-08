"""
LiteLLM Router LLM provider — pure pass-through to ``litellm.Router``.

The full configuration object is forwarded verbatim. We do not translate model
names, infer fallbacks, validate shape, or introspect Router internals:
whatever the user puts in ``HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG`` becomes
``Router(**config)``. If the shape is wrong, LiteLLM Router raises.

The only Hindsight-imposed convention is that one entry in ``model_list``
must have ``model_name: "default"`` — that's the entrypoint we issue
completions against. Everything else (ordering, fallbacks, load-balancing,
weighted picks, rate limits, retries, cooldowns) is whatever the user
configures via LiteLLM's own keys.

See https://docs.litellm.ai/docs/routing for the supported keys (``model_list``,
``fallbacks``, ``context_window_fallbacks``, ``num_retries``, ``cooldown_time``,
``routing_strategy``, ``allowed_fails``, …).

The retry/parse/metrics loop is shared with ``LiteLLMLLM`` via inheritance:
this class only overrides the completion fn, the call kwargs, and the model
name reported in metrics.

Example ``HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG``::

    {
      "model_list": [
        {"model_name": "default",  "litellm_params": {"model": "openai/gpt-4o-mini",       "api_key": "sk-..."}},
        {"model_name": "fallback", "litellm_params": {"model": "anthropic/claude-sonnet-4", "api_key": "sk-ant-..."}}
      ],
      "fallbacks": [{"default": ["fallback"]}],
      "num_retries": 0,
      "cooldown_time": 60
    }
"""

import logging
from typing import Any

from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

logger = logging.getLogger(__name__)


# Hindsight always issues completions against this ``model_name``. Users must
# include at least one entry with ``model_name: "default"`` in their config's
# ``model_list``; that entry is the entrypoint, and any other entries become
# fallback / load-balance / weighted-pool members per the user's own
# ``fallbacks`` / ``routing_strategy`` settings.
_ENTRYPOINT_MODEL_NAME = "default"


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
        self.config = config

        from litellm import Router

        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        # Pure pass-through: whatever the user gave goes straight to LiteLLM Router.
        # If the shape is invalid, Router raises its own error — we don't pre-validate
        # or introspect Router internals.
        self._router = Router(**config)

        # Pre-compute the most conservative output-tokens cap across every configured
        # deployment so a single max_completion_tokens value works no matter which
        # deployment Router picks. Uses LiteLLM's own per-model registry; unknown
        # models contribute no cap. See LiteLLMLLM._cap_max_completion_tokens.
        self._router_output_cap = self._compute_router_output_cap(config)

        logger.info("LiteLLM Router initialized; entrypoint model_name=%r", _ENTRYPOINT_MODEL_NAME)

    def _compute_router_output_cap(self, config: dict[str, Any]) -> int | None:
        caps: list[int] = []
        for deployment in (config.get("model_list") or []) if isinstance(config, dict) else []:
            if not isinstance(deployment, dict):
                continue
            params = deployment.get("litellm_params") or {}
            model_str = params.get("model") if isinstance(params, dict) else None
            if not model_str:
                continue
            try:
                cap = self._litellm.get_max_tokens(model_str)
            except Exception:
                cap = None
            if cap:
                caps.append(int(cap))
        return min(caps) if caps else None

    # ── overrides for the shared retry/parse loop ───────────────────────────

    @property
    def _stage_label(self) -> str:
        return "litellmrouter"

    async def _acompletion(self, **kwargs: Any) -> Any:
        return await self._router.acompletion(**kwargs)

    def _resolve_completion_model(self, response: Any) -> str:
        hidden = getattr(response, "_hidden_params", None) or {}
        return hidden.get("model") or _ENTRYPOINT_MODEL_NAME

    def _get_model_output_cap(self) -> int | None:
        return self._router_output_cap

    def _build_common_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        # Always issue against the entrypoint group; Router handles deployment selection,
        # cross-group fallbacks, retries, cooldowns — whatever the user configured.
        kwargs: dict[str, Any] = {
            "model": _ENTRYPOINT_MODEL_NAME,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self._cap_max_completion_tokens(max_completion_tokens)
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
