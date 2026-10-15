"""Pick the LLM — or just the key — for each call from the bank and operation it runs for.

Routes are bank-id glob patterns, tried in order; the first match wins::

    HINDSIGHT_API_LLM_REQUEST_EXTENSION=hindsight_ext_bank_llm_routing:BankLLMRoutingExtension
    HINDSIGHT_API_LLM_REQUEST_ROUTES='[
        {"bank": "acme-*", "operation": "reflect", "llm": {"provider": "anthropic", "model": "claude-sonnet-5-5", "api_key": "sk-ant"}},
        {"bank": "acme-*", "api_key": "sk-acme"},
        {"bank": "globex-*", "api_key": "sk-globex", "headers": {"x-team": "globex"}}
    ]'
    HINDSIGHT_API_LLM_REQUEST_UNMATCHED=deny   # or "server"
"""

from __future__ import annotations

import json
import logging
from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hindsight_api.config import LLMMemberConfig
from hindsight_api.extensions.llm_request import (
    LLMRequestExtension,
    LLMRequestOverrides,
    LLMRequestRefused,
    LLMRequestTarget,
)

logger = logging.getLogger(__name__)

UNMATCHED_DENY = "deny"
UNMATCHED_SERVER = "server"


class RouteLLM(BaseModel):
    """A whole LLM to send a route's calls to, in place of the server's."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key: str | None = Field(default=None, min_length=1, repr=False)

    def member(self) -> LLMMemberConfig:
        return LLMMemberConfig(provider=self.provider, api_key=self.api_key, model=self.model, base_url=self.base_url)


class Route(BaseModel):
    """Send calls for banks matching ``bank`` to ``llm``, and/or with ``api_key`` and ``headers``.

    Strict and closed: parsed straight from operator JSON, where a misspelled key
    (``"api_ky"``) must fail startup instead of quietly sending that bank's calls
    somewhere else.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    bank: str = Field(min_length=1)
    """Glob over the bank id (``fnmatch``, case-sensitive): ``acme-*``, ``user-?``, ``*``."""
    schema_: str | None = Field(default=None, alias="schema", min_length=1)
    """Only match banks in this tenant schema. Bank ids are unique per schema only."""
    operation: str | None = Field(default=None, min_length=1)
    """Only match calls of operations matching this glob: ``retain``, ``reflect``,
    ``consolidation*``, ``mental_model_*``. Unset matches every operation."""
    provider: str | None = Field(default=None, min_length=1)
    """Only match calls the server would send to this provider. With a multi-LLM chain,
    or a provider per operation, a key-only route otherwise hands its key to every one."""
    model: str | None = Field(default=None, min_length=1)
    """Only match calls the server would send to models matching this glob."""
    llm: RouteLLM | None = None
    api_key: str | None = Field(default=None, min_length=1, repr=False)
    """The key for the LLM the call goes to — the server's, or ``llm``'s."""
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    def overrides(self) -> LLMRequestOverrides:
        return LLMRequestOverrides(
            llm=self.llm.member() if self.llm is not None else None, api_key=self.api_key, headers=self.headers
        )

    def describe(self) -> str:
        """The route for a log line: patterns and LLM, never a credential."""
        where = self.bank if self.schema_ is None else f"{self.schema_}/{self.bank}"
        if self.operation is not None:
            where += f"[{self.operation}]"
        if self.provider is not None or self.model is not None:
            where += f"@{self.provider or '*'}/{self.model or '*'}"
        return where if self.llm is None else f"{where} -> {self.llm.provider}/{self.llm.model}"

    def matches(self, target: LLMRequestTarget) -> bool:
        if target.bank_id is None:
            return False
        if self.schema_ is not None and self.schema_ != target.schema:
            return False
        if self.operation is not None and not fnmatchcase(target.operation or "", self.operation):
            return False
        if self.provider is not None and self.provider != target.provider:
            return False
        if self.model is not None and not fnmatchcase(target.model, self.model):
            return False
        return fnmatchcase(target.bank_id, self.bank)


class BankLLMRoutingExtension(LLMRequestExtension):
    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)
        self.routes = _parse_routes(config.get("routes"))
        self.unmatched = (config.get("unmatched") or UNMATCHED_DENY).strip().lower()
        if self.unmatched not in (UNMATCHED_DENY, UNMATCHED_SERVER):
            raise ValueError(
                f"HINDSIGHT_API_LLM_REQUEST_UNMATCHED must be '{UNMATCHED_DENY}' or '{UNMATCHED_SERVER}', "
                f"got {self.unmatched!r}"
            )
        # Patterns only: the keys stay out of the log.
        logger.info(
            "bank-llm-routing: %d route(s) %s, unmatched banks -> %s",
            len(self.routes),
            [route.describe() for route in self.routes],
            self.unmatched,
        )

    async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides | None:
        for route in self.routes:
            if route.matches(target):
                return route.overrides()
        if self.unmatched == UNMATCHED_SERVER:
            return None
        raise LLMRequestRefused(
            f"No LLM route matches bank {target.bank_id!r} (schema {target.schema!r}, "
            f"operation {target.operation!r}); refusing to send the call on the server's LLM. Add a route "
            f"to HINDSIGHT_API_LLM_REQUEST_ROUTES, or set HINDSIGHT_API_LLM_REQUEST_UNMATCHED=server."
        )


def _parse_routes(raw: str | None) -> tuple[Route, ...]:
    if not raw or not raw.strip():
        raise ValueError("HINDSIGHT_API_LLM_REQUEST_ROUTES is required: a JSON list of {bank, llm|api_key} routes")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Not `from e`: the decode error quotes the input, which holds the keys.
        raise ValueError(
            f"HINDSIGHT_API_LLM_REQUEST_ROUTES is not valid JSON (line {e.lineno}, column {e.colno})"
        ) from None
    if not isinstance(data, list) or not data:
        raise ValueError("HINDSIGHT_API_LLM_REQUEST_ROUTES must be a non-empty JSON list")
    routes = []
    for i, item in enumerate(data):
        try:
            route = Route.model_validate(item)
        except ValidationError as e:
            # Field locations and messages only — pydantic's default text echoes the input values.
            problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'route'}: {err['msg']}" for err in e.errors())
            raise ValueError(f"HINDSIGHT_API_LLM_REQUEST_ROUTES[{i}] is invalid: {problems}") from None
        if route.llm is None and route.api_key is None and not route.headers:
            raise ValueError(f"HINDSIGHT_API_LLM_REQUEST_ROUTES[{i}] sets none of llm, api_key, headers")
        # Validates header names/values (no CR/LF) at startup rather than on the first call.
        try:
            route.overrides()
        except ValueError as e:
            raise ValueError(f"HINDSIGHT_API_LLM_REQUEST_ROUTES[{i}]: {e}") from None
        routes.append(route)
    return tuple(routes)
