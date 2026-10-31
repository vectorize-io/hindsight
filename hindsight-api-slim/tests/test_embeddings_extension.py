"""HINDSIGHT_API_EMBEDDINGS_EXTENSION: an extension supplies the engine's embeddings model."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.engine.embeddings import Embeddings, RemoteTEIEmbeddings
from hindsight_api.engine.memory_engine import _current_schema, get_current_schema
from hindsight_api.engine.retain import embedding_utils
from hindsight_api.extensions import EmbeddingsExtension
from hindsight_api.extensions.loader import EXTENSION_KINDS


class _FixedEmbeddings(Embeddings):
    """Every vector is ``[value] * dim``, so a test can tell which model made it."""

    def __init__(self, value: float, dim: int = 384) -> None:
        self.value = value
        self._dim = dim
        self.initialized = False

    @property
    def provider_name(self) -> str:
        return f"fixed-{self.value}"

    @property
    def dimension(self) -> int:
        return self._dim

    async def initialize(self) -> None:
        self.initialized = True

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[self.value] * self._dim for _ in texts]


class _PerSchemaEmbeddings(Embeddings):
    """The documented use: route each call by the current tenant schema, refuse the rest."""

    def __init__(self, routes: dict[str, Embeddings]) -> None:
        self.routes = routes

    @property
    def provider_name(self) -> str:
        return "per-schema"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        for model in self.routes.values():
            await model.initialize()

    async def encode(self, texts: list[str]) -> list[list[float]]:
        schema = get_current_schema()
        if schema not in self.routes:
            raise RuntimeError(f"no embeddings model for {schema}")
        return await self.routes[schema].encode(texts)


# Extensions under test. Class attributes rather than instance state because the
# loader builds the instance; each test resets what it reads.


class _Replaces(EmbeddingsExtension):
    def create_embeddings(self, build_default):
        return _FixedEmbeddings(1.0)


class _Wraps(EmbeddingsExtension):
    def create_embeddings(self, build_default):
        default = build_default()
        return _PerSchemaEmbeddings({"tenant_old": default, "tenant_new": _FixedEmbeddings(2.0)})


class _ReturnsJunk(EmbeddingsExtension):
    def create_embeddings(self, build_default):
        return object()


class _Raises(EmbeddingsExtension):
    def create_embeddings(self, build_default):
        raise RuntimeError("model store unreachable")


class _ReadsTheEngine(EmbeddingsExtension):
    seen = None

    def create_embeddings(self, build_default):
        _ReadsTheEngine.seen = self.context.get_memory_engine()
        return build_default()


class _Routes(EmbeddingsExtension):
    def create_embeddings(self, build_default):
        return _PerSchemaEmbeddings(
            {
                "tenant_a": _FixedEmbeddings(1.0),
                "tenant_b": _FixedEmbeddings(2.0),
                "tenant_c": _FixedEmbeddings(3.0, 768),
            }
        )


def _env(ext: str | None, **extra: str) -> dict[str, str]:
    env = {
        "HINDSIGHT_API_LLM_PROVIDER": "none",
        "HINDSIGHT_API_LLM_MODEL": "none",
        "HINDSIGHT_API_LLM_API_KEY": "test-key",
        # A remote default: building it opens no connection and loads no model.
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "tei",
        "HINDSIGHT_API_EMBEDDINGS_TEI_URL": "http://tei.invalid:8080",
        **extra,
    }
    if ext is not None:
        env["HINDSIGHT_API_EMBEDDINGS_EXTENSION"] = f"{__name__}:{ext}"
    return env


def _build_engine(env: dict[str, str], embeddings=None):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import MemoryEngine

    environ = {k: v for k, v in os.environ.items() if k != "HINDSIGHT_API_EMBEDDINGS_EXTENSION"}
    try:
        with patch.dict(os.environ, {**environ, **env}, clear=True):
            clear_config_cache()
            return MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=embeddings)
    finally:
        # Keep the patched config from bleeding into other tests on this worker.
        clear_config_cache()


def test_without_the_extension_the_server_settings_build_the_model():
    engine = _build_engine(_env(None))
    assert isinstance(engine.embeddings, RemoteTEIEmbeddings)
    assert engine._embeddings_extension is None


def test_the_extension_replaces_the_model():
    engine = _build_engine(_env("_Replaces"))
    assert isinstance(engine._embeddings_extension, _Replaces)
    assert isinstance(engine.embeddings, _FixedEmbeddings) and engine.embeddings.value == 1.0


def test_the_extension_can_wrap_the_default():
    engine = _build_engine(_env("_Wraps"))
    routes = engine.embeddings.routes
    assert isinstance(routes["tenant_old"], RemoteTEIEmbeddings)
    assert isinstance(routes["tenant_new"], _FixedEmbeddings)


def test_an_explicit_model_wins_and_the_extension_is_not_loaded():
    given = MagicMock()
    given.dimension = 384
    engine = _build_engine(_env("_Replaces"), embeddings=given)
    assert engine.embeddings is given
    assert engine._embeddings_extension is None


def test_the_extension_gets_the_engine_context_and_its_config():
    engine = _build_engine(_env("_Replaces", HINDSIGHT_API_EMBEDDINGS_MY_OPTION="x"))
    ext = engine._embeddings_extension
    assert ext.context is engine._ext_ctx
    assert ext.config["my_option"] == "x"


def test_the_extension_can_use_its_context_while_building_the_model():
    _ReadsTheEngine.seen = None
    engine = _build_engine(_env("_ReadsTheEngine"))
    assert _ReadsTheEngine.seen is engine


def test_returning_something_other_than_embeddings_fails_startup():
    with pytest.raises(TypeError, match="not an Embeddings"):
        _build_engine(_env("_ReturnsJunk"))


def test_an_extension_that_raises_fails_startup():
    with pytest.raises(RuntimeError, match="model store unreachable"):
        _build_engine(_env("_Raises"))


def test_the_kind_is_registered():
    assert "EmbeddingsExtension" in {kind.class_name for kind in EXTENSION_KINDS}


@pytest.mark.asyncio
async def test_initialize_reaches_the_wrapped_models():
    engine = _build_engine(_env("_Routes"))
    await engine.embeddings.initialize()
    assert all(model.initialized for model in engine.embeddings.routes.values())


async def _embed_as(schema: str, engine, input_type="document"):
    token = _current_schema.set(schema)
    try:
        return await embedding_utils.generate_embeddings_batch(engine.embeddings, ["hello"], input_type=input_type)
    finally:
        _current_schema.reset(token)


@pytest.mark.asyncio
async def test_each_tenant_gets_its_own_model_on_the_engine_path():
    engine = _build_engine(_env("_Routes"))
    assert (await _embed_as("tenant_a", engine))[0][0] == 1.0
    assert (await _embed_as("tenant_b", engine))[0][0] == 2.0
    # Query side too: recall must search with the model the tenant stored with.
    assert (await _embed_as("tenant_b", engine, input_type="query"))[0][0] == 2.0


@pytest.mark.asyncio
async def test_an_unroutable_tenant_is_refused_not_defaulted():
    engine = _build_engine(_env("_Routes"))
    # The engine wraps backend errors; the refusal is what fails the call.
    with pytest.raises(Exception, match="no embeddings model for tenant_x") as caught:
        await _embed_as("tenant_x", engine)
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_a_route_of_another_dimension_fails_loudly():
    """The engine sized every vector column from ``dimension`` (384); a route to a 768-d
    model must fail the call, never store a vector of the wrong size."""
    engine = _build_engine(_env("_Routes"))
    with pytest.raises(Exception, match="768|384|dimension"):
        await _embed_as("tenant_c", engine)
