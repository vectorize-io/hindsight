"""Unit tests for Hindsight Composio custom tools."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from hindsight_composio import (
    RecallInput,
    ReflectInput,
    RetainInput,
    configure,
    memory_instructions,
    register_hindsight_tools,
    reset_config,
)
from hindsight_composio.errors import HindsightError
from hindsight_composio.tools import _resolve_client

try:
    from composio import Composio

    _HAS_COMPOSIO = True
except Exception:  # pragma: no cover - composio is a hard dep
    _HAS_COMPOSIO = False


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _FakeExperimental:
    """Identity ``tool`` decorator that mirrors the real CustomTool slug."""

    def tool(self, fn):
        fn.slug = fn.__name__.upper()
        return fn


class FakeComposio:
    """Stand-in for a Composio instance for fast, offline unit tests."""

    def __init__(self):
        self.experimental = _FakeExperimental()


class FakeCtx:
    """Stand-in for the Composio SessionContext injected into tools."""

    def __init__(self, user_id=None):
        self.user_id = user_id


def _mock_client():
    client = MagicMock()
    client.retain = MagicMock()
    client.recall = MagicMock()
    client.reflect = MagicMock()
    client.create_bank = MagicMock()
    return client


def _mock_recall_response(texts):
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


def _mock_reflect_response(text):
    response = MagicMock()
    response.text = text
    return response


def _tools_by_name(composio, **kwargs):
    """Register tools on a FakeComposio and return them keyed by function name."""
    tools = register_hindsight_tools(composio, **kwargs)
    return {t.__name__: t for t in tools}


# ---------------------------------------------------------------------------
# _resolve_client
# ---------------------------------------------------------------------------


class TestResolveClient:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_explicit_client(self):
        client = _mock_client()
        assert _resolve_client(client, None, None) is client

    def test_creates_client_from_url(self):
        with patch("hindsight_composio.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", None)
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_creates_client_with_api_key(self):
        with patch("hindsight_composio.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", "my-key")
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0, api_key="my-key")

    def test_falls_back_to_global_config_url(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_composio.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once_with(base_url="http://config:8888", timeout=30.0)

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_composio.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://explicit:9999", None)
            mock_cls.assert_called_once_with(base_url="http://explicit:9999", timeout=30.0)

    def test_raises_without_url_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            _resolve_client(None, None, None)


# ---------------------------------------------------------------------------
# register_hindsight_tools — construction
# ---------------------------------------------------------------------------


class TestRegister:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_creates_three_tools_by_default(self):
        tools = _tools_by_name(FakeComposio(), client=_mock_client())
        assert set(tools) == {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}

    def test_enable_retain_only(self):
        tools = register_hindsight_tools(
            FakeComposio(),
            client=_mock_client(),
            enable_retain=True,
            enable_recall=False,
            enable_reflect=False,
        )
        assert len(tools) == 1
        assert tools[0].__name__ == "hindsight_retain"

    def test_no_tools_when_all_disabled(self):
        tools = register_hindsight_tools(
            FakeComposio(),
            client=_mock_client(),
            enable_retain=False,
            enable_recall=False,
            enable_reflect=False,
        )
        assert tools == []

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            register_hindsight_tools(FakeComposio())


# ---------------------------------------------------------------------------
# Bank resolution from ctx.user_id
# ---------------------------------------------------------------------------


class TestBankResolution:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_uses_user_id_as_bank(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_retain"](RetainInput(content="hi"), FakeCtx(user_id="alice"))
        assert result == {"status": "stored", "bank": "alice"}
        client.retain.assert_called_once_with(bank_id="alice", content="hi")
        client.create_bank.assert_called_once_with(bank_id="alice", name="alice")

    def test_falls_back_to_default_bank_without_user_id(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client, default_bank="shared")
        result = tools["hindsight_retain"](RetainInput(content="hi"), FakeCtx(user_id=None))
        assert result["bank"] == "shared"
        client.retain.assert_called_once_with(bank_id="shared", content="hi")

    def test_default_bank_from_config(self):
        configure(hindsight_api_url="http://localhost:8888", default_bank="config-bank")
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client)
        client.recall.return_value = _mock_recall_response(["x"])
        tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id=None))
        assert client.recall.call_args[1]["bank_id"] == "config-bank"

    def test_user_id_takes_precedence_over_default_bank(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client, default_bank="shared")
        tools["hindsight_retain"](RetainInput(content="hi"), FakeCtx(user_id="alice"))
        assert client.retain.call_args[1]["bank_id"] == "alice"

    def test_raises_when_no_user_id_and_no_default(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client)
        with pytest.raises(HindsightError, match="No Hindsight bank"):
            tools["hindsight_retain"](RetainInput(content="hi"), FakeCtx(user_id=None))

    def test_bank_created_only_once_per_bank(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client)
        tools["hindsight_retain"](RetainInput(content="a"), FakeCtx(user_id="alice"))
        tools["hindsight_retain"](RetainInput(content="b"), FakeCtx(user_id="alice"))
        client.create_bank.assert_called_once()

    def test_bank_created_per_distinct_user(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client)
        tools["hindsight_retain"](RetainInput(content="a"), FakeCtx(user_id="alice"))
        tools["hindsight_retain"](RetainInput(content="b"), FakeCtx(user_id="bob"))
        assert client.create_bank.call_count == 2


# ---------------------------------------------------------------------------
# Retain
# ---------------------------------------------------------------------------


class TestRetain:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_retain_with_tags(self):
        client = _mock_client()
        tools = _tools_by_name(FakeComposio(), client=client, tags=["env:test"])
        tools["hindsight_retain"](RetainInput(content="c"), FakeCtx(user_id="alice"))
        assert client.retain.call_args[1]["tags"] == ["env:test"]

    def test_retain_bank_already_exists(self):
        client = _mock_client()
        client.create_bank.side_effect = Exception("already exists")
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_retain"](RetainInput(content="c"), FakeCtx(user_id="alice"))
        assert result["status"] == "stored"

    def test_retain_failure_raises_hindsight_error(self):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("network error")
        tools = _tools_by_name(FakeComposio(), client=client)
        with pytest.raises(HindsightError, match="Retain failed"):
            tools["hindsight_retain"](RetainInput(content="c"), FakeCtx(user_id="alice"))

    def test_retain_failure_logs_error(self, caplog):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("network error")
        tools = _tools_by_name(FakeComposio(), client=client)
        with caplog.at_level(logging.ERROR), pytest.raises(HindsightError):
            tools["hindsight_retain"](RetainInput(content="c"), FakeCtx(user_id="alice"))
        assert "Retain failed" in caplog.text


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_recall_returns_memories(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f1", "f2", "f3"])
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        assert result == {"memories": ["f1", "f2", "f3"], "count": 3}

    def test_recall_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        assert result == {"memories": [], "count": 0}

    def test_recall_none_results(self):
        client = _mock_client()
        response = MagicMock()
        response.results = None
        client.recall.return_value = response
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        assert result == {"memories": [], "count": 0}

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        tools = _tools_by_name(FakeComposio(), client=client, budget="high", max_tokens=2048)
        tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_with_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        tools = _tools_by_name(
            FakeComposio(),
            client=client,
            recall_tags=["scope:global"],
            recall_tags_match="all",
        )
        tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:global"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_without_tags_omits_tag_kwargs(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        tools = _tools_by_name(FakeComposio(), client=client)
        tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        call_kwargs = client.recall.call_args[1]
        assert "tags" not in call_kwargs
        assert "tags_match" not in call_kwargs

    def test_recall_does_not_create_bank(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        tools = _tools_by_name(FakeComposio(), client=client)
        tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))
        client.create_bank.assert_not_called()

    def test_recall_failure_raises_hindsight_error(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("network error")
        tools = _tools_by_name(FakeComposio(), client=client)
        with pytest.raises(HindsightError, match="Recall failed"):
            tools["hindsight_recall"](RecallInput(query="q"), FakeCtx(user_id="alice"))


# ---------------------------------------------------------------------------
# Reflect
# ---------------------------------------------------------------------------


class TestReflect:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_reflect_returns_answer(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("Synthesized answer")
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_reflect"](ReflectInput(query="q"), FakeCtx(user_id="alice"))
        assert result == {"answer": "Synthesized answer"}

    def test_reflect_empty_text_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        tools = _tools_by_name(FakeComposio(), client=client)
        result = tools["hindsight_reflect"](ReflectInput(query="q"), FakeCtx(user_id="alice"))
        assert result == {"answer": "No relevant memories found."}

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("a")
        tools = _tools_by_name(FakeComposio(), client=client, budget="high")
        tools["hindsight_reflect"](ReflectInput(query="q"), FakeCtx(user_id="alice"))
        assert client.reflect.call_args[1]["budget"] == "high"

    def test_reflect_failure_raises_hindsight_error(self):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("network error")
        tools = _tools_by_name(FakeComposio(), client=client)
        with pytest.raises(HindsightError, match="Reflect failed"):
            tools["hindsight_reflect"](ReflectInput(query="q"), FakeCtx(user_id="alice"))


# ---------------------------------------------------------------------------
# Real Composio SDK registration (guards against experimental API drift)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_COMPOSIO, reason="composio not installed")
class TestRealComposioRegistration:
    def test_registers_three_custom_tools(self):
        composio = Composio(api_key="dummy")
        tools = register_hindsight_tools(composio, client=_mock_client())
        assert len(tools) == 3
        slugs = {t.slug for t in tools}
        assert slugs == {"HINDSIGHT_RETAIN", "HINDSIGHT_RECALL", "HINDSIGHT_REFLECT"}

    def test_custom_tools_carry_input_schema(self):
        composio = Composio(api_key="dummy")
        tools = register_hindsight_tools(
            composio,
            client=_mock_client(),
            enable_recall=False,
            enable_reflect=False,
        )
        retain = tools[0]
        assert retain.input_params is RetainInput


# ---------------------------------------------------------------------------
# memory_instructions
# ---------------------------------------------------------------------------


class TestMemoryInstructions:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_formats_results_with_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["likes tea", "lives in NYC"])
        out = memory_instructions(bank_id="b", client=client)
        assert out == "Relevant memories:\n\n1. likes tea\n2. lives in NYC"

    def test_custom_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        out = memory_instructions(bank_id="b", client=client, prefix="Known:\n")
        assert out.startswith("Known:\n")

    def test_caps_at_max_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["a", "b", "c", "d"])
        out = memory_instructions(bank_id="b", client=client, max_results=2)
        assert "1. a" in out and "2. b" in out
        assert "c" not in out and "d" not in out

    def test_returns_empty_when_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        assert memory_instructions(bank_id="b", client=client) == ""

    def test_returns_empty_on_recall_failure(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("boom")
        assert memory_instructions(bank_id="b", client=client) == ""

    def test_passes_tags_through_to_recall(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["x"])
        memory_instructions(bank_id="b", client=client, tags=["t1"], tags_match="all")
        kwargs = client.recall.call_args[1]
        assert kwargs["tags"] == ["t1"]
        assert kwargs["tags_match"] == "all"

    def test_omits_tags_when_none(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["x"])
        memory_instructions(bank_id="b", client=client)
        kwargs = client.recall.call_args[1]
        assert "tags" not in kwargs

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError):
            memory_instructions(bank_id="b")
