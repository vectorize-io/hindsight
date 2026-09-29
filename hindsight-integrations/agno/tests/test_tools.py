"""Unit tests for Hindsight Agno tools."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from hindsight_agno import (
    HindsightTools,
    configure,
    memory_instructions,
    reset_config,
)
from hindsight_agno.errors import HindsightError
from hindsight_agno.tools import _TOOL_INSTRUCTIONS, _resolve_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client():
    """Create a mock Hindsight client."""
    client = MagicMock()
    client.retain = MagicMock()
    client.recall = MagicMock()
    client.reflect = MagicMock()
    client.create_bank = MagicMock()
    return client


def _mock_run_context(user_id=None, session_id=None):
    """Create a mock Agno RunContext."""
    ctx = MagicMock()
    ctx.user_id = user_id
    ctx.session_id = session_id
    return ctx


def _mock_recall_response(texts: list[str]):
    """Create a mock RecallResponse with results."""
    response = MagicMock()
    results = []
    for t in texts:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    return response


def _mock_reflect_response(text: str):
    """Create a mock ReflectResponse."""
    response = MagicMock()
    response.text = text
    return response


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

    def test_explicit_client_ignores_url_and_key(self):
        client = _mock_client()
        result = _resolve_client(client, "http://ignored", "ignored-key")
        assert result is client

    def test_creates_client_from_url(self):
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", None)
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_creates_client_with_api_key(self):
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", "my-key")
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0, api_key="my-key")

    def test_falls_back_to_global_config_url(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once_with(base_url="http://config:8888", timeout=30.0)

    def test_falls_back_to_global_config_api_key(self):
        configure(hindsight_api_url="http://config:8888", api_key="config-key")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once_with(base_url="http://config:8888", timeout=30.0, api_key="config-key")

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://explicit:9999", None)
            mock_cls.assert_called_once_with(base_url="http://explicit:9999", timeout=30.0)

    def test_explicit_api_key_overrides_config(self):
        configure(hindsight_api_url="http://config:8888", api_key="config-key")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, "explicit-key")
            mock_cls.assert_called_once_with(base_url="http://config:8888", timeout=30.0, api_key="explicit-key")

    def test_raises_without_url_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            _resolve_client(None, None, None)

    def test_raises_with_empty_config_no_url(self):
        # Config exists but has no url override and default is set,
        # so this should NOT raise since default URL exists in config
        configure()
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# HindsightTools initialization
# ---------------------------------------------------------------------------


class TestHindsightToolsInit:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_creates_three_tools_by_default(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert "retain_memory" in toolkit.functions
        assert "recall_memory" in toolkit.functions
        assert "reflect_on_memory" in toolkit.functions
        assert len(toolkit.functions) == 3

    def test_enable_retain_only(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=True,
            enable_recall=False,
            enable_reflect=False,
        )
        assert "retain_memory" in toolkit.functions
        assert "recall_memory" not in toolkit.functions
        assert "reflect_on_memory" not in toolkit.functions

    def test_enable_recall_only(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=True,
            enable_reflect=False,
        )
        assert "retain_memory" not in toolkit.functions
        assert "recall_memory" in toolkit.functions
        assert "reflect_on_memory" not in toolkit.functions

    def test_enable_reflect_only(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=False,
            enable_reflect=True,
        )
        assert "retain_memory" not in toolkit.functions
        assert "recall_memory" not in toolkit.functions
        assert "reflect_on_memory" in toolkit.functions

    def test_enable_two_tools(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=True,
            enable_recall=True,
            enable_reflect=False,
        )
        assert len(toolkit.functions) == 2
        assert "retain_memory" in toolkit.functions
        assert "recall_memory" in toolkit.functions

    def test_no_tools_when_all_disabled(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_recall=False,
            enable_reflect=False,
        )
        assert len(toolkit.functions) == 0

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            HindsightTools(bank_id="test")

    def test_falls_back_to_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            toolkit = HindsightTools(bank_id="test")
            assert "retain_memory" in toolkit.functions
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_explicit_url_overrides_config(self):
        configure(hindsight_api_url="http://config:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            HindsightTools(bank_id="test", hindsight_api_url="http://explicit:9999")
            mock_cls.assert_called_once_with(base_url="http://explicit:9999", timeout=30.0)

    def test_toolkit_name(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.name == "hindsight_tools"

    def test_toolkit_has_instructions(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.instructions == _TOOL_INSTRUCTIONS
        assert "retain_memory" in toolkit.instructions
        assert "recall_memory" in toolkit.instructions
        assert "reflect_on_memory" in toolkit.instructions

    def test_constructor_defaults_override_config_for_budget(self):
        """Constructor default budget='mid' is truthy, so config budget is not used.
        This matches pydantic-ai integration behavior — pass budget explicitly to override."""
        configure(hindsight_api_url="http://localhost:8888", budget="low")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            toolkit = HindsightTools(bank_id="test")
        client = mock_cls.return_value
        client.recall.return_value = _mock_recall_response(["fact"])
        ctx = _mock_run_context()
        toolkit.recall_memory(ctx, "q")
        # Constructor default "mid" wins over config "low" due to `or` logic
        assert client.recall.call_args[1]["budget"] == "mid"

    def test_explicit_budget_overrides_default(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            toolkit = HindsightTools(bank_id="test", budget="low")
        client = mock_cls.return_value
        client.recall.return_value = _mock_recall_response(["fact"])
        ctx = _mock_run_context()
        toolkit.recall_memory(ctx, "q")
        assert client.recall.call_args[1]["budget"] == "low"

    def test_config_defaults_for_tags(self):
        """Tags use 'is not None' check, so config tags ARE picked up when not explicitly set."""
        configure(
            hindsight_api_url="http://localhost:8888",
            tags=["config-tag"],
            recall_tags=["config-recall"],
        )
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            toolkit = HindsightTools(bank_id="test")
        client = mock_cls.return_value
        client.retain.return_value = None
        client.recall.return_value = _mock_recall_response(["fact"])
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")
        assert client.retain.call_args[1]["tags"] == ["config-tag"]

        toolkit.recall_memory(ctx, "q")
        assert client.recall.call_args[1]["tags"] == ["config-recall"]

    def test_explicit_tags_override_config(self):
        configure(
            hindsight_api_url="http://localhost:8888",
            tags=["config-tag"],
            recall_tags=["config-recall"],
        )
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            toolkit = HindsightTools(
                bank_id="test",
                tags=["explicit-tag"],
                recall_tags=["explicit-recall"],
            )
        client = mock_cls.return_value
        client.retain.return_value = None
        client.recall.return_value = _mock_recall_response(["fact"])
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")
        assert client.retain.call_args[1]["tags"] == ["explicit-tag"]

        toolkit.recall_memory(ctx, "q")
        assert client.recall.call_args[1]["tags"] == ["explicit-recall"]

    def test_api_key_passed_to_client(self):
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            HindsightTools(
                bank_id="test",
                hindsight_api_url="http://localhost:8888",
                api_key="secret",
            )
            mock_cls.assert_called_once_with(
                base_url="http://localhost:8888",
                timeout=30.0,
                api_key="secret",
            )


# ---------------------------------------------------------------------------
# Bank ID resolution
# ---------------------------------------------------------------------------


class TestBankIdResolution:
    def test_static_bank_id(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="my-bank", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        assert client.recall.call_args[1]["bank_id"] == "my-bank"

    def test_bank_id_from_run_context_user_id(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context(user_id="user-456")

        toolkit.recall_memory(ctx, "query")

        assert client.recall.call_args[1]["bank_id"] == "user-456"

    def test_custom_bank_resolver(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        resolver = MagicMock(return_value="resolved-bank")
        toolkit = HindsightTools(bank_resolver=resolver, client=client)
        ctx = _mock_run_context(user_id="user-789")

        toolkit.recall_memory(ctx, "query")

        resolver.assert_called_once_with(ctx)
        assert client.recall.call_args[1]["bank_id"] == "resolved-bank"

    def test_bank_resolver_takes_priority_over_bank_id(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        resolver = MagicMock(return_value="resolver-wins")
        toolkit = HindsightTools(
            bank_id="static-bank",
            bank_resolver=resolver,
            client=client,
        )
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        assert client.recall.call_args[1]["bank_id"] == "resolver-wins"

    def test_static_bank_id_takes_priority_over_user_id(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="static-bank", client=client)
        ctx = _mock_run_context(user_id="user-id-ignored")

        toolkit.recall_memory(ctx, "query")

        assert client.recall.call_args[1]["bank_id"] == "static-bank"

    def test_missing_bank_id_raises_error(self):
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.recall_memory(ctx, "query")

    def test_missing_bank_id_no_user_id_attr(self):
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = MagicMock(spec=[])  # No attributes at all

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.recall_memory(ctx, "query")

    def test_bank_resolver_error_propagates(self):
        client = _mock_client()

        def bad_resolver(ctx):
            raise ValueError("resolver broke")

        toolkit = HindsightTools(bank_resolver=bad_resolver, client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="resolver broke"):
            toolkit.retain_memory(ctx, "content")

    def test_bank_id_consistent_across_tools(self):
        """All tools resolve the same bank_id from the same context."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context(user_id="shared-user")

        toolkit.retain_memory(ctx, "content")
        toolkit.recall_memory(ctx, "query")
        toolkit.reflect_on_memory(ctx, "question")

        assert client.retain.call_args[1]["bank_id"] == "shared-user"
        assert client.recall.call_args[1]["bank_id"] == "shared-user"
        assert client.reflect.call_args[1]["bank_id"] == "shared-user"


# ---------------------------------------------------------------------------
# Bank auto-creation (_ensure_bank)
# ---------------------------------------------------------------------------


class TestEnsureBank:
    def test_creates_bank_on_first_tool_use(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="new-bank", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")

        client.create_bank.assert_called_once_with(bank_id="new-bank", name="new-bank")

    def test_does_not_recreate_bank(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "first")
        toolkit.retain_memory(ctx, "second")

        assert client.create_bank.call_count == 1

    def test_bank_creation_failure_is_swallowed(self):
        client = _mock_client()
        client.create_bank.side_effect = RuntimeError("bank exists")
        toolkit = HindsightTools(bank_id="existing-bank", client=client)
        ctx = _mock_run_context()

        # Should not raise — bank creation failure is tolerated
        result = toolkit.retain_memory(ctx, "content")
        assert result == "Memory stored successfully."

    def test_bank_creation_failure_marks_as_created(self):
        """After a bank creation failure, it shouldn't retry."""
        client = _mock_client()
        client.create_bank.side_effect = RuntimeError("conflict")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "first")
        toolkit.retain_memory(ctx, "second")

        assert client.create_bank.call_count == 1

    def test_different_bank_ids_created_separately(self):
        """When bank_resolver returns different IDs, each is created once."""
        client = _mock_client()

        def resolver(ctx):
            return f"bank-{ctx.user_id}"

        toolkit = HindsightTools(bank_resolver=resolver, client=client)

        toolkit.retain_memory(_mock_run_context(user_id="alice"), "content")
        toolkit.retain_memory(_mock_run_context(user_id="bob"), "content")
        toolkit.retain_memory(_mock_run_context(user_id="alice"), "more")

        assert client.create_bank.call_count == 2
        bank_ids = [c[1]["bank_id"] for c in client.create_bank.call_args_list]
        assert "bank-alice" in bank_ids
        assert "bank-bob" in bank_ids

    def test_recall_does_not_create_bank(self):
        """Recall doesn't call _ensure_bank (only retain does)."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        client.create_bank.assert_not_called()

    def test_reflect_does_not_create_bank(self):
        """Reflect doesn't call _ensure_bank (only retain does)."""
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "query")

        client.create_bank.assert_not_called()


# ---------------------------------------------------------------------------
# Retain tool
# ---------------------------------------------------------------------------


class TestRetainTool:
    def test_retain_stores_memory(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test-bank", client=client)
        ctx = _mock_run_context()

        result = toolkit.retain_memory(ctx, "The user likes Python")

        assert result == "Memory stored successfully."
        client.retain.assert_called_once_with(bank_id="test-bank", content="The user likes Python")

    def test_retain_passes_tags(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test-bank", client=client, tags=["source:chat"])
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "some content")

        call_kwargs = client.retain.call_args[1]
        assert call_kwargs["tags"] == ["source:chat"]

    def test_retain_no_tags_key_when_none(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")

        call_kwargs = client.retain.call_args[1]
        assert "tags" not in call_kwargs

    def test_retain_with_empty_content(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.retain_memory(ctx, "")

        assert result == "Memory stored successfully."
        client.retain.assert_called_once_with(bank_id="test", content="")

    def test_retain_raises_hindsight_error(self):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("connection refused")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Retain failed"):
            toolkit.retain_memory(ctx, "content")

    def test_retain_preserves_hindsight_error(self):
        """HindsightError from bank resolution is not double-wrapped."""
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context()  # No user_id, no bank_id

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.retain_memory(ctx, "content")

    def test_retain_error_chains_original_exception(self):
        client = _mock_client()
        original = RuntimeError("original error")
        client.retain.side_effect = original
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError) as exc_info:
            toolkit.retain_memory(ctx, "content")

        assert exc_info.value.__cause__ is original

    def test_retain_with_bank_resolver(self):
        client = _mock_client()
        resolver = MagicMock(return_value="resolved")
        toolkit = HindsightTools(bank_resolver=resolver, client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")

        assert client.retain.call_args[1]["bank_id"] == "resolved"


# ---------------------------------------------------------------------------
# Recall tool
# ---------------------------------------------------------------------------


class TestRecallTool:
    def test_recall_returns_numbered_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["User likes Python", "User is in NYC"])
        toolkit = HindsightTools(bank_id="test-bank", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "user preferences")

        assert "1. User likes Python" in result
        assert "2. User is in NYC" in result

    def test_recall_single_result(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["only fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "query")

        assert result == "1. only fact"

    def test_recall_many_results(self):
        client = _mock_client()
        facts = [f"fact {i}" for i in range(1, 11)]
        client.recall.return_value = _mock_recall_response(facts)
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "query")

        for i in range(1, 11):
            assert f"{i}. fact {i}" in result

    def test_recall_empty_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "anything")

        assert result == "No relevant memories found."

    def test_recall_none_results(self):
        client = _mock_client()
        response = MagicMock()
        response.results = None
        client.recall.return_value = response
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "anything")

        assert result == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client, budget="high", max_tokens=2048)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_default_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["budget"] == "mid"
        assert call_kwargs["max_tokens"] == 4096

    def test_recall_passes_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user"],
            recall_tags_match="all",
        )
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_no_tags_key_when_none(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        call_kwargs = client.recall.call_args[1]
        assert "tags" not in call_kwargs
        assert "tags_match" not in call_kwargs

    def test_recall_raises_hindsight_error(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("timeout")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Recall failed"):
            toolkit.recall_memory(ctx, "query")

    def test_recall_preserves_hindsight_error(self):
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.recall_memory(ctx, "query")

    def test_recall_error_chains_original_exception(self):
        client = _mock_client()
        original = ConnectionError("network down")
        client.recall.side_effect = original
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError) as exc_info:
            toolkit.recall_memory(ctx, "query")

        assert exc_info.value.__cause__ is original

    def test_recall_passes_query(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "specific query text")

        assert client.recall.call_args[1]["query"] == "specific query text"


# ---------------------------------------------------------------------------
# Reflect tool
# ---------------------------------------------------------------------------


class TestReflectTool:
    def test_reflect_returns_text(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response(
            "The user is a Python developer who prefers functional patterns."
        )
        toolkit = HindsightTools(bank_id="test-bank", client=client)
        ctx = _mock_run_context()

        result = toolkit.reflect_on_memory(ctx, "What do you know about the user?")

        assert result == "The user is a Python developer who prefers functional patterns."

    def test_reflect_empty_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.reflect_on_memory(ctx, "anything")

        assert result == "No relevant memories found."

    def test_reflect_none_text_returns_fallback(self):
        client = _mock_client()
        response = MagicMock()
        response.text = None
        client.reflect.return_value = response
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.reflect_on_memory(ctx, "anything")

        assert result == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="test", client=client, budget="high")
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "query")

        call_kwargs = client.reflect.call_args[1]
        assert call_kwargs["budget"] == "high"

    def test_reflect_default_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "query")

        assert client.reflect.call_args[1]["budget"] == "mid"

    def test_reflect_passes_query(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "What is the user's favorite color?")

        assert client.reflect.call_args[1]["query"] == "What is the user's favorite color?"

    def test_reflect_passes_bank_id(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="my-bank", client=client)
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "query")

        assert client.reflect.call_args[1]["bank_id"] == "my-bank"

    def test_reflect_raises_hindsight_error(self):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("timeout")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Reflect failed"):
            toolkit.reflect_on_memory(ctx, "query")

    def test_reflect_preserves_hindsight_error(self):
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.reflect_on_memory(ctx, "query")

    def test_reflect_error_chains_original_exception(self):
        client = _mock_client()
        original = TimeoutError("timed out")
        client.reflect.side_effect = original
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError) as exc_info:
            toolkit.reflect_on_memory(ctx, "query")

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# memory_instructions
# ---------------------------------------------------------------------------


class TestMemoryInstructions:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_returns_formatted_memories(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["Likes Python", "Lives in NYC", "Prefers dark mode"])

        result = memory_instructions(bank_id="test-bank", client=client)

        assert "Relevant memories:" in result
        assert "1. Likes Python" in result
        assert "2. Lives in NYC" in result
        assert "3. Prefers dark mode" in result

    def test_returns_string_type(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        result = memory_instructions(bank_id="test", client=client)

        assert isinstance(result, str)

    def test_respects_max_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2", "fact3", "fact4", "fact5"])

        result = memory_instructions(bank_id="test", client=client, max_results=2)

        assert "1. fact1" in result
        assert "2. fact2" in result
        assert "3." not in result

    def test_max_results_larger_than_available(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2"])

        result = memory_instructions(bank_id="test", client=client, max_results=10)

        assert "1. fact1" in result
        assert "2. fact2" in result

    def test_custom_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        result = memory_instructions(bank_id="test", client=client, prefix="Memory context:\n")

        assert result.startswith("Memory context:")

    def test_empty_results_returns_empty_string(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])

        result = memory_instructions(bank_id="test", client=client)

        assert result == ""

    def test_none_results_returns_empty_string(self):
        client = _mock_client()
        response = MagicMock()
        response.results = None
        client.recall.return_value = response

        result = memory_instructions(bank_id="test", client=client)

        assert result == ""

    def test_error_returns_empty_string(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("connection error")

        result = memory_instructions(bank_id="test", client=client)

        assert result == ""

    def test_error_does_not_raise(self):
        """Errors should be silently swallowed, never propagated."""
        client = _mock_client()
        client.recall.side_effect = ConnectionError("unreachable")

        # Should not raise
        result = memory_instructions(bank_id="test", client=client)
        assert result == ""

    def test_passes_query_and_budget(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(
            bank_id="test",
            client=client,
            query="user preferences and context",
            budget="high",
        )

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["query"] == "user preferences and context"
        assert call_kwargs["budget"] == "high"

    def test_passes_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="test", client=client, max_tokens=2048)

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    def test_default_parameters(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="test", client=client)

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["query"] == "relevant context about the user"
        assert call_kwargs["budget"] == "low"
        assert call_kwargs["max_tokens"] == 4096

    def test_passes_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(
            bank_id="test",
            client=client,
            tags=["scope:user"],
            tags_match="all",
        )

        call_kwargs = client.recall.call_args[1]
        assert call_kwargs["tags"] == ["scope:user"]
        assert call_kwargs["tags_match"] == "all"

    def test_no_tags_key_when_none(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="test", client=client)

        call_kwargs = client.recall.call_args[1]
        assert "tags" not in call_kwargs
        assert "tags_match" not in call_kwargs

    def test_raises_without_client_or_config(self):
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            memory_instructions(bank_id="test")

    def test_uses_global_config(self):
        configure(hindsight_api_url="http://localhost:8888")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_client = _mock_client()
            mock_client.recall.return_value = _mock_recall_response(["fact"])
            mock_cls.return_value = mock_client

            result = memory_instructions(bank_id="test")

            assert "1. fact" in result
            mock_cls.assert_called_once_with(base_url="http://localhost:8888", timeout=30.0)

    def test_uses_sync_recall(self):
        """memory_instructions should use sync client.recall, not arecall."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="test", client=client)

        client.recall.assert_called_once()
        # Should NOT call async variant
        assert not hasattr(client, "arecall") or not client.arecall.called


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_all_exports_importable(self):
        import hindsight_agno

        for name in hindsight_agno.__all__:
            assert hasattr(hindsight_agno, name), f"{name} in __all__ but not importable"

    def test_version(self):
        import hindsight_agno

        assert hindsight_agno.__version__ == "0.1.0"

    def test_hindsight_error_importable_from_top_level(self):
        from hindsight_agno import HindsightError

        assert issubclass(HindsightError, Exception)

    def test_config_class_importable(self):
        from hindsight_agno import HindsightAgnoConfig

        config = HindsightAgnoConfig()
        assert config.budget == "mid"


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class TestHindsightError:
    def test_is_exception(self):
        assert issubclass(HindsightError, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(HindsightError, match="test error"):
            raise HindsightError("test error")

    def test_str_representation(self):
        err = HindsightError("something went wrong")
        assert str(err) == "something went wrong"


# ---------------------------------------------------------------------------
# Multiple toolkit instances
# ---------------------------------------------------------------------------


class TestMultipleToolkits:
    def test_separate_created_banks_sets(self):
        """Each toolkit instance tracks its own created banks."""
        client = _mock_client()
        toolkit1 = HindsightTools(bank_id="bank-1", client=client)
        toolkit2 = HindsightTools(bank_id="bank-2", client=client)
        ctx = _mock_run_context()

        toolkit1.retain_memory(ctx, "content")
        toolkit2.retain_memory(ctx, "content")

        assert client.create_bank.call_count == 2

    def test_separate_config_per_toolkit(self):
        """Each toolkit can have different settings."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        toolkit_low = HindsightTools(bank_id="test", client=client, budget="low")
        toolkit_high = HindsightTools(bank_id="test", client=client, budget="high")
        ctx = _mock_run_context()

        toolkit_low.recall_memory(ctx, "query")
        low_budget = client.recall.call_args[1]["budget"]

        toolkit_high.recall_memory(ctx, "query")
        high_budget = client.recall.call_args[1]["budget"]

        assert low_budget == "low"
        assert high_budget == "high"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_retain_logs_error(self, caplog):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("boom")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with caplog.at_level(logging.ERROR, logger="hindsight_agno.tools"):
            with pytest.raises(HindsightError):
                toolkit.retain_memory(ctx, "content")

        assert "Retain failed: boom" in caplog.text

    def test_recall_logs_error(self, caplog):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("timeout")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with caplog.at_level(logging.ERROR, logger="hindsight_agno.tools"):
            with pytest.raises(HindsightError):
                toolkit.recall_memory(ctx, "query")

        assert "Recall failed: timeout" in caplog.text

    def test_reflect_logs_error(self, caplog):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("service down")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with caplog.at_level(logging.ERROR, logger="hindsight_agno.tools"):
            with pytest.raises(HindsightError):
                toolkit.reflect_on_memory(ctx, "query")

        assert "Reflect failed: service down" in caplog.text

    def test_hindsight_error_not_logged(self, caplog):
        """HindsightError (e.g. missing bank_id) should not log — it's re-raised directly."""
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context()  # No user_id

        with caplog.at_level(logging.ERROR, logger="hindsight_agno.tools"):
            with pytest.raises(HindsightError, match="No bank_id"):
                toolkit.retain_memory(ctx, "content")

        assert "Retain failed" not in caplog.text


# ---------------------------------------------------------------------------
# Edge cases: falsy and special values
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_bank_id_is_falsy(self):
        """bank_id='' is falsy, so resolution falls through to user_id."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="", client=client)
        ctx = _mock_run_context(user_id="user-from-ctx")

        # "" is not None, so _bank_id check passes, but "" is also the bank_id
        # Actually: `if self._bank_id is not None:` — "" is not None, so "" is used
        toolkit.recall_memory(ctx, "query")
        assert client.recall.call_args[1]["bank_id"] == ""

    def test_empty_string_user_id_is_falsy(self):
        """user_id='' is falsy, so resolution raises error."""
        client = _mock_client()
        toolkit = HindsightTools(client=client)
        ctx = _mock_run_context(user_id="")

        with pytest.raises(HindsightError, match="No bank_id available"):
            toolkit.recall_memory(ctx, "query")

    def test_unicode_content_in_retain(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.retain_memory(ctx, "The user likes coffee and books")

        assert result == "Memory stored successfully."
        assert client.retain.call_args[1]["content"] == "The user likes coffee and books"

    def test_unicode_in_recall_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["Cafe du Monde"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "query")

        assert "Cafe du Monde" in result

    def test_multiline_content_in_retain(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()
        content = "Line 1\nLine 2\nLine 3"

        result = toolkit.retain_memory(ctx, content)

        assert result == "Memory stored successfully."
        assert client.retain.call_args[1]["content"] == content

    def test_newlines_in_recall_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact with\nnewline", "normal fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "query")

        assert "1. fact with\nnewline" in result
        assert "2. normal fact" in result

    def test_special_chars_in_bank_id(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="org/user-123_v2", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        assert client.recall.call_args[1]["bank_id"] == "org/user-123_v2"

    def test_empty_tags_list_not_sent(self):
        """tags=[] is falsy, so tags should not be sent to retain."""
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client, tags=[])
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")

        assert "tags" not in client.retain.call_args[1]

    def test_empty_recall_tags_list_not_sent(self):
        """recall_tags=[] is falsy, so tags should not be sent to recall."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client, recall_tags=[])
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        assert "tags" not in client.recall.call_args[1]

    def test_very_long_content(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()
        content = "x" * 100_000

        result = toolkit.retain_memory(ctx, content)

        assert result == "Memory stored successfully."
        assert len(client.retain.call_args[1]["content"]) == 100_000

    def test_recall_with_empty_query(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "")

        assert client.recall.call_args[1]["query"] == ""

    def test_reflect_with_empty_query(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        toolkit.reflect_on_memory(ctx, "")

        assert client.reflect.call_args[1]["query"] == ""


# ---------------------------------------------------------------------------
# Error type variations
# ---------------------------------------------------------------------------


class TestErrorTypes:
    """Verify various exception types are properly wrapped."""

    def test_retain_wraps_type_error(self):
        client = _mock_client()
        client.retain.side_effect = TypeError("bad type")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Retain failed.*bad type"):
            toolkit.retain_memory(ctx, "content")

    def test_recall_wraps_key_error(self):
        client = _mock_client()
        client.recall.side_effect = KeyError("missing key")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Recall failed"):
            toolkit.recall_memory(ctx, "query")

    def test_reflect_wraps_os_error(self):
        client = _mock_client()
        client.reflect.side_effect = OSError("disk full")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Reflect failed.*disk full"):
            toolkit.reflect_on_memory(ctx, "query")

    def test_retain_wraps_value_error(self):
        client = _mock_client()
        client.retain.side_effect = ValueError("invalid input")
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        with pytest.raises(HindsightError, match="Retain failed.*invalid input"):
            toolkit.retain_memory(ctx, "content")


# ---------------------------------------------------------------------------
# Tool docstrings (these become LLM tool descriptions)
# ---------------------------------------------------------------------------


class TestToolDocstrings:
    """Tool docstrings are critical — Agno exposes them to the LLM."""

    def test_retain_has_docstring(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.retain_memory.__doc__ is not None
        assert "Store" in toolkit.retain_memory.__doc__
        assert "memory" in toolkit.retain_memory.__doc__.lower()

    def test_recall_has_docstring(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.recall_memory.__doc__ is not None
        assert "Search" in toolkit.recall_memory.__doc__
        assert "memory" in toolkit.recall_memory.__doc__.lower()

    def test_reflect_has_docstring(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.reflect_on_memory.__doc__ is not None
        assert "Synthesize" in toolkit.reflect_on_memory.__doc__


# ---------------------------------------------------------------------------
# Toolkit integration with Agno's registration
# ---------------------------------------------------------------------------


class TestToolkitIntegration:
    def test_get_functions_returns_registered_tools(self):
        """Verify tools are accessible via Toolkit.get_functions()."""
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        functions = toolkit.get_functions()
        assert "retain_memory" in functions
        assert "recall_memory" in functions
        assert "reflect_on_memory" in functions

    def test_get_functions_respects_enable_flags(self):
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            enable_retain=False,
            enable_reflect=False,
        )
        functions = toolkit.get_functions()
        assert "retain_memory" not in functions
        assert "recall_memory" in functions
        assert "reflect_on_memory" not in functions

    def test_instructions_are_set(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert toolkit.instructions is not None
        assert len(toolkit.instructions) > 0

    def test_instructions_mention_all_tools(self):
        client = _mock_client()
        toolkit = HindsightTools(bank_id="test", client=client)
        assert "retain_memory" in toolkit.instructions
        assert "recall_memory" in toolkit.instructions
        assert "reflect_on_memory" in toolkit.instructions


# ---------------------------------------------------------------------------
# Bank creation across tool types
# ---------------------------------------------------------------------------


class TestBankCreationCrossTool:
    def test_retain_then_recall_creates_bank_once(self):
        """Bank created by retain is reused by recall (no re-creation)."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="shared-bank", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")
        toolkit.recall_memory(ctx, "query")

        # create_bank only called by retain (once)
        client.create_bank.assert_called_once()

    def test_retain_then_reflect_creates_bank_once(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("answer")
        toolkit = HindsightTools(bank_id="shared-bank", client=client)
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")
        toolkit.reflect_on_memory(ctx, "query")

        client.create_bank.assert_called_once()


# ---------------------------------------------------------------------------
# memory_instructions edge cases
# ---------------------------------------------------------------------------


class TestMemoryInstructionsEdgeCases:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_max_results_zero(self):
        """max_results=0 → results[:0] is empty → returns ''."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2"])

        result = memory_instructions(bank_id="test", client=client, max_results=0)

        assert result == ""

    def test_max_results_one(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["only-fact", "extra"])

        result = memory_instructions(bank_id="test", client=client, max_results=1)

        assert "1. only-fact" in result
        assert "extra" not in result

    def test_empty_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        result = memory_instructions(bank_id="test", client=client, prefix="")

        # Empty prefix still appears as first line (empty string before \n join)
        assert result.startswith("\n") or result == "\n1. fact"

    def test_prefix_without_newline(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        result = memory_instructions(bank_id="test", client=client, prefix="Context: ")

        assert result == "Context: \n1. fact"

    def test_bank_id_passed_to_recall(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="specific-bank", client=client)

        assert client.recall.call_args[1]["bank_id"] == "specific-bank"

    def test_empty_tags_list_not_sent(self):
        """tags=[] is falsy, so tags should not be included in recall kwargs."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        memory_instructions(bank_id="test", client=client, tags=[])

        assert "tags" not in client.recall.call_args[1]

    def test_multiple_calls_with_same_client(self):
        """memory_instructions can be called multiple times with the same client."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        r1 = memory_instructions(bank_id="bank-1", client=client)
        r2 = memory_instructions(bank_id="bank-2", client=client)

        assert client.recall.call_count == 2
        assert r1 == r2  # Same response mock

    def test_client_resolution_error_propagates(self):
        """_resolve_client error during memory_instructions IS raised (not swallowed)."""
        with pytest.raises(HindsightError, match="No Hindsight API URL"):
            memory_instructions(bank_id="test")

    def test_recall_error_swallowed_not_raised(self):
        """Once client is resolved, recall errors are swallowed (returns '')."""
        client = _mock_client()
        client.recall.side_effect = Exception("any error")

        result = memory_instructions(bank_id="test", client=client)

        assert result == ""


# ---------------------------------------------------------------------------
# _resolve_client edge cases
# ---------------------------------------------------------------------------


class TestResolveClientEdgeCases:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_empty_string_api_key_not_passed(self):
        """api_key='' is falsy, so it should not be passed to Hindsight."""
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", "")
            call_kwargs = mock_cls.call_args[1]
            assert "api_key" not in call_kwargs

    def test_none_api_key_not_passed(self):
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", None)
            call_kwargs = mock_cls.call_args[1]
            assert "api_key" not in call_kwargs

    def test_whitespace_api_key_is_passed(self):
        """Non-empty whitespace string IS truthy, so it gets passed."""
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, "http://localhost:8888", "  ")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "  "

    def test_config_api_key_used_when_no_explicit(self):
        configure(hindsight_api_url="http://config:8888", api_key="config-key")
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            assert mock_cls.call_args[1]["api_key"] == "config-key"

    def test_config_none_api_key_not_passed(self):
        configure(hindsight_api_url="http://config:8888", api_key=None)
        with patch("hindsight_agno.tools.Hindsight") as mock_cls:
            mock_cls.return_value = _mock_client()
            _resolve_client(None, None, None)
            assert "api_key" not in mock_cls.call_args[1]


# ---------------------------------------------------------------------------
# Config interaction with toolkit
# ---------------------------------------------------------------------------


class TestConfigToolkitInteraction:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_config_set_after_toolkit_creation_not_used(self):
        """Config is captured at construction time, not at tool call time."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(bank_id="test", client=client, tags=["original"])
        ctx = _mock_run_context()

        # Change config after construction
        configure(hindsight_api_url="http://new:8888", tags=["updated"])

        toolkit.retain_memory(ctx, "content")

        # Should use the original tags, not updated
        assert client.retain.call_args[1]["tags"] == ["original"]

    def test_toolkit_without_config_then_configure(self):
        """Creating toolkit with explicit client doesn't require config."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])

        # No config set
        toolkit = HindsightTools(bank_id="test", client=client)

        # Config set after — doesn't affect existing toolkit
        configure(hindsight_api_url="http://new:8888", budget="high")

        toolkit.recall_memory(_mock_run_context(), "query")
        assert client.recall.call_args[1]["budget"] == "mid"  # Not "high"

    def test_multiple_tags(self):
        """Multiple tags are passed as a list."""
        client = _mock_client()
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            tags=["source:chat", "env:prod", "version:2"],
        )
        ctx = _mock_run_context()

        toolkit.retain_memory(ctx, "content")

        tags = client.retain.call_args[1]["tags"]
        assert tags == ["source:chat", "env:prod", "version:2"]

    def test_multiple_recall_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact"])
        toolkit = HindsightTools(
            bank_id="test",
            client=client,
            recall_tags=["scope:user", "type:preference"],
            recall_tags_match="all_strict",
        )
        ctx = _mock_run_context()

        toolkit.recall_memory(ctx, "query")

        kwargs = client.recall.call_args[1]
        assert kwargs["tags"] == ["scope:user", "type:preference"]
        assert kwargs["tags_match"] == "all_strict"


# ---------------------------------------------------------------------------
# Recall output formatting
# ---------------------------------------------------------------------------


class TestRecallFormatting:
    def test_result_numbering_starts_at_one(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["alpha"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "q")

        assert result.startswith("1.")
        assert "0." not in result

    def test_results_separated_by_newlines(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["a", "b", "c"])
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "q")

        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "1. a"
        assert lines[1] == "2. b"
        assert lines[2] == "3. c"

    def test_double_digit_numbering(self):
        client = _mock_client()
        facts = [f"fact-{i}" for i in range(1, 13)]
        client.recall.return_value = _mock_recall_response(facts)
        toolkit = HindsightTools(bank_id="test", client=client)
        ctx = _mock_run_context()

        result = toolkit.recall_memory(ctx, "q")

        assert "10. fact-10" in result
        assert "11. fact-11" in result
        assert "12. fact-12" in result


# ---------------------------------------------------------------------------
# memory_instructions output formatting
# ---------------------------------------------------------------------------


class TestMemoryInstructionsFormatting:
    def test_default_prefix_includes_trailing_newline(self):
        """Default prefix is 'Relevant memories:\\n', so join adds a blank line."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2"])

        result = memory_instructions(bank_id="test", client=client)

        # prefix="Relevant memories:\n" joined with "\n" produces:
        # "Relevant memories:\n\n1. fact1\n2. fact2"
        assert result == "Relevant memories:\n\n1. fact1\n2. fact2"

    def test_single_result_with_default_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["single"])

        result = memory_instructions(bank_id="test", client=client)

        assert result == "Relevant memories:\n\n1. single"

    def test_clean_prefix_no_double_newline(self):
        """Using a prefix without trailing newline avoids the blank line."""
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact1", "fact2"])

        result = memory_instructions(bank_id="test", client=client, prefix="Relevant memories:")

        assert result == "Relevant memories:\n1. fact1\n2. fact2"


# ---------------------------------------------------------------------------
# Keyword-only constructor enforcement
# ---------------------------------------------------------------------------


class TestConstructorAPI:
    def test_hindsight_tools_requires_keyword_args(self):
        """All HindsightTools args are keyword-only (due to *)."""
        client = _mock_client()
        with pytest.raises(TypeError):
            HindsightTools("test", client)  # type: ignore[misc]

    def test_memory_instructions_requires_keyword_args(self):
        """All memory_instructions args are keyword-only (due to *)."""
        client = _mock_client()
        with pytest.raises(TypeError):
            memory_instructions("test", client)  # type: ignore[misc]
