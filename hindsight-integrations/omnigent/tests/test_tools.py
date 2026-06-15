"""Unit tests for the Hindsight Omnigent tools.

The bulk of these tests drive the tool callables through ``_invoke_like_omnigent``
— a faithful replica of how Omnigent's ``LocalCallableTool.invoke`` calls a
``type: function`` tool (parse the LLM's JSON args into kwargs, call the
callable, coerce the return to a string). That exercises the real framework
contract without installing the (alpha, 3.12-only) ``omnigent`` package.

``TestRealOmnigentInvocation`` additionally loads the callables through the real
``omnigent`` machinery when it's importable, guarding against contract drift.
"""

import json
import logging
from unittest.mock import MagicMock

import pytest
from hindsight_omnigent import (
    OmnigentToolSpec,
    configure,
    memory_instructions,
    recall,
    reflect,
    retain,
    reset_config,
    tool_specs,
    tools_yaml,
)
from hindsight_omnigent.errors import HindsightError
from hindsight_omnigent.tools import _reset_created_banks

try:
    from omnigent.spec.types import LocalToolInfo
    from omnigent.tools.local_callable import load_local_callable_tools

    _HAS_OMNIGENT = True
except Exception:  # pragma: no cover - omnigent is an optional, 3.12-only dep
    _HAS_OMNIGENT = False


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _stringify(value):
    """Replica of ``omnigent.tools.local_callable._stringify``."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)


def _invoke_like_omnigent(fn, **arguments) -> str:
    """Replicate ``LocalCallableTool.invoke``: JSON args -> kwargs -> stringify."""
    kwargs = json.loads(json.dumps(arguments))  # round-trip as the LLM payload would
    return _stringify(fn(**kwargs))


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


class _Base:
    def setup_method(self):
        reset_config()
        _reset_created_banks()

    def teardown_method(self):
        reset_config()
        _reset_created_banks()


# ---------------------------------------------------------------------------
# Bank / connection resolution
# ---------------------------------------------------------------------------


class TestBankResolution(_Base):
    def test_retain_uses_configured_bank(self):
        client = _mock_client()
        configure(client=client, bank_id="alice")
        out = _invoke_like_omnigent(retain, content="hi")
        assert out == "Stored to long-term memory."
        client.retain.assert_called_once_with(bank_id="alice", content="hi")
        client.create_bank.assert_called_once_with(bank_id="alice", name="alice")

    def test_bank_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "from-env")
        client = _mock_client()
        configure(client=client)  # reads HINDSIGHT_BANK_ID
        _invoke_like_omnigent(retain, content="hi")
        assert client.retain.call_args[1]["bank_id"] == "from-env"

    def test_missing_bank_raises(self):
        configure(client=_mock_client())  # no bank_id, no env
        with pytest.raises(HindsightError, match="No Hindsight bank configured"):
            retain(content="hi")

    def test_bank_created_once(self):
        client = _mock_client()
        configure(client=client, bank_id="alice")
        retain(content="a")
        retain(content="b")
        client.create_bank.assert_called_once()


# ---------------------------------------------------------------------------
# Retain
# ---------------------------------------------------------------------------


class TestRetain(_Base):
    def test_retain_with_tags(self):
        client = _mock_client()
        configure(client=client, bank_id="alice", tags=["env:test"])
        retain(content="c")
        assert client.retain.call_args[1]["tags"] == ["env:test"]

    def test_retain_bank_already_exists(self):
        client = _mock_client()
        client.create_bank.side_effect = Exception("already exists")
        configure(client=client, bank_id="alice")
        assert retain(content="c") == "Stored to long-term memory."

    def test_retain_failure_raises_and_logs(self, caplog):
        client = _mock_client()
        client.retain.side_effect = RuntimeError("network error")
        configure(client=client, bank_id="alice")
        with caplog.at_level(logging.ERROR), pytest.raises(HindsightError, match="Retain failed"):
            retain(content="c")
        assert "Retain failed" in caplog.text


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


class TestRecall(_Base):
    def test_recall_returns_bullet_list(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f1", "f2"])
        configure(client=client, bank_id="alice")
        out = _invoke_like_omnigent(recall, query="q")
        assert out == "- f1\n- f2"

    def test_recall_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        configure(client=client, bank_id="alice")
        assert recall(query="q") == "No relevant memories found."

    def test_recall_none_results(self):
        client = _mock_client()
        response = MagicMock()
        response.results = None
        client.recall.return_value = response
        configure(client=client, bank_id="alice")
        assert recall(query="q") == "No relevant memories found."

    def test_recall_passes_budget_and_max_tokens(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        configure(client=client, bank_id="alice", budget="high", max_tokens=2048)
        recall(query="q")
        call = client.recall.call_args[1]
        assert call["budget"] == "high"
        assert call["max_tokens"] == 2048

    def test_recall_with_tags(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        configure(client=client, bank_id="alice", recall_tags=["scope:global"], recall_tags_match="all")
        recall(query="q")
        call = client.recall.call_args[1]
        assert call["tags"] == ["scope:global"]
        assert call["tags_match"] == "all"

    def test_recall_without_tags_omits_tag_kwargs(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        configure(client=client, bank_id="alice")
        recall(query="q")
        call = client.recall.call_args[1]
        assert "tags" not in call
        assert "tags_match" not in call

    def test_recall_does_not_create_bank(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["f"])
        configure(client=client, bank_id="alice")
        recall(query="q")
        client.create_bank.assert_not_called()

    def test_recall_failure_raises(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("network error")
        configure(client=client, bank_id="alice")
        with pytest.raises(HindsightError, match="Recall failed"):
            recall(query="q")


# ---------------------------------------------------------------------------
# Reflect
# ---------------------------------------------------------------------------


class TestReflect(_Base):
    def test_reflect_returns_answer(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("Synthesized answer")
        configure(client=client, bank_id="alice")
        assert _invoke_like_omnigent(reflect, query="q") == "Synthesized answer"

    def test_reflect_empty_text_returns_fallback(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("")
        configure(client=client, bank_id="alice")
        assert reflect(query="q") == "No relevant memories found."

    def test_reflect_passes_budget(self):
        client = _mock_client()
        client.reflect.return_value = _mock_reflect_response("a")
        configure(client=client, bank_id="alice", budget="high")
        reflect(query="q")
        assert client.reflect.call_args[1]["budget"] == "high"

    def test_reflect_failure_raises(self):
        client = _mock_client()
        client.reflect.side_effect = RuntimeError("network error")
        configure(client=client, bank_id="alice")
        with pytest.raises(HindsightError, match="Reflect failed"):
            reflect(query="q")


# ---------------------------------------------------------------------------
# memory_instructions
# ---------------------------------------------------------------------------


class TestMemoryInstructions(_Base):
    def test_formats_results_with_prefix(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["likes tea", "lives in NYC"])
        configure(client=client, bank_id="b")
        out = memory_instructions()
        assert out == "Relevant memories:\n\n1. likes tea\n2. lives in NYC"

    def test_caps_at_max_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["a", "b", "c", "d"])
        configure(client=client, bank_id="b")
        out = memory_instructions(max_results=2)
        assert "1. a" in out and "2. b" in out
        assert "c" not in out and "d" not in out

    def test_explicit_bank_overrides_config(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["x"])
        configure(client=client, bank_id="config-bank")
        memory_instructions(bank_id="explicit-bank")
        assert client.recall.call_args[1]["bank_id"] == "explicit-bank"

    def test_returns_empty_without_bank(self):
        configure(client=_mock_client())  # no bank
        assert memory_instructions() == ""

    def test_returns_empty_when_no_results(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response([])
        configure(client=client, bank_id="b")
        assert memory_instructions() == ""

    def test_returns_empty_on_failure(self):
        client = _mock_client()
        client.recall.side_effect = RuntimeError("boom")
        configure(client=client, bank_id="b")
        assert memory_instructions() == ""


# ---------------------------------------------------------------------------
# Agent-YAML helpers
# ---------------------------------------------------------------------------


class TestToolSpecs(_Base):
    def test_default_three_specs(self):
        specs = tool_specs()
        assert [s.name for s in specs] == [
            "hindsight_retain",
            "hindsight_recall",
            "hindsight_reflect",
        ]
        assert all(isinstance(s, OmnigentToolSpec) for s in specs)
        assert all(s.callable_path.startswith("hindsight_omnigent.tools.") for s in specs)

    def test_enable_flags(self):
        specs = tool_specs(enable_retain=False, enable_reflect=False)
        assert [s.name for s in specs] == ["hindsight_recall"]

    def test_tools_yaml_is_parseable_and_complete(self):
        text = tools_yaml(enable_reflect=False)
        assert text.startswith("tools:\n")
        # Each tool's parameters line is inline JSON (valid YAML) — load it back.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("parameters:"):
                payload = stripped[len("parameters:") :].strip()
                schema = json.loads(payload)
                assert schema["type"] == "object"
                assert schema["required"]
        assert "hindsight_omnigent.tools.retain" in text
        assert "hindsight_omnigent.tools.recall" in text
        assert "hindsight_omnigent.tools.reflect" not in text


# ---------------------------------------------------------------------------
# Real Omnigent invocation (guards against the function-tool contract drifting)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OMNIGENT, reason="omnigent not installed (3.12-only alpha)")
class TestRealOmnigentInvocation(_Base):
    def _info(self, name, path, field):
        return LocalToolInfo(
            name=name,
            path=path,
            language="omnigent-python-callable",
            parameters={
                "type": "object",
                "properties": {field: {"type": "string"}},
                "required": [field],
            },
        )

    def test_recall_through_local_callable_tool(self):
        client = _mock_client()
        client.recall.return_value = _mock_recall_response(["fact one", "fact two"])
        configure(client=client, bank_id="alice")

        tools = load_local_callable_tools([self._info("hindsight_recall", "hindsight_omnigent.tools.recall", "query")])
        assert len(tools) == 1
        tool = tools[0]
        assert tool.name() == "hindsight_recall"
        assert tool.get_schema()["function"]["name"] == "hindsight_recall"

        out = tool.invoke(json.dumps({"query": "what tech?"}), None)
        assert "fact one" in out and "fact two" in out
        assert client.recall.call_args[1]["bank_id"] == "alice"

    def test_retain_through_local_callable_tool(self):
        client = _mock_client()
        configure(client=client, bank_id="alice")

        tools = load_local_callable_tools(
            [self._info("hindsight_retain", "hindsight_omnigent.tools.retain", "content")]
        )
        out = tools[0].invoke(json.dumps({"content": "remember this"}), None)
        assert out == "Stored to long-term memory."
        client.retain.assert_called_once_with(bank_id="alice", content="remember this")
