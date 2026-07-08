"""Unit tests for AuditLogger's optional per-bank gate.

These are pure (no DB): they exercise the decision logic in ``should_log``
directly. The point is to pin the backward-compatibility contract — with no
gate set, behavior equals the original ``is_enabled(action)`` — and the
new per-bank behavior when a gate IS set.
"""

from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig
from hindsight_api.engine.audit import AuditLogger, audit_context


class TestAuditEnabledIsConfigurable:
    """The per-bank ``audit_enabled`` field must be overridable per bank so
    the bank-config API accepts it (unknown fields are rejected)."""

    def test_audit_enabled_in_configurable_fields(self):
        assert "audit_enabled" in HindsightConfig.get_configurable_fields()

    def test_default_true(self):
        assert HindsightConfig.from_env().audit_enabled is True


def _logger(enabled: bool, allowed_actions=None) -> AuditLogger:
    return AuditLogger(
        pool_getter=lambda: None,
        schema_getter=lambda: "public",
        enabled=enabled,
        allowed_actions=allowed_actions or [],
    )


class TestNoGateBackwardCompat:
    """No bank gate set → should_log == is_enabled (the original behavior)."""

    @pytest.mark.asyncio
    async def test_enabled_no_gate_logs_all_banks(self):
        al = _logger(enabled=True)
        assert al.is_enabled("retain") is True
        assert await al.should_log("retain", "bank-a") is True
        assert await al.should_log("retain", "bank-b") is True
        # bank_id is irrelevant without a gate — even None logs.
        assert await al.should_log("retain", None) is True

    @pytest.mark.asyncio
    async def test_disabled_no_gate_logs_nothing(self):
        al = _logger(enabled=False)
        assert await al.should_log("retain", "bank-a") is False

    @pytest.mark.asyncio
    async def test_action_allowlist_still_applies(self):
        al = _logger(enabled=True, allowed_actions=["retain"])
        assert await al.should_log("retain", "bank-a") is True
        assert await al.should_log("recall", "bank-a") is False


class TestWithBankGate:
    """A gate is set → global gate AND the per-bank predicate."""

    @pytest.mark.asyncio
    async def test_gate_selects_specific_banks(self):
        al = _logger(enabled=True)
        allowed = {"bank-on"}

        async def gate(bank_id):
            return bank_id in allowed

        al.set_bank_gate(gate)
        assert await al.should_log("retain", "bank-on") is True
        assert await al.should_log("retain", "bank-off") is False

    @pytest.mark.asyncio
    async def test_global_off_short_circuits_gate(self):
        al = _logger(enabled=False)
        called = False

        async def gate(bank_id):
            nonlocal called
            called = True
            return True

        al.set_bank_gate(gate)
        # Global gate is off, so the (potentially expensive) per-bank gate is
        # never awaited.
        assert await al.should_log("retain", "bank-on") is False
        assert called is False

    @pytest.mark.asyncio
    async def test_gate_failure_fails_closed(self):
        al = _logger(enabled=True)

        async def gate(bank_id):
            raise RuntimeError("control plane unreachable")

        al.set_bank_gate(gate)
        # A broken gate must not enable auditing for a bank meant to be off.
        assert await al.should_log("retain", "bank-x") is False

    @pytest.mark.asyncio
    async def test_clearing_gate_restores_all_banks(self):
        al = _logger(enabled=True)

        async def gate(bank_id):
            return False

        al.set_bank_gate(gate)
        assert await al.should_log("retain", "bank-a") is False
        al.set_bank_gate(None)
        assert await al.should_log("retain", "bank-a") is True


class TestAuditContextHonorsGate:
    """audit_context is a separate write path (used by the retain engine). It
    must honor the per-bank gate too, or a gated-off bank leaks audit rows."""

    @pytest.mark.asyncio
    async def test_gated_off_bank_does_not_log(self):
        al = _logger(enabled=True)

        async def gate(bank_id):
            return False

        al.set_bank_gate(gate)
        with patch.object(al, "log_fire_and_forget") as write:
            async with audit_context(al, "retain", "http", bank_id="bank-off") as entry:
                entry.response = {"ok": True}
        write.assert_not_called()

    @pytest.mark.asyncio
    async def test_gated_on_bank_logs(self):
        al = _logger(enabled=True)

        async def gate(bank_id):
            return True

        al.set_bank_gate(gate)
        with patch.object(al, "log_fire_and_forget") as write:
            async with audit_context(al, "retain", "http", bank_id="bank-on") as entry:
                entry.response = {"ok": True}
        write.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_gate_logs_as_before(self):
        al = _logger(enabled=True)
        with patch.object(al, "log_fire_and_forget") as write:
            async with audit_context(al, "retain", "http", bank_id="bank-a") as entry:
                entry.response = {"ok": True}
        write.assert_called_once()
