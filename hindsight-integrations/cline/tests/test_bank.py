"""Bank ID derivation."""

from conftest import base_config, make_hook

from lib.bank import derive_bank_id


def test_static_bank_is_the_default():
    assert derive_bank_id(make_hook(), base_config()) == "cline"


def test_static_bank_respects_prefix():
    cfg = base_config(bankId="cline", bankIdPrefix="prod")
    assert derive_bank_id(make_hook(), cfg) == "prod-cline"


def test_dynamic_bank_uses_agent_and_project():
    cfg = base_config(dynamicBankId=True, dynamicBankGranularity=["agent", "project"])
    hook = make_hook(workspace="/home/user/myproject")
    assert derive_bank_id(hook, cfg) == "cline::myproject"


def test_dynamic_bank_session_is_task_id():
    cfg = base_config(dynamicBankId=True, dynamicBankGranularity=["session"])
    assert derive_bank_id(make_hook(task_id="task-42"), cfg) == "task-42"


def test_dynamic_bank_handles_missing_workspace():
    cfg = base_config(dynamicBankId=True, dynamicBankGranularity=["project"])
    assert derive_bank_id(make_hook(workspace=""), cfg) == "unknown"
