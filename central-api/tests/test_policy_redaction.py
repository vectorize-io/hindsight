from app.policy.classifier import classify_sensitivity
from app.policy.redactor import redact_secrets
from app.policy.rules import PolicyDecision, Sensitivity, policy_decision, run_write_gate


def test_redactor_blocks_fake_aws_key():
    content, count, kinds = redact_secrets("key AKIAIOSFODNN7EXAMPLE here")
    assert count == 1
    assert "aws_access_key" in kinds
    assert "AKIAIOSFODNN7EXAMPLE" not in content
    assert "[REDACTED:aws_access_key]" in content


def test_redactor_blocks_password_kv():
    _, count, _ = redact_secrets("password: hunter2")
    assert count >= 1


def test_classifier_levels():
    assert classify_sensitivity("anything", ["openai_key"]) is Sensitivity.secret_blocked
    assert classify_sensitivity("my seed phrase ...", []) is Sensitivity.sensitive
    assert classify_sensitivity("production rollout", []) is Sensitivity.internal
    assert classify_sensitivity("hello world", []) is Sensitivity.public


def test_policy_decision_mapping():
    assert policy_decision(Sensitivity.secret_blocked) is PolicyDecision.reject
    assert policy_decision(Sensitivity.sensitive) is PolicyDecision.quarantine
    assert policy_decision(Sensitivity.public) is PolicyDecision.allow


def test_write_gate_rejects_secret_and_redacts():
    gate = run_write_gate("token=ghp_0123456789abcdef0123456789abcdef0123")
    assert gate.decision is PolicyDecision.reject
    assert gate.redactions >= 1
    assert "ghp_" not in gate.content


def test_write_gate_allows_benign():
    gate = run_write_gate("we use reciprocal rank fusion")
    assert gate.decision is PolicyDecision.allow
    assert gate.redactions == 0
    assert gate.sensitivity is Sensitivity.public
