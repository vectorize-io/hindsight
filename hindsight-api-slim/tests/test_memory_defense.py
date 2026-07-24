"""Memory Defense — the OSS regex extension end to end.

Sections:
  * policy parsing (unit)
  * regex screening (unit)
  * extension loading (unit)
  * extension-context wiring (unit)
  * bank config validation (DB)
  * retain: allow / redact / block / webhook (DB)
  * document-body scrubbing (DB)
"""

import json

import pytest

from hindsight_api.extensions.builtin.memory_defense_regex import MemoryDefenseRegexExtension
from hindsight_api.extensions.loader import ExtensionLoadError, load_extension
from hindsight_api.extensions.memory_defense import (
    DefenseAction,
    MemoryDefenseExtension,
    _fingerprint_value,
    apply_redaction,
    parse_policy,
)
from tests.conftest import enable_audit_default

# ---------------------------------------------------------------------------
# Policy parsing (unit)
# ---------------------------------------------------------------------------


def test_parse_minimal_policy() -> None:
    policy = parse_policy({"enabled": True})
    assert policy.enabled is True
    assert policy.rules == ()


def test_parse_policy_with_rule() -> None:
    policy = parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    assert {r.on for r in policy.rules} == {"sensitive_data"}
    assert policy.rules[0].action is DefenseAction.REDACT


def test_parse_policy_accepts_block_action() -> None:
    policy = parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]})
    assert policy.rules[0].action is DefenseAction.BLOCK


def test_parse_policy_rejects_invalid_action() -> None:
    # Use a valid ``on`` so the parser progresses to action validation.
    with pytest.raises(ValueError, match="action"):
        parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "lol"}]})


@pytest.mark.parametrize("on", [None, "", 123])
def test_parse_policy_rejects_empty_or_non_string_on(on: object) -> None:
    with pytest.raises(ValueError, match="invalid on"):
        parse_policy({"enabled": True, "rules": [{"on": on, "action": "block"}]})


@pytest.mark.parametrize(
    "detector",
    [
        "sensitive_data",
        "prompt_injection",
        "size_anomaly",
        "protected_keys",
        "detect_secrets",
        "base64_decode",
        "llm_screen",
        # An unknown future name passes too: the parser doesn't gate ``on``
        # against a fixed roster.
        "some_future_cloud_detector",
    ],
)
def test_parse_policy_accepts_any_detector_name(detector: str) -> None:
    """The parser accepts any non-empty detector name so cloud-shape policies
    pass through the OSS PATCH layer unchanged. The OSS regex extension only
    actually screens ``sensitive_data``; the rest are silent no-ops here and
    are dispatched by downstream extensions (e.g. hindsight-cloud)."""
    policy = parse_policy({"enabled": True, "rules": [{"on": detector, "action": "block"}]})
    assert len(policy.rules) == 1
    assert policy.rules[0].on == detector
    assert policy.rules[0].action is DefenseAction.BLOCK


def test_disabled_policy_is_inert() -> None:
    policy = parse_policy({"enabled": False, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    assert policy.enabled is False


def test_defense_action_string_round_trip() -> None:
    assert DefenseAction("redact") is DefenseAction.REDACT
    assert DefenseAction.BLOCK.value == "block"


# ---------------------------------------------------------------------------
# Fingerprinting (unit)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        # Length > 15 → first-4 + ellipsis + last-4.
        ("ghp_" + "A" * 36, "ghp_...AAAA"),
        ("AKIA" + "B" * 16, "AKIA...BBBB"),
        ("sk-ant-" + "Z" * 40, "sk-a...ZZZZ"),
        # Length 6–15 → first-2 + ellipsis + last-2.
        ("123-45-6789", "12...89"),
        ("xoxb-12345", "xo...45"),
        # Length < 6 → fully masked; we don't preview anything.
        ("abcde", "[redacted]"),
        ("", "[redacted]"),
    ],
)
def test_fingerprint_value_shape(value: str, expected: str) -> None:
    """_fingerprint_value never returns the raw value and uses length-aware
    bracketing so short matches don't leak material."""
    out = _fingerprint_value(value)
    assert out == expected
    if value:
        assert value not in out, f"raw value leaked into fingerprint: {out!r}"


def test_apply_redaction_hits_carry_fingerprinted_previews() -> None:
    """apply_redaction returns per-match fingerprinted previews — one entry
    per matched substring — with the raw secret nowhere present in the hits."""
    s1 = "ghp_" + "A" * 36
    s2 = "AKIA" + "B" * 16
    s3 = "123-45-6789"
    content = f"rotate {s1}, drop {s2}, also ssn {s3}"

    result = apply_redaction(content)

    # Same-shape labels still flow to matched_types (deduplicated).
    assert set(result.matched_types) >= {"github_token", "aws_access_key", "ssn_us"}

    # One hit per matched substring; raw secret never appears.
    by_detector = {h["detector"]: h["preview"] for h in result.hits}
    assert by_detector["github_token"] == "ghp_...AAAA"
    assert by_detector["aws_access_key"] == "AKIA...BBBB"
    assert by_detector["ssn_us"] == "12...89"
    for h in result.hits:
        assert s1 not in h["preview"]
        assert s2 not in h["preview"]
        assert s3 not in h["preview"]


def test_apply_redaction_multiple_hits_per_pattern() -> None:
    """Two matches of the same pattern produce two hits — receivers can count
    occurrences, not just types."""
    a = "ghp_" + "A" * 36
    b = "ghp_" + "B" * 36
    content = f"old {a} new {b}"
    result = apply_redaction(content)

    gh_hits = [h for h in result.hits if h["detector"] == "github_token"]
    assert len(gh_hits) == 2
    previews = {h["preview"] for h in gh_hits}
    assert previews == {"ghp_...AAAA", "ghp_...BBBB"}


# ---------------------------------------------------------------------------
# Regex screening (unit)
# ---------------------------------------------------------------------------


@pytest.fixture
def regex_defense() -> MemoryDefenseRegexExtension:
    return MemoryDefenseRegexExtension({})


@pytest.fixture
def redact_policy() -> dict:
    return {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}


@pytest.mark.asyncio
async def test_screen_allows_innocuous_content(regex_defense, redact_policy) -> None:
    decision = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b1",
        document_id="d1",
        content="The Q3 roadmap meeting is on Friday.",
        tags=["session:abc"],
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_screen_redacts_secret(regex_defense, redact_policy) -> None:
    secret = "ghp_" + "A" * 36
    decision = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b1",
        document_id="d1",
        content=f"rotate this token: {secret}",
        tags=[],
    )
    assert decision.action is DefenseAction.REDACT
    assert decision.redacted_content is not None
    assert secret not in decision.redacted_content
    assert "[REDACTED:github_token]" in decision.redacted_content
    assert "github_token" in decision.matched_types
    # The decision carries a per-match fingerprinted preview — never the raw
    # value — so SIEM receivers can correlate without the secret crossing
    # the wire.
    assert decision.hits, "OSS should populate at least one hit"
    hit = decision.hits[0]
    assert hit["detector"] == "github_token"
    assert hit["preview"] == "ghp_...AAAA"
    assert secret not in hit["preview"]


@pytest.mark.asyncio
async def test_screen_blocks_secret(regex_defense) -> None:
    """A sensitive_data rule with action=block returns BLOCK (no redacted content)."""
    policy = parse_policy({"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]})
    secret = "AKIA" + "A" * 16
    decision = await regex_defense.screen(
        policy=policy,
        bank_id="b1",
        document_id="d1",
        content=f"key={secret}",
        tags=[],
    )
    assert decision.action is DefenseAction.BLOCK
    assert decision.redacted_content is None
    assert "aws_access_key" in decision.matched_types


@pytest.mark.asyncio
async def test_screen_allows_when_no_sensitive_data_rule(regex_defense) -> None:
    policy = parse_policy({"enabled": True, "rules": []})
    decision = await regex_defense.screen(
        policy=policy, bank_id="b1", document_id="d1", content="ghp_" + "Z" * 36, tags=[]
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
async def test_screen_disabled_policy_is_inert(regex_defense) -> None:
    policy = parse_policy({"enabled": False, "rules": [{"on": "sensitive_data", "action": "redact"}]})
    decision = await regex_defense.screen(
        policy=policy, bank_id="b1", document_id="d1", content="ghp_" + "Z" * 36, tags=[]
    )
    assert decision.action is DefenseAction.ALLOW


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "ghp_" + "A" * 36,
        "sk-ant-" + "B" * 40,
        "sk-" + "C" * 30,
        "AKIA" + "D" * 16,
    ],
)
async def test_screen_redacts_known_patterns(payload: str, regex_defense, redact_policy) -> None:
    d = await regex_defense.screen(
        policy=parse_policy(redact_policy),
        bank_id="b",
        document_id="d",
        content=f"my key is {payload}",
        tags=[],
    )
    assert d.action is DefenseAction.REDACT, f"expected redact for {payload!r}, got {d.action}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        "The roadmap meeting is on Friday",
        "Product launch planning notes",
        "Reminder about Tuesday",
    ],
)
async def test_screen_allows_benign_payloads(payload: str, regex_defense, redact_policy) -> None:
    d = await regex_defense.screen(
        policy=parse_policy(redact_policy), bank_id="b", document_id="d", content=payload, tags=[]
    )
    assert d.action is DefenseAction.ALLOW


# ---------------------------------------------------------------------------
# Extension loading (unit)
# ---------------------------------------------------------------------------


def test_regex_is_default_when_no_env(monkeypatch) -> None:
    monkeypatch.delenv("HINDSIGHT_API_MEMORY_DEFENSE_EXTENSION", raising=False)
    ext = load_extension("MEMORY_DEFENSE", MemoryDefenseExtension) or MemoryDefenseRegexExtension({})
    assert isinstance(ext, MemoryDefenseRegexExtension)


def test_custom_extension_loaded_from_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "HINDSIGHT_API_MEMORY_DEFENSE_EXTENSION",
        "hindsight_api.extensions.builtin.memory_defense_regex:MemoryDefenseRegexExtension",
    )
    ext = load_extension("MEMORY_DEFENSE", MemoryDefenseExtension)
    assert isinstance(ext, MemoryDefenseRegexExtension)


def test_malformed_extension_path_raises(monkeypatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_MEMORY_DEFENSE_EXTENSION", "no_colon_here")
    with pytest.raises(ExtensionLoadError):
        load_extension("MEMORY_DEFENSE", MemoryDefenseExtension)


def test_non_subclass_extension_raises(monkeypatch) -> None:
    monkeypatch.setenv("HINDSIGHT_API_MEMORY_DEFENSE_EXTENSION", "builtins:dict")
    with pytest.raises(ExtensionLoadError):
        load_extension("MEMORY_DEFENSE", MemoryDefenseExtension)


# ---------------------------------------------------------------------------
# Extension-context wiring (unit)
# ---------------------------------------------------------------------------


def _make_minimal_engine():
    """Construct a MemoryEngine with minimal env config (no network/GPU).

    Uses the "none" LLM provider and a mocked embeddings model so __init__
    runs without external calls; the pool is never started (no DB access).
    """
    import os
    from unittest.mock import MagicMock, patch

    mock_embeddings = MagicMock()
    mock_embeddings.dimension = 384

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import MemoryEngine

    with patch.dict(
        os.environ,
        {
            "HINDSIGHT_API_LLM_PROVIDER": "none",
            "HINDSIGHT_API_LLM_MODEL": "none",
            "HINDSIGHT_API_LLM_API_KEY": "test-key",
        },
        clear=False,
    ):
        clear_config_cache()
        engine = MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=mock_embeddings)

    # Constructing the engine above repopulated the process-global config cache
    # from the patched env (provider="none" forces retain_extraction_mode="chunks").
    # Now that the patched env is gone, drop that cache so the leaked "none"/chunks
    # config does not bleed into other tests on this xdist worker — their retains
    # would silently skip entity extraction (0 unit_entities) and fail unrelated
    # assertions. The next get_config() rebuilds from the real env.
    clear_config_cache()
    return engine


def test_engine_memory_defense_shares_ext_ctx() -> None:
    """The defense extension's context is the engine's _ext_ctx, and webhook_manager
    starts None (it is wired in initialize())."""
    engine = _make_minimal_engine()
    assert engine._memory_defense._context is engine._ext_ctx
    assert engine._ext_ctx.webhook_manager is None


def test_engine_ext_ctx_current_schema_propagation() -> None:
    """Writing _ext_ctx.current_schema is visible through _memory_defense.context."""
    engine = _make_minimal_engine()
    engine._ext_ctx.current_schema = "tenant_x"
    assert engine._memory_defense.context.current_schema == "tenant_x"


# ---------------------------------------------------------------------------
# Bank config validation (DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_accepts_and_persists_policy(api_client) -> None:
    await api_client.put("/v1/default/banks/md-cfg-1", json={})
    r = await api_client.patch(
        "/v1/default/banks/md-cfg-1/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}
        },
    )
    assert r.status_code == 200, r.text

    r2 = await api_client.get("/v1/default/banks/md-cfg-1/config")
    assert r2.json()["config"]["memory_defense"]["enabled"] is True


@pytest.mark.asyncio
async def test_patch_rejects_invalid_action(api_client) -> None:
    await api_client.put("/v1/default/banks/md-cfg-2", json={})
    # Valid ``on`` so the parser reaches action validation.
    r = await api_client.patch(
        "/v1/default/banks/md-cfg-2/config",
        json={
            "updates": {
                "memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "delete_everything"}]}
            }
        },
    )
    assert r.status_code == 422, r.text
    assert "action" in str(r.json()["detail"]).lower()


@pytest.mark.asyncio
async def test_patch_accepts_cloud_only_detector(api_client) -> None:
    # A cloud-only detector the OSS extension doesn't implement still persists
    # through the PATCH layer (it's a silent no-op here, dispatched downstream).
    await api_client.put("/v1/default/banks/md-cfg-3", json={})
    r = await api_client.patch(
        "/v1/default/banks/md-cfg-3/config",
        json={
            "updates": {"memory_defense": {"enabled": True, "rules": [{"on": "prompt_injection", "action": "block"}]}}
        },
    )
    assert r.status_code == 200, r.text
    r2 = await api_client.get("/v1/default/banks/md-cfg-3/config")
    assert r2.json()["config"]["memory_defense"]["rules"][0]["on"] == "prompt_injection"


@pytest.mark.asyncio
async def test_patch_rejects_empty_detector(api_client) -> None:
    await api_client.put("/v1/default/banks/md-cfg-4", json={})
    r = await api_client.patch(
        "/v1/default/banks/md-cfg-4/config",
        json={"updates": {"memory_defense": {"enabled": True, "rules": [{"on": "", "action": "redact"}]}}},
    )
    assert r.status_code == 422, r.text
    assert "on" in str(r.json()["detail"]).lower()


# ---------------------------------------------------------------------------
# Retain: allow / redact / block / webhook (DB)
# ---------------------------------------------------------------------------

_REDACT_POLICY = {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "redact"}]}}


async def _set_policy(api_client, bank: str, updates: dict) -> None:
    r = await api_client.patch(f"/v1/default/banks/{bank}/config", json={"updates": updates})
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_retain_allows_clean_content(api_client) -> None:
    await api_client.put("/v1/default/banks/md-retain-1", json={})
    await _set_policy(api_client, "md-retain-1", _REDACT_POLICY)
    r = await api_client.post(
        "/v1/default/banks/md-retain-1/memories",
        json={"items": [{"content": "the meeting is friday"}]},
    )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_retain_stores_redacted_text(api_client, memory) -> None:
    await api_client.put("/v1/default/banks/md-retain-2", json={})
    await _set_policy(api_client, "md-retain-2", _REDACT_POLICY)
    secret = "ghp_" + "A" * 36
    r = await api_client.post(
        "/v1/default/banks/md-retain-2/memories",
        json={"items": [{"content": f"my token is {secret}"}]},
    )
    assert r.status_code == 200, r.text
    async with memory._pool.acquire() as conn:
        texts = [row["text"] for row in await conn.fetch("SELECT text FROM memory_units WHERE bank_id = 'md-retain-2'")]
    assert all(secret not in t for t in texts), texts


@pytest.mark.asyncio
async def test_retain_blocks_secret_item(api_client) -> None:
    await api_client.put("/v1/default/banks/md-retain-3", json={})
    await _set_policy(
        api_client,
        "md-retain-3",
        {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]}},
    )
    # A single item that contains a secret is fully blocked → 422.
    secret = "sk-ant-" + "B" * 40
    r = await api_client.post(
        "/v1/default/banks/md-retain-3/memories",
        json={"items": [{"content": f"key={secret}"}]},
    )
    assert r.status_code == 422, r.text
    # Content with no sensitive_data hit still passes (nothing to block).
    r2 = await api_client.post(
        "/v1/default/banks/md-retain-3/memories",
        json={"items": [{"content": "the roadmap meeting is on friday"}]},
    )
    assert r2.status_code == 200, r2.text


async def _memory_defense_webhook_events(memory, bank: str) -> list[dict]:
    """Return the fully-parsed WebhookEvent bodies of the memory_defense.triggered
    deliveries queued for ``bank``. The webhook_delivery task_payload nests the
    serialized event under ``payload`` (a JSON string)."""
    async with memory._pool.acquire() as conn:
        # Order most-recent-first so callers using ``events[0]`` always see
        # the latest queued delivery — otherwise pollution from earlier test
        # runs against the same bank surfaces stale payloads.
        rows = await conn.fetch(
            "SELECT task_payload FROM async_operations "
            "WHERE operation_type = 'webhook_delivery' AND bank_id = $1 "
            "ORDER BY created_at DESC",
            bank,
        )
    events: list[dict] = []
    for row in rows:
        task = row["task_payload"]
        if isinstance(task, str):
            task = json.loads(task)
        if task.get("event_type") != "memory_defense.triggered":
            continue
        events.append(json.loads(task["payload"]))
    return events


@pytest.mark.asyncio
async def test_retain_fires_webhook_on_redact(api_client, memory) -> None:
    """A redact decision queues a memory_defense.triggered delivery whose payload
    reports the action, detector, and matched pattern labels."""
    bank = "md-retain-wh"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    wr = await api_client.post(
        f"/v1/default/banks/{bank}/webhooks",
        json={"url": "https://example.com/hook", "event_types": ["memory_defense.triggered"]},
    )
    assert wr.status_code in {200, 201}, wr.text
    await _set_policy(api_client, bank, _REDACT_POLICY)

    secret = "ghp_" + "A" * 36
    rr = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": f"rotate {secret}"}]},
    )
    assert rr.status_code == 200, rr.text

    events = await _memory_defense_webhook_events(memory, bank)
    assert len(events) >= 1, events
    ev = events[0]
    assert ev["event"] == "memory_defense.triggered"
    assert ev["status"] == "redact"
    data = ev["data"]
    assert data["action"] == "redact"
    assert data["detector"] == "sensitive_data"
    assert "github_token" in data["matched_types"]
    assert data["message"]
    # The webhook payload carries a per-match fingerprinted preview — the raw
    # secret never crosses the wire, but a SIEM can still correlate against
    # its credential inventory using the leading provider prefix + trailing
    # discriminator (e.g. `ghp_...AAAA`). Populated by OSS as of #2157.
    hits = data.get("hits") or []
    assert any(h.get("detector") == "github_token" and h.get("preview") == "ghp_...AAAA" for h in hits), hits
    for h in hits:
        assert secret not in (h.get("preview") or ""), "raw secret leaked into preview"


@pytest.mark.asyncio
async def test_retain_fires_webhook_on_block(api_client, memory) -> None:
    """A block decision also fires the webhook (before the 422 is raised), with
    action=block in the payload."""
    bank = "md-retain-wh-block"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    wr = await api_client.post(
        f"/v1/default/banks/{bank}/webhooks",
        json={"url": "https://example.com/hook", "event_types": ["memory_defense.triggered"]},
    )
    assert wr.status_code in {200, 201}, wr.text
    await _set_policy(
        api_client,
        bank,
        {"memory_defense": {"enabled": True, "rules": [{"on": "sensitive_data", "action": "block"}]}},
    )

    secret = "AKIA" + "A" * 16
    rr = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": f"key {secret}"}]},
    )
    assert rr.status_code == 422, rr.text  # all items blocked

    events = await _memory_defense_webhook_events(memory, bank)
    assert any(ev["data"]["action"] == "block" for ev in events), events
    blocked = next(ev for ev in events if ev["data"]["action"] == "block")
    assert blocked["status"] == "block"
    assert blocked["data"]["detector"] == "sensitive_data"
    assert "aws_access_key" in blocked["data"]["matched_types"]


@pytest.mark.asyncio
async def test_retain_writes_audit_log(api_client, memory) -> None:
    """A non-allow decision writes a 'memory_defense' audit entry recording the
    action taken and what matched (when audit logging is enabled)."""
    import asyncio

    # Audit logging is hierarchical (env -> tenant -> bank) and defaults off.
    # Enable it deployment-wide for this case: the logger flag covers actions
    # with no bank in scope, enable_audit_default sets the resolver default the
    # per-bank gate reads.
    memory._audit_logger._enabled = True
    enable_audit_default(memory, True)
    try:
        bank = "md-audit"
        await api_client.put(f"/v1/default/banks/{bank}", json={})
        await _set_policy(api_client, bank, _REDACT_POLICY)

        secret = "ghp_" + "A" * 36
        rr = await api_client.post(
            f"/v1/default/banks/{bank}/memories",
            json={"items": [{"content": f"rotate {secret}", "document_id": "doc-audit"}]},
        )
        assert rr.status_code == 200, rr.text

        # Audit writes are fire-and-forget — poll briefly for the row.
        row = None
        for _ in range(20):
            async with memory._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT action, transport, metadata FROM audit_log "
                    "WHERE bank_id = $1 AND action = 'memory_defense' ORDER BY started_at DESC LIMIT 1",
                    bank,
                )
            if row is not None:
                break
            await asyncio.sleep(0.1)
        assert row is not None, "no memory_defense audit entry written"
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        assert meta["action"] == "redact"
        assert meta["detector"] == "sensitive_data"
        assert "github_token" in meta["matched_types"]
        assert meta["document_id"] == "doc-audit"
    finally:
        memory._audit_logger._enabled = False


# ---------------------------------------------------------------------------
# Document-body scrubbing (DB)
# ---------------------------------------------------------------------------
#
# Regression coverage for the "ghp_AAA... persists in raw documents" leak:
# per-chunk screen() mutates the chunk content, but the document body is built
# either from the raw dict or from document_body_override (the FULL original
# body for oversized inputs). Both paths must be scrubbed.

# Mix of secret patterns covered by the redactor (keys, tokens, PII, DB URLs).
_SECRETS = {
    "ssn": "123-45-6789",
    "github_pat": "ghp_" + "A" * 36,
    "github_app": "ghs_" + "B" * 36,
    "anthropic": "sk-ant-" + "C" * 40,
    "xai": "xai-" + "D" * 40,
    "groq": "gsk_" + "E" * 30,
    "huggingface": "hf_" + "F" * 35,
    "stripe_live": "sk_live_" + "G" * 30,
    "twilio_sid": "AC" + "0" * 32,
    "sendgrid": "SG." + "H" * 22 + "." + "I" * 43,
    "aws_access": "AKIA" + "J" * 16,
    "postgres_url": "postgres://user:p4ssw0rd@db.example.com:5432/app",
}


@pytest.mark.asyncio
async def test_scrubs_secrets_from_document_body(api_client) -> None:
    bank = "md-doc-body-1"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await _set_policy(api_client, bank, _REDACT_POLICY)

    doc_id = "leak-test-doc-1"
    body = "Audit log:\n" + "\n".join(f"- {label} = {value}" for label, value in _SECRETS.items())
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": body, "document_id": doc_id}]},
    )
    assert r.status_code == 200, r.text

    # 1) Memory units must not contain ANY secret value verbatim.
    r2 = await api_client.get(f"/v1/default/banks/{bank}/memories/list", params={"limit": 200})
    for label, value in _SECRETS.items():
        for unit in r2.json()["items"]:
            assert value not in unit["text"], f"memory_unit leaked {label}={value!r}: {unit['text']!r}"

    # 2) Document body must not contain ANY secret value verbatim.
    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    for label, value in _SECRETS.items():
        assert value not in original_text, f"document.original_text leaked {label}={value!r}"


@pytest.mark.asyncio
async def test_scrubs_ssn_from_short_message(api_client) -> None:
    bank = "md-doc-body-ssn"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await _set_policy(api_client, bank, _REDACT_POLICY)

    doc_id = "ssn-1"
    ssn = "123-45-6789"
    body = f"The user pasted their ssn us for debugging: {ssn} — please scrub and rotate."
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": body, "document_id": doc_id}]},
    )
    assert r.status_code == 200, r.text

    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    assert ssn not in original_text, f"document.original_text leaked SSN: {original_text!r}"
    assert "[REDACTED:ssn_us]" in original_text, original_text


@pytest.mark.asyncio
async def test_scrubs_secrets_in_multi_doc_batch(api_client) -> None:
    """Multiple items with distinct document_ids in a single POST trigger the
    multi-doc grouping recursion in retain_batch(); screening must run for each."""
    bank = "md-multi-doc-batch"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await _set_policy(api_client, bank, _REDACT_POLICY)

    secrets = [
        ("anthropic", "sk-ant-" + "A" * 40),
        ("xai", "xai-" + "G" * 80),
        ("databricks", "dapi" + "L" * 32),
        ("ssn", "123-45-6789"),
    ]
    items = [
        {"content": f"User pasted {label}: {value} — scrub it.", "document_id": f"multi-doc-{label}"}
        for label, value in secrets
    ]
    r = await api_client.post(f"/v1/default/banks/{bank}/memories", json={"items": items})
    assert r.status_code == 200, r.text

    for label, value in secrets:
        r2 = await api_client.get(f"/v1/default/banks/{bank}/documents/multi-doc-{label}")
        assert r2.status_code == 200, r2.text
        assert value not in r2.json()["original_text"], f"{label} leaked in multi-doc batch"


@pytest.mark.asyncio
async def test_scrubs_secrets_from_oversized_chunked_input(api_client) -> None:
    """A single content item over retain_batch_tokens is chunked and carries the
    FULL original body in document_body_override, which bypasses per-chunk
    screen() — the orchestrator must scrub it before persisting."""
    bank = "md-doc-body-oversized"
    await api_client.put(f"/v1/default/banks/{bank}", json={})
    await _set_policy(api_client, bank, _REDACT_POLICY)

    secret = "ghp_" + "Z" * 36
    ssn = "987-65-4321"
    padding = ("The quick brown fox jumps over the lazy dog. " * 50 + "\n") * 5  # ~12KB
    body = f"Audit:\n{padding}\nCredential: {secret}\nUser SSN: {ssn}\n{padding}{padding}{padding}"  # >45KB

    doc_id = "oversized-leak-1"
    r = await api_client.post(
        f"/v1/default/banks/{bank}/memories",
        json={"items": [{"content": body, "document_id": doc_id}]},
    )
    assert r.status_code == 200, r.text

    r3 = await api_client.get(f"/v1/default/banks/{bank}/documents/{doc_id}")
    assert r3.status_code == 200, r3.text
    original_text = r3.json()["original_text"]
    assert secret not in original_text, "oversized document.original_text leaked github token"
    assert ssn not in original_text, "oversized document.original_text leaked SSN"
