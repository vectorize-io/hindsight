"""Tests for StaticKeysTenantExtension (env-configured per-user API keys)."""

import asyncio
import hashlib
import hmac
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.extensions.context import ExtensionContext
from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.tenant import AuthenticationError, Tenant, TenantContext, TenantExtension
from hindsight_api.models import RequestContext
from hindsight_ext_static_keys_tenant.extension import StaticKeysTenantExtension, _KeyEntry


def _make_config(**overrides) -> dict[str, str]:
    """Build a minimal valid config, overridable per test."""
    config = {
        "users": "rafael:key-a,sophie:key-b",
    }
    config.update(overrides)
    return config


def _make_extension(**overrides) -> StaticKeysTenantExtension:
    return StaticKeysTenantExtension(_make_config(**overrides))


def _expected_entry(user_id: str, key: str, schema_prefix: str = "user") -> _KeyEntry:
    """Build the _KeyEntry the extension must produce for one configured pair."""
    return _KeyEntry(
        user_id=user_id.lower(),
        schema_name=f"{schema_prefix}_{user_id.lower().replace('-', '_')}",
        key_id=hashlib.sha256(key.encode("utf-8", "surrogateescape")).hexdigest()[:16],
        key_bytes=key.encode("utf-8", "surrogateescape"),
    )


class TestStaticKeysTenantExtensionInit:
    """Tests for initialization and configuration parsing."""

    def test_init_with_valid_config(self):
        ext = _make_extension()
        assert ext.schema_prefix == "user"
        assert ext._key_to_user == {
            "key-a": _expected_entry("rafael", "key-a"),
            "key-b": _expected_entry("sophie", "key-b"),
        }
        assert ext._users == {"rafael": "user_rafael", "sophie": "user_sophie"}

    def test_init_missing_users(self):
        with pytest.raises(ValueError, match="HINDSIGHT_API_TENANT_USERS is required"):
            _make_extension(users="")

    def test_init_default_schema_prefix(self):
        ext = _make_extension()
        assert ext.schema_prefix == "user"

    def test_init_custom_schema_prefix(self):
        ext = _make_extension(schema_prefix="tenant")
        assert ext._users == {"rafael": "tenant_rafael", "sophie": "tenant_sophie"}

    def test_init_rejects_invalid_schema_prefix(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="1bad")

    def test_init_rejects_schema_prefix_with_dash(self):
        with pytest.raises(ValueError, match="Invalid schema_prefix"):
            _make_extension(schema_prefix="bad-prefix")

    def test_init_accepts_underscore_prefix(self):
        ext = _make_extension(schema_prefix="my_user")
        assert ext.schema_prefix == "my_user"

    def test_init_rejects_duplicate_api_key(self):
        # The error must name the colliding users via the derived key_id, never
        # the key itself — it may end up pasted into an issue or a chat.
        with pytest.raises(ValueError, match=r"Duplicate API key \(key_id '[0-9a-f]{16}'\)"):
            _make_extension(users="alice:k1,bob:k1")

    def test_init_duplicate_api_key_error_does_not_leak_key(self):
        with pytest.raises(ValueError) as excinfo:
            _make_extension(users="alice:k1,bob:k1")
        assert "k1" not in str(excinfo.value)

    def test_init_duplicate_api_key_error_names_both_users(self):
        with pytest.raises(ValueError, match="alice") as first:
            _make_extension(users="alice:k1,bob:k1")
        with pytest.raises(ValueError, match="bob") as second:
            _make_extension(users="alice:k1,bob:k1")
        assert "alice" in str(first.value) and "bob" in str(second.value)

    def test_init_rejects_entry_without_colon_does_not_leak_key(self):
        # `entry` is the raw user_id:api_key pair; quoting it in the error would
        # disclose the key of a malformed entry. Only the index may be reported.
        with pytest.raises(ValueError, match="entry at index 1") as excinfo:
            _make_extension(users="rafael:key-a,sophie")  # missing colon → whole entry malformed
        assert "sophie" not in str(excinfo.value)  # cannot report user_id without ':'; index only

    def test_init_rejects_empty_api_key_does_not_leak_entry(self):
        # HINDSIGHT_API_TENANT_USERS=rafael: — the malformed entry contains no
        # key here, but the same message path is shared with entries that do,
        # so assert the message stays free of entry/key material.
        with pytest.raises(ValueError, match="api_key must be non-empty"):
            _make_extension(users="rafael:")

    def test_init_rejects_entry_without_colon(self):
        with pytest.raises(ValueError, match="Invalid HINDSIGHT_API_TENANT_USERS entry"):
            _make_extension(users="rafael")

    def test_init_rejects_sql_injection_user_id(self):
        with pytest.raises(ValueError, match="Invalid user_id"):
            _make_extension(users='rafael"; DROP TABLE memory_units;--:key-a')

    def test_init_rejects_empty_user_id(self):
        with pytest.raises(ValueError, match="user_id must be non-empty"):
            _make_extension(users=":key-a")

    def test_init_rejects_empty_api_key(self):
        with pytest.raises(ValueError, match="api_key must be non-empty"):
            _make_extension(users="rafael:")

    def test_init_rejects_user_id_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid user_id"):
            _make_extension(users="1rafael:key-a")

    def test_key_id_is_stable_and_derived_from_key(self):
        # key_id is the truncated sha256 of the key bytes — stable across
        # restarts (so metering can attribute usage to a key long-term) and
        # short enough to quote in error messages.
        ext = _make_extension(users="rafael:key-a")
        entry = ext._key_to_user["key-a"]
        assert entry.key_id == hashlib.sha256(b"key-a").hexdigest()[:16]
        assert len(entry.key_id) == 16
        assert entry.key_id != "key-a"  # not the secret itself
        ext2 = _make_extension(users="rafael:key-a")
        assert ext2._key_to_user["key-a"].key_id == entry.key_id  # deterministic

    def test_key_id_differs_between_keys(self):
        ext = _make_extension(users="rafael:key-a,rafael:key-b")
        assert ext._key_to_user["key-a"].key_id != ext._key_to_user["key-b"].key_id

    def test_init_normalizes_dashes_in_user_id(self):
        ext = _make_extension(users="my-user-1:key-a")
        assert ext._users == {"my-user-1": "user_my_user_1"}

    def test_init_normalizes_mixed_case_user_id_to_lowercase(self):
        # "Rafael" must land on schema "user_rafael": Postgres folds unquoted
        # identifiers to lowercase at runtime (fq_table does not quote the
        # schema), so a mixed-case schema would break every query — or worse,
        # collapse two users onto one schema. Lowercasing up front keeps the
        # isolation guarantee.
        ext = _make_extension(users="Rafael:key-a")
        assert ext._users == {"rafael": "user_rafael"}
        assert ext._key_to_user == {"key-a": _expected_entry("rafael", "key-a")}

    def test_init_normalizes_mixed_case_prefix_and_dashes(self):
        # Mixed case + dashes must normalize to a single stable lowercased schema.
        ext = _make_extension(users="My-User-1:key-a")
        assert ext._users == {"my-user-1": "user_my_user_1"}

    def test_init_rejects_mixed_case_dash_collision(self):
        # "Jane-Doe" and "jane_doe" are different spellings that normalize to
        # the same schema (lowercased + dashes to underscores) — two distinct
        # user ids must never share one isolated schema.
        with pytest.raises(ValueError, match="already claimed"):
            _make_extension(users="Jane-Doe:k1,jane_doe:k2")

    def test_init_accepts_case_insensitive_duplicate_user(self):
        # The same user written with different case is the same tenant, so its
        # keys may coexist on one schema.
        ext = _make_extension(users="Rafael:k1,rafael:k2")
        assert ext._users == {"rafael": "user_rafael"}
        assert ext._key_to_user == {
            "k1": _expected_entry("Rafael", "k1"),
            "k2": _expected_entry("rafael", "k2"),
        }

    def test_init_multiple_keys_same_user(self):
        ext = _make_extension(users="rafael:key-a,rafael:key-b")
        assert ext._key_to_user == {
            "key-a": _expected_entry("rafael", "key-a"),
            "key-b": _expected_entry("rafael", "key-b"),
        }
        assert ext._users == {"rafael": "user_rafael"}

    def test_init_rejects_schema_collision_after_dash_normalization(self):
        # "jane-doe" and "jane_doe" both normalize to user_jane_doe — two
        # distinct users must never share one isolated schema.
        with pytest.raises(ValueError, match="already claimed"):
            _make_extension(users="jane-doe:k1,jane_doe:k2")

    def test_init_rejects_schema_name_over_63_chars(self):
        long_user = "u" * 62
        with pytest.raises(ValueError, match="exceeds the PostgreSQL identifier limit"):
            _make_extension(users=f"{long_user}:key-a")

    def test_init_rejects_duplicate_api_key(self):
        with pytest.raises(ValueError, match="Duplicate API key"):
            _make_extension(users="alice:k1,bob:k1")

    def test_init_multiple_keys_same_user_does_not_collide(self):
        # The legitimate multi-key-per-user case must not raise.
        ext = _make_extension(users="alice:k1,alice:k2")
        assert ext._users == {"alice": "user_alice"}

    def test_is_tenant_extension_subclass(self):
        ext = _make_extension()
        assert isinstance(ext, TenantExtension)


class TestStaticKeysTenantExtensionAuthenticate:
    """Tests for authentication."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        result = await ext.authenticate(RequestContext(api_key="key-a"))

        assert isinstance(result, TenantContext)
        assert result.schema_name == "user_rafael"
        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_missing_key(self):
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Missing Authorization header"):
            await ext.authenticate(RequestContext(api_key=None))

    @pytest.mark.asyncio
    async def test_authenticate_unknown_key(self):
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="wrong-key"))

    @pytest.mark.asyncio
    async def test_authenticate_non_ascii_key_is_401_not_500(self):
        # Header values arrive latin-1-decoded, so a byte >= 0x80 (here 'é',
        # U+00E9) is a legitimate non-ASCII str. It must be rejected as an
        # invalid key (AuthenticationError -> 401), not raise TypeError out of
        # authenticate() (which would be a 500) the way str-vs-str
        # hmac.compare_digest does.
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="\xe9"))

    @pytest.mark.asyncio
    async def test_authenticate_compares_all_keys_in_constant_time(self, monkeypatch):
        # Regression pin for the auth fix: there is no exact-equality fast path
        # anymore — an unknown key must run hmac.compare_digest over EVERY
        # configured key, on bytes (not str, which would TypeError on non-ASCII).
        ext = _make_extension()
        compared: list[tuple[bytes, bytes]] = []
        orig = hmac.compare_digest

        def spy(a, b):
            compared.append((a, b))
            return orig(a, b)

        monkeypatch.setattr("hindsight_ext_static_keys_tenant.extension.hmac.compare_digest", spy)

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate(RequestContext(api_key="wrong-key"))

        assert len(compared) == len(ext._key_to_user), "every configured key must be compared"
        assert compared, "compare_digest must always run over all keys (no fast path)"
        assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in compared)
        # Configured keys are pre-encoded at init: the compared bytes equal the
        # utf-8/surrogateescape encoding of the stored keys.
        expected = {entry.key_bytes for entry in ext._key_to_user.values()}
        assert {b for _, b in compared} == expected

    @pytest.mark.asyncio
    async def test_authenticate_valid_key_matches_preencoded_bytes(self):
        # A valid key authenticates through the pre-encoded-bytes comparison.
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        result = await ext.authenticate(RequestContext(api_key="key-a"))
        assert result.schema_name == "user_rafael"
        assert ext._key_to_user["key-a"].key_bytes == b"key-a"

    @pytest.mark.asyncio
    async def test_authenticate_sets_usage_metering_fields(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        ctx = RequestContext(api_key="key-a")
        await ext.authenticate(ctx)

        assert ctx.tenant_id == "rafael"
        # api_key_id identifies *the key*, not the user — with multiple keys
        # per user, metering must be able to tell which key was used. It is
        # the derived, non-secret key_id (never the key itself).
        assert ctx.api_key_id == hashlib.sha256(b"key-a").hexdigest()[:16]
        assert ctx.api_key_id != ctx.tenant_id

    @pytest.mark.asyncio
    async def test_authenticate_metering_distinguishes_keys_for_same_user(self):
        ext = _make_extension(users="rafael:key-a,rafael:key-b")
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        ctx_a = RequestContext(api_key="key-a")
        await ext.authenticate(ctx_a)
        ctx_b = RequestContext(api_key="key-b")
        await ext.authenticate(ctx_b)

        # Same user/tenant, different keys → different api_key_id.
        assert ctx_a.tenant_id == ctx_b.tenant_id == "rafael"
        assert ctx_a.api_key_id != ctx_b.api_key_id

    @pytest.mark.asyncio
    async def test_authenticate_mixed_case_key_maps_to_lowercase_schema(self):
        # A user configured with a mixed-case id authenticates onto the
        # lowercased schema, matching what the runtime's fq_table() resolves.
        ext = _make_extension(users="Rafael:key-a")
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        result = await ext.authenticate(RequestContext(api_key="key-a"))

        assert result.schema_name == "user_rafael"
        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_provisions_schema_once(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        await ext.authenticate(RequestContext(api_key="key-a"))
        await ext.authenticate(RequestContext(api_key="key-a"))

        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_concurrent_first_requests_provision_once(self):
        # Two first requests for the same user racing each other must not both
        # run migrations: the per-schema lock serializes them and the loser
        # skips the (now cached) migration on re-check.
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        await asyncio.gather(
            ext.authenticate(RequestContext(api_key="key-a")),
            ext.authenticate(RequestContext(api_key="key-a")),
        )

        mock_context.run_migration.assert_called_once_with("user_rafael")

    @pytest.mark.asyncio
    async def test_authenticate_schema_init_failure_not_cached(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock(side_effect=RuntimeError("Migration failed"))
        ext._context = mock_context

        with pytest.raises(AuthenticationError, match="Failed to initialize tenant"):
            await ext.authenticate(RequestContext(api_key="key-a"))

        assert "user_rafael" not in ext._initialized_schemas


class TestStaticKeysTenantExtensionMcp:
    """Tests for MCP auth."""

    @pytest.mark.asyncio
    async def test_authenticate_mcp_delegates(self):
        ext = _make_extension()
        mock_context = AsyncMock(spec=ExtensionContext)
        mock_context.run_migration = AsyncMock()
        ext._context = mock_context

        with patch.object(ext, "authenticate") as mock_authenticate:
            mock_authenticate.return_value = TenantContext(schema_name="user_rafael")
            result = await ext.authenticate_mcp(RequestContext(api_key="key-a"))
            mock_authenticate.assert_awaited_once()
        assert result.schema_name == "user_rafael"

    @pytest.mark.asyncio
    async def test_authenticate_mcp_disabled_is_refused_at_init(self):
        # The flag would let unauthenticated MCP clients into the base schema
        # of a deployment whose whole purpose is per-user isolation — refuse
        # at startup (fail-fast contract) instead of silently downgrading auth.
        with pytest.raises(ValueError, match="not supported by StaticKeysTenantExtension"):
            _make_extension(mcp_auth_disabled="true")

    @pytest.mark.asyncio
    async def test_authenticate_mcp_requires_auth(self):
        # Without the flag there is no MCP bypass: a bad key is rejected on the
        # MCP path exactly as on HTTP.
        ext = _make_extension()
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await ext.authenticate_mcp(RequestContext(api_key="wrong-key"))


class TestStaticKeysTenantExtensionListTenants:
    """Tests for list_tenants."""

    @pytest.mark.asyncio
    async def test_list_tenants_returns_all_configured(self):
        ext = _make_extension()
        tenants = await ext.list_tenants()
        assert tenants == [
            Tenant(schema="user_rafael", tenant_id="rafael"),
            Tenant(schema="user_sophie", tenant_id="sophie"),
        ]

    @pytest.mark.asyncio
    async def test_list_tenants_includes_tenant_id(self):
        ext = _make_extension()
        tenants = await ext.list_tenants()
        assert all(t.tenant_id is not None for t in tenants)
        assert {t.tenant_id for t in tenants} == {"rafael", "sophie"}

    @pytest.mark.asyncio
    async def test_list_tenants_uses_lowercased_user_ids(self):
        # Mixed-case ids are normalized at init, so list_tenants() must hand
        # the worker the same lowercased tenant_id and schema it authenticates
        # onto — otherwise the consolidation reconcile sweep would build a
        # RequestContext that never matches an authenticated tenant.
        ext = _make_extension(users="Rafael:key-a,Sophie:key-b")
        tenants = await ext.list_tenants()
        assert tenants == [
            Tenant(schema="user_rafael", tenant_id="rafael"),
            Tenant(schema="user_sophie", tenant_id="sophie"),
        ]


class TestStaticKeysTenantExtensionLoader:
    """Tests for loading via the extension loader."""

    def test_load_via_extension_loader(self, monkeypatch):
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "hindsight_ext_static_keys_tenant:StaticKeysTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TENANT_USERS", "rafael:key-a,sophie:key-b")
        monkeypatch.setenv("HINDSIGHT_API_TENANT_SCHEMA_PREFIX", "tenant")

        ext = load_extension("TENANT", TenantExtension)

        assert ext is not None
        assert isinstance(ext, StaticKeysTenantExtension)
        assert ext.schema_prefix == "tenant"
        assert ext._users == {"rafael": "tenant_rafael", "sophie": "tenant_sophie"}
