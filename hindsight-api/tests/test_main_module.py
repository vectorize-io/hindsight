"""
Tests for hindsight_api.main module (single-worker code path).

The main.py module is used when running with a single worker:
    hindsight-api  (or hindsight-api --workers 1)

When workers=1, main.py creates the app directly and passes it to uvicorn.
These tests ensure that extensions are properly loaded in this code path.

Compare with test_server_module.py which tests the multi-worker path (workers > 1).
"""

import sys
from unittest.mock import MagicMock, patch


class TestMainModuleExtensionLoading:
    """Tests that main.py correctly loads extensions when configured via environment."""

    def test_main_loads_tenant_extension_when_configured(self, monkeypatch):
        """
        Verify that main.py loads tenant extension from HINDSIGHT_API_TENANT_EXTENSION.

        This ensures extension loading works in the single-worker code path.
        """
        # Set up environment to configure a tenant extension
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "tests.test_main_module:MockTenantExtension",
        )
        # Ensure single worker mode
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")

        # Track what extensions were loaded via load_extension
        loaded_extensions = {}

        # Get the real load_extension function
        from hindsight_api.extensions.loader import load_extension as real_load_extension

        def tracking_load_extension(name, base_class):
            """Track calls to load_extension and delegate to original."""
            result = real_load_extension(name, base_class)
            loaded_extensions[name] = result
            return result

        with patch("hindsight_api.main.MemoryEngine") as mock_engine, \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.load_extension", side_effect=tracking_load_extension), \
             patch("hindsight_api.main.DefaultExtensionContext"), \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run"):  # Don't actually start uvicorn

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            # Mock sys.argv to simulate CLI invocation
            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        # Verify TENANT extension was loaded
        assert "TENANT" in loaded_extensions, \
            "main.py did not call load_extension('TENANT', ...) - extensions not loaded!"
        assert loaded_extensions["TENANT"] is not None, \
            "load_extension('TENANT', ...) returned None despite env var being set"
        assert isinstance(loaded_extensions["TENANT"], MockTenantExtension), \
            f"Expected MockTenantExtension, got {type(loaded_extensions['TENANT'])}"

    def test_main_loads_operation_validator_when_configured(self, monkeypatch):
        """
        Verify that main.py loads operation validator from HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION.
        """
        monkeypatch.setenv(
            "HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION",
            "tests.test_main_module:MockOperationValidator",
        )
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")

        loaded_extensions = {}

        from hindsight_api.extensions.loader import load_extension as real_load_extension

        def tracking_load_extension(name, base_class):
            result = real_load_extension(name, base_class)
            loaded_extensions[name] = result
            return result

        with patch("hindsight_api.main.MemoryEngine") as mock_engine, \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.load_extension", side_effect=tracking_load_extension), \
             patch("hindsight_api.main.DefaultExtensionContext"), \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run"):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        assert "OPERATION_VALIDATOR" in loaded_extensions, \
            "main.py did not call load_extension('OPERATION_VALIDATOR', ...)"
        assert loaded_extensions["OPERATION_VALIDATOR"] is not None
        assert isinstance(loaded_extensions["OPERATION_VALIDATOR"], MockOperationValidator)

    def test_main_passes_extensions_to_memory_engine(self, monkeypatch):
        """
        Verify that main.py passes loaded extensions to MemoryEngine constructor.

        This is the critical test - even if extensions are loaded, they must be
        passed to MemoryEngine for authentication to work.
        """
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "tests.test_main_module:MockTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")

        memory_engine_calls = []

        def capture_memory_engine(*args, **kwargs):
            memory_engine_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        with patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine), \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.DefaultExtensionContext"), \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run"):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        # Verify MemoryEngine was called
        assert len(memory_engine_calls) == 1, "MemoryEngine should be called exactly once"

        call_kwargs = memory_engine_calls[0]["kwargs"]

        # THE CRITICAL ASSERTION: tenant_extension must be passed and not None
        assert "tenant_extension" in call_kwargs, \
            "MemoryEngine was not called with tenant_extension parameter!"
        assert call_kwargs["tenant_extension"] is not None, \
            "tenant_extension was None - main.py did not pass loaded extension to MemoryEngine!"

    def test_main_sets_extension_context_on_tenant_extension(self, monkeypatch):
        """
        Verify that main.py sets the extension context on tenant extension.

        This is required for tenant extensions that need to provision schemas.
        """
        monkeypatch.setenv(
            "HINDSIGHT_API_TENANT_EXTENSION",
            "tests.test_main_module:MockTenantExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")

        captured_tenant_ext = [None]

        def capture_memory_engine(*args, **kwargs):
            captured_tenant_ext[0] = kwargs.get("tenant_extension")
            return MagicMock()

        context_created = []

        def capture_context(*args, **kwargs):
            ctx = MagicMock()
            context_created.append(ctx)
            return ctx

        with patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine), \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.DefaultExtensionContext", side_effect=capture_context), \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run"):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        # Verify context was created and set
        assert len(context_created) == 1, "DefaultExtensionContext should be created"
        assert captured_tenant_ext[0] is not None, "Tenant extension should be captured"
        assert captured_tenant_ext[0]._context_set, \
            "set_context was not called on tenant extension"

    def test_main_works_without_extensions(self, monkeypatch):
        """
        Verify that main.py works correctly when no extensions are configured.
        """
        # Ensure no extension env vars are set
        monkeypatch.delenv("HINDSIGHT_API_TENANT_EXTENSION", raising=False)
        monkeypatch.delenv("HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION", raising=False)
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")

        memory_engine_calls = []

        def capture_memory_engine(*args, **kwargs):
            memory_engine_calls.append({"args": args, "kwargs": kwargs})
            return MagicMock()

        with patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine), \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run"):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        # Should work without extensions
        assert len(memory_engine_calls) == 1
        call_kwargs = memory_engine_calls[0]["kwargs"]

        # Extensions should be None when not configured
        assert call_kwargs.get("tenant_extension") is None
        assert call_kwargs.get("operation_validator") is None

    def test_main_uses_app_object_for_single_worker(self, monkeypatch):
        """
        Verify that main.py passes the app object (not import string) when workers=1.

        This is important because it means single-worker mode uses the app created
        in main.py (with extensions loaded), not server.py.
        """
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")
        monkeypatch.delenv("HINDSIGHT_API_TENANT_EXTENSION", raising=False)

        uvicorn_calls = []

        def capture_uvicorn_run(**kwargs):
            uvicorn_calls.append(kwargs)

        mock_app = MagicMock()

        with patch("hindsight_api.main.MemoryEngine") as mock_engine, \
             patch("hindsight_api.main.create_app", return_value=mock_app), \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run", side_effect=capture_uvicorn_run):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api', '--workers', '1']):
                from hindsight_api.main import main
                main()

        assert len(uvicorn_calls) == 1
        # With workers=1, should pass app object, not import string
        assert uvicorn_calls[0]["app"] is mock_app, \
            "main.py should pass app object (not import string) when workers=1"

    def test_main_uses_import_string_for_multiple_workers(self, monkeypatch):
        """
        Verify that main.py uses import string when workers > 1.

        This is important because multi-worker mode requires server.py to be imported
        by each worker process.
        """
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "2")
        monkeypatch.delenv("HINDSIGHT_API_TENANT_EXTENSION", raising=False)

        uvicorn_calls = []

        def capture_uvicorn_run(**kwargs):
            uvicorn_calls.append(kwargs)

        with patch("hindsight_api.main.MemoryEngine") as mock_engine, \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run", side_effect=capture_uvicorn_run):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api', '--workers', '2']):
                from hindsight_api.main import main
                main()

        assert len(uvicorn_calls) == 1
        # With workers > 1, should use import string
        assert uvicorn_calls[0]["app"] == "hindsight_api.server:app", \
            "main.py should use import string when workers > 1"
        assert uvicorn_calls[0]["workers"] == 2

    def test_main_sets_keepalive_timeout(self, monkeypatch):
        """
        Verify that uvicorn is configured with timeout_keep_alive > aiohttp's
        default client keepalive timeout (15s), so the server never closes
        connections before the client does.
        """
        monkeypatch.setenv("HINDSIGHT_API_WORKERS", "1")
        monkeypatch.delenv("HINDSIGHT_API_TENANT_EXTENSION", raising=False)

        uvicorn_calls = []

        def capture_uvicorn_run(**kwargs):
            uvicorn_calls.append(kwargs)

        with patch("hindsight_api.main.MemoryEngine") as mock_engine, \
             patch("hindsight_api.main.create_app") as mock_create_app, \
             patch("hindsight_api.main._get_raw_config") as mock_get_config, \
             patch("hindsight_api.main.print_banner"), \
             patch("uvicorn.run", side_effect=capture_uvicorn_run):

            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, 'argv', ['hindsight-api']):
                from hindsight_api.main import main
                main()

        assert len(uvicorn_calls) == 1
        assert "timeout_keep_alive" in uvicorn_calls[0], \
            "uvicorn config must set timeout_keep_alive"
        assert uvicorn_calls[0]["timeout_keep_alive"] > 15, \
            "timeout_keep_alive must exceed aiohttp's 15s client default"


# Mock extensions for testing
from hindsight_api.extensions import (
    OperationValidatorExtension,
    RecallContext,
    ReflectContext,
    RequestContext,
    RetainContext,
    TenantContext,
    TenantExtension,
    ValidationResult,
)


class MockTenantExtension(TenantExtension):
    """Mock tenant extension for testing main.py extension loading."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._context_set = False

    async def authenticate(self, request_context: RequestContext) -> TenantContext:
        return TenantContext(schema_name="public")

    async def list_tenants(self) -> list:
        from hindsight_api.extensions.tenant import Tenant

        return [Tenant(schema="public")]

    def set_context(self, context) -> None:
        self._context_set = True


class MockOperationValidator(OperationValidatorExtension):
    """Mock operation validator for testing main.py extension loading."""

    def __init__(self, config: dict):
        super().__init__(config)

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()
