"""
Tests for hindsight_api.main module (single-worker code path).

The main.py module is used when running with a single worker:
    hindsight-api  (or hindsight-api --workers 1)

When workers=1, main.py creates the app directly and passes it to uvicorn.
These tests ensure that extensions are properly loaded in this code path.

Compare with test_server_module.py which tests the multi-worker path (workers > 1).
"""

import errno
import socket
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestMainModuleExtensionLoading:
    """Tests that main.py correctly loads extensions when configured via environment."""

    @pytest.fixture(autouse=True)
    def _port_is_free(self):
        # These tests run main() against a fixed port; don't let a local listener on it fail them.
        with patch("hindsight_api.main._port_bind_error", return_value=None):
            yield

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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.load_extension", side_effect=tracking_load_extension),
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run"),
        ):  # Don't actually start uvicorn
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            # Mock sys.argv to simulate CLI invocation
            with patch.object(sys, "argv", ["hindsight-api"]):
                from hindsight_api.main import main

                main()

        # Verify TENANT extension was loaded
        assert "TENANT" in loaded_extensions, (
            "main.py did not call load_extension('TENANT', ...) - extensions not loaded!"
        )
        assert loaded_extensions["TENANT"] is not None, (
            "load_extension('TENANT', ...) returned None despite env var being set"
        )
        assert isinstance(loaded_extensions["TENANT"], MockTenantExtension), (
            f"Expected MockTenantExtension, got {type(loaded_extensions['TENANT'])}"
        )

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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.load_extension", side_effect=tracking_load_extension),
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run"),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api"]):
                from hindsight_api.main import main

                main()

        assert "OPERATION_VALIDATOR" in loaded_extensions, (
            "main.py did not call load_extension('OPERATION_VALIDATOR', ...)"
        )
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

        with (
            patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine),
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run"),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api"]):
                from hindsight_api.main import main

                main()

        # Verify MemoryEngine was called
        assert len(memory_engine_calls) == 1, "MemoryEngine should be called exactly once"

        call_kwargs = memory_engine_calls[0]["kwargs"]

        # THE CRITICAL ASSERTION: tenant_extension must be passed and not None
        assert "tenant_extension" in call_kwargs, "MemoryEngine was not called with tenant_extension parameter!"
        assert call_kwargs["tenant_extension"] is not None, (
            "tenant_extension was None - main.py did not pass loaded extension to MemoryEngine!"
        )

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

        with (
            patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine),
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run"),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api"]):
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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app", return_value=mock_app),
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run", side_effect=capture_uvicorn_run),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api", "--workers", "1"]):
                from hindsight_api.main import main

                main()

        assert len(uvicorn_calls) == 1
        # With workers=1, should pass app object, not import string
        assert uvicorn_calls[0]["app"] is mock_app, "main.py should pass app object (not import string) when workers=1"

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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run", side_effect=capture_uvicorn_run),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api", "--workers", "2"]):
                from hindsight_api.main import main

                main()

        assert len(uvicorn_calls) == 1
        # With workers > 1, should use import string
        assert uvicorn_calls[0]["app"] == "hindsight_api.server:app", (
            "main.py should use import string when workers > 1"
        )
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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.print_banner"),
            patch("uvicorn.run", side_effect=capture_uvicorn_run),
        ):
            mock_config = MagicMock()
            mock_config.host = "0.0.0.0"
            mock_config.port = 8888
            mock_config.log_level = "info"
            # argparse defaults: a MagicMock would compare against ints later on.
            mock_config.workers = 1
            mock_config.access_log = False
            mock_config.mcp_enabled = False
            mock_config.run_migrations_on_startup = False
            mock_config.database_url = "postgresql://test:test@localhost/test"
            mock_get_config.return_value = mock_config
            mock_engine.return_value = MagicMock()
            mock_create_app.return_value = MagicMock()

            with patch.object(sys, "argv", ["hindsight-api"]):
                from hindsight_api.main import main

                main()

        assert len(uvicorn_calls) == 1
        assert "timeout_keep_alive" in uvicorn_calls[0], "uvicorn config must set timeout_keep_alive"
        assert uvicorn_calls[0]["timeout_keep_alive"] > 15, (
            "timeout_keep_alive must exceed aiohttp's 15s client default"
        )


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

    async def authenticate(self, request_context: RequestContext) -> TenantContext:
        return TenantContext(schema_name="public")

    async def list_tenants(self) -> list:
        from hindsight_api.extensions.tenant import Tenant

        return [Tenant(schema="public")]


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


class TestPortPreflight:
    """#4281: an occupied port must fail before MemoryEngine, embedded PG, models or migrations."""

    @pytest.mark.parametrize("flags", [[], ["--workers", "2"], ["--reload"], ["--daemon"]])
    def test_occupied_port_exits_before_initialization(self, monkeypatch, capsys, flags):
        with socket.create_server(("127.0.0.1", 0)) as listener:
            port = listener.getsockname()[1]
            monkeypatch.setattr(sys, "argv", ["hindsight-api", "--host", "127.0.0.1", "--port", str(port), *flags])
            with (
                patch("hindsight_api.main._PORT_IN_USE_GRACE_SECONDS", 0),
                patch("hindsight_api.main.load_dotenv_for_entrypoint"),
                patch("hindsight_api.main.daemonize") as mock_daemonize,
                patch("hindsight_api.main.MemoryEngine") as mock_engine,
                patch("hindsight_api.main.load_extension") as mock_load_extension,
                patch("uvicorn.run") as mock_uvicorn_run,
            ):
                from hindsight_api.main import main

                with pytest.raises(SystemExit) as exc:
                    main()

            assert exc.value.code == 1
            assert f"cannot bind 127.0.0.1:{port}" in capsys.readouterr().err
            mock_daemonize.assert_not_called()
            mock_engine.assert_not_called()
            mock_load_extension.assert_not_called()
            mock_uvicorn_run.assert_not_called()
            # The existing listener is left alone and still serves.
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                conn, _ = listener.accept()
                conn.close()

    def test_port_released_within_grace_is_not_an_error(self):
        """A previous instance still releasing the port must not make startup fail."""
        from hindsight_api.main import _wait_for_port

        # side_effect would RAISE exception instances, so hand them back through a function.
        results = iter([OSError(errno.EADDRINUSE, "Address already in use")] * 2 + [None])
        with (
            patch("hindsight_api.main._port_bind_error", side_effect=lambda host, port: next(results)) as probe,
            patch("hindsight_api.main.time.sleep") as sleep,
        ):
            assert _wait_for_port("127.0.0.1", 9177) is None
        assert probe.call_count == 3
        assert sleep.call_count == 2

    def test_port_still_in_use_after_grace_fails(self):
        from hindsight_api.main import _wait_for_port

        with (
            socket.create_server(("127.0.0.1", 0)) as listener,
            patch("hindsight_api.main._PORT_IN_USE_GRACE_SECONDS", 0.3),
            patch("hindsight_api.main._PORT_IN_USE_RETRY_INTERVAL", 0.05),
        ):
            error = _wait_for_port("127.0.0.1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE

    def test_other_bind_errors_are_not_retried(self):
        from hindsight_api.main import _wait_for_port

        denied = OSError(errno.EACCES, "Permission denied")
        with (
            patch("hindsight_api.main._port_bind_error", return_value=denied) as probe,
            patch("hindsight_api.main.time.sleep") as sleep,
        ):
            assert _wait_for_port("127.0.0.1", 80) is denied
        assert probe.call_count == 1
        sleep.assert_not_called()

    def test_bind_error_reports_the_occupied_port(self):
        from hindsight_api.main import _port_bind_error

        with socket.create_server(("127.0.0.1", 0)) as listener:
            error = _port_bind_error("127.0.0.1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE

    def test_free_port_is_released_after_the_probe(self):
        """The probe must not hold or listen on the port: uvicorn binds it right after."""
        from hindsight_api.main import _port_bind_error

        with socket.create_server(("127.0.0.1", 0)) as reserved:
            port = reserved.getsockname()[1]
        assert _port_bind_error("127.0.0.1", port) is None
        with socket.socket() as probe_target:
            assert probe_target.connect_ex(("127.0.0.1", port)) != 0
        with socket.create_server(("127.0.0.1", port)):
            pass

    def test_ipv6_host_probes_an_ipv6_socket(self):
        """Mirror uvicorn: a host containing ':' is bound as AF_INET6 (dual-stack untouched)."""
        from hindsight_api.main import _port_bind_error

        if not socket.has_ipv6:
            pytest.skip("IPv6 unavailable")
        try:
            listener = socket.create_server(("::1", 0), family=socket.AF_INET6)
        except OSError:
            pytest.skip("IPv6 loopback unavailable")
        with listener:
            error = _port_bind_error("::1", listener.getsockname()[1])
        assert isinstance(error, OSError)
        assert error.errno == errno.EADDRINUSE
