"""
Tests for hindsight_api.main module (single-worker code path).

The main.py module is used when running with a single worker:
    hindsight-api  (or hindsight-api --workers 1)

When workers=1, main.py creates the app directly and passes it to uvicorn.
These tests ensure that extensions are properly loaded in this code path.

Compare with test_server_module.py which tests the multi-worker path (workers > 1).
"""

import socket
import sys
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig


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

        with (
            patch("hindsight_api.main.MemoryEngine") as mock_engine,
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.load_extension", side_effect=tracking_load_extension),
            patch("hindsight_api.main.DefaultExtensionContext"),
            patch("hindsight_api.main.print_banner"),
            patch("hindsight_api.main._run_uvicorn"),
        ):  # Don't actually start uvicorn
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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
            patch("hindsight_api.main.DefaultExtensionContext"),
            patch("hindsight_api.main.print_banner"),
            patch("hindsight_api.main._run_uvicorn"),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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
            patch("hindsight_api.main.DefaultExtensionContext"),
            patch("hindsight_api.main.print_banner"),
            patch("hindsight_api.main._run_uvicorn"),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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

        with (
            patch("hindsight_api.main.MemoryEngine", side_effect=capture_memory_engine),
            patch("hindsight_api.main.create_app") as mock_create_app,
            patch("hindsight_api.main._get_raw_config") as mock_get_config,
            patch("hindsight_api.main.DefaultExtensionContext", side_effect=capture_context),
            patch("hindsight_api.main.print_banner"),
            patch("hindsight_api.main._run_uvicorn"),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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

        # Verify context was created and set
        assert len(context_created) == 1, "DefaultExtensionContext should be created"
        assert captured_tenant_ext[0] is not None, "Tenant extension should be captured"
        assert captured_tenant_ext[0]._context_set, "set_context was not called on tenant extension"

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
            patch("hindsight_api.main._run_uvicorn"),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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
            patch("hindsight_api.main._run_uvicorn", side_effect=capture_uvicorn_run),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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
            patch("hindsight_api.main._run_uvicorn", side_effect=capture_uvicorn_run),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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
            patch("hindsight_api.main._run_uvicorn", side_effect=capture_uvicorn_run),
        ):
            mock_config = HindsightConfig.from_env()
            mock_config.host = "0.0.0.0"
            mock_config.port = 0
            mock_config.log_level = "info"
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


@pytest.mark.parametrize("flags", [[], ["--workers", "2"], ["--reload"], ["--daemon"]])
def test_occupied_port_fails_before_lazy_imports(monkeypatch, flags):
    import hindsight_api.main as entrypoint
    from hindsight_api.config import HindsightConfig

    imports = []

    def unexpected_import(name):
        imports.append(name)
        raise AssertionError(f"expensive import before bind: {name}")

    for name in entrypoint._LAZY_IMPORTS:
        monkeypatch.delitem(vars(entrypoint), name, raising=False)
    monkeypatch.setattr(entrypoint, "__getattr__", unexpected_import)
    monkeypatch.setattr(entrypoint, "_get_raw_config", lambda: HindsightConfig.from_env())
    monkeypatch.setattr(entrypoint, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(entrypoint, "daemonize", lambda: None)
    with socket.create_server(("127.0.0.1", 0)) as listener:
        port = listener.getsockname()[1]
        monkeypatch.setattr(sys, "argv", ["hindsight-api", "--host", "127.0.0.1", "--port", str(port), *flags])
        with pytest.raises(SystemExit) as exc:
            entrypoint.main()
        assert exc.value.code == 1
        assert imports == []
        # The existing listener remains usable; the CLI never kills or reclaims it.
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            connection, _ = listener.accept()
            connection.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_prebound_sockets_reserve_addresses_until_closed(host):
    import asyncio
    import errno

    from hindsight_api.main import _bind_sockets

    try:
        sockets = asyncio.run(_bind_sockets(host, 0))
    except OSError as exc:
        if host == "::1" and exc.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
            pytest.skip("IPv6 loopback unavailable")
        raise
    addresses = [(sock.family, sock.getsockname()) for sock in sockets]
    try:
        assert addresses
        assert len({address[1] for _, address in addresses}) == 1
        for family, address in addresses:
            assert address[1] > 0
            with socket.socket(family) as contender:
                contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                with pytest.raises(OSError):
                    contender.bind(address)
                    contender.listen()
    finally:
        for sock in sockets:
            sock.close()
    for family, address in addresses:
        with socket.socket(family) as retry:
            retry.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            retry.bind(address)
            retry.listen()


@pytest.mark.parametrize("conflicts", [1, 5])
def test_ephemeral_port_retries_conflicts_without_leaking_sockets(monkeypatch, conflicts):
    import asyncio
    import errno

    from hindsight_api.main import _bind_sockets

    probes = []
    blockers = []
    reserved = []
    attempts = 0

    async def exercise():
        nonlocal attempts
        loop = asyncio.get_running_loop()
        create_server = loop.create_server

        async def race(*args, **kwargs):
            nonlocal attempts
            if kwargs["port"]:
                attempts += 1
                if attempts <= conflicts:
                    # Occupy the selected port after the probe closes, just as a
                    # competing process could. The actual asyncio bind must fail.
                    blockers.append(socket.create_server(("127.0.0.1", kwargs["port"])))
            server = await create_server(*args, **kwargs)
            if kwargs["port"] == 0:
                probes.extend(server.sockets)
            return server

        monkeypatch.setattr(loop, "create_server", race)
        if conflicts == 5:
            with pytest.raises(OSError) as exc:
                await _bind_sockets("localhost", 0)
            assert exc.value.errno == errno.EADDRINUSE
        else:
            reserved.extend(await _bind_sockets("localhost", 0))
            assert len({sock.getsockname()[1] for sock in reserved}) == 1

    try:
        asyncio.run(exercise())
        assert attempts == min(conflicts + 1, 5)
        assert probes and all(sock.fileno() == -1 for sock in probes)
        for blocker in blockers:
            with socket.create_connection(blocker.getsockname(), timeout=1):
                connection, _ = blocker.accept()
                connection.close()
    finally:
        for sock in reserved + blockers:
            sock.close()


def test_main_closes_sockets_when_initialization_fails(monkeypatch):
    import hindsight_api.main as entrypoint
    from hindsight_api.config import HindsightConfig

    monkeypatch.setattr(entrypoint, "_get_raw_config", lambda: HindsightConfig.from_env())
    monkeypatch.setattr(entrypoint, "load_dotenv_for_entrypoint", lambda: None)
    monkeypatch.setattr(sys, "argv", ["hindsight-api", "--host", "127.0.0.1", "--port", "0"])
    held = []

    def fail(args, config, is_daemon, sockets):
        held.extend(sockets)
        assert config.port == args.port == sockets[0].getsockname()[1] > 0
        raise RuntimeError("engine initialization failed")

    monkeypatch.setattr(entrypoint, "_serve", fail)
    with pytest.raises(RuntimeError, match="engine initialization failed"):
        entrypoint.main()
    assert held and all(sock.fileno() == -1 for sock in held)


@pytest.mark.parametrize("mode", ["single", "workers", "reload"])
def test_uvicorn_receives_reserved_sockets_in_every_mode(monkeypatch, mode):
    import asyncio

    import uvicorn

    from hindsight_api.main import _bind_sockets, _run_uvicorn

    server = MagicMock(started=True)
    monkeypatch.setattr(uvicorn, "Server", lambda config: server)
    with patch("uvicorn.supervisors.ChangeReload") as reload, patch("uvicorn.supervisors.Multiprocess") as workers:
        sockets = asyncio.run(_bind_sockets("127.0.0.1", 0))
        try:
            _run_uvicorn(
                sockets=sockets,
                app="hindsight_api.server:app",
                workers=2 if mode == "workers" else 1,
                reload=mode == "reload",
                log_level="error",
            )
            if mode == "single":
                server.run.assert_called_once_with(sockets=sockets)
                workers.assert_not_called()
                reload.assert_not_called()
            else:
                supervisor = workers if mode == "workers" else reload
                assert supervisor.call_args.kwargs["sockets"] is sockets
                assert supervisor.call_args.kwargs["target"] == server.run
                supervisor.return_value.run.assert_called_once_with()
                server.run.assert_not_called()
        finally:
            for sock in sockets:
                sock.close()


def test_prebound_socket_serves_http_and_keeps_startup_failure_exit_code(monkeypatch):
    import asyncio
    import threading
    import urllib.request

    import uvicorn

    from hindsight_api.main import _bind_sockets, _run_uvicorn

    real_server = uvicorn.Server
    created = []
    failures = []
    ready = threading.Event()

    class Server(real_server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            ready.set()

    def make_server(config):
        server = Server(config)
        created.append(server)
        return server

    monkeypatch.setattr(uvicorn, "Server", make_server)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"reserved listener"})

    sockets = asyncio.run(_bind_sockets("127.0.0.1", 0))
    port = sockets[0].getsockname()[1]

    def run():
        try:
            _run_uvicorn(sockets=sockets, app=app, loop="asyncio", lifespan="off", log_level="error")
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert ready.wait(10), failures
        with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=5) as response:
            assert response.read() == b"reserved listener"
    finally:
        for server in created:
            server.should_exit = True
        thread.join(10)
        for sock in sockets:
            sock.close()
    assert not thread.is_alive()
    assert failures == []

    failed = MagicMock(started=False)
    monkeypatch.setattr(uvicorn, "Server", lambda config: failed)
    with pytest.raises(SystemExit) as exc:
        _run_uvicorn(sockets=[], app=app, log_level="error")
    assert exc.value.code == 3
