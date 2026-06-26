"""Tests for the Hindsight extensions system."""

from collections import defaultdict

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from hindsight_api.extensions import (
    ApiKeyTenantExtension,
    AuthenticationError,
    BankReadContext,
    BankReadOperation,
    BankWriteContext,
    BankWriteOperation,
    # Consolidation operation
    ConsolidateContext,
    ConsolidateResult,
    Extension,
    HttpExtension,
    OperationValidationError,
    OperationValidatorExtension,
    PrecheckContext,
    PrecheckOperation,
    RecallContext,
    RecallResult,
    ReflectContext,
    ReflectResultContext,
    RequestContext,
    RetainContext,
    RetainResult,
    ValidationResult,
    load_extension,
)


class TestExtensionLoader:
    """Tests for extension loading and lifecycle."""

    def test_load_extension_with_config(self, monkeypatch):
        """Extension receives config from prefixed env vars and supports lifecycle."""
        monkeypatch.setenv(
            "HINDSIGHT_API_TEST_EXTENSION",
            "tests.test_extensions:LifecycleTestExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_TEST_API_URL", "https://example.com")
        monkeypatch.setenv("HINDSIGHT_API_TEST_MAX_RETRIES", "5")

        ext = load_extension("TEST", Extension)

        assert ext is not None
        assert ext.config["api_url"] == "https://example.com"
        assert ext.config["max_retries"] == "5"

    @pytest.mark.asyncio
    async def test_extension_lifecycle(self, monkeypatch):
        """Extension on_startup and on_shutdown are called."""
        monkeypatch.setenv(
            "HINDSIGHT_API_TEST_EXTENSION",
            "tests.test_extensions:LifecycleTestExtension",
        )

        ext = load_extension("TEST", Extension)

        assert not ext.started
        assert not ext.stopped

        await ext.on_startup()
        assert ext.started

        await ext.on_shutdown()
        assert ext.stopped

    def test_operation_enums_remain_string_compatible(self):
        """Operation enums centralize names without breaking string comparisons."""
        request_context = RequestContext(tenant_id="tenant-1")

        precheck_ctx = PrecheckContext(
            bank_id="bank-1",
            operation=PrecheckOperation.RETAIN,
            request_context=request_context,
        )
        read_ctx = BankReadContext(
            bank_id="bank-1",
            operation=BankReadOperation.GET_BANK_STATS,
            request_context=request_context,
        )
        write_ctx = BankWriteContext(
            bank_id="bank-1",
            operation=BankWriteOperation.UPDATE_BANK_CONFIG,
            request_context=request_context,
        )

        assert precheck_ctx.operation is PrecheckOperation.RETAIN
        assert read_ctx.operation is BankReadOperation.GET_BANK_STATS
        assert write_ctx.operation is BankWriteOperation.UPDATE_BANK_CONFIG
        assert precheck_ctx.operation == "retain"
        assert read_ctx.operation == "get_bank_stats"
        assert write_ctx.operation == "update_bank_config"
        assert isinstance(precheck_ctx.operation, str)
        assert isinstance(read_ctx.operation, str)
        assert isinstance(write_ctx.operation, str)


class LifecycleTestExtension(Extension):
    """Test extension for config and lifecycle tests."""

    def __init__(self, config):
        super().__init__(config)
        self.started = False
        self.stopped = False

    async def on_startup(self):
        self.started = True

    async def on_shutdown(self):
        self.stopped = True


class RateLimitingValidator(OperationValidatorExtension):
    """
    Mock validator that blocks after N attempts per bank_id.

    Used for testing the extension integration with MemoryEngine.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.max_attempts = int(config.get("max_attempts", "2"))
        self.retain_counts: dict[str, int] = defaultdict(int)
        self.recall_counts: dict[str, int] = defaultdict(int)
        self.reflect_counts: dict[str, int] = defaultdict(int)

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        self.retain_counts[ctx.bank_id] += 1
        if self.retain_counts[ctx.bank_id] > self.max_attempts:
            return ValidationResult.reject(f"Retain limit exceeded for bank {ctx.bank_id}")
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        self.recall_counts[ctx.bank_id] += 1
        if self.recall_counts[ctx.bank_id] > self.max_attempts:
            return ValidationResult.reject(f"Recall limit exceeded for bank {ctx.bank_id}")
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        self.reflect_counts[ctx.bank_id] += 1
        if self.reflect_counts[ctx.bank_id] > self.max_attempts:
            return ValidationResult.reject(f"Reflect limit exceeded for bank {ctx.bank_id}")
        return ValidationResult.accept()


class TrackingValidator(OperationValidatorExtension):
    """
    Mock validator that tracks all pre and post hook calls with full parameters.

    Used for testing that hooks receive all user-provided parameters.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Pre-hook tracking - Core operations
        self.pre_retain_calls: list[RetainContext] = []
        self.pre_recall_calls: list[RecallContext] = []
        self.pre_reflect_calls: list[ReflectContext] = []
        # Post-hook tracking - Core operations
        self.post_retain_calls: list[RetainResult] = []
        self.post_recall_calls: list[RecallResult] = []
        self.post_reflect_calls: list[ReflectResultContext] = []
        # Pre-hook tracking - Consolidation
        self.pre_consolidate_calls: list[ConsolidateContext] = []
        # Post-hook tracking - Consolidation
        self.post_consolidate_calls: list[ConsolidateResult] = []

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        self.pre_retain_calls.append(ctx)
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        self.pre_recall_calls.append(ctx)
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        self.pre_reflect_calls.append(ctx)
        return ValidationResult.accept()

    async def on_retain_complete(self, result: RetainResult) -> None:
        self.post_retain_calls.append(result)

    async def on_recall_complete(self, result: RecallResult) -> None:
        self.post_recall_calls.append(result)

    async def on_reflect_complete(self, result: ReflectResultContext) -> None:
        self.post_reflect_calls.append(result)

    # Consolidation hooks
    async def validate_consolidate(self, ctx: ConsolidateContext) -> ValidationResult:
        self.pre_consolidate_calls.append(ctx)
        return ValidationResult.accept()

    async def on_consolidate_complete(self, result: ConsolidateResult) -> None:
        self.post_consolidate_calls.append(result)


class TestMemoryEngineValidation:
    """Tests for validation integration with MemoryEngine.

    The OperationValidatorExtension is integrated at the MemoryEngine level,
    so all interfaces (HTTP API, MCP, SDK) get the same validation behavior.

    For retain, the batch is validated as a whole (all or nothing) using
    retain_batch_async which is the public method used by the HTTP API.
    """

    @pytest.mark.asyncio
    async def test_retain_batch_validation(self, memory_with_validator):
        """Retain batch is validated as a whole - accepts or rejects entire batch."""
        memory = memory_with_validator
        bank_id = "test-retain-batch"
        ctx = RequestContext()

        # First batch should succeed
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "First item"},
                {"content": "Second item"},
            ],
            request_context=ctx,
        )

        # Second batch should succeed (2nd attempt)
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Third item"}],
            request_context=ctx,
        )

        # Third batch should be blocked entirely (exceeds limit)
        with pytest.raises(OperationValidationError) as exc_info:
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[
                    {"content": "Should not be stored"},
                    {"content": "Neither should this"},
                ],
                request_context=ctx,
            )

        assert "limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_recall_validation(self, memory_with_validator):
        """Recall is validated before execution."""
        memory = memory_with_validator
        bank_id = "test-recall-validation"
        ctx = RequestContext()

        # First recall should pass validation
        await memory.recall_async(bank_id, "test query", fact_type=["world"], request_context=ctx)

        # Second recall should pass validation
        await memory.recall_async(bank_id, "another query", fact_type=["world"], request_context=ctx)

        # Third recall should be blocked by validator
        with pytest.raises(OperationValidationError) as exc_info:
            await memory.recall_async(bank_id, "blocked query", fact_type=["world"], request_context=ctx)

        assert "limit exceeded" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reflect_validation(self, memory_with_validator):
        """Reflect is validated before execution."""
        memory = memory_with_validator
        bank_id = "test-reflect-validation"
        ctx = RequestContext()

        # First reflect should pass validation (may fail internally but validation passes)
        try:
            await memory.reflect_async(bank_id, "test question", request_context=ctx)
        except OperationValidationError:
            raise  # Re-raise validation errors
        except Exception:
            pass  # Other errors are fine (e.g., no data)

        # Second reflect should pass validation
        try:
            await memory.reflect_async(bank_id, "another question", request_context=ctx)
        except OperationValidationError:
            raise
        except Exception:
            pass

        # Third reflect should be blocked by validator
        with pytest.raises(OperationValidationError) as exc_info:
            await memory.reflect_async(bank_id, "blocked question", request_context=ctx)

        assert "limit exceeded" in str(exc_info.value).lower()


@pytest.fixture
def memory_with_validator(memory):
    """Memory engine with a rate-limiting validator (max 2 attempts per bank)."""
    validator = RateLimitingValidator({"max_attempts": "2"})
    memory._operation_validator = validator
    return memory


@pytest.fixture
def memory_with_tracking_validator(memory):
    """Memory engine with a tracking validator that records all hook calls."""
    validator = TrackingValidator({})
    memory._operation_validator = validator
    return memory, validator


class TestOperationHooksParameters:
    """Tests for pre and post operation hooks receiving all user-provided parameters."""

    @pytest.mark.asyncio
    async def test_retain_pre_hook_receives_all_parameters(self, memory_with_tracking_validator):
        """Pre-retain hook receives all user-provided parameters."""
        memory, validator = memory_with_tracking_validator
        bank_id = "test-retain-params"
        ctx = RequestContext(api_key="test-key")
        contents = [{"content": "Test content", "context": "test context"}]
        document_id = "doc-123"

        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            document_id=document_id,
            fact_type_override="world",
            request_context=ctx,
        )

        assert len(validator.pre_retain_calls) == 1
        pre_ctx = validator.pre_retain_calls[0]

        # Verify all parameters are present
        assert pre_ctx.bank_id == bank_id
        # Note: contents is copied before document_id is applied to individual items
        assert len(pre_ctx.contents) == len(contents)
        assert pre_ctx.contents[0]["content"] == contents[0]["content"]
        assert pre_ctx.document_id == document_id
        assert pre_ctx.fact_type_override == "world"
        assert pre_ctx.request_context == ctx

    @pytest.mark.asyncio
    async def test_retain_post_hook_receives_all_parameters_and_result(self, memory_with_tracking_validator):
        """Post-retain hook receives all parameters plus the result."""
        memory, validator = memory_with_tracking_validator
        bank_id = "test-retain-post"
        ctx = RequestContext(api_key="test-key")
        contents = [{"content": "Test content for post hook"}]
        document_id = "doc-456"

        result = await memory.retain_batch_async(
            bank_id=bank_id,
            contents=contents,
            document_id=document_id,
            fact_type_override="experience",
            request_context=ctx,
        )

        assert len(validator.post_retain_calls) == 1
        post_result = validator.post_retain_calls[0]

        # Verify all parameters are present
        assert post_result.bank_id == bank_id
        assert post_result.document_id == document_id
        assert post_result.fact_type_override == "experience"
        assert post_result.request_context == ctx

        # Verify result data
        assert post_result.success is True
        assert post_result.error is None
        assert post_result.unit_ids == result  # Should match the return value

        # Verify actual LLM token usage is populated
        assert post_result.llm_input_tokens is not None
        assert post_result.llm_input_tokens > 0
        assert post_result.llm_output_tokens is not None
        assert post_result.llm_output_tokens > 0
        assert post_result.llm_total_tokens is not None
        assert post_result.llm_total_tokens == post_result.llm_input_tokens + post_result.llm_output_tokens

    @pytest.mark.asyncio
    async def test_recall_pre_hook_receives_all_parameters(self, memory_with_tracking_validator):
        """Pre-recall hook receives all user-provided parameters."""
        from datetime import datetime, timezone

        from hindsight_api.engine.memory_engine import Budget

        memory, validator = memory_with_tracking_validator
        bank_id = "test-recall-params"
        ctx = RequestContext(api_key="test-key")
        query = "test query"
        question_date = datetime(2024, 1, 15, tzinfo=timezone.utc)

        await memory.recall_async(
            bank_id=bank_id,
            query=query,
            budget=Budget.HIGH,
            max_tokens=2048,
            enable_trace=True,
            fact_type=["world", "experience"],
            question_date=question_date,
            include_entities=True,
            max_entity_tokens=300,
            include_chunks=True,
            max_chunk_tokens=4096,
            request_context=ctx,
        )

        assert len(validator.pre_recall_calls) == 1
        pre_ctx = validator.pre_recall_calls[0]

        # Verify all parameters are present
        assert pre_ctx.bank_id == bank_id
        assert pre_ctx.query == query
        assert pre_ctx.budget == Budget.HIGH
        assert pre_ctx.max_tokens == 2048
        assert pre_ctx.enable_trace is True
        assert pre_ctx.fact_types == ["world", "experience"]
        assert pre_ctx.question_date == question_date
        assert pre_ctx.include_entities is True
        assert pre_ctx.max_entity_tokens == 300
        assert pre_ctx.include_chunks is True
        assert pre_ctx.max_chunk_tokens == 4096
        assert pre_ctx.request_context == ctx

    @pytest.mark.asyncio
    async def test_recall_post_hook_receives_all_parameters_and_result(self, memory_with_tracking_validator):
        """Post-recall hook receives all parameters plus the result."""
        from hindsight_api.engine.memory_engine import Budget

        memory, validator = memory_with_tracking_validator
        bank_id = "test-recall-post"
        ctx = RequestContext(api_key="test-key")

        result = await memory.recall_async(
            bank_id=bank_id,
            query="test query for post",
            budget=Budget.LOW,
            max_tokens=1024,
            fact_type=["world"],
            request_context=ctx,
        )

        assert len(validator.post_recall_calls) == 1
        post_result = validator.post_recall_calls[0]

        # Verify all parameters are present
        assert post_result.bank_id == bank_id
        assert post_result.query == "test query for post"
        assert post_result.budget == Budget.LOW
        assert post_result.max_tokens == 1024
        assert post_result.fact_types == ["world"]
        assert post_result.request_context == ctx

        # Verify result data
        assert post_result.success is True
        assert post_result.error is None
        assert post_result.result == result  # Should match the return value

    @pytest.mark.asyncio
    async def test_reflect_pre_hook_receives_all_parameters(self, memory_with_tracking_validator):
        """Pre-reflect hook receives all user-provided parameters."""
        from hindsight_api.engine.memory_engine import Budget

        memory, validator = memory_with_tracking_validator
        bank_id = "test-reflect-params"
        ctx = RequestContext(api_key="test-key")

        try:
            await memory.reflect_async(
                bank_id=bank_id,
                query="test question",
                budget=Budget.MID,
                context="additional context",
                request_context=ctx,
            )
        except Exception:
            pass  # May fail if no data, but pre-hook should still be called

        assert len(validator.pre_reflect_calls) == 1
        pre_ctx = validator.pre_reflect_calls[0]

        # Verify all parameters are present
        assert pre_ctx.bank_id == bank_id
        assert pre_ctx.query == "test question"
        assert pre_ctx.budget == Budget.MID
        assert pre_ctx.context == "additional context"
        assert pre_ctx.request_context == ctx

    @pytest.mark.asyncio
    async def test_reflect_post_hook_receives_all_parameters_and_result(self, memory_with_tracking_validator):
        """Post-reflect hook receives all parameters plus the result on success."""
        from hindsight_api.engine.memory_engine import Budget

        memory, validator = memory_with_tracking_validator
        bank_id = "test-reflect-post"
        ctx = RequestContext(api_key="test-key")

        # Store some content first so reflect has something to work with
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Alice is a software engineer at Google."}],
            request_context=ctx,
        )

        result = await memory.reflect_async(
            bank_id=bank_id,
            query="What does Alice do?",
            budget=Budget.LOW,
            context="work context",
            request_context=ctx,
        )

        assert len(validator.post_reflect_calls) == 1
        post_result = validator.post_reflect_calls[0]

        # Verify all parameters are present
        assert post_result.bank_id == bank_id
        assert post_result.query == "What does Alice do?"
        assert post_result.budget == Budget.LOW
        assert post_result.context == "work context"
        assert post_result.request_context == ctx

        # Verify result data
        assert post_result.success is True
        assert post_result.error is None
        assert post_result.result == result  # Should match the return value
        assert post_result.result.text is not None

    @pytest.mark.asyncio
    async def test_post_hooks_called_in_order_after_pre_hooks(self, memory_with_tracking_validator):
        """Post hooks are called after pre hooks and after operation completes."""
        memory, validator = memory_with_tracking_validator
        bank_id = "test-hook-order"
        ctx = RequestContext()

        # Retain operation
        await memory.retain_batch_async(
            bank_id=bank_id,
            contents=[{"content": "Test content"}],
            request_context=ctx,
        )

        # Pre-hook should be called before post-hook
        assert len(validator.pre_retain_calls) == 1
        assert len(validator.post_retain_calls) == 1

        # Recall operation
        await memory.recall_async(
            bank_id=bank_id,
            query="test",
            fact_type=["world"],
            request_context=ctx,
        )

        # Use >= 1 since consolidation may trigger internal recall calls when observations are enabled
        assert len(validator.pre_recall_calls) >= 1
        assert len(validator.post_recall_calls) >= 1


class TestTenantExtension:
    """Tests for TenantExtension and ApiKeyTenantExtension."""

    @pytest.mark.asyncio
    async def test_api_key_tenant_extension_valid_key(self):
        """ApiKeyTenantExtension accepts valid API key."""
        ext = ApiKeyTenantExtension({"api_key": "secret-key-123"})

        result = await ext.authenticate(RequestContext(api_key="secret-key-123"))

        assert result.schema_name == "public"

    @pytest.mark.asyncio
    async def test_api_key_tenant_extension_invalid_key(self):
        """ApiKeyTenantExtension rejects invalid API key."""
        ext = ApiKeyTenantExtension({"api_key": "secret-key-123"})

        with pytest.raises(AuthenticationError) as exc_info:
            await ext.authenticate(RequestContext(api_key="wrong-key"))

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_key_tenant_extension_missing_key(self):
        """ApiKeyTenantExtension rejects missing API key."""
        ext = ApiKeyTenantExtension({"api_key": "secret-key-123"})

        with pytest.raises(AuthenticationError):
            await ext.authenticate(RequestContext(api_key=None))

    def test_api_key_tenant_extension_requires_config(self):
        """ApiKeyTenantExtension requires api_key in config."""
        with pytest.raises(ValueError) as exc_info:
            ApiKeyTenantExtension({})

        assert "HINDSIGHT_API_TENANT_API_KEY is required" in str(exc_info.value)


class TestMemoryEngineTenantAuth:
    """Tests for tenant authentication in MemoryEngine."""

    @pytest.mark.asyncio
    async def test_retain_requires_tenant_request_when_extension_configured(self, memory_with_tenant):
        """Retain fails without RequestContext when tenant extension is configured."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=None,  # Missing!
            )

        assert "RequestContext is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retain_succeeds_with_valid_tenant_request(self, memory_with_tenant):
        """Retain succeeds with valid RequestContext."""
        memory = memory_with_tenant

        # Should not raise
        await memory.retain_batch_async(
            bank_id="test-bank-tenant",
            contents=[{"content": "test content"}],
            request_context=RequestContext(api_key="test-api-key"),
        )

    @pytest.mark.asyncio
    async def test_retain_fails_with_invalid_api_key(self, memory_with_tenant):
        """Retain fails with invalid API key."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.retain_batch_async(
                bank_id="test-bank",
                contents=[{"content": "test"}],
                request_context=RequestContext(api_key="wrong-key"),
            )

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_recall_requires_tenant_request_when_extension_configured(self, memory_with_tenant):
        """Recall fails without RequestContext when tenant extension is configured."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError):
            await memory.recall_async(
                bank_id="test-bank",
                query="test query",
                fact_type=["world"],
                request_context=None,
            )

    @pytest.mark.asyncio
    async def test_list_audit_logs_fails_with_invalid_api_key(self, memory_with_tenant):
        """Audit log listing goes through tenant auth like other ops."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError) as exc_info:
            await memory.list_audit_logs(
                "test-bank",
                request_context=RequestContext(api_key="wrong-key"),
            )

        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_audit_log_stats_fails_with_invalid_api_key(self, memory_with_tenant):
        """Audit log stats goes through tenant auth like other ops."""
        memory = memory_with_tenant

        with pytest.raises(AuthenticationError):
            await memory.audit_log_stats(
                "test-bank",
                request_context=RequestContext(api_key="wrong-key"),
            )

    @pytest.mark.asyncio
    async def test_no_tenant_request_needed_without_extension(self, memory):
        """Operations work with empty RequestContext when no tenant extension configured."""
        # Should not raise - no tenant extension configured, just pass empty RequestContext
        await memory.retain_batch_async(
            bank_id="test-bank-no-tenant",
            contents=[{"content": "test content"}],
            request_context=RequestContext(),
        )


@pytest.fixture
def memory_with_tenant(memory):
    """Memory engine with a tenant extension (API key auth)."""
    tenant_ext = ApiKeyTenantExtension({"api_key": "test-api-key"})
    memory._tenant_extension = tenant_ext
    return memory


class SampleHttpExtension(HttpExtension):
    """Sample HTTP extension for testing that provides custom endpoints."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.started = False
        self.stopped = False
        self.request_count = 0

    async def on_startup(self):
        self.started = True

    async def on_shutdown(self):
        self.stopped = True

    def get_router(self, memory) -> APIRouter:
        router = APIRouter()

        @router.get("/hello")
        async def hello():
            self.request_count += 1
            return {"message": "Hello from extension!"}

        @router.get("/config")
        async def get_config():
            return {"config": self.config}

        @router.get("/health-check")
        async def extension_health():
            health = await memory.health_check()
            return {"extension": "healthy", "memory": health}

        @router.post("/echo")
        async def echo(data: dict):
            return {"echoed": data}

        return router


class TestHttpExtensionIntegration:
    """Tests for HTTP extension integration."""

    def test_load_http_extension(self, monkeypatch):
        """HttpExtension can be loaded from environment variable."""
        monkeypatch.setenv(
            "HINDSIGHT_API_HTTP_EXTENSION",
            "tests.test_extensions:SampleHttpExtension",
        )
        monkeypatch.setenv("HINDSIGHT_API_HTTP_CUSTOM_PARAM", "custom_value")

        ext = load_extension("HTTP", HttpExtension)

        assert ext is not None
        assert isinstance(ext, SampleHttpExtension)
        assert ext.config["custom_param"] == "custom_value"

    def test_http_extension_router_mounted_at_ext(self, memory):
        """HTTP extension router is mounted at /ext/."""
        from hindsight_api.api.http import create_app

        ext = SampleHttpExtension({"test_key": "test_value"})
        app = create_app(memory, initialize_memory=False, http_extension=ext)

        client = TestClient(app)

        # Extension endpoint should be accessible at /ext/
        response = client.get("/ext/hello")
        assert response.status_code == 200
        assert response.json() == {"message": "Hello from extension!"}

        # Should track request count
        assert ext.request_count == 1

        # Old path should NOT work
        response = client.get("/extension/hello")
        assert response.status_code == 404

    def test_http_extension_config_endpoint(self, memory):
        """Extension can expose its config via custom endpoint."""
        from hindsight_api.api.http import create_app

        ext = SampleHttpExtension({"api_key": "secret", "limit": "100"})
        app = create_app(memory, initialize_memory=False, http_extension=ext)

        client = TestClient(app)

        response = client.get("/ext/config")
        assert response.status_code == 200
        assert response.json()["config"]["api_key"] == "secret"
        assert response.json()["config"]["limit"] == "100"

    def test_http_extension_can_access_memory(self, memory):
        """Extension endpoints can access memory engine."""
        from hindsight_api.api.http import create_app

        ext = SampleHttpExtension({})
        app = create_app(memory, initialize_memory=False, http_extension=ext)

        client = TestClient(app)

        response = client.get("/ext/health-check")
        assert response.status_code == 200
        data = response.json()
        assert data["extension"] == "healthy"
        assert "memory" in data

    def test_http_extension_post_endpoint(self, memory):
        """Extension can handle POST requests with JSON body."""
        from hindsight_api.api.http import create_app

        ext = SampleHttpExtension({})
        app = create_app(memory, initialize_memory=False, http_extension=ext)

        client = TestClient(app)

        response = client.post("/ext/echo", json={"key": "value", "number": 42})
        assert response.status_code == 200
        assert response.json() == {"echoed": {"key": "value", "number": 42}}

    def test_http_extension_not_mounted_when_none(self, memory):
        """No extension routes when http_extension is None."""
        from hindsight_api.api.http import create_app

        app = create_app(memory, initialize_memory=False, http_extension=None)

        client = TestClient(app)

        # Extension endpoint should not exist
        response = client.get("/ext/hello")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_http_extension_lifecycle(self):
        """HTTP extension on_startup and on_shutdown are called."""
        ext = SampleHttpExtension({})

        assert not ext.started
        assert not ext.stopped

        await ext.on_startup()
        assert ext.started

        await ext.on_shutdown()
        assert ext.stopped

    def test_core_routes_still_work_with_extension(self, memory):
        """Core API routes still work when extension is mounted."""
        from hindsight_api.api.http import create_app

        ext = SampleHttpExtension({})
        app = create_app(memory, initialize_memory=False, http_extension=ext)

        client = TestClient(app)

        # Health endpoint should work
        response = client.get("/health")
        assert response.status_code in (200, 503)  # May be unhealthy if DB not connected

        # Banks list endpoint should work
        response = client.get("/v1/default/banks")
        assert response.status_code in (200, 500)  # May fail if DB not ready


# ============================================================================
# Precheck (pre-body-parse) tests
# ============================================================================
#
# The precheck() hook is wired as a FastAPI Depends on the billable POST
# routes. FastAPI resolves dependencies before deserialising the route's body
# parameter, so a rejecting precheck never causes the request body to be read
# or materialised in memory. The test below uses a Pydantic model_validator
# that records every parse to assert that body parsing never runs on the
# rejection path.


class RecordingPrecheckValidator(OperationValidatorExtension):
    """Validator that records every precheck call and can be configured to reject.

    Used to drive the FastAPI dependency that runs precheck() before body parse.
    The validate_* hooks below are required-abstract no-ops so the class is
    instantiable; the tests here only exercise precheck.
    """

    def __init__(self, *, reject: bool = False, status_code: int = 402, reason: str = "rejected by precheck") -> None:
        super().__init__(config={})
        self.reject = reject
        self.status_code = status_code
        self.reason = reason
        self.precheck_calls: list[PrecheckContext] = []

    async def precheck(self, ctx: PrecheckContext) -> ValidationResult:
        self.precheck_calls.append(ctx)
        if self.reject:
            return ValidationResult.reject(self.reason, status_code=self.status_code)
        return ValidationResult.accept()

    async def validate_retain(self, ctx: RetainContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_recall(self, ctx: RecallContext) -> ValidationResult:
        return ValidationResult.accept()

    async def validate_reflect(self, ctx: ReflectContext) -> ValidationResult:
        return ValidationResult.accept()


class TestPrecheckDefault:
    """The base OperationValidatorExtension.precheck is a no-op accept."""

    @pytest.mark.asyncio
    async def test_default_precheck_accepts(self):
        validator = RecordingPrecheckValidator(reject=False)
        # Bypass our override by calling the base implementation directly.
        ctx = PrecheckContext(
            operation=PrecheckOperation.RETAIN,
            bank_id="bank-x",
            request_context=RequestContext(),
        )
        result = await OperationValidatorExtension.precheck(validator, ctx)
        assert result.allowed is True
        assert result.reason is None


class TestPrecheckHttpWiring:
    """precheck() is wired as a FastAPI Depends on the billable POST routes.

    These tests do NOT use the heavy ``memory`` fixture (which requires a
    running pg0 + migrations). Instead they construct a minimal FastAPI
    app that mirrors the same Depends ordering used in
    ``hindsight_api.api.http`` (a ``Depends(precheck_for(...))`` resolved
    before the Pydantic body parameter), so the contract under test —
    "rejection happens before body parse" — can be exercised in isolation.

    The critical assertion in test_precheck_rejection_skips_body_parse is
    that a rejection response is returned without the request body being
    deserialised by Pydantic — i.e. the body parser was never invoked on
    the rejection path.
    """

    @staticmethod
    def _build_app(validator):
        """Mirror the precheck wiring from ``hindsight_api.api.http`` in a
        standalone FastAPI app."""
        from fastapi import Depends, FastAPI, HTTPException, Request
        from pydantic import BaseModel, model_validator

        from hindsight_api.extensions import PrecheckContext, PrecheckOperation
        from hindsight_api.models import RequestContext

        body_parses: list[str] = []

        class _RetainBody(BaseModel):
            items: list

            @model_validator(mode="before")
            @classmethod
            def _record(cls, v):
                body_parses.append("retain")
                return v

        class _RecallBody(BaseModel):
            query: str

            @model_validator(mode="before")
            @classmethod
            def _record(cls, v):
                body_parses.append("recall")
                return v

        class _ReflectBody(BaseModel):
            query: str

            @model_validator(mode="before")
            @classmethod
            def _record(cls, v):
                body_parses.append("reflect")
                return v

        async def _request_context() -> RequestContext:
            return RequestContext()

        def _precheck_for(operation: PrecheckOperation):
            async def _dep(
                bank_id: str,
                request: Request,
                request_context: RequestContext = Depends(_request_context),
            ) -> None:
                cl_header = request.headers.get("content-length")
                content_length: int | None = None
                if cl_header is not None:
                    try:
                        parsed = int(cl_header)
                    except ValueError:
                        parsed = -1
                    if parsed >= 0:
                        content_length = parsed
                ctx = PrecheckContext(
                    operation=operation,
                    bank_id=bank_id,
                    request_context=request_context,
                    content_length=content_length,
                )
                result = await validator.precheck(ctx)
                if not result.allowed:
                    raise HTTPException(
                        status_code=result.status_code,
                        detail=result.reason or "Operation not allowed",
                    )

            return _dep

        app = FastAPI()

        @app.post("/v1/default/banks/{bank_id}/memories")
        async def retain(
            bank_id: str,
            body: _RetainBody,
            _: None = Depends(_precheck_for(PrecheckOperation.RETAIN)),
        ):
            return {"ok": True, "bank_id": bank_id, "n": len(body.items)}

        @app.post("/v1/default/banks/{bank_id}/memories/recall")
        async def recall(
            bank_id: str,
            body: _RecallBody,
            _: None = Depends(_precheck_for(PrecheckOperation.RECALL)),
        ):
            return {"ok": True}

        @app.post("/v1/default/banks/{bank_id}/reflect")
        async def reflect(
            bank_id: str,
            body: _ReflectBody,
            _: None = Depends(_precheck_for(PrecheckOperation.REFLECT)),
        ):
            return {"ok": True}

        @app.get("/v1/default/banks/{bank_id}/memories/list")
        async def list_memories(bank_id: str):
            return {"ok": True}

        return app, body_parses

    def test_precheck_accept_lets_request_through_to_body_parse(self):
        validator = RecordingPrecheckValidator(reject=False)
        app, body_parses = self._build_app(validator)
        client = TestClient(app)

        resp = client.post(
            "/v1/default/banks/precheck-bank/memories",
            json={"items": [{"content": "x"}]},
        )
        assert resp.status_code == 200
        assert len(validator.precheck_calls) == 1
        assert validator.precheck_calls[0].operation == "retain"
        assert validator.precheck_calls[0].bank_id == "precheck-bank"
        assert body_parses == ["retain"]

    def test_precheck_rejection_returns_status_and_reason(self):
        validator = RecordingPrecheckValidator(reject=True, status_code=402, reason="Insufficient credits")
        app, _ = self._build_app(validator)
        client = TestClient(app)

        resp = client.post(
            "/v1/default/banks/precheck-bank/memories",
            json={"items": [{"content": "x"}]},
        )
        assert resp.status_code == 402
        assert resp.json()["detail"] == "Insufficient credits"

    def test_precheck_rejection_skips_body_parse(self):
        """The critical assertion: rejection happens before Pydantic
        deserialises the body. We send an oversized body and verify the
        body-parse counter never incremented.
        """
        validator = RecordingPrecheckValidator(reject=True, status_code=402, reason="rejected by precheck")
        app, body_parses = self._build_app(validator)
        client = TestClient(app)

        resp = client.post(
            "/v1/default/banks/precheck-bank/memories",
            json={"items": [{"content": "x" * 100_000} for _ in range(50)]},
        )
        assert resp.status_code == 402
        assert "rejected by precheck" in resp.json()["detail"]
        assert body_parses == [], (
            "request body was deserialised despite a rejecting precheck — "
            "the Depends-before-body-parse contract is broken"
        )

    def test_precheck_rejection_skips_body_parse_for_recall(self):
        validator = RecordingPrecheckValidator(reject=True, status_code=402, reason="rejected")
        app, body_parses = self._build_app(validator)
        client = TestClient(app)

        resp = client.post(
            "/v1/default/banks/precheck-bank/memories/recall",
            json={"query": "x" * 100_000},
        )
        assert resp.status_code == 402
        assert validator.precheck_calls[-1].operation == "recall"
        assert body_parses == []

    def test_precheck_rejection_skips_body_parse_for_reflect(self):
        validator = RecordingPrecheckValidator(reject=True, status_code=402, reason="rejected")
        app, body_parses = self._build_app(validator)
        client = TestClient(app)

        resp = client.post(
            "/v1/default/banks/precheck-bank/reflect",
            json={"query": "x" * 100_000},
        )
        assert resp.status_code == 402
        assert validator.precheck_calls[-1].operation == "reflect"
        assert body_parses == []

    def test_precheck_does_not_run_on_get(self):
        validator = RecordingPrecheckValidator(reject=True)
        app, _ = self._build_app(validator)
        client = TestClient(app)

        resp = client.get("/v1/default/banks/precheck-bank/memories/list")
        assert resp.status_code == 200
        assert len(validator.precheck_calls) == 0

    def test_precheck_context_carries_content_length(self):
        """Content-Length header is exposed to the precheck so a validator
        can make size-aware decisions (e.g. upper-bound cost estimate)
        before the body is deserialised."""
        validator = RecordingPrecheckValidator(reject=False)
        app, _ = self._build_app(validator)
        client = TestClient(app)

        # Body must contain at least 500 'x' bytes; check the surfaced
        # Content-Length is within a tight band around that floor (allows
        # for JSON envelope + httpx's serialisation choices without
        # depending on exact byte counts).
        payload = {"items": [{"content": "x" * 500}]}
        resp = client.post(
            "/v1/default/banks/precheck-bank/memories",
            json=payload,
        )
        assert resp.status_code == 200
        assert len(validator.precheck_calls) == 1
        ctx = validator.precheck_calls[0]
        assert ctx.content_length is not None
        assert 500 <= ctx.content_length <= 600

    def test_precheck_context_content_length_zero_is_not_none(self):
        """An empty POST body has Content-Length: 0. That should surface
        as the int 0, not None — None means 'unknown', 0 means 'known to
        be empty'."""
        validator = RecordingPrecheckValidator(reject=False)
        app, _ = self._build_app(validator)
        client = TestClient(app)

        # Empty body fails Pydantic parse (422), but precheck runs first
        # and records the Content-Length.
        client.post(
            "/v1/default/banks/precheck-bank/memories",
            content=b"",
            headers={"content-type": "application/json"},
        )
        assert len(validator.precheck_calls) >= 1
        ctx = validator.precheck_calls[-1]
        assert ctx.content_length == 0

    @pytest.mark.asyncio
    async def test_precheck_context_content_length_none_when_header_missing(self):
        """When the Content-Length header isn't set (e.g. chunked transfer
        encoding) the validator sees None, not a crash and not a default 0."""
        from starlette.requests import Request as _StarletteRequest

        from hindsight_api.extensions import PrecheckContext
        from hindsight_api.models import RequestContext

        validator = RecordingPrecheckValidator(reject=False)

        # Replicate the wiring's parse step inline so the test exercises
        # the same code-path semantics introduced in
        # ``hindsight_api.api.http._precheck_dep``.
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/default/banks/bank-x/memories",
            "headers": [],  # no content-length
            "query_string": b"",
        }
        req = _StarletteRequest(scope)
        cl_header = req.headers.get("content-length")
        content_length: int | None = None
        if cl_header is not None:
            try:
                parsed = int(cl_header)
            except ValueError:
                parsed = -1
            if parsed >= 0:
                content_length = parsed

        ctx = PrecheckContext(
            operation=PrecheckOperation.RETAIN,
            bank_id="bank-x",
            request_context=RequestContext(),
            content_length=content_length,
        )
        await validator.precheck(ctx)
        assert validator.precheck_calls[-1].content_length is None
