"""
OpenTelemetry metrics instrumentation for Hindsight API.

This module provides metrics for:
- Operation latency (retain, recall, reflect) with percentiles
- Token usage (input/output) per operation
- Per-bank granularity via labels
- LLM call latency and token usage with scope dimension
- HTTP request metrics (latency, count by endpoint/method/status)
- Process metrics (CPU, memory, file descriptors, threads)
- Database connection pool metrics
"""

import logging
import os
import resource
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource

if TYPE_CHECKING:
    import asyncpg


def _get_tenant() -> str:
    """Get current tenant (schema) from context for metrics labeling."""
    # Import here to avoid circular imports
    from hindsight_api.engine.memory_engine import get_current_schema

    return get_current_schema()


# Custom bucket boundaries for operation duration (in seconds)
# Fine granularity in 0-30s range where most operations complete
DURATION_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0)

# LLM duration buckets (finer granularity for faster LLM calls)
LLM_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0)

# HTTP request duration buckets (millisecond-level for fast endpoints)
HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def get_token_bucket(token_count: int) -> str:
    """
    Convert a token count to a bucket label for use as a dimension.

    This allows analyzing token usage patterns without high-cardinality issues.

    Buckets:
    - "0-100": Very small requests/responses
    - "100-500": Small requests/responses
    - "500-1k": Medium requests/responses
    - "1k-5k": Large requests/responses
    - "5k-10k": Very large requests/responses
    - "10k-50k": Huge requests/responses
    - "50k+": Extremely large requests/responses

    Args:
        token_count: Number of tokens

    Returns:
        Bucket label string
    """
    if token_count < 100:
        return "0-100"
    elif token_count < 500:
        return "100-500"
    elif token_count < 1000:
        return "500-1k"
    elif token_count < 5000:
        return "1k-5k"
    elif token_count < 10000:
        return "5k-10k"
    elif token_count < 50000:
        return "10k-50k"
    else:
        return "50k+"


logger = logging.getLogger(__name__)

# Global meter instance
_meter = None


def initialize_metrics(service_name: str = "hindsight-api", service_version: str = "1.0.0"):
    """
    Initialize OpenTelemetry metrics with Prometheus exporter.

    This should be called once during application startup.

    Args:
        service_name: Name of the service for resource attributes
        service_version: Version of the service

    Returns:
        PrometheusMetricReader instance (for accessing metrics endpoint)
    """
    global _meter

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
        }
    )

    # Create Prometheus metric reader
    prometheus_reader = PrometheusMetricReader()

    # Create view with custom bucket boundaries for duration histogram
    duration_view = View(
        instrument_name="hindsight.operation.duration",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=DURATION_BUCKETS),
    )

    # Create view with custom bucket boundaries for LLM duration histogram
    llm_duration_view = View(
        instrument_name="hindsight.llm.duration",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=LLM_DURATION_BUCKETS),
    )

    # Create view with custom bucket boundaries for HTTP request duration histogram
    http_duration_view = View(
        instrument_name="hindsight.http.duration",
        aggregation=ExplicitBucketHistogramAggregation(boundaries=HTTP_DURATION_BUCKETS),
    )

    # Create meter provider with Prometheus exporter and custom views
    provider = MeterProvider(
        resource=resource,
        metric_readers=[prometheus_reader],
        views=[duration_view, llm_duration_view, http_duration_view],
    )

    # Set the global meter provider
    metrics.set_meter_provider(provider)

    # Get meter for this application
    _meter = metrics.get_meter(__name__)

    return prometheus_reader


def get_meter():
    """Get the global meter instance."""
    if _meter is None:
        raise RuntimeError("Metrics not initialized. Call initialize_metrics() first.")
    return _meter


class MetricsCollectorBase:
    """Base class for metrics collectors."""

    @contextmanager
    def record_operation(
        self,
        operation: str,
        bank_id: str,
        source: str = "api",
        budget: str | None = None,
        max_tokens: int | None = None,
    ):
        """Context manager to record operation duration and status."""
        raise NotImplementedError

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ):
        """
        Record metrics for an LLM call.

        Args:
            provider: LLM provider name (openai, anthropic, gemini, groq, ollama, lmstudio)
            model: Model name
            scope: Scope identifier (e.g., "memory", "reflect", "consolidation")
            duration: Call duration in seconds
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            success: Whether the call was successful
        """
        raise NotImplementedError

    @contextmanager
    def record_http_request(self, method: str, endpoint: str, status_code_getter: Callable[[], int]):
        """Context manager to record HTTP request metrics."""
        raise NotImplementedError

    def set_db_pool(self, pool: "asyncpg.Pool"):
        """Set the database pool for metrics collection."""
        pass


class NoOpMetricsCollector(MetricsCollectorBase):
    """No-op metrics collector that does nothing. Used when metrics are disabled."""

    @contextmanager
    def record_operation(
        self,
        operation: str,
        bank_id: str,
        source: str = "api",
        budget: str | None = None,
        max_tokens: int | None = None,
    ):
        """No-op context manager."""
        yield

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ):
        """No-op LLM call recording."""
        pass

    @contextmanager
    def record_http_request(self, method: str, endpoint: str, status_code_getter: Callable[[], int]):
        """No-op HTTP request recording."""
        yield


class MetricsCollector(MetricsCollectorBase):
    """
    Collector for Hindsight API metrics.

    Provides methods to record latency and token usage for operations.
    """

    def __init__(self):
        self.meter = get_meter()

        # Operation latency histogram (in seconds)
        # Records duration of retain, recall, reflect operations
        self.operation_duration = self.meter.create_histogram(
            name="hindsight.operation.duration", description="Duration of Hindsight operations in seconds", unit="s"
        )

        # Operation counter (success/failure)
        self.operation_total = self.meter.create_counter(
            name="hindsight.operation.total", description="Total number of operations executed", unit="operations"
        )

        # LLM call latency histogram (in seconds)
        # Records duration of LLM API calls with provider, model, and scope dimensions
        self.llm_duration = self.meter.create_histogram(
            name="hindsight.llm.duration", description="Duration of LLM API calls in seconds", unit="s"
        )

        # LLM token usage counters with bucket labels
        self.llm_tokens_input = self.meter.create_counter(
            name="hindsight.llm.tokens.input", description="Number of input tokens for LLM calls", unit="tokens"
        )

        self.llm_tokens_output = self.meter.create_counter(
            name="hindsight.llm.tokens.output", description="Number of output tokens from LLM calls", unit="tokens"
        )

        # LLM call counter (success/failure)
        self.llm_calls_total = self.meter.create_counter(
            name="hindsight.llm.calls.total", description="Total number of LLM API calls", unit="calls"
        )

        # HTTP request metrics
        self.http_request_duration = self.meter.create_histogram(
            name="hindsight.http.duration", description="Duration of HTTP requests in seconds", unit="s"
        )

        self.http_requests_total = self.meter.create_counter(
            name="hindsight.http.requests.total", description="Total number of HTTP requests", unit="requests"
        )

        self.http_requests_in_progress = self.meter.create_up_down_counter(
            name="hindsight.http.requests.in_progress",
            description="Number of HTTP requests in progress",
            unit="requests",
        )

        # Process metrics (observable gauges - collected on scrape)
        self._setup_process_metrics()

        # DB pool metrics holder (set via set_db_pool)
        self._db_pool: "asyncpg.Pool | None" = None

    @contextmanager
    def record_operation(
        self,
        operation: str,
        bank_id: str,
        source: str = "api",
        budget: str | None = None,
        max_tokens: int | None = None,
    ):
        """
        Context manager to record operation duration and status.

        Usage:
            with metrics.record_operation("recall", bank_id="user123", source="api", budget="mid", max_tokens=4096):
                # ... perform operation
                pass

        Args:
            operation: Operation name (retain, recall, reflect, consolidation)
            bank_id: Memory bank ID
            source: Source of the operation (api, reflect, internal)
            budget: Optional budget level (low, mid, high)
            max_tokens: Optional max tokens for the operation
        """
        start_time = time.time()
        attributes = {
            "operation": operation,
            "bank_id": bank_id,
            "source": source,
            "tenant": _get_tenant(),
        }
        if budget:
            attributes["budget"] = budget
        if max_tokens:
            attributes["max_tokens"] = str(max_tokens)

        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            attributes["success"] = str(success).lower()

            # Record duration
            self.operation_duration.record(duration, attributes)

            # Record operation count
            self.operation_total.add(1, attributes)

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ):
        """
        Record metrics for an LLM call.

        Args:
            provider: LLM provider name (openai, anthropic, gemini, groq, ollama, lmstudio)
            model: Model name
            scope: Scope identifier (e.g., "memory", "reflect", "consolidation")
            duration: Call duration in seconds
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            success: Whether the call was successful
        """
        # Base attributes for all metrics
        base_attributes = {
            "provider": provider,
            "model": model,
            "scope": scope,
            "success": str(success).lower(),
            "tenant": _get_tenant(),
        }

        # Record duration
        self.llm_duration.record(duration, base_attributes)

        # Record call count
        self.llm_calls_total.add(1, base_attributes)

        # Record tokens with bucket labels for cardinality control
        if input_tokens > 0:
            input_attributes = {
                **base_attributes,
                "token_bucket": get_token_bucket(input_tokens),
            }
            self.llm_tokens_input.add(input_tokens, input_attributes)

        if output_tokens > 0:
            output_attributes = {
                **base_attributes,
                "token_bucket": get_token_bucket(output_tokens),
            }
            self.llm_tokens_output.add(output_tokens, output_attributes)

    @contextmanager
    def record_http_request(self, method: str, endpoint: str, status_code_getter: Callable[[], int]):
        """
        Context manager to record HTTP request metrics.

        Usage:
            status_code = [200]  # Use list for mutability
            with metrics.record_http_request("GET", "/api/banks", lambda: status_code[0]):
                # ... handle request
                status_code[0] = response.status_code

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: Request endpoint path
            status_code_getter: Callable that returns the status code after request completes
        """
        start_time = time.time()
        base_attributes = {"method": method, "endpoint": endpoint}

        # Track in-progress
        self.http_requests_in_progress.add(1, base_attributes)

        try:
            yield
        finally:
            duration = time.time() - start_time
            status_code = status_code_getter()
            status_class = f"{status_code // 100}xx"

            # Get tenant from context (may be set during request processing)
            tenant = _get_tenant()

            attributes = {
                **base_attributes,
                "status_code": str(status_code),
                "status_class": status_class,
                "tenant": tenant,
            }

            # Record duration and count
            self.http_request_duration.record(duration, attributes)
            self.http_requests_total.add(1, attributes)

            # Decrement in-progress
            self.http_requests_in_progress.add(-1, base_attributes)

    def _setup_process_metrics(self):
        """Set up observable gauges for process metrics."""

        def get_cpu_times(_options):
            """Get process CPU times."""
            try:
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                yield metrics.Observation(rusage.ru_utime, {"type": "user"})
                yield metrics.Observation(rusage.ru_stime, {"type": "system"})
            except Exception:
                pass

        def get_memory_usage(_options):
            """Get process memory usage in bytes."""
            try:
                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # ru_maxrss is in kilobytes on Linux, bytes on macOS
                max_rss = rusage.ru_maxrss
                if os.uname().sysname == "Linux":
                    max_rss *= 1024  # Convert KB to bytes
                yield metrics.Observation(max_rss, {"type": "rss_max"})
            except Exception:
                pass

        def get_open_file_descriptors(_options):
            """Get number of open file descriptors."""
            try:
                # Try to count open FDs by checking /proc on Linux
                if os.path.exists("/proc/self/fd"):
                    count = len(os.listdir("/proc/self/fd"))
                    yield metrics.Observation(count)
                else:
                    # Fallback: use resource limits
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    yield metrics.Observation(soft, {"limit": "soft"})
            except Exception:
                pass

        def get_thread_count(_options):
            """Get number of active threads."""
            try:
                yield metrics.Observation(threading.active_count())
            except Exception:
                pass

        # Create observable gauges
        self.meter.create_observable_gauge(
            name="hindsight.process.cpu.seconds",
            callbacks=[get_cpu_times],
            description="Process CPU time in seconds",
            unit="s",
        )

        self.meter.create_observable_gauge(
            name="hindsight.process.memory.bytes",
            callbacks=[get_memory_usage],
            description="Process memory usage in bytes",
            unit="By",
        )

        self.meter.create_observable_gauge(
            name="hindsight.process.open_fds",
            callbacks=[get_open_file_descriptors],
            description="Number of open file descriptors",
            unit="{fds}",
        )

        self.meter.create_observable_gauge(
            name="hindsight.process.threads",
            callbacks=[get_thread_count],
            description="Number of active threads",
            unit="{threads}",
        )

    def set_db_pool(self, pool: "asyncpg.Pool"):
        """
        Set the database pool for metrics collection.

        Args:
            pool: asyncpg connection pool instance
        """
        self._db_pool = pool
        self._setup_db_pool_metrics()

    def _setup_db_pool_metrics(self):
        """Set up observable gauges for database pool metrics."""

        def get_pool_size(_options):
            """Get current pool size."""
            if self._db_pool is not None:
                try:
                    yield metrics.Observation(self._db_pool.get_size())
                except Exception:
                    pass

        def get_pool_free_size(_options):
            """Get number of free connections in pool."""
            if self._db_pool is not None:
                try:
                    yield metrics.Observation(self._db_pool.get_idle_size())
                except Exception:
                    pass

        def get_pool_min_size(_options):
            """Get pool minimum size."""
            if self._db_pool is not None:
                try:
                    yield metrics.Observation(self._db_pool.get_min_size())
                except Exception:
                    pass

        def get_pool_max_size(_options):
            """Get pool maximum size."""
            if self._db_pool is not None:
                try:
                    yield metrics.Observation(self._db_pool.get_max_size())
                except Exception:
                    pass

        # Create observable gauges for pool metrics
        self.meter.create_observable_gauge(
            name="hindsight.db.pool.size",
            callbacks=[get_pool_size],
            description="Current number of connections in the pool",
            unit="{connections}",
        )

        self.meter.create_observable_gauge(
            name="hindsight.db.pool.idle",
            callbacks=[get_pool_free_size],
            description="Number of idle connections in the pool",
            unit="{connections}",
        )

        self.meter.create_observable_gauge(
            name="hindsight.db.pool.min",
            callbacks=[get_pool_min_size],
            description="Minimum pool size",
            unit="{connections}",
        )

        self.meter.create_observable_gauge(
            name="hindsight.db.pool.max",
            callbacks=[get_pool_max_size],
            description="Maximum pool size",
            unit="{connections}",
        )


# Global metrics collector instance (defaults to no-op)
_metrics_collector: MetricsCollectorBase = NoOpMetricsCollector()


def get_metrics_collector() -> MetricsCollectorBase:
    """
    Get the global metrics collector instance.

    Returns a no-op collector if metrics are not initialized.
    """
    return _metrics_collector


def create_metrics_collector() -> MetricsCollector:
    """
    Create and set the global metrics collector.

    Should be called after initialize_metrics().
    """
    global _metrics_collector
    _metrics_collector = MetricsCollector()
    return _metrics_collector
