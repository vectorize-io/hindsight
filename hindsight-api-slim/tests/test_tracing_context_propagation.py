"""Incoming W3C trace-context propagation on the HTTP API (issue #3604).

Before this, every memory operation opened a new root trace, so a caller's
request and the Hindsight work it triggered were two unrelated traces. The
FastAPI/ASGI instrumentation installed by ``_instrument_app_for_tracing``
extracts ``traceparent`` and opens a SERVER span, which the engine's existing
spans then nest under through the ambient context.

These tests assert the observable contract — what trace the handler runs in —
rather than the middleware wiring, so they hold whether or not an SDK tracer
provider happens to be installed in the test process.
"""

import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import trace

from hindsight_api.api.http import _instrument_app_for_tracing
from hindsight_api.config import _get_raw_config

# A well-formed W3C traceparent, from the spec's own example.
REMOTE_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
REMOTE_SPAN_ID = "00f067aa0ba902b7"
TRACEPARENT = f"00-{REMOTE_TRACE_ID}-{REMOTE_SPAN_ID}-01"


def _config(**overrides):
    """Raw config with tracing switched on unless a test says otherwise."""
    defaults = {
        "otel_traces_enabled": True,
        "otel_exporter_otlp_endpoint": "http://localhost:4318",
    }
    return dataclasses.replace(_get_raw_config(), **{**defaults, **overrides})


def _app_reporting_current_trace(config) -> FastAPI:
    """An app whose routes report the trace they are executing in."""
    app = FastAPI()

    def _current_trace_id() -> dict[str, str]:
        span_context = trace.get_current_span().get_span_context()
        return {"trace_id": trace.format_trace_id(span_context.trace_id)}

    @app.get("/v1/probe")
    def probe():
        return _current_trace_id()

    @app.get("/health")
    def health():
        return _current_trace_id()

    _instrument_app_for_tracing(app, config)
    return app


def test_incoming_traceparent_becomes_the_handlers_trace(monkeypatch):
    """A caller's traceparent is extracted, so Hindsight's work joins its trace."""
    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    app = _app_reporting_current_trace(_config())

    with TestClient(app) as client:
        response = client.get("/v1/probe", headers={"traceparent": TRACEPARENT})

    assert response.json()["trace_id"] == REMOTE_TRACE_ID


def test_request_without_traceparent_starts_its_own_trace(monkeypatch):
    """Un-instrumented callers are unaffected — the change is backwards compatible."""
    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    app = _app_reporting_current_trace(_config())

    with TestClient(app) as client:
        response = client.get("/v1/probe")

    assert response.status_code == 200
    assert response.json()["trace_id"] != REMOTE_TRACE_ID


def test_probe_endpoints_are_excluded_from_tracing(monkeypatch):
    """Health/metrics scrapes must not flood the trace stream."""
    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    app = _app_reporting_current_trace(_config())

    with TestClient(app) as client:
        response = client.get("/health", headers={"traceparent": TRACEPARENT})

    # Excluded URLs skip the middleware entirely, so the extracted context is
    # never made current and the handler runs outside any trace.
    assert response.json()["trace_id"] != REMOTE_TRACE_ID


def test_excluded_urls_are_configurable(monkeypatch):
    """OTEL_PYTHON_FASTAPI_EXCLUDED_URLS overrides the built-in exclusions."""
    monkeypatch.setenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", "nothing-matches-this")
    app = _app_reporting_current_trace(_config())

    with TestClient(app) as client:
        response = client.get("/health", headers={"traceparent": TRACEPARENT})

    assert response.json()["trace_id"] == REMOTE_TRACE_ID


def test_app_is_not_instrumented_when_tracing_is_disabled():
    """No tracing configured means no instrumentation overhead."""
    app = _app_reporting_current_trace(_config(otel_traces_enabled=False))

    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is False


def test_app_is_not_instrumented_without_an_endpoint():
    """Enabled-but-unexportable tracing leaves the app alone, matching the bootstrap."""
    app = _app_reporting_current_trace(_config(otel_exporter_otlp_endpoint=None))

    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is False


def test_instrumentation_failure_does_not_break_app_creation(monkeypatch):
    """Instrumentation is best-effort: a failure must not stop the API booting."""
    import opentelemetry.instrumentation.fastapi as fastapi_instrumentation

    monkeypatch.delenv("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", raising=False)
    monkeypatch.setattr(
        fastapi_instrumentation.FastAPIInstrumentor,
        "instrument_app",
        staticmethod(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    app = _app_reporting_current_trace(_config())  # must not raise

    with TestClient(app) as client:
        assert client.get("/v1/probe").status_code == 200
