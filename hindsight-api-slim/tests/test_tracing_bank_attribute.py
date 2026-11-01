"""Every span inside a bank-scoped operation carries hindsight.bank_id.

Per-bank routing (an OTel collector sending each bank's spans to its own backend project)
keys on this attribute. The model-call spans that carry token counts are opened deep
inside the operation, so without the processor only the operation span had the bank and
every costed span routed to the catch-all.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from hindsight_api.engine.llm_trace import operation_run
from hindsight_api.tracing import bank_id_span_processor


def _tracer():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(bank_id_span_processor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__), exporter


def test_child_spans_in_a_bank_operation_get_the_bank():
    tracer, exporter = _tracer()
    with operation_run("bank-a", "retain"):
        with tracer.start_as_current_span("hindsight.retain"):
            tracer.start_span("hindsight.retain_extract_facts").end()

    assert {s.name: s.attributes.get("hindsight.bank_id") for s in exporter.get_finished_spans()} == {
        "hindsight.retain": "bank-a",
        "hindsight.retain_extract_facts": "bank-a",
    }


def test_bank_bound_without_an_operation_run_is_used():
    """Worker tasks bind the bank (``_bind_bank_id``) before any operation run exists."""
    from hindsight_api.schema_context import _current_bank_id

    tracer, exporter = _tracer()
    token = _current_bank_id.set("bank-b")
    try:
        tracer.start_span("hindsight.consolidation").end()
    finally:
        _current_bank_id.reset(token)

    assert exporter.get_finished_spans()[0].attributes["hindsight.bank_id"] == "bank-b"


def test_spans_outside_a_bank_operation_get_no_bank():
    tracer, exporter = _tracer()
    tracer.start_span("hindsight.startup_check").end()

    assert "hindsight.bank_id" not in exporter.get_finished_spans()[0].attributes


def test_an_explicit_bank_attribute_is_kept():
    tracer, exporter = _tracer()
    with operation_run("bank-a", "reflect"):
        tracer.start_span("x", attributes={"hindsight.bank_id": "bank-b"}).end()

    assert exporter.get_finished_spans()[0].attributes["hindsight.bank_id"] == "bank-b"
