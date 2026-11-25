"""An older client must parse enum values the server added after its release (#6862)."""

from hindsight_client_api.models.budget import Budget
from hindsight_client_api.models.document_event_kind import DocumentEventKind
from hindsight_client_api.models.mental_model_refresh_trace import MentalModelRefreshTrace


def test_unknown_inline_enum_value_parses_as_string():
    trace = MentalModelRefreshTrace.from_dict(
        {
            "recorded_at": "2026-01-01T00:00:00Z",
            "effective_mode": "delta",
            "mode_fallback_reason": "a_reason_added_later",
            "outcome": "an_outcome_added_later",
        }
    )
    assert trace.mode_fallback_reason == "a_reason_added_later"
    assert trace.outcome == "an_outcome_added_later"


def test_unknown_named_enum_value_parses():
    kind = DocumentEventKind.from_json('"a_kind_added_later"')
    assert kind == "a_kind_added_later"
    assert kind.value == "a_kind_added_later"
    assert DocumentEventKind("retain") is DocumentEventKind.RETAIN
    assert Budget("low") is Budget.LOW


def test_unknown_union_shape_is_kept_raw():
    from hindsight_client_api.models.mental_model_trigger_output import MentalModelTriggerOutput

    trigger = MentalModelTriggerOutput.from_dict({"tag_groups": [{"tags": ["a"]}, {"xor": [{"tags": ["b"]}]}]})
    assert trigger.tag_groups[1].actual_instance == {"xor": [{"tags": ["b"]}]}
    assert trigger.to_dict()["tag_groups"][1] == {"xor": [{"tags": ["b"]}]}
