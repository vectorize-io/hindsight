"""#5938: an UPDATE's ``reason`` is generated first, and it and ``current_text`` are advertised as required.

Field order is generation order. Written after ``text``, the reason only rationalised the edit and
gemini-3.1-flash-lite let an older fact overwrite a newer observation every time; written first,
opening with the time comparison, it keeps the newer value. Both fields are optional for
validation, so a grammar-constrained model skipped them: required in the schema, the reason does
the time check, and the copied ``current_text`` stops the rewrite from erasing what the fact did not
touch. The behaviour is graded by the system evals ``recency-same-day-amendment-arrives-first`` and
``noisy-service-catalogue``; this pins the schema they depend on.
"""

from hindsight_api.engine.consolidation.consolidator import _build_response_model, _UpdateAction


def _order(props: dict) -> list[str]:
    return [p for p in props if p in ("reason", "current_text", "text")]


def test_update_reason_comes_before_text() -> None:
    assert _order(_UpdateAction.model_json_schema()["properties"]) == ["reason", "current_text", "text"]


def test_constrained_update_keeps_the_order() -> None:
    schema = _build_response_model(3, fact_ids=["f1"]).model_json_schema()
    assert _order(schema["$defs"]["_Update"]["properties"]) == ["reason", "current_text", "text"]


def test_fields_are_advertised_as_required_but_still_default() -> None:
    defs = _build_response_model(3, fact_ids=["f1"]).model_json_schema()["$defs"]
    assert {"reason", "current_text"} <= set(defs["_Update"]["required"])
    assert "reason" in defs["_Create"]["required"]
    update = _UpdateAction(observation_id="o", text="t", source_fact_ids=["f1"])
    assert (update.reason, update.current_text) == ("", "")
