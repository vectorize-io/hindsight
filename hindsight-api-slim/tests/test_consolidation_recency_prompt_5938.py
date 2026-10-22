"""#5938: what the consolidation prompt and schema give the model to decide recency.

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


def test_prompt_times_share_one_format() -> None:
    # A fact's time arrives as a datetime with sub-second offsets, an observation's as an ISO string:
    # the model compares them, so both must read the same (#5938).
    from datetime import datetime, timezone

    from hindsight_api.engine.consolidation.consolidator import _prompt_time

    fact = datetime(2026, 10, 21, 9, 48, 0, 10000, tzinfo=timezone.utc)
    assert _prompt_time(fact) == "2026-10-21T09:48Z"
    assert _prompt_time("2026-10-21T10:48:00+00:00") == "2026-10-21T10:48Z"
    assert _prompt_time(None) is None


def test_observation_newer_than_every_input_fact_is_marked() -> None:
    # The model is told the comparison instead of making it (#5938). A message's own time can live
    # only in occurred_* when its session started earlier (#2550), so the latest time counts.
    from datetime import datetime, timezone

    from hindsight_api.engine.consolidation.consolidator import _build_observations_for_llm, _latest_time
    from hindsight_api.engine.response_models import MemoryFact

    amendment = MemoryFact(
        id="new", text="25 free units", fact_type="observation", mentioned_at="2026-10-21T10:48:00+00:00"
    )
    older = MemoryFact(
        id="old", text="no free units", fact_type="observation", mentioned_at="2026-10-21T08:00:00+00:00"
    )
    batch = _latest_time(datetime(2026, 10, 21, 9, 48, tzinfo=timezone.utc), "2026-10-21T00:00:00+00:00")

    marked, unmarked = _build_observations_for_llm([amendment, older], {}, batch)
    assert marked["newer_than_input_facts"] is True
    assert "newer_than_input_facts" not in unmarked
    # A fact stated in an older session but sent later is newer than an observation in between.
    assert _latest_time("2026-07-01T09:00:00+00:00", "2026-07-10T18:30:00+00:00").day == 10
