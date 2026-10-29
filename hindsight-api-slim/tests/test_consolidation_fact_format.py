"""The consolidation answer, one entry per new fact, turned into actions.

The model writes one entry per new fact and names each observation it repeats, changes or ends
(``effect``). These tests pin how that answer becomes creates, updates and skips — the
model's own verdicts decide, the code only reads them — and what the prompt is shown.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from hindsight_api.engine.consolidation.consolidator import (
    _BatchLLMResult,
    _build_fact_response_model,
    _consolidate_batch_with_llm,
    _CreateAction,
    _settle_skips,
    _SkipAction,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.response_models import LLMCallResult

POOL = {"o-port": "ledger-sink listens on port 8444.", "o-fav": "The user's favorite framework is Svelte."}


def _answer(*facts: dict[str, Any], ids: tuple[str, ...] = ("f1", "f2", "f3")) -> Any:
    return _build_fact_response_model(list(ids)).model_validate({"facts": list(facts)})


def _actions(answer: Any, later: set[tuple[str, str]] = frozenset(), ids: tuple[str, ...] = ("f1", "f2", "f3")) -> Any:
    return answer.to_actions(POOL, list(ids), lambda f, o: (f, o) in later)


def _change(oid: str, effect: str, text: str = "") -> dict[str, str]:
    return {"observation_id": oid, "observation_text": POOL.get(oid, ""), "effect": effect, "text": text}


def test_a_fact_touching_no_observation_is_created() -> None:
    actions = _actions(
        _answer(
            {
                "fact_id": "f1",
                "kind": "lasting",
                "existing": [],
                "adds": "yes",
                "new": ["ledger-sync listens on port 8443."],
            }
        )
    )
    assert actions.updates == []
    assert [(c.text, c.source_fact_ids) for c in actions.creates] == [("ledger-sync listens on port 8443.", ["f1"])]


def test_a_repeat_is_evidence_with_the_text_unchanged() -> None:
    actions = _actions(
        _answer(
            {
                "fact_id": "f1",
                "kind": "lasting",
                "existing": [_change("o-port", "repeats")],
                "adds": "no",
                "new": ["dup"],
            }
        )
    )
    assert [(u.observation_id, u.text, u.source_fact_ids) for u in actions.updates] == [
        ("o-port", POOL["o-port"], ["f1"])
    ]
    assert actions.creates == []  # adds: "no" drops what it wrote anyway


def test_a_change_rewrites_the_observation() -> None:
    actions = _actions(
        _answer(
            {
                "fact_id": "f1",
                "kind": "lasting",
                "existing": [_change("o-port", "changes", "ledger-sink listens on port 9443.")],
                "adds": "no",
                "new": [],
            }
        )
    )
    assert [u.text for u in actions.updates] == ["ledger-sink listens on port 9443."]


def test_an_older_fact_only_adds_the_earlier_state_and_creates_nothing() -> None:
    answer = _answer(
        {
            "fact_id": "f1",
            "kind": "lasting",
            "existing": [_change("o-fav", "changes", "Before that, their favorite was React.")],
            "adds": "yes",
            "new": ["The user's favorite framework is React."],
        }
    )
    actions = _actions(answer, later={("f1", "o-fav")})
    assert [u.text for u in actions.updates] == [
        "The user's favorite framework is Svelte. Before that, their favorite was React."
    ]
    assert actions.creates == []


def test_a_momentary_fact_is_skipped_as_chatter_whatever_it_wrote() -> None:
    actions = _actions(
        _answer({"fact_id": "f1", "kind": "momentary", "adds": "yes", "new": ["The user is having coffee."]})
    )
    assert actions.creates == []
    assert [s.fact_id for s in actions.skips] == ["f1"]


def test_a_restated_fact_joins_the_earlier_facts_create() -> None:
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "adds": "yes", "new": ["ledger-sync listens on port 8443."]},
            {"fact_id": "f2", "kind": "lasting", "adds": "no", "new": [], "same_as_fact": "f1"},
        )
    )
    assert [c.source_fact_ids for c in actions.creates] == [["f1", "f2"]]


def test_a_repeat_of_this_answers_own_create_under_an_invented_id_joins_it() -> None:
    created = "Companies are moving to open-weight models."
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "adds": "yes", "new": [created]},
            {
                "fact_id": "f2",
                "kind": "lasting",
                "existing": [
                    {"observation_id": "made-up", "observation_text": created, "effect": "repeats", "text": ""}
                ],
                "adds": "no",
                "new": [],
            },
        )
    )
    assert [c.source_fact_ids for c in actions.creates] == [["f1", "f2"]]
    assert actions.updates == []


def test_a_change_aimed_at_the_fact_itself_is_a_new_observation() -> None:
    actions = _actions(
        _answer({"fact_id": "f1", "kind": "lasting", "existing": [_change("f1", "changes", "Kotlin.")], "adds": "no"})
    )
    assert [(c.text, c.source_fact_ids) for c in actions.creates] == [("Kotlin.", ["f1"])]
    assert actions.updates == []


def test_entries_without_fact_ids_are_matched_by_position() -> None:
    actions = _actions(
        _answer(
            {"kind": "lasting", "adds": "yes", "new": ["a"]},
            {"kind": "lasting", "adds": "yes", "new": ["b"]},
            ids=("f1", "f2"),
        ),
        ids=("f1", "f2"),
    )
    assert [c.source_fact_ids for c in actions.creates] == [["f1"], ["f2"]]


def test_a_broken_entry_costs_only_itself() -> None:
    answer = _answer(
        ["f1", "positional", "garbage"], {"fact_id": "f2", "kind": "lasting", "adds": "yes", "new": ["kept"]}
    )
    assert [f.fact_id for f in answer.facts] == ["f2"]


def test_the_schema_pins_ids_and_verdicts() -> None:
    schema = str(_build_fact_response_model(["f1"]).model_json_schema())
    for value in ("'f1'", "'lasting'", "'momentary'", "'unrelated'", "'repeats'", "'changes'", "'ends'"):
        assert value in schema


def test_a_chatter_skip_settles_a_fact_no_action_cites() -> None:
    result = _BatchLLMResult(
        creates=[_CreateAction(text="Pages go to #oncall.", source_fact_ids=["f2"])],
        skips=[_SkipAction(fact_id="f1"), _SkipAction(fact_id="f2")],
    )
    assert _settle_skips(result) == {"f1"}


async def test_the_prompt_uses_short_ids_oldest_first_and_marks_later_observations() -> None:
    def at(day: int) -> datetime:
        return datetime(2026, 10, day, tzinfo=timezone.utc)

    observation = SimpleNamespace(
        id="o-fav", text=POOL["o-fav"], occurred_start=None, occurred_end=None, mentioned_at=at(20), source_fact_ids=[]
    )
    sent: dict[str, Any] = {}

    class _LLM:
        async def call(self, **kwargs: Any) -> Any:
            sent.update(kwargs)
            answer = {
                "facts": [
                    {
                        "fact_id": "F2",
                        "kind": "lasting",
                        "existing": [{"observation_id": "O1", "effect": "changes", "text": "Now React."}],
                        "adds": "no",
                    }
                ]
            }
            return LLMCallResult(content=kwargs["response_format"].model_validate(answer))

    config = SimpleNamespace(
        llm_output_language=None,
        observations_mission=None,
        llm_supports_max_items=True,
        consolidation_max_attempts=1,
        consolidation_llm_max_retries=None,
        llm_temperature_consolidation=0.0,
        llm_strict_schema_consolidation=False,
        consolidation_max_completion_tokens=None,
    )
    memories = [
        {"id": "f-new", "text": "newer", "mentioned_at": at(25)},
        {"id": "f-old", "text": "older", "mentioned_at": at(10)},
    ]
    result = await _consolidate_batch_with_llm(
        _LLM(), memories, [observation], {}, config, closest={"f-old": ["o-fav"], "f-new": ["o-fav"]}
    )

    user = sent["messages"][-1]["content"]
    assert "f-old" not in user and "o-fav" not in user
    assert user.index("[F1] older") < user.index("[F2] newer")
    assert (
        "[F1] older (mentioned_at=2026-10-10T00:00Z)\n  closest existing observations: O1 (stated after this fact)"
        in user
    )
    assert "closest existing observations: O1\n" in user + "\n"  # F2 is the more recent: unmarked
    assert '"id": "O1"' in user
    # The answer's short ids come back as the real ones.
    assert [(u.observation_id, u.source_fact_ids, u.text) for u in result.updates] == [
        ("o-fav", ["f-new"], "Now React.")
    ]
    assert sent["messages"][0]["content"] == build_consolidation_system_prompt()
