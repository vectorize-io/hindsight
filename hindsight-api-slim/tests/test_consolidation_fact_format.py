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
    _FactBatchResponse,
    _settle_skips,
    _SkipAction,
)
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.response_models import LLMCallResult

POOL = {"o-port": "ledger-sink listens on port 8444.", "o-fav": "The user's favorite framework is Svelte."}


def _answer(*facts: dict[str, Any], ids: tuple[str, ...] = ("f1", "f2", "f3")) -> Any:
    return _build_fact_response_model(list(ids)).model_validate({"facts": list(facts)})


def _actions(
    answer: Any,
    later: set[tuple[str, str]] = frozenset(),
    ids: tuple[str, ...] = ("f1", "f2", "f3"),
    unmarked: set[tuple[str, str]] = frozenset(),
) -> Any:
    """``later``: observations stated after the fact, marked so in the prompt unless also in ``unmarked``."""
    texts = {fid: f"{fid}: Fact {fid} says so" for fid in ids}
    return answer.to_actions(POOL, texts, lambda f, o: (f, o) in later, lambda f, o: (f, o) not in unmarked)


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
            "existing": [_change("o-fav", "changes", "Until it was removed, React.")],
            "adds": "yes",
            "new": ["The user's favorite framework is React."],
        }
    )
    actions = _actions(answer, later={("f1", "o-fav")})
    # The note is the fact's own text, whatever the model wrote.
    assert [u.text for u in actions.updates] == ["The user's favorite framework is Svelte. (f1: Fact f1 says so)"]
    assert actions.creates == []


def test_two_older_facts_both_land_as_earlier_states() -> None:
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "existing": [_change("o-fav", "changes", "Before, Vue.")]},
            {"fact_id": "f2", "kind": "lasting", "existing": [_change("o-fav", "changes", "Before that, React.")]},
        ),
        later={("f1", "o-fav"), ("f2", "o-fav")},
    )
    assert [(u.text, u.source_fact_ids) for u in actions.updates] == [
        (
            # Facts come oldest first: the later one's earlier state reads first.
            "The user's favorite framework is Svelte. (f2: Fact f2 says so) (f1: Fact f1 says so)",
            ["f1", "f2"],
        )
    ]


def test_a_repeat_after_a_change_keeps_the_change() -> None:
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "existing": [_change("o-port", "changes", "port 9443.")]},
            {
                "fact_id": "f2",
                "kind": "lasting",
                "existing": [_change("o-port", "repeats"), _change("o-port", "repeats")],
            },
        )
    )
    assert [(u.text, u.source_fact_ids) for u in actions.updates] == [("port 9443.", ["f1", "f2"])]


def test_an_earlier_state_in_another_language_gets_no_words_of_ours() -> None:
    pool = {"o-zh": "用户最喜欢的框架是Svelte。"}
    answer = _answer({"fact_id": "f1", "kind": "lasting", "existing": [_change("o-zh", "changes")]})
    actions = answer.to_actions(pool, {"f1": "2026-10-10 09:00: 用户最喜欢Vue"}, lambda f, o: True, lambda f, o: True)
    assert [u.text for u in actions.updates] == ["用户最喜欢的框架是Svelte。 (2026-10-10 09:00: 用户最喜欢Vue)"]


def test_a_change_before_an_earlier_state_note_is_kept() -> None:
    """An undated fact sorts first and is never "stated after": its change comes before a note."""
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "existing": [_change("o-fav", "changes", "Fav is Solid.")]},
            {"fact_id": "f2", "kind": "lasting", "existing": [_change("o-fav", "changes")]},
        ),
        later={("f2", "o-fav")},
    )
    assert [u.text for u in actions.updates] == ["Fav is Solid. (f2: Fact f2 says so)"]


def test_a_change_the_prompt_did_not_mark_as_older_keeps_the_newer_text() -> None:
    """Outside a fact's closest list the model was not told the observation is newer: its rewrite
    is the older fact written as current, so the observation keeps its text and only cites it."""
    actions = _actions(
        _answer({"fact_id": "f1", "kind": "lasting", "existing": [_change("o-fav", "changes", "Fav is React.")]}),
        later={("f1", "o-fav")},
        unmarked={("f1", "o-fav")},
    )
    assert [(u.text, u.source_fact_ids) for u in actions.updates] == [(POOL["o-fav"], ["f1"])]


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


def test_a_restated_fact_that_also_wrote_new_joins_the_earlier_create() -> None:
    actions = _actions(
        _answer(
            {"fact_id": "f1", "kind": "lasting", "adds": "yes", "new": ["ledger-sync listens on port 8443."]},
            {
                "fact_id": "f2",
                "kind": "lasting",
                "adds": "no",
                "new": ["ledger-sync is on 8443."],
                "same_as_fact": "f1",
            },
        )
    )
    assert [c.source_fact_ids for c in actions.creates] == [["f1", "f2"]]


def test_short_ids_map_back_and_unknown_ids_stay() -> None:
    answer = _FactBatchResponse.model_validate(
        {
            "facts": [
                {
                    "fact_id": "F1",
                    "existing": [{"observation_id": "O1", "effect": "repeats"}],
                    "same_as_fact": "F2",
                },
                {"fact_id": "zz", "existing": [{"observation_id": "made-up", "effect": "repeats"}]},
            ],
            "deletes": [{"observation_id": "O1", "merged_into": "O2"}],
        }
    ).with_real_ids({"F1": "f-1", "F2": "f-2", "O1": "o-1", "O2": "o-2"})
    first, second = answer.facts
    assert (first.fact_id, first.same_as_fact, first.existing[0].observation_id) == ("f-1", "f-2", "o-1")
    assert (second.fact_id, second.existing[0].observation_id) == ("zz", "made-up")
    assert (answer.deletes[0].observation_id, answer.deletes[0].merged_into) == ("o-1", "o-2")


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

    def observation(oid: str, text: str, day: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=oid, text=text, occurred_start=None, occurred_end=None, mentioned_at=at(day), source_fact_ids=[]
        )

    # Facts are stated on day 10 (f-old) and 25 (f-new). O1 is in f-old's closest list and later
    # than it; O2 is later than every fact; O3 is later than f-old but neither: unmarked.
    pool = [
        observation("o-fav", POOL["o-fav"], 20),
        observation("o-lang", "The user codes in Rust.", 28),
        observation("o-city", "The user lives in Lisbon.", 15),
    ]
    sent: dict[str, Any] = {}

    class _LLM:
        async def call(self, **kwargs: Any) -> Any:
            sent.update(kwargs)
            answer = {
                "facts": [
                    {
                        "fact_id": "F1",
                        "kind": "lasting",
                        "existing": [
                            {"observation_id": "O1", "effect": "changes", "text": "Before, Vue."},
                            {"observation_id": "O2", "effect": "changes", "text": "Before, Go."},
                            {"observation_id": "O3", "effect": "changes", "text": "The user lives in Rome."},
                        ],
                        "adds": "no",
                    },
                ]
            }
            return LLMCallResult(content=kwargs["response_format"].model_validate(answer))

    config = SimpleNamespace(
        llm_output_language=None,
        observations_mission=None,
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
        _LLM(),
        memories,
        pool,
        {},
        config,
        remaining_observation_slots=0,
        max_observations_per_scope=5,
        closest={"f-old": ["o-fav"], "f-new": ["o-fav"]},
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
    assert '"id": "O2"' in user and '"newer_than_input_facts": true' in user
    # The answer's short ids come back as the real ones. Where the prompt said the observation is
    # newer (closest-list mark, or newer than every fact), the older fact's change is its earlier
    # state, added after the text; where it did not, the observation keeps its text.
    assert [(u.observation_id, u.source_fact_ids, u.text) for u in result.updates] == [
        ("o-fav", ["f-old"], f"{POOL['o-fav']} (2026-10-10 00:00: older)"),
        ("o-lang", ["f-old"], "The user codes in Rust. (2026-10-10 00:00: older)"),
        ("o-city", ["f-old"], "The user lives in Lisbon."),
    ]
    assert sent["messages"][0]["content"] == build_consolidation_system_prompt()
    # The prompt says what those brackets are, for the next call that sees them in the pool.
    assert "is an earlier state that the text before it replaced" in sent["messages"][0]["content"]
    # A full scope is told in the answer's own terms: no `new` sentences.
    assert 'Write no `new` sentences (set `adds` to "no")' in user
