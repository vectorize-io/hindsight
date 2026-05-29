from hindsight_api.api.http import MentalModelTrigger
from hindsight_api.engine.search.tags import TagGroupOr


def test_mental_model_trigger_model_dump_preserves_or_tag_group():
    trigger = MentalModelTrigger.model_validate(
        {
            "tag_groups": [
                {
                    "or": [
                        {"tags": ["ns:a"], "match": "all_strict"},
                        {"tags": ["ns:b"], "match": "all_strict"},
                    ]
                }
            ]
        }
    )

    dumped = trigger.model_dump()

    assert dumped["tag_groups"] == [
        {
            "or": [
                {"tags": ["ns:a"], "match": "all_strict"},
                {"tags": ["ns:b"], "match": "all_strict"},
            ]
        }
    ]


def test_mental_model_trigger_or_tag_group_survives_storage_round_trip():
    trigger = MentalModelTrigger.model_validate(
        {
            "tag_groups": [
                {
                    "or": [
                        {"tags": ["ns:a"], "match": "all_strict"},
                        {"tags": ["ns:b"], "match": "all_strict"},
                    ]
                }
            ]
        }
    )

    round_tripped = MentalModelTrigger.model_validate(trigger.model_dump())

    assert isinstance(round_tripped.tag_groups[0], TagGroupOr)
    assert round_tripped.model_dump()["tag_groups"] == trigger.model_dump()["tag_groups"]
