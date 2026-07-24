import json
from types import SimpleNamespace

import pytest

from hindsight_api.engine.response_models import TokenUsage
from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.fact_extraction import CausalRelation, Fact
from hindsight_api.engine.retain.types import RetainContent


@pytest.mark.asyncio
async def test_causal_targets_are_offset_from_extraction_group_start(monkeypatch):
    async def extract_facts_from_text(*, text, **_kwargs):
        if text == "preceding":
            facts = [Fact(fact=f"prior {index}", fact_type="world") for index in range(4)]
            return facts, [(text, len(facts))], TokenUsage()

        facts = [
            Fact(fact="cause", fact_type="world"),
            Fact(
                fact="effect",
                fact_type="world",
                causal_relations=[CausalRelation(target_fact_index=0, relation_type="caused_by")],
            ),
        ]
        return facts, [(text, len(facts))], TokenUsage()

    monkeypatch.setattr(fact_extraction, "extract_facts_from_text", extract_facts_from_text)
    monkeypatch.setattr(fact_extraction, "_add_temporal_offsets", lambda *_args: None)
    monkeypatch.setattr(fact_extraction, "_inject_label_tags", lambda *_args: None)

    config = SimpleNamespace(retain_extraction_mode="normal", retain_batch_enabled=False)
    facts, _, _ = await fact_extraction.extract_facts_from_contents(
        [RetainContent(content="preceding"), RetainContent(content="causal group")],
        llm_config=None,
        agent_name="test",
        config=config,
    )

    assert facts[5].causal_relations[0].target_fact_index == 4


@pytest.mark.asyncio
async def test_each_chunk_uses_its_own_causal_index_base(monkeypatch):
    async def extract_facts_from_text(**_kwargs):
        facts = [
            Fact(fact="first cause", fact_type="world"),
            Fact(
                fact="first effect",
                fact_type="world",
                causal_relations=[CausalRelation(target_fact_index=0, relation_type="caused_by")],
            ),
            Fact(fact="second cause", fact_type="world"),
            Fact(
                fact="second effect",
                fact_type="world",
                causal_relations=[CausalRelation(target_fact_index=0, relation_type="caused_by")],
            ),
        ]
        return facts, [("first", 2), ("second", 2)], TokenUsage()

    monkeypatch.setattr(fact_extraction, "extract_facts_from_text", extract_facts_from_text)
    monkeypatch.setattr(fact_extraction, "_add_temporal_offsets", lambda *_args: None)
    monkeypatch.setattr(fact_extraction, "_inject_label_tags", lambda *_args: None)

    config = SimpleNamespace(retain_extraction_mode="normal", retain_batch_enabled=False)
    facts, _, _ = await fact_extraction.extract_facts_from_contents(
        [RetainContent(content="two chunks")],
        llm_config=None,
        agent_name="test",
        config=config,
    )

    assert facts[1].causal_relations[0].target_fact_index == 0
    assert facts[3].causal_relations[0].target_fact_index == 2


def test_invalid_local_causal_targets_are_dropped():
    relations = [
        CausalRelation(target_fact_index=-1, relation_type="caused_by"),
        CausalRelation(target_fact_index=2, relation_type="caused_by"),
        SimpleNamespace(target_fact_index=True, relation_type="caused_by"),
        SimpleNamespace(target_fact_index=0.5, relation_type="caused_by"),
        SimpleNamespace(target_fact_index="0", relation_type="caused_by"),
    ]

    assert fact_extraction._convert_causal_relations(relations, 4, 2) == []


@pytest.mark.asyncio
async def test_batch_causal_targets_use_each_chunk_start(monkeypatch):
    fact_groups = [
        [{"what": f"prior {index}", "fact_type": "world"} for index in range(4)],
        [
            {"what": "first cause", "fact_type": "world"},
            {
                "what": "first effect",
                "fact_type": "world",
                "causal_relations": [{"target_index": 0, "relation_type": "caused_by"}],
            },
        ],
        [
            {"what": "second cause", "fact_type": "world"},
            {
                "what": "second effect",
                "fact_type": "world",
                "causal_relations": [
                    {"target_index": 0, "relation_type": "caused_by"},
                    {"target_index": 2, "relation_type": "caused_by"},
                ],
            },
        ],
    ]

    class Provider:
        async def supports_batch_api(self):
            return True

        async def submit_batch(self, _requests):
            return {"batch_id": "test"}

        async def get_batch_status(self, _batch_id):
            return {"status": "completed", "request_counts": {"completed": 3, "total": 3}}

        async def retrieve_batch_results(self, _batch_id):
            return [
                {
                    "custom_id": f"chunk_{index}",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": json.dumps({"facts": facts})}}],
                            "usage": {},
                        }
                    },
                }
                for index, facts in enumerate(fact_groups)
            ]

    monkeypatch.setattr(fact_extraction, "chunk_text", lambda text, **_kwargs: text.split("|"))
    monkeypatch.setattr(fact_extraction, "_build_extraction_prompt_and_schema", lambda _config: ("", object))
    monkeypatch.setattr(fact_extraction, "_retain_mission_preamble", lambda _config: "")
    monkeypatch.setattr(fact_extraction, "_build_user_message", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(fact_extraction, "_build_request_body", lambda *_args: {})
    monkeypatch.setattr(fact_extraction, "_add_temporal_offsets", lambda *_args: None)
    monkeypatch.setattr(fact_extraction, "_inject_label_tags", lambda *_args: None)

    config = SimpleNamespace(
        retain_extract_causal_links=True,
        retain_chunk_size=100,
        retain_structured_chunk_size=100,
        retain_batch_poll_interval_seconds=0,
    )
    llm_config = SimpleNamespace(_provider_impl=Provider(), provider="test")
    facts, _, _ = await fact_extraction.extract_facts_from_contents_batch_api(
        [RetainContent(content="preceding"), RetainContent(content="first|second")],
        llm_config=llm_config,
        agent_name="test",
        config=config,
    )

    assert facts[5].causal_relations[0].target_fact_index == 4
    assert facts[7].causal_relations[0].target_fact_index == 6
    assert len(facts[7].causal_relations) == 1
