"""
Test that first-person agent experiences are classified as 'experience' fact_type,
not 'world'. This is critical for AI agent systems that store their own operational
experiences (debugging, code changes, user interactions) separately from world knowledge.
"""
from datetime import datetime

import pytest

from hindsight_api import LLMConfig
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.retain.fact_extraction import (
    _reclassify_chinese_experience,
    extract_facts_from_text,
)


class TestAgentExperienceClassification:
    """Tests that first-person coding agent experiences get classified as 'experience'."""

    @pytest.mark.asyncio
    async def test_code_changes_classified_as_experience(self):
        """First-person code change descriptions should be experience, not world."""
        text = """
I changed the return type of the `process_request` function from `dict` to `ResponseModel`.
After that, I updated the three callers in `api/handlers.py` to destructure the new model fields.
The type checker was happy after the change but I noticed one test was still using the old dict keys.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"First-person code changes should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_debugging_session_classified_as_experience(self):
        """First-person debugging narrative should be experience, not world."""
        text = """
The tests were failing with a ConnectionRefusedError on the Redis integration suite.
I traced it to the connection pool not being initialized before the first test ran.
I added a setup fixture that ensures the pool is warmed up, and all 47 tests pass now.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"First-person debugging should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_user_interaction_classified_as_experience(self):
        """Agent describing interactions with the user should be experience."""
        text = """
The user asked me to refactor the authentication middleware to support JWT tokens.
I proposed splitting it into two modules: token_validation.py and session_management.py.
The user approved my approach and I started with the token validation logic.
I discovered that the existing tests were mocking the wrong interface, so I had to rewrite them first.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) > len(world_facts), (
            f"Agent-user interactions should be mostly 'experience', "
            f"got {len(experience_facts)} experience vs {len(world_facts)} world. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_mixed_agent_and_world_facts(self):
        """Mix of agent experiences and world knowledge should be classified correctly."""
        text = """
Python 3.12 introduced a new type parameter syntax for generic classes.
I migrated our codebase from the old TypeVar approach to the new syntax.
The migration touched 23 files but was mostly mechanical.
PEP 695 defines the new type statement that makes generics more readable.
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        world_facts = [f for f in facts if f.fact_type == "world"]
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        # Should have both types - world facts about Python 3.12/PEP 695,
        # experience facts about the migration work
        assert len(world_facts) >= 1, (
            f"Should have at least 1 world fact about Python 3.12/PEP 695. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )
        assert len(experience_facts) >= 1, (
            f"Should have at least 1 experience fact about the migration. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_chinese_first_person_classified_as_experience(self):
        """Chinese first-person agent actions should be classified as experience."""
        text = """
我通过 app_id + app_secret 获取 token 写入了飞书表格。
我修复了数据库连接池的内存泄漏问题，并部署了新版本。
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        assert len(experience_facts) >= 1, (
            f"Chinese first-person actions should produce at least 1 experience fact, "
            f"got {len(experience_facts)}. Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )

    @pytest.mark.asyncio
    async def test_chinese_mixed_experience_and_world(self):
        """Mix of Chinese agent experiences and world knowledge should be classified correctly."""
        text = """
我通过 app_id + app_secret 获取 token 写入了飞书表格。
王先生要求每周一执行Reddit监控。
"""
        llm_config = LLMConfig.from_env()
        facts, _, _ = await extract_facts_from_text(
            text=text,
            event_date=datetime(2025, 3, 28),
            llm_config=llm_config,
            agent_name="coding-agent",
            context="agent work log",
            config=_get_raw_config(),
        )

        assert len(facts) > 0, "Should extract at least one fact"
        experience_facts = [f for f in facts if f.fact_type == "experience"]
        world_facts = [f for f in facts if f.fact_type == "world"]
        assert len(experience_facts) >= 1, (
            f"Should have at least 1 experience fact for first-person Chinese action. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )
        assert len(world_facts) >= 1, (
            f"Should have at least 1 world fact for 王先生's requirement. "
            f"Facts: {[(f.fact, f.fact_type) for f in facts]}"
        )


class TestChineseExperienceReclassification:
    """Unit tests for the _reclassify_chinese_experience heuristic (no LLM needed)."""

    def test_chinese_first_person_with_action_verb_reclassified(self):
        """Chinese first-person subject + action verb should be reclassified to experience."""
        assert _reclassify_chinese_experience("我修复了数据库连接问题", "world") == "experience"
        assert _reclassify_chinese_experience("我们部署了新版本", "world") == "experience"
        assert _reclassify_chinese_experience("助手完成了代码审查", "world") == "experience"
        assert _reclassify_chinese_experience("助理分析了日志文件", "world") == "experience"
        assert _reclassify_chinese_experience("我写入了飞书表格", "world") == "experience"
        assert _reclassify_chinese_experience("咱们搭建了测试环境", "world") == "experience"

    def test_chinese_world_facts_not_reclassified(self):
        """Chinese world facts (no first-person subject) should stay as world."""
        assert _reclassify_chinese_experience("王先生要求每周一执行Reddit监控", "world") == "world"
        assert _reclassify_chinese_experience("Python 3.12 引入了新的类型语法", "world") == "world"
        assert _reclassify_chinese_experience("服务器在凌晨三点宕机了", "world") == "world"

    def test_already_experience_not_changed(self):
        """Facts already classified as experience should not be changed."""
        assert _reclassify_chinese_experience("我修复了bug", "experience") == "experience"
        assert _reclassify_chinese_experience("some world fact", "experience") == "experience"

    def test_english_text_not_affected(self):
        """English text should not be reclassified by the Chinese heuristic."""
        assert _reclassify_chinese_experience("I fixed the database bug", "world") == "world"
        assert _reclassify_chinese_experience("The server crashed", "world") == "world"

    def test_chinese_subject_with_gap_before_verb(self):
        """Chinese subject with a small gap before the action verb should match."""
        assert _reclassify_chinese_experience("我通过API修复了问题", "world") == "experience"
        assert _reclassify_chinese_experience("我用新方法解决了bug", "world") == "experience"

    def test_agent_keyword_as_subject(self):
        """The word 'agent' as subject followed by Chinese verb should match."""
        assert _reclassify_chinese_experience("agent执行了任务", "world") == "experience"
        assert _reclassify_chinese_experience("agent成功部署", "world") == "experience"
