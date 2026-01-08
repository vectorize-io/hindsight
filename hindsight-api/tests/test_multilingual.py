"""
Test multilingual support for retain and reflect operations.

Tests that the system correctly handles non-English input and produces
output in the same language as the input.
"""

import pytest
import logging
from datetime import datetime, timezone
from hindsight_api.engine.memory_engine import Budget
from hindsight_api import RequestContext

logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_retain_chinese_content(memory, request_context):
    """
    Test that retain correctly extracts facts from Chinese content
    and keeps the output in Chinese.

    This test verifies:
    1. Facts are extracted from Chinese text
    2. The extracted facts contain Chinese characters
    3. Entity names are preserved in Chinese
    """
    bank_id = f"test_chinese_retain_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Chinese content about a person and their activities
        chinese_content = """
        张伟是一位资深软件工程师，在腾讯工作了五年。他专门研究分布式系统，
        并领导了公司微服务架构的开发。他以编写干净、文档完善的代码而闻名。

        李明上个月加入团队担任初级开发人员。他正在学习React和Node.js。
        李明很有热情，在代码审查中提出很好的问题。他最近完成了他的第一个功能，
        这是一个用户认证流程。

        团队使用Kubernetes进行容器编排，并部署到阿里云。他们遵循敏捷方法论，
        采用两周冲刺周期。合并前必须进行代码审查。
        """

        # Retain the Chinese content
        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=chinese_content,
            context="团队概述",  # Chinese context
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        logger.info(f"Retained {len(unit_ids)} facts from Chinese content")
        assert len(unit_ids) > 0, "Should have extracted and stored facts from Chinese content"

        # Recall the facts with a Chinese query
        result = await memory.recall_async(
            bank_id=bank_id,
            query="告诉我关于张伟的信息",  # "Tell me about Zhang Wei"
            budget=Budget.MID,
            max_tokens=1000,
            fact_type=["world"],
            request_context=request_context,
        )

        logger.info(f"Recalled {len(result.results)} facts")
        assert len(result.results) > 0, "Should recall facts about Zhang Wei"

        # Verify that the facts contain Chinese characters
        # At least one fact should mention 张伟 (Zhang Wei) or related Chinese content
        chinese_facts_found = 0
        for fact in result.results:
            logger.info(f"Fact: {fact.text[:100]}...")
            # Check for common Chinese characters or the name
            if any(
                char in fact.text
                for char in ["张", "伟", "腾讯", "软件", "工程师", "分布式", "系统", "代码"]
            ):
                chinese_facts_found += 1

        logger.info(f"Found {chinese_facts_found} facts with Chinese content")
        assert chinese_facts_found > 0, (
            f"Expected facts to contain Chinese characters, but none found. "
            f"Facts: {[f.text for f in result.results]}"
        )

        logger.info("Chinese retain test passed - facts preserved in Chinese")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_reflect_chinese_content(memory, request_context):
    """
    Test that reflect correctly generates responses in Chinese
    when given Chinese facts and a Chinese query.

    This test verifies:
    1. Reflection produces a response in Chinese
    2. The response references the Chinese facts
    3. Opinions are formed and expressed in Chinese

    Note: LLM responses are non-deterministic, so we retry up to 3 times
    to account for occasional hallucinations of different names.
    """
    bank_id = f"test_chinese_reflect_{datetime.now(timezone.utc).timestamp()}"
    max_retries = 3

    try:
        # Store some Chinese facts to give context for opinion formation
        await memory.retain_async(
            bank_id=bank_id,
            content="张伟是一位优秀的软件工程师，完成了五个重大项目。他总是按时交付，代码整洁有良好的文档。",
            context="绩效评估",  # "Performance review"
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        await memory.retain_async(
            bank_id=bank_id,
            content="李明最近加入团队。他错过了第一个截止日期，代码有很多bug。",
            context="绩效评估",
            event_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            request_context=request_context,
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                # Reflect with a Chinese query
                query = "谁是更可靠的工程师？"  # "Who is a more reliable engineer?"
                result = await memory.reflect_async(
                    bank_id=bank_id,
                    query=query,
                    budget=Budget.LOW,
                    request_context=request_context,
                )

                logger.info(f"Reflection answer (attempt {attempt + 1}): {result.text}")

                # Verify we got an answer
                assert result.text, "Reflection should return an answer"

                # Check that the response contains Chinese characters
                # The response should be in Chinese, not English
                chinese_chars_found = sum(1 for char in result.text if "\u4e00" <= char <= "\u9fff")
                total_chars = len(result.text.replace(" ", "").replace("\n", ""))

                logger.info(f"Chinese characters: {chinese_chars_found}, Total characters: {total_chars}")

                # At least 30% of characters should be Chinese (allowing for numbers, punctuation)
                chinese_ratio = chinese_chars_found / max(total_chars, 1)
                assert chinese_ratio > 0.3, (
                    f"Expected response to be in Chinese (>30% Chinese characters), "
                    f"but only {chinese_ratio:.1%} are Chinese. Response: {result.text}"
                )

                # Check that Chinese names are mentioned
                # The LLM should use names from the based_on facts, not hallucinate different names
                # Extract Chinese names from the based_on world facts
                expected_names = set()
                for fact in result.based_on.get("world", []):
                    # Extract Chinese entity names from the fact
                    for entity in (fact.entities or []):
                        # Check if entity contains Chinese characters
                        if any("\u4e00" <= char <= "\u9fff" for char in entity):
                            expected_names.add(entity)

                # Also check for the specific names we stored
                expected_names.update(["张伟", "李明"])

                # At least one expected name should appear in the response
                found_name = any(name in result.text for name in expected_names)
                assert found_name, (
                    f"Expected response to mention one of the Chinese names: {expected_names}. Response: {result.text}"
                )

                logger.info("Chinese reflect test passed - response generated in Chinese")
                return  # Test passed, exit

            except AssertionError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    raise e

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_japanese_content(memory, request_context):
    """
    Test that retain correctly handles Japanese content.

    This test verifies multilingual support extends beyond Chinese
    to other non-Latin languages.

    Note: LLM fact extraction is non-deterministic and may sometimes translate
    content to English despite instructions. We retry up to 3 times.
    """
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        # Use unique bank_id per attempt to avoid stale data
        bank_id = f"test_japanese_retain_{datetime.now(timezone.utc).timestamp()}_{attempt}"

        try:
            # Japanese content about a developer
            japanese_content = """
            田中さんはソフトウェアエンジニアで、東京のスタートアップで働いています。
            彼女はPythonとTypeScriptが得意で、毎日コードレビューをしています。
            先週、新しいAPIを完成させました。
            """

            unit_ids = await memory.retain_async(
                bank_id=bank_id,
                content=japanese_content,
                context="チームプロフィール",  # "Team profile"
                event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                request_context=request_context,
            )

            logger.info(f"Retained {len(unit_ids)} facts from Japanese content (attempt {attempt + 1})")
            assert len(unit_ids) > 0, "Should have extracted facts from Japanese content"

            # Recall with Japanese query
            result = await memory.recall_async(
                bank_id=bank_id,
                query="田中さんについて教えてください",  # "Tell me about Tanaka-san"
                budget=Budget.MID,
                max_tokens=1000,
                fact_type=["world"],
                request_context=request_context,
            )

            assert len(result.results) > 0, "Should recall facts about Tanaka"

            # Check for Japanese content in facts
            japanese_facts_found = 0
            for fact in result.results:
                logger.info(f"Fact: {fact.text[:100]}...")
                # Check for Japanese characters (hiragana, katakana, or kanji)
                if any(
                    ("\u3040" <= char <= "\u309f")  # Hiragana
                    or ("\u30a0" <= char <= "\u30ff")  # Katakana
                    or ("\u4e00" <= char <= "\u9fff")  # Kanji
                    for char in fact.text
                ):
                    japanese_facts_found += 1

            assert japanese_facts_found > 0, (
                f"Expected facts to contain Japanese characters. "
                f"Facts: {[f.text for f in result.results]}"
            )

            logger.info("Japanese retain test passed - facts preserved in Japanese")
            return  # Test passed

        except AssertionError as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                raise e
        finally:
            # Cleanup the bank
            try:
                await memory.delete_bank(bank_id, request_context=request_context)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_mixed_language_entities(memory, request_context):
    """
    Test that entity extraction works correctly with mixed language content.

    Some entities (like company names) might be in English while the
    description is in Chinese.
    """
    bank_id = f"test_mixed_lang_{datetime.now(timezone.utc).timestamp()}"

    try:
        # Mixed language content - Chinese with English company names
        mixed_content = """
        王芳在Google北京办公室工作，她是一名高级产品经理。
        之前她在Microsoft和Amazon工作过。
        她负责管理YouTube在中国市场的推广策略。
        """

        unit_ids = await memory.retain_async(
            bank_id=bank_id,
            content=mixed_content,
            context="员工资料",
            event_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            request_context=request_context,
        )

        assert len(unit_ids) > 0, "Should extract facts from mixed language content"

        # Recall and check entities
        result = await memory.recall_async(
            bank_id=bank_id,
            query="王芳在哪里工作？",  # "Where does Wang Fang work?"
            budget=Budget.MID,
            max_tokens=1000,
            fact_type=["world"],
            include_entities=True,
            request_context=request_context,
        )

        assert len(result.results) > 0, "Should recall facts about Wang Fang"

        # Check that both Chinese and English entities are preserved
        all_text = " ".join(f.text for f in result.results)
        logger.info(f"Combined facts: {all_text}")

        # Should contain Chinese name and/or English company names
        has_chinese_name = "王芳" in all_text
        has_english_company = any(
            company in all_text for company in ["Google", "Microsoft", "Amazon", "YouTube"]
        )

        assert has_chinese_name or has_english_company, (
            f"Expected mixed language entities. Facts: {all_text}"
        )

        logger.info("Mixed language entity test passed")

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
