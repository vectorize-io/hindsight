#!/usr/bin/env python3
"""
Multilingual examples for Hindsight.
Run: python examples/api/multilingual.py
"""
import os

HINDSIGHT_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

# =============================================================================
# Doc Examples
# =============================================================================

# [docs:retain-chinese]
from hindsight_client import Hindsight

hindsight = Hindsight(base_url=HINDSIGHT_URL)

# Retain Chinese content
hindsight.retain(
    bank_id="multilingual-user",
    content="""
    张伟是一位资深软件工程师，在腾讯工作了五年。
    他专门研究分布式系统，并领导了公司微服务架构的开发。
    """,
    context="团队概述"
)

# Query in Chinese - get Chinese results
results = hindsight.recall(
    bank_id="multilingual-user",
    query="告诉我关于张伟的信息"
)

# Facts are returned in Chinese:
# - 张伟是一位资深软件工程师，在腾讯工作了五年
# - 张伟专门研究分布式系统，并领导了公司微服务架构的开发
# [/docs:retain-chinese]


# [docs:retain-japanese]
hindsight.retain(
    bank_id="multilingual-user",
    content="""
    田中さんはソフトウェアエンジニアで、東京のスタートアップで働いています。
    彼女はPythonとTypeScriptが得意で、毎日コードレビューをしています。
    """,
    context="チームプロフィール"
)

# Query in Japanese
results = hindsight.recall(
    bank_id="multilingual-user",
    query="田中さんについて教えてください"
)
# [/docs:retain-japanese]


# [docs:reflect-chinese]
# Store facts about team members (in Chinese)
hindsight.retain(
    bank_id="multilingual-team-eval",
    content="张伟是一位优秀的软件工程师，完成了五个重大项目。他总是按时交付，代码整洁有良好的文档。",
    context="绩效评估"
)

hindsight.retain(
    bank_id="multilingual-team-eval",
    content="李明最近加入团队。他错过了第一个截止日期，代码有很多bug。",
    context="绩效评估"
)

# Reflect in Chinese
result = hindsight.reflect(
    bank_id="multilingual-team-eval",
    query="谁是更可靠的工程师？"
)

# Response is in Chinese:
# "我认为张伟更可靠。张伟完成了五个重大项目，按时交付，代码质量高..."
# [/docs:reflect-chinese]


# [docs:mixed-language]
hindsight.retain(
    bank_id="multilingual-user",
    content="""
    王芳在Google北京办公室工作，她是一名高级产品经理。
    之前她在Microsoft和Amazon工作过。
    她负责管理YouTube在中国市场的推广策略。
    """,
    context="员工资料"
)

# Facts preserve both languages:
# - 王芳在Google北京办公室工作，担任高级产品经理
# - 王芳曾在Microsoft和Amazon工作过
# - 王芳负责管理YouTube在中国市场的推广策略
# [/docs:mixed-language]


# =============================================================================
# Cleanup (not shown in docs)
# =============================================================================
hindsight.delete_bank("multilingual-user")
hindsight.delete_bank("multilingual-team-eval")
hindsight.close()

print("multilingual.py: All examples passed")
