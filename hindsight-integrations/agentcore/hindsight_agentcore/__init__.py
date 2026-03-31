"""
hindsight-agentcore
===================

Persistent memory for Amazon Bedrock AgentCore Runtime agents using Hindsight.

AgentCore Runtime sessions are ephemeral — they terminate on inactivity and
reprovision fresh. This package adds durable cross-session memory so agents
remember users, decisions, and patterns across any number of Runtime sessions.

Quick start:
    from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure

    configure(
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key=os.environ["HINDSIGHT_API_KEY"],
    )

    adapter = HindsightRuntimeAdapter(agent_name="support-agent")

    # In your AgentCore Runtime handler:
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["userId"],           # from validated auth context
        agent_name="support-agent",
        tenant_id=event.get("tenantId"),
    )

    result = await adapter.run_turn(
        context=context,
        payload={"prompt": user_message},
        agent_callable=my_agent,
    )
"""

from __future__ import annotations

from .adapter import HindsightRuntimeAdapter, RecallPolicy, RetentionPolicy
from .bank import BankResolver, TurnContext, default_bank_resolver
from .config import (
    HindsightAgentCoreConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import BankResolutionError, HindsightAgentCoreError

__version__ = "0.1.0"

__all__ = [
    # Adapter
    "HindsightRuntimeAdapter",
    "RecallPolicy",
    "RetentionPolicy",
    # Identity + bank
    "TurnContext",
    "BankResolver",
    "default_bank_resolver",
    # Configuration
    "configure",
    "get_config",
    "reset_config",
    "HindsightAgentCoreConfig",
    # Errors
    "HindsightAgentCoreError",
    "BankResolutionError",
]
