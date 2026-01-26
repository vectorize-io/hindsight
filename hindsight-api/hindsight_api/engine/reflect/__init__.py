"""
Reflect agent module for agentic reflection with tools.

The reflect agent uses an iterative loop with tools to:
1. Lookup mental models (existing knowledge)
2. Recall facts (semantic + temporal search)
3. Learn new insights (create/update observations)
4. Expand memories (get chunk/document context)
"""

from .agent import ReflectAgentResult, run_reflect_agent
from .models import ObservationInput, ReflectAction, ReflectActionBatch

__all__ = [
    "run_reflect_agent",
    "ReflectAgentResult",
    "ReflectAction",
    "ReflectActionBatch",
    "ObservationInput",
]
