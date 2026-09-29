"""Hindsight-Pipecat: Persistent memory for voice AI pipelines.

Provides a Hindsight-backed FrameProcessor for Pipecat pipelines,
giving them long-term memory via automatic recall and retain.

Basic usage::

    from pipecat.pipeline.pipeline import Pipeline
    from hindsight_pipecat import HindsightMemoryService

    memory = HindsightMemoryService(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )

    pipeline = Pipeline([
        transport.input(),
        stt_service,
        user_aggregator,
        memory,            # recall before LLM, retain after each turn
        llm_service,
        assistant_aggregator,
        tts_service,
        transport.output(),
    ])
"""

from .config import (
    HindsightPipecatConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightPipecatError
from .memory import HindsightMemoryService

__version__ = "0.1.0"

__all__ = [
    "configure",
    "get_config",
    "reset_config",
    "HindsightPipecatConfig",
    "HindsightPipecatError",
    "HindsightMemoryService",
]
