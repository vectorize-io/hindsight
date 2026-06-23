"""Basic Pipecat pipeline with Hindsight persistent memory.

This example shows how to add long-term memory to a voice AI pipeline using
HindsightMemoryService. The processor slots between the user context aggregator
and the LLM service — recalling relevant memories before each turn and retaining
conversation content after.

Requirements:
    pip install pipecat-hindsight pipecat-ai[deepgram,openai,cartesia,silero,daily]

Environment variables:
    DEEPGRAM_API_KEY     - Deepgram speech-to-text key
    OPENAI_API_KEY       - OpenAI API key
    CARTESIA_API_KEY     - Cartesia text-to-speech key
    DAILY_API_KEY        - Daily.co WebRTC transport key
    HINDSIGHT_API_URL    - Hindsight API URL (default: http://localhost:8888)
    HINDSIGHT_API_KEY    - Hindsight API key (required for Hindsight Cloud)
"""

import asyncio
import os

from hindsight_pipecat import HindsightMemoryService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

SYSTEM_PROMPT = """You are a friendly voice assistant with long-term memory.
You remember details from past conversations and use them naturally.
Keep responses concise — this is a voice interface."""


async def main() -> None:
    transport = DailyTransport(
        room_url=os.environ["DAILY_ROOM_URL"],
        token=os.environ.get("DAILY_TOKEN", ""),
        bot_name="Hindsight Bot",
        params=DailyParams(
            audio_out_enabled=True,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Reading Lady
    )
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")

    # HindsightMemoryService — add between user aggregator and LLM.
    # bank_id should be stable per user so memories persist across sessions.
    memory = HindsightMemoryService(
        bank_id="demo-user-001",
        hindsight_api_url=os.environ.get("HINDSIGHT_API_URL", "http://localhost:8888"),
        api_key=os.environ.get("HINDSIGHT_API_KEY"),
        recall_budget="mid",
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            memory,  # ← recall before LLM, retain after each turn
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):  # type: ignore[misc]
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
