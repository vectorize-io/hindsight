"""The three memory arms behind one interface.

Every arm answers the same two calls: ingest one (split) conversation, and
retrieve a context string for a task. The answer generator and judge never
know which arm produced the context — arm differences stay confined to what
each deployment can actually remember.

* HindsightArm — one private bank per (conversation, agent); each bank
  ingests only that agent's sessions. ``blind`` tasks are unanswerable by
  construction.
* GraphitiArm — one shared graph group per conversation; ALL sessions go in
  via Graphiti's own episode pipeline (its extraction is part of the method
  under test). No private/shared distinction.
* DualArm — both of the above; retrieval returns the private recall AND the
  world-graph facts side by side (the zero-code approximation of plan C's
  answer quality — see HINDSIGHT_GRAPHITI_EVAL_4WAY.md §3.2).
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from benchmarks.locomo.locomo_benchmark import LoComoDataset

from .taskset import SessionSplit, Task

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

_RECALL_MAX_TOKENS = 2048
_GRAPH_SEARCH_RESULTS = 15


class MemoryArm(ABC):
    """One memory deployment under comparison."""

    name: str

    async def setup(self) -> None:  # noqa: B027 — optional hook
        pass

    @abstractmethod
    async def ingest(self, item: dict[str, Any], split: SessionSplit) -> None: ...

    @abstractmethod
    async def retrieve(self, task: Task) -> str:
        """Return the memory context (a human-readable/JSON string) for a task."""
        ...

    async def close(self) -> None:  # noqa: B027 — optional hook
        pass


def _session_number(document_id: str) -> int:
    # prepare_sessions_for_ingestion ids are "<sample_id>_session_<n>"
    return int(document_id.rsplit("_", 1)[1])


class HindsightArm(MemoryArm):
    """Per-agent private banks; each agent holds only its own sessions."""

    name = "hindsight"

    def __init__(self, memory: "MemoryEngine", run_id: str):
        self._memory = memory
        self._run_id = run_id
        self._dataset = LoComoDataset()

    def bank_id(self, conv_id: str, agent: str) -> str:
        return f"dm-{self._run_id}-{conv_id}-{agent}"

    async def ingest(self, item: dict[str, Any], split: SessionSplit) -> None:
        from hindsight_api.models import RequestContext

        sessions = self._dataset.prepare_sessions_for_ingestion(item)
        for agent, owned in (("a", split.a_sessions), ("b", split.b_sessions)):
            contents = [s for s in sessions if _session_number(s["document_id"]) in owned]
            if contents:
                await self._memory.retain_batch_async(
                    bank_id=self.bank_id(split.conv_id, agent),
                    contents=contents,
                    request_context=RequestContext(),
                )

    async def retrieve(self, task: Task) -> str:
        from hindsight_api.engine.memory_engine import Budget
        from hindsight_api.models import RequestContext

        result = await self._memory.recall_async(
            bank_id=self.bank_id(task.conv_id, task.asker),
            query=task.question,
            budget=Budget.MID,
            max_tokens=_RECALL_MAX_TOKENS,
            fact_type=["world", "experience"],
            request_context=RequestContext(),
            _quiet=True,
        )
        memories = [
            {"text": r.text, "occurred_start": r.occurred_start, "occurred_end": r.occurred_end} for r in result.results
        ]
        return json.dumps({"private_memory": memories}, ensure_ascii=False)


class GraphitiArm(MemoryArm):
    """One shared Graphiti graph per conversation (FalkorDB backend).

    Requires ``graphiti-core[falkordb]`` plus an OpenAI-compatible LLM
    endpoint (GRAPHITI_LLM_* env vars, falling back to HINDSIGHT_API_LLM_*).
    Embeddings run locally (the same BGE model Hindsight uses) so the arm
    works without an embeddings API.
    """

    name = "graphiti"

    def __init__(self, run_id: str):
        self._run_id = run_id
        self._graphiti: Any = None

    def group_id(self, conv_id: str) -> str:
        return f"dm-{self._run_id}-{conv_id}"

    async def setup(self) -> None:
        import os

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.falkordb_driver import FalkorDriver
            from graphiti_core.llm_client import LLMConfig as GraphitiLLMConfig
            from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        except ImportError as e:
            raise RuntimeError(
                "GraphitiArm needs graphiti-core: uv pip install 'graphiti-core[falkordb]' "
                "(and a running FalkorDB: docker run -p 6379:6379 falkordb/falkordb)"
            ) from e

        llm_config = GraphitiLLMConfig(
            api_key=os.getenv("GRAPHITI_LLM_API_KEY", os.getenv("HINDSIGHT_API_LLM_API_KEY", "")),
            model=os.getenv("GRAPHITI_LLM_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "")),
            small_model=os.getenv("GRAPHITI_LLM_SMALL_MODEL", os.getenv("HINDSIGHT_API_LLM_MODEL", "")),
            base_url=os.getenv("GRAPHITI_LLM_BASE_URL", os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None),
        )
        driver = FalkorDriver(
            host=os.getenv("FALKORDB_HOST", "localhost"),
            port=int(os.getenv("FALKORDB_PORT", "6379")),
        )
        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=_LocalSTEmbedder(),
            cross_encoder=_local_reranker(),
        )
        await self._graphiti.build_indices_and_constraints()

    async def ingest(self, item: dict[str, Any], split: SessionSplit) -> None:
        from graphiti_core.nodes import EpisodeType

        dataset = LoComoDataset()
        conv = item["conversation"]
        group = self.group_id(split.conv_id)
        for session in dataset.prepare_sessions_for_ingestion(item):
            turns = json.loads(session["content"])
            body = "\n".join(f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in turns)
            await self._graphiti.add_episode(
                name=session["document_id"],
                episode_body=body,
                source=EpisodeType.message,
                source_description=f"conversation between {conv['speaker_a']} and {conv['speaker_b']}",
                reference_time=session["event_date"],
                group_id=group,
            )

    async def retrieve(self, task: Task) -> str:
        edges = await self._graphiti.search(
            query=task.question,
            group_ids=[self.group_id(task.conv_id)],
            num_results=_GRAPH_SEARCH_RESULTS,
        )
        facts = [
            {
                "fact": e.fact,
                "valid_at": e.valid_at.isoformat() if e.valid_at else None,
                "invalid_at": e.invalid_at.isoformat() if e.invalid_at else None,
            }
            for e in edges
        ]
        return json.dumps({"world_graph": facts}, ensure_ascii=False)

    async def close(self) -> None:
        if self._graphiti is not None:
            await self._graphiti.close()


class DualArm(MemoryArm):
    """Private banks AND the shared graph mounted side by side."""

    name = "dual"

    def __init__(self, hindsight: HindsightArm, graphiti: GraphitiArm):
        self._hindsight = hindsight
        self._graphiti = graphiti

    async def setup(self) -> None:
        await self._graphiti.setup()

    async def ingest(self, item: dict[str, Any], split: SessionSplit) -> None:
        # Each sub-arm was (or will be) ingested when it runs standalone; the
        # orchestrator calls ingest once per arm with shared run_id, so dual
        # reuses the standalone arms' stores instead of re-ingesting.
        del item, split

    async def retrieve(self, task: Task) -> str:
        private = json.loads(await self._hindsight.retrieve(task))
        world = json.loads(await self._graphiti.retrieve(task))
        return json.dumps({**private, **world}, ensure_ascii=False)


class _LocalSTEmbedder:
    """Graphiti EmbedderClient backed by the locally cached BGE model."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    async def create(self, input_data: Any) -> list[float]:
        texts = input_data if isinstance(input_data, list) else [input_data]
        return self._model.encode(texts[0], normalize_embeddings=True).tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(input_data_list, normalize_embeddings=True)]


def _local_reranker() -> Any:
    """Local BGE cross-encoder if graphiti ships one; None otherwise.

    The basic ``graphiti.search`` recipe (EDGE_HYBRID_SEARCH_RRF) doesn't
    invoke the reranker, but Graphiti's constructor defaults to an OpenAI
    reranker — pass something harmless so no OpenAI key is required.
    """
    try:
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()
    except Exception:  # pragma: no cover — optional dependency surface
        logger.warning("No local BGE reranker available in graphiti-core; relying on RRF-only recipes")
        return None
