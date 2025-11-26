"""
Common benchmark runner framework based on the LoComo implementation.

This module provides a unified interface for running benchmarks with the same
optimizations as the working LoComo benchmark:
- Batch ingestion for speed
- Parallel question processing with semaphores
- Parallel LLM judging with rate limiting
- Progress tracking with Rich
- Comprehensive metrics collection
- Support for both traditional (search + LLM) and integrated (think API) approaches

The framework supports two answer generation patterns:
1. Traditional: Benchmark runner performs search, then passes results to answer generator
2. Integrated: Answer generator performs its own retrieval (e.g., think API)
   - Indicated by needs_external_search() returning False
   - Skips the search step for efficiency
"""

import json
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box
import pydantic

from hindsight_api import MemoryEngine
from openai import AsyncOpenAI

console = Console()


class HindsightClientAdapter:
    """
    Adapter that wraps the Hindsight Python client to provide the interface
    expected by the benchmark runner.

    This allows benchmarks to use the Python client instead of RemoteMemoryClient
    while maintaining the same interface for put_batch_async, search_async, etc.
    """

    def __init__(self, base_url: str = "http://localhost:8888", timeout: float = 300.0):
        """
        Initialize the adapter with the Hindsight client.

        Args:
            base_url: Base URL of the Hindsight API server
            timeout: Request timeout in seconds
        """
        from hindsight_client import Hindsight
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = Hindsight(base_url=base_url, timeout=timeout)

    async def initialize(self):
        """Initialize the client (no-op for HTTP client)."""
        pass

    async def close(self):
        """Close the HTTP client."""
        self.client.close()

    async def put_batch_async(
        self,
        agent_id: str,
        contents: List[Dict[str, Any]],
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store multiple memory items via API.

        Args:
            agent_id: Agent identifier
            contents: List of content dicts with 'content', 'event_date', 'context' keys
            document_id: Optional document identifier

        Returns:
            Result dict with success status
        """
        # Convert to format expected by client
        items = []
        for content in contents:
            item = {"content": content["content"]}
            if "event_date" in content and content["event_date"]:
                item["event_date"] = content["event_date"]
            if "context" in content and content["context"]:
                item["context"] = content["context"]
            items.append(item)

        return await self.client.aput_batch(
            agent_id=agent_id,
            items=items,
            document_id=document_id,
        )

    async def search_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 100,
        max_tokens: int = 4096,
        enable_trace: bool = False,
        reranker: str = "heuristic",
        fact_type: Optional[List[str]] = None,
        question_date: Optional[datetime] = None
    ) -> 'SearchResult':
        """
        Search memories via API.

        Returns:
            SearchResult object with results list
        """
        from hindsight_client_api.models import search_request

        request_obj = search_request.SearchRequest(
            query=query,
            fact_type=fact_type,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            trace=enable_trace,
            question_date=question_date.isoformat() if question_date else None,
        )

        response = await self.client._memory_api.search_memories(agent_id, request_obj)

        # Convert to expected format - wrap results in an object with .results attribute
        class SearchResult:
            def __init__(self, results):
                self.results = results

        class MemoryFact:
            def __init__(self, data):
                self._data = data

            def model_dump(self):
                return self._data

        results = []
        if hasattr(response, 'results'):
            for r in response.results:
                data = r.to_dict() if hasattr(r, 'to_dict') else r
                results.append(MemoryFact(data))

        return SearchResult(results)

    async def think_async(
        self,
        agent_id: str,
        query: str,
        thinking_budget: int = 50,
        context: str = None
    ) -> 'ThinkResult':
        """
        Generate answer using think API.

        Returns:
            ThinkResult object with text, based_on, and new_opinions
        """
        response = await self.client.athink(
            agent_id=agent_id,
            query=query,
            thinking_budget=thinking_budget,
            context=context,
        )

        # Convert to expected format with attribute access
        class MemoryFact:
            def __init__(self, data):
                self._data = data

            def model_dump(self):
                return self._data

            def get(self, key, default=None):
                return self._data.get(key, default)

        class ThinkResult:
            def __init__(self, data):
                self.text = data.get('text', '')
                # Convert based_on facts to MemoryFact objects
                based_on_raw = data.get('based_on', {})
                self.based_on = {
                    'world': [MemoryFact(f) for f in based_on_raw.get('world', [])],
                    'agent': [MemoryFact(f) for f in based_on_raw.get('agent', [])],
                    'opinion': [MemoryFact(f) for f in based_on_raw.get('opinion', [])],
                }
                self.new_opinions = data.get('new_opinions', [])

        return ThinkResult(response)

    async def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Delete all data for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Result dict
        """
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.delete(f"{self.base_url}/api/v1/agents/{agent_id}")
            if response.status_code == 404:
                return {"success": True, "message": "Agent not found (already deleted)"}
            response.raise_for_status()
            return response.json()

    async def list_agents(self) -> List[str]:
        """
        List all agents.

        Returns:
            List of agent IDs
        """
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/agents")
            response.raise_for_status()
            result = response.json()
            return [a.get('agent_id', a) if isinstance(a, dict) else a for a in result.get("agents", [])]

    async def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Get statistics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with statistics including total_nodes, total_links, and pending_operations
        """
        import httpx
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/v1/agents/{agent_id}/stats")
            response.raise_for_status()
            return response.json()

    async def wait_for_backlog_completion(
        self,
        agent_id: str,
        poll_interval: float = 1.0,
        timeout: float = 300.0,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Poll agent stats until pending_operations is zero or timeout is reached.

        Args:
            agent_id: Agent identifier
            poll_interval: Time to wait between polls in seconds
            timeout: Maximum time to wait in seconds
            verbose: Whether to print status updates

        Returns:
            Final stats dict

        Raises:
            TimeoutError: If pending_operations doesn't clear within timeout
        """
        import time
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Timeout waiting for pending operations to clear for agent '{agent_id}' "
                    f"after {timeout}s"
                )

            stats = await self.get_agent_stats(agent_id)
            pending_operations = stats.get("pending_operations", 0)

            if verbose:
                print(
                    f"Agent '{agent_id}' pending operations: {pending_operations} "
                    f"(elapsed: {elapsed:.1f}s)"
                )

            if pending_operations == 0:
                if verbose:
                    print(f"All operations completed for agent '{agent_id}' in {elapsed:.1f}s")
                return stats

            await asyncio.sleep(poll_interval)


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""

    @abstractmethod
    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load dataset from file.

        Returns:
            List of dataset items
        """
        pass

    @abstractmethod
    def get_item_id(self, item: Dict) -> str:
        """Get unique identifier for an item."""
        pass

    @abstractmethod
    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare conversation sessions for batch ingestion.

        Returns:
            List of session dicts with keys: 'content', 'context', 'event_date'
        """
        pass

    @abstractmethod
    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from an item.

        Returns:
            List of QA dicts with keys: 'question', 'answer', 'category' (optional)
        """
        pass


class LLMAnswerGenerator(ABC):
    """Abstract base class for LLM-based answer generation."""

    def needs_external_search(self) -> bool:
        """
        Whether this generator needs external search to be performed.

        Returns:
            True if the benchmark runner should perform search before calling generate_answer.
            False if the generator does its own retrieval (e.g., integrated think API).
        """
        return True

    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        memories: List[Dict[str, Any]],
        question_date: Optional[datetime] = None
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories.

        Args:
            question: The question text
            memories: Retrieved memories to use for answering
            question_date: Optional date when the question was asked (for temporal context)

        Returns:
            Tuple of (answer, reasoning, retrieved_memories_override)
            - answer: The generated answer text
            - reasoning: Explanation of how the answer was derived
            - retrieved_memories_override: Optional list of memories to include in results
              - None: Use memories passed in (traditional mode)
              - List: Use these memories instead (integrated mode like think API)
        """
        pass


class JudgeResponse(pydantic.BaseModel):
    """Judge response format."""
    correct: bool
    reasoning: str


class LLMAnswerEvaluator:
    """LLM-based answer evaluator with configurable provider."""

    def __init__(self):
        """Initialize with LLM configuration for judge/evaluator."""
        from hindsight_api.engine.llm_wrapper import LLMConfig
        self.llm_config = LLMConfig.for_judge()
        self.client = self.llm_config._client
        self.model = self.llm_config.model

    async def judge_answer(
        self,
        question: str,
        correct_answer: str,
        predicted_answer: str,
        semaphore: asyncio.Semaphore
    ) -> Tuple[bool, str]:
        """
        Evaluate predicted answer using LLM-as-judge.

        Args:
            question: The question
            correct_answer: Gold/correct answer
            predicted_answer: Predicted answer
            semaphore: Semaphore for rate limiting

        Returns:
            Tuple of (is_correct, reasoning)
        """
        async with semaphore:
            try:
                judgement = await self.llm_config.call(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert grader that determines if answers to questions match a gold standard answer"
                        },
                        {
                            "role": "user",
                            "content": f"""
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know, it should be counted as CORRECT. 

    Now it's time for the real question:
    Question: {question}
    Gold answer: {correct_answer}
    Generated answer: {predicted_answer}

    First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
    If it's correct, set correct=true.
"""
                        }
                    ],
                    response_format=JudgeResponse,
                    scope="judge",
                    temperature=0,
                    max_tokens=4096
                )

                return judgement.correct, judgement.reasoning

            except Exception as e:
                print(f"Error judging answer: {e}")
                return False, f"Error: {str(e)}"


class BenchmarkRunner:
    """
    Common benchmark runner using the proven LoComo approach.

    Optimizations:
    - Batch ingestion (put_batch_async)
    - Parallel question processing with rate limiting
    - Parallel LLM judging with rate limiting
    - Progress tracking
    """

    def __init__(
        self,
        dataset: BenchmarkDataset,
        answer_generator: LLMAnswerGenerator,
        answer_evaluator: LLMAnswerEvaluator,
        memory: Optional[MemoryEngine] = None
    ):
        """
        Initialize benchmark runner.

        Args:
            dataset: Dataset implementation
            answer_generator: Answer generator implementation
            answer_evaluator: Answer evaluator implementation
            memory: Memory system instance (creates new if None)
        """
        import os
        self.dataset = dataset
        self.answer_generator = answer_generator
        self.answer_evaluator = answer_evaluator
        self.memory = memory or MemoryEngine(
            db_url=os.getenv("HINDSIGHT_API_DATABASE_URL"),
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,  # Use None to get provider defaults
        )

    def calculate_data_stats(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the data to be ingested.

        Returns:
            Dict with statistics: total_sessions, total_chars, avg_session_length, etc.
        """
        total_sessions = 0
        total_chars = 0
        session_lengths = []

        for item in items:
            batch_contents = self.dataset.prepare_sessions_for_ingestion(item)
            total_sessions += len(batch_contents)

            for session in batch_contents:
                content_len = len(session['content'])
                total_chars += content_len
                session_lengths.append(content_len)

        avg_length = total_chars / total_sessions if total_sessions > 0 else 0

        return {
            'total_sessions': total_sessions,
            'total_chars': total_chars,
            'total_items': len(items),
            'avg_session_length': avg_length,
            'min_session_length': min(session_lengths) if session_lengths else 0,
            'max_session_length': max(session_lengths) if session_lengths else 0
        }

    async def ingest_conversation(
        self,
        item: Dict[str, Any],
        agent_id: str
    ) -> int:
        """
        Ingest conversation into memory using batch ingestion.

        Uses put_batch_async for maximum efficiency.

        Returns:
            Number of sessions ingested
        """
        batch_contents = self.dataset.prepare_sessions_for_ingestion(item)

        if batch_contents:
            await self.memory.put_batch_async(
                agent_id=agent_id,
                contents=batch_contents
            )

            # If using remote API, wait for this batch to complete before continuing
            if isinstance(self.memory, HindsightClientAdapter):
                await self.memory.wait_for_backlog_completion(agent_id, verbose=False)

        return len(batch_contents)

    async def answer_question(
        self,
        agent_id: str,
        question: str,
        thinking_budget: int = 500,
        max_tokens: int = 4096,
        question_date: Optional[datetime] = None,
    ) -> Tuple[str, str, List[Dict]]:
        """
        Answer a question using memory retrieval.

        Args:
            agent_id: Agent ID
            question: Question text
            thinking_budget: Thinking budget for search
            max_tokens: Maximum tokens to retrieve
            question_date: Date when the question was asked (for temporal filtering)

        Returns:
            Tuple of (answer, reasoning, retrieved_memories)
        """
        # Check if generator needs external search
        if self.answer_generator.needs_external_search():
            # Traditional flow: search then generate
            # Search both 'world' and 'agent' fact types in parallel
            search_result = await self.memory.search_async(
                agent_id=agent_id,
                query=question,
                thinking_budget=thinking_budget,
                max_tokens=max_tokens,
                fact_type=["world", "agent"],
                question_date=question_date,
                include_entities=True
            )

            # Convert MemoryFact objects to dictionaries for compatibility
            results = [fact.model_dump() for fact in search_result.results]

            if not results:
                return "I don't have enough information to answer that question.", "No relevant memories found.", []

            # Generate answer using LLM
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(question, results, question_date)

            # Use override if provided, otherwise use search results
            final_memories = memories_override if memories_override is not None else results

            return answer, reasoning, final_memories
        else:
            # Integrated flow: generator does its own search (e.g., think API)
            # Pass empty memories list since generator doesn't need them
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(question, [], question_date)

            # Use memories from generator (should not be None for integrated mode)
            final_memories = memories_override if memories_override is not None else []

            return answer, reasoning, final_memories

    async def evaluate_qa_task(
        self,
        agent_id: str,
        qa_pairs: List[Dict],
        item_id: str,
        thinking_budget: int,
        max_tokens: int,
        max_questions: Optional[int] = None,
        semaphore: asyncio.Semaphore = None,
    ) -> List[Dict]:
        """
        Evaluate QA task with parallel question processing.

        Args:
            semaphore: Semaphore to limit concurrent question processing

        Returns:
            List of QA results
        """
        # Filter out questions without answers (category 5)
        qa_pairs = [pair for pair in qa_pairs if pair.get('answer')]
        questions_to_eval = qa_pairs[:max_questions] if max_questions else qa_pairs

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating QA for {item_id} - {len(questions_to_eval)} questions",
                total=len(questions_to_eval)
            )

            # Create tasks for all questions
            async def process_question(qa):
                async with semaphore:
                    question = qa['question']
                    correct_answer = qa['answer']
                    category = qa.get('category', 0)
                    question_date = qa.get('question_date')

                    try:
                        # Get predicted answer, reasoning, and retrieved memories
                        predicted_answer, reasoning, retrieved_memories = await self.answer_question(
                            agent_id, question, thinking_budget, max_tokens, question_date
                        )

                        # Remove embeddings from retrieved memories to reduce file size
                        memories_without_embeddings = [
                            {k: v for k, v in mem.items() if k != 'embedding'}
                            for mem in retrieved_memories
                        ]

                        return {
                            'question': question,
                            'correct_answer': correct_answer,
                            'predicted_answer': predicted_answer,
                            'reasoning': reasoning,
                            'category': category,
                            'retrieved_memories': memories_without_embeddings,
                            'is_invalid': False,
                            'error': None
                        }
                    except Exception as e:
                        logging.exception(f"Failed to answer question: {question[:100]}")
                        # Mark as invalid if answer generation failed
                        console.print(f"      [red]✗[/red] Failed to answer question: {question[:50]}... Error: {str(e)[:100]}")
                        return {
                            'question': question,
                            'correct_answer': correct_answer,
                            'predicted_answer': 'ERROR: Failed to generate answer',
                            'reasoning': f'Error: {str(e)}',
                            'category': category,
                            'retrieved_memories': [],
                            'is_invalid': True,
                            'error': str(e)
                        }

            question_tasks = [process_question(qa) for qa in questions_to_eval]

            # Use as_completed to update progress as results come in
            results = []
            for coro in asyncio.as_completed(question_tasks):
                result = await coro
                results.append(result)
                progress.update(task, advance=1)

        return results

    async def calculate_metrics(self, results: List[Dict], eval_semaphore_size: int = 8) -> Dict:
        """
        Calculate evaluation metrics using parallel LLM-as-judge.

        Args:
            results: QA results to evaluate
            eval_semaphore_size: Max concurrent LLM judge requests

        Returns:
            Dict with evaluation metrics
        """
        total = len(results)

        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(eval_semaphore_size)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[yellow]Judging answers with LLM (parallel, max {eval_semaphore_size})...",
                total=total
            )

            # Create all judgment tasks
            async def judge_single(result):
                # Skip judging if already marked as invalid
                if result.get('is_invalid', False):
                    result['is_correct'] = None
                    result['correctness_reasoning'] = f"Question invalid due to error: {result.get('error', 'Unknown error')}"
                    return result

                try:
                    is_correct, eval_reasoning = await self.answer_evaluator.judge_answer(
                        result['question'],
                        result['correct_answer'],
                        result['predicted_answer'],
                        semaphore
                    )
                    result['is_correct'] = is_correct
                    result['correctness_reasoning'] = eval_reasoning
                    return result
                except Exception as e:
                    # Mark as invalid if judging failed
                    logging.exception(f"Failed to judge answer for question: {result.get('question', 'unknown')[:100]}")
                    console.print(f"      [red]✗[/red] Failed to judge answer: {result.get('question', '')[:50]}... Error: {str(e)[:100]}")
                    result['is_invalid'] = True
                    result['is_correct'] = None
                    result['correctness_reasoning'] = f"Judge error: {str(e)}"
                    result['error'] = str(e)
                    return result

            judgment_tasks = [judge_single(result) for result in results]

            # Process in parallel with progress updates
            judged_results = []
            for coro in asyncio.as_completed(judgment_tasks):
                judged_result = await coro
                judged_results.append(judged_result)
                progress.update(task, advance=1)

        # Calculate stats
        correct = sum(1 for r in judged_results if r.get('is_correct', False))
        invalid = sum(1 for r in judged_results if r.get('is_invalid', False))
        valid_total = total - invalid
        category_stats = {}

        for result in judged_results:
            category = result.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0, 'invalid': 0}
            category_stats[category]['total'] += 1
            if result.get('is_invalid', False):
                category_stats[category]['invalid'] += 1
            elif result.get('is_correct', False):
                category_stats[category]['correct'] += 1

        # Calculate accuracy excluding invalid questions
        accuracy = (correct / valid_total * 100) if valid_total > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'invalid': invalid,
            'valid_total': valid_total,
            'category_stats': category_stats,
            'detailed_results': judged_results
        }

    async def _agent_has_data(self, agent_id: str) -> bool:
        """
        Check if an agent has any indexed memory units.

        Args:
            agent_id: Agent ID to check

        Returns:
            True if agent has at least one memory unit, False otherwise
        """
        try:
            # Check if we're using a remote client or local memory
            if isinstance(self.memory, HindsightClientAdapter):
                # Use stats API for remote client
                stats = await self.memory.get_agent_stats(agent_id)
                total_nodes = stats.get("total_nodes", 0)
                return total_nodes > 0
            else:
                # Use direct database access for local memory
                pool = await self.memory._get_pool()
                async with pool.acquire() as conn:
                    result = await conn.fetchrow(
                        "SELECT COUNT(*) as count FROM memory_units WHERE agent_id = $1 LIMIT 1",
                        agent_id
                    )
                    return result['count'] > 0
        except Exception as e:
            console.print(f"  [red]Warning: Error checking agent data: {e}[/red]")
            return False

    async def process_single_item(
        self,
        item: Dict,
        agent_id: str,
        i: int,
        total_items: int,
        thinking_budget: int,
        max_tokens: int,
        max_questions_per_item: Optional[int],
        skip_ingestion: bool,
        question_semaphore: asyncio.Semaphore,
        eval_semaphore_size: int = 8,
        clear_this_agent: bool = True,
    ) -> Dict:
        """
        Process a single item (ingest + evaluate).

        Args:
            clear_this_agent: Whether to clear this agent's data before ingesting.
                             Set to False to skip clearing (e.g., when agent_id is shared and already cleared)

        Returns:
            Result dict with metrics
        """
        item_id = self.dataset.get_item_id(item)

        console.print(f"\n[bold blue]Item {i}/{total_items}[/bold blue] (ID: {item_id})")

        if not skip_ingestion:
            # Clear agent data before ingesting
            if clear_this_agent:
                console.print("  [1] Clearing previous agent data...")
                await self.memory.delete_agent(agent_id)
                console.print(f"      [green]✓[/green] Cleared '{agent_id}' agent data")

            # Ingest conversation
            console.print("  [2] Ingesting conversation (batch mode)...")
            num_sessions = await self.ingest_conversation(item, agent_id)
            console.print(f"      [green]✓[/green] Ingested {num_sessions} sessions")
        else:
            num_sessions = -1

        # Evaluate QA
        qa_pairs = self.dataset.get_qa_pairs(item)
        console.print(f"  [3] Evaluating {len(qa_pairs)} QA pairs (parallel)...")
        qa_results = await self.evaluate_qa_task(
            agent_id,
            qa_pairs,
            item_id,
            thinking_budget,
            max_tokens,
            max_questions_per_item,
            question_semaphore,
        )

        # Calculate metrics
        console.print("  [4] Calculating metrics...")
        metrics = await self.calculate_metrics(qa_results, eval_semaphore_size)

        console.print(f"      [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")

        return {
            'item_id': item_id,
            'metrics': metrics,
            'num_sessions': num_sessions
        }

    async def run(
        self,
        dataset_path: Path,
        agent_id: str,
        max_items: Optional[int] = None,
        max_questions_per_item: Optional[int] = None,
        thinking_budget: int = 500,
        max_tokens: int = 4096,
        skip_ingestion: bool = False,
        max_concurrent_questions: int = 1,  # Default to 1 for sequential processing
        eval_semaphore_size: int = 8,
        clear_agent_per_item: bool = False,
        specific_item: Optional[str] = None,
        separate_ingestion_phase: bool = False,
        filln: bool = False,
        max_concurrent_items: int = 1,  # Max concurrent items (conversations) to process in parallel
    ) -> Dict[str, Any]:
        """
        Run the full benchmark evaluation.

        Args:
            dataset_path: Path to dataset file
            agent_id: Agent ID to use
            max_items: Maximum number of items to evaluate
            max_questions_per_item: Maximum questions per item
            thinking_budget: Thinking budget for search
            max_tokens: Maximum tokens to retrieve from memories
            skip_ingestion: Skip ingestion and use existing data
            max_concurrent_questions: Max concurrent question processing
            eval_semaphore_size: Max concurrent LLM judge requests
            clear_agent_per_item: Use unique agent ID per item for isolation (deprecated when separate_ingestion_phase=True)
            specific_item: If provided, only run this specific item ID (e.g., conversation)
            separate_ingestion_phase: If True, ingest all data first, then evaluate all questions (single agent)
            filln: If True, only process items where the agent has no indexed data yet
            max_concurrent_items: Max concurrent items to process in parallel (requires clear_agent_per_item=True)

        Returns:
            Dict with complete benchmark results
        """
        console.print(f"\n[bold cyan]Benchmark Evaluation[/bold cyan]")
        console.print("=" * 80)

        # Load dataset
        console.print(f"\n[1] Loading dataset from {dataset_path}...")
        items = self.dataset.load(dataset_path, max_items)

        # Filter for specific item if requested
        if specific_item is not None:
            items = [item for item in items if self.dataset.get_item_id(item) == specific_item]
            if not items:
                console.print(f"    [red]✗[/red] No item found with ID: {specific_item}")
                raise ValueError(f"Item with ID '{specific_item}' not found in dataset")
            console.print(f"    [green]✓[/green] Filtering to specific item: {specific_item}")

        console.print(f"    [green]✓[/green] Loaded {len(items)} items")

        # Initialize memory system
        console.print(f"\n[2] Initializing memory system...")
        console.print(f"    [green]✓[/green] Memory system initialized")

        if separate_ingestion_phase:
            # New two-phase approach: ingest all, then evaluate all
            return await self._run_two_phase(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item,
                max_concurrent_questions, eval_semaphore_size
            )
        else:
            # Original approach: process each item independently
            return await self._run_single_phase(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item,
                max_concurrent_questions, eval_semaphore_size,
                clear_agent_per_item, filln, max_concurrent_items
            )

    async def _run_single_phase(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        max_concurrent_questions: int,
        eval_semaphore_size: int,
        clear_agent_per_item: bool,
        filln: bool = False,
        max_concurrent_items: int = 1,
    ) -> Dict[str, Any]:
        """Original single-phase approach: process each item independently."""
        # Create semaphore for question processing
        question_semaphore = asyncio.Semaphore(max_concurrent_questions)

        # Process items - either in parallel or sequentially
        if max_concurrent_items > 1 and clear_agent_per_item:
            # Parallel item processing (requires unique agent IDs)
            all_results = await self._process_items_parallel(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item, question_semaphore,
                eval_semaphore_size, filln, max_concurrent_items
            )
        else:
            # Sequential item processing (original behavior)
            all_results = await self._process_items_sequential(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item, question_semaphore,
                eval_semaphore_size, clear_agent_per_item, filln
            )

        # Calculate overall metrics
        total_correct = sum(r['metrics']['correct'] for r in all_results)
        total_questions = sum(r['metrics']['total'] for r in all_results)
        total_invalid = sum(r['metrics'].get('invalid', 0) for r in all_results)
        total_valid = total_questions - total_invalid
        # Calculate accuracy excluding invalid questions
        overall_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'total_invalid': total_invalid,
            'total_valid': total_valid,
            'num_items': len(items),
            'item_results': all_results
        }

    async def _process_items_sequential(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        question_semaphore: asyncio.Semaphore,
        eval_semaphore_size: int,
        clear_agent_per_item: bool,
        filln: bool,
    ) -> List[Dict]:
        """Process items sequentially (original behavior)."""
        all_results = []

        for i, item in enumerate(items, 1):
            # Use unique agent ID per item if requested (for isolation in benchmarks like LongMemEval)
            # This avoids deadlocks from deleting agent data
            if clear_agent_per_item:
                item_id = self.dataset.get_item_id(item)
                item_agent_id = f"{agent_id}_{item_id}"
                # Always clear for unique agents (each agent_id is used only once)
                clear_this_agent = True
            else:
                item_agent_id = agent_id
                # Only clear on first item for shared agent_id
                clear_this_agent = (i == 1)

            # Check if we should skip this item (filln mode)
            if filln:
                has_data = await self._agent_has_data(item_agent_id)
                if has_data:
                    console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {self.dataset.get_item_id(item)})")
                    console.print(f"  [yellow]⊘[/yellow] Skipping - agent '{item_agent_id}' already has indexed data")
                    continue

            result = await self.process_single_item(
                item, item_agent_id, i, len(items),
                thinking_budget, max_tokens, max_questions_per_item,
                skip_ingestion, question_semaphore, eval_semaphore_size,
                clear_this_agent,
            )
            all_results.append(result)

        return all_results

    async def _process_items_parallel(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        question_semaphore: asyncio.Semaphore,
        eval_semaphore_size: int,
        filln: bool,
        max_concurrent_items: int,
    ) -> List[Dict]:
        """Process items in parallel (requires unique agent IDs per item)."""
        # Create semaphore for item-level parallelism
        item_semaphore = asyncio.Semaphore(max_concurrent_items)

        async def process_item_wrapper(i: int, item: Dict) -> Optional[Dict]:
            """Wrapper to process a single item with semaphore control."""
            async with item_semaphore:
                item_id = self.dataset.get_item_id(item)
                item_agent_id = f"{agent_id}_{item_id}"

                # Check if we should skip this item (filln mode)
                if filln:
                    has_data = await self._agent_has_data(item_agent_id)
                    if has_data:
                        console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                        console.print(f"  [yellow]⊘[/yellow] Skipping - agent '{item_agent_id}' already has indexed data")
                        return None

                # Process the item
                result = await self.process_single_item(
                    item, item_agent_id, i, len(items),
                    thinking_budget, max_tokens, max_questions_per_item,
                    skip_ingestion, question_semaphore, eval_semaphore_size,
                    clear_this_agent=True,  # Always clear for parallel processing
                )
                return result

        # Create all tasks
        tasks = [process_item_wrapper(i, item) for i, item in enumerate(items, 1)]

        # Run in parallel and collect results
        results = await asyncio.gather(*tasks)

        # Filter out None results (skipped items)
        all_results = [r for r in results if r is not None]

        return all_results

    async def _run_two_phase(
        self,
        items: List[Dict[str, Any]],
        agent_id: str,
        thinking_budget: int,
        max_tokens: int,
        skip_ingestion: bool,
        max_questions_per_item: Optional[int],
        max_concurrent_questions: int,
        eval_semaphore_size: int,
    ) -> Dict[str, Any]:
        """
        Two-phase approach: ingest all data into single agent, then evaluate all questions.

        More realistic scenario where agent accumulates memories over time.
        """
        # Check if using remote API client
        is_remote = isinstance(self.memory, HindsightClientAdapter)

        # Phase 1: Ingestion
        if not skip_ingestion:
            # Calculate and display data statistics
            console.print(f"\n[3] Analyzing data to be ingested...")
            stats = self.calculate_data_stats(items)
            console.print(f"    [cyan]Total items:[/cyan] {stats['total_items']}")
            console.print(f"    [cyan]Total sessions:[/cyan] {stats['total_sessions']}")
            console.print(f"    [cyan]Total characters:[/cyan] {stats['total_chars']:,}")
            console.print(f"    [cyan]Avg session length:[/cyan] {stats['avg_session_length']:.0f} chars")
            console.print(f"    [cyan]Session length range:[/cyan] {stats['min_session_length']}-{stats['max_session_length']} chars")

            console.print(f"\n[4] Phase 1: Ingesting all data into agent '{agent_id}'...")
            console.print(f"    [yellow]Clearing previous agent data...[/yellow]")
            await self.memory.delete_agent(agent_id)
            console.print(f"    [green]✓[/green] Cleared agent data")

            if is_remote:
                # For remote API: send one request per instance, then poll
                console.print(f"    [yellow]Sending {len(items)} instances (one request per instance)...[/yellow]")
                total_sessions = 0

                for i, item in enumerate(items, 1):
                    item_sessions = self.dataset.prepare_sessions_for_ingestion(item)
                    total_sessions += len(item_sessions)

                    if item_sessions:
                        await self.memory.put_batch_async(
                            agent_id=agent_id,
                            contents=item_sessions
                        )

                    if i % 10 == 0 or i == len(items):
                        console.print(f"        Sent {i}/{len(items)} instances ({total_sessions} sessions so far)")

                console.print(f"    [green]✓[/green] Sent all {len(items)} instances ({total_sessions} sessions total)")

                # Wait for all background processing to complete
                console.print(f"    [yellow]Waiting for background processing to complete...[/yellow]")
                await self.memory.wait_for_backlog_completion(agent_id, verbose=False)
                console.print(f"    [green]✓[/green] Background processing complete")
            else:
                # For local memory: collect all and send in one batch (faster with auto-chunking)
                console.print(f"    [yellow]Collecting sessions from all items...[/yellow]")
                all_sessions = []
                for item in items:
                    item_sessions = self.dataset.prepare_sessions_for_ingestion(item)
                    all_sessions.extend(item_sessions)

                console.print(f"    [cyan]Collected {len(all_sessions)} sessions from {len(items)} items[/cyan]")
                console.print(f"    [yellow]Ingesting in one batch (auto-chunks if needed)...[/yellow]")

                # Ingest all sessions in one batch call (will auto-chunk if too large)
                await self.memory.put_batch_async(
                    agent_id=agent_id,
                    contents=all_sessions
                )

                console.print(f"    [green]✓[/green] Ingested {len(all_sessions)} sessions from {len(items)} items")
        else:
            console.print(f"\n[3] Skipping ingestion (using existing data)")

        # Phase 2: Evaluation
        console.print(f"\n[5] Phase 2: Evaluating all questions...")

        # Create semaphore for question processing
        question_semaphore = asyncio.Semaphore(max_concurrent_questions)

        all_results = []
        for i, item in enumerate(items, 1):
            item_id = self.dataset.get_item_id(item)
            console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")

            # Get QA pairs
            qa_pairs = self.dataset.get_qa_pairs(item)
            console.print(f"  Evaluating {len(qa_pairs)} QA pairs (parallel)...")

            qa_results = await self.evaluate_qa_task(
                agent_id,
                qa_pairs,
                item_id,
                thinking_budget,
                max_tokens,
                max_questions_per_item,
                question_semaphore,
            )

            # Calculate metrics
            metrics = await self.calculate_metrics(qa_results, eval_semaphore_size)
            console.print(f"  [green]✓[/green] Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")

            all_results.append({
                'item_id': item_id,
                'metrics': metrics,
                'num_sessions': -1  # Not tracked in two-phase mode
            })

        # Calculate overall metrics
        total_correct = sum(r['metrics']['correct'] for r in all_results)
        total_questions = sum(r['metrics']['total'] for r in all_results)
        total_invalid = sum(r['metrics'].get('invalid', 0) for r in all_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'total_invalid': total_invalid,
            'total_valid': total_valid,
            'num_items': len(items),
            'item_results': all_results
        }

    def display_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table."""
        console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")

        # Display results table
        table = Table(title="Benchmark Results", box=box.ROUNDED)
        table.add_column("Item ID", style="cyan")
        table.add_column("Sessions", justify="right", style="yellow")
        table.add_column("Questions", justify="right", style="blue")
        table.add_column("Correct", justify="right", style="green")
        table.add_column("Invalid", justify="right", style="red")
        table.add_column("Accuracy", justify="right", style="magenta")

        for result in results['item_results']:
            metrics = result['metrics']
            invalid_count = metrics.get('invalid', 0)
            invalid_str = str(invalid_count) if invalid_count > 0 else "-"
            table.add_row(
                result['item_id'],
                str(result['num_sessions']),
                str(metrics['total']),
                str(metrics['correct']),
                invalid_str,
                f"{metrics['accuracy']:.1f}%"
            )

        overall_invalid = results.get('total_invalid', 0)
        invalid_str = str(overall_invalid) if overall_invalid > 0 else "-"
        table.add_row(
            "[bold]OVERALL[/bold]",
            "-",
            f"[bold]{results['total_questions']}[/bold]",
            f"[bold]{results['total_correct']}[/bold]",
            f"[bold]{invalid_str}[/bold]",
            f"[bold]{results['overall_accuracy']:.1f}%[/bold]"
        )

        console.print(table)

        # Display note about invalid questions if any
        if overall_invalid > 0:
            console.print(f"\n[yellow]Note: {overall_invalid} question(s) marked as invalid due to errors (excluded from accuracy calculation)[/yellow]")

    def merge_results(self, new_results: Dict[str, Any], existing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new results into existing results.

        Updates or adds item results, then recalculates overall metrics.

        Args:
            new_results: New results to merge (typically from a specific item run)
            existing_results: Existing results to merge into

        Returns:
            Merged results with updated overall metrics
        """
        # Start with existing item results
        merged_item_results = existing_results.get('item_results', [])

        # Update or add new item results
        for new_item in new_results['item_results']:
            item_id = new_item['item_id']

            # Find if item already exists
            found = False
            for i, existing_item in enumerate(merged_item_results):
                if existing_item['item_id'] == item_id:
                    # Replace existing item result
                    merged_item_results[i] = new_item
                    found = True
                    console.print(f"    [yellow]→[/yellow] Updated results for item: {item_id}")
                    break

            if not found:
                # Add new item result
                merged_item_results.append(new_item)
                console.print(f"    [green]+[/green] Added results for item: {item_id}")

        # Recalculate overall metrics from all item results
        total_correct = sum(r['metrics']['correct'] for r in merged_item_results)
        total_questions = sum(r['metrics']['total'] for r in merged_item_results)
        total_invalid = sum(r['metrics'].get('invalid', 0) for r in merged_item_results)
        total_valid = total_questions - total_invalid
        # Calculate accuracy excluding invalid questions
        overall_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'total_invalid': total_invalid,
            'total_valid': total_valid,
            'num_items': len(merged_item_results),
            'item_results': merged_item_results
        }

    def save_results(self, results: Dict[str, Any], output_path: Path, merge_with_existing: bool = False):
        """
        Save results to JSON file.

        Args:
            results: Results to save
            output_path: Path to save results to
            merge_with_existing: If True, merge with existing results file if it exists
        """
        if merge_with_existing and output_path.exists():
            # Load existing results
            with open(output_path, 'r') as f:
                existing_results = json.load(f)

            console.print(f"\n[cyan]Merging with existing results from {output_path}...[/cyan]")
            results = self.merge_results(results, existing_results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]✓[/green] Results saved to {output_path}")
