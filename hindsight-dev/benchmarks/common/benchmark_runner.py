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
from hindsight_api.engine.memory_engine import Budget
from openai import AsyncOpenAI
import os

console = Console()


def get_model_config() -> Dict[str, Dict[str, str]]:
    """
    Get the model configuration for all three LLM roles.

    Reads directly from environment variables without instantiating LLM clients.

    Returns:
        Dict with 'hindsight', 'answer_generation', and 'judge' keys,
        each containing 'provider' and 'model' info.
    """
    # Memory/Hindsight config (base config)
    memory_provider = os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq")
    memory_model = os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b")

    # Answer generation config (falls back to memory config)
    answer_provider = os.getenv("HINDSIGHT_API_ANSWER_LLM_PROVIDER", memory_provider)
    answer_model = os.getenv("HINDSIGHT_API_ANSWER_LLM_MODEL", memory_model)

    # Judge config (falls back to memory config)
    judge_provider = os.getenv("HINDSIGHT_API_JUDGE_LLM_PROVIDER", memory_provider)
    judge_model = os.getenv("HINDSIGHT_API_JUDGE_LLM_MODEL", memory_model)

    return {
        'hindsight': {
            'provider': memory_provider,
            'model': memory_model,
        },
        'answer_generation': {
            'provider': answer_provider,
            'model': answer_model,
        },
        'judge': {
            'provider': judge_provider,
            'model': judge_model,
        }
    }


def print_model_config():
    """Print the model configuration to console."""
    config = get_model_config()

    console.print("\n[bold cyan]Model Configuration:[/bold cyan]")
    console.print(f"  Hindsight:         {config['hindsight']['provider']}/{config['hindsight']['model']}")
    console.print(f"  Answer Generation: {config['answer_generation']['provider']}/{config['answer_generation']['model']}")
    console.print(f"  LLM Judge:         {config['judge']['provider']}/{config['judge']['model']}")
    console.print()


async def create_memory_engine() -> MemoryEngine:
    """
    Create and initialize a MemoryEngine instance from environment variables.

    Reads configuration from:
    - HINDSIGHT_API_DATABASE_URL (default: "pg0")
    - HINDSIGHT_API_LLM_PROVIDER (default: "groq")
    - HINDSIGHT_API_LLM_API_KEY
    - HINDSIGHT_API_LLM_MODEL (default: "openai/gpt-oss-120b")
    - HINDSIGHT_API_LLM_BASE_URL (optional)

    Returns:
        Initialized MemoryEngine instance
    """
    memory = MemoryEngine(
        db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
        memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
        memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
        memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
        memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,  # Use None to get provider defaults
    )
    await memory.initialize()
    return memory


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
        recall_result: Dict[str, Any],
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories.

        Args:
            question: The question text
            recall_result: Full RecallResult dict containing results, entities, chunks, and trace
            question_date: Optional date when the question was asked (for temporal context)
            question_type: Optional question category/type (e.g., 'multi-session', 'temporal-reasoning')

        Returns:
            Tuple of (answer, reasoning, retrieved_memories_override)
            - answer: The generated answer text
            - reasoning: Explanation of how the answer was derived
            - retrieved_memories_override: Optional list of memories to include in results
              - None: Use memories from recall_result (traditional mode)
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
        semaphore: asyncio.Semaphore,
        category: Optional[str] = None,
        max_retries: int = 3
    ) -> Tuple[bool, str]:
        """
        Evaluate predicted answer using LLM-as-judge with category-specific prompts.

        Args:
            question: The question
            correct_answer: Gold/correct answer
            predicted_answer: Predicted answer
            semaphore: Semaphore for rate limiting
            category: Question category for LongMemEval-specific evaluation
            max_retries: Maximum retry attempts for validation errors

        Returns:
            Tuple of (is_correct, reasoning)
        """
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    # LongMemEval-specific evaluation prompts
                    if category in ['single-session-user', 'single-session-assistant', 'multi-session']:
                        prompt_content = f"""Evaluate if the model response contains the correct answer to the question.
                        
I will give you a question, a correct answer, and a response from a model. 
Please set correct=true if the response contains the correct answer. Otherwise, set correct=no. 
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. 
If the response only contains a subset of the information required by the answer, set correct=false

Question: {question}

Correct Answer: {correct_answer}

Model Response: {predicted_answer}

Evaluation criteria:
- Set correct=true if the response contains the correct answer
- Set correct=true if the response is equivalent to the correct answer or contains intermediate steps
- Set correct=false if the response is incorrect or missing key information

Provide your evaluation as JSON with:
- reasoning: One sentence explanation
- correct: true or false"""

                    elif category == 'temporal-reasoning':
                        prompt_content = f"""
I will give you a question, a correct answer, and a response from a model. 
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. 
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also set correct=true. 
If the response only contains a subset of the information required by the answer, answer correct=false. 
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.
"""

                    elif category == 'knowledge-update':
                        prompt_content = f"""
I will give you a question, a correct answer, and a response from a model. 
Please set correct=true if the response contains the correct answer. Otherwise, set correct=false. 
If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.
"""

                    elif category == 'single-session-preference':
                        prompt_content = f"""
I will give you a question, a answer for desired personalized response, and a response from a model. 
Please set correct=true if the response satisfies the desired response. Otherwise, set correct=false. 
The model does not need to reflect all the points in the desired response. The response is correct as long as it recalls and utilizes the user's personal information correctly.
"""

                    else:
                        # Default LoComo-style evaluation
                        prompt_content = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
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
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.
"""

                    judgement = await self.llm_config.call(
                        messages=[
                            {
                                "role": "user",
                                "content": f"""{prompt_content}
                                

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
                        max_completion_tokens=4096
                    )

                    return judgement.correct, judgement.reasoning

                except Exception as e:
                    # Check if it's a validation error (LLM returned malformed JSON)
                    error_str = str(e)
                    is_validation_error = "ValidationError" in error_str or "Field required" in error_str

                    # Retry on validation errors, fail immediately on other errors
                    if is_validation_error and attempt < max_retries - 1:
                        print(f"Judge validation error on attempt {attempt + 1}/{max_retries}, retrying...")
                        await asyncio.sleep(0.5)  # Small delay before retry
                        continue

                    # Final attempt or non-validation error - log and return error
                    print(f"Error judging answer after {attempt + 1} attempts: {e}")
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
            db_url=os.getenv("HINDSIGHT_API_DATABASE_URL", "pg0"),
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-20b"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,
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
            await self.memory.retain_batch_async(
                bank_id=agent_id,
                contents=batch_contents
            )

        return len(batch_contents)

    async def answer_question(
        self,
        agent_id: str,
        question: str,
        thinking_budget: int = 500,
        max_tokens: int = 4096,
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None,
    ) -> Tuple[str, str, List[Dict], Dict[str, Dict]]:
        """
        Answer a question using memory retrieval.

        Args:
            agent_id: Agent ID
            question: Question text
            thinking_budget: Thinking budget for search
            max_tokens: Maximum tokens to retrieve
            question_date: Date when the question was asked (for temporal filtering)
            question_type: Question category/type (e.g., 'multi-session', 'temporal-reasoning')

        Returns:
            Tuple of (answer, reasoning, retrieved_memories, chunks)
        """
        # Check if generator needs external search
        if self.answer_generator.needs_external_search():
            # Traditional flow: search then generate
            # Use MemoryEngine directly
            # Map thinking_budget to budget level
            budget = Budget.LOW if thinking_budget <= 30 else Budget.MID if thinking_budget <= 70 else Budget.HIGH

            import time
            recall_start_time = time.time()
            search_result = await self.memory.recall_async(
                bank_id=agent_id,
                query=question,
                budget=budget,
                max_tokens=max_tokens,
                fact_type=["world", "experience"],
                question_date=question_date,
                include_entities=True,
                max_entity_tokens=2048,
                include_chunks=True,

            )
            recall_time = time.time() - recall_start_time

            # Log recall stats
            num_results = len(search_result.results) if search_result.results else 0
            num_chunks = len(search_result.chunks) if search_result.chunks else 0
            num_entities = len(search_result.entities) if search_result.entities else 0

            # Convert entire RecallResult to dictionary for answer generation
            recall_result_dict = search_result.model_dump()

            # Extract chunks from search result
            chunks = {}
            if search_result.chunks:
                for chunk_key, chunk_info in search_result.chunks.items():
                    chunks[chunk_key] = chunk_info.model_dump()

            # Check if we have any results
            if not search_result.results:
                return "I don't have enough information to answer that question.", "No relevant memories found.", [], {}

            # Generate answer using LLM - pass entire recall result
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(question, recall_result_dict, question_date, question_type)

            # Use override if provided, otherwise use the results from recall
            final_memories = memories_override if memories_override is not None else [fact.model_dump() for fact in search_result.results]

            return answer, reasoning, final_memories, chunks
        else:
            # Integrated flow: generator does its own search (e.g., think API)
            # Pass empty recall result since generator doesn't need them
            answer, reasoning, memories_override = await self.answer_generator.generate_answer(question, {"results": []}, question_date, question_type)

            # Use memories from generator (should not be None for integrated mode)
            final_memories = memories_override if memories_override is not None else []

            return answer, reasoning, final_memories, {}

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
        # First, identify and log category 5 questions that will be skipped
        category_5_questions = [pair for pair in qa_pairs if pair.get('category') == 5]
        if category_5_questions:
            logging.info(f"Skipping {len(category_5_questions)} category=5 questions for {item_id}")
            for q in category_5_questions:
                logging.debug(f"  Skipped category=5 question: {q.get('question', 'N/A')[:100]}")

        # Filter out category 5 and questions without answers
        qa_pairs = [pair for pair in qa_pairs if pair.get('category') != 5 and pair.get('answer')]
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
                        # Get predicted answer, reasoning, retrieved memories, and chunks
                        predicted_answer, reasoning, retrieved_memories, chunks = await self.answer_question(
                            agent_id, question, thinking_budget, max_tokens, question_date, category
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
                        semaphore,
                        category=result.get('category')
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
            # Use direct database access for local memory
            pool = await self.memory._get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT COUNT(*) as count FROM memory_units WHERE bank_id = $1 LIMIT 1",
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
                await self.memory.delete_bank(agent_id)
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
        output_path: Optional[Path] = None,  # Path to save results incrementally
        merge_with_existing: bool = False,  # Whether to merge with existing results
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

        # Print model configuration
        print_model_config()

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
                max_concurrent_questions, eval_semaphore_size,
                output_path, merge_with_existing
            )
        else:
            # Original approach: process each item independently
            return await self._run_single_phase(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item,
                max_concurrent_questions, eval_semaphore_size,
                clear_agent_per_item, filln, max_concurrent_items,
                output_path, merge_with_existing
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
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
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
                eval_semaphore_size, filln, max_concurrent_items,
                output_path, merge_with_existing
            )
        else:
            # Sequential item processing (original behavior)
            all_results = await self._process_items_sequential(
                items, agent_id, thinking_budget, max_tokens,
                skip_ingestion, max_questions_per_item, question_semaphore,
                eval_semaphore_size, clear_agent_per_item, filln,
                output_path, merge_with_existing
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
            'model_config': get_model_config(),
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
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
    ) -> List[Dict]:
        """Process items sequentially (original behavior)."""
        all_results = []
        existing_item_ids = set()

        # Load existing results if merge_with_existing is True
        if merge_with_existing and output_path and output_path.exists():
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                if 'item_results' in existing_data:
                    all_results = existing_data['item_results']
                    existing_item_ids = {r['item_id'] for r in all_results}
                    console.print(f"[cyan]Loaded {len(all_results)} existing results from {output_path}[/cyan]")

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

            # Check if we should skip this item (fill mode - skip if already in results file)
            item_id = self.dataset.get_item_id(item)
            if filln:
                if item_id in existing_item_ids:
                    console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                    console.print(f"  [yellow]⊘[/yellow] Skipping - already has results in output file")
                    continue

            result = await self.process_single_item(
                item, item_agent_id, i, len(items),
                thinking_budget, max_tokens, max_questions_per_item,
                skip_ingestion, question_semaphore, eval_semaphore_size,
                clear_this_agent,
            )

            # Replace existing result or append new one
            result_item_id = result['item_id']
            if result_item_id in existing_item_ids:
                # Replace existing result
                all_results = [r for r in all_results if r['item_id'] != result_item_id]
                console.print(f"  [cyan]↻[/cyan] Updating existing result for {result_item_id}")
            all_results.append(result)
            existing_item_ids.add(result_item_id)

            # Save results incrementally after each item
            if output_path:
                self._save_incremental_results(all_results, output_path)

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
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
    ) -> List[Dict]:
        """Process items in parallel (requires unique agent IDs per item)."""
        # Load existing results if merge_with_existing is True
        all_results = []
        existing_item_ids = set()
        result_lock = asyncio.Lock()  # Lock for thread-safe updates to all_results

        if merge_with_existing and output_path and output_path.exists():
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
                if 'item_results' in existing_data:
                    all_results = existing_data['item_results']
                    existing_item_ids = {r['item_id'] for r in all_results}
                    console.print(f"[cyan]Loaded {len(all_results)} existing results from {output_path}[/cyan]")

        # Create semaphore for item-level parallelism
        item_semaphore = asyncio.Semaphore(max_concurrent_items)

        async def process_item_wrapper(i: int, item: Dict) -> Optional[Dict]:
            """Wrapper to process a single item with semaphore control."""
            async with item_semaphore:
                item_id = self.dataset.get_item_id(item)
                item_agent_id = f"{agent_id}_{item_id}"

                # Check if we should skip this item (fill mode - skip if already in results file)
                if filln:
                    if item_id in existing_item_ids:
                        console.print(f"\n[bold blue]Item {i}/{len(items)}[/bold blue] (ID: {item_id})")
                        console.print(f"  [yellow]⊘[/yellow] Skipping - already has results in output file")
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

        # Run in parallel and collect results incrementally
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result is not None:
                async with result_lock:
                    # Replace existing result or append new one
                    result_item_id = result['item_id']
                    if result_item_id in existing_item_ids:
                        # Replace existing result
                        all_results = [r for r in all_results if r['item_id'] != result_item_id]
                        console.print(f"  [cyan]↻[/cyan] Updating existing result for {result_item_id}")
                    all_results.append(result)
                    existing_item_ids.add(result_item_id)

                    # Save results incrementally after each item completes
                    if output_path:
                        self._save_incremental_results(all_results, output_path)

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
        output_path: Optional[Path] = None,
        merge_with_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Two-phase approach: ingest all data into single agent, then evaluate all questions.

        More realistic scenario where agent accumulates memories over time.
        """
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
            await self.memory.delete_bank(agent_id)
            console.print(f"    [green]✓[/green] Cleared agent data")

            # Collect all sessions and send in one batch (with auto-chunking)
            console.print(f"    [yellow]Collecting sessions from all items...[/yellow]")
            all_sessions = []
            for item in items:
                item_sessions = self.dataset.prepare_sessions_for_ingestion(item)
                all_sessions.extend(item_sessions)

            console.print(f"    [cyan]Collected {len(all_sessions)} sessions from {len(items)} items[/cyan]")
            console.print(f"    [yellow]Ingesting in one batch (auto-chunks if needed)...[/yellow]")

            # Ingest all sessions in one batch call (will auto-chunk if too large)
            await self.memory.retain_batch_async(
                bank_id=agent_id,
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

        # Display model configuration
        if 'model_config' in results:
            config = results['model_config']
            console.print("[bold cyan]Model Configuration:[/bold cyan]")
            console.print(f"  Hindsight:         {config['hindsight']['provider']}/{config['hindsight']['model']}")
            console.print(f"  Answer Generation: {config['answer_generation']['provider']}/{config['answer_generation']['model']}")
            console.print(f"  LLM Judge:         {config['judge']['provider']}/{config['judge']['model']}")
            console.print()

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

    def _save_incremental_results(self, all_results: List[Dict], output_path: Path):
        """
        Save results incrementally to JSON file.

        Args:
            all_results: Current list of all item results
            output_path: Path to save results to
        """
        # Calculate metrics from current results
        total_correct = sum(r['metrics']['correct'] for r in all_results)
        total_questions = sum(r['metrics']['total'] for r in all_results)
        total_invalid = sum(r['metrics'].get('invalid', 0) for r in all_results)
        total_valid = total_questions - total_invalid
        overall_accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0

        results_dict = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_questions': total_questions,
            'total_invalid': total_invalid,
            'total_valid': total_valid,
            'num_items': len(all_results),
            'model_config': get_model_config(),
            'item_results': all_results
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

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
