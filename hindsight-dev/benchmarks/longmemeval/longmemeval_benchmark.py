"""
LongMemEval-specific benchmark implementations.

Provides dataset, answer generator, and evaluator for the LongMemEval benchmark.
"""
import sys
from pathlib import Path

from benchmarks.common.benchmark_runner import BenchmarkRunner
from hindsight_api import MemoryEngine

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import pydantic
from openai import AsyncOpenAI
import os

from benchmarks.common.benchmark_runner import BenchmarkDataset, LLMAnswerGenerator, LLMAnswerEvaluator
from hindsight_api.engine.llm_wrapper import LLMConfig


class LongMemEvalDataset(BenchmarkDataset):
    """LongMemEval dataset implementation."""

    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LongMemEval dataset from JSON file."""
        with open(path, 'r') as f:
            dataset = json.load(f)

        if max_items:
            dataset = dataset[:max_items]

        return dataset

    def get_item_id(self, item: Dict) -> str:
        """Get question ID from LongMemEval item."""
        return item.get("question_id", "unknown")

    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare LongMemEval conversation sessions for batch ingestion.

        Returns:
            List of session dicts with 'content', 'context', 'event_date'
        """
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates", [])
        session_ids = item.get("haystack_session_ids", [])

        # Ensure all lists have same length
        if not (len(sessions) == len(dates) == len(session_ids)):
            min_len = min(len(sessions), len(dates), len(session_ids))
            sessions = sessions[:min_len]
            dates = dates[:min_len]
            session_ids = session_ids[:min_len]

        batch_contents = []

        # Process each session
        for session_turns, date_str, session_id in zip(sessions, dates, session_ids):
            # Parse session date
            session_date = self._parse_date(date_str) if date_str else datetime.now(timezone.utc)

            # Combine all turns in the session into one content string
            session_content_parts = []
            for turn_dict in session_turns:
                role = turn_dict.get("role", "")
                content = turn_dict.get("content", "")

                if not content.strip():
                    continue

                # Format as "role: content"
                session_content_parts.append(f"{role}: {content}")

            # Add session to batch
            if session_content_parts:
                session_content = "\n".join(session_content_parts)
                question_id = item.get("question_id", "unknown")
                document_id = f"{question_id}_{session_id}"
                batch_contents.append({
                    "content": session_content,
                    "context": f"Session {session_id}",
                    "event_date": session_date,
                    "document_id": document_id
                })

        return batch_contents

    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from LongMemEval item.

        For LongMemEval, each item has one question.

        Returns:
            List with single QA dict with 'question', 'answer', 'category', 'question_date'
        """
        # Parse question_date if available
        question_date = None
        if 'question_date' in item:
            question_date = self._parse_date(item['question_date'])

        return [{
            'question': item.get("question", ""),
            'answer': item.get("answer", ""),
            'category': item.get("question_type", "unknown"),
            'question_date': question_date
        }]

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            # LongMemEval format: "2023/05/20 (Sat) 02:21"
            # Try to parse the main part before the day name
            date_str_cleaned = date_str.split('(')[0].strip() if '(' in date_str else date_str

            # Try multiple formats
            for fmt in ["%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                try:
                    dt = datetime.strptime(date_str_cleaned, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

            # Fallback: try ISO format
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except Exception:
            return datetime.now(timezone.utc)


class QuestionAnswer(pydantic.BaseModel):
    answer: str
    reasoning: str

class LongMemEvalAnswerGenerator(LLMAnswerGenerator):
    """LongMemEval-specific answer generator using configurable LLM provider."""

    def __init__(self):
        """Initialize with LLM configuration for memory operations."""
        self.llm_config = LLMConfig.for_memory()
        self.client = self.llm_config._client
        self.model = self.llm_config.model

    async def generate_answer(
                self,
                question: str,
                memories: List[Dict[str, Any]],
                question_date: Optional[datetime] = None
        ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
            """
            Generate answer from retrieved memories using Groq.

            Args:
                question: The question text
                memories: Retrieved memories
                question_date: Date when the question was asked (for temporal context)

            Returns:
                Tuple of (answer, reasoning, None)
                - None indicates to use the memories passed in
            """
            # Format context
            context_parts = []
            for result in memories:
                context_parts.append({"text": result.get("text"), "context": result.get("context"),
                                      "event_date": result.get("event_date")})

            context = json.dumps(context_parts)

            # Format question date if provided
            question_date_str = ""
            if question_date:
                question_date_str = f"\n# CURRENT DATE:\nThe question is being asked on: {question_date.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"

            # Use LLM to generate answer
            try:
                answer_obj = await self.llm_config.call(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful expert assistant answering questions from lme_experiment users based on the provided context."
                        },
                        {
                            "role": "user",
                            "content": f"""
    # CONTEXT:
    You have access to facts and entities from a conversation.
{question_date_str}
    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. Always convert relative time references to specific dates, months, or years.
    6. Be as specific as possible when talking about people, places, and events
    7. Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.

    Clarification:
    When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

    Example:

    Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
    Question: What day did I go to the vet?
    Correct Answer: March 15, 2023
    Explanation:
    Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.


    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references
    8. If you're not exactly sure, still try to attempt an answer. Sometimes the terms are sligtly different from the question, so it's better to try with the current evidence than just say you don't know.
    9. Say that you cannot answer if no evidence is related to the question.
    10. Instead of saying "I don't know", you can use the most relevant information you found in the memories to construct a best-effort answer (but you need to use the provided context).
    11. Provide a complete answer with your reasoning.

    Context:

    {context}

    Question: {question}
    Answer:

    """
                        }
                    ],
                    response_format=QuestionAnswer,
                    scope="memory"
                )
                return answer_obj.answer, answer_obj.reasoning, None
            except Exception as e:
                return f"Error generating answer: {str(e)}", "Error occurred during answer generation.", None


async def run_benchmark(
    max_instances: int = None,
    max_questions_per_instance: int = None,
    thinking_budget: int = 500,
    max_tokens: int = 8192,
    skip_ingestion: bool = False,
    api_url: str = None,
    filln: bool = False,
    question_id: str = None,
    only_failed: bool = False,
    only_invalid: bool = False
):
    """
    Run the LongMemEval benchmark.

    Args:
        max_instances: Maximum number of instances to evaluate (None for all)
        max_questions_per_instance: Maximum questions per instance (for testing)
        thinking_budget: Thinking budget for spreading activation search
        max_tokens: Maximum tokens to retrieve from memories
        skip_ingestion: Whether to skip ingestion and use existing data
        api_url: Optional API URL to connect to (default: use local memory)
        filln: If True, only process questions where the agent has no indexed data yet
        question_id: Optional question ID to filter (e.g., 'e47becba'). Useful with --skip-ingestion.
        only_failed: If True, only run questions that were previously marked as incorrect (is_correct=False)
        only_invalid: If True, only run questions that were previously marked as invalid (is_invalid=True)
    """
    from rich.console import Console
    console = Console()

    # Check dataset exists, download if needed
    dataset_path = Path(__file__).parent / "datasets" / "longmemeval_s_cleaned.json"
    if not dataset_path.exists():
        if not download_dataset(dataset_path):
            console.print(f"[red]Failed to download dataset. Please download manually:[/red]")
            console.print("[yellow]curl -L 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json' -o benchmarks/longmemeval/datasets/longmemeval_s_cleaned.json[/yellow]")
            return

    # Load previous results if filtering for failed/invalid questions
    failed_question_ids = set()
    invalid_question_ids = set()
    if only_failed or only_invalid:
        results_path = Path(__file__).parent / 'results' / 'benchmark_results.json'
        if not results_path.exists():
            console.print(f"[red]Error: Cannot use --only-failed or --only-invalid without existing results file[/red]")
            console.print(f"[yellow]Results file not found: {results_path}[/yellow]")
            return

        with open(results_path, 'r') as f:
            previous_results = json.load(f)

        # Extract question IDs that failed or are invalid
        for item_result in previous_results.get('item_results', []):
            item_id = item_result['item_id']
            for detail in item_result['metrics'].get('detailed_results', []):
                if only_failed and detail.get('is_correct') == False and not detail.get('is_invalid', False):
                    failed_question_ids.add(item_id)
                if only_invalid and detail.get('is_invalid', False):
                    invalid_question_ids.add(item_id)

        if only_failed:
            console.print(f"[cyan]Filtering to {len(failed_question_ids)} questions that failed (is_correct=False)[/cyan]")
        if only_invalid:
            console.print(f"[cyan]Filtering to {len(invalid_question_ids)} questions that were invalid (is_invalid=True)[/cyan]")

    # Initialize components
    dataset = LongMemEvalDataset()

    # Filter dataset based on failed/invalid flags
    if only_failed or only_invalid:
        target_ids = failed_question_ids if only_failed else invalid_question_ids
        if not target_ids:
            filter_type = "failed" if only_failed else "invalid"
            console.print(f"[yellow]No {filter_type} questions found in previous results. Nothing to run.[/yellow]")
            return
        # Override question_id to be None if we're filtering by failed/invalid
        # The filtering will happen when we load the dataset
        original_dataset_items = dataset.load(dataset_path, max_instances)
        filtered_items = [item for item in original_dataset_items if dataset.get_item_id(item) in target_ids]
        console.print(f"[green]Found {len(filtered_items)} items to re-evaluate[/green]")

    answer_generator = LongMemEvalAnswerGenerator()
    answer_evaluator = LLMAnswerEvaluator()

    # Use remote API client if api_url is provided, otherwise use local memory
    if api_url:
        from benchmarks.common.benchmark_runner import HindsightClientAdapter
        memory = HindsightClientAdapter(base_url=api_url)
    else:
        memory = MemoryEngine(
            db_url=os.getenv("HINDSIGHT_API_DATABASE_URL"),
            memory_llm_provider=os.getenv("HINDSIGHT_API_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("HINDSIGHT_API_LLM_API_KEY"),
            memory_llm_model=os.getenv("HINDSIGHT_API_LLM_MODEL", "openai/gpt-oss-120b"),
            memory_llm_base_url=os.getenv("HINDSIGHT_API_LLM_BASE_URL") or None,  # Use None to get provider defaults
        )

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        answer_generator=answer_generator,
        answer_evaluator=answer_evaluator,
        memory=memory
    )

    # If filtering by failed/invalid, we need to use a custom dataset that only returns those items
    # We'll temporarily replace the dataset's load method
    if only_failed or only_invalid:
        original_load = dataset.load
        def filtered_load(path: Path, max_items: Optional[int] = None):
            return filtered_items[:max_items] if max_items else filtered_items
        dataset.load = filtered_load

    # Run benchmark
    # Single-phase approach: each question gets its own isolated agent_id
    # This ensures each question only has access to its own context
    results = await runner.run(
        dataset_path=dataset_path,
        agent_id="longmemeval",  # Will be suffixed with question_id per item
        max_items=max_instances,
        max_questions_per_item=max_questions_per_instance,
        thinking_budget=thinking_budget,
        max_tokens=max_tokens,
        skip_ingestion=skip_ingestion,
        max_concurrent_questions=8,
        eval_semaphore_size=8,
        separate_ingestion_phase=False,  # Process each question independently
        clear_agent_per_item=True,  # Use unique agent_id per question
        filln=filln,  # Only process questions without indexed data
        specific_item=question_id  # Optional filter for specific question ID
    )

    # Display and save results
    runner.display_results(results)
    runner.save_results(
        results,
        Path(__file__).parent / 'results' / 'benchmark_results.json',
        merge_with_existing=(filln or question_id is not None or only_failed or only_invalid)  # Merge when using --fill, --only-failed, --only-invalid flags or specific question
    )

    # Generate detailed report by question type
    generate_type_report(results)

    return results


def download_dataset(dataset_path: Path) -> bool:
    """
    Download the LongMemEval dataset if it doesn't exist.

    Returns:
        True if successful, False otherwise
    """
    import subprocess
    from rich.console import Console
    console = Console()

    url = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

    console.print(f"[yellow]Dataset not found. Downloading from HuggingFace...[/yellow]")
    console.print(f"[dim]URL: {url}[/dim]")
    console.print(f"[dim]Destination: {dataset_path}[/dim]")

    # Create parent directory if it doesn't exist
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use curl to download with progress
        result = subprocess.run(
            ["curl", "-L", "-o", str(dataset_path), url],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0 and dataset_path.exists():
            console.print(f"[green]✓ Dataset downloaded successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ Download failed: {result.stderr}[/red]")
            return False

    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Download timed out after 5 minutes[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Download error: {e}[/red]")
        return False


def generate_type_report(results: dict):
    """Generate a detailed report by question type."""
    from rich.table import Table
    from rich.console import Console
    console = Console()

    # Aggregate stats by question type
    type_stats = {}

    for item_result in results['item_results']:
        metrics = item_result['metrics']
        by_category = metrics.get('category_stats', {})

        for qtype, stats in by_category.items():
            if qtype not in type_stats:
                type_stats[qtype] = {'total': 0, 'correct': 0}
            type_stats[qtype]['total'] += stats['total']
            type_stats[qtype]['correct'] += stats['correct']

    # Display table
    table = Table(title="Performance by Question Type")
    table.add_column("Question Type", style="cyan")
    table.add_column("Total", justify="right", style="yellow")
    table.add_column("Correct", justify="right", style="green")
    table.add_column("Accuracy", justify="right", style="magenta")

    for qtype, stats in sorted(type_stats.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        table.add_row(
            qtype,
            str(stats['total']),
            str(stats['correct']),
            f"{acc:.1f}%"
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Limit number of instances to evaluate (default: all 500)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions per instance (for quick testing)"
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=500,
        help="Thinking budget for spreading activation search"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to retrieve from memories"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip ingestion and use existing data"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Hindsight API URL (default: use local memory, example: http://localhost:8888)"
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Only process questions where the agent has no indexed data yet (for resuming interrupted runs)"
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Filter to specific question ID (e.g., 'e47becba'). Useful with --skip-ingestion to test a single question."
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only run questions that were previously marked as incorrect (is_correct=False). Requires existing results file."
    )
    parser.add_argument(
        "--only-invalid",
        action="store_true",
        help="Only run questions that were previously marked as invalid (is_invalid=True). Requires existing results file."
    )

    args = parser.parse_args()

    # Validate that only one of --only-failed or --only-invalid is set
    if args.only_failed and args.only_invalid:
        parser.error("Cannot use both --only-failed and --only-invalid at the same time")

    results = asyncio.run(run_benchmark(
        max_instances=args.max_instances,
        max_questions_per_instance=args.max_questions,
        thinking_budget=args.thinking_budget,
        max_tokens=args.max_tokens,
        skip_ingestion=args.skip_ingestion,
        api_url=args.api_url,
        filln=args.fill,
        question_id=args.question_id,
        only_failed=args.only_failed,
        only_invalid=args.only_invalid
    ))
