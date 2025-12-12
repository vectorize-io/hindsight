"""
LoComo-specific benchmark implementations.

Provides dataset, answer generator, and evaluator for the LoComo benchmark.
"""
import sys
from pathlib import Path

from benchmarks.common.benchmark_runner import BenchmarkRunner

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import pydantic
from openai import AsyncOpenAI
import os

from benchmarks.common.benchmark_runner import BenchmarkDataset, LLMAnswerGenerator, LLMAnswerEvaluator
from hindsight_api.engine.llm_wrapper import LLMConfig

class LoComoDataset(BenchmarkDataset):
    """LoComo dataset implementation."""

    def load(self, path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LoComo dataset from JSON file."""
        with open(path, 'r') as f:
            dataset = json.load(f)

        if max_items:
            dataset = dataset[:max_items]

        return dataset

    def get_item_id(self, item: Dict) -> str:
        """Get sample ID from LoComo item."""
        return item['sample_id']

    def prepare_sessions_for_ingestion(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Prepare LoComo conversation for batch ingestion.

        Each session is ingested as a separate item with its own date.

        Returns:
            List of session dicts, each containing 'content', 'context', 'event_date', 'document_id'
        """
        conv = item['conversation']
        speaker_a = conv['speaker_a']
        speaker_b = conv['speaker_b']

        # Get all session keys sorted
        session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])

        session_items = []

        for session_key in session_keys:
            if session_key not in conv or not isinstance(conv[session_key], list):
                continue

            session_data = conv[session_key]

            # Get session date
            date_key = f"{session_key}_date_time"
            session_date = self._parse_date(conv.get(date_key))
            session_content = json.dumps(session_data)
            document_id = f"{item['sample_id']}_{session_key}"
            session_items.append({
                "content": session_content,
                "context": f"Conversation between {speaker_a} and {speaker_b} ({session_key} of {item['sample_id']})",
                "event_date": session_date,
                "document_id": document_id
            })

        return session_items

    def get_qa_pairs(self, item: Dict) -> List[Dict[str, Any]]:
        """
        Extract QA pairs from LoComo item.

        Returns:
            List of QA dicts with 'question', 'answer', 'category'
        """
        return item['qa']

    def _parse_date(self, date_string: str) -> datetime:
        """Parse LoComo date format to datetime."""
        # Format: "1:56 pm on 8 May, 2023"
        try:
            dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
            return dt.replace(tzinfo=timezone.utc)
        except:
            raise


class QuestionAnswer(pydantic.BaseModel):
    """Answer format for LoComo questions."""
    answer: str
    reasoning: str


class LoComoAnswerGenerator(LLMAnswerGenerator):
    """LoComo-specific answer generator using configurable LLM provider."""

    def __init__(self):
        """Initialize with LLM configuration for answer generation."""
        self.llm_config = LLMConfig.for_answer_generation()
        self.client = self.llm_config._client
        self.model = self.llm_config.model

    async def generate_answer(
        self,
        question: str,
        recall_result: Dict[str, Any],
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories using Groq.

        Args:
            question: The question text
            recall_result: Full RecallResult dict containing results, entities, chunks, and trace
            question_date: Date when the question was asked (for temporal context)
            question_type: Question category (unused in Locomo)

        Returns:
            Tuple of (answer, reasoning, None)
            - None indicates to use the memories from recall_result
        """
        context = json.dumps(recall_result)

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
4. If the memories contain contradictory information or multiple instances of an event, say them all
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events
7. If the answer is not explicitly stated in the memories, use logical reasoning based on the information available to answer (e.g. calculate duration of an event from different memories).

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


class LoComoThinkAnswerGenerator(LLMAnswerGenerator):
    """LoComo answer generator using the think API instead of search + LLM.

    This generator performs its own retrieval internally via the think API,
    so it doesn't need external search to be performed by the benchmark runner.
    """

    def __init__(self, memory: 'MemoryEngine', agent_id: str, thinking_budget: int = 500):
        """Initialize with memory instance and agent_id.

        Args:
            memory: MemoryEngine instance
            agent_id: Agent identifier for think queries
            thinking_budget: Budget for memory exploration
        """
        self.memory = memory
        self.agent_id = agent_id
        self.thinking_budget = thinking_budget

    def needs_external_search(self) -> bool:
        """Think API does its own retrieval, so no external search needed."""
        return False

    async def generate_answer(
        self,
        question: str,
        recall_result: Dict[str, Any],
        question_date: Optional[datetime] = None,
        question_type: Optional[str] = None
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer using the integrated think API.

        The think API performs both search and answer generation in a single call,
        combining agent facts, world facts, and opinions to formulate a response.

        Args:
            question: Question to answer
            recall_result: Not used (empty dict), as think does its own retrieval
            question_date: Date when the question was asked (currently not used by think API)
            question_type: Question category (unused in think API)

        Returns:
            Tuple of (answer, reasoning, retrieved_memories)
            - retrieved_memories: Combined list of all facts from based_on (world, agent, opinion)
        """
        try:
            # Use the think API which does both search and answer generation
            result = await self.memory.think_async(
                agent_id=self.agent_id,
                query=question,
                thinking_budget=self.thinking_budget,
            )

            # Extract answer and reasoning
            answer = result.text

            # Extract memories from based_on
            based_on = result.based_on
            world_facts = based_on.get('world', [])
            agent_facts = based_on.get('agent', [])
            opinion_facts = based_on.get('opinion', [])

            # Combine all facts into retrieved_memories
            retrieved_memories = []

            # Add world facts
            for fact in world_facts:
                retrieved_memories.append(fact.model_dump())

            for fact in agent_facts:
                retrieved_memories.append(fact.model_dump())
            for fact in opinion_facts:
                retrieved_memories.append(fact.model_dump())
            # Build reasoning summary
            num_world = len(world_facts)
            num_agent = len(agent_facts)
            num_opinion = len(opinion_facts)

            reasoning = f"Think API: {num_world} world facts, {num_agent} agent facts, {num_opinion} opinions"

            return answer, reasoning, retrieved_memories
        except Exception as e:
            return f"Error generating answer: {str(e)}", "Error occurred during think API call.", []


async def run_benchmark(
    max_conversations: int = None,
    max_questions_per_conv: int = None,
    skip_ingestion: bool = False,
    use_think: bool = False,
    conversation: str = None,
    api_url: str = None,
    max_concurrent_questions_override: int = None,
    only_failed: bool = False,
    only_invalid: bool = False
):
    """
    Run the LoComo benchmark.

    Args:
        max_conversations: Maximum number of conversations to evaluate (None for all)
        max_questions_per_conv: Maximum questions per conversation (None for all)
        skip_ingestion: Whether to skip ingestion and use existing data
        use_think: Whether to use the think API instead of search + LLM
        conversation: Specific conversation ID to run (e.g., "conv-26")
        api_url: Optional API URL to connect to (default: use local memory)
        only_failed: If True, only run conversations that have failed questions (is_correct=False)
        only_invalid: If True, only run conversations that have invalid questions (is_invalid=True)
    """
    from rich.console import Console
    console = Console()

    # Load previous results if filtering for failed/invalid conversations
    failed_conversation_ids = set()
    invalid_conversation_ids = set()
    if only_failed or only_invalid:
        suffix = "_think" if use_think else ""
        results_filename = f'benchmark_results{suffix}.json'
        results_path = Path(__file__).parent / 'results' / results_filename

        if not results_path.exists():
            console.print(f"[red]Error: Cannot use --only-failed or --only-invalid without existing results file[/red]")
            console.print(f"[yellow]Results file not found: {results_path}[/yellow]")
            return

        with open(results_path, 'r') as f:
            previous_results = json.load(f)

        # Extract conversation IDs that have failed or invalid questions
        for item_result in previous_results.get('item_results', []):
            item_id = item_result['item_id']
            for detail in item_result['metrics'].get('detailed_results', []):
                if only_failed and detail.get('is_correct') == False and not detail.get('is_invalid', False):
                    failed_conversation_ids.add(item_id)
                if only_invalid and detail.get('is_invalid', False):
                    invalid_conversation_ids.add(item_id)

        if only_failed:
            console.print(f"[cyan]Filtering to {len(failed_conversation_ids)} conversations with failed questions (is_correct=False)[/cyan]")
        if only_invalid:
            console.print(f"[cyan]Filtering to {len(invalid_conversation_ids)} conversations with invalid questions (is_invalid=True)[/cyan]")

        target_ids = failed_conversation_ids if only_failed else invalid_conversation_ids
        if not target_ids:
            filter_type = "failed" if only_failed else "invalid"
            console.print(f"[yellow]No conversations with {filter_type} questions found in previous results. Nothing to run.[/yellow]")
            return

    # Initialize components
    dataset = LoComoDataset()

    # Use remote API client if api_url is provided, otherwise use local memory
    if api_url:
        from benchmarks.common.benchmark_runner import HindsightClientAdapter
        memory = HindsightClientAdapter(base_url=api_url)
        await memory.initialize()
    else:
        from benchmarks.common.benchmark_runner import create_memory_engine
        memory = await create_memory_engine()

    if use_think:
        answer_generator = LoComoThinkAnswerGenerator(
            memory=memory,
            agent_id="locomo",
            thinking_budget=500
        )
        max_concurrent_questions = max_concurrent_questions_override or 4
        eval_semaphore_size = 4
    else:
        answer_generator = LoComoAnswerGenerator()
        # Reduced from 32 to 10 to match search semaphore limit
        # Prevents "too many connections" errors
        max_concurrent_questions = max_concurrent_questions_override or 10
        eval_semaphore_size = 8

    answer_evaluator = LLMAnswerEvaluator()

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        answer_generator=answer_generator,
        answer_evaluator=answer_evaluator,
        memory=memory
    )

    # Filter dataset if using --only-failed or --only-invalid
    dataset_path = Path(__file__).parent / 'datasets' / 'locomo10.json'

    if only_failed or only_invalid:
        # Load and filter dataset
        target_ids = failed_conversation_ids if only_failed else invalid_conversation_ids
        original_items = dataset.load(dataset_path, max_conversations)
        filtered_items = [item for item in original_items if dataset.get_item_id(item) in target_ids]
        console.print(f"[green]Found {len(filtered_items)} conversations to re-evaluate[/green]")

        # Temporarily replace dataset's load method
        original_load = dataset.load
        def filtered_load(path: Path, max_items: Optional[int] = None):
            return filtered_items[:max_items] if max_items else filtered_items
        dataset.load = filtered_load

    # Determine output filename based on mode
    suffix = "_think" if use_think else ""
    results_filename = f'benchmark_results{suffix}.json'
    output_path = Path(__file__).parent / 'results' / results_filename

    # Create results directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing results if running a specific conversation or using filters
    merge_with_existing = conversation is not None or only_failed or only_invalid

    # Run benchmark with parallel conversation processing
    # Each conversation gets its own agent ID (locomo_conv-26, locomo_conv-30, etc.)
    # This allows conversations to run in parallel (up to max_concurrent_items at a time)
    results = await runner.run(
        dataset_path=dataset_path,
        agent_id="locomo",
        max_items=max_conversations,
        max_questions_per_item=max_questions_per_conv,
        thinking_budget=500,
        max_tokens=4096,
        skip_ingestion=skip_ingestion,
        max_concurrent_questions=max_concurrent_questions,
        eval_semaphore_size=eval_semaphore_size,
        specific_item=conversation,
        clear_agent_per_item=True,  # Use unique agent ID per conversation
        max_concurrent_items=3,  # Process up to 3 conversations in parallel
        output_path=output_path,  # Save results incrementally
        merge_with_existing=merge_with_existing
    )

    # Display results (final save already happened incrementally)
    runner.display_results(results)
    console.print(f"\n[green]✓[/green] Results saved incrementally to {output_path}")

    # Generate markdown table
    generate_markdown_table(results, use_think)

    return results


def generate_markdown_table(results: dict, use_think: bool = False):
    """
    Generate a markdown table with benchmark results.

    Category mapping:
    1 = Multi-hop
    2 = Single-hop
    3 = Temporal
    4 = Open-domain
    """
    from rich.console import Console
    console = Console()

    category_names = {
        '1': 'Multi-hop',
        '2': 'Single-hop',
        '3': 'Temporal',
        '4': 'Open-domain'
    }

    # Build markdown content
    lines = []
    mode_str = " (Think Mode)" if use_think else ""
    lines.append(f"# LoComo Benchmark Results{mode_str}")
    lines.append("")

    # Add model configuration
    if 'model_config' in results:
        config = results['model_config']
        lines.append("## Model Configuration")
        lines.append("")
        lines.append(f"- **Hindsight**: {config['hindsight']['provider']}/{config['hindsight']['model']}")
        lines.append(f"- **Answer Generation**: {config['answer_generation']['provider']}/{config['answer_generation']['model']}")
        lines.append(f"- **LLM Judge**: {config['judge']['provider']}/{config['judge']['model']}")
        lines.append("")

    lines.append(f"**Overall Accuracy**: {results['overall_accuracy']:.2f}% ({results['total_correct']}/{results['total_questions']})")
    lines.append("")
    lines.append("| Sample ID | Sessions | Questions | Correct | Accuracy | Multi-hop | Single-hop | Temporal | Open-domain |")
    lines.append("|-----------|----------|-----------|---------|----------|-----------|------------|----------|-------------|")

    for item_result in results['item_results']:
        item_id = item_result['item_id']
        num_sessions = item_result['num_sessions']
        metrics = item_result['metrics']

        # Calculate category accuracies
        cat_stats = metrics.get('category_stats', {})
        cat_accuracies = {}

        for cat_id in ['1', '2', '3', '4']:
            if cat_id in cat_stats:
                stats = cat_stats[cat_id]
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                cat_accuracies[cat_id] = f"{acc:.1f}% ({stats['correct']}/{stats['total']})"
            else:
                cat_accuracies[cat_id] = "N/A"

        lines.append(
            f"| {item_id} | {num_sessions} | {metrics['total']} | {metrics['correct']} | "
            f"{metrics['accuracy']:.2f}% | {cat_accuracies['1']} | {cat_accuracies['2']} | "
            f"{cat_accuracies['3']} | {cat_accuracies['4']} |"
        )

    # Write to file with suffix
    suffix = "_think" if use_think else ""
    output_file = Path(__file__).parent / 'results' / f'results_table{suffix}.md'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text('\n'.join(lines))
    console.print(f"\n[green]✓[/green] Results table saved to {output_file}")


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description='Run LoComo benchmark')
    parser.add_argument('--max-conversations', type=int, default=None, help='Maximum conversations to evaluate')
    parser.add_argument('--max-questions', type=int, default=None, help='Maximum questions per conversation')
    parser.add_argument('--skip-ingestion', action='store_true', help='Skip ingestion and use existing data')
    parser.add_argument('--use-think', action='store_true', help='Use think API instead of search + LLM')
    parser.add_argument('--conversation', type=str, default=None, help='Run only specific conversation (e.g., "conv-26")')
    parser.add_argument('--api-url', type=str, default=None, help='Hindsight API URL (default: use local memory, example: http://localhost:8888)')
    parser.add_argument('--max-concurrent-questions', type=int, default=None, help='Max concurrent questions per conversation (default: 4 for think, 10 for search)')
    parser.add_argument('--only-failed', action='store_true', help='Only run conversations that have failed questions (is_correct=False). Requires existing results file.')
    parser.add_argument('--only-invalid', action='store_true', help='Only run conversations that have invalid questions (is_invalid=True). Requires existing results file.')

    args = parser.parse_args()

    # Validate that only one of --only-failed or --only-invalid is set
    if args.only_failed and args.only_invalid:
        parser.error("Cannot use both --only-failed and --only-invalid at the same time")

    results = asyncio.run(run_benchmark(
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        skip_ingestion=args.skip_ingestion,
        use_think=args.use_think,
        conversation=args.conversation,
        api_url=args.api_url,
        max_concurrent_questions_override=args.max_concurrent_questions,
        only_failed=args.only_failed,
        only_invalid=args.only_invalid
    ))
