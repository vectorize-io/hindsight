"""
LoComo-specific benchmark implementations.

Provides dataset, answer generator, and evaluator for the LoComo benchmark.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import pydantic
from openai import AsyncOpenAI
import os

# Import common framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.benchmark_runner import BenchmarkDataset, LLMAnswerGenerator, LLMAnswerEvaluator
from memora.llm_wrapper import LLMConfig

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
        Prepare LoComo conversation sessions for batch ingestion.

        Returns:
            List of session dicts with 'content', 'context', 'event_date'
        """
        conv = item['conversation']
        speaker_a = conv['speaker_a']
        speaker_b = conv['speaker_b']

        # Get all session keys sorted
        session_keys = sorted([k for k in conv.keys() if k.startswith('session_') and not k.endswith('_date_time')])

        batch_contents = []

        for session_key in session_keys:
            if session_key not in conv or not isinstance(conv[session_key], list):
                continue

            session_data = conv[session_key]

            # Build session content from all turns
            session_parts = []
            for turn in session_data:
                speaker = turn['speaker']
                text = turn['text']
                session_parts.append(f"{speaker}: {text}")

            if not session_parts:
                continue

            # Get session date
            date_key = f"{session_key}_date_time"
            session_date = self._parse_date(conv.get(date_key, "1:00 pm on 1 January, 2023"))

            # Add to batch
            session_content = "\n".join(session_parts)
            document_id = f"{item['sample_id']}_{session_key}"
            batch_contents.append({
                "content": session_content,
                "context": f"Conversation session between {speaker_a} and {speaker_b} (conversation {item['sample_id']} session {session_key})",
                "event_date": session_date,
                "document_id": document_id
            })

        return batch_contents

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
            return datetime.now(timezone.utc)


class QuestionAnswer(pydantic.BaseModel):
    """Answer format for LoComo questions."""
    answer: str
    reasoning: str


class LoComoAnswerGenerator(LLMAnswerGenerator):
    """LoComo-specific answer generator using configurable LLM provider."""

    def __init__(self):
        """Initialize with LLM configuration for memory operations."""
        self.llm_config = LLMConfig.for_memory()
        self.client = self.llm_config.client
        self.model = self.llm_config.model

    async def generate_answer(
        self,
        question: str,
        memories: List[Dict[str, Any]]
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer from retrieved memories using Groq.

        Returns:
            Tuple of (answer, reasoning, None)
            - None indicates to use the memories passed in
        """
        # Format context
        context_parts = []
        for result in memories:
            context_parts.append({"text": result.get("text"), "context": result.get("context"), "event_date": result.get("event_date")})

        context = json.dumps(context_parts)

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

    def __init__(self, memory: 'TemporalSemanticMemory', agent_id: str, thinking_budget: int = 500):
        """Initialize with memory instance and agent_id.

        Args:
            memory: TemporalSemanticMemory instance
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
        memories: List[Dict[str, Any]]
    ) -> Tuple[str, str, Optional[List[Dict[str, Any]]]]:
        """
        Generate answer using the integrated think API.

        The think API performs both search and answer generation in a single call,
        combining agent facts, world facts, and opinions to formulate a response.

        Args:
            question: Question to answer
            memories: Not used (empty list), as think does its own retrieval

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
            answer = result.get('text', '')

            # Extract memories from based_on
            based_on = result.get('based_on', {})
            world_facts = based_on.get('world', [])
            agent_facts = based_on.get('agent', [])
            opinion_facts = based_on.get('opinion', [])

            # Combine all facts into retrieved_memories
            retrieved_memories = []

            # Add world facts
            for fact in world_facts:
                retrieved_memories.append({
                    'id': fact.get('id'),
                    'text': fact.get('text'),
                    'context': fact.get('context'),
                    'event_date': fact.get('event_date'),
                    'score': fact.get('score', 0.0),
                    'fact_type': 'world'
                })

            # Add agent facts
            for fact in agent_facts:
                retrieved_memories.append({
                    'id': fact.get('id'),
                    'text': fact.get('text'),
                    'context': fact.get('context'),
                    'event_date': fact.get('event_date'),
                    'score': fact.get('score', 0.0),
                    'fact_type': 'agent'
                })

            # Add opinion facts
            for fact in opinion_facts:
                retrieved_memories.append({
                    'id': fact.get('id'),
                    'text': fact.get('text'),
                    'context': fact.get('context'),
                    'event_date': fact.get('event_date'),
                    'score': fact.get('score', 0.0),
                    'fact_type': 'opinion'
                })

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
    api_url: str = None
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
    """
    # Initialize components
    dataset = LoComoDataset()

    # Use remote API client if api_url is provided, otherwise use local memory
    if api_url:
        from memora.remote_client import RemoteMemoryClient
        memory = RemoteMemoryClient(base_url=api_url)
    else:
        memory = TemporalSemanticMemory(
            db_url=os.getenv("DATABASE_URL"),
            memory_llm_provider=os.getenv("MEMORY_LLM_PROVIDER", "groq"),
            memory_llm_api_key=os.getenv("MEMORY_LLM_API_KEY"),
            memory_llm_model=os.getenv("MEMORY_LLM_MODEL", "openai/gpt-oss-120b"),
            memory_llm_base_url=os.getenv("MEMORY_LLM_BASE_URL") or None,  # Use None to get provider defaults
        )
    await memory.initialize()

    if use_think:
        answer_generator = LoComoThinkAnswerGenerator(
            memory=memory,
            agent_id="locomo",
            thinking_budget=500
        )
        max_concurrent_questions = 4
        eval_semaphore_size = 4
    else:
        answer_generator = LoComoAnswerGenerator()
        # Reduced from 32 to 10 to match search semaphore limit
        # Prevents "too many connections" errors
        max_concurrent_questions = 10
        eval_semaphore_size = 8

    answer_evaluator = LLMAnswerEvaluator()

    # Create benchmark runner
    runner = BenchmarkRunner(
        dataset=dataset,
        answer_generator=answer_generator,
        answer_evaluator=answer_evaluator,
        memory=memory
    )

    # Run benchmark
    dataset_path = Path(__file__).parent / 'datasets' / 'locomo10.json'
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
        specific_item=conversation
    )

    # Display and save results
    runner.display_results(results)

    # Determine output filename based on mode
    suffix = "_think" if use_think else ""
    results_filename = f'benchmark_results{suffix}.json'

    # Merge with existing results if running a specific conversation
    merge_with_existing = conversation is not None
    runner.save_results(results, Path(__file__).parent / 'results' / results_filename, merge_with_existing=merge_with_existing)

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
    parser.add_argument('--api-url', type=str, default=None, help='Memora API URL (default: use local memory, example: http://localhost:8000)')

    args = parser.parse_args()

    results = asyncio.run(run_benchmark(
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        skip_ingestion=args.skip_ingestion,
        use_think=args.use_think,
        conversation=args.conversation,
        api_url=args.api_url
    ))
