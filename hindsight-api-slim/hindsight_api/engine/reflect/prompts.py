"""
System prompts for the reflect agent.

The reflect agent uses hierarchical retrieval:
1. search_mental_models - User-curated summaries (highest quality)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts as ground truth fallback
"""

import json
from datetime import datetime, timezone
from typing import Any

from ..prompt_utils import default_language_section, escape_for_prompt, output_language_directive
from .tokenization import count_prompt_tokens

# Fraction of max_context_tokens reserved for tool results in the final synthesis prompt.
# The remainder covers the system prompt, question, bank context, and output tokens.
_FINAL_PROMPT_CONTEXT_FRACTION = 0.8

_DEFAULT_ROLE = "You are a reflection agent that answers questions by reasoning over retrieved memories."
_DEFAULT_FINAL_ROLE = "You are a thoughtful assistant that synthesizes answers from retrieved memories."


def _current_utc_datetime() -> str:
    """Return the current UTC date and time for time-relative reflect reasoning.

    Minute precision (not seconds) so requests within the same minute share an
    identical prompt string — the finest granularity that still keeps prompt
    caching viable for bursty traffic.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _current_datetime_section() -> str:
    return f"## Current Date and Time\nThe current date and time is {_current_utc_datetime()}."


def _extract_directive_rules(directives: list[dict[str, Any]]) -> list[str]:
    """Extract directive rules as a list of strings."""
    rules = []
    for directive in directives:
        name = directive.get("name", "")
        content = directive.get("content", "")
        if content:
            rules.append(f"**{name}**: {content}" if name else content)
    return rules


def build_directives_section(directives: list[dict[str, Any]]) -> str:
    """Build the directives section for the system prompt.

    Directives are hard rules that MUST be followed in all responses.
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "## DIRECTIVES (MANDATORY)",
        "These are hard rules you MUST follow in ALL responses:",
        "",
    ]

    for rule in rules:
        parts.append(f"- {rule}")

    parts.extend(
        [
            "",
            "NEVER violate these directives, even if other context suggests otherwise.",
            "IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.",
            "",
        ]
    )
    return "\n".join(parts)


def build_directives_reminder(directives: list[dict[str, Any]]) -> str:
    """
    Build a reminder section for directives to place at the end of the prompt.

    Args:
        directives: List of directive mental models with observations
    """
    if not directives:
        return ""

    rules = _extract_directive_rules(directives)
    if not rules:
        return ""

    parts = [
        "",
        "## REMINDER: MANDATORY DIRECTIVES",
        "Before responding, ensure your answer complies with ALL of these directives:",
        "",
    ]

    for i, rule in enumerate(rules, 1):
        parts.append(f"{i}. {rule}")

    parts.append("")
    parts.append("Your response will be REJECTED if it violates any directive above.")
    parts.append("Do NOT include any commentary about how you handled directives - just follow them.")
    return "\n".join(parts)


# The reasoning-loop language rule. Emitted only when no output language is configured
# — see default_language_section(). Like _FINAL_LANGUAGE_RULE it defers to a directive
# "above", and nothing ever put one there for a configured language: the rule stayed,
# out-ranked the directive, and the done() answer came back in the question's language
# on every run that never fell through to forced synthesis (#3776, review).
_TOOLS_LANGUAGE_RULE = (
    "## LANGUAGE RULE (default - directives take precedence)\n"
    "- By default, detect the language of the user's question and respond in that SAME language.\n"
    "- If the question is in Chinese, respond in Chinese. If in Japanese, respond in Japanese.\n"
    "- IMPORTANT: The DIRECTIVES section above has HIGHER PRIORITY than this rule.\n"
    "  If a directive specifies a language (e.g. 'Always respond in French'), follow the directive."
)


def build_agent_user_prompt(query: str, llm_output_language: str | None = None) -> str:
    """The reasoning loop's opening user message: the question, closed by the directive.

    The ``done()`` answer is written by the tool-calling model, whose system prompt is
    :func:`build_system_prompt_for_tools`. For the same reason :func:`build_final_prompt`
    carries the directive instead of :func:`build_final_system_prompt`, it goes on the user
    message here rather than at the end of the system prompt: the question arrives after
    the system prompt and out-ranks it. Tool results still follow over later turns, so
    this is "last" for the question the model is answering, not for the whole
    conversation — the system prompt dropping its contradicting rule is what makes the
    directive uncontested (#3776).
    """
    return query + output_language_directive(llm_output_language)


def build_system_prompt_for_tools(
    bank_profile: dict[str, Any],
    context: str | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    include_observations: bool = True,
    budget: str | None = None,
    answer_as_document: bool = False,
    llm_output_language: str | None = None,
) -> str:
    """
    Build the system prompt for tool-calling reflect agent.

    The agent uses hierarchical retrieval:
    1. search_mental_models - User-curated summaries (try first, if available)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    The retrieval-strategy and workflow sections are built to match the tools
    actually exposed to the LLM — mentioning a tool the agent has disabled
    causes weaker LLMs to either hallucinate the call (rejected by the agent)
    or give up with "I cannot find any information…" (see #1724).

    Args:
        bank_profile: Bank profile with name and mission
        context: Optional additional context
        directives: Optional list of directive mental models to inject as hard rules
        has_mental_models: Whether the bank has any mental models (skip if not)
        include_observations: Whether search_observations is in the tool list.
        budget: Search depth budget - "low", "mid", or "high". Controls exploration thoroughness.
        answer_as_document: Whether done() takes a structured document instead of markdown.
        llm_output_language: Configured output language; drops the default language rule
            (the directive itself goes on the user message, see build_agent_user_prompt).
    """
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts = []

    # Anti-hallucination rule at the very top
    parts.extend(
        [
            "CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.",
            "",
        ]
    )

    # Inject directives after anti-hallucination rule
    if directives:
        parts.append(build_directives_section(directives))

    parts.extend(
        [
            mission.strip() if mission else _DEFAULT_ROLE,
            "",
            "Answer the user's question by reasoning over retrieved memories.",
            "",
        ]
    )

    # Mutually exclusive with the configured output language — the rule is dropped
    # outright rather than left to out-rank the directive (#3776). The directive itself
    # is not appended here: it rides on the user message, see build_agent_user_prompt().
    tools_language_section = default_language_section(_TOOLS_LANGUAGE_RULE, llm_output_language)
    if tools_language_section:
        parts.extend([tools_language_section.rstrip("\n"), ""])

    parts.extend(
        [
            "## CRITICAL RULES",
            "- ONLY use information from tool results - no external knowledge or guessing",
            "- You SHOULD synthesize, infer, and reason from the retrieved memories",
            "- You MUST search before saying you don't have information",
            "",
            "## How to Reason",
            "- If memories mention someone did an activity, you can infer they likely enjoyed it",
            "- Synthesize a coherent narrative from related memories",
            "- Be a thoughtful interpreter, not just a literal repeater",
            "- When the exact answer isn't stated, use what IS stated to give a best-effort answer AND surface any uncertainty — never invent confidence the data doesn't support.",
            "",
            "## Temporal Reasoning",
            "Every memory and observation carries temporal fields in the JSON tool result:",
            "- `mentioned_at` — when the user retained the fact (always set).",
            "- `occurred_start` / `occurred_end` — when the underlying event happened (optional, set for dated events).",
            "",
            "When facts about the SAME facet conflict — counts, statuses, ownership, location, presence, etc. — the fact with the LATEST `mentioned_at` is authoritative. Later statements SUPERSEDE earlier ones. Do NOT average, sum, or favor an explicitly-dated fact over a more recent one.",
            "",
            "Example: three count facts come back from recall:",
            "  - 'Team has 2 engineers' (mentioned_at=T1)",
            "  - 'Team now has 1 engineer' (mentioned_at=T2, occurred_start=2026-05-25)",
            "  - 'Team has 5 engineers' (mentioned_at=T3)",
            "with T1 < T2 < T3. The current size is 5, not 1. Then apply later events (e.g. someone leaving after T3) on top of that.",
            "",
            "For reconstructing a TIMELINE of events, order by `occurred_start` / `occurred_end` (when things happened), not `mentioned_at` (when they were retained).",
            "",
            "## Conflicts and Ambiguity",
            "Not every retrieval converges on a single answer. Distinguish two cases:",
            "",
            "- RESOLVABLE conflict — the temporal rule above (latest `mentioned_at` wins) cleanly picks a winner. Apply it and move on.",
            "- UNRESOLVABLE ambiguity — the data is internally inconsistent in a way the temporal rule does NOT settle. Examples: a recent aggregate (count, total) is incompatible with the individual entities you can enumerate; two equally-recent facts disagree and no later fact resolves them; events are described but their relative order is unclear; the user's own statements contradict each other and nothing later reconciles them.",
            "",
            "When the data is genuinely ambiguous: SAY SO in your answer. Name the conflicting facts. Explain why they can't be reconciled. Give a range or a best-effort interpretation with explicit uncertainty (e.g. 'between X and Y, depending on [unresolved condition]'; or 'the most recent statement says A, but B was stated earlier and the gap isn't accounted for in any later fact').",
            "",
            "An honest 'the data is inconsistent about X' beats a confident wrong answer. Do NOT pick a value arbitrarily, average conflicting values, or smooth over gaps in confident prose. Acknowledging ambiguity is a successful answer, not a failure mode.",
            "",
            "## Showing Your Reasoning",
            "For any answer that resolves a conflict between facts, applies events on top of a count or status, or settles an ambiguity — show your work in the answer text so a reader can audit it.",
            "",
            "Walk through these steps explicitly:",
            "1. **List the relevant facts in `mentioned_at` order (oldest → newest)**, each with the value it asserts. Use a short bulleted list.",
            "2. **Identify the authoritative fact** under the temporal rule (latest `mentioned_at` for the contested facet). Write its date down.",
            "3. **List candidate events to apply on top** — anything that changes the count, status, or state being asked about. Write each event's date down next to it.",
            "4. **Sanity-check each candidate event against the authoritative date** — for EVERY event from step 3, write a one-line check in the form `<event> (<event_date>) vs authoritative (<authoritative_date>) → BEFORE/AFTER → KEEP/DROP`. If the event is BEFORE or EQUAL to the authoritative date, DROP it: it is already reflected in the authoritative fact, and applying it again is double-counting. This is the single most common mistake — do not skip this step even if you feel confident.",
            "5. **Show the arithmetic or derivation explicitly** using only the KEEP events from step 4 — e.g. 'authoritative count = 5 (at 2025-02-12); kept events: Shadow died (2025-03-12, AFTER); 5 − 1 = 4'.",
            "6. If step 2 or 3 cannot be done cleanly (no clear winner, overlapping timestamps, unclear event order), STOP and surface this as an UNRESOLVABLE ambiguity per the section above — do not fabricate a derivation.",
            "",
            "For simple factual lookups that don't involve conflict or arithmetic, you can answer directly without this scaffolding.",
            "",
            "## HIERARCHICAL RETRIEVAL STRATEGY",
            "",
        ]
    )

    # Assemble the retrieval-level blocks for whatever tools are exposed.
    # MM and Observations bodies are unconditional; recall's fallback wording
    # adapts to which upstream tools precede it (telling the LLM to fall back
    # to a tool that isn't in its list is the bug at the root of #1724).
    levels: list[tuple[str, list[str]]] = []
    if has_mental_models:
        levels.append(
            (
                "MENTAL MODELS (search_mental_models)",
                [
                    "- User-curated summaries about specific topics",
                    "- HIGHEST quality - manually created and maintained",
                    "- If a relevant mental model exists and is FRESH, it may fully answer the question",
                    "- Check `is_stale` field - if stale, also verify with lower levels",
                ],
            )
        )
    if include_observations:
        levels.append(
            (
                "OBSERVATIONS (search_observations)",
                [
                    "- Auto-consolidated knowledge from memories",
                    "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                    "- Good for understanding patterns and summaries",
                ],
            )
        )
    recall_body = ["- Individual memories (world facts and experiences)"]
    if has_mental_models and include_observations:
        recall_body.extend(
            [
                "- Use when: no mental models/observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_mental_models and search_observations both return 0 results, you MUST call recall() before giving up",
                "- This is the source of truth that other levels are built from",
                "",
                "**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).",
                "",
            ]
        )
    elif has_mental_models:
        recall_body.extend(
            [
                "- Use when: no mental model exists, it's stale, or you need specific details",
                "- MANDATORY: If search_mental_models returns 0 results, you MUST call recall() before giving up",
                "- This is the source of truth that mental models are built from",
            ]
        )
    elif include_observations:
        recall_body.extend(
            [
                "- Use when: no observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_observations returns 0 results or count=0, you MUST call recall() before giving up",
                "- This is the source of truth that observations are built from",
                "",
                "**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).",
                "",
            ]
        )
    else:
        recall_body.extend(
            [
                "- MANDATORY: Call recall() to gather facts before giving up",
                "- This is the source of truth.",
            ]
        )
    levels.append(("RAW FACTS (recall) - Ground Truth", recall_body))

    # Position-dependent suffix for upstream tools; recall already carries its
    # fixed "- Ground Truth" suffix in the header text.
    suffixes = [""] * len(levels)
    if len(levels) >= 2:
        suffixes[0] = " - Try First"
    if len(levels) == 3:
        suffixes[1] = " - Second Priority"

    if len(levels) == 1:
        parts.append("You have access to ONE level of knowledge:")
    else:
        word = "TWO" if len(levels) == 2 else "THREE"
        parts.append(f"You have access to {word} levels of knowledge. Use them in this order:")
    parts.append("")
    for idx, ((header, body), suffix) in enumerate(zip(levels, suffixes), 1):
        parts.append(f"### {idx}. {header}{suffix}")
        parts.extend(body)
        parts.append("")

    parts.extend(
        [
            "## Query Strategy",
            "recall() uses semantic search. NEVER just echo the user's question - decompose it into targeted searches:",
            "",
            "BAD: User asks 'recurring lesson themes between students' → recall('recurring lesson themes between students')",
            "GOOD: Break it down into component searches:",
            "  1. recall('lessons') - find all lesson-related memories",
            "  2. recall('teaching sessions') - alternative phrasing",
            "  3. recall('student progress') - find student-related memories",
            "",
            "Think: What ENTITIES and CONCEPTS does this question involve? Search for each separately.",
            "",
        ]
    )

    # Add budget guidance
    if budget:
        budget_lower = budget.lower()
        if budget_lower == "low":
            parts.extend(
                [
                    "## RESEARCH DEPTH: SHALLOW (Quick Response)",
                    "- Prioritize speed over completeness",
                    "- If mental models or observations provide a reasonable answer, stop there",
                    "- Only dig deeper if the initial results are clearly insufficient",
                    "- Prefer a quick overview rather than exhaustive details",
                    "- Answer promptly with available information",
                    "",
                ]
            )
        elif budget_lower == "mid":
            parts.extend(
                [
                    "## RESEARCH DEPTH: MODERATE (Balanced)",
                    "- Balance thoroughness with efficiency",
                    "- Check multiple sources when the question warrants it",
                    "- Verify stale data if it's central to the answer",
                    "- Don't over-explore, but ensure reasonable coverage",
                    "",
                ]
            )
        elif budget_lower == "high":
            parts.extend(
                [
                    "## RESEARCH DEPTH: DEEP (Thorough Exploration)",
                    "- Explore comprehensively before answering",
                    "- Search across all available knowledge levels",
                    "- Use multiple query variations to ensure coverage",
                    "- Verify information across different retrieval levels",
                    "- Use expand() to get full context on important memories",
                    "- Take time to synthesize a complete, well-researched answer",
                    "",
                ]
            )

    parts.append("## Workflow")

    steps: list[str] = []
    if has_mental_models:
        steps.append("First, try search_mental_models() - check if a curated summary exists")
    if include_observations:
        if has_mental_models:
            steps.append("If no mental model or it's stale, try search_observations() for consolidated knowledge")
        else:
            steps.append("First, try search_observations() - check for consolidated knowledge")
    # Recall step phrasing varies with whichever upstream tool(s) precede it.
    if include_observations:
        steps.append(
            "If observations are stale OR you need specific details, use recall() for raw facts"
            if has_mental_models
            else "If search_observations returns 0 results OR observations are stale, you MUST call recall() for raw facts"
        )
    elif has_mental_models:
        steps.append("If no mental model or it's stale, use recall() for raw facts")
    else:
        steps.append("Call recall() to gather raw facts")
    steps.append("Use expand() if you need more context on specific memories")
    steps.append("When ready, call done() with your answer and supporting IDs")
    parts.extend(f"{idx}. {step}" for idx, step in enumerate(steps, 1))

    common_output_rules = [
        "- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text",
        "- Put IDs ONLY in the memory_ids/mental_model_ids/observation_ids arrays, not in the answer",
        "- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.",
    ]
    if answer_as_document:
        # The document is stored as structure and the markdown is rendered from
        # it, so asking for "a markdown answer" here would be asking for the one
        # thing the caller does not want (see tools_schema's document tool).
        parts.extend(
            [
                "",
                "## Output Format: Structured Document",
                "Call done() with a 'document' field. Do NOT write a markdown document — "
                "state its structure and the markdown is generated from it.",
                # Name the wrapper and show it. Every field of a *section* was spelled out
                # here — heading, level, blocks — and the array holding them never was, so
                # the prose described a section while the tool schema described a document
                # containing sections. Models resolved that disagreement in favour of the
                # prose and emitted the bare section, which parsed to zero sections, an
                # empty render, and a failed refresh. A shape stated in one place and shown
                # in another is a shape that gets filled correctly.
                "- 'document' holds a 'sections' array — one entry per section, in order. "
                'Shape: {"sections": [{"heading": "Overview", "level": 2, "blocks": '
                '["First paragraph.", "- a list item\\n- another"]}]}',
                "- Even a one-section document uses the 'sections' array; never emit a bare section",
                "- Each section carries its heading text (no '#') and a level; the heading is NOT a block",
                "- 'blocks' holds the section's content, ONE block per paragraph, list, table or code fence",
                "- Never put two paragraphs in one block, and never put a heading inside a block",
                "- Inside a block, write list items and table rows on their own lines as usual",
                # Stating the structure is a more clerical task than writing prose, and a
                # model doing clerical work gets terse: measured against the markdown
                # answer it replaced, the first version of this produced noticeably
                # shorter documents that judges liked less. Say plainly that the
                # structure is the shape, not a budget.
                "- The structure is how the answer is laid out, NOT a reason to say less: give each "
                "block the same specifics, numbers, names and examples you would write in prose",
                "- Prefer a section with several blocks over one block that summarises them away",
                *common_output_rules,
            ]
        )
    else:
        parts.extend(
            [
                "",
                "## Output Format: Well-Formatted Markdown Answer",
                "Call done() with a well-formatted markdown 'answer' field.",
                "- USE markdown formatting for structure (headers, lists, bold, italic, code blocks, tables, etc.)",
                "- CRITICAL: Add blank lines before and after block elements (tables, code blocks, lists)",
                "- Format for clarity and readability with proper spacing and hierarchy",
                *common_output_rules,
            ]
        )

    # Volatile "now" reference goes here — after all the static instructions and
    # right before the bank-specific/custom data. Everything above is identical
    # across banks and requests, so it stays a cacheable prefix; only this
    # timestamp and the custom tail below fall outside the cache.
    parts.append("")
    parts.append(_current_datetime_section())

    parts.append("")
    parts.append(f"## Memory Bank: {name}")

    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits
    disposition = bank_profile.get("disposition", {})
    if disposition:
        traits = []
        if "skepticism" in disposition:
            traits.append(f"skepticism={disposition['skepticism']}")
        if "literalism" in disposition:
            traits.append(f"literalism={disposition['literalism']}")
        if "empathy" in disposition:
            traits.append(f"empathy={disposition['empathy']}")
        if traits:
            parts.append(f"Disposition: {', '.join(traits)}")

    if context:
        parts.append(f"\n## Additional Context\n{context}")

    # Add directive reminder at the END for recency effect
    if directives:
        parts.append(build_directives_reminder(directives))

    return "\n".join(parts)


#: Result-list keys a tool output can carry; an over-budget block is split on
#: these entry boundaries so no retrieved evidence is dropped.
_SPLITTABLE_RESULT_KEYS = ("observations", "memories", "results")

#: Above this many synthesis chunks the retrieval volume is pathological
#: (each chunk is ~0.8 * max_context_tokens); we still process everything,
#: but loudly, so the real cause (an unbounded tool result) gets looked at.
_SPLIT_SYNTHESIS_WARN_CHUNKS = 4

#: Floor for the per-chunk budget during splitting. A tiny configured
#: ``max_context_tokens`` (tests use 1) would otherwise shred the history into
#: one chunk per result entry — an LLM call per fact. A ~1k-token prompt is
#: safe for any real model, so the floor caps fan-out without dropping data.
_MIN_SPLIT_CHUNK_TOKENS = 1024

_FINAL_INSTRUCTIONS = (
    "Provide a thoughtful answer by synthesizing and reasoning from the retrieved data above. "
    "You can make reasonable inferences from the memories, but don't completely fabricate information. "
    "If the exact answer isn't stated, use what IS stated to give the best possible answer. "
    "Only say 'I don't have information' if the retrieved data is truly unrelated to the question.\n\n"
    "IMPORTANT: Output ONLY the final answer. Do NOT include meta-commentary like "
    '"I\'ll search..." or "Let me analyze...". Do NOT explain your reasoning process. '
    "Just provide the direct synthesized answer."
)


def _render_history_block(entry: dict) -> str:
    """Render one context-history entry as a fenced JSON block."""
    tool = entry["tool"]
    output = entry["output"]
    try:
        output_str = json.dumps(output, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        output_str = str(output)
    return f"\n### From {tool}:\n```json\n{output_str}\n```"


def _cut_entry_to_budget(entry: dict, token_budget: int) -> dict:
    """Token-bound one indivisible over-budget entry by cutting its serialized text.

    Only reachable when a single result entry (or a list-less output like a
    document expand) alone exceeds the whole per-chunk budget — the one case
    where "split, don't drop" cannot be honored without exceeding the model's
    window. The cut text is wrapped back into an output dict so the entry
    renders like any other block.
    """
    output = entry["output"]
    try:
        output_str = json.dumps(output, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        output_str = str(output)
    tokens = count_prompt_tokens(output_str)
    while output_str and tokens > token_budget:
        # Proportional shrink with a safety margin; the loop guards against the
        # estimate landing high, and always makes progress.
        keep = min(len(output_str) - 1, max(1, int(len(output_str) * token_budget / tokens * 0.95)))
        output_str = output_str[:keep]
        tokens = count_prompt_tokens(output_str)
    return {**entry, "output": {"truncated": True, "content": output_str}}


def split_context_history(context_history: list[dict], max_context_tokens: int) -> list[list[dict]]:
    """Partition tool-result history into chunks that each fit the prompt budget.

    Greedy chronological packing: blocks keep their order, and a chunk closes
    when the next block would push its rendered size past the budget. A single
    block bigger than the whole budget is split on result-entry boundaries
    (``observations``/``memories``/``results``) into synthetic partial blocks,
    so evidence is split across chunks rather than dropped — the failure mode
    of the old ``break`` was answering from nothing while citing everything
    (#3122). Only an *indivisible* over-budget entry gets token-cut.

    Returns at least one chunk when history is non-empty; every original
    result entry appears in exactly one chunk.
    """
    budget = max(_MIN_SPLIT_CHUNK_TOKENS, int(max_context_tokens * _FINAL_PROMPT_CONTEXT_FRACTION))
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    def _close_current() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(current)
            current = []
            current_tokens = 0

    def _append_block(entry: dict, tokens: int) -> None:
        nonlocal current_tokens
        if current and current_tokens + tokens > budget:
            _close_current()
        current.append(entry)
        current_tokens += tokens

    for entry in context_history:
        tokens = count_prompt_tokens(_render_history_block(entry))
        if tokens <= budget:
            _append_block(entry, tokens)
            continue

        # Over-budget block: split it on result-entry boundaries.
        output = entry["output"]
        split_key = next(
            (
                k
                for k in _SPLITTABLE_RESULT_KEYS
                if isinstance(output, dict) and isinstance(output.get(k), list) and output.get(k)
            ),
            None,
        )
        if split_key is None:
            cut = _cut_entry_to_budget(entry, budget)
            _append_block(cut, count_prompt_tokens(_render_history_block(cut)))
            continue

        items = output[split_key]
        piece: list = []
        for item in items:
            candidate = {**entry, "output": {**output, split_key: piece + [item]}}
            if piece and count_prompt_tokens(_render_history_block(candidate)) > budget:
                partial = {**entry, "output": {**output, split_key: piece}}
                _append_block(partial, count_prompt_tokens(_render_history_block(partial)))
                piece = []
                candidate = {**entry, "output": {**output, split_key: [item]}}
            single_tokens = count_prompt_tokens(_render_history_block(candidate))
            if not piece and single_tokens > budget:
                cut = _cut_entry_to_budget({**entry, "output": {**output, split_key: [item]}}, budget)
                _append_block(cut, count_prompt_tokens(_render_history_block(cut)))
            else:
                piece.append(item)
        if piece:
            partial = {**entry, "output": {**output, split_key: piece}}
            _append_block(partial, count_prompt_tokens(_render_history_block(partial)))

    _close_current()
    return chunks


def _bank_identity_section(bank_profile: dict, additional_context: str | None) -> list[str]:
    """The shared bank-identity/disposition/context head of a synthesis prompt."""
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts = [f"## Memory Bank Context\nName: {name}"]
    if mission:
        parts.append(f"Mission: {mission}")

    disposition = bank_profile.get("disposition", {})
    if disposition:
        traits = []
        if "skepticism" in disposition:
            traits.append(f"skepticism={disposition['skepticism']}")
        if "literalism" in disposition:
            traits.append(f"literalism={disposition['literalism']}")
        if "empathy" in disposition:
            traits.append(f"empathy={disposition['empathy']}")
        if traits:
            parts.append(f"Disposition: {', '.join(traits)}")

    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")
    return parts


def _length_directive(max_tokens: int | None) -> str | None:
    """Soft visible-length directive for a synthesis prompt, or None.

    ``max_tokens`` is the desired *visible* length of the answer (e.g. a mental
    model page's ``max_tokens``). It is communicated as a prompt directive
    rather than enforced by truncating the provider call: on thinking models the
    provider budget is consumed by reasoning tokens, so a hard cap cuts the page
    off mid-word (#3365). The hard length guarantee is the post-hoc rewrite in
    the agent; this directive just steers the model toward the target so the
    rewrite rarely has to fire.
    """
    if max_tokens is None:
        return None
    return (
        "\n## Length\n"
        f"Aim for a complete, self-contained answer of approximately {max_tokens} tokens. "
        "Finishing cleanly matters more than length: end on a complete sentence and NEVER stop "
        "mid-word, mid-list, or mid-code-fence. If you near the budget, wrap up gracefully rather "
        "than cutting off."
    )


def build_final_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
    max_context_tokens: int = 100_000,
    max_tokens: int | None = None,
    llm_output_language: str | None = None,
) -> str:
    """Build the final prompt when forcing a text response (no tools).

    ``max_tokens`` is the soft visible-length target (see ``_length_directive``).

    ``llm_output_language`` closes the prompt with :func:`output_language_directive`.
    It rides on the USER message, not the system prompt, because it has to be the last
    thing the model reads — see :func:`build_final_system_prompt` for the measurement.

    Callers overflow-proof this via ``split_context_history``: when the whole
    history fits one chunk this renders it directly, and the per-block budget
    walk below never trims. (The walk is kept as a defensive bound for direct
    callers that skip splitting.)
    """
    parts = _bank_identity_section(bank_profile, additional_context)

    # Tool call history — include as many entries as fit within the token budget,
    # preferring the most recent calls (they tend to be the most targeted).
    if context_history:
        parts.append("\n## Retrieved Data (synthesize and reason from this data)")
        token_budget = int(max_context_tokens * _FINAL_PROMPT_CONTEXT_FRACTION)
        # Render entries newest-first, then reverse so the prompt reads chronologically.
        rendered: list[str] = []
        truncated = False
        for entry in reversed(context_history):
            block = _render_history_block(entry)
            block_tokens = count_prompt_tokens(block)
            if block_tokens > token_budget:
                truncated = True
                break
            rendered.append(block)
            token_budget -= block_tokens
        for block in reversed(rendered):
            parts.append(block)
        if truncated:
            parts.append("\n*Note: Some earlier tool results were omitted to stay within the context window.*")
    else:
        parts.append("\n## Retrieved Data\nNo data was retrieved.")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Final instructions
    parts.append("\n## Instructions\n" + _FINAL_INSTRUCTIONS)

    length_directive = _length_directive(max_tokens)
    if length_directive is not None:
        parts.append(length_directive)

    return "\n".join(parts) + output_language_directive(llm_output_language)


#: System prompt for the intermediate (map) calls of split synthesis. They do
#: NOT answer the question — they compress one chunk of retrieved data into
#: dated, cited claims that the reduce call can reason over. Dates and ids are
#: mandatory because conflicting facts can land in different chunks: only the
#: reduce call sees every chunk's claims, and it needs each claim's
#: ``mentioned_at`` to apply the latest-statement-wins supersession rule.
CLAIMS_SYSTEM_PROMPT = (
    "You extract evidence from retrieved memory data. You MUST ONLY use information "
    "from the provided data. NEVER make up names, people, events, or entities.\n\n"
    "Output a markdown bulleted list of factual claims relevant to the question. For EVERY claim:\n"
    "- state the fact in one sentence, in the same language as the question;\n"
    "- append its provenance in parentheses, exactly: "
    "(mentioned_at: <ISO date or unknown>; occurred: <ISO date/range or unknown>; memory_ids: <comma-separated ids>)\n\n"
    "Rules:\n"
    "- Be exhaustive over RELEVANT evidence; skip clearly irrelevant entries.\n"
    "- Do NOT synthesize, conclude, resolve conflicts, or answer the question — report conflicting "
    "claims as separate bullets with their dates; a later pass reconciles them.\n"
    "- Copy memory ids exactly as they appear in the data.\n"
    "- If nothing in the data is relevant, output exactly: (no relevant evidence)"
)


def build_chunk_claims_prompt(query: str, chunk: list[dict]) -> str:
    """Build the user prompt for one intermediate (map) call of split synthesis."""
    parts = ["## Retrieved Data (extract relevant claims from this data)"]
    for entry in chunk:
        parts.append(_render_history_block(entry))
    parts.append(f"\n## Question\n{query}")
    parts.append(
        "\n## Instructions\n"
        "List every claim in the retrieved data relevant to the question, one bullet per claim, "
        "each with its (mentioned_at: ...; occurred: ...; memory_ids: ...) provenance. "
        "Do not answer the question."
    )
    return "\n".join(parts)


def build_reduce_prompt(
    query: str,
    claim_sections: list[str],
    bank_profile: dict,
    additional_context: str | None = None,
    max_tokens: int | None = None,
    llm_output_language: str | None = None,
) -> str:
    """Build the final prompt that synthesizes the answer from per-chunk claims.

    The retrieved data exceeded the context budget, so it was split into chunks
    and each chunk was compressed to dated, cited claims by a parallel LLM call.
    This prompt hands ALL the claim sets to one model. Conflicting facts may sit
    in different sections — that is why the claims carry ``mentioned_at``: the
    supersession rule (latest statement wins) must be applied across sections,
    not within one.

    Carries the output-language directive for the same reason, and in the same place, as
    :func:`build_final_prompt` — this is the other prompt that writes a user-visible answer.
    """
    parts = _bank_identity_section(bank_profile, additional_context)

    parts.append(
        "\n## Retrieved Evidence (synthesize and reason from these claims)\n"
        "The retrieved data was processed in parallel passes; each section below holds one pass's "
        "extracted claims with provenance dates and memory ids. Treat the sections as ONE evidence "
        "pool: related and conflicting claims may appear in different sections."
    )
    for i, section in enumerate(claim_sections, 1):
        parts.append(f"\n### Evidence pass {i}:\n{section}")

    parts.append(f"\n## Question\n{query}")

    parts.append(
        "\n## Instructions\n"
        "When claims about the same fact conflict, the claim with the LATEST mentioned_at date is "
        "authoritative — later statements supersede earlier ones, regardless of which section they "
        "appear in. If equally-recent claims disagree and nothing resolves them, say so explicitly "
        "rather than picking one.\n\n" + _FINAL_INSTRUCTIONS
    )

    length_directive = _length_directive(max_tokens)
    if length_directive is not None:
        parts.append(length_directive)

    return "\n".join(parts) + output_language_directive(llm_output_language)


_FINAL_SYSTEM_PROMPT_BASE = """CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.

{role_section}

Your approach:
- Reason over the retrieved memories to answer the question
- Make reasonable inferences when the exact answer isn't explicitly stated
- Connect related memories to form a complete picture
- Be helpful - if you have related information, use it to give the best possible answer
- ONLY use information from tool results - no external knowledge or guessing

Only say "I don't have information" if the retrieved data is truly unrelated to the question.

FORMATTING: Use proper markdown formatting in your answer:
- Headers (##, ###) for sections
- Lists (bullet or numbered) for enumerations
- Bold/italic for emphasis
- Tables with proper syntax (ensure blank line before and after)
- Code blocks where appropriate
- CRITICAL: Always add blank lines before and after block elements (tables, code blocks, lists)
- Proper spacing between sections

CRITICAL: Output ONLY the final synthesized answer. Do NOT include:
- Meta-commentary about what you're doing ("I'll search...", "Let me analyze...")
- Explanations of your reasoning process
- Descriptions of your approach
Just provide the direct answer with proper markdown formatting.

CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer to search again, suggest alternatives, or end with anything like "Would you like me to..." or "Let me know if...". The user cannot reply. Your answer must be complete and self-contained."""


# The final synthesis is a SEPARATE LLM call with its own system prompt — the
# agent/reasoning system prompt (which carries directives and the language rule)
# is NOT in scope here. So this default language rule, and the directives, must
# be repeated for the answer-writing model. Without it, weaker models drift to
# English even when the question/facts are in another language or a directive
# demands a specific one (the cause of flaky multilingual reflect tests).
#
# Emitted only when no output language is configured — see default_language_section().
# The escape hatch below defers to a directive "above", but nothing ever put one there:
# with a configured language the model saw this rule, phrased more forcefully, and
# answered in the question's language instead. Retain and consolidation drop their rule
# for exactly this reason (#3776); reflect does too.
_FINAL_LANGUAGE_RULE = (
    "## LANGUAGE\n"
    "- Respond in the SAME language as the user's question "
    "(e.g. a question in Chinese gets a Chinese answer; Japanese → Japanese).\n"
    "- If a directive above specifies a response language, follow the directive — "
    "it takes precedence over this default."
)


def build_final_system_prompt(
    mission: str | None = None,
    llm_output_language: str | None = None,
    directives: list[dict[str, Any]] | None = None,
) -> str:
    """Build the final synthesis system prompt, using mission as role when set.

    ``directives`` are re-injected here (they live in the agent/reasoning prompt,
    but the final answer is a separate call) so output-constraining rules — most
    visibly response language — are honoured by the model that actually writes
    the answer.

    ``llm_output_language`` drops the answer-in-the-question's-language default rather
    than leaving it to contradict the directive. It does NOT add the directive here.
    That is deliberate and measured: this prompt is the system message, and the question
    and the retrieved data both arrive *after* it in the user message. Appending the
    directive at the end of this string still leaves it out-ranked by everything the
    model reads next — a Chinese question with the output language set to English came
    back in Chinese 12 times out of 12 on gemini-2.5-flash-lite, and 0/5 on
    gemini-2.5-flash, with no contradicting rule anywhere in the prompt. Moving the same
    directive to the end of the user message is 12/12 and 5/5 English. So
    :func:`build_final_prompt` and :func:`build_reduce_prompt` carry it instead, where it
    is genuinely the last thing the model reads (#3776).
    """
    role_section = escape_for_prompt(mission.strip()) if mission else _DEFAULT_FINAL_ROLE

    parts = [build_directives_section(directives) if directives else ""]
    parts.append(_FINAL_SYSTEM_PROMPT_BASE.format(role_section=role_section))
    parts.append(default_language_section(_FINAL_LANGUAGE_RULE, llm_output_language))
    parts.append(build_directives_reminder(directives) if directives else "")
    # Volatile "now" reference last, so the static/per-bank instructions above
    # remain a cacheable prefix and only this timestamp falls outside the cache.
    parts.append(_current_datetime_section())

    return "\n\n".join(p.strip() for p in parts if p.strip())


# Backward-compatible constant for non-identity missions
FINAL_SYSTEM_PROMPT = build_final_system_prompt()


STRUCTURED_DELTA_SYSTEM_PROMPT = """You are integrating *new information* into an existing structured document.

You will be given:
1. TOPIC — the question this document answers. Content that does not help
   answer this question is OFF-TOPIC and should be removed.
2. CURRENT DOCUMENT (JSON) — the existing structured mental model. It is a list
   of sections; each section has a stable ``id``, a ``heading``, a ``level``
   (1..6) and an ordered list of ``blocks``. Each block has a stable ``id`` and
   a ``text`` field holding one markdown fragment — a paragraph, a list, a
   table, or a fenced code block.
3. NEW INFORMATION SYNTHESIS (markdown) — a synthesis showing how the new facts
   relate to the document's topic. Use it to understand context and relevance,
   but do NOT copy its formatting or wording wholesale.
4. SUPPORTING FACTS — observations and facts created since the last refresh.
   These are genuinely new — they were NOT available when the current document
   was written.

Your task: output a JSON object ``{"operations": [...]}``. Applied to CURRENT
DOCUMENT, the operations must produce a document that best answers the TOPIC
by integrating the new facts.

RULES
- These facts are NEW since the last refresh. The existing document already
  captures all prior information from earlier refreshes. Your job is to
  integrate the new facts into the existing document.
- **Preserve existing content**: The current document was built from prior facts
  that you cannot see. Do NOT remove or replace existing sections just because
  the new facts do not reference them. Only remove content when the new facts
  explicitly contradict or supersede it.
- **Merge overlapping topics**: When new facts cover topics that overlap with
  existing sections, merge the new information INTO the existing section
  rather than creating duplicates. When new facts provide more specific or
  authoritative guidance on a topic already covered generically, update the
  existing content to reflect the more specific guidance.
- **Preserve examples**: Concrete examples, before/after pairs, sample sentences,
  and illustrative ✅/❌ comparisons are MORE valuable than abstract rules.
  When facts contain examples, include them. Never drop an example to make
  room for an abstract restatement of the same point.
- Operations target sections by ``section_id`` and blocks by ``block_id``. Both
  are the ``id`` values printed in CURRENT DOCUMENT — **copy them exactly**,
  never invent one, never use a heading in place of an id, and never guess an
  id for a block you cannot see. An operation naming an id that does not exist
  is dropped, and its content never reaches the document.
- **Add** new content with ``append_block``, ``insert_block``, or ``add_section``
  when facts introduce information not yet covered. Prefer extending an
  existing section over creating a new one.
- **Update** existing content with ``replace_block`` or ``replace_section_blocks``
  when new facts provide corrections, updates, or more specific information
  about topics already in the document.
- **Remove** content with ``remove_block`` or ``remove_section`` ONLY when
  the new facts explicitly contradict or supersede it.
- Prefer the *smallest* operation that expresses the change: appending or
  replacing one block leaves every other block byte-identical, while
  ``replace_section_blocks`` makes you retype the whole section and risks
  losing detail you did not intend to change.
- NEVER emit operations whose only effect is to reword unchanged content.
- NEVER emit operations to "normalize" formatting (numbered → bulleted, casing
  changes, paragraph → list, etc).
- Every operation MUST be justifiable by a specific fact in SUPPORTING FACTS.
- Output ``{"operations": []}`` only if the new facts are already reflected
  in the document (e.g., from a concurrent update).

ALLOWED OPERATIONS (each line shows the JSON shape)
- ``{"op": "append_block", "section_id": "...", "text": "..."}``
- ``{"op": "insert_block", "section_id": "...", "after_block_id": "...", "text": "..."}``
- ``{"op": "replace_block", "section_id": "...", "block_id": "...", "text": "..."}``
- ``{"op": "remove_block", "section_id": "...", "block_id": "..."}``
- ``{"op": "add_section", "heading": "...", "level": 2, "blocks": ["...", "..."], "after_section_id": "..."}``
- ``{"op": "remove_section", "section_id": "..."}``
- ``{"op": "replace_section_blocks", "section_id": "...", "blocks": ["...", "..."]}``
- ``{"op": "rename_section", "section_id": "...", "new_heading": "..."}``

BLOCK TEXT RULES
- Every ``text`` (and every entry of ``blocks``) is ONE markdown fragment:
  a single paragraph, a single list, a single table, or a single fenced code
  block. Do not put two paragraphs in one block — emit two operations, or pass
  two entries in ``blocks``.
- Write real markdown inside it, with real line breaks encoded as ``\\n``:
  a list is ``"- one\\n- two"``, a table is
  ``"| col | col |\\n| --- | --- |\\n| a | b |"``. A table whose rows are not
  separated by ``\\n`` is not a table.
- To add a row to an existing table, or an item to an existing list, use
  ``replace_block`` on that block and re-emit it *with* the addition. A lone
  table row or bullet appended as its own block is a separate fragment, and a
  table row on its own is not a table.
- ``insert_block`` places the new block directly after ``after_block_id``; use
  ``"after_block_id": null`` to place it first in the section.

JSON STRING RULES (critical)
- Every string must be valid JSON: escape ``"`` as ``\\"``, backslashes as
  ``\\\\``, and every line break as ``\\n``. Never put a raw newline inside a
  JSON string.
- Return ONLY a single JSON object, with no prose before or after, no markdown
  code fences, no commentary. The object must have exactly one top-level key,
  ``operations``, whose value is an array of operation objects (empty array
  when nothing changes).
- Do not append extra ``]`` or ``}`` after the closing ``}`` of the root object.

Examples
- No changes needed → ``{"operations": []}``
- Add one bullet list to an existing "Members" section →
  ``{"operations": [{"op": "append_block", "section_id": "members",
  "text": "- Carol — junior engineer"}]}``
- Replace a paragraph that new facts have corrected →
  ``{"operations": [{"op": "replace_block", "section_id": "overview",
  "block_id": "b3f9a1c2", "text": "Updated summary."}]}``
- Remove an obsolete block →
  ``{"operations": [{"op": "remove_block", "section_id": "status",
  "block_id": "b17c4d80"}]}``"""

_STRUCTURED_DELTA_DEFAULT_MAX_INPUT_TOKENS = 24_000


def _truncate_prompt_text(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens under the configured encoding."""
    if max_tokens <= 0:
        return ""
    from ..token_encoding import truncate_to_tokens

    return truncate_to_tokens(text, max_tokens).text


def _fit_structured_delta_prompt_parts(
    *,
    source_query: str,
    current_document_json: str,
    candidate_markdown: str,
    facts_block: str,
    budget_hint: str,
    task_footer: str,
    max_input_tokens: int,
) -> tuple[str, str, str, bool]:
    """Shrink large prompt sections to fit within max_input_tokens (tokenizer estimate)."""
    from .tokenization import count_prompt_tokens

    fixed = (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT (apply ops to this; copy section and block ids from it verbatim)\n"
        f"```json\n\n```\n\n"
        f"## NEW INFORMATION SYNTHESIS (context for how new facts relate to the topic)\n"
        f"```markdown\n\n```\n\n"
        f"## SUPPORTING FACTS (new since last refresh — integrate these)\n"
        f"{budget_hint}\n\n"
        f"{task_footer}"
    )
    facts_header = "## SUPPORTING FACTS (new since last refresh — integrate these)\n"
    facts_prefix_tokens = count_prompt_tokens(facts_header)
    reserved_facts = min(4096, max(512, max_input_tokens // 8))
    doc_budget = max(1024, (max_input_tokens - count_prompt_tokens(fixed) - reserved_facts) * 55 // 100)
    cand_budget = max(512, (max_input_tokens - count_prompt_tokens(fixed) - reserved_facts) * 30 // 100)
    facts_budget = max(256, reserved_facts - facts_prefix_tokens)
    doc_json = _truncate_prompt_text(current_document_json, doc_budget)
    candidate = _truncate_prompt_text(candidate_markdown, cand_budget)
    facts_body = _truncate_prompt_text(facts_block, facts_budget)
    truncated = doc_json != current_document_json or candidate != candidate_markdown or facts_body != facts_block
    return doc_json, candidate, facts_body, truncated


def build_structured_delta_prompt(
    *,
    current_document_json: str,
    candidate_markdown: str,
    supporting_facts: list[dict[str, Any]],
    source_query: str,
    max_output_tokens: int | None = None,
    max_input_tokens: int | None = None,
    document_tokens: int | None = None,
    document_budget: int | None = None,
) -> str:
    """Build the user prompt for a structured-delta mental model refresh.

    The LLM's job is to emit operations against ``current_document_json``;
    the surrounding ``candidate_markdown`` and ``supporting_facts`` are
    references for *what new information exists*, not templates to mimic.

    ``max_output_tokens`` is surfaced in the prompt so the model can keep its
    op list within the provider's response cap. The actual cap is enforced by
    the caller; this is just an advisory anchor — without it the model often
    returns op lists whose JSON gets truncated mid-string.
    """
    fact_lines: list[str] = []
    for f in supporting_facts:
        fid = f.get("id", "")
        text = (f.get("text") or "").strip().replace("\n", " ")
        ftype = f.get("type", "")
        fact_lines.append(f"- [{ftype}:{fid}] {text}")
    facts_block = "\n".join(fact_lines) if fact_lines else "(no supporting facts retrieved)"

    budget_hint = ""
    if max_output_tokens is not None:
        budget_hint = (
            f"\n\n## Output budget\n"
            f"Your JSON response must fit within ~{max_output_tokens} tokens. If you "
            "would need more than this to express every change, emit the highest-"
            "leverage edits first and drop the rest — a truncated response parses as "
            "nothing and loses every change, while a short one keeps the ones you sent."
        )

    # The *document's* budget, which is a different thing from the response budget
    # above and was previously not mentioned to this call at all. A delta refresh
    # only ever adds, so a long-lived page grows a little on every round and
    # crosses its configured budget after a few hundred of them with nothing in
    # the pipeline noticing. Enforcing it by truncation would delete knowledge
    # nobody asked to delete, so it is stated as a constraint the model satisfies
    # the same way it does everything else here — with operations, on the content
    # that has actually gone stale.
    document_hint = ""
    if document_budget is not None and document_tokens is not None:
        if document_tokens >= document_budget:
            document_hint = (
                f"\n\n## Document budget (EXCEEDED)\n"
                f"The document is ~{document_tokens} tokens against a budget of {document_budget}. "
                "As well as integrating the new facts, make room: remove or merge blocks that are "
                "superseded, duplicated, or no longer help answer the TOPIC, using remove_block, "
                "replace_block or replace_section_blocks. Reclaim space from stale content — never "
                "by dropping the facts you are integrating, and never by summarising a section that "
                "is still current into a sentence."
            )
        elif document_tokens >= int(document_budget * 0.8):
            document_hint = (
                f"\n\n## Document budget\n"
                f"The document is ~{document_tokens} tokens against a budget of {document_budget}. "
                "It is close to the limit, so prefer replacing stale blocks over appending new ones "
                "where the new facts update something the document already covers."
            )

    task_footer = (
        "## Task\n"
        "Output a JSON object matching the operations schema. Integrate the new "
        "supporting facts into CURRENT DOCUMENT. Add, update, or remove content "
        "as needed. Preserve unchanged sections and blocks by not mentioning them."
    )
    input_cap = max_input_tokens if max_input_tokens is not None else _STRUCTURED_DELTA_DEFAULT_MAX_INPUT_TOKENS
    doc_json, candidate, facts_body, input_truncated = _fit_structured_delta_prompt_parts(
        source_query=source_query,
        current_document_json=current_document_json,
        candidate_markdown=candidate_markdown,
        facts_block=facts_block,
        budget_hint=budget_hint,
        task_footer=task_footer,
        max_input_tokens=input_cap,
    )
    truncation_note = ""
    if input_truncated:
        truncation_note = (
            "\n\n*Note: Document, synthesis, or facts were truncated to fit the model "
            "context window. Prefer minimal, high-leverage operations.*"
        )

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT (apply ops to this; copy section and block ids from it verbatim)\n"
        f"```json\n{doc_json}\n```\n\n"
        f"## NEW INFORMATION SYNTHESIS (context for how new facts relate to the topic)\n"
        f"```markdown\n{candidate}\n```\n\n"
        f"## SUPPORTING FACTS (new since last refresh — integrate these)\n{facts_body}"
        f"{document_hint}{budget_hint}{truncation_note}\n\n"
        f"{task_footer}"
    )


STRUCTURED_RETRACTION_SYSTEM_PROMPT = """You are removing *retracted information* from an existing structured document.

You will be given:
1. TOPIC — the question this document answers.
2. CURRENT DOCUMENT (JSON) — the existing document. Each section has a stable
   ``id``, a ``heading``, a ``level`` (1..6), and an ordered list of ``blocks``.
   Each block has a stable ``id`` and a ``text`` field holding one markdown
   fragment — a paragraph, a list, a table, or a fenced code block.
3. RETRACTED FACTS — facts this document was built from that have since been
   removed from the memory bank. They are no longer true, no longer supported,
   or were withdrawn. The document may still state them.
4. STILL-SUPPORTED FACTS — other facts this document is built on that remain
   valid. They are listed so you can tell which content survives.

Your task: output a JSON object ``{"operations": [...]}`` that removes from
CURRENT DOCUMENT anything that rests on the RETRACTED FACTS, and nothing else.

RULES
- **Remove only what the retracted facts support.** If a sentence, bullet, row, or
  section states a retracted fact, remove it (``remove_block``, ``remove_section``)
  or rewrite it to drop just that claim (``replace_block``,
  ``replace_section_blocks``) when the block also carries content that survives.
- **When in doubt, keep it.** The document was written from far more facts than you
  are shown, and blocks do not record which fact they came from. Content that merely
  looks related to a retracted fact, or that could plausibly rest on evidence you
  cannot see, must be left exactly as it is. Removing something still true is worse
  than leaving something stale: the deletion is not recoverable, and no later pass
  can restore it.
- **A restated fact was not retracted.** If a retracted fact's content also appears
  in STILL-SUPPORTED FACTS, the underlying information was re-ingested rather than
  withdrawn — only its identifier changed. Keep that content.
- **Do not add anything.** No ``append_block``, no ``insert_block``, no
  ``add_section``. This pass only takes away.
- **Do not tidy.** No rewording, reordering, or reformatting of surviving content.
  Never leave a note saying something was removed — the document must read as though
  the retracted claim was never there.
- If removing a block would leave its section empty and the section exists only for
  that content, remove the section instead.
- Output ``{"operations": []}`` when nothing in the document rests on the retracted
  facts. That is a normal, expected answer.

ALLOWED OPERATIONS (each line shows the JSON shape)
- ``{"op": "remove_block", "section_id": "...", "block_id": "..."}``
- ``{"op": "remove_section", "section_id": "..."}``
- ``{"op": "replace_block", "section_id": "...", "block_id": "...", "text": "..."}``
- ``{"op": "replace_section_blocks", "section_id": "...", "blocks": ["...", "..."]}``

Blocks are addressed by ``block_id`` — the ``id`` printed beside each block in
CURRENT DOCUMENT. Copy it exactly; never invent one, and never use a position.
An operation naming an id that does not exist is dropped, so a retraction that
guesses removes nothing.

Every ``text`` (and every entry of ``blocks``) is ONE markdown fragment, written
as it should appear: ``"- one\\n- two"`` for a list,
``"| col | col |\\n| --- | --- |\\n| a | b |"`` for a table.

OUTPUT FORMAT
Return ONLY a single JSON object, with no prose before or after, no markdown code
fences, no commentary. The object must have exactly one top-level key,
``operations``, whose value is an array of operation objects (empty array when
nothing changes).

JSON STRING RULES (critical)
- Every string must be valid JSON: escape ``"`` as ``\\"``, backslashes as
  ``\\\\``, and every line break as ``\\n``. Never put a raw newline inside a
  JSON string.
- Do not append extra ``]`` or ``}`` after the closing ``}`` of the root object."""


def build_structured_retraction_prompt(
    *,
    current_document_json: str,
    retracted_facts: list[dict[str, Any]],
    surviving_facts: list[dict[str, Any]],
    source_query: str,
    max_output_tokens: int | None = None,
    max_input_tokens: int | None = None,
) -> str:
    """Build the user prompt for the retraction ("unsay") pass of a delta refresh.

    ``retracted_facts`` and ``surviving_facts`` are stored ``based_on`` entries —
    ``{id, text, type, context}``. The retracted ones are quoted from the document's
    own record because the rows themselves are gone: an observation swept away with
    its source keeps no history, so this copy is all that is left of what it said.

    ``surviving_facts`` are passed so the model can tell re-ingestion from
    withdrawal. When a document is re-retained its facts return under fresh ids, and
    the old ids read as retracted even though nothing was actually withdrawn; seeing
    the same content on both lists is what stops that from deleting live content.
    """

    def _lines(facts: list[dict[str, Any]]) -> str:
        out: list[str] = []
        for fact in facts:
            text = (fact.get("text") or "").strip().replace("\n", " ")
            out.append(f"- [{fact.get('type', '')}] {text}")
        return "\n".join(out)

    retracted_block = _lines(retracted_facts) or "(none)"
    surviving_block = _lines(surviving_facts) or "(none listed)"

    budget_hint = ""
    if max_output_tokens is not None:
        budget_hint = (
            f"\n\n## Output budget\n"
            f"Your JSON response must fit within ~{max_output_tokens} tokens. If more "
            "removals are needed than fit, prefer the highest-leverage ones first "
            "(``replace_section_blocks`` over many block-level ops) so the response "
            "always parses as valid JSON."
        )

    task_footer = (
        "## Task\n"
        "Output a JSON object matching the operations schema, removing content that "
        "rests on the RETRACTED FACTS. Leave everything else untouched. Emit "
        '``{"operations": []}`` if nothing in the document rests on them.'
    )

    # Reuse the delta fitter: same three oversized inputs (document, and two fact
    # lists standing in for synthesis + facts), same budget split, so a large
    # document cannot push the retracted list out of the window.
    input_cap = max_input_tokens if max_input_tokens is not None else _STRUCTURED_DELTA_DEFAULT_MAX_INPUT_TOKENS
    doc_json, surviving_body, retracted_body, input_truncated = _fit_structured_delta_prompt_parts(
        source_query=source_query,
        current_document_json=current_document_json,
        candidate_markdown=surviving_block,
        facts_block=retracted_block,
        budget_hint=budget_hint,
        task_footer=task_footer,
        max_input_tokens=input_cap,
    )
    truncation_note = ""
    if input_truncated:
        truncation_note = (
            "\n\n*Note: Document or fact lists were truncated to fit the model context "
            "window. Prefer minimal, high-leverage operations, and keep anything you "
            "cannot confidently attribute to a retracted fact.*"
        )

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT (apply ops to this; reference section ids as listed)\n"
        f"```json\n{doc_json}\n```\n\n"
        f"## STILL-SUPPORTED FACTS (these remain valid — do not remove content resting on them)\n"
        f"{surviving_body}\n\n"
        f"## RETRACTED FACTS (no longer in the memory bank — remove content resting on these)\n"
        f"{retracted_body}"
        f"{budget_hint}{truncation_note}\n\n"
        f"{task_footer}"
    )


DELTA_SYSTEM_PROMPT = """You are performing a surgical delta update to an existing mental model document.

You will be given:
1. CURRENT DOCUMENT: the existing mental model content (markdown).
2. CANDIDATE UPDATE: a freshly generated synthesis based on the latest retrieved memories.
3. SUPPORTING FACTS: the observations and facts that support the CANDIDATE UPDATE.

Your task: produce an updated version of the CURRENT DOCUMENT that reflects the new reality, with the MINIMUM possible changes.

ABSOLUTE RULES:
- Preserve unchanged content BYTE-FOR-BYTE. If a sentence, heading, bullet, code block, or section is still accurate according to the CANDIDATE UPDATE and SUPPORTING FACTS, copy it verbatim — same wording, same punctuation, same whitespace, same markdown structure.
- Do NOT reformat, rephrase, or re-style content that is still accurate. No "light edits for clarity", no reordering for flow, no synonym swaps.
- Remove content that is contradicted by the CANDIDATE UPDATE or SUPPORTING FACTS (stale content).
- Add new content ONLY when the SUPPORTING FACTS contain information not already in the CURRENT DOCUMENT.
- When adding new content, prefer appending to an existing relevant section. Creating a new section is acceptable when the new information does not fit any existing section.
- When creating a new section, match the heading style, tone, and formatting conventions used in the CURRENT DOCUMENT.
- Every assertion in your output MUST be grounded in either (a) the CURRENT DOCUMENT (preserved) or (b) the SUPPORTING FACTS. Never introduce outside knowledge.
- If nothing in the SUPPORTING FACTS contradicts or extends the CURRENT DOCUMENT, return the CURRENT DOCUMENT UNCHANGED, character for character.

OUTPUT FORMAT:
- Output ONLY the updated markdown document. No preamble, no explanation, no diff markers, no commentary.
- Do not wrap the output in code fences unless the CURRENT DOCUMENT itself was entirely a code fence."""
