"""
System prompts for the reflect agent.

The reflect agent uses hierarchical retrieval:
1. search_mental_models - User-curated summaries (highest quality)
2. search_observations - Consolidated knowledge with freshness awareness
3. recall - Raw facts as ground truth fallback
"""

import json
from typing import Any

import tiktoken

_TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")

# Fraction of max_context_tokens reserved for tool results in the final synthesis prompt.
# The remainder covers the system prompt, question, bank context, and output tokens.
_FINAL_PROMPT_CONTEXT_FRACTION = 0.8


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


def build_system_prompt_for_tools(
    bank_profile: dict[str, Any],
    context: str | None = None,
    directives: list[dict[str, Any]] | None = None,
    has_mental_models: bool = False,
    budget: str | None = None,
) -> str:
    """
    Build the system prompt for tool-calling reflect agent.

    The agent uses hierarchical retrieval:
    1. search_mental_models - User-curated summaries (try first, if available)
    2. search_observations - Consolidated knowledge with freshness
    3. recall - Raw facts as ground truth

    Args:
        bank_profile: Bank profile with name and mission
        context: Optional additional context
        directives: Optional list of directive mental models to inject as hard rules
        has_mental_models: Whether the bank has any mental models (skip if not)
        budget: Search depth budget - "low", "mid", or "high". Controls exploration thoroughness.
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
            "You are a reflection agent that answers questions by reasoning over retrieved memories.",
            "",
        ]
    )

    parts.extend(
        [
            "## LANGUAGE RULE (default - directives take precedence)",
            "- By default, detect the language of the user's question and respond in that SAME language.",
            "- If the question is in Chinese, respond in Chinese. If in Japanese, respond in Japanese.",
            "- IMPORTANT: The DIRECTIVES section above has HIGHER PRIORITY than this rule.",
            "  If a directive specifies a language (e.g. 'Always respond in French'), follow the directive.",
            "",
            "## CRITICAL RULES",
            "- ONLY use information from tool results - no external knowledge or guessing",
            "- You SHOULD synthesize, infer, and reason from the retrieved memories",
            "- You MUST search before saying you don't have information",
            "",
            "## How to Reason",
            "- If memories mention someone did an activity, you can infer they likely enjoyed it",
            "- Synthesize a coherent narrative from related memories",
            "- Be a thoughtful interpreter, not just a literal repeater",
            "- When the exact answer isn't stated, use what IS stated to give the best answer",
            "",
            "## HIERARCHICAL RETRIEVAL STRATEGY",
            "",
        ]
    )

    # Build retrieval levels based on what's available
    if has_mental_models:
        parts.extend(
            [
                "You have access to THREE levels of knowledge. Use them in this order:",
                "",
                "### 1. MENTAL MODELS (search_mental_models) - Try First",
                "- User-curated summaries about specific topics",
                "- HIGHEST quality - manually created and maintained",
                "- If a relevant mental model exists and is FRESH, it may fully answer the question",
                "- Check `is_stale` field - if stale, also verify with lower levels",
                "",
                "### 2. OBSERVATIONS (search_observations) - Second Priority",
                "- Auto-consolidated knowledge from memories",
                "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                "- Good for understanding patterns and summaries",
                "",
                "### 3. RAW FACTS (recall) - Ground Truth",
                "- Individual memories (world facts and experiences)",
                "- Use when: no mental models/observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_mental_models and search_observations both return 0 results, you MUST call recall() before giving up",
                "- This is the source of truth that other levels are built from",
                "",
            ]
        )
    else:
        parts.extend(
            [
                "You have access to TWO levels of knowledge. Use them in this order:",
                "",
                "### 1. OBSERVATIONS (search_observations) - Try First",
                "- Auto-consolidated knowledge from memories",
                "- Check `is_stale` field - if stale, ALSO use recall() to verify",
                "- Good for understanding patterns and summaries",
                "",
                "### 2. RAW FACTS (recall) - Ground Truth",
                "- Individual memories (world facts and experiences)",
                "- Use when: no observations exist, they're stale, or you need specific details",
                "- MANDATORY: If search_observations returns 0 results or count=0, you MUST call recall() before giving up",
                "- This is the source of truth that observations are built from",
                "",
            ]
        )

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

    if has_mental_models:
        parts.extend(
            [
                "1. First, try search_mental_models() - check if a curated summary exists",
                "2. If no mental model or it's stale, try search_observations() for consolidated knowledge",
                "3. If observations are stale OR you need specific details, use recall() for raw facts",
                "4. Use expand() if you need more context on specific memories",
                "5. When ready, call done() with your answer and supporting IDs",
            ]
        )
    else:
        parts.extend(
            [
                "1. First, try search_observations() - check for consolidated knowledge",
                "2. If search_observations returns 0 results OR observations are stale, you MUST call recall() for raw facts",
                "3. Use expand() if you need more context on specific memories",
                "4. When ready, call done() with your answer and supporting IDs",
            ]
        )

    parts.extend(
        [
            "",
            "## Output Format: Well-Formatted Markdown Answer",
            "Call done() with a well-formatted markdown 'answer' field.",
            "- USE markdown formatting for structure (headers, lists, bold, italic, code blocks, tables, etc.)",
            "- CRITICAL: Add blank lines before and after block elements (tables, code blocks, lists)",
            "- Format for clarity and readability with proper spacing and hierarchy",
            "- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text",
            "- Put IDs ONLY in the memory_ids/mental_model_ids/observation_ids arrays, not in the answer",
            "- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.",
        ]
    )

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


def build_agent_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
) -> str:
    """Build the user prompt for the reflect agent."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits if present
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

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history
    if context_history:
        parts.append("\n## Tool Results (synthesize and reason from this data)")
        for i, entry in enumerate(context_history, 1):
            tool = entry["tool"]
            output = entry["output"]
            # Format as proper JSON for LLM readability
            try:
                output_str = json.dumps(output, indent=2, default=str)
            except (TypeError, ValueError):
                output_str = str(output)
            parts.append(f"\n### Call {i}: {tool}\n```json\n{output_str}\n```")

    # The question
    parts.append(f"\n## Question\n{query}")

    # Instructions
    if context_history:
        parts.append(
            "\n## Instructions\n"
            "Based on the tool results above, either call more tools or provide your final answer. "
            "Synthesize and reason from the data - make reasonable inferences when helpful. "
            "If you have related information, use it to give the best possible answer."
        )
    else:
        parts.append(
            "\n## Instructions\n"
            "Start by searching for relevant information using the hierarchical retrieval strategy:\n"
            "1. Try search_mental_models() first for curated summaries\n"
            "2. Try search_observations() for consolidated knowledge\n"
            "3. Use recall() for specific details or to verify stale data"
        )

    return "\n".join(parts)


def build_final_prompt(
    query: str,
    context_history: list[dict],
    bank_profile: dict,
    additional_context: str | None = None,
    max_context_tokens: int = 100_000,
) -> str:
    """Build the final prompt when forcing a text response (no tools)."""
    parts = []

    # Bank identity
    name = bank_profile.get("name", "Assistant")
    mission = bank_profile.get("mission", "")

    parts.append(f"## Memory Bank Context\nName: {name}")
    if mission:
        parts.append(f"Mission: {mission}")

    # Disposition traits if present
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

    # Additional context from caller
    if additional_context:
        parts.append(f"\n## Additional Context\n{additional_context}")

    # Tool call history — include as many entries as fit within the token budget,
    # preferring the most recent calls (they tend to be the most targeted).
    if context_history:
        parts.append("\n## Retrieved Data (synthesize and reason from this data)")
        token_budget = int(max_context_tokens * _FINAL_PROMPT_CONTEXT_FRACTION)
        # Render entries newest-first, then reverse so the prompt reads chronologically.
        rendered: list[str] = []
        truncated = False
        for entry in reversed(context_history):
            tool = entry["tool"]
            output = entry["output"]
            try:
                output_str = json.dumps(output, indent=2, default=str)
            except (TypeError, ValueError):
                output_str = str(output)
            block = f"\n### From {tool}:\n```json\n{output_str}\n```"
            block_tokens = len(_TIKTOKEN_ENCODING.encode(block))
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
    parts.append(
        "\n## Instructions\n"
        "Provide a thoughtful answer by synthesizing and reasoning from the retrieved data above. "
        "You can make reasonable inferences from the memories, but don't completely fabricate information. "
        "If the exact answer isn't stated, use what IS stated to give the best possible answer. "
        "Only say 'I don't have information' if the retrieved data is truly unrelated to the question.\n\n"
        "IMPORTANT: Output ONLY the final answer. Do NOT include meta-commentary like "
        '"I\'ll search..." or "Let me analyze...". Do NOT explain your reasoning process. '
        "Just provide the direct synthesized answer."
    )

    return "\n".join(parts)


FINAL_SYSTEM_PROMPT = """CRITICAL: You MUST ONLY use information from retrieved tool results. NEVER make up names, people, events, or entities.

You are a thoughtful assistant that synthesizes answers from retrieved memories.

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


STRUCTURED_DELTA_SYSTEM_PROMPT = """You are computing a *minimal patch* to a structured document.

You will be given:
1. CURRENT DOCUMENT (JSON) — the existing structured mental model. Each section
   has a stable ``id``, a ``heading``, a ``level`` (1..6), and an ordered list
   of ``blocks``. Blocks are typed: ``paragraph``, ``bullet_list``,
   ``ordered_list``, or ``code``.
2. CANDIDATE SUMMARY (markdown) — a freshly generated synthesis of the latest
   memories, useful only as a hint about *what new information exists*. You
   MUST NOT copy its formatting or wording wholesale; it is not the target.
3. SUPPORTING FACTS — the observations and facts the candidate is grounded in.
   Treat these as the only source of new information.

Your task: output a JSON object ``{"operations": [...]}``. Applied to CURRENT
DOCUMENT, the operations must produce the smallest possible change that
reflects the new facts.

ABSOLUTE RULES
- If CURRENT DOCUMENT already covers all the supporting facts, output
  exactly ``{"operations": []}``. An empty operation list IS the correct
  answer when nothing new has come in. This is the most common case.
- Operations target sections by ``section_id`` (use the ``id`` field of the
  section in CURRENT DOCUMENT, NOT the heading). Block operations target
  blocks by ``index`` (0-based, against the section's current block list).
- Add new content with ``append_block``, ``insert_block``, or ``add_section``.
  Prefer extending an existing section over creating a new one.
- Modify existing content with ``replace_block`` or ``replace_section_blocks``
  ONLY when the supporting facts contradict the current text. Do NOT rewrite
  for style, brevity, or "improvement".
- Remove stale content with ``remove_block`` or ``remove_section`` ONLY when
  the supporting facts directly contradict it.
- NEVER emit operations whose only effect is to reword unchanged content.
- NEVER emit operations to "normalize" formatting (numbered → bulleted, casing
  changes, paragraph → list, etc).
- Every operation MUST be justifiable by a specific fact in SUPPORTING FACTS.

ALLOWED OPERATIONS (each line shows the JSON shape)
- ``{"op": "append_block", "section_id": "...", "block": {...}}``
- ``{"op": "insert_block", "section_id": "...", "index": N, "block": {...}}``
- ``{"op": "replace_block", "section_id": "...", "index": N, "block": {...}}``
- ``{"op": "remove_block", "section_id": "...", "index": N}``
- ``{"op": "add_section", "heading": "...", "level": 2, "blocks": [...], "after_section_id": "..."}``
- ``{"op": "remove_section", "section_id": "..."}``
- ``{"op": "replace_section_blocks", "section_id": "...", "blocks": [...]}``
- ``{"op": "rename_section", "section_id": "...", "new_heading": "..."}``

Block shapes
- ``{"type": "paragraph", "text": "..."}``
- ``{"type": "bullet_list", "items": ["...", "..."]}``
- ``{"type": "ordered_list", "items": ["...", "..."]}``
- ``{"type": "code", "language": "json", "text": "..."}``

OUTPUT FORMAT
Return ONLY a single JSON object on its own, with no prose before or after,
no markdown code fences, no commentary. The object must have exactly one
top-level key, ``operations``, whose value is an array of operation objects
(empty array when nothing changes).

Examples
- No changes needed → ``{"operations": []}``
- Add one bullet to an existing "Members" section →
  ``{"operations": [{"op": "append_block", "section_id": "members",
  "block": {"type": "bullet_list", "items": ["Carol — junior engineer"]}}]}``"""


def build_structured_delta_prompt(
    *,
    current_document_json: str,
    candidate_markdown: str,
    supporting_facts: list[dict[str, Any]],
    source_query: str,
    max_output_tokens: int | None = None,
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
            "would need more than this to express every change, prefer the highest-"
            "leverage edits first (a few ``replace_section_blocks`` ops over many "
            "block-level ops) so the response always parses as valid JSON."
        )

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT (apply ops to this; reference section ids as listed)\n"
        f"```json\n{current_document_json}\n```\n\n"
        f"## CANDIDATE SUMMARY (hint only — do NOT copy wording wholesale)\n"
        f"```markdown\n{candidate_markdown}\n```\n\n"
        f"## SUPPORTING FACTS (the only source of new information)\n{facts_block}"
        f"{budget_hint}\n\n"
        "## Task\n"
        "Output a JSON object matching the operations schema. Use an empty list "
        "if no new fact requires a change. Otherwise, emit the smallest set of "
        "operations that reflects the new facts in CURRENT DOCUMENT, preserving "
        "all unchanged sections and blocks by simply not mentioning them."
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


def build_delta_prompt(
    *,
    current_content: str,
    candidate_content: str,
    supporting_facts: list[dict[str, Any]],
    source_query: str,
) -> str:
    """Build the user prompt for a delta-mode mental model refresh.

    Args:
        current_content: The existing mental model content (to preserve as much as possible).
        candidate_content: Fresh synthesis from the reflect agent reflecting new reality.
        supporting_facts: Flat list of fact dicts (id, text, type) supporting the candidate.
        source_query: The mental model's source query, for topical framing.
    """
    fact_lines: list[str] = []
    for f in supporting_facts:
        fid = f.get("id", "")
        text = (f.get("text") or "").strip().replace("\n", " ")
        ftype = f.get("type", "")
        fact_lines.append(f"- [{ftype}:{fid}] {text}")
    facts_block = "\n".join(fact_lines) if fact_lines else "(no supporting facts retrieved)"

    return (
        f"## Topic\n{source_query}\n\n"
        f"## CURRENT DOCUMENT\n```markdown\n{current_content}\n```\n\n"
        f"## CANDIDATE UPDATE\n```markdown\n{candidate_content}\n```\n\n"
        f"## SUPPORTING FACTS\n{facts_block}\n\n"
        "## Task\n"
        "Produce the updated mental model document by applying the minimum necessary changes "
        "to CURRENT DOCUMENT so that it reflects CANDIDATE UPDATE and SUPPORTING FACTS. "
        "Preserve unchanged content byte-for-byte. Output only the final markdown."
    )
