"""Exact-output unit tests for ``build_system_prompt_for_tools``.

The reflect agent's behaviour depends entirely on the prompt text — telling
the LLM to call a tool that isn't exposed (see #1724) causes weaker models
to either hallucinate the call or give up. These tests pin the prompt
output byte-for-byte across every if/else branch of the builder:

- The 4 ``(has_mental_models, include_observations)`` combinations for the
  HIERARCHICAL RETRIEVAL STRATEGY and Workflow sections.
- ``budget`` ∈ {None, "low", "mid", "high"}.
- ``mission`` present / absent.
- ``directives`` present / absent (header section + end-of-prompt reminder).
- ``disposition`` present / absent.
- ``context`` present / absent.

Each test calls the builder and asserts the full prompt string equals an
assembled expected value. Shared section constants keep the tests DRY
without hiding what each one produces — the ``_assemble`` helper is a
mechanical join, not a re-implementation of the builder.
"""

from hindsight_api.engine.reflect.prompts import build_final_system_prompt, build_system_prompt_for_tools

BANK = {"name": "TestBank", "mission": ""}

_HEADER = (
    "CRITICAL: You MUST ONLY use information from retrieved tool results. "
    "NEVER make up names, people, events, or entities."
)

_DEFAULT_ROLE = "You are a reflection agent that answers questions by reasoning over retrieved memories."

_LANGUAGE_AND_RULES = """\
## LANGUAGE RULE (default - directives take precedence)
- By default, detect the language of the user's question and respond in that SAME language.
- If the question is in Chinese, respond in Chinese. If in Japanese, respond in Japanese.
- IMPORTANT: The DIRECTIVES section above has HIGHER PRIORITY than this rule.
  If a directive specifies a language (e.g. 'Always respond in French'), follow the directive.

## CRITICAL RULES
- ONLY use information from tool results - no external knowledge or guessing
- You SHOULD synthesize, infer, and reason from the retrieved memories
- You MUST search before saying you don't have information

## How to Reason
- If memories mention someone did an activity, you can infer they likely enjoyed it
- Synthesize a coherent narrative from related memories
- Be a thoughtful interpreter, not just a literal repeater
- When the exact answer isn't stated, use what IS stated to give a best-effort answer AND surface any uncertainty — never invent confidence the data doesn't support.

## Temporal Reasoning
Every memory and observation carries temporal fields in the JSON tool result:
- `mentioned_at` — when the user retained the fact (always set).
- `occurred_start` / `occurred_end` — when the underlying event happened (optional, set for dated events).

When facts about the SAME facet conflict — counts, statuses, ownership, location, presence, etc. — the fact with the LATEST `mentioned_at` is authoritative. Later statements SUPERSEDE earlier ones. Do NOT average, sum, or favor an explicitly-dated fact over a more recent one.

Example: three count facts come back from recall:
  - 'Team has 2 engineers' (mentioned_at=T1)
  - 'Team now has 1 engineer' (mentioned_at=T2, occurred_start=2026-05-25)
  - 'Team has 5 engineers' (mentioned_at=T3)
with T1 < T2 < T3. The current size is 5, not 1. Then apply later events (e.g. someone leaving after T3) on top of that.

For reconstructing a TIMELINE of events, order by `occurred_start` / `occurred_end` (when things happened), not `mentioned_at` (when they were retained).

## Conflicts and Ambiguity
Not every retrieval converges on a single answer. Distinguish two cases:

- RESOLVABLE conflict — the temporal rule above (latest `mentioned_at` wins) cleanly picks a winner. Apply it and move on.
- UNRESOLVABLE ambiguity — the data is internally inconsistent in a way the temporal rule does NOT settle. Examples: a recent aggregate (count, total) is incompatible with the individual entities you can enumerate; two equally-recent facts disagree and no later fact resolves them; events are described but their relative order is unclear; the user's own statements contradict each other and nothing later reconciles them.

When the data is genuinely ambiguous: SAY SO in your answer. Name the conflicting facts. Explain why they can't be reconciled. Give a range or a best-effort interpretation with explicit uncertainty (e.g. 'between X and Y, depending on [unresolved condition]'; or 'the most recent statement says A, but B was stated earlier and the gap isn't accounted for in any later fact').

An honest 'the data is inconsistent about X' beats a confident wrong answer. Do NOT pick a value arbitrarily, average conflicting values, or smooth over gaps in confident prose. Acknowledging ambiguity is a successful answer, not a failure mode.

## Showing Your Reasoning
For any answer that resolves a conflict between facts, applies events on top of a count or status, or settles an ambiguity — show your work in the answer text so a reader can audit it.

Walk through these steps explicitly:
1. **List the relevant facts in `mentioned_at` order (oldest → newest)**, each with the value it asserts. Use a short bulleted list.
2. **Identify the authoritative fact** under the temporal rule (latest `mentioned_at` for the contested facet). Write its date down.
3. **List candidate events to apply on top** — anything that changes the count, status, or state being asked about. Write each event's date down next to it.
4. **Sanity-check each candidate event against the authoritative date** — for EVERY event from step 3, write a one-line check in the form `<event> (<event_date>) vs authoritative (<authoritative_date>) → BEFORE/AFTER → KEEP/DROP`. If the event is BEFORE or EQUAL to the authoritative date, DROP it: it is already reflected in the authoritative fact, and applying it again is double-counting. This is the single most common mistake — do not skip this step even if you feel confident.
5. **Show the arithmetic or derivation explicitly** using only the KEEP events from step 4 — e.g. 'authoritative count = 5 (at 2025-02-12); kept events: Shadow died (2025-03-12, AFTER); 5 − 1 = 4'.
6. If step 2 or 3 cannot be done cleanly (no clear winner, overlapping timestamps, unclear event order), STOP and surface this as an UNRESOLVABLE ambiguity per the section above — do not fabricate a derivation.

For simple factual lookups that don't involve conflict or arithmetic, you can answer directly without this scaffolding.

## HIERARCHICAL RETRIEVAL STRATEGY
"""

_QUERY_STRATEGY = """\
## Query Strategy
recall() uses semantic search. NEVER just echo the user's question - decompose it into targeted searches:

BAD: User asks 'recurring lesson themes between students' → recall('recurring lesson themes between students')
GOOD: Break it down into component searches:
  1. recall('lessons') - find all lesson-related memories
  2. recall('teaching sessions') - alternative phrasing
  3. recall('student progress') - find student-related memories

Think: What ENTITIES and CONCEPTS does this question involve? Search for each separately.
"""

_OUTPUT_FORMAT = """\
## Output Format: Well-Formatted Markdown Answer
Call done() with a well-formatted markdown 'answer' field.
- USE markdown formatting for structure (headers, lists, bold, italic, code blocks, tables, etc.)
- CRITICAL: Add blank lines before and after block elements (tables, code blocks, lists)
- Format for clarity and readability with proper spacing and hierarchy
- NEVER include memory IDs, UUIDs, or 'Memory references' in the answer text
- Put IDs ONLY in the memory_ids/mental_model_ids/observation_ids arrays, not in the answer
- CRITICAL: This is a NON-CONVERSATIONAL system. NEVER ask follow-up questions, offer further assistance, or suggest next steps. Your answer must be complete and self-contained. The user cannot reply.\
"""

_BANK_HEADER = "## Memory Bank: TestBank"

# --- Retrieval-strategy variants (one per (has_mental_models, include_observations) combo) ---

_RETRIEVAL_MM_AND_OBS = """\
You have access to THREE levels of knowledge. Use them in this order:

### 1. MENTAL MODELS (search_mental_models) - Try First
- User-curated summaries about specific topics
- HIGHEST quality - manually created and maintained
- If a relevant mental model exists and is FRESH, it may fully answer the question
- Check `is_stale` field - if stale, also verify with lower levels

### 2. OBSERVATIONS (search_observations) - Second Priority
- Auto-consolidated knowledge from memories
- Check `is_stale` field - if stale, ALSO use recall() to verify
- Good for understanding patterns and summaries

### 3. RAW FACTS (recall) - Ground Truth
- Individual memories (world facts and experiences)
- Use when: no mental models/observations exist, they're stale, or you need specific details
- MANDATORY: If search_mental_models and search_observations both return 0 results, you MUST call recall() before giving up
- This is the source of truth that other levels are built from

**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).

"""

_RETRIEVAL_MM_ONLY = """\
You have access to TWO levels of knowledge. Use them in this order:

### 1. MENTAL MODELS (search_mental_models) - Try First
- User-curated summaries about specific topics
- HIGHEST quality - manually created and maintained
- If a relevant mental model exists and is FRESH, it may fully answer the question
- Check `is_stale` field - if stale, also verify with lower levels

### 2. RAW FACTS (recall) - Ground Truth
- Individual memories (world facts and experiences)
- Use when: no mental model exists, it's stale, or you need specific details
- MANDATORY: If search_mental_models returns 0 results, you MUST call recall() before giving up
- This is the source of truth that mental models are built from
"""

_RETRIEVAL_OBS_ONLY = """\
You have access to TWO levels of knowledge. Use them in this order:

### 1. OBSERVATIONS (search_observations) - Try First
- Auto-consolidated knowledge from memories
- Check `is_stale` field - if stale, ALSO use recall() to verify
- Good for understanding patterns and summaries

### 2. RAW FACTS (recall) - Ground Truth
- Individual memories (world facts and experiences)
- Use when: no observations exist, they're stale, or you need specific details
- MANDATORY: If search_observations returns 0 results or count=0, you MUST call recall() before giving up
- This is the source of truth that observations are built from

**Tool result ordering:** `recall()` and `search_observations()` return their `memories` / `observations` arrays sorted by SEMANTIC RELEVANCE to the query, NOT by time. The POSITION of an entry tells you nothing about when it was retained. For any temporal reasoning — recency, supersession, applying events on top of a state — IGNORE the position and read the per-entry `mentioned_at` field (and `occurred_start` / `occurred_end` for events).

"""

_RETRIEVAL_RECALL_ONLY = """\
You have access to ONE level of knowledge:

### 1. RAW FACTS (recall) - Ground Truth
- Individual memories (world facts and experiences)
- MANDATORY: Call recall() to gather facts before giving up
- This is the source of truth.
"""

# --- Workflow variants ---

_WORKFLOW_MM_AND_OBS = """\
## Workflow
1. First, try search_mental_models() - check if a curated summary exists
2. If no mental model or it's stale, try search_observations() for consolidated knowledge
3. If observations are stale OR you need specific details, use recall() for raw facts
4. Use expand() if you need more context on specific memories
5. When ready, call done() with your answer and supporting IDs\
"""

_WORKFLOW_MM_ONLY = """\
## Workflow
1. First, try search_mental_models() - check if a curated summary exists
2. If no mental model or it's stale, use recall() for raw facts
3. Use expand() if you need more context on specific memories
4. When ready, call done() with your answer and supporting IDs\
"""

_WORKFLOW_OBS_ONLY = """\
## Workflow
1. First, try search_observations() - check for consolidated knowledge
2. If search_observations returns 0 results OR observations are stale, you MUST call recall() for raw facts
3. Use expand() if you need more context on specific memories
4. When ready, call done() with your answer and supporting IDs\
"""

_WORKFLOW_RECALL_ONLY = """\
## Workflow
1. Call recall() to gather raw facts
2. Use expand() if you need more context on specific memories
3. When ready, call done() with your answer and supporting IDs\
"""

# --- Budget variants (inserted between Query Strategy and Workflow) ---

_BUDGET_LOW = """\
## RESEARCH DEPTH: SHALLOW (Quick Response)
- Prioritize speed over completeness
- If mental models or observations provide a reasonable answer, stop there
- Only dig deeper if the initial results are clearly insufficient
- Prefer a quick overview rather than exhaustive details
- Answer promptly with available information
"""

_BUDGET_MID = """\
## RESEARCH DEPTH: MODERATE (Balanced)
- Balance thoroughness with efficiency
- Check multiple sources when the question warrants it
- Verify stale data if it's central to the answer
- Don't over-explore, but ensure reasonable coverage
"""

_BUDGET_HIGH = """\
## RESEARCH DEPTH: DEEP (Thorough Exploration)
- Explore comprehensively before answering
- Search across all available knowledge levels
- Use multiple query variations to ensure coverage
- Verify information across different retrieval levels
- Use expand() to get full context on important memories
- Take time to synthesize a complete, well-researched answer
"""

# --- Directives header block (inserted between anti-hallucination and role) ---

_DIRECTIVES_HEADER = """\
## DIRECTIVES (MANDATORY)
These are hard rules you MUST follow in ALL responses:

- **No Competitors**: Never mention competitor names.

NEVER violate these directives, even if other context suggests otherwise.
IMPORTANT: Do NOT explain or justify how you handled directives in your answer. Just follow them silently.
"""

# --- Directives reminder block (appended after Memory Bank header) ---

_DIRECTIVES_REMINDER = """

## REMINDER: MANDATORY DIRECTIVES
Before responding, ensure your answer complies with ALL of these directives:

1. **No Competitors**: Never mention competitor names.

Your response will be REJECTED if it violates any directive above.
Do NOT include any commentary about how you handled directives - just follow them.\
"""


def _assemble(
    retrieval: str,
    workflow: str,
    *,
    role: str = _DEFAULT_ROLE,
    directives_header: str = "",
    budget: str = "",
    trailer: str = "",
) -> str:
    """Mechanically join the sections in the same order the builder does.

    This is not a re-implementation of the builder — it just stitches
    precomputed text fragments together. The shape mirrors the
    ``"\\n".join(parts)`` pattern in ``build_system_prompt_for_tools``.
    """
    parts: list[str] = []
    parts.append(_HEADER)
    parts.append("")
    if directives_header:
        parts.append(directives_header)
    parts.append(role)
    parts.append("")
    parts.append("Answer the user's question by reasoning over retrieved memories.")
    parts.append("")
    parts.append(_LANGUAGE_AND_RULES)
    parts.append(retrieval)
    parts.append(_QUERY_STRATEGY)
    if budget:
        parts.append(budget)
    parts.append(workflow)
    parts.append("")
    parts.append(_OUTPUT_FORMAT)
    parts.append("")
    parts.append(_BANK_HEADER + trailer)
    return "\n".join(parts)


# =========================================================================
# Retrieval × Workflow combinations (the 4 branches added/preserved by #1724)
# =========================================================================


class TestRetrievalAndWorkflowBranches:
    """Exact-output checks for each (has_mental_models, include_observations) combo."""

    def test_mm_and_observations_renders_three_levels(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=True,
            include_observations=True,
        )
        assert actual == _assemble(_RETRIEVAL_MM_AND_OBS, _WORKFLOW_MM_AND_OBS)

    def test_mm_only_renders_two_levels_without_observations(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=True,
            include_observations=False,
        )
        assert actual == _assemble(_RETRIEVAL_MM_ONLY, _WORKFLOW_MM_ONLY)
        # The whole point of the fix: when observations are disabled, the
        # prompt must not name the tool.
        assert "search_observations" not in actual

    def test_observations_only_renders_two_levels_without_mental_models(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=True,
        )
        assert actual == _assemble(_RETRIEVAL_OBS_ONLY, _WORKFLOW_OBS_ONLY)
        assert "search_mental_models" not in actual

    def test_recall_only_renders_one_level(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY)
        # The exact bug from #1724: this branch must not advertise either
        # upstream tool the agent has disabled.
        assert "search_observations" not in actual
        assert "search_mental_models" not in actual


# =========================================================================
# Budget guidance branches
# =========================================================================


class TestBudgetBranches:
    """The ``budget`` parameter inserts a RESEARCH DEPTH section between
    Query Strategy and Workflow. ``None`` (default) inserts nothing.
    """

    def test_no_budget_omits_research_depth_section(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget=None,
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY)
        assert "RESEARCH DEPTH" not in actual

    def test_budget_low_inserts_shallow_block(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="low",
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_LOW)

    def test_budget_mid_inserts_moderate_block(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="mid",
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_MID)

    def test_budget_high_inserts_deep_block(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="high",
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_HIGH)

    def test_unknown_budget_inserts_nothing(self):
        # The builder only recognises low/mid/high; any other value is a no-op.
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="extreme",
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY)


# =========================================================================
# Bank-profile branches: mission, disposition
# =========================================================================


class TestBankProfileBranches:
    def test_mission_replaces_default_role_and_appends_mission_line(self):
        profile = {"name": "TestBank", "mission": "Track customer feedback"}
        actual = build_system_prompt_for_tools(
            bank_profile=profile,
            has_mental_models=False,
            include_observations=False,
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY,
            _WORKFLOW_RECALL_ONLY,
            role="Track customer feedback",
            trailer="\nMission: Track customer feedback",
        )

    def test_empty_mission_uses_default_role_and_no_mission_line(self):
        # Both the mission-as-role substitution and the trailing Mission line
        # are gated on ``mission`` being truthy.
        actual = build_system_prompt_for_tools(
            bank_profile={"name": "TestBank", "mission": ""},
            has_mental_models=False,
            include_observations=False,
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY)
        assert "Mission:" not in actual

    def test_disposition_appends_trait_line(self):
        profile = {
            "name": "TestBank",
            "mission": "",
            "disposition": {"skepticism": 3, "literalism": 2, "empathy": 4},
        }
        actual = build_system_prompt_for_tools(
            bank_profile=profile,
            has_mental_models=False,
            include_observations=False,
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY,
            _WORKFLOW_RECALL_ONLY,
            trailer="\nDisposition: skepticism=3, literalism=2, empathy=4",
        )

    def test_no_disposition_omits_trait_line(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
        )
        assert "Disposition:" not in actual


# =========================================================================
# Optional caller-supplied content: directives, additional context
# =========================================================================


class TestDirectivesAndContextBranches:
    def test_directives_add_header_section_and_reminder(self):
        directives = [{"name": "No Competitors", "content": "Never mention competitor names."}]
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            directives=directives,
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY,
            _WORKFLOW_RECALL_ONLY,
            directives_header=_DIRECTIVES_HEADER,
            trailer=_DIRECTIVES_REMINDER,
        )

    def test_no_directives_omits_both_sections(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            directives=None,
        )
        assert actual == _assemble(_RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY)
        assert "## DIRECTIVES" not in actual
        assert "## REMINDER: MANDATORY DIRECTIVES" not in actual

    def test_additional_context_appends_section_after_bank_header(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            context="Focus on Q3 metrics.",
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY,
            _WORKFLOW_RECALL_ONLY,
            trailer="\n\n## Additional Context\nFocus on Q3 metrics.",
        )

    def test_no_additional_context_omits_section(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            context=None,
        )
        assert "## Additional Context" not in actual


# =========================================================================
# include_observations default value
# =========================================================================


def test_include_observations_defaults_to_true():
    """Callers that don't pass ``include_observations`` get the original
    observations-enabled prompt — this guards the API default so reflect
    paths that don't gate the flag aren't silently changed."""
    actual = build_system_prompt_for_tools(bank_profile=BANK, has_mental_models=False)
    assert actual == _assemble(_RETRIEVAL_OBS_ONLY, _WORKFLOW_OBS_ONLY)


# =========================================================================
# build_final_system_prompt: language rule + directives
#
# The final synthesis is a SEPARATE LLM call from the agent loop, so its
# system prompt must independently carry the language rule and any directives
# — otherwise the answer-writing model has no instruction to stay in the
# query's language / obey a language directive and weaker models drift to
# English (flaky multilingual reflect tests).
# =========================================================================

_FRENCH_DIRECTIVE = {
    "name": "Language Policy",
    "content": "ALWAYS respond in French language. Never respond in English.",
}


def test_final_prompt_always_includes_language_rule():
    prompt = build_final_system_prompt()
    assert "## LANGUAGE" in prompt
    assert "SAME language as the user's question" in prompt


def test_final_prompt_without_directives_omits_directives_section():
    prompt = build_final_system_prompt()
    assert "## DIRECTIVES (MANDATORY)" not in prompt
    assert "REMINDER: MANDATORY DIRECTIVES" not in prompt


def test_final_prompt_injects_directives_so_answer_obeys_them():
    """The answer-writing model — not just the reasoning loop — must see the
    directive, else a 'respond in French' rule is silently dropped at synthesis."""
    prompt = build_final_system_prompt(directives=[_FRENCH_DIRECTIVE])
    assert "## DIRECTIVES (MANDATORY)" in prompt
    assert "respond in French" in prompt
    # End-of-prompt reminder reinforces compliance, mirroring the reasoning prompt.
    assert "REMINDER: MANDATORY DIRECTIVES" in prompt
    # The default language rule still defers to the directive.
    assert "takes precedence over this default" in prompt


def test_final_prompt_output_language_override_is_appended_last():
    """HINDSIGHT_API_LLM_OUTPUT_LANGUAGE forces a language regardless of query/directive."""
    prompt = build_final_system_prompt(llm_output_language="Spanish")
    assert "Respond exclusively in Spanish" in prompt
    # The config override is appended after the default LANGUAGE rule so it wins.
    assert prompt.index("Respond exclusively in Spanish") > prompt.index("## LANGUAGE")
