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

from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

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
- When the exact answer isn't stated, use what IS stated to give the best answer

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
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_LOW
        )

    def test_budget_mid_inserts_moderate_block(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="mid",
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_MID
        )

    def test_budget_high_inserts_deep_block(self):
        actual = build_system_prompt_for_tools(
            bank_profile=BANK,
            has_mental_models=False,
            include_observations=False,
            budget="high",
        )
        assert actual == _assemble(
            _RETRIEVAL_RECALL_ONLY, _WORKFLOW_RECALL_ONLY, budget=_BUDGET_HIGH
        )

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
    actual = build_system_prompt_for_tools(
        bank_profile=BANK, has_mental_models=False
    )
    assert actual == _assemble(_RETRIEVAL_OBS_ONLY, _WORKFLOW_OBS_ONLY)
