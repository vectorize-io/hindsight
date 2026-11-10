"""The session rule writer, with a real LLM: what it writes from an agent's verdict on a note.

The mechanics (when it runs, the guards, what it stores) are covered with stubs in
test_sessions.py. This covers how the model reads WRITE_PROMPT: a rejected note must turn into the
fact the agent states instead, worded as an instruction for later changes (the session check's
decision model picks a rule by that wording), and a reason that settles nothing writes no rule.
"""

import pytest

from hindsight_api import LLMConfig
from hindsight_api.engine.sessions import learn as L
from tests.llm_judge import assert_meets_criteria

pytestmark = pytest.mark.hs_llm_core


async def _draft(note: str, verdict: str, reason: str, goal: str) -> L.RuleDraft:
    # Built as service._learn_one builds it: a rejected note is not shown to the writer.
    prompt = L.WRITE_PROMPT.format(
        note=note if verdict == "used" else L.REJECTED_NOTE,
        verdict=verdict,
        reason=reason,
        goal=goal,
        guide=L.VERDICT_GUIDE[verdict],
        existing=L.existing_rules_text([]),
    )
    result = await LLMConfig.from_env().call(
        messages=[{"role": "user", "content": prompt}],
        response_format=L.RuleDraft,
        scope="session_learn",
        strict_schema=True,
    )
    return result.content if isinstance(result.content, L.RuleDraft) else L.RuleDraft.model_validate(result.content)


async def test_a_wrong_note_becomes_the_rule_the_agent_states_for_later_changes():
    draft = await _draft(
        note="Before you change shell.tsx: Files, Knowledge and Settings must each stay their own floating panel.",
        verdict="wrong",
        reason="that is outdated: the user asked to merge Files, Knowledge and Settings into one System panel "
        "in shell.tsx, and they now open as tabs inside it.",
        goal="Merge the three floating panels in shell.tsx into one System panel",
    )
    await assert_meets_criteria(
        response=f"create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria=(
            "create is True. The rule says Files, Knowledge and Settings are tabs of one System panel in "
            "shell.tsx, and is worded as guidance for any later change there (for example where a new panel "
            "goes), not only as a description of the merge that was done. It never says they stay separate "
            "floating panels."
        ),
        context="An agent rejected a memory note saying the panels stay floating; the user had them merged.",
    )


async def test_a_reason_that_settles_nothing_writes_no_rule():
    draft = await _draft(
        note="Before you change units.py: K is 1024.",
        verdict="not_needed",
        reason="I was already doing it.",
        goal="Fix the unit sizes in units.py",
    )
    await assert_meets_criteria(
        response=f"create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria="The response shows create=False, meaning no rule is written. Empty name and content are expected.",
        context="The agent dismissed a note because the work already did what it said.",
    )


async def test_a_value_the_agent_chose_for_its_run_is_not_the_rule():
    draft = await _draft(
        note="Local database tests should use `HINDSIGHT_TEST_PG_PORT=5617`.",
        verdict="used",
        reason="I ran the tests with `HINDSIGHT_TEST_PG_PORT=5617`. The default port was taken by another "
        "session, and that is why the first runs failed.",
        goal="Fix the flaky queue test and run the suite",
    )
    await assert_meets_criteria(
        response=f"create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria=(
            "The rule says to set HINDSIGHT_TEST_PG_PORT to a free port when the default one is taken. It does "
            "not require the specific port 5617: that was only the free port one run happened to pick."
        ),
        context="An agent picked port 5617 for its own test run because the default port was busy.",
    )


async def test_a_known_flaky_test_is_never_a_rule_to_ignore_its_failures():
    draft = await _draft(
        note="test_document_mutation_and_relink_preserve_queue is flaky: under load it treats a blocked "
        "command as the queue insert.",
        verdict="used",
        reason="it said this test is flaky because, under load, it can watch the wrong blocked database "
        "command and wrongly treat it as the queue insert.",
        goal="Ship the PR",
    )
    await assert_meets_criteria(
        response=f"create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria=(
            "The rule says the test is flaky under load and why. It does NOT tell agents to skip, ignore or not "
            "investigate the test's failures; it may say how to tell a flake from a real regression (rerun it)."
        ),
        context="A rule that switched the test off would hide a real regression in it from every later session.",
    )


async def test_a_used_note_the_agent_acted_against_becomes_what_the_agent_did():
    draft = await _draft(
        note="Before you change transactions.go: Count transfers to piva account as tax already set aside",
        verdict="used",
        reason="I removed the shortcut that counted money moved to the tax pot as already set aside. With the "
        "Salvadanaio loaded, the dashboard now uses its real balance.",
        goal="access fiscozen to understand my tax situation",
    )
    await assert_meets_criteria(
        response=f"create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria=(
            "The rule says NOT to count transfers to the piva/tax account as tax already set aside, and to use the "
            "real Salvadanaio balance instead. A rule that says to count those transfers as set aside fails."
        ),
        context="The agent used the note to remove that shortcut, so the note as written is no longer true.",
    )


async def _correction(agent: str, user: str, asked: str) -> L.CorrectionDraft:
    # Built as service._learn_correction builds it.
    prompt = L.CORRECTION_PROMPT.format(agent=agent, user=user, goal=asked, existing=L.existing_rules_text([]))
    result = await LLMConfig.from_env().call(
        messages=[{"role": "user", "content": prompt}],
        response_format=L.CorrectionDraft,
        scope="session_learn",
        strict_schema=True,
    )
    c = result.content
    return c if isinstance(c, L.CorrectionDraft) else L.CorrectionDraft.model_validate(c)


async def test_a_users_correction_of_a_value_becomes_a_rule():
    # Live: the user had to say it, though memory had the price since Oct 22.
    draft = await _correction(
        agent='I read "0.4% per M" as **$0.40 per 1M tokens** for typesafe; at that price the forwarded calls '
        "flip the margin to negative.",
        user="it's 4 cent, not 40",
        asked="Compute customer COGS and margin, and include typesafe in the table",
    )
    await assert_meets_criteria(
        response=f"corrects={draft.corrects} create={draft.create}\nname: {draft.name}\ncontent: {draft.content}",
        criteria=(
            "corrects is True and create is True. The rule states that typesafe costs $0.04 (4 cents) per 1M "
            "tokens, and never states $0.40 as the price."
        ),
        context="The agent priced typesafe at $0.40 per 1M tokens; the user corrected it to 4 cents.",
    )


async def test_a_users_next_request_is_no_correction():
    # Asked only "create?", the writer made rules of replies like this one (239 from 298 replies).
    draft = await _correction(
        agent="The knowledge bank page now has the Documents list and a Model page with the schema.",
        user="we need to rename Knowledge page to Documents and Model page to Structured Data for now",
        asked="Build the knowledge bank UI",
    )
    await assert_meets_criteria(
        response=f"corrects={draft.corrects} create={draft.create}\ncontent: {draft.content}",
        criteria="corrects is False or create is False: the reply is a new request, not a correction of the agent.",
        context="The user asked for a rename after the agent built the pages; the agent got nothing wrong.",
    )
