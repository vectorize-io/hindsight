"""Blackbox quality evals: a real server, a real model, an independent judge.

The sibling of ``hindsight-system-tests``, and deliberately a separate package.
Those tests stub every LLM call so they are deterministic and need no secrets —
which is exactly what an eval cannot do, because a stubbed model means scoring
the stub. So these need provider credentials, cannot run on fork PRs, and report
RATES rather than equalities.

Read ``README.md`` for the two modes and what each is for.
"""

from hindsight_system_evals.answers import AnswerOutcome, ask, seed_bank
from hindsight_system_evals.judge import Verdict, evaluate, judge_model
from hindsight_system_evals.pages import PageOutcome, build_page, facts, questions, split_into_waves
from hindsight_system_evals.server import EvalServer, provider_environment, start_eval_server
from hindsight_system_evals.waiting import wait_until_settled

__all__ = [
    "AnswerOutcome",
    "EvalServer",
    "PageOutcome",
    "Verdict",
    "ask",
    "build_page",
    "evaluate",
    "facts",
    "judge_model",
    "provider_environment",
    "questions",
    "seed_bank",
    "split_into_waves",
    "start_eval_server",
    "wait_until_settled",
]
