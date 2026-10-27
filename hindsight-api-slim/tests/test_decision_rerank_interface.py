"""decision_rerank_interface is loaded by path outside hindsight-api, so it must stand alone."""

import subprocess
import sys

from hindsight_api.engine import decision_rerank_interface

_LOAD_ALONE = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("decision_rerank_interface", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rerank = module.DecisionRerank(count=lambda t: len(t.split()), truncate=lambda t, n: " ".join(t.split()[:n]))
questions = rerank.questions(["a", "b"])
assert [q.key for q in questions] == ["m0", "m1"], questions
assert rerank.split("m", rerank.state("q"), {q.key: q.system_one() for q in questions}) == [["m0", "m1"]]
assert rerank.scores([None, 0.9]) == [0.5, 1.0]
leaked = sorted(name for name in sys.modules if name.split(".")[0] in {"hindsight_api", "toktok", "pydantic"})
assert not leaked, leaked
"""


def test_it_loads_by_path_with_the_standard_library_only():
    subprocess.run([sys.executable, "-I", "-c", _LOAD_ALONE, decision_rerank_interface.__file__], check=True)


def test_the_tokenizer_is_the_callers():
    words = decision_rerank_interface.DecisionRerank(
        count=lambda t: len(t.split()), truncate=lambda t, n: " ".join(t.split()[:n]), max_request_tokens=40
    )
    # A quarter of a 40-token request: 10 words of memory, whatever the cap says.
    (question,) = words.questions(["word " * 50])
    assert question.memory == " ".join(["word"] * 10)


def test_pruning_without_sending_anything():
    """What an outside caller runs on the answers: the no-help ones at exactly 0.0."""
    words = decision_rerank_interface.DecisionRerank(count=lambda t: len(t.split()), truncate=lambda t, n: t)
    assert words.pruned([0.9, 0.1, 0.8]) == [1.0, 0.0, 2 / 3]
    # Below the floor, the best still stays.
    assert words.pruned([0.05, 0.01]) == [1.0, 0.0]
