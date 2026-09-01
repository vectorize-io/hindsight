"""Tests for `query_tag_gate` — restricting recall to the identities a query names.

The matching itself is pure and deterministic, so it is asserted directly here rather
than through a fixture. The separation matters: what the gate decides must not depend
on embeddings or on a reranker being present.
"""

import pytest

from hindsight_api.engine.response_models import QueryTagGate
from hindsight_api.engine.search.query_tag_gate import is_typo_of, match_tags, tokenize

VOCAB = [
    "name:kubernetes",
    "name:k8s",
    "name:mongo",
    "name:error-handling",
    "name:typescript",
    "name:session-backend",
    "other:kubernetes",
]


class TestTokenize:
    def test_splits_on_punctuation_and_lowercases(self):
        assert tokenize("What's my K8s deploy story?") == ["what", "s", "my", "k8s", "deploy", "story"]

    def test_empty_and_none_are_empty(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestIsTypoOf:
    @pytest.mark.parametrize(
        ("typed", "intended"),
        [
            ("error", "error"),  # identical
            ("eror", "error"),  # deletion
            ("moongoose", "mongoose"),  # insertion
            ("typsecript", "typescript"),  # transposition
            ("mongod", "mongodb"),  # trailing deletion
        ],
    )
    def test_accepts_one_slip_of_the_same_word(self, typed, intended):
        assert is_typo_of(typed, intended)

    @pytest.mark.parametrize(
        ("typed", "intended"),
        [
            ("mango", "mongo"),  # substitution — a different word at the same distance
            ("k9s", "k8s"),  # substitution of a digit
            ("postgres", "mongodb"),  # unrelated
            ("eor", "error"),  # two deletions — beyond one slip
        ],
    )
    def test_rejects_substitutions_and_larger_distances(self, typed, intended):
        assert not is_typo_of(typed, intended)

    @pytest.mark.parametrize("typed", ["erorr", "errro"])
    def test_transposed_letters_are_still_the_same_word(self, typed):
        """Both are `error` with one adjacent swap — a slip, not a different word."""
        assert is_typo_of(typed, "error")

    def test_substitution_and_typo_are_indistinguishable_by_distance(self):
        """Why the rule is about the *kind* of edit, not a threshold.

        'mango'->'mongo' and 'eror'->'error' are both one edit away from their target, so no
        distance cutoff can accept the typo and reject the different word. Only excluding
        substitutions separates them.
        """
        assert is_typo_of("eror", "error")
        assert not is_typo_of("mango", "mongo")


class TestMatchTags:
    def test_matches_the_named_identity_only(self):
        assert match_tags("what's my k8s deploy story?", VOCAB, prefix="name:") == ["name:k8s"]

    def test_ignores_tags_outside_the_prefix(self):
        """`other:kubernetes` is in the bank but not in the gated namespace."""
        assert match_tags("kubernetes rollout", VOCAB, prefix="name:") == ["name:kubernetes"]

    def test_matches_multi_word_names_as_a_contiguous_phrase(self):
        assert match_tags("session backend design", VOCAB, prefix="name:") == ["name:session-backend"]

    def test_does_not_match_a_phrase_split_across_the_query(self):
        assert match_tags("the session is fine but the backend is not", VOCAB, prefix="name:") == []

    def test_typos_are_matched_by_default(self):
        assert match_tags("eror handling patterns", VOCAB, prefix="name:") == ["name:error-handling"]
        assert match_tags("typsecript compilation", VOCAB, prefix="name:") == ["name:typescript"]

    def test_exact_mode_rejects_typos(self):
        assert match_tags("eror handling patterns", VOCAB, prefix="name:", match="exact") == []
        assert match_tags("error handling patterns", VOCAB, prefix="name:", match="exact") == ["name:error-handling"]

    def test_a_different_word_is_not_a_match(self):
        """The prefix guards: neither of these names anything in the vocabulary."""
        assert match_tags("mango connection pooling", VOCAB, prefix="name:") == []
        assert match_tags("k9s pod scheduling", VOCAB, prefix="name:") == []

    def test_short_tokens_must_match_exactly(self):
        """`k8s` is three characters — fuzzy matching there would match almost anything."""
        assert match_tags("k8s cluster", VOCAB, prefix="name:", min_token_length=4) == ["name:k8s"]
        assert match_tags("k9s cluster", VOCAB, prefix="name:", min_token_length=4) == []

    def test_naming_nothing_known_matches_nothing(self):
        assert match_tags("what is the weather in Lisbon", VOCAB, prefix="name:") == []

    def test_empty_query_matches_nothing(self):
        assert match_tags("", VOCAB, prefix="name:") == []
        assert match_tags("   ", VOCAB, prefix="name:") == []

    def test_returns_every_named_identity(self):
        matched = match_tags("kubernetes and typescript", VOCAB, prefix="name:")
        assert set(matched) == {"name:kubernetes", "name:typescript"}


class TestQueryTagGateModel:
    def test_defaults_are_typo_tolerant_and_abstaining(self):
        gate = QueryTagGate(prefix="name:")
        assert gate.match == "typos"
        assert gate.on_no_match == "abstain"
        assert gate.min_token_length == 4

    def test_prefix_is_required(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            QueryTagGate()

    @pytest.mark.parametrize(("field", "value"), [("min_token_length", 0), ("max_vocabulary", 0)])
    def test_rejects_non_positive_limits(self, field, value):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            QueryTagGate(prefix="name:", **{field: value})
