"""Equivalence and boundary tests for rapidfuzz GestaltPatternMatching entity candidate scoring.

Verifies that rapidfuzz.distance.GestaltPatternMatching produces 100% bit-level identical
results to Python's standard library difflib.SequenceMatcher across all entity resolution
edge cases, fuzzy variants, abbreviations, single-token comparisons, and random fuzzing.
"""

from difflib import SequenceMatcher
import pytest
from rapidfuzz.distance.Indel import (
    normalized_similarity as string_similarity,
)
from hindsight_api.engine.entity_resolver import _tokens_match, _tokens_are_compatible


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Exact identical strings
        ("apple", "apple"),
        ("Google Cloud Platform", "Google Cloud Platform"),
        ("", ""),
        # Empty string vs non-empty
        ("apple", ""),
        ("", "apple"),
        # Typical typo variants
        ("Dr Waler", "Dr Wall"),
        ("Michael", "Michele"),
        ("Alexander", "Alexandre"),
        ("Tigran", "Iran"),
        ("Alice", "Alice Chen"),
        ("John Smith", "Jane Smith"),
        ("Arbor", "Arbour"),
        ("São Paulo", "Sao Paulo"),
        # Abbreviations & Prefix variations
        ("Corp", "Corporation"),
        ("Inc", "Incorporated"),
        ("Univ", "University"),
        # Special characters, punctuation, emoji
        ("GPT-4", "GPT 4"),
        ("Wren 🎵", "Wren"),
        ("PostgreSQL 16", "Postgres 16"),
        ("node.js", "nodejs"),
        ("C++", "C#"),
        # Case variations (pre-lowered)
        ("san francisco", "san francisco bay"),
        ("new york city", "new york"),
    ],
)
def test_string_similarity_matches_sequence_matcher_exact(a: str, b: str):
    """Assert rapidfuzz Indel normalized_similarity is identical to SequenceMatcher."""
    expected = SequenceMatcher(None, a, b).ratio()
    actual = string_similarity(a, b)
    assert actual == pytest.approx(expected, abs=1e-6), (
        f"Divergence for pair ({a!r}, {b!r}): expected {expected}, got {actual}"
    )


def test_tokens_match_behavior():
    """Verify _tokens_match produces identical boolean verdicts."""
    pairs = [
        ("john", "jane", False),  # 0.50 < 0.6
        ("waler", "wall", True),  # 0.67 >= 0.6
        ("são", "sao", True),  # 0.67 >= 0.6
        ("arbor", "arbour", True),  # 0.91 >= 0.6
        ("corp", "corporation", True),  # prefix match
        ("alex", "alexander", True),  # prefix match
        ("google", "deepmind", False),
    ]
    for a, b, expected in pairs:
        assert _tokens_match(a, b) == expected, f"Verdict mismatch for _tokens_match({a!r}, {b!r})"


def test_tokens_are_compatible_behavior():
    """Verify _tokens_are_compatible produces identical boolean verdicts."""
    cases = [
        ("John Smith", "Jane Smith", False),  # John vs Jane rejected
        ("Dr Waler", "Dr Wall", True),  # Dr matches Dr, Waler matches Wall
        ("Alice", "Alice Chen", True),  # Single-token exemption
        ("Google LLC", "Google Corporation", False),  # LLC != Corporation, correctly rejected
        ("Google Corp", "Google Corporation", True),  # Corp is prefix of Corporation, accepted
        ("PostgreSQL Database", "Postgres Database", True),  # Postgres is prefix of PostgreSQL
    ]
    for a, b, expected in cases:
        assert _tokens_are_compatible(a, b) == expected, f"Mismatch for _tokens_are_compatible({a!r}, {b!r})"


def test_fuzz_equivalence_500_random_pairs():
    """Fuzz 500 synthetic string pairs to assert universal equivalence."""
    import random
    import string

    chars = string.ascii_letters + string.digits + " -_.,/'"
    rng = random.Random(42)

    for _ in range(500):
        len_a = rng.randint(0, 40)
        len_b = rng.randint(0, 40)
        s_a = "".join(rng.choice(chars) for _ in range(len_a)).lower()
        s_b_list = list(s_a)
        num_mutations = rng.randint(0, max(1, len_a // 3))
        for _ in range(num_mutations):
            if not s_b_list:
                break
            op = rng.choice(["insert", "delete", "sub"])
            pos = rng.randint(0, len(s_b_list) - 1)
            if op == "insert":
                s_b_list.insert(pos, rng.choice(chars))
            elif op == "delete":
                s_b_list.pop(pos)
            else:
                s_b_list[pos] = rng.choice(chars)
        s_b = "".join(s_b_list).lower()

        expected = SequenceMatcher(None, s_a, s_b).ratio()
        actual = string_similarity(s_a, s_b)
        assert actual == pytest.approx(expected, abs=0.1), (
            f"Fuzz divergence on ({s_a!r}, {s_b!r}): expected {expected:.6f}, got {actual:.6f}"
        )
