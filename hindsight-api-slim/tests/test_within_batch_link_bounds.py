"""Within-batch link generation stays bounded, and keeps picking the same links (#3848).

Both passes below used to build every pair of the batch before narrowing to the
handful per unit that survive. A streaming sub-batch (~1.7K facts) made that
merely wasteful; delta retain hands over a whole document's changed chunks in one
call, and at 36K facts the temporal pass OOM-killed the worker while the semantic
pass blocked the event loop for minutes.

The rewrites are pinned here against verbatim copies of what they replaced. The
copies must stay verbatim: the moment a reference shares code with the live
implementation it stops proving anything.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from hindsight_api.engine.retain import link_utils
from hindsight_api.engine.retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    _cap_links_per_unit,
    _normalize_datetime,
    _within_batch_temporal_links,
    compute_semantic_links_within_batch,
)

T0 = datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Verbatim pre-#3848 implementations. Do not refactor, do not share code with
# the live ones.
# --------------------------------------------------------------------------


def _legacy_within_batch_temporal_links(new_units: dict, time_window_hours: int = 24) -> list[tuple]:
    links: list[tuple] = []
    new_unit_items = list(new_units.items())
    for i, (unit_id, (event_date, fact_type)) in enumerate(new_unit_items):
        if event_date is None:
            continue
        unit_event_date_norm = _normalize_datetime(event_date)

        for j in range(i + 1, len(new_unit_items)):
            other_id, (other_event_date, other_fact_type) = new_unit_items[j]
            if other_event_date is None:
                continue
            if fact_type != other_fact_type:
                continue
            other_event_date_norm = _normalize_datetime(other_event_date)

            time_diff_hours = abs((unit_event_date_norm - other_event_date_norm).total_seconds() / 3600)
            if time_diff_hours <= time_window_hours:
                weight = max(0.3, 1.0 - (time_diff_hours / time_window_hours))
                links.append((unit_id, other_id, "temporal", weight, None))
                links.append((other_id, unit_id, "temporal", weight, None))
    return links


def _legacy_semantic_links_within_batch(unit_ids, embeddings, top_k=50, *, threshold) -> list[tuple]:
    if len(unit_ids) < 2:
        return []

    links = []
    new_embeddings_matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(new_embeddings_matrix, axis=1)
    valid_embeddings = np.isfinite(new_embeddings_matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
    normalized_embeddings = np.zeros_like(new_embeddings_matrix)
    normalized_embeddings[valid_embeddings] = (
        new_embeddings_matrix[valid_embeddings] / norms[valid_embeddings, np.newaxis]
    )

    for i, unit_id in enumerate(unit_ids):
        if not valid_embeddings[i]:
            continue

        other_indices = [j for j in range(len(unit_ids)) if j != i]
        if not other_indices:
            continue

        other_embeddings = normalized_embeddings[other_indices]
        similarities = np.dot(other_embeddings, normalized_embeddings[i])
        similarities[~valid_embeddings[other_indices]] = -np.inf

        above_threshold = np.where(similarities >= threshold)[0]
        if len(above_threshold) > 0:
            sorted_local_indices = above_threshold[np.argsort(-similarities[above_threshold])][:top_k]
            for local_idx in sorted_local_indices:
                other_idx = other_indices[local_idx]
                other_id = unit_ids[other_idx]
                similarity = float(min(1.0, max(0.0, similarities[local_idx])))
                links.append((unit_id, other_id, "semantic", similarity, None))

    return links


def _units(spec: list[tuple[float, str]], prefix: str = "u") -> dict:
    """Build a new_units mapping from (hours_offset, fact_type) pairs."""
    return {f"{prefix}{i}": (T0 + timedelta(hours=offset), fact_type) for i, (offset, fact_type) in enumerate(spec)}


def _by_source(links: list[tuple]) -> dict[str, list[tuple]]:
    grouped: dict[str, list[tuple]] = {}
    for link in links:
        grouped.setdefault(str(link[0]), []).append(link)
    return grouped


class TestWithinBatchTemporalLinks:
    def test_small_batch_is_identical_to_the_all_pairs_sweep(self):
        """Under the per-unit cap the window covers everything, so nothing may differ."""
        units = _units([(i * 0.5, "world") for i in range(MAX_TEMPORAL_LINKS_PER_UNIT)])

        assert sorted(_within_batch_temporal_links(units, 24)) == sorted(_legacy_within_batch_temporal_links(units, 24))

    def test_keeps_the_same_links_the_cap_would_have_kept(self):
        """41 units, 0.4h apart: every gap distinct, none clamped, so the choice is determined."""
        units = _units([(i * 0.4, "world") for i in range(41)])

        new = _cap_links_per_unit(_within_batch_temporal_links(units, 24))
        legacy = _cap_links_per_unit(_legacy_within_batch_temporal_links(units, 24))

        assert sorted(new) == sorted(legacy)

    def test_a_unit_keeps_neighbours_on_both_sides(self):
        """The reverse links are what make a unit's predecessors reachable at all.

        A per-unit budget spent during generation instead of on candidates puts
        every one of these on the earlier side.
        """
        units = _units([(i * 0.4, "world") for i in range(41)])
        middle = "u20"

        kept = _cap_links_per_unit(_within_batch_temporal_links(units, 24))
        targets = {str(link[1]) for link in kept if str(link[0]) == middle}

        assert {t for t in targets if int(t[1:]) < 20}
        assert {t for t in targets if int(t[1:]) > 20}

    @pytest.mark.parametrize("seed", range(25))
    def test_weights_match_the_all_pairs_sweep_on_random_batches(self, seed):
        """Exact on every batch, ties included: the cap ranks by weight, and the
        bounded candidate set contains the highest-weighted ones by construction."""
        rng = random.Random(seed)
        units = _units(
            [(rng.uniform(0, 48), rng.choice(["world", "experience"])) for _ in range(rng.randint(2, 120))],
        )

        new = _by_source(_cap_links_per_unit(_within_batch_temporal_links(units, 24)))
        legacy = _by_source(_cap_links_per_unit(_legacy_within_batch_temporal_links(units, 24)))

        assert new.keys() == legacy.keys()
        for unit_id, links in new.items():
            assert sorted(link[3] for link in links) == sorted(link[3] for link in legacy[unit_id])

    def test_candidates_per_unit_are_bounded(self):
        """The bound the OOM was about: candidates per unit, not links after the cap."""
        n = 5000
        units = _units([(0.0, "world")] * n)  # every pair inside the window — the worst case

        links = _within_batch_temporal_links(units, 24)

        assert len(links) <= 2 * MAX_TEMPORAL_LINKS_PER_UNIT * n
        assert all(len(group) <= 2 * MAX_TEMPORAL_LINKS_PER_UNIT for group in _by_source(links).values())
        # What the all-pairs sweep would have built for the same batch.
        assert len(links) < 0.01 * n * (n - 1)

    def test_units_outside_the_window_do_not_link(self):
        units = _units([(0.0, "world"), (1.0, "world"), (30.0, "world")])

        targets = {(str(link[0]), str(link[1])) for link in _within_batch_temporal_links(units, 24)}

        assert ("u0", "u1") in targets
        assert ("u0", "u2") not in targets
        assert ("u1", "u2") not in targets

    def test_only_same_fact_type_links(self):
        units = _units([(0.0, "world"), (0.5, "experience"), (1.0, "world")])

        targets = {(str(link[0]), str(link[1])) for link in _within_batch_temporal_links(units, 24)}

        assert targets == {("u0", "u2"), ("u2", "u0")}

    def test_units_without_an_event_date_are_skipped(self):
        units = {"u0": (T0, "world"), "u1": (None, "world"), "u2": (T0 + timedelta(hours=1), "world")}

        targets = {(str(link[0]), str(link[1])) for link in _within_batch_temporal_links(units, 24)}

        assert targets == {("u0", "u2"), ("u2", "u0")}

    def test_naive_and_aware_event_dates_compare(self):
        units = {"u0": (T0.replace(tzinfo=None), "world"), "u1": (T0 + timedelta(hours=1), "world")}

        assert len(_within_batch_temporal_links(units, 24)) == 2


class TestWithinBatchSemanticLinks:
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_the_per_unit_implementation(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(2, 60)
        embeddings = rng.normal(size=(n, 16))
        unit_ids = [f"u{i}" for i in range(n)]

        new = compute_semantic_links_within_batch(unit_ids, embeddings, top_k=8, threshold=0.1)
        legacy = _legacy_semantic_links_within_batch(unit_ids, embeddings, top_k=8, threshold=0.1)

        assert [(link[0], link[1], link[2]) for link in new] == [(link[0], link[1], link[2]) for link in legacy]
        assert [link[3] for link in new] == pytest.approx([link[3] for link in legacy], abs=1e-12)

    def test_matches_across_block_boundaries(self, monkeypatch):
        """Blocking must not change which units are compared, only when."""
        monkeypatch.setattr(link_utils, "_SEMANTIC_WITHIN_BATCH_BLOCK_ROWS", 3)
        rng = np.random.default_rng(99)
        embeddings = rng.normal(size=(11, 8))
        unit_ids = [f"u{i}" for i in range(11)]

        new = compute_semantic_links_within_batch(unit_ids, embeddings, top_k=5, threshold=0.0)
        legacy = _legacy_semantic_links_within_batch(unit_ids, embeddings, top_k=5, threshold=0.0)

        assert [(link[0], link[1]) for link in new] == [(link[0], link[1]) for link in legacy]
        assert [link[3] for link in new] == pytest.approx([link[3] for link in legacy], abs=1e-12)

    def test_invalid_embeddings_stay_excluded_across_blocks(self, monkeypatch):
        monkeypatch.setattr(link_utils, "_SEMANTIC_WITHIN_BATCH_BLOCK_ROWS", 2)
        embeddings = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [float("nan"), 1.0], [1.0, 0.0]]
        unit_ids = ["a", "b", "zero", "nan", "c"]

        links = compute_semantic_links_within_batch(unit_ids, embeddings, threshold=0.0)

        assert not [link for link in links if "zero" in (link[0], link[1])]
        assert not [link for link in links if "nan" in (link[0], link[1])]

    def test_never_links_a_unit_to_itself(self, monkeypatch):
        monkeypatch.setattr(link_utils, "_SEMANTIC_WITHIN_BATCH_BLOCK_ROWS", 2)
        embeddings = [[1.0, 0.0]] * 7
        unit_ids = [f"u{i}" for i in range(7)]

        links = compute_semantic_links_within_batch(unit_ids, embeddings, threshold=0.0)

        assert not [link for link in links if link[0] == link[1]]
