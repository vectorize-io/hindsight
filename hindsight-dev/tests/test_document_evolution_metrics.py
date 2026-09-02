"""Unit tests for the document-evolution benchmark's damage detector.

The detector decides whether a release is safe, so it gets the same scrutiny as
the code it measures. Two failure modes would quietly invalidate a comparison: a
detector that misses the damage it exists to catch, and one that reports an
edit the model made on purpose as corruption.
"""

from __future__ import annotations

from benchmarks.document_evolution.metrics import (
    compare_shape,
    damage_in_untouched_sections,
    describe,
    untouched_sections_drifted,
)

WELL_FORMED = """## Environments

| Environment | Region |
| --- | --- |
| staging | eu-west-1 |
| production | eu-west-1 |

- top
  - nested

Contact the platform team  
via their channel.

> Provisioned from infrastructure-as-code.
"""

# The same document after a markdown round-trip flattened it: the table is one
# physical line, the nesting is gone and the hard line break is gone.
COLLAPSED = """## Environments

| Environment | Region | | --- | --- | | staging | eu-west-1 | | production | eu-west-1 |

- top
- nested

Contact the platform team via their channel.

> Provisioned from infrastructure-as-code.
"""


class TestDescribe:
    def test_counts_the_constructs_that_break(self):
        shape = describe(WELL_FORMED)
        assert shape.table_rows == 4
        assert shape.indented_list_lines == 1
        assert shape.hard_break_lines == 1
        assert shape.blockquote_lines == 1
        assert shape.collapsed_tables == 0

    def test_detects_a_welded_table(self):
        assert describe(COLLAPSED).collapsed_tables == 1

    def test_a_separator_row_alone_is_not_a_collapse(self):
        assert describe("| a | b |\n| --- | --- |\n| 1 | 2 |\n").collapsed_tables == 0

    def test_aligned_separator_is_not_a_collapse(self):
        assert describe("| a | b |\n|:---|---:|\n| 1 | 2 |\n").collapsed_tables == 0

    def test_fences_are_counted_and_balanced(self):
        shape = describe("```python\nx = 1\n```\n")
        assert shape.fenced_blocks == 1
        assert not shape.unbalanced_fence

    def test_unbalanced_fence_is_reported(self):
        assert describe("```python\nx = 1\n").unbalanced_fence

    def test_table_syntax_inside_a_fence_is_not_a_table(self):
        assert describe("```\n| a | b |\n| --- | --- |\n```\n").table_rows == 0


class TestCompareShape:
    def test_reports_every_loss(self):
        delta = compare_shape(describe(WELL_FORMED), describe(COLLAPSED))
        assert delta.collapsed_tables_introduced == 1
        assert delta.table_rows_lost == 3
        assert delta.indented_list_lines_lost == 1
        assert delta.hard_break_lines_lost == 1
        assert delta.damaged

    def test_growth_is_not_damage(self):
        grown = WELL_FORMED.replace("| production | eu-west-1 |", "| production | eu-west-1 |\n| canary | eu-west-1 |")
        assert not compare_shape(describe(WELL_FORMED), describe(grown)).damaged

    def test_identical_documents_are_undamaged(self):
        assert not compare_shape(describe(WELL_FORMED), describe(WELL_FORMED)).damaged


class TestDamageAttribution:
    """Damage is only damage in a section nobody asked to change."""

    def test_loss_in_an_untouched_section_is_damage(self):
        assert damage_in_untouched_sections(WELL_FORMED, COLLAPSED, set()).damaged

    def test_loss_in_a_targeted_section_is_an_edit_not_damage(self):
        delta = damage_in_untouched_sections(WELL_FORMED, COLLAPSED, {"## Environments"})
        assert not delta.damaged

    def test_damage_elsewhere_still_counts_when_another_section_was_edited(self):
        before = "## A\n\n| x | y |\n| --- | --- |\n| 1 | 2 |\n\n## B\n\nprose\n"
        after = "## A\n\n| x | y | | --- | --- | | 1 | 2 |\n\n## B\n\nrewritten prose\n"
        assert damage_in_untouched_sections(before, after, {"## B"}).damaged


class TestDrift:
    def test_untouched_section_that_changed_is_drift(self):
        before = "## A\n\none\n\n## B\n\ntwo\n"
        after = "## A\n\none\n\n## B\n\ntwo changed\n"
        assert untouched_sections_drifted(before, after, set()) == ["## B"]

    def test_touched_section_that_changed_is_not_drift(self):
        before = "## A\n\none\n\n## B\n\ntwo\n"
        after = "## A\n\none\n\n## B\n\ntwo changed\n"
        assert untouched_sections_drifted(before, after, {"## B"}) == []

    def test_a_removed_section_is_not_reported_as_drift(self):
        """Removal is a content decision the coverage judge scores, not structural drift."""
        before = "## A\n\none\n\n## B\n\ntwo\n"
        after = "## A\n\none\n"
        assert untouched_sections_drifted(before, after, set()) == []

    def test_added_sections_do_not_disturb_the_others(self):
        before = "## A\n\none\n"
        after = "## A\n\none\n\n## New\n\nadded\n"
        assert untouched_sections_drifted(before, after, set()) == []
