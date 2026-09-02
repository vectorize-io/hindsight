"""Unit tests for the structured document schema, renderer, splitter, and
delta-operation applicator.

These tests are pure-Python (no DB, no LLM) and run fast. They guard the
mechanical guarantees that the structured-delta architecture relies on:

- Deterministic rendering (same input → same bytes).
- ``split_markdown`` is *lossless*: every non-blank line of the input comes
  back verbatim and in order, whatever markdown construct it belongs to. This
  is the guarantee v1's typed parser did not have — it flattened anything it
  could not classify onto one line, permanently (#3361).
- Section and block IDs are stable and unique.
- Operations target sections and blocks by id and never silently corrupt the
  document: an id that does not resolve is dropped and reported, never applied
  to whatever happens to be nearby (#3273).
- Sections and blocks not mentioned by any op come through byte-identical.
"""

from __future__ import annotations

import pytest

from hindsight_api.engine.reflect.delta_ops import (
    AddSectionOp,
    AppendBlockOp,
    InsertBlockOp,
    RemoveBlockOp,
    RemoveSectionOp,
    RenameSectionOp,
    ReplaceBlockOp,
    ReplaceSectionBlocksOp,
    apply_operations,
)
from hindsight_api.engine.reflect.structured_doc import (
    Block,
    Section,
    StructuredDocument,
    make_block_id,
    make_unique_id,
    normalize_block_text,
    render_document,
    render_section,
    slugify_heading,
    split_markdown,
    structured_document_from_stored,
)

# Markdown corpus ------------------------------------------------------------
#
# Every construct here was destroyed by the v1 parser (see #3361 and the
# investigation that preceded this rewrite). They are kept together because the
# fidelity guarantee is about *all* markdown, not about the handful of shapes
# the old block union happened to model.

MARKDOWN_CORPUS: dict[str, str] = {
    "paragraph": "## S\n\nA sentence.\n",
    "multi_line_paragraph": "## S\n\nA sentence.\nAnother sentence.\n",
    "hard_line_break": "## S\n\nline one  \nline two\n",
    "bullets": "## S\n\n- one\n- two\n",
    "nested_bullets": "## S\n\n- top\n  - child\n    - grandchild\n- second\n",
    "bullet_continuation": "## S\n\n- item one\n  continued line of item one\n- item two\n",
    "ordered_from_five": "## S\n\n5. five\n6. six\n",
    "ordered_paren": "## S\n\n1) one\n2) two\n",
    "mixed_list": "## S\n\n- bullet\n1. ordered\n",
    "task_list": "## S\n\n- [ ] todo\n- [x] done\n",
    "blockquote": "## S\n\n> quoted line one\n> quoted line two\n",
    "horizontal_rule": "## S\n\npara a\n\n---\n\npara b\n",
    "setext_heading": "Title\n=====\n\nbody\n",
    "html_block": "## S\n\n<details>\n<summary>x</summary>\ntext\n</details>\n",
    "indented_code": "## S\n\n    def f():\n        return 1\n",
    "fenced_code": '## S\n\n```python\ndef f():\n\n    return {"a": 1}\n```\n',
    "fenced_code_with_heading": "## S\n\n```\n# not a heading\n```\n",
    "table": "## S\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n",
    "table_aligned": "## S\n\n| a | b |\n|:--|--:|\n| 1 | 2 |\n",
    "table_no_trailing_pipe": "## S\n\n| a | b |\n|---|---|\n| 1 | 2\n",
    "table_no_leading_pipe": "## S\n\n| a | b |\n|---|---|\na | 1 | 2 |\n",
    "table_single_dash_separator": "## S\n\n| a | b |\n| - | - |\n| 1 | 2 |\n",
    "table_pipe_in_cell": "## S\n\n| a | b |\n| --- | --- |\n| `x \\| y` | 2 |\n",
    "preamble": "Intro prose.\n\n## S\n\nbody\n",
    "many_sections": "# T\n\nlead\n\n## A\n\na body\n\n### B\n\nb body\n",
    "unicode": "## 概要\n\n日本語のテキスト。\n",
}


def _significant_lines(markdown: str) -> list[str]:
    """Non-blank lines, right-stripped of nothing — compared verbatim."""
    return [line for line in markdown.splitlines() if line.strip()]


# Slug / ids -----------------------------------------------------------------


class TestIds:
    def test_basic(self):
        assert slugify_heading("Purpose") == "purpose"

    def test_multi_word(self):
        assert slugify_heading("Stop Conditions") == "stop-conditions"

    def test_punctuation_collapses(self):
        assert slugify_heading("Inputs / Context !") == "inputs-context"

    def test_unicode_falls_back(self):
        assert slugify_heading("???") == "section"

    def test_make_unique_id_no_collision(self):
        assert make_unique_id("rules", set()) == "rules"

    def test_make_unique_id_collision(self):
        assert make_unique_id("rules", {"rules"}) == "rules-2"
        assert make_unique_id("rules", {"rules", "rules-2"}) == "rules-3"

    def test_block_id_is_deterministic(self):
        assert make_block_id("hello", set()) == make_block_id("hello", set())

    def test_block_id_differs_by_content(self):
        assert make_block_id("hello", set()) != make_block_id("goodbye", set())

    def test_block_id_disambiguates_identical_text(self):
        first = make_block_id("same", set())
        assert make_block_id("same", {first}) == f"{first}-2"

    def test_identical_blocks_get_distinct_ids(self):
        doc = split_markdown("## S\n\nrepeated\n\nrepeated\n")
        ids = [b.id for b in doc.sections[0].blocks]
        assert len(ids) == len(set(ids)) == 2


# Renderer -------------------------------------------------------------------


class TestRenderer:
    def test_section_with_heading(self):
        section = Section(id="s", heading="Title", level=3, blocks=[Block(id="b1", text="body")])
        assert render_section(section) == "### Title\n\nbody"

    def test_section_without_heading_renders_bare_blocks(self):
        section = Section(id="preamble", heading="", blocks=[Block(id="b1", text="body")])
        assert render_section(section) == "body"

    def test_blocks_are_blank_line_separated(self):
        section = Section(
            id="s",
            heading="T",
            blocks=[Block(id="b1", text="one"), Block(id="b2", text="two")],
        )
        assert render_section(section) == "## T\n\none\n\ntwo"

    def test_block_text_is_rendered_verbatim(self):
        table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        section = Section(id="s", heading="T", blocks=[Block(id="b1", text=table)])
        assert table in render_section(section)

    def test_empty_document_renders_empty(self):
        assert render_document(StructuredDocument()) == ""

    def test_document_ends_with_single_newline(self):
        doc = split_markdown("## S\n\nbody\n")
        assert render_document(doc).endswith("body\n")

    def test_render_is_deterministic(self):
        doc = split_markdown(MARKDOWN_CORPUS["many_sections"])
        assert render_document(doc) == render_document(doc)

    def test_empty_blocks_are_not_rendered(self):
        section = Section(id="s", heading="T", blocks=[Block(id="b1", text="   ")])
        assert render_section(section) == "## T"


class TestNormalizeBlockText:
    def test_strips_blank_edges(self):
        assert normalize_block_text("\n\nbody\n\n") == "body"

    def test_keeps_hard_line_break_spaces(self):
        assert normalize_block_text("line one  \nline two") == "line one  \nline two"

    def test_keeps_leading_indentation(self):
        assert normalize_block_text("    indented code") == "    indented code"


# Splitter fidelity ----------------------------------------------------------


class TestSplitFidelity:
    """The core guarantee: splitting never loses or rewrites content."""

    @pytest.mark.parametrize("name", sorted(MARKDOWN_CORPUS))
    def test_every_significant_line_survives_verbatim(self, name: str):
        markdown = MARKDOWN_CORPUS[name]
        rendered = render_document(split_markdown(markdown))
        assert _significant_lines(rendered) == _significant_lines(markdown)

    @pytest.mark.parametrize("name", sorted(MARKDOWN_CORPUS))
    def test_render_of_split_is_idempotent(self, name: str):
        markdown = MARKDOWN_CORPUS[name]
        once = render_document(split_markdown(markdown))
        twice = render_document(split_markdown(once))
        assert once == twice

    @pytest.mark.parametrize("name", sorted(MARKDOWN_CORPUS))
    def test_canonical_markdown_round_trips_byte_for_byte(self, name: str):
        """Our own render is a fixed point of the splitter."""
        markdown = MARKDOWN_CORPUS[name]
        canonical = render_document(split_markdown(markdown))
        doc = split_markdown(canonical)
        assert render_document(doc) == canonical
        assert [b.text for s in doc.sections for b in s.blocks] == [
            b.text for s in split_markdown(canonical).sections for b in s.blocks
        ]

    @pytest.mark.parametrize(
        "name",
        [
            "table",
            "table_aligned",
            "table_no_trailing_pipe",
            "table_no_leading_pipe",
            "table_single_dash_separator",
            "table_pipe_in_cell",
        ],
    )
    def test_tables_keep_one_row_per_line(self, name: str):
        """#3361: a table — malformed rows included — is never welded onto one line."""
        markdown = MARKDOWN_CORPUS[name]
        rendered = render_document(split_markdown(markdown))
        assert markdown.strip() in rendered
        # The separator row is still its own physical line.
        for line in _significant_lines(markdown):
            assert line in rendered.splitlines()

    def test_table_is_one_block(self):
        doc = split_markdown(MARKDOWN_CORPUS["table_no_trailing_pipe"])
        assert len(doc.sections[0].blocks) == 1
        assert doc.sections[0].blocks[0].text.count("\n") == 2

    def test_nested_list_indentation_survives(self):
        rendered = render_document(split_markdown(MARKDOWN_CORPUS["nested_bullets"]))
        assert "  - child" in rendered
        assert "    - grandchild" in rendered

    def test_ordered_list_numbering_is_not_rewritten(self):
        rendered = render_document(split_markdown(MARKDOWN_CORPUS["ordered_from_five"]))
        assert "5. five" in rendered
        assert "6. six" in rendered

    def test_horizontal_rule_survives(self):
        rendered = render_document(split_markdown(MARKDOWN_CORPUS["horizontal_rule"]))
        assert "\n---\n" in rendered

    def test_hard_line_break_spaces_survive(self):
        rendered = render_document(split_markdown(MARKDOWN_CORPUS["hard_line_break"]))
        assert "line one  \nline two" in rendered


class TestSplitStructure:
    def test_sections_and_levels(self):
        doc = split_markdown(MARKDOWN_CORPUS["many_sections"])
        assert [(s.id, s.level) for s in doc.sections] == [("t", 1), ("a", 2), ("b", 3)]

    def test_content_before_first_heading_becomes_preamble(self):
        doc = split_markdown(MARKDOWN_CORPUS["preamble"])
        assert doc.sections[0].id == "preamble"
        assert doc.sections[0].heading == ""
        assert doc.sections[0].blocks[0].text == "Intro prose."

    def test_document_without_any_heading_is_all_preamble(self):
        doc = split_markdown("just prose\n\nand more\n")
        assert len(doc.sections) == 1
        assert doc.sections[0].id == "preamble"
        assert len(doc.sections[0].blocks) == 2

    def test_duplicate_headings_get_unique_ids(self):
        doc = split_markdown("## Rules\n\na\n\n## Rules\n\nb\n")
        assert [s.id for s in doc.sections] == ["rules", "rules-2"]

    def test_blank_lines_inside_a_fence_do_not_split_blocks(self):
        doc = split_markdown(MARKDOWN_CORPUS["fenced_code"])
        assert len(doc.sections[0].blocks) == 1
        assert "\n\n    return" in doc.sections[0].blocks[0].text

    def test_heading_inside_a_fence_is_not_a_section(self):
        doc = split_markdown(MARKDOWN_CORPUS["fenced_code_with_heading"])
        assert [s.id for s in doc.sections] == ["s"]
        assert "# not a heading" in doc.sections[0].blocks[0].text

    def test_tilde_fence_is_honoured(self):
        doc = split_markdown("## S\n\n~~~\na\n\nb\n~~~\n")
        assert len(doc.sections[0].blocks) == 1

    def test_empty_markdown_yields_no_sections(self):
        assert split_markdown("").sections == []
        assert split_markdown("   \n\n  \n").sections == []

    def test_heading_only_document(self):
        doc = split_markdown("## Empty\n")
        assert doc.sections[0].blocks == []
        assert render_document(doc) == "## Empty\n"

    def test_runs_of_blank_lines_collapse(self):
        rendered = render_document(split_markdown("## S\n\n\n\na\n\n\n\nb\n"))
        assert rendered == "## S\n\na\n\nb\n"

    def test_split_is_deterministic(self):
        markdown = MARKDOWN_CORPUS["many_sections"]
        assert split_markdown(markdown) == split_markdown(markdown)


class TestStructuredDocumentFromStored:
    def test_valid_v2_is_used_as_is(self):
        doc = split_markdown("## S\n\nbody\n")
        loaded = structured_document_from_stored(doc.model_dump(), "## Other\n\nignored\n")
        assert loaded == doc

    def test_v1_document_is_rebuilt_from_markdown(self):
        """v1's typed blocks are a lossy projection of the same markdown — drop them."""
        v1 = {
            "version": 1,
            "sections": [
                {
                    "id": "s",
                    "heading": "S",
                    "level": 2,
                    "blocks": [{"type": "paragraph", "text": "| a | b | |---|---| | 1 | 2 |"}],
                }
            ],
        }
        markdown = "## S\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
        loaded = structured_document_from_stored(v1, markdown)
        assert render_document(loaded) == markdown

    def test_missing_structure_is_rebuilt_from_markdown(self):
        markdown = "## S\n\nbody\n"
        assert render_document(structured_document_from_stored(None, markdown)) == markdown

    def test_corrupt_structure_falls_back_to_markdown(self):
        markdown = "## S\n\nbody\n"
        loaded = structured_document_from_stored({"version": 2, "sections": "nonsense"}, markdown)
        assert render_document(loaded) == markdown

    def test_empty_markdown_and_no_structure_is_an_empty_document(self):
        assert structured_document_from_stored(None, "").sections == []


# Operations -----------------------------------------------------------------


def _doc() -> StructuredDocument:
    return split_markdown(
        "# Team Overview\n\n"
        "Quick summary of the engineering team.\n\n"
        "## Members\n\n"
        "- **Alice** — team lead.\n- **Bob** — senior engineer.\n\n"
        "## Cadence\n\n"
        "Standups happen daily at 9am.\n"
    )


def _block_id(doc: StructuredDocument, section_id: str, index: int = 0) -> str:
    section = doc.section_by_id(section_id)
    assert section is not None
    return section.blocks[index].id


class TestApplyOperations:
    def test_zero_ops_returns_identical_document(self):
        doc = _doc()
        outcome = apply_operations(doc, [])
        assert outcome.document == doc
        assert outcome.applied == []
        assert not outcome.changed

    def test_original_document_is_not_mutated(self):
        doc = _doc()
        before = render_document(doc)
        apply_operations(doc, [RemoveSectionOp(section_id="members")])
        assert render_document(doc) == before

    def test_append_block(self):
        doc = _doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="cadence", text="Retro on Fridays.")])
        assert outcome.applied[0]["op"] == "append_block"
        assert "Retro on Fridays." in render_document(outcome.document)

    def test_append_block_assigns_a_unique_id(self):
        doc = _doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="cadence", text="New.")])
        ids = [b.id for s in outcome.document.sections for b in s.blocks]
        assert len(ids) == len(set(ids))

    def test_insert_block_after_anchor(self):
        doc = _doc()
        anchor = _block_id(doc, "members")
        outcome = apply_operations(
            doc, [InsertBlockOp(section_id="members", after_block_id=anchor, text="Team is remote.")]
        )
        section = outcome.document.section_by_id("members")
        assert section is not None
        assert [b.text for b in section.blocks][1] == "Team is remote."

    def test_insert_block_at_start(self):
        doc = _doc()
        outcome = apply_operations(doc, [InsertBlockOp(section_id="members", after_block_id=None, text="Roster:")])
        section = outcome.document.section_by_id("members")
        assert section is not None
        assert section.blocks[0].text == "Roster:"

    def test_replace_block_keeps_id_and_position(self):
        doc = _doc()
        block_id = _block_id(doc, "cadence")
        outcome = apply_operations(
            doc, [ReplaceBlockOp(section_id="cadence", block_id=block_id, text="Standups at 10am.")]
        )
        section = outcome.document.section_by_id("cadence")
        assert section is not None
        assert section.blocks[0].id == block_id
        assert section.blocks[0].text == "Standups at 10am."

    def test_remove_block(self):
        doc = _doc()
        block_id = _block_id(doc, "cadence")
        outcome = apply_operations(doc, [RemoveBlockOp(section_id="cadence", block_id=block_id)])
        section = outcome.document.section_by_id("cadence")
        assert section is not None
        assert section.blocks == []

    def test_add_section_at_end(self):
        doc = _doc()
        outcome = apply_operations(doc, [AddSectionOp(heading="Tools", blocks=["- Linear\n- GitHub"])])
        assert outcome.document.sections[-1].id == "tools"
        assert outcome.applied[0]["assigned_id"] == "tools"

    def test_add_section_after_another(self):
        doc = _doc()
        outcome = apply_operations(doc, [AddSectionOp(heading="Tools", blocks=["x"], after_section_id="members")])
        assert [s.id for s in outcome.document.sections] == ["team-overview", "members", "tools", "cadence"]

    def test_add_section_disambiguates_colliding_id(self):
        doc = _doc()
        outcome = apply_operations(doc, [AddSectionOp(heading="Members", blocks=["x"])])
        assert outcome.applied[0]["assigned_id"] == "members-2"

    def test_remove_section(self):
        doc = _doc()
        outcome = apply_operations(doc, [RemoveSectionOp(section_id="members")])
        assert outcome.document.section_by_id("members") is None

    def test_replace_section_blocks(self):
        doc = _doc()
        outcome = apply_operations(doc, [ReplaceSectionBlocksOp(section_id="members", blocks=["- Only Alice now."])])
        section = outcome.document.section_by_id("members")
        assert section is not None
        assert [b.text for b in section.blocks] == ["- Only Alice now."]

    def test_rename_section_keeps_id(self):
        doc = _doc()
        outcome = apply_operations(doc, [RenameSectionOp(section_id="members", new_heading="The Team")])
        section = outcome.document.section_by_id("members")
        assert section is not None
        assert section.heading == "The Team"
        assert "## The Team" in render_document(outcome.document)

    def test_multi_line_block_text_survives_an_op(self):
        doc = _doc()
        table = "| Name | Role |\n| --- | --- |\n| Alice | Lead |"
        outcome = apply_operations(doc, [AppendBlockOp(section_id="members", text=table)])
        assert table in render_document(outcome.document)


class TestOperationsAreConservative:
    def test_unknown_section_is_skipped(self):
        doc = _doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="nope", text="x")])
        assert outcome.applied == []
        assert "unknown section_id" in outcome.skipped[0]["reason"]
        assert outcome.document == doc

    def test_unknown_block_is_skipped(self):
        doc = _doc()
        outcome = apply_operations(doc, [ReplaceBlockOp(section_id="cadence", block_id="bdeadbeef", text="x")])
        assert outcome.applied == []
        assert "unknown block_id" in outcome.skipped[0]["reason"]
        assert outcome.document == doc

    def test_block_id_from_another_section_is_skipped(self):
        """#3273: a mis-targeted block op must never edit the block it names."""
        doc = _doc()
        foreign = _block_id(doc, "members")
        outcome = apply_operations(doc, [ReplaceBlockOp(section_id="cadence", block_id=foreign, text="WRONG")])
        assert outcome.applied == []
        assert "belongs to section members" in outcome.skipped[0]["reason"]
        assert "WRONG" not in render_document(outcome.document)
        assert outcome.document == doc

    def test_unknown_insert_anchor_is_skipped_not_appended(self):
        doc = _doc()
        outcome = apply_operations(doc, [InsertBlockOp(section_id="cadence", after_block_id="bnope", text="x")])
        assert outcome.applied == []
        assert outcome.document == doc

    def test_unknown_after_section_is_skipped(self):
        doc = _doc()
        outcome = apply_operations(doc, [AddSectionOp(heading="Tools", blocks=["x"], after_section_id="nope")])
        assert outcome.applied == []
        assert outcome.document == doc

    def test_empty_block_text_is_skipped(self):
        doc = _doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="cadence", text="   ")])
        assert outcome.applied == []
        assert "empty" in outcome.skipped[0]["reason"]

    def test_valid_ops_still_apply_when_one_is_skipped(self):
        doc = _doc()
        outcome = apply_operations(
            doc,
            [
                AppendBlockOp(section_id="nope", text="dropped"),
                AppendBlockOp(section_id="cadence", text="kept"),
            ],
        )
        assert len(outcome.applied) == 1
        assert len(outcome.skipped) == 1
        assert "kept" in render_document(outcome.document)
        assert "dropped" not in render_document(outcome.document)

    def test_untouched_sections_are_byte_identical(self):
        doc = _doc()
        before = render_section(doc.sections[1])
        outcome = apply_operations(doc, [AppendBlockOp(section_id="cadence", text="Retro on Fridays.")])
        assert render_section(outcome.document.sections[1]) == before

    def test_untouched_blocks_are_byte_identical_after_a_sibling_edit(self):
        doc = split_markdown("## S\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nA paragraph.\n")
        table_before = doc.sections[0].blocks[0].text
        target = doc.sections[0].blocks[1].id
        outcome = apply_operations(doc, [ReplaceBlockOp(section_id="s", block_id=target, text="Changed.")])
        assert outcome.document.sections[0].blocks[0].text == table_before

    def test_op_summary_omits_block_bodies(self):
        doc = _doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="cadence", text="a very long body")])
        assert "text" not in outcome.applied[0]
        assert outcome.applied[0]["section_id"] == "cadence"


class TestOrphanTableRowMerge:
    """A bare table row appended as its own block joins the table above it.

    A real model does this when asked to add a row (seen in the multi-round
    stability eval). Left alone it renders as a blank line followed by a lone
    ``| ... |``, which is not a table row at all.
    """

    def _table_doc(self) -> StructuredDocument:
        return split_markdown("## Ops\n\n| A | B |\n| --- | --- |\n| x | 1 |\n\nProse after.\n")

    def test_appended_row_joins_the_table(self):
        doc = split_markdown("## Ops\n\n| A | B |\n| --- | --- |\n| x | 1 |\n")
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="| y | 2 |")])
        rendered = render_document(outcome.document)
        assert "| x | 1 |\n| y | 2 |" in rendered
        assert "\n\n| y | 2 |" not in rendered

    def test_append_after_a_trailing_paragraph_is_not_merged(self):
        """Only a row landing directly after the table joins it."""
        doc = self._table_doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="| y | 2 |")])
        assert len(outcome.document.sections[0].blocks) == 3

    def test_merge_is_recorded_in_the_audit_trail(self):
        doc = split_markdown("## Ops\n\n| A | B |\n| --- | --- |\n| x | 1 |\n")
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="| y | 2 |")])
        table_id = doc.sections[0].blocks[0].id
        assert outcome.applied[0]["merged_into_block_id"] == table_id
        assert len(outcome.document.sections[0].blocks) == 1

    def test_inserted_row_joins_the_table_it_follows(self):
        doc = self._table_doc()
        table_id = doc.sections[0].blocks[0].id
        outcome = apply_operations(doc, [InsertBlockOp(section_id="ops", after_block_id=table_id, text="| y | 2 |")])
        assert len(outcome.document.sections[0].blocks) == 2
        assert outcome.document.sections[0].blocks[0].text.endswith("| y | 2 |")

    def test_a_real_table_is_not_merged(self):
        """A block with its own separator row is a table, not an orphan row."""
        doc = self._table_doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="| C | D |\n| --- | --- |\n| z | 3 |")])
        assert len(outcome.document.sections[0].blocks) == 3

    def test_prose_is_not_merged(self):
        doc = self._table_doc()
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="Another paragraph.")])
        assert len(outcome.document.sections[0].blocks) == 3

    def test_row_after_a_non_table_block_is_kept_as_its_own_block(self):
        doc = split_markdown("## Ops\n\nJust prose.\n")
        outcome = apply_operations(doc, [AppendBlockOp(section_id="ops", text="| y | 2 |")])
        assert len(outcome.document.sections[0].blocks) == 2
        assert outcome.applied[0].get("merged_into_block_id") is None

    def test_row_at_the_start_of_a_section_is_kept(self):
        doc = self._table_doc()
        outcome = apply_operations(doc, [InsertBlockOp(section_id="ops", after_block_id=None, text="| y | 2 |")])
        assert outcome.document.sections[0].blocks[0].text == "| y | 2 |"
