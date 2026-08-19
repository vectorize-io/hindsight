"""Unit tests for the structured document schema, renderer, parser, and
delta-operation applicator.

These tests are pure-Python (no DB, no LLM) and run fast. They guard the
mechanical guarantees that the structured-delta architecture relies on:

- Deterministic rendering (same input → same bytes).
- Round-trip parse → render is stable for canonical markdown.
- Section IDs are stable slugs and survive disambiguation.
- Operations target sections/blocks by id/index and never silently corrupt
  the document; invalid ops are dropped, not applied half-way.
- Sections and blocks not mentioned by any op come through byte-identical.
"""

from __future__ import annotations

import json

import pytest

from hindsight_api.engine.reflect.delta_ops import (
    AddSectionOp,
    AppendBlockOp,
    DeltaOperationList,
    InsertBlockOp,
    RemoveBlockOp,
    RemoveSectionOp,
    RenameSectionOp,
    ReplaceBlockOp,
    ReplaceSectionBlocksOp,
    _block_content_text,
    apply_operations,
    serialize_document_for_delta_prompt,
)
from hindsight_api.engine.reflect.structured_doc import (
    BulletListBlock,
    CodeBlock,
    OrderedListBlock,
    ParagraphBlock,
    Section,
    StructuredDocument,
    TableBlock,
    make_unique_id,
    parse_markdown,
    render_block,
    render_document,
    render_section,
    slugify_heading,
)

# Helpers --------------------------------------------------------------------


def _team_overview_doc() -> StructuredDocument:
    return StructuredDocument(
        sections=[
            Section(
                id="team-overview",
                heading="Team Overview",
                level=1,
                blocks=[ParagraphBlock(text="Quick summary of the engineering team.")],
            ),
            Section(
                id="members",
                heading="Members",
                level=2,
                blocks=[
                    BulletListBlock(
                        items=[
                            "**Alice** — team lead, owns planning.",
                            "**Bob** — senior engineer, mentors juniors.",
                        ]
                    )
                ],
            ),
            Section(
                id="cadence",
                heading="Cadence",
                level=2,
                blocks=[ParagraphBlock(text="Standups happen daily at 9am.")],
            ),
        ]
    )


# Slug ----------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert slugify_heading("Purpose") == "purpose"

    def test_multi_word(self):
        assert slugify_heading("Stop Conditions") == "stop-conditions"

    def test_punctuation_collapses(self):
        assert slugify_heading("Inputs / Context !") == "inputs-context"

    def test_unicode_falls_back(self):
        # Non-ASCII chars are stripped; if nothing remains, slug becomes "section".
        assert slugify_heading("???") == "section"

    def test_make_unique_id_no_collision(self):
        assert make_unique_id("rules", set()) == "rules"

    def test_make_unique_id_collision(self):
        assert make_unique_id("rules", {"rules"}) == "rules-2"
        assert make_unique_id("rules", {"rules", "rules-2"}) == "rules-3"


# Renderer ------------------------------------------------------------------


class TestRenderer:
    def test_paragraph(self):
        assert render_block(ParagraphBlock(text="hello world")) == "hello world"

    def test_bullet_list(self):
        block = BulletListBlock(items=["one", "two"])
        assert render_block(block) == "- one\n- two"

    def test_ordered_list_uses_sequential_numbering(self):
        block = OrderedListBlock(items=["one", "two", "three"])
        assert render_block(block) == "1. one\n2. two\n3. three"

    def test_code_block_with_language(self):
        block = CodeBlock(language="json", text='{"a": 1}')
        assert render_block(block) == '```json\n{"a": 1}\n```'

    def test_code_block_no_language(self):
        block = CodeBlock(text="raw text")
        assert render_block(block) == "```\nraw text\n```"

    def test_table_one_row_per_line(self):
        block = TableBlock(headers=["Layer", "Role"], rows=[["API", "HTTP"], ["Engine", "Memory"]])
        assert render_block(block) == ("| Layer | Role |\n| --- | --- |\n| API | HTTP |\n| Engine | Memory |")

    def test_table_escapes_pipes_in_cells(self):
        block = TableBlock(headers=["col"], rows=[["a | b"]])
        assert render_block(block) == "| col |\n| --- |\n| a \\| b |"

    def test_table_cell_newline_collapses_to_one_line(self):
        block = TableBlock(headers=["col"], rows=[["first\nsecond"]])
        assert render_block(block) == "| col |\n| --- |\n| first second |"

    def test_table_short_row_is_padded(self):
        block = TableBlock(headers=["a", "b", "c"], rows=[["1"]])
        assert render_block(block).splitlines()[-1] == "| 1 |  |  |"

    def test_table_row_wider_than_headers_keeps_every_cell(self):
        block = TableBlock(headers=["a"], rows=[["1", "2"]])
        assert render_block(block) == "| a |  |\n| --- | --- |\n| 1 | 2 |"

    def test_table_without_headers_still_renders_rows(self):
        block = TableBlock(headers=[], rows=[["1", "2"]])
        assert render_block(block) == "|  |  |\n| --- | --- |\n| 1 | 2 |"

    def test_empty_table_renders_empty(self):
        assert render_block(TableBlock()) == ""

    def test_section_heading_level(self):
        section = Section(id="purpose", heading="Purpose", level=3, blocks=[ParagraphBlock(text="hi")])
        assert render_section(section).startswith("### Purpose\n\nhi")

    def test_document_round_trip_is_stable(self):
        doc = _team_overview_doc()
        rendered = render_document(doc)
        # Re-rendering must produce the same bytes.
        assert render_document(doc) == rendered
        # Headings, members, cadence all present.
        assert "# Team Overview" in rendered
        assert "## Members" in rendered
        assert "## Cadence" in rendered
        assert "- **Alice**" in rendered
        assert "Standups happen daily at 9am" in rendered
        # Sections separated by exactly one blank line, document ends with newline.
        assert rendered.endswith("\n")
        assert "\n\n\n" not in rendered

    def test_empty_document_renders_empty(self):
        assert render_document(StructuredDocument()) == ""


# Parser --------------------------------------------------------------------


class TestParser:
    def test_simple_document(self):
        markdown = (
            "# Team Overview\n\nQuick summary.\n\n## Members\n\n- Alice\n- Bob\n\n## Cadence\n\nStandups daily.\n"
        )
        doc = parse_markdown(markdown)
        assert [s.id for s in doc.sections] == ["team-overview", "members", "cadence"]
        assert [s.level for s in doc.sections] == [1, 2, 2]
        assert isinstance(doc.sections[0].blocks[0], ParagraphBlock)
        assert isinstance(doc.sections[1].blocks[0], BulletListBlock)
        assert doc.sections[1].blocks[0].items == ["Alice", "Bob"]

    def test_horizontal_rule_treated_as_blank(self):
        markdown = "## Rules\n\n- one\n\n---\n\n## Stop\n\nstop here.\n"
        doc = parse_markdown(markdown)
        assert [s.id for s in doc.sections] == ["rules", "stop"]
        # Horizontal rule must NOT become a paragraph.
        assert all(not (isinstance(b, ParagraphBlock) and "---" in b.text) for s in doc.sections for b in s.blocks)

    def test_horizontal_rule_inside_code_fence_is_preserved(self):
        # `---` is the YAML document separator, so treating it as a section
        # separator inside a fence merges two documents into one invalid one.
        markdown = (
            "## Deploy Manifest\n\n```yaml\napiVersion: v1\nkind: ConfigMap\n---\napiVersion: v1\nkind: Secret\n```\n"
        )
        doc = parse_markdown(markdown)
        block = doc.sections[0].blocks[0]
        assert isinstance(block, CodeBlock)
        assert block.text == "apiVersion: v1\nkind: ConfigMap\n---\napiVersion: v1\nkind: Secret"
        assert render_document(doc) == markdown

    def test_rule_spellings_inside_code_fence_are_preserved(self):
        for rule in ("---", "***", "___", "-----", "  ---"):
            markdown = f"## Snippet\n\n```text\nbefore\n{rule}\nafter\n```\n"
            doc = parse_markdown(markdown)
            block = doc.sections[0].blocks[0]
            assert isinstance(block, CodeBlock), rule
            assert block.text == f"before\n{rule}\nafter", rule

    def test_ordered_list(self):
        markdown = "## Steps\n\n1. one\n2. two\n3. three\n"
        doc = parse_markdown(markdown)
        block = doc.sections[0].blocks[0]
        assert isinstance(block, OrderedListBlock)
        assert block.items == ["one", "two", "three"]

    def test_code_block(self):
        markdown = '## Example\n\n```json\n{"a": 1}\n```\n'
        doc = parse_markdown(markdown)
        block = doc.sections[0].blocks[0]
        assert isinstance(block, CodeBlock)
        assert block.language == "json"
        assert block.text == '{"a": 1}'

    def test_implicit_overview_when_content_before_first_heading(self):
        markdown = "preamble paragraph.\n\n## Members\n\n- Alice\n"
        doc = parse_markdown(markdown)
        assert doc.sections[0].id == "overview"
        assert isinstance(doc.sections[0].blocks[0], ParagraphBlock)
        assert doc.sections[1].id == "members"

    def test_duplicate_headings_get_unique_ids(self):
        markdown = "## Notes\n\nfirst.\n\n## Notes\n\nsecond.\n"
        doc = parse_markdown(markdown)
        assert [s.id for s in doc.sections] == ["notes", "notes-2"]

    def test_table(self):
        md = "## Layers\n\n| Layer | Role |\n| --- | --- |\n| API | HTTP |\n| Engine | Memory |\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert block.headers == ["Layer", "Role"]
        assert block.rows == [["API", "HTTP"], ["Engine", "Memory"]]

    def test_table_alignment_colons_are_a_separator(self):
        md = "## T\n\n| a | b |\n|:---|---:|\n| 1 | 2 |\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert block.headers == ["a", "b"]
        assert block.rows == [["1", "2"]]

    def test_table_escaped_pipe_stays_in_one_cell(self):
        md = "## T\n\n| col | expr |\n| --- | --- |\n| a | x \\| y |\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert block.rows == [["a", "x | y"]]

    def test_table_keeps_other_backslash_sequences_verbatim(self):
        md = "## T\n\n| col |\n| --- |\n| C:\\path |\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, TableBlock)
        assert block.rows == [["C:\\path"]]

    def test_prose_containing_a_pipe_is_not_a_table(self):
        md = "## T\n\nUse a | b to pipe.\nSecond line.\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, ParagraphBlock)

    def test_pipe_rows_without_separator_are_not_a_table(self):
        md = "## T\n\n| a | b |\n| 1 | 2 |\n"
        block = parse_markdown(md).sections[0].blocks[0]
        assert isinstance(block, ParagraphBlock)

    def test_table_round_trip_via_render(self):
        doc = StructuredDocument(
            sections=[
                Section(
                    id="layers",
                    heading="Layers",
                    blocks=[TableBlock(headers=["Layer", "Note"], rows=[["API", "a | b"], ["Engine", ""]])],
                )
            ]
        )
        markdown = render_document(doc)
        roundtripped = parse_markdown(markdown)
        assert roundtripped.sections[0].blocks[0] == doc.sections[0].blocks[0]
        assert render_document(roundtripped) == markdown

    def test_round_trip_via_render(self):
        original = _team_overview_doc()
        markdown = render_document(original)
        roundtripped = parse_markdown(markdown)
        # Re-render must match the original render exactly.
        assert render_document(roundtripped) == markdown


# Operation applicator ------------------------------------------------------


class TestApplyOperations:
    def test_zero_ops_returns_identical_document(self):
        doc = _team_overview_doc()
        result = apply_operations(doc, [])
        assert result.document.model_dump() == doc.model_dump()
        assert render_document(result.document) == render_document(doc)
        assert result.applied == []
        assert result.changed is False

    def test_unknown_section_op_is_skipped(self):
        doc = _team_overview_doc()
        op = AppendBlockOp(section_id="does-not-exist", block=ParagraphBlock(text="x"))
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert len(result.skipped) == 1
        assert "unknown section_id" in result.skipped[0]["reason"]
        # Document unchanged.
        assert render_document(result.document) == render_document(doc)

    def test_append_block_to_existing_section(self):
        doc = _team_overview_doc()
        op = AppendBlockOp(
            section_id="members",
            block=BulletListBlock(items=["**Carol** — junior engineer."]),
        )
        result = apply_operations(doc, [op])
        members = result.document.section_by_id("members")
        assert members is not None
        assert len(members.blocks) == 2  # original list + new bullet block
        # Other sections byte-identical
        original = doc.model_dump()
        new = result.document.model_dump()
        assert new["sections"][0] == original["sections"][0]  # team-overview
        assert new["sections"][2] == original["sections"][2]  # cadence

    def test_insert_block_at_index(self):
        doc = _team_overview_doc()
        op = InsertBlockOp(
            section_id="members",
            index=0,
            # Verbatim excerpt of the block currently at index 0 (the bullet
            # list this insert lands before).
            anchor="**Alice** — team lead",
            block=ParagraphBlock(text="Roster as of 2026:"),
        )
        result = apply_operations(doc, [op])
        members = result.document.section_by_id("members")
        assert isinstance(members.blocks[0], ParagraphBlock)
        assert members.blocks[0].text.startswith("Roster")

    def test_insert_block_out_of_range_skipped(self):
        doc = _team_overview_doc()
        op = InsertBlockOp(section_id="members", index=99, block=ParagraphBlock(text="x"))
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert "index out of range" in result.skipped[0]["reason"]

    def test_insert_block_at_append_position_needs_no_anchor(self):
        """index == len(blocks) is a pure append: there is no existing block
        there to anchor against, so an empty/omitted anchor still applies."""
        doc = _team_overview_doc()
        members_len = len(doc.section_by_id("members").blocks)
        op = InsertBlockOp(section_id="members", index=members_len, block=ParagraphBlock(text="Appended."))
        result = apply_operations(doc, [op])
        assert result.applied
        members = result.document.section_by_id("members")
        assert members.blocks[-1].text == "Appended."

    def test_insert_block_wrong_anchor_skipped(self):
        """index < len(blocks) DOES name an existing block, so a wrong
        anchor there must still be caught."""
        doc = _team_overview_doc()
        op = InsertBlockOp(
            section_id="members",
            index=0,
            anchor="Standups happen daily",  # content from a different section
            block=ParagraphBlock(text="x"),
        )
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert "anchor" in result.skipped[0]["reason"]
        # Nothing was inserted.
        assert len(result.document.section_by_id("members").blocks) == 1

    def test_replace_block(self):
        doc = _team_overview_doc()
        op = ReplaceBlockOp(
            section_id="cadence",
            index=0,
            anchor="Standups happen daily at 9am.",
            block=ParagraphBlock(text="Standups happen daily at 10am."),
        )
        result = apply_operations(doc, [op])
        cadence = result.document.section_by_id("cadence")
        assert isinstance(cadence.blocks[0], ParagraphBlock)
        assert cadence.blocks[0].text.endswith("10am.")

    def test_replace_table_block_anchor_uses_cell_text(self):
        doc = StructuredDocument(
            sections=[
                Section(
                    id="layers",
                    heading="Layers",
                    blocks=[
                        TableBlock(headers=["Layer", "Role"], rows=[["API", "HTTP"], ["Engine", "Memory"]]),
                    ],
                )
            ]
        )
        op = ReplaceBlockOp(
            section_id="layers",
            index=0,
            anchor="Layer Role API HTTP",
            block=ParagraphBlock(text="replaced"),
        )
        result = apply_operations(doc, [op])
        assert result.applied
        assert isinstance(result.document.section_by_id("layers").blocks[0], ParagraphBlock)

    def test_table_block_wrong_anchor_is_skipped_not_raised(self):
        """A table used to raise TypeError in _block_content_text and abort the refresh."""
        doc = StructuredDocument(
            sections=[
                Section(
                    id="layers",
                    heading="Layers",
                    blocks=[TableBlock(headers=["Layer", "Role"], rows=[["API", "HTTP"]])],
                )
            ]
        )
        op = ReplaceBlockOp(
            section_id="layers",
            index=0,
            anchor="this text is not in the table",
            block=ParagraphBlock(text="x"),
        )
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert "anchor" in result.skipped[0]["reason"]
        assert isinstance(result.document.section_by_id("layers").blocks[0], TableBlock)

    def test_replace_block_missing_anchor_skipped(self):
        doc = _team_overview_doc()
        op = ReplaceBlockOp(section_id="cadence", index=0, block=ParagraphBlock(text="new text"))
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert result.skipped[0]["reason"] == "missing anchor"
        # Original content untouched.
        assert result.document.section_by_id("cadence").blocks[0].text == "Standups happen daily at 9am."

    def test_replace_block_anchor_whitespace_normalized(self):
        """Irregular whitespace in the quoted anchor (extra spaces, a
        line-wrapped newline) must not cause a false mismatch."""
        doc = _team_overview_doc()
        op = ReplaceBlockOp(
            section_id="cadence",
            index=0,
            anchor="Standups   happen\ndaily  at 9am.",
            block=ParagraphBlock(text="Standups happen daily at 10am."),
        )
        result = apply_operations(doc, [op])
        assert result.applied
        assert result.document.section_by_id("cadence").blocks[0].text.endswith("10am.")

    def test_off_by_one_index_with_neighbor_anchor_is_skipped(self):
        """Regression test for the measured production defect: a delta
        refresh computed an off-by-one ``index`` (miscounting an 8-block
        section's array elements) and replaced the wrong block, because
        range validation alone cannot tell a wrong-but-in-range index from a
        correct one.

        Here the op claims ``index=4`` but its anchor actually names the
        content of the block at index 5 (its neighbor) -- exactly the shape
        of the measured defect, where the model's intended target and its
        claimed index pointed at different blocks. The anchor guard must
        catch the mismatch and skip the op rather than silently overwriting
        block 4 with content meant for block 5.
        """
        blocks = [ParagraphBlock(text=f"Paragraph number {i} of the long section.") for i in range(8)]
        doc = StructuredDocument(sections=[Section(id="long", heading="Long Section", level=2, blocks=blocks)])
        neighbor_text = blocks[5].text
        op = ReplaceBlockOp(
            section_id="long",
            index=4,
            anchor=neighbor_text[:40],
            block=ParagraphBlock(text="REPLACED CONTENT"),
        )
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert len(result.skipped) == 1
        assert "anchor" in result.skipped[0]["reason"]
        section = result.document.section_by_id("long")
        # Neither the wrongly-targeted block (4) nor its neighbor (5) changed.
        assert section.blocks[4].model_dump() == blocks[4].model_dump()
        assert section.blocks[5].model_dump() == blocks[5].model_dump()

    def test_remove_block(self):
        doc = _team_overview_doc()
        op = RemoveBlockOp(section_id="members", index=0, anchor="**Alice** — team lead")
        result = apply_operations(doc, [op])
        members = result.document.section_by_id("members")
        assert members.blocks == []

    def test_remove_block_missing_anchor_skipped(self):
        doc = _team_overview_doc()
        op = RemoveBlockOp(section_id="members", index=0)
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert result.skipped[0]["reason"] == "missing anchor"
        assert len(result.document.section_by_id("members").blocks) == 1

    def test_remove_block_wrong_anchor_skipped(self):
        doc = _team_overview_doc()
        op = RemoveBlockOp(section_id="members", index=0, anchor="Standups happen daily")
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert "anchor" in result.skipped[0]["reason"]
        assert len(result.document.section_by_id("members").blocks) == 1

    def test_add_section_at_end(self):
        doc = _team_overview_doc()
        op = AddSectionOp(
            heading="Open Questions",
            blocks=[ParagraphBlock(text="None right now.")],
        )
        result = apply_operations(doc, [op])
        assert result.document.sections[-1].id == "open-questions"
        assert result.document.sections[-1].heading == "Open Questions"

    def test_add_section_after_existing(self):
        doc = _team_overview_doc()
        op = AddSectionOp(
            heading="Charter",
            after_section_id="team-overview",
            blocks=[ParagraphBlock(text="Mission statement.")],
        )
        result = apply_operations(doc, [op])
        ids = [s.id for s in result.document.sections]
        assert ids == ["team-overview", "charter", "members", "cadence"]

    def test_add_section_after_unknown_skipped(self):
        doc = _team_overview_doc()
        op = AddSectionOp(
            heading="Charter",
            after_section_id="nope",
            blocks=[],
        )
        result = apply_operations(doc, [op])
        assert result.applied == []
        assert "unknown after_section_id" in result.skipped[0]["reason"]

    def test_add_section_with_id_collision_disambiguates(self):
        doc = _team_overview_doc()
        op = AddSectionOp(heading="Members", blocks=[])
        result = apply_operations(doc, [op])
        # Two sections with heading "Members": the new one gets "members-2".
        ids = [s.id for s in result.document.sections]
        assert "members" in ids
        assert "members-2" in ids

    def test_remove_section(self):
        doc = _team_overview_doc()
        op = RemoveSectionOp(section_id="cadence")
        result = apply_operations(doc, [op])
        assert [s.id for s in result.document.sections] == ["team-overview", "members"]

    def test_replace_section_blocks_preserves_id_and_heading(self):
        doc = _team_overview_doc()
        op = ReplaceSectionBlocksOp(
            section_id="members",
            blocks=[ParagraphBlock(text="See the org chart.")],
        )
        result = apply_operations(doc, [op])
        members = result.document.section_by_id("members")
        assert members.heading == "Members"
        assert members.id == "members"
        assert len(members.blocks) == 1
        assert isinstance(members.blocks[0], ParagraphBlock)

    def test_rename_section_keeps_id(self):
        doc = _team_overview_doc()
        op = RenameSectionOp(section_id="cadence", new_heading="Operating Cadence")
        result = apply_operations(doc, [op])
        section = result.document.section_by_id("cadence")
        assert section.heading == "Operating Cadence"
        # ID stable so future ops still resolve.
        assert section.id == "cadence"

    def test_unmodified_sections_byte_identical_in_render(self):
        """The structural guarantee: sections not touched by any op render
        identically character-for-character.
        """
        doc = _team_overview_doc()
        op = AppendBlockOp(
            section_id="members",
            block=ParagraphBlock(text="New: Carol joined as junior engineer."),
        )
        result = apply_operations(doc, [op])
        before_overview = render_section(doc.section_by_id("team-overview"))
        after_overview = render_section(result.document.section_by_id("team-overview"))
        before_cadence = render_section(doc.section_by_id("cadence"))
        after_cadence = render_section(result.document.section_by_id("cadence"))
        assert before_overview == after_overview
        assert before_cadence == after_cadence


class TestSerializeDocumentForDeltaPrompt:
    """``serialize_document_for_delta_prompt`` is the prompt-facing view that
    annotates each block with its own index, so the model reads its position
    instead of silently counting array elements (the root cause behind the
    off-by-one regression tested above). The tests below check that this
    annotation is absent from the real persisted dump, and that feeding the
    annotated view back through ``StructuredDocument``'s own strict schema
    validation (the only parse path currently used to reconstruct a
    document) is rejected rather than silently accepted.
    """

    def test_annotates_each_block_with_its_index(self):
        doc = _team_overview_doc()
        payload = json.loads(serialize_document_for_delta_prompt(doc))
        members_section = next(s for s in payload["sections"] if s["id"] == "members")
        assert members_section["blocks"][0]["index"] == 0
        cadence_section = next(s for s in payload["sections"] if s["id"] == "cadence")
        assert cadence_section["blocks"][0]["index"] == 0

    def test_multi_block_section_indices_are_sequential(self):
        blocks = [ParagraphBlock(text=f"p{i}") for i in range(5)]
        doc = StructuredDocument(sections=[Section(id="s", heading="S", level=2, blocks=blocks)])
        payload = json.loads(serialize_document_for_delta_prompt(doc))
        assert [b["index"] for b in payload["sections"][0]["blocks"]] == [0, 1, 2, 3, 4]

    def test_index_annotation_does_not_leak_into_persisted_document(self):
        doc = _team_overview_doc()
        # None of this fixture's blocks carry an "index" key in the real
        # persisted dump (checked for every section/block in _team_overview_doc).
        persisted = json.loads(doc.model_dump_json())
        for section in persisted["sections"]:
            for block in section["blocks"]:
                assert "index" not in block

        # The annotated, prompt-facing view is rejected by the strict
        # (extra="forbid") Block/StructuredDocument schema when fed back
        # through model_validate -- the same validation path used to parse
        # a document today -- rather than being silently accepted.
        annotated_payload = json.loads(serialize_document_for_delta_prompt(doc))
        with pytest.raises(Exception):  # pydantic ValidationError
            StructuredDocument.model_validate(annotated_payload)


class TestBlockContentText:
    """Anchor extraction must tolerate every block type, including ones the
    union grew after the first delta-ops commit, and must never raise."""

    def test_table_joins_headers_and_rows(self):
        text = _block_content_text(TableBlock(headers=["Layer", "Role"], rows=[["API", "HTTP"], ["Engine", "Memory"]]))
        assert text == "Layer Role API HTTP Engine Memory"

    def test_unknown_block_with_table_shape_does_not_raise(self):
        class _FutureTable:
            headers = ["a"]
            rows = [["b", "c"]]

        assert _block_content_text(_FutureTable()) == "a b c"

    def test_opaque_unknown_block_is_empty_not_raised(self):
        class _Opaque:
            pass

        assert _block_content_text(_Opaque()) == ""


class TestDeltaOperationListSchema:
    """Sanity-check that the discriminated-union schema serialises as the LLM
    will see it: each op has a literal ``op`` string that picks the variant.
    """

    def test_round_trip_via_json(self):
        ops = DeltaOperationList(
            operations=[
                AppendBlockOp(
                    section_id="members",
                    block=ParagraphBlock(text="hi"),
                ),
                AddSectionOp(
                    heading="Open Questions",
                    after_section_id="cadence",
                    blocks=[ParagraphBlock(text="None.")],
                ),
                RemoveSectionOp(section_id="charter"),
            ]
        )
        payload = ops.model_dump_json()
        roundtripped = DeltaOperationList.model_validate_json(payload)
        assert len(roundtripped.operations) == 3

    def test_invalid_op_field_rejected(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            DeltaOperationList.model_validate({"operations": [{"op": "not_a_real_op", "section_id": "x"}]})

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            DeltaOperationList.model_validate(
                {
                    "operations": [
                        {
                            "op": "append_block",
                            "section_id": "members",
                            "block": {"type": "paragraph", "text": "hi"},
                            "extra_field": "no",
                        }
                    ]
                }
            )
