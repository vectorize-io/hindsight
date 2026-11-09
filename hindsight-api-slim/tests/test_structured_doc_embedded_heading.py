"""A heading the model writes inside block text becomes a real section (#6379)."""

from hindsight_api.engine.reflect.delta_ops import AppendBlockOp, apply_operations
from hindsight_api.engine.reflect.structured_doc import document_from_sections, render_document, split_markdown


def _shape(doc):
    return [(s.heading, [b.text for b in s.blocks]) for s in doc.sections]


def test_appended_heading_opens_its_own_section_and_later_edits_stay_put():
    doc = split_markdown("## Constraints\n\nBudget is 200ms.\n\n## Procedure\n\n1. Step\n")
    first = apply_operations(
        doc, [AppendBlockOp(section_id="constraints", text="### History\n\n- 6 Nov: schema errors name the field")]
    ).document
    # The structure matches what the render shows a reader.
    assert _shape(first) == _shape(split_markdown(render_document(first)))
    assert [s.id for s in first.sections] == ["constraints", "history", "procedure"]
    untouched = first.sections[0].blocks[0].id
    assert untouched == doc.sections[0].blocks[0].id

    # An edit to Constraints now renders under Constraints, not under History.
    second = apply_operations(first, [AppendBlockOp(section_id="constraints", text="Measured at p95.")]).document
    rendered = split_markdown(render_document(second))
    assert [b.text for b in rendered.section_by_id("history").blocks] == ["- 6 Nov: schema errors name the field"]
    assert [b.text for b in rendered.section_by_id("constraints").blocks] == ["Budget is 200ms.", "Measured at p95."]
    assert len(second.block_ids()) == sum(len(s.blocks) for s in second.sections)


def test_done_document_with_a_heading_inside_a_block_is_split():
    doc = document_from_sections(
        {"sections": [{"heading": "Overview", "level": 2, "blocks": ["Intro.", "### Details\n\nMore."]}]}
    )
    assert _shape(doc) == [("Overview", ["Intro."]), ("Details", ["More."])]


def test_heading_inside_a_code_fence_is_not_a_section():
    doc = split_markdown("## Example\n\nIntro.\n")
    out = apply_operations(doc, [AppendBlockOp(section_id="example", text="```md\n# not a heading\n```")]).document
    assert [s.id for s in out.sections] == ["example"]
