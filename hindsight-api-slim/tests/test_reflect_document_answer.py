"""The reflect agent can answer with a document instead of markdown.

A document that is generated as markdown has to be read back to work out its
structure, and reading markdown back is where #3361 destroyed tables. In
document mode the model states the structure and the markdown is rendered from
it, so nothing the model writes is ever parsed to find out what it meant.

Most of these are pure-Python tests of the schema, the prompt text and the
parsing of the tool call — the mechanics. ``TestRealModelFillsTheDocumentShape``
is the exception: whether a real model actually fills the shape is the thing
that broke, and no amount of schema testing can answer it.
"""

from __future__ import annotations

import uuid

import pytest

from hindsight_api.engine.reflect.structured_doc import (
    document_from_sections,
    render_document,
)
from hindsight_api.engine.reflect.tools_schema import get_reflect_tools


def _done_tool(**kwargs) -> dict:
    return next(t for t in get_reflect_tools(**kwargs) if t["function"]["name"] == "done")


class TestDoneToolSchema:
    def test_markdown_mode_is_the_default(self):
        params = _done_tool()["function"]["parameters"]
        assert "answer" in params["properties"]
        assert "document" not in params["properties"]

    def test_document_mode_replaces_the_answer_field(self):
        params = _done_tool(answer_as_document=True)["function"]["parameters"]
        assert "document" in params["properties"]
        assert "answer" not in params["properties"], "the model must not have a markdown escape hatch"
        assert params["required"] == ["document"]

    def test_document_mode_keeps_the_supporting_id_fields(self):
        params = _done_tool(answer_as_document=True)["function"]["parameters"]
        for field in ("memory_ids", "mental_model_ids", "observation_ids"):
            assert field in params["properties"]

    def test_document_mode_composes_with_directives(self):
        params = _done_tool(directive_rules=["Be concise"], answer_as_document=True)["function"]["parameters"]
        assert set(params["required"]) == {"document", "directive_compliance"}

    def test_schema_has_no_union_types(self):
        """A tool schema goes to the provider verbatim, and Gemini rejects ``oneOf``."""
        rendered = repr(_done_tool(answer_as_document=True))
        for keyword in ("oneOf", "anyOf", "allOf", "discriminator"):
            assert keyword not in rendered

    def test_section_shape_is_heading_level_blocks(self):
        document = _done_tool(answer_as_document=True)["function"]["parameters"]["properties"]["document"]
        section = document["properties"]["sections"]["items"]
        assert set(section["required"]) == {"heading", "level", "blocks"}
        assert section["properties"]["blocks"]["items"]["type"] == "string"


class TestDocumentFromSections:
    def test_renders_headings_and_blocks(self):
        doc = document_from_sections(
            {"sections": [{"heading": "Ops", "level": 2, "blocks": ["Intro.", "- one\n- two"]}]}
        )
        assert render_document(doc) == "## Ops\n\nIntro.\n\n- one\n- two\n"

    def test_ids_are_assigned(self):
        doc = document_from_sections({"sections": [{"heading": "Ops", "level": 2, "blocks": ["Intro."]}]})
        assert doc.sections[0].id == "ops"
        assert doc.sections[0].blocks[0].id

    def test_table_survives_verbatim(self):
        table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        doc = document_from_sections({"sections": [{"heading": "T", "level": 2, "blocks": [table]}]})
        assert doc.sections[0].blocks[0].text == table

    def test_a_block_holding_several_fragments_is_split(self):
        """The model sometimes packs a whole section into one string."""
        doc = document_from_sections({"sections": [{"heading": "T", "level": 2, "blocks": ["One para.\n\nTwo para."]}]})
        assert [b.text for b in doc.sections[0].blocks] == ["One para.", "Two para."]

    def test_blank_lines_inside_a_fence_do_not_split(self):
        fence = "```python\ndef f():\n\n    return 1\n```"
        doc = document_from_sections({"sections": [{"heading": "T", "level": 2, "blocks": [fence]}]})
        assert [b.text for b in doc.sections[0].blocks] == [fence]

    def test_heading_hashes_are_stripped(self):
        doc = document_from_sections({"sections": [{"heading": "## Ops", "level": 2, "blocks": ["x"]}]})
        assert doc.sections[0].heading == "Ops"
        assert render_document(doc).startswith("## Ops\n")

    def test_empty_heading_renders_without_one(self):
        doc = document_from_sections({"sections": [{"heading": "", "level": 2, "blocks": ["lead in"]}]})
        assert doc.sections[0].id == "preamble"
        assert render_document(doc) == "lead in\n"

    def test_out_of_range_level_is_clamped(self):
        doc = document_from_sections({"sections": [{"heading": "T", "level": 99, "blocks": ["x"]}]})
        assert doc.sections[0].level == 6

    def test_missing_level_defaults_to_two(self):
        doc = document_from_sections({"sections": [{"heading": "T", "blocks": ["x"]}]})
        assert doc.sections[0].level == 2

    def test_duplicate_headings_get_unique_ids(self):
        doc = document_from_sections(
            {"sections": [{"heading": "T", "level": 2, "blocks": ["a"]}, {"heading": "T", "level": 2, "blocks": ["b"]}]}
        )
        assert [s.id for s in doc.sections] == ["t", "t-2"]

    def test_empty_and_malformed_entries_are_dropped(self):
        doc = document_from_sections(
            {"sections": [{"heading": "", "level": 2, "blocks": ["   ", None]}, "not a section", {}]}
        )
        assert doc.sections == []

    def test_empty_payload_is_an_empty_document(self):
        assert document_from_sections({}).sections == []
        assert render_document(document_from_sections({"sections": []})) == ""


class TestBareSectionIsReadAsOne:
    """A one-section document the model emitted without its wrapper.

    The most common near-miss on this schema: the model states the section *as*
    the document — heading, level and blocks at the top level, no ``sections``
    array. Every fact is present and correctly structured; only the wrapper is
    absent. Read literally it has no sections, which rendered to "" and made
    reflect raise ReflectNoAnswerError — the whole refresh discarded, and retried
    against the same prompt, over one missing key.
    """

    def test_a_bare_section_is_taken_as_the_document(self):
        doc = document_from_sections(
            {"heading": "Team Information Summary", "level": 2, "blocks": ["Alice leads.", "Bob reviews."]}
        )
        assert [s.heading for s in doc.sections] == ["Team Information Summary"]
        rendered = render_document(doc)
        assert "## Team Information Summary" in rendered
        # The content is what was nearly thrown away — assert it survives, not just the shape.
        assert "Alice leads." in rendered
        assert "Bob reviews." in rendered

    def test_a_bare_section_keeps_the_block_granularity(self):
        """Blocks must not be folded together on this path — delta operations address
        them individually, so a document imported here has to be as addressable as one
        that arrived wrapped."""
        doc = document_from_sections({"heading": "H", "level": 2, "blocks": ["One.", "- a\n- b"]})
        assert len(doc.sections[0].blocks) == 2

    def test_a_wrapped_document_is_unaffected(self):
        """The normalisation triggers on the absence of ``sections``, so a correct
        payload — including one whose section happens to carry no blocks — takes the
        ordinary path."""
        doc = document_from_sections({"sections": [{"heading": "H", "level": 2, "blocks": ["x"]}]})
        assert [s.heading for s in doc.sections] == ["H"]
        assert document_from_sections({"sections": []}).sections == []

    def test_a_payload_that_is_neither_stays_empty(self):
        """Not a licence to guess: something with no sections and no blocks is not a
        document, and inventing one from it would be the silent overwrite #2959 fixed."""
        assert document_from_sections({"heading": "H", "level": 2}).sections == []
        assert document_from_sections({"blocks": "not a list"}).sections == []


class TestDocumentPromptStatesTheShape:
    """The prompt has to name the field the schema requires.

    It described every field of a *section* — heading, level, blocks — and never
    named the ``sections`` array holding them, so the prose described a section
    while the tool schema described a document containing sections. Models
    resolved that disagreement in favour of the prose. These are direct asserts,
    not judged: whether the text reaches the prompt is mechanical.
    """

    @staticmethod
    def _document_block() -> str:
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(bank_profile={"name": "T", "mission": ""}, answer_as_document=True)
        return prompt[prompt.index("## Output Format: Structured Document") :]

    def test_the_sections_array_is_named(self):
        assert "'sections' array" in self._document_block()

    def test_an_example_of_the_shape_is_shown(self):
        """A nested shape stated but never shown is the gap that caused this: the
        example is the part a model copies."""
        block = self._document_block()
        assert '{"sections": [{"heading"' in block

    def test_the_single_section_case_is_called_out(self):
        """The failure only ever showed up on one-section documents — that is the case
        where flattening looks harmless to the model."""
        assert "never emit a bare section" in self._document_block()

    def test_markdown_mode_does_not_carry_the_document_shape(self):
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(bank_profile={"name": "T", "mission": ""})
        assert "'sections' array" not in prompt


class TestOverBudgetRewrite:
    """A document too long for its budget is trimmed as a document, not as prose.

    The trim is the one place a long answer gets regenerated, so asking for prose
    there would put the model back in charge of the markdown that gets stored —
    on exactly the documents whose structure matters most.
    """

    def test_a_json_rewrite_is_read_back_as_a_document(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        rewritten = (
            '{"sections": [{"heading": "Ops", "level": 2, "blocks": ["| a | b |\\n| --- | --- |\\n| 1 | 2 |"]}]}'
        )
        trimmed = _document_from_rewrite(rewritten, "previous")
        assert [s.heading for s in trimmed.structure.sections] == ["Ops"]
        assert trimmed.markdown == "## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"

    def test_markdown_rewrite_falls_back_to_a_lossless_split(self):
        """A model that ignores the format must not cost us the whole reflect."""
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        trimmed = _document_from_rewrite("## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n", "previous")
        assert [s.heading for s in trimmed.structure.sections] == ["Ops"]
        assert "| 1 | 2 |" in trimmed.markdown.splitlines()

    def test_empty_rewrite_keeps_the_previous_answer(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        trimmed = _document_from_rewrite("   ", "## Kept\n\nbody\n")
        assert trimmed.markdown == "## Kept\n\nbody\n"
        assert [s.heading for s in trimmed.structure.sections] == ["Kept"]

    def test_the_render_always_matches_the_structure(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite
        from hindsight_api.engine.reflect.structured_doc import render_document

        for rewritten in (
            '{"sections": [{"heading": "A", "level": 2, "blocks": ["one", "two"]}]}',
            "## A\n\none\n\ntwo\n",
            "",
        ):
            trimmed = _document_from_rewrite(rewritten, "## Kept\n\nbody\n")
            assert trimmed.markdown.strip() == render_document(trimmed.structure).strip()


@pytest.mark.hs_llm_core
@pytest.mark.asyncio
class TestRealModelFillsTheDocumentShape:
    """A real model, asked for a document, produces one that renders.

    The schema tests above all pass while the pipeline is broken: they check the
    shape we *accept*, not the shape a model *sends*. What actually happened is
    that the model sent a bare section, the parse yielded zero sections, and
    reflect raised ReflectNoAnswerError — so the assertion that matters is that a
    real document-mode reflect comes back with content in it.

    Structural asserts, not a judge: section count and non-empty text are
    deterministic properties of any usable answer, whatever the model wrote.
    """

    async def test_document_mode_reflect_returns_a_rendered_document(self, memory_real_llm, request_context):
        bank_id = f"test-doc-shape-{uuid.uuid4().hex[:8]}"
        await memory_real_llm.get_bank_profile(bank_id, request_context=request_context)
        await memory_real_llm.retain_batch_async(
            bank_id=bank_id,
            contents=[
                {"content": "Alice is the team lead and owns quarterly project planning."},
                {"content": "Bob is a senior engineer who reviews every database migration."},
            ],
            request_context=request_context,
        )
        await memory_real_llm.wait_for_background_tasks()

        # A narrow question, because a one-section answer is where the model flattens
        # the wrapper away — a broad one would produce several sections and hide it.
        result = await memory_real_llm.reflect_async(
            bank_id=bank_id,
            query="Who is the team lead?",
            answer_as_document=True,
            request_context=request_context,
        )

        assert result.text.strip(), (
            "document-mode reflect returned no text: the model's 'document' did not parse into "
            "anything renderable, which is the failure this covers (an empty render makes the "
            "mental-model refresh raise and discard the run)"
        )
        # ``document`` is not guaranteed: a model that answers in prose without calling
        # done sends reflect down the forced-synthesis path, which returns text and no
        # structure — legitimate, and handled downstream by splitting the markdown. What
        # is guaranteed is that a document it *did* state parses into something.
        if result.document is not None:
            assert len(result.document.sections) >= 1, (
                "the model stated a document and it parsed to nothing — the shape it sent is not the shape being read"
            )
            assert render_document(result.document).strip()

        await memory_real_llm.delete_bank(bank_id, request_context=request_context)
