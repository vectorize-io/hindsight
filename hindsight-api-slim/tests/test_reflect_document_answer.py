"""The reflect agent can answer with a document instead of a plain answer.

In document mode the closing ``done`` carries the page as one markdown string,
which ``split_markdown`` cuts into sections at its headings and into blocks at
its blank lines, losslessly. It used to be a section array the model filled in
(the answer to #3361, where reading markdown back destroyed tables); held to a
response schema, models wrote that array thinner than the same page as
markdown, and the lossless split keeps tables intact.

Most of these are pure-Python tests of the schema, the prompt text and the
parsing of the tool call — the mechanics. ``TestRealModelFillsTheDocumentShape``
is the exception: whether a real model actually fills the shape is the thing
that broke, and no amount of schema testing can answer it.
"""

from __future__ import annotations

import uuid

import pytest

from hindsight_api.engine.reflect.structured_doc import (
    DocumentSectionsInvalidError,
    render_document,
)
from hindsight_api.engine.reflect.prompts import (
    NO_IDS_IN_TEXT_RULE,
    build_final_system_prompt,
    build_system_prompt_for_tools,
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
        assert "markdown" in params["properties"]
        assert "answer" not in params["properties"]
        assert params["required"][:2] == ["answer_found", "markdown"]

    def test_document_mode_keeps_the_supporting_id_fields(self):
        params = _done_tool(answer_as_document=True)["function"]["parameters"]
        for field in ("memory_ids", "mental_model_ids", "observation_ids"):
            assert field in params["properties"]

    def test_document_mode_composes_with_directives(self):
        params = _done_tool(directive_rules=["Be concise"], answer_as_document=True)["function"]["parameters"]
        assert {"markdown", "answer_found", "directive_compliance"} <= set(params["required"])

    def test_every_variant_requires_answer_found(self):
        """The absence gate in the agent loop reads it (#4567), so no variant may drop it."""
        for kwargs in ({}, {"directive_rules": ["Be concise"]}, {"answer_as_document": True}):
            params = _done_tool(**kwargs)["function"]["parameters"]
            assert params["properties"]["answer_found"]["type"] == "boolean"
            assert "answer_found" in params["required"]

    def test_the_document_keeps_ids_out_of_the_text(self):
        """Dropping ``answer`` dropped its no-ids rule, and cited handles resolve to UUIDs in pages (#5833)."""
        markdown = _done_tool(answer_as_document=True)["function"]["parameters"]["properties"]["markdown"]
        assert NO_IDS_IN_TEXT_RULE in markdown["description"]

    def test_every_text_writing_prompt_keeps_ids_out(self):
        """Document mode, and forced synthesis in both modes, write text a reader sees (#5833)."""
        document_prompt = build_system_prompt_for_tools(bank_profile={"name": "B"}, answer_as_document=True)
        assert NO_IDS_IN_TEXT_RULE in document_prompt
        assert "IDs ONLY in the memory_ids" not in document_prompt, "document mode has no answer field"
        assert NO_IDS_IN_TEXT_RULE in build_final_system_prompt()

    def test_schema_has_no_union_types(self):
        """A tool schema goes to the provider verbatim, and Gemini rejects ``oneOf``."""
        rendered = repr(_done_tool(answer_as_document=True))
        for keyword in ("oneOf", "anyOf", "allOf", "discriminator"):
            assert keyword not in rendered


class TestDocumentPromptStatesTheShape:
    """The prompt names the field the schema requires: one markdown string, a '## ' heading per section."""

    @staticmethod
    def _document_block() -> str:
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(bank_profile={"name": "T", "mission": ""}, answer_as_document=True)
        return prompt[prompt.index("## Output Format: Markdown Document") :]

    def test_the_markdown_field_is_named(self):
        assert "'markdown' field" in self._document_block()

    def test_answer_mode_does_not_carry_the_document_shape(self):
        from hindsight_api.engine.reflect.prompts import build_system_prompt_for_tools

        prompt = build_system_prompt_for_tools(bank_profile={"name": "T", "mission": ""})
        assert "## Output Format: Markdown Document" not in prompt

    def test_the_closing_turn_states_the_page_shape_in_markdown_terms(self):
        """A blank line splits a markdown list into blocks, so the rule says so, not "one block"."""
        from hindsight_api.engine.reflect.prompts import build_done_request_prompt

        page = build_done_request_prompt("Q", as_document=True)
        assert "## Document shape" in page
        assert "no blank lines between its items" in page
        assert "## Document shape" not in build_done_request_prompt("Q")


class TestOverBudgetRewrite:
    """A document too long for its budget is trimmed as a document, or not at all.

    The trim is the one place a long answer gets regenerated, so a response that
    is not a document is refused rather than read back as prose: that would put
    the model in charge of the markdown that gets stored, on exactly the
    documents whose structure matters most, and would let a schema violation
    reach stored content unreported (#4910). The caller re-asks and otherwise
    keeps the full document.
    """

    def test_a_json_rewrite_is_read_back_as_a_document(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        rewritten = (
            '{"sections": [{"heading": "Ops", "level": 2, "blocks": ["| a | b |\\n| --- | --- |\\n| 1 | 2 |"]}]}'
        )
        trimmed = _document_from_rewrite(rewritten)
        assert [s.heading for s in trimmed.structure.sections] == ["Ops"]
        assert trimmed.markdown == "## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |"

    def test_markdown_instead_of_json_is_refused(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        with pytest.raises(DocumentSectionsInvalidError):
            _document_from_rewrite("## Ops\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n")

    def test_object_blocks_are_refused(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        with pytest.raises(DocumentSectionsInvalidError) as exc:
            _document_from_rewrite('{"sections": [{"heading": "Ops", "blocks": [{"text": "x"}]}]}')
        assert "blocks[0]" in str(exc.value)

    def test_empty_rewrite_is_refused(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        with pytest.raises(DocumentSectionsInvalidError):
            _document_from_rewrite("   ")

    def test_sections_that_render_to_nothing_are_refused(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite

        with pytest.raises(DocumentSectionsInvalidError):
            _document_from_rewrite('{"sections": [{"heading": "", "blocks": ["  "]}]}')

    def test_the_render_always_matches_the_structure(self):
        from hindsight_api.engine.reflect.agent import _document_from_rewrite
        from hindsight_api.engine.reflect.structured_doc import render_document

        trimmed = _document_from_rewrite('{"sections": [{"heading": "A", "level": 2, "blocks": ["one", "two"]}]}')
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
        await memory_real_llm.ensure_bank_profile(bank_id, request_context=request_context)
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
