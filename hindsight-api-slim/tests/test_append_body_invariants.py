"""The invariants that keep an append from losing a document (#3989).

Fast, DB-free unit tests for the two rules the retain path now rests on:

- one definition of how a document's parts become its body, so the body the SPLITTER predicts for
  an oversized append and the body ``retain_batch`` actually builds cannot disagree;
- an append is monotonic, so a body that does not extend the stored one is refused rather than
  written over it.

The end-to-end consequences live in ``test_oversized_append_truncates_document.py``; these pin the
mechanics so a future change breaks here first, where the failure is legible.
"""

import json

import pytest

from hindsight_api.engine.retain.orchestrator import (
    AppendWouldTruncateDocument,
    append_document_body,
    assert_append_extends_stored_body,
    join_document_parts,
    merge_json_array_parts,
)


class TestJoinDocumentParts:
    def test_plain_text_parts_join_with_a_newline(self):
        assert join_document_parts(["alpha", "beta"]) == "alpha\nbeta"

    def test_jsonl_is_plain_text_and_joins_with_a_newline(self):
        # JSONL is the coding-agent transcript shape: each LINE is JSON, the document is not.
        a, b = json.dumps({"role": "user"}), json.dumps({"role": "assistant"})
        assert join_document_parts([a, b]) == f"{a}\n{b}"

    def test_json_arrays_merge_into_one_array(self):
        # Newline-joining these would produce "[...]\n[...]", which the next append cannot parse.
        merged = join_document_parts(['[{"a": 1}]', '[{"b": 2}]'])
        assert json.loads(merged) == [{"a": 1}, {"b": 2}]

    def test_a_non_array_part_defeats_the_merge(self):
        assert merge_json_array_parts(['[{"a": 1}]', "not json"]) is None

    def test_an_array_of_scalars_is_not_a_document_array(self):
        assert merge_json_array_parts(["[1, 2]", "[3]"]) is None


class TestAppendDocumentBody:
    def test_the_appended_body_extends_the_stored_one(self):
        assert append_document_body("stored", "tail").startswith("stored")

    def test_it_is_the_same_join_the_retain_path_uses(self):
        # The property the splitter's prediction depends on: predicting the body is exactly
        # joining the parts, so there is no second implementation to drift.
        assert append_document_body("stored", "tail") == join_document_parts(["stored", "tail"])


class TestAppendMonotonicity:
    def test_a_merged_conversation_array_is_allowed(self):
        stored = '[{"role":"user","content":"café"}]'
        body = append_document_body(stored, '[{"role":"assistant","content":"new turn"}]')
        assert not body.startswith(stored)
        assert_append_extends_stored_body(stored, body, document_id="d")

    def test_empty_conversation_array_can_receive_first_turn(self):
        assert_append_extends_stored_body("[]", '[{"content":"first"}]', document_id="d")

    @pytest.mark.parametrize(
        "stored,body",
        [
            pytest.param(
                '[{"role":"user","content":"keep","content":"committed"}]',
                '[{"role":"user","content":"committed"},{"role":"assistant","content":"new"}]',
                id="stored-duplicate-content",
            ),
            pytest.param(
                '[{"content":"keep","content":"keep"}]',
                '[{"content":"keep"},{"content":"new"}]',
                id="stored-duplicate-same-value",
            ),
            pytest.param(
                '[{"metadata":{"tags":[{"name":"keep","name":"committed"}]}}]',
                '[{"metadata":{"tags":[{"name":"committed"}]}},{"content":"new"}]',
                id="stored-nested-duplicate",
            ),
            pytest.param(
                r'[{"content":"keep","\u0063ontent":"committed"}]',
                '[{"content":"committed"},{"content":"new"}]',
                id="stored-escaped-duplicate-key",
            ),
            pytest.param(
                '[{"content":"keep"}]',
                '[{"content":"changed","content":"keep"},{"content":"new"}]',
                id="candidate-duplicate-prefix",
            ),
            pytest.param(
                '[{"content":"keep"}]',
                '[{"content":"keep"},{"metadata":{"name":"first","name":"last"}}]',
                id="candidate-nested-duplicate-tail",
            ),
        ],
    )
    def test_structural_append_with_duplicate_object_keys_is_refused(self, stored: str, body: str):
        with pytest.raises(AppendWouldTruncateDocument):
            assert_append_extends_stored_body(stored, body, document_id="d")

    def test_literal_append_preserves_duplicate_keys_without_parsing(self):
        stored = '[{"content":"keep","content":"committed"}]'
        assert_append_extends_stored_body(stored, stored + "\nnew turn", document_id="d")

    @pytest.mark.parametrize(
        "stored,body",
        [
            ('[{"content":"old"}]', '[{"content":"new"}]'),
            ('[{"content":"old"}]', '[{"content":"changed"},{"content":"new"}]'),
            ('[{"a":1},{"a":2}]', '[{"a":2},{"a":1},{"a":3}]'),
            ('[{"a":1},{"a":2}]', '[{"a":1}]'),
            ('[{"a":1}]', '[{"a":1}]'),
            ('[{"a":true}]', '[{"a":1},{"a":2}]'),
            ('[{"a":1}]', '[{"a":1},2]'),
            ("[1]", "[1,2]"),
            ('[{"a":1}]', '[{"a":1},'),
        ],
    )
    def test_json_replacement_truncation_and_invalid_shapes_are_refused(self, stored: str, body: str):
        with pytest.raises(AppendWouldTruncateDocument):
            assert_append_extends_stored_body(stored, body, document_id="d")

    def test_extending_the_stored_body_is_allowed(self):
        assert_append_extends_stored_body("stored", "stored\ntail", document_id="d")

    def test_no_stored_body_is_allowed(self):
        # The document's first write has nothing to extend.
        assert_append_extends_stored_body(None, "anything", document_id="d")

    def test_the_tail_alone_is_refused(self):
        # Exactly #3989: the splitter reported the new tail as the whole document.
        with pytest.raises(AppendWouldTruncateDocument, match="does not extend"):
            assert_append_extends_stored_body("stored", "tail", document_id="d")

    def test_an_unchanged_body_is_refused(self):
        # An append that adds nothing is not an append; writing it would be a no-op at best and a
        # lost tail at worst, so it is worth hearing about rather than silently accepting.
        with pytest.raises(AppendWouldTruncateDocument):
            assert_append_extends_stored_body("stored", "stored", document_id="d")

    def test_the_message_names_the_document_and_both_sizes(self):
        with pytest.raises(AppendWouldTruncateDocument) as excinfo:
            assert_append_extends_stored_body("stored body", "tail", document_id="conversation:s1")
        assert "conversation:s1" in str(excinfo.value)
