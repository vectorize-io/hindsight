"""Turn boundaries must agree with what the conversation extractor keeps.

`slice_last_turns_by_user_boundary()` counted every `role == "user"` record as a
turn boundary. But `_extract_text_content()` already discards tool results as
"operational results, not conversation", and the recall composer additionally
strips channel envelopes and injected memory blocks. So a window could be filled
entirely by user-role records that contribute nothing, exhausting the requested
turn count before reaching any actual conversation.

In Claude Code this is the common case, not an edge case — tool results are
stored as `role: "user"` messages. Measured on a real 1,639-message transcript,
566 user-role records of which 514 (91%) were tool results:

    recallContextTurns=3  ->  1 conversational turn,     82 chars   (before)
    recallContextTurns=3  ->  3 conversational turns, 8,081 chars   (after)

Boundary selection is the only thing that changes. Records *inside* the selected
window are returned untouched, so retention/formatting policy still decides what
is kept from them.
"""

from lib.content import compose_recall_query, slice_last_turns_by_user_boundary

TOOL_RESULT = {"role": "user", "content": [{"type": "tool_result", "content": "exit 0"}]}
MEMORY_ONLY = {"role": "user", "content": "<hindsight_memories>prior stuff</hindsight_memories>"}


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


class TestBoundaryQualification:
    def test_tool_result_only_record_is_not_a_boundary(self):
        # Intention: the core defect — tool results are role="user" in Claude Code.
        # Expected: the single real turn is found past the tool results, and the
        # returned window still contains them.
        msgs = [_user("the real question"), _assistant("the real answer"), TOOL_RESULT, TOOL_RESULT]
        out = slice_last_turns_by_user_boundary(msgs, 1)
        assert out[0] == msgs[0], "window must start at the conversational user message"
        assert len(out) == 4, "records inside the window must be preserved, including tool results"

    def test_plain_string_user_content_is_a_boundary(self):
        # Intention: non-Claude-Code hosts and the flat test format send plain strings.
        # Expected: unchanged behaviour.
        msgs = [_user("first"), _assistant("a"), _user("second"), _assistant("b")]
        assert slice_last_turns_by_user_boundary(msgs, 1)[0]["content"] == "second"

    def test_text_block_user_content_is_a_boundary(self):
        # Intention: structured text blocks are real conversation.
        msgs = [_user("older"), {"role": "user", "content": [{"type": "text", "text": "newer"}]}]
        out = slice_last_turns_by_user_boundary(msgs, 1)
        assert out[0]["content"] == [{"type": "text", "text": "newer"}]

    def test_empty_and_whitespace_user_records_are_not_boundaries(self):
        # Intention: an empty record contributes nothing, so it must not consume a turn.
        msgs = [_user("real question"), _assistant("real answer"), _user("   "), _user("")]
        out = slice_last_turns_by_user_boundary(msgs, 1)
        assert out[0]["content"] == "real question"

    def test_memory_only_user_record_is_not_a_boundary(self):
        # Intention: a UserPromptSubmit hook injects <hindsight_memories> into the user
        # message. A record that is ONLY that injection is not a turn — the composer
        # strips it and it contributes nothing.
        msgs = [_user("real question"), _assistant("real answer"), MEMORY_ONLY]
        out = slice_last_turns_by_user_boundary(msgs, 1)
        assert out[0]["content"] == "real question"

    def test_memory_tags_plus_real_text_is_a_boundary(self):
        # Intention: don't over-correct — injected memory alongside a real prompt is
        # still a real prompt.
        mixed = _user("<hindsight_memories>prior</hindsight_memories>\nactual question")
        msgs = [_user("older"), _assistant("a"), mixed]
        assert slice_last_turns_by_user_boundary(msgs, 1)[0] is mixed

    def test_mixed_text_and_tool_result_blocks_count_once(self):
        # Intention: a record carrying both a tool result and genuine text is a turn.
        mixed = {
            "role": "user",
            "content": [{"type": "tool_result", "content": "exit 0"}, {"type": "text", "text": "and also this"}],
        }
        msgs = [_user("older"), _assistant("a"), mixed]
        assert slice_last_turns_by_user_boundary(msgs, 1)[0] is mixed

    def test_counts_n_conversational_turns_across_tool_noise(self):
        # Intention: the headline behaviour. Tool results between prompts must not
        # consume the turn budget.
        # Expected: N=2 reaches the second-to-last real prompt, not the last one.
        msgs = [
            _user("turn one"),
            _assistant("answer one"),
            TOOL_RESULT,
            _user("turn two"),
            _assistant("answer two"),
            TOOL_RESULT,
            TOOL_RESULT,
            _user("turn three"),
            _assistant("answer three"),
        ]
        out = slice_last_turns_by_user_boundary(msgs, 2)
        assert out[0]["content"] == "turn two"
        assert any(m is TOOL_RESULT for m in out), "intervening tool records must be preserved"

    def test_fewer_boundaries_than_requested_returns_everything(self):
        # Intention: preserve the existing fallback when N turns don't exist.
        msgs = [TOOL_RESULT, _user("only one real turn"), _assistant("a")]
        assert slice_last_turns_by_user_boundary(msgs, 5) == msgs

    def test_all_tool_results_returns_everything(self):
        # Intention: with no conversational boundary at all, fall back rather than
        # returning an empty window.
        msgs = [TOOL_RESULT, TOOL_RESULT]
        assert slice_last_turns_by_user_boundary(msgs, 2) == msgs


class TestRecallQueryComposition:
    def test_recall_context_reaches_prior_turns_through_tool_noise(self):
        # Intention: end-to-end — the recall query must carry real prior turns even
        # when the transcript is dominated by tool results.
        # Expected: earlier conversation present, tool-result text absent.
        msgs = [
            _user("why did the flame sensor trip"),
            _assistant("the 74HC14 ground went marginal"),
            TOOL_RESULT,
            TOOL_RESULT,
            _user("what fixed it"),
            _assistant("resoldering VCC and GND"),
            TOOL_RESULT,
        ]
        q = compose_recall_query("and what about the water sensor", msgs, 3, ["user", "assistant"])
        assert "why did the flame sensor trip" in q
        assert "74HC14 ground went marginal" in q
        assert "exit 0" not in q, "tool-result content must not enter the recall query"
        assert q.endswith("and what about the water sensor")

    def test_injected_memory_blocks_stay_out_of_the_query(self):
        # Intention: recall must not feed its own previous output back into itself.
        msgs = [_user("real prior question"), _assistant("real prior answer"), MEMORY_ONLY]
        q = compose_recall_query("the latest question", msgs, 3, ["user", "assistant"])
        assert "real prior question" in q
        assert "prior stuff" not in q
        assert "<hindsight_memories>" not in q
