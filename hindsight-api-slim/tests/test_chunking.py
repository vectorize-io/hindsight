"""
Test chunking functionality for large documents.

These assert the EXACT chunk output for small, controlled inputs (so a change
in splitting behavior is caught precisely), plus a few property/scale tests for
large inputs where spelling out every chunk would be unwieldy.
"""

import json

import pytest

from hindsight_api.engine.retain import fact_extraction
from hindsight_api.engine.retain.fact_extraction import chunk_text


def _is_valid_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


def test_chunk_text_small():
    """Text within the budget is returned unchanged, as a single chunk."""
    text = "This is a short text. It should not be chunked."
    assert chunk_text(text, max_chars=1000) == [text]


def test_chunk_text_exact_split():
    """Plain text splits at sentence boundaries — exact chunks."""
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta sentence four."

    chunks = chunk_text(text, max_chars=40)

    assert chunks == [
        "Alpha sentence one. Beta sentence two",
        ". Gamma sentence three",
        ". Delta sentence four.",
    ]
    # Sentence-boundary splitting here is lossless: concatenation rebuilds the input.
    assert "".join(chunks) == text


def test_chunk_text_large():
    """Test that large text is chunked at sentence boundaries."""
    # Create a text with 10 sentences of ~100 chars each
    sentences = [f"This is sentence number {i}. " + "x" * 80 for i in range(10)]
    text = " ".join(sentences)

    # Chunk with max 300 chars - should create multiple chunks
    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1, "Large text should be chunked"

    # Verify all chunks are under the limit
    for chunk in chunks:
        assert len(chunk) <= 300, f"Chunk exceeds max_chars: {len(chunk)}"

    # Verify we didn't lose any content
    combined = " ".join(chunks)
    # Account for possible whitespace differences
    assert len(combined.replace(" ", "")) >= len(text.replace(" ", "")) * 0.95


def test_chunk_text_64k():
    """Test chunking a 64k character text (like a podcast transcript)."""
    # Create a 64k character text
    sentence = "This is a typical podcast conversation sentence. "
    text = sentence * (64000 // len(sentence))

    chunks = chunk_text(text, max_chars=120000)

    # Should create at least 1 chunk (if text fits) or more
    assert len(chunks) >= 1

    # All chunks should be under the limit
    for chunk in chunks:
        assert len(chunk) <= 120000, f"Chunk exceeds max_chars: {len(chunk)}"

    # Verify we didn't lose content
    combined_length = sum(len(chunk) for chunk in chunks)
    assert combined_length >= len(text) * 0.95, "Lost too much content during chunking"


# ---------------------------------------------------------------------------
# JSONL (newline-delimited JSON objects)
# ---------------------------------------------------------------------------


def test_chunk_jsonl_small():
    """JSONL that fits in one chunk is returned unchanged."""
    lines = [json.dumps({"role": "user", "content": f"message {i}"}) for i in range(3)]
    text = "\n".join(lines)

    assert chunk_text(text, max_chars=10000) == [text]


def test_chunk_jsonl_packs_multiple_short_lines():
    """Short JSONL lines are packed together — exact chunk boundaries."""
    lines = [json.dumps({"i": i}) for i in range(6)]  # each '{"i": N}' is 8 chars
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=40)

    # Four lines (8 chars + newline = 9 each -> 36) fit; the fifth would hit 45 > 40.
    assert chunks == [
        '{"i": 0}\n{"i": 1}\n{"i": 2}\n{"i": 3}',
        '{"i": 4}\n{"i": 5}',
    ]


def test_chunk_jsonl_one_line_per_chunk():
    """When two lines don't fit together, each lands in its own chunk."""
    lines = [json.dumps({"k": "a" * 10}) for _ in range(3)]  # 19 chars each
    text = "\n".join(lines)

    # Budget 25: one line (20 w/ newline) fits, two (40) don't.
    assert chunk_text(text, max_chars=25) == lines


def test_chunk_jsonl_splits_at_line_boundaries():
    """Large JSONL is chunked at line boundaries without splitting any line."""
    lines = [json.dumps({"role": "user", "content": f"message {i} " + "x" * 80}) for i in range(10)]
    text = "\n".join(lines)

    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1, "Large JSONL should be chunked"

    # Every line across all chunks must remain a complete, parseable JSON object.
    seen = []
    for chunk in chunks:
        for line in chunk.split("\n"):
            seen.append(json.loads(line))  # raises if a line was split mid-object
    assert seen == [json.loads(line) for line in lines], "Lines must be preserved in order"


def test_chunk_jsonl_default_structured_unit_limit_matches_budget():
    """An oversized JSONL scalar is split into valid records within the budget."""
    big = json.dumps({"c": "y" * 20})  # 29 chars; budget 25 -> split
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])

    chunks = chunk_text(text, max_chars=25)

    assert "".join(json.loads(chunk)["c"] for chunk in chunks[:-1]) == "y" * 20
    assert chunks[-1] == small
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_chunk_jsonl_custom_structured_unit_limit_keeps_overflow_whole():
    """A JSONL line over the budget is kept whole when the explicit cap allows it."""
    big = json.dumps({"c": "y" * 20})  # 29 chars
    small = json.dumps({"c": "ok"})
    text = "\n".join([big, small])

    chunks = chunk_text(text, max_chars=25, structured_chunk_size=len(big))

    assert chunks == [big, small]


def test_chunk_structured_unit_limit_above_chunk_size_preserves_small_overflows():
    """Structured units between max_chars and the structured cap remain intact."""
    jsonl_line = json.dumps({"c": "y" * 20})  # 29 chars; over budget 25, within cap 29
    conversation = json.dumps([{"c": "y" * 20}])

    jsonl_chunks = chunk_text(
        "\n".join([jsonl_line, json.dumps({"c": "ok"})]),
        max_chars=25,
        structured_chunk_size=29,
    )
    conversation_chunks = chunk_text(conversation, max_chars=25, structured_chunk_size=len(conversation))

    assert jsonl_chunks[0] == jsonl_line
    assert conversation_chunks == [conversation]


def test_chunk_jsonl_structured_unit_limit_can_be_below_chunk_size():
    """An oversized JSONL line is split by the structured cap, not the larger chunk budget."""
    huge = json.dumps({"c": "y" * 40})  # 49 chars; over cap 20 but under budget 55
    small = json.dumps({"c": "ok"})
    text = "\n".join([huge, small])

    chunks = chunk_text(text, max_chars=55, structured_chunk_size=20)

    assert "".join(json.loads(chunk)["c"] for chunk in chunks[:-1]) == "y" * 40
    assert chunks[-1] == small
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_jsonl_huge_line_is_split():
    """A JSONL line past the structured cap keeps its JSON envelope."""
    huge = json.dumps({"c": "y" * 40})  # 49 chars; budget/cap 20 -> must split
    small = json.dumps({"c": "ok"})
    text = "\n".join([huge, small])

    chunks = chunk_text(text, max_chars=20)

    assert "".join(json.loads(chunk)["c"] for chunk in chunks[:-1]) == "y" * 40
    assert chunks[-1] == small
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_jsonl_oversized_tool_call_preserves_nested_envelope():
    """Every tool-call fragment keeps actor, time, identity, and path context."""
    payload = "write payload. " * 30
    record = {
        "type": "message",
        "id": "msg-1",
        "timestamp": "2026-06-20T01:54:38.248Z",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "toolCall",
                    "id": "functions.write:21",
                    "name": "write",
                    "arguments": {"path": "/project/notes.md", "content": payload},
                },
                {"type": "text", "text": "after"},
            ],
        },
    }
    small = {"type": "message", "message": {"role": "user", "content": "done"}}
    chunks = chunk_text(
        "\n".join([json.dumps(record), json.dumps(small)]),
        max_chars=320,
    )

    oversized_fragments = [json.loads(chunk) for chunk in chunks[:-1]]
    content_items = []
    for fragment in oversized_fragments:
        assert fragment["type"] == "message"
        assert fragment["id"] == "msg-1"
        assert fragment["timestamp"] == "2026-06-20T01:54:38.248Z"
        assert fragment["message"]["role"] == "assistant"
        content_items.extend(fragment["message"]["content"])

    assert content_items[0] == {"type": "text", "text": "before"}
    assert content_items[-1] == {"type": "text", "text": "after"}
    reconstructed = []
    for tool_call in content_items[1:-1]:
        assert tool_call["id"] == "functions.write:21"
        assert tool_call["arguments"]["path"] == "/project/notes.md"
        reconstructed.append(tool_call["arguments"]["content"])

    assert "".join(reconstructed) == payload
    assert json.loads(chunks[-1]) == small
    assert all(len(chunk) <= 320 for chunk in chunks)


# ---------------------------------------------------------------------------
# JSON conversation array
# ---------------------------------------------------------------------------


def test_chunk_conversation_packs_turns():
    """A conversation array packs whole turns per chunk — exact JSON-array chunks."""
    turns = [
        {"r": "u", "c": "hi"},
        {"r": "a", "c": "yo"},
        {"r": "u", "c": "bye"},
        {"r": "a", "c": "ok"},
    ]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=50)

    assert chunks == [
        '[{"r": "u", "c": "hi"}, {"r": "a", "c": "yo"}]',
        '[{"r": "u", "c": "bye"}, {"r": "a", "c": "ok"}]',
    ]
    # Each chunk is itself a valid JSON array of complete turns.
    assert [json.loads(c) for c in chunks] == [turns[:2], turns[2:]]


def test_chunk_conversation_splits_at_turn_boundaries():
    """A large conversation array chunks at turn boundaries, keeping turns whole."""
    turns = [{"role": "user", "content": f"message {i} " + "x" * 80} for i in range(10)]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=300)

    assert len(chunks) > 1
    seen = []
    for chunk in chunks:
        parsed = json.loads(chunk)
        assert isinstance(parsed, list)
        seen.extend(parsed)
    assert seen == turns


def test_chunk_conversation_custom_structured_unit_limit_keeps_overflow_whole():
    """A conversation turn over the budget is kept whole when the explicit cap allows it."""
    turns = [{"c": "y" * 20}, {"c": "ok"}]
    text = json.dumps(turns)
    turn_size = len(json.dumps([turns[0]]))

    chunks = chunk_text(text, max_chars=25, structured_chunk_size=turn_size)

    assert chunks == [
        '[{"c": "yyyyyyyyyyyyyyyyyyyy"}]',
        '[{"c": "ok"}]',
    ]


def test_chunk_conversation_array_wrapper_counts_toward_budget():
    """The single-turn array wrapper cannot push a chunk over the limit."""
    turn = {"c": "y" * 20}
    max_chars = len(json.dumps(turn))

    chunks = chunk_text(json.dumps([turn]), max_chars=max_chars)

    assert "".join(json.loads(chunk)[0]["c"] for chunk in chunks) == turn["c"]
    assert all(len(chunk) <= max_chars for chunk in chunks)


def test_chunk_conversation_structured_unit_limit_can_be_below_chunk_size():
    """An oversized conversation turn is split by the structured cap, not the larger chunk budget."""
    turns = [{"c": "y" * 40}, {"c": "ok"}]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=55, structured_chunk_size=20)

    assert "".join(json.loads(chunk)[0]["c"] for chunk in chunks[:-1]) == "y" * 40
    assert json.loads(chunks[-1]) == [{"c": "ok"}]
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_conversation_huge_turn_is_split():
    """A single turn past the structured cap keeps its array and role envelope."""
    turns = [{"c": "y" * 40}, {"c": "ok"}]
    text = json.dumps(turns)

    chunks = chunk_text(text, max_chars=20)

    assert "".join(json.loads(chunk)[0]["c"] for chunk in chunks[:-1]) == "y" * 40
    assert json.loads(chunks[-1]) == [{"c": "ok"}]
    for chunk in chunks:
        assert len(chunk) <= 20


def test_chunk_conversation_oversized_turn_preserves_role_and_timestamp():
    """Conversation fragments stay valid arrays with the original scalar envelope."""
    content = "alpha sentence. beta sentence. " * 15
    turns = [
        {"role": "assistant", "timestamp": "2026-06-15T17:22:29.716Z", "content": content},
        {"role": "user", "content": "done"},
    ]

    chunks = chunk_text(json.dumps(turns), max_chars=170)

    split_turns = [json.loads(chunk)[0] for chunk in chunks[:-1]]
    assert all(turn["role"] == "assistant" for turn in split_turns)
    assert all(turn["timestamp"] == "2026-06-15T17:22:29.716Z" for turn in split_turns)
    assert "".join(turn["content"] for turn in split_turns) == content
    assert json.loads(chunks[-1]) == [{"role": "user", "content": "done"}]
    assert all(len(chunk) <= 170 for chunk in chunks)


def test_structured_string_escaping_respects_serialized_budget():
    """Quotes and backslashes are budgeted after JSON escaping, not by raw length."""
    content = 'path="C:\\\\work\\\\file". ' * 20
    text = json.dumps({"role": "assistant", "content": content})

    chunks = chunk_text(text, max_chars=95)

    parsed = [json.loads(chunk) for chunk in chunks]
    assert all(fragment["role"] == "assistant" for fragment in parsed)
    assert "".join(fragment["content"] for fragment in parsed) == content
    assert all(len(chunk) <= 95 for chunk in chunks)


def test_escaped_unicode_normalizes_to_one_valid_record():
    """A compact non-ASCII rendering is kept structured when it fits."""
    record = {"role": "user", "content": "你好"}
    text = json.dumps(record)
    normalized = json.dumps(record, ensure_ascii=False)

    chunks = chunk_text(text, max_chars=len(normalized))

    assert len(text) > len(normalized)
    assert chunks == [normalized]
    assert chunk_text(chunks[0], max_chars=len(normalized)) == chunks


def test_whitespace_heavy_json_normalizes_to_one_valid_record():
    """Insignificant source whitespace does not force a plain-text split."""
    record = {"role": "user", "content": "ok"}
    normalized = json.dumps(record, ensure_ascii=False)
    text = '  {  "role"  :  "user",  "content"  :  "ok"  }  '

    chunks = chunk_text(text, max_chars=len(normalized))

    assert len(text) > len(normalized)
    assert chunks == [normalized]
    assert chunk_text(chunks[0], max_chars=len(normalized)) == chunks


def test_large_structured_string_stays_lossless_and_idempotent():
    """Large payloads scale across many fragments without losing structure."""
    content = "large structured payload. " * 4000
    text = json.dumps({"role": "assistant", "timestamp": "2026-06-15T17:22:29.716Z", "content": content})

    chunks = chunk_text(text, max_chars=600)

    parsed = [json.loads(chunk) for chunk in chunks]
    assert len(chunks) > 100
    assert all(fragment["role"] == "assistant" for fragment in parsed)
    assert all(fragment["timestamp"] == "2026-06-15T17:22:29.716Z" for fragment in parsed)
    assert "".join(fragment["content"] for fragment in parsed) == content
    assert all(len(chunk) <= 600 for chunk in chunks)
    assert all(chunk_text(chunk, max_chars=600) == [chunk] for chunk in chunks)


def test_multiple_payload_strings_keep_plain_text_fallback():
    """A generic name is not duplicated as presumed envelope data."""
    text = json.dumps({"name": "Ada", "answer": "a" * 500})

    chunks = chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


def test_oversized_list_item_without_unique_id_keeps_plain_text_fallback():
    """One array element is never expanded into ambiguous anonymous parts."""
    text = json.dumps({"role": "assistant", "content": [{"type": "text", "text": "x" * 500}]})

    chunks = chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


def test_unique_list_item_identity_is_preserved_across_fragments():
    """A unique identity can authorize splitting another field, but never itself."""
    payload = "x" * 500
    text = json.dumps({"content": [{"id": "item-1", "payload": payload}]})

    chunks = chunk_text(text, max_chars=120)

    items = [json.loads(chunk)["content"][0] for chunk in chunks]
    assert all(item["id"] == "item-1" for item in items)
    assert "".join(item["payload"] for item in items) == payload
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_oversized_identity_only_item_keeps_plain_text_fallback():
    """A stable identity is context, not a payload that may be fragmented."""
    text = json.dumps({"content": [{"id": "x" * 500}]})

    chunks = chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([{"id": "x" * 500}, {"id": "small"}]),
        "\n".join([json.dumps({"id": "x" * 500}), json.dumps({"id": "small"})]),
    ],
    ids=["conversation", "jsonl"],
)
def test_top_level_identity_is_never_split_as_payload(text):
    """Conversation and JSONL entry points both protect identity fields."""
    chunks = chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


def test_oversized_list_identity_counts_are_built_once(monkeypatch):
    """Identity lookup remains linear when many oversized siblings are split."""
    original = fact_extraction._count_stable_item_identities
    calls = 0

    def counted(items):
        nonlocal calls
        calls += 1
        return original(items)

    monkeypatch.setattr(fact_extraction, "_count_stable_item_identities", counted)
    records = [{"id": f"item-{index}", "payload": "x" * 100} for index in range(25)]

    chunks = chunk_text(json.dumps({"content": records}), max_chars=90)

    assert calls == 1
    assert all(_is_valid_json(chunk) for chunk in chunks)


def test_thin_payload_budget_keeps_plain_text_fallback():
    """A nearly full envelope cannot amplify one character into one record."""
    envelope = {"role": "r" * 70, "content": ""}
    max_chars = len(json.dumps(envelope)) + 1
    text = json.dumps({**envelope, "content": "x" * 500})

    chunks = chunk_text(text, max_chars=max_chars)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


def test_many_small_array_items_preserve_boundaries():
    """Large arrays are packed without rebuilding the growing prefix per item."""
    items = [str(index % 10) for index in range(10_000)]
    text = json.dumps({"role": "assistant", "content": items})

    chunks = chunk_text(text, max_chars=20_000)

    parsed = [json.loads(chunk) for chunk in chunks]
    assert len(chunks) > 1
    assert all(fragment["role"] == "assistant" for fragment in parsed)
    assert [item for fragment in parsed for item in fragment["content"]] == items
    assert all(len(chunk) <= 20_000 for chunk in chunks)


def test_ambiguous_structured_record_keeps_plain_text_fallback():
    """Multiple nested payload branches are not duplicated heuristically."""
    text = json.dumps({"left": ["x" * 80], "right": ["y" * 80]})

    chunks = chunk_text(text, max_chars=45)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


def test_deep_structured_record_keeps_bounded_plain_text_fallback():
    """Deeply nested input cannot force unbounded structured recursion."""
    record = {"payload": "x" * 500}
    for _ in range(40):
        record = {"nested": record}

    chunks = chunk_text(json.dumps(record), max_chars=120)

    assert len(chunks) > 1
    assert not all(_is_valid_json(chunk) for chunk in chunks)


# ---------------------------------------------------------------------------
# Detection guard
# ---------------------------------------------------------------------------


def test_plain_text_lines_not_treated_as_jsonl():
    """Plain (non-JSON) lines fall back to text splitting, not JSONL chunking."""
    text = "\n".join(["Line one here.", "Line two here.", "Line three now."])

    chunks = chunk_text(text, max_chars=20)

    # Each line fits the budget, so text splitting emits one line per chunk.
    assert chunks == ["Line one here.", "Line two here.", "Line three now."]
    # Sanity: these are not JSON objects (so the JSONL path correctly declined).
    with pytest.raises(json.JSONDecodeError):
        json.loads(chunks[0])


# ---------------------------------------------------------------------------
# Idempotency — re-chunking a produced chunk must be a no-op (issue #2301)
# ---------------------------------------------------------------------------
#
# The streaming retain pipeline pre-chunks each document once (producer) and then
# re-chunks every piece during extraction (consumer), stamping all sub-chunks of
# one piece with that piece's single chunk_index. If a piece re-split, its
# sub-chunks would derive the same chunk_id = {bank}_{doc}_{index} and the
# ON CONFLICT upsert would fail with CardinalityViolationError. This can only
# happen when structured_chunk_size > max_chars (a chunk legitimately exceeds the
# re-chunk budget); the defaults (structured == max_chars) never trip it.


def _assert_idempotent(text: str, *, max_chars: int, structured_chunk_size: int) -> list[str]:
    chunks = chunk_text(text, max_chars=max_chars, structured_chunk_size=structured_chunk_size)
    for chunk in chunks:
        rechunked = chunk_text(chunk, max_chars=max_chars, structured_chunk_size=structured_chunk_size)
        assert rechunked == [chunk], (
            f"re-chunking a produced chunk split it again ({len(chunk)} chars -> "
            f"{len(rechunked)} pieces) — not idempotent (issue #2301)"
        )
    return chunks


def test_conversation_turn_over_chunk_size_is_rechunk_stable():
    """A conversation turn larger than max_chars but kept whole by the larger
    structured cap must survive a re-chunk unchanged (issue #2301)."""
    content = json.dumps([{"role": "assistant", "content": "x" * 6000}])

    _assert_idempotent(content, max_chars=3000, structured_chunk_size=5000)


def test_jsonl_line_over_chunk_size_is_rechunk_stable():
    """A single oversized JSONL line, kept whole within the structured cap, must
    not be re-split when handed back through chunk_text (issue #2301)."""
    text = "\n".join([json.dumps({"event": "x" * 3800}), json.dumps({"event": "small"})])

    _assert_idempotent(text, max_chars=3000, structured_chunk_size=4500)


def test_oversized_unit_fragments_stay_within_chunk_budget():
    """A unit past even the structured cap is fragmented as text; no fragment may
    exceed max_chars, so a re-chunk leaves the fragments intact (issue #2301)."""
    text = "\n".join([json.dumps({"event": "z" * 9000}), json.dumps({"e": "s"})])

    chunks = _assert_idempotent(text, max_chars=3000, structured_chunk_size=4500)

    assert all(len(c) <= 3000 for c in chunks)


def test_single_json_object_kept_whole_within_structured_cap():
    """A lone JSON object over max_chars but within the structured cap is returned
    whole rather than plain-text-split (the basis of re-chunk stability)."""
    obj = json.dumps({"role": "assistant", "content": "x" * 4000})

    assert chunk_text(obj, max_chars=3000, structured_chunk_size=5000) == [obj]


def test_rechunk_preserves_one_chunk_id_per_pre_chunk():
    """End-to-end of the producer/consumer chunk_id derivation: each pre-chunk
    (one global index) must re-chunk to exactly one piece, so the derived
    chunk_ids stay unique within an upsert batch (issue #2301)."""
    content = json.dumps([{"role": "assistant", "content": "x" * 6000}])

    pre_chunks = chunk_text(content, max_chars=3000, structured_chunk_size=5000)
    chunk_ids = []
    for global_idx, pre in enumerate(pre_chunks):
        for _ in chunk_text(pre, max_chars=3000, structured_chunk_size=5000):
            chunk_ids.append(f"bank_doc_{global_idx}")

    assert len(chunk_ids) == len(set(chunk_ids)), f"duplicate chunk_ids in one batch: {chunk_ids}"


# ---------------------------------------------------------------------------
# Append-mode JSON array merge simulation (issue #2409)
# ---------------------------------------------------------------------------


def test_newline_joined_json_arrays_bypass_conversation_chunking():
    """Newline-joined JSON arrays (the pre-fix append-mode storage format)
    fail both the conversation and JSONL detection paths and fall through
    to sentence-boundary text splitting.

    This test documents the broken state that issue #2409 fixes at the
    orchestrator level. chunk_text() itself is not changed; the fix
    merges the arrays before they reach chunk_text().
    """
    turn1 = json.dumps([{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}])
    turn2 = json.dumps([{"role": "user", "content": "How are you"}, {"role": "assistant", "content": "Fine"}])
    corrupted = turn1 + "\n" + turn2

    chunks = chunk_text(corrupted, max_chars=80)

    # The corrupted format does NOT route through _chunk_conversation.
    # At least one chunk will not be a valid JSON array of dicts.
    has_non_json_chunk = False
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
            if not (isinstance(parsed, list) and all(isinstance(e, dict) for e in parsed)):
                has_non_json_chunk = True
        except json.JSONDecodeError:
            has_non_json_chunk = True
    assert has_non_json_chunk, (
        "Newline-joined JSON arrays should NOT produce valid conversation chunks. "
        "If this fails, chunk_text() learned to handle the format and the "
        "orchestrator-level merge in #2409 may be redundant."
    )


def test_merged_json_array_routes_to_conversation_chunking():
    """A properly merged flat JSON array (the post-fix format) routes
    through _chunk_conversation and produces chunks that are each valid
    JSON arrays of complete message dicts.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "How are you"},
        {"role": "assistant", "content": "Fine, thanks for asking"},
    ]
    text = json.dumps(messages)

    chunks = chunk_text(text, max_chars=120)

    assert len(chunks) > 1, "Should produce multiple chunks at this budget"
    for chunk in chunks:
        parsed = json.loads(chunk)
        assert isinstance(parsed, list), f"Chunk must be a JSON array: {chunk[:60]}"
        assert all(isinstance(e, dict) for e in parsed), f"Every element must be a dict: {chunk[:60]}"
        assert all("role" in e for e in parsed), f"Every element must have a role key: {chunk[:60]}"
