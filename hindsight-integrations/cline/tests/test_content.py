"""Content helpers and the per-task transcript accumulator."""

from lib import content


def test_accumulate_and_read_transcript():
    content.append_turn("t1", "user", "first")
    content.append_turn("t1", "assistant", "second")
    msgs = content.read_transcript("t1")
    assert [m["content"] for m in msgs] == ["first", "second"]


def test_append_skips_empty_content():
    content.append_turn("t2", "user", "   ")
    assert content.read_transcript("t2") == []


def test_clear_transcript():
    content.append_turn("t3", "user", "x")
    content.clear_transcript("t3")
    assert content.read_transcript("t3") == []


def test_transcripts_are_isolated_per_task():
    content.append_turn("a", "user", "for-a")
    content.append_turn("b", "user", "for-b")
    assert content.read_transcript("a") == [{"role": "user", "content": "for-a"}]
    assert content.read_transcript("b") == [{"role": "user", "content": "for-b"}]


def test_format_retention_renders_roles_and_strips_memories():
    msgs = [
        {"role": "user", "content": "do the thing <hindsight_memories>secret</hindsight_memories>"},
        {"role": "assistant", "content": "done"},
    ]
    out = content.format_retention(msgs)
    assert "[user]" in out and "[assistant]" in out
    assert "secret" not in out  # injected memories must not be re-stored


def test_strip_memory_tags():
    assert content.strip_memory_tags("a<hindsight_memories>x</hindsight_memories>b") == "ab"


def test_format_memories():
    out = content.format_memories([{"text": "Acme uses SOC2", "type": "world", "mentioned_at": "2026-01-01"}])
    assert out == "- Acme uses SOC2 [world] (2026-01-01)"


def test_format_memories_empty():
    assert content.format_memories([]) == ""


def test_compose_recall_query_single_turn_is_latest_only():
    msgs = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}]
    assert content.compose_recall_query("latest", msgs, 1) == "latest"


def test_compose_recall_query_multi_turn_includes_prior():
    msgs = [
        {"role": "user", "content": "set up auth"},
        {"role": "assistant", "content": "used JWT"},
        {"role": "user", "content": "now add refresh"},
    ]
    out = content.compose_recall_query("now add refresh", msgs, 3)
    assert "Prior context:" in out
    assert "used JWT" in out
    assert out.endswith("now add refresh")


def test_truncate_recall_query_preserves_latest():
    long_prior = "\n".join(f"user: line {i}" for i in range(50))
    query = f"Prior context:\n\n{long_prior}\n\nLATEST"
    out = content.truncate_recall_query(query, "LATEST", 60)
    assert out.endswith("LATEST")
    assert len(out) <= 60
