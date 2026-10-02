"""Profile env sync cases from a production ``local_embedded`` host.

Ported from @mabaty's setup in #4694: the profile env file is written on one side (Windows) and
read back on the other (POSIX), hand-edited files happen, and a rewrite can truncate the file a
second provider init is reading. Those are the shapes where a fixture and a live host disagree.

Out of scope here: the write path itself. Operator-owned keys surviving a *genuine* drift rewrite
(``_secure_write_profile_env`` opens ``O_TRUNC``) is the other half of the hotspot and is pinned
red on #4662, not on this PR.
"""

from hindsight_hermes import embedded

CONFIG = {
    "profile": "taller",
    "llm_provider": "openai_compatible",
    "llm_model": "some-model",
    "llmApiKey": "test-key",
    "llm_base_url": "https://llm.example/v1",
}


def _materialize(config: dict | None = None):
    """Write the profile env the way the plugin does, then return its path."""
    return embedded._materialize_embedded_profile_env(dict(config or CONFIG))


def test_a_crlf_body_from_a_windows_write_still_compares_equal(hermes_env):
    """An editor rewriting the file with CRLF must not start a rewrite loop."""
    path = _materialize()
    body = path.read_text(encoding="utf-8")
    path.write_bytes(body.replace("\n", "\r\n").encode("utf-8"))

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False


def test_a_bom_on_the_file_is_not_a_mismatch(hermes_env):
    """Pins the ``utf-8-sig`` read as load-bearing.

    A plain ``utf-8`` read would glue the BOM onto the first key's name
    (``\\ufeffHINDSIGHT_API_LLM_PROVIDER``) and the file would drift forever.
    """
    path = _materialize()
    body = path.read_text(encoding="utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False


def test_duplicate_governed_lines_are_last_wins(hermes_env):
    """A hand-edited file with the key twice: only the last line counts.

    Both directions are pinned because the parser is last-wins: a stale value *after* the good
    line is drift, a stale value *before* it is not (it resolves to the good one).
    """
    path = _materialize()
    lines = path.read_text(encoding="utf-8").splitlines()
    at = next(index for index, line in enumerate(lines) if line.startswith("HINDSIGHT_API_LLM_MODEL="))
    lines.insert(at + 1, "HINDSIGHT_API_LLM_MODEL=stale-model")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True

    lines = [line for line in lines if line != "HINDSIGHT_API_LLM_MODEL=stale-model"]
    at = next(index for index, line in enumerate(lines) if line.startswith("HINDSIGHT_API_LLM_MODEL="))
    lines.insert(at, "HINDSIGHT_API_LLM_MODEL=stale-model")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False

    _materialize(dict(CONFIG))
    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False  # settles after one rewrite


def test_a_malformed_line_without_an_equals_sign_reads_as_a_missing_key(hermes_env):
    """A half-written file (key with no value) must drift, and the rewrite repairs it."""
    path = _materialize()
    lines = [
        line.split("=", 1)[0] if line.startswith("HINDSIGHT_API_LLM_MODEL") else line
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True


def test_a_truncated_file_from_a_racing_rewrite_reads_as_drift_and_settles(hermes_env):
    """The truncation window of a racing rewrite, and why it is one-shot rather than a loop.

    One init truncates the file while another reads it: the reader sees the governed keys gone and
    queues its own rewrite. After that rewrite plus the daemon re-appending its own keys, the next
    init is in sync — a single extra restart, not an intermittent restart loop.
    """
    _materialize()
    path = embedded._embedded_profile_env_path(dict(CONFIG))
    path.write_text("HINDSIGHT_API_PORT=9177\n", encoding="utf-8")  # the truncation window

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True

    _materialize(dict(CONFIG))
    with path.open("a", encoding="utf-8") as fh:
        fh.write("HINDSIGHT_API_PORT=9177\n")  # what the daemon does on start

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False
