"""The profile env sync check: only the keys this build governs count as a mismatch.

hindsight-embed appends its own keys (``HINDSIGHT_API_PORT``, embedding config, ...) to the
same profile env file the plugin writes, so a whole-mapping equality check is permanently
unequal — every daemon start rewrote the file and stopped a healthy daemon.
"""

from hindsight_hermes import embedded

CONFIG = {
    "profile": "taller",
    "llm_provider": "openai_compatible",
    "llm_model": "some-model",
    "llmApiKey": "test-key",
    "llm_base_url": "https://llm.example/v1",
}
# Same profile with no optional knob set, and with both of them set: ``llm_base_url`` and
# ``idle_timeout`` are the two keys the build only emits when someone asks for them.
BARE_CONFIG = {key: value for key, value in CONFIG.items() if key != "llm_base_url"}
OPTIONAL_CONFIG = {**CONFIG, "idle_timeout": 900}


def _materialize(config: dict | None = None):
    """Write the profile env the way the plugin does, then return its path."""
    return embedded._materialize_embedded_profile_env(dict(config or CONFIG))


def test_keys_the_build_does_not_own_are_not_a_mismatch(hermes_env):
    """The regression: hindsight-embed's own keys must not trigger a rewrite.

    The file it maintains carries more keys than this build produces (it appends
    ``HINDSIGHT_API_PORT`` on start), so the whole-mapping comparison was always unequal:
    every start rewrote the env and killed the running daemon, losing in-flight retains.
    """
    path = _materialize()
    with path.open("a", encoding="utf-8") as fh:
        fh.write("HINDSIGHT_API_PORT=9176\n")
        fh.write("HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5\n")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False


def test_a_missing_governed_key_is_a_mismatch(hermes_env):
    path = _materialize()
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if "LLM_MODEL" not in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True


def test_a_changed_governed_value_is_a_mismatch(hermes_env):
    """Config drift must still rewrite and restart: the check is not a no-op."""
    _materialize()

    assert embedded._profile_env_out_of_sync({**CONFIG, "llm_model": "another-model"}) is True


def test_an_absent_profile_env_is_a_mismatch(hermes_env):
    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True


def test_whitespace_and_comments_around_governed_keys_are_not_a_mismatch(hermes_env):
    """The parser already strips and ignores lines; the sync check must inherit that."""
    path = _materialize()
    body = path.read_text(encoding="utf-8").replace("=", " = ", 1)
    path.write_text("# managed by hindsight-embed\n" + body, encoding="utf-8")

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False


def test_a_cleared_optional_knob_is_a_mismatch(hermes_env):
    """Dropping a knob from config must not leave its old value live in the file.

    ``llm_base_url`` and ``idle_timeout`` only appear in the build while the knob is set,
    so a check that walks the built mapping alone finds nothing to disagree with — and the
    daemon keeps dialing the old base URL (or honouring the old idle timeout) from the file.
    Those two keys are governed in both directions: absent from the build, absent from the file.
    """
    _materialize(OPTIONAL_CONFIG)

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True


def test_a_cleared_base_url_is_a_mismatch(hermes_env):
    """Same hole on the other optional key: the URL must not survive being unset."""
    _materialize()

    assert embedded._profile_env_out_of_sync(dict(BARE_CONFIG)) is True


def test_every_optional_key_the_build_can_emit_is_governed(hermes_env):
    """Guard: a new optional knob must join the governed set, or its removal escapes."""
    with_knobs = embedded._build_embedded_profile_env(dict(OPTIONAL_CONFIG))
    without_knobs = embedded._build_embedded_profile_env(dict(BARE_CONFIG))

    assert set(with_knobs) - set(without_knobs) == set(embedded._OPTIONAL_PROFILE_ENV_KEYS)


def test_a_cleared_knob_settles_after_one_rewrite(hermes_env):
    """The mismatch is a one-shot, not an every-start rewrite.

    The rewrite drops the key this build no longer governs, so the check is satisfied
    afterwards: fixing the stale-key hole must not trade it for a rewrite-and-restart loop
    on every provider init (which is the bug this PR set out to kill).
    """
    _materialize(OPTIONAL_CONFIG)
    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is True

    _materialize(dict(CONFIG))  # what the daemon-start path does on a mismatch

    assert embedded._profile_env_out_of_sync(dict(CONFIG)) is False
