"""A discovered ``.env`` must win even if something already built the config.

``HindsightConfig`` is cached for the process on first build, and an entry point
imports its whole module graph before ``main()`` calls
``load_dotenv_for_entrypoint()``. Modules that read the config at import scope
(``llm_wrapper`` sizes its LLM semaphores there) therefore build it from an
environment the ``.env`` has not been applied to yet.

Without the cache clear in ``load_dotenv_for_entrypoint()`` that stale config stays
for the life of the process and the ``.env`` is silently ignored — the failure mode
being a server that had always started refusing to, with "LLM API key is required",
because the key only ever lived in the ``.env``.

Run in a subprocess: the config cache and ``sys.modules`` are process-global, so an
in-process version would be a no-op once another test has loaded these modules.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_dotenv_applies_even_when_the_config_was_built_before_it_loaded(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("HINDSIGHT_API_LLM_MODEL=model-from-dotenv\n")

    probe = textwrap.dedent("""
        import hindsight_api.config as config

        # Stand in for any module that reads the config at import scope, before the
        # entry point has had a chance to load the .env.
        early = config.get_config().llm_model
        assert config._config_cache is not None, "probe did not actually build the config"

        config.load_dotenv_for_entrypoint()

        print("EARLY:" + early)
        print("AFTER:" + config.get_config().llm_model)
    """)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert "AFTER:model-from-dotenv" in result.stdout, (
        "load_dotenv_for_entrypoint() left a config cached from before the .env was "
        f"applied, so the .env had no effect:\n{result.stdout}"
    )
