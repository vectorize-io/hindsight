"""Starting up must not load code the configuration doesn't use (#5479).

`hindsight-api --help` and every spawned worker build the config. If that pulls in
`hindsight_api.engine` or `sentence_transformers` (and so torch), each one pays ~15 s;
sqlalchemy adds another ~0.5 s. Loading the app itself must not pull in the SDK of
every LLM provider (google-genai alone is ~4 s), nor the tracing exporter while
tracing is off.
"""

import os
import subprocess
import sys

_CONFIG_PROBE = """
import sys
from hindsight_api.config import _get_raw_config
_get_raw_config()
heavy = [m for m in ("hindsight_api.engine", "sentence_transformers", "torch", "sqlalchemy") if m in sys.modules]
assert not heavy, heavy
"""

_APP_PROBE = """
import sys
from hindsight_api.api import create_app
from hindsight_api.engine import MemoryEngine
from hindsight_api.engine import providers
providers.MockLLM  # picking one provider must not load the others
heavy = [
    m
    for m in ("google.genai", "google.oauth2", "openai", "anthropic", "litellm", "opentelemetry.sdk.trace")
    if m in sys.modules
]
assert not heavy, heavy
"""


def _run(probe: str) -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HINDSIGHT_API_")}
    env["HINDSIGHT_API_EMBEDDINGS_PROVIDER"] = "local"
    env["HINDSIGHT_API_RERANKER_PROVIDER"] = "local"
    env["HINDSIGHT_API_LLM_PROVIDER"] = "mock"
    env["HINDSIGHT_API_LLM_1_PROVIDER"] = "mock"
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_building_config_does_not_import_engine_or_ml_stack():
    _run(_CONFIG_PROBE)


def test_loading_app_does_not_import_unused_provider_sdks():
    _run(_APP_PROBE)
