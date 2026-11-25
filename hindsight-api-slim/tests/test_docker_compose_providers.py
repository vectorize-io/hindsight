"""Every docker-compose recipe names embeddings/reranker providers the server accepts (#6884)."""

import re
from pathlib import Path

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.cross_encoder import create_cross_encoder_from_env
from hindsight_api.engine.embeddings import create_embeddings_from_env

COMPOSE_FILES = sorted(
    (Path(__file__).resolve().parents[2] / "docker" / "docker-compose").glob("*/docker-compose.yaml")
)
# Matches both "KEY: value" and "- KEY=value" environment entries.
_ENTRY = re.compile(r"^\s*-?\s*(HINDSIGHT_API_(?:EMBEDDINGS|RERANKER)_[A-Z0-9_]+)\s*[:=]\s*(.*?)\s*$", re.M)
_DEFAULT = re.compile(r"\$\{[A-Z0-9_]+:-([^}]*)\}")


def test_compose_files_found() -> None:
    assert COMPOSE_FILES


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.parent.name)
def test_compose_providers_are_valid(compose: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in _ENTRY.findall(compose.read_text()):
        monkeypatch.setenv(key, _DEFAULT.sub(r"\1", value).strip("\"'"))
    clear_config_cache()
    create_cross_encoder_from_env()
    create_embeddings_from_env()
