"""Shared fixtures and helpers for the Continue adapter tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hindsight_continue import reset_config


@pytest.fixture(autouse=True)
def _clean_config():
    """Ensure each test starts and ends with no global config."""
    reset_config()
    yield
    reset_config()


def make_recall_response(texts: list[str]):
    """Build a stand-in RecallResponse with ``results[].text`` like the real client."""
    results = [SimpleNamespace(text=t) for t in texts]
    return SimpleNamespace(results=results)


def make_client(texts: list[str] | None = None) -> MagicMock:
    """A mock Hindsight client whose ``recall`` returns the given memory texts."""
    client = MagicMock()
    client.recall = MagicMock(return_value=make_recall_response(texts or []))
    return client


def continue_request(query: str = "", full_input: str = "", **extra) -> dict:
    """The exact JSON body Continue's ``http`` context provider POSTs.

    Mirrors ``HttpContextProvider.getContextItems`` in continuedev/continue.
    """
    body = {
        "query": query,
        "fullInput": full_input,
        "options": extra.pop("options", {}),
        "workspacePath": extra.pop("workspacePath", "/home/user/project"),
    }
    body.update(extra)
    return body
