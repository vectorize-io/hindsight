"""MCP tool error payloads must be valid JSON, whatever the message contains.

The bank-id-parameter variants of the MCP tools declare ``-> str`` and return
JSON text, so their error branch has to return JSON too. Every one of them used
to build it by interpolation::

    return f'{{"error": "{e}"}}'

which emits invalid JSON as soon as the exception message contains a double
quote, a backslash or a newline. That is not a corner case: PostgreSQL quotes
identifiers with double quotes, so a plain ``relation "memory_units" does not
exist`` already produced something the caller could not parse. The bug lived
only in the bank-id half of each duplicated tool -- the single-bank half
returned a dict and was always correct -- which is exactly why no test caught
it.
"""

import json

import pytest

from hindsight_api.mcp_tools import _error_json

# Messages a real backend actually produces. The PG ones are the reason this
# matters; the others cover the remaining JSON metacharacters.
HOSTILE_MESSAGES = [
    'relation "memory_units" does not exist',
    'column "tags" is of type text[] but expression is of type text',
    'duplicate key value violates unique constraint "banks_pkey"',
    "back\\slash",
    "line one\nline two",
    'tab\there and "quotes"',
    "unicode: — é 中文",
    "",
]


@pytest.mark.parametrize("message", HOSTILE_MESSAGES)
def test_error_payload_is_parseable_json(message):
    parsed = json.loads(_error_json(Exception(message)))
    assert parsed == {"error": message}


@pytest.mark.parametrize("message", HOSTILE_MESSAGES)
def test_error_payload_keeps_its_empty_collection(message):
    """Tools that return a collection include an empty one so callers keep their shape."""
    parsed = json.loads(_error_json(Exception(message), results=[]))
    assert parsed == {"error": message, "results": []}


def test_error_payload_accepts_a_plain_string():
    """The 'no bank configured' branch passes a str, not an exception."""
    assert json.loads(_error_json("No bank_id configured")) == {"error": "No bank_id configured"}


def test_no_tool_builds_error_json_by_interpolation():
    """Structural guard: the f-string form must not come back.

    A single reintroduced ``f'{{"error": "{e}"}}'`` is invisible until a message
    with a quote reaches that one branch, so this asserts over the whole file
    rather than over any one tool.
    """
    import pathlib
    import re

    source = pathlib.Path(__file__).resolve().parent.parent / "hindsight_api" / "mcp_tools.py"
    text = source.read_text()
    # Skip the docstring that quotes the old form as the thing not to do.
    body = text.replace('``f\'{{"error": "{e}"}}\'``', "")
    offenders = [
        (i + 1, line.strip())
        for i, line in enumerate(body.split("\n"))
        if re.search(r"""return\s+f['"]\{\{["']error""", line)
    ]
    assert not offenders, f"mcp_tools.py builds error JSON by interpolation at {offenders}"
