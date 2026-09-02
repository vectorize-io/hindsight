"""Retain ingestion must not store null metadata values (issue #3209).

The retain API accepts arbitrary JSON metadata; a null value (e.g.
{"ocr_engine": null}) stored verbatim poisons the read path, which validates
MemoryFact.metadata as dict[str, str] and made every recall fail for the
affected rows. RetainContent drops null-valued keys at construction, so facts
extracted from it (metadata=content.metadata in fact_extraction) stay canonical
on the write side; the read path drops nulls again for legacy rows.
Non-string values are preserved as-is here and coerced by the read path.

The delta paths, which sync document metadata onto units they preserve rather
than re-extracting them, are covered on a real database by
test_delta_retain.py::test_delta_retain_drops_null_metadata_values.
"""

from hindsight_api.engine.metadata_utils import as_string_metadata, drop_null_values
from hindsight_api.engine.retain.orchestrator import _build_contents
from hindsight_api.engine.retain.types import RetainContent


def test_drop_null_values_keeps_everything_else_untouched():
    assert drop_null_values({"a": None, "b": "x", "c": 5, "d": ""}) == {"b": "x", "c": 5, "d": ""}


def test_drop_null_values_normalizes_absent_metadata_to_empty_dict():
    assert drop_null_values(None) == {}


def test_as_string_metadata_drops_nulls_and_stringifies_the_rest():
    assert as_string_metadata({"a": None, "n": 348}) == {"n": "348"}


def test_retain_content_drops_null_metadata_values():
    content = RetainContent(content="hi", metadata={"ocr_engine": None, "source": "slack"})
    assert content.metadata == {"source": "slack"}


def test_retain_content_keeps_non_null_values_as_given():
    content = RetainContent(content="hi", metadata={"n": 5, "source": "slack"})
    assert content.metadata == {"n": 5, "source": "slack"}


def test_retain_content_normalizes_null_metadata_to_empty_dict():
    """``"metadata": null`` in the request reaches the dataclass as None; the
    field is declared dict[str, str], so it must not stay None."""
    assert RetainContent(content="hi", metadata=None).metadata == {}


def test_build_contents_normalizes_null_metadata_from_api():
    """The ingestion path accepts JSON null metadata; stored facts must not
    carry null values (regression for the reported retain-with-null case)."""
    contents = _build_contents(
        [{"content": "hi", "metadata": {"ocr_engine": None, "n": 5, "source": "slack"}}],
        None,
    )
    assert contents[0].metadata == {"n": 5, "source": "slack"}
