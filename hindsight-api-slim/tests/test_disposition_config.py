"""Tests for write-side validation of bank disposition config overrides.

Disposition traits (skepticism / literalism / empathy) are integers on a 1-5
scale. The ``PATCH /v1/{tenant}/banks/{id}/config`` write path must reject
out-of-contract values (floats, 0-1 scales, ints outside 1-5) at write time;
otherwise a single malformed bank 500s the entire bank list because the read
overlay injects the stored value verbatim into a strict
``DispositionTraits(int, ge=1, le=5)``. See issue #2348.
"""

import pytest

from hindsight_api.config_resolver import _validate_disposition_updates


_DISPOSITION_FIELD_NAMES = (
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
)


class TestValidateDispositionUpdates:
    def test_no_op_passes(self):
        _validate_disposition_updates({})
        _validate_disposition_updates({"unrelated_field": 123})

    def test_valid_in_range_integers_pass(self):
        for key in _DISPOSITION_FIELD_NAMES:
            for value in (1, 2, 3, 4, 5):
                _validate_disposition_updates({key: value})

    def test_none_clears_override(self):
        # None is the "unset this per-bank override" sentinel (field is int | None).
        for key in _DISPOSITION_FIELD_NAMES:
            _validate_disposition_updates({key: None})

    def test_out_of_range_integer_raises(self):
        for key in _DISPOSITION_FIELD_NAMES:
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: 0})
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: 6})
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: -1})

    def test_float_raises(self):
        # The reported v0.8.3 case: a 0-1 scale used by mistake.
        for key in _DISPOSITION_FIELD_NAMES:
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: 0.7})
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: 3.0})  # float, even if in 1-5 range

    def test_bool_raises(self):
        # bool is an int subclass and would sneak past a naive isinstance(int) check.
        for key in _DISPOSITION_FIELD_NAMES:
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: True})

    def test_string_raises(self):
        for key in _DISPOSITION_FIELD_NAMES:
            with pytest.raises(ValueError, match=key):
                _validate_disposition_updates({key: "3"})
