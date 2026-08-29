import numpy as np
import pytest

from preprocess_task1 import normalize_bx


def test_normalize_bx_positive_to_one():
    assert normalize_bx("Positive") == 1


def test_normalize_bx_negative_to_zero():
    assert normalize_bx("Negative") == 0


def test_normalize_bx_string_none_to_nan():
    assert np.isnan(normalize_bx("None"))


def test_normalize_bx_json_null_to_nan():
    assert np.isnan(normalize_bx(None))


def test_normalize_bx_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported BX"):
        normalize_bx("unknown-status")
