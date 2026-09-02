from __future__ import annotations

import numpy as np
import pytest

from ptv_flow.validity import (
    valid_component_samples,
    valid_vector_samples,
    validate_invalid_samples,
)


def test_valid_component_samples_modes():
    data = np.array([0.0, 1.0, np.nan, np.inf])

    np.testing.assert_array_equal(
        valid_component_samples(data), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        valid_component_samples(data, "zero"), [False, True, True, True]
    )
    np.testing.assert_array_equal(
        valid_component_samples(data, "nan"), [True, True, False, False]
    )
    np.testing.assert_array_equal(
        valid_component_samples(data, "zero-or-nan"), [False, True, False, False]
    )
    np.testing.assert_array_equal(
        valid_component_samples(data, "none"), [True, True, True, True]
    )


def test_valid_vector_samples_keeps_any_valid_component():
    components = {
        "u": np.array([0.0, 1.0, 0.0, np.nan, np.nan]),
        "v": np.array([0.0, 0.0, 2.0, np.nan, 2.0]),
        "w": np.array([0.0, 0.0, 0.0, np.nan, 3.0]),
    }

    np.testing.assert_array_equal(
        valid_vector_samples(components),
        [True, True, True, False, False],
    )
    np.testing.assert_array_equal(
        valid_vector_samples(components, "zero"),
        [False, True, True, True, True],
    )
    np.testing.assert_array_equal(
        valid_vector_samples(components, "nan"),
        [True, True, True, False, False],
    )
    np.testing.assert_array_equal(
        valid_vector_samples(components, "zero-or-nan"),
        [False, True, True, False, False],
    )


def test_validate_invalid_samples_rejects_unknown_mode():
    with pytest.raises(ValueError, match="invalid_samples"):
        validate_invalid_samples("missing-value")
