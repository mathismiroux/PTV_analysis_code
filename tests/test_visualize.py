from __future__ import annotations

import numpy as np
import pytest

from ptv_flow.visualize import _temporal_average_quantity


def test_temporal_average_quantity_selection():
    plane = {
        "u": np.array([[1.0]]),
        "v": np.array([[2.0]]),
        "w": np.array([[3.0]]),
        "speed": np.array([[4.0]]),
    }

    scalar, label, cmap = _temporal_average_quantity(plane, "speed")
    np.testing.assert_array_equal(scalar, [[4.0]])
    assert label == "Mean 3D velocity magnitude"
    assert cmap == "viridis"

    for quantity in ("u", "v", "w"):
        scalar, label, cmap = _temporal_average_quantity(plane, quantity)
        np.testing.assert_array_equal(scalar, plane[quantity])
        assert label == f"Mean {quantity} velocity"
        assert cmap == "coolwarm"


def test_temporal_average_quantity_rejects_unknown():
    with pytest.raises(ValueError, match="quantity"):
        _temporal_average_quantity({"speed": np.array([[1.0]])}, "bad")
