from __future__ import annotations

import numpy as np
import pytest

from ptv_flow.visualize import (
    _apply_phase_plane_valid_fraction,
    _apply_plane_valid_fraction,
    _phase_average_quantity,
    _temporal_average_quantity,
)


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


def test_phase_average_quantity_selection():
    plane = {
        "field": "coherent",
        "u": np.array([[1.0]]),
        "v": np.array([[2.0]]),
        "w": np.array([[3.0]]),
        "speed": np.array([[4.0]]),
    }

    scalar, label, cmap = _phase_average_quantity(plane, "speed")
    np.testing.assert_array_equal(scalar, [[4.0]])
    assert label == "Coherent 3D velocity magnitude"
    assert cmap == "viridis"

    scalar, label, cmap = _phase_average_quantity(plane, "u")
    np.testing.assert_array_equal(scalar, [[1.0]])
    assert label == "Coherent u velocity"
    assert cmap == "coolwarm"


def test_apply_plane_valid_fraction_masks_component():
    plane = {
        "u_count": np.array([[4, 2]]),
        "v_count": np.array([[4, 4]]),
        "w_count": np.array([[4, 4]]),
    }
    scalar = np.array([[1.0, 2.0]])

    masked = _apply_plane_valid_fraction(
        plane,
        scalar,
        quantity="u",
        min_valid_count=3,
    )

    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1])
    np.testing.assert_array_equal(scalar, [[1.0, 2.0]])


def test_apply_plane_valid_fraction_masks_speed_if_any_component_sparse():
    plane = {
        "u_count": np.array([[4, 4]]),
        "v_count": np.array([[4, 2]]),
        "w_count": np.array([[4, 4]]),
    }
    scalar = np.array([[1.0, 2.0]])

    masked = _apply_plane_valid_fraction(
        plane,
        scalar,
        quantity="speed",
        min_valid_count=3,
    )

    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1])


def test_apply_phase_plane_valid_fraction_masks_component():
    plane = {
        "u_count": np.array([[4, 2]]),
        "v_count": np.array([[4, 4]]),
        "w_count": np.array([[4, 4]]),
    }
    scalar = np.array([[1.0, 2.0]])

    masked = _apply_phase_plane_valid_fraction(
        plane,
        scalar,
        quantity="u",
        min_valid_count=3,
    )

    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1])
