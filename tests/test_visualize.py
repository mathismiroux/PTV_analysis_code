from __future__ import annotations

import numpy as np
import pytest

from ptv_flow.visualize import (
    _apply_harmonic_valid_fraction,
    _apply_phase_plane_valid_fraction,
    _apply_plane_valid_fraction,
    _phase_average_quantity,
    _temporal_average_quantity,
)


def _write_minimal_phase_average(path):
    import h5py

    with h5py.File(path, "w") as h5:
        shape = (3, 1, 1, 1)
        h5.create_dataset("x", data=np.array([1.0]))
        h5.create_dataset("y", data=np.array([2.0]))
        h5.create_dataset("z", data=np.array([3.0]))
        h5.create_dataset("phase", data=np.array([0.0, 2.0, 4.0]))
        h5.create_dataset("phase_degrees", data=np.array([0.0, 120.0, 240.0]))
        h5.create_dataset("phase_sample_count", data=np.array([4, 4, 4], dtype=np.uint32))
        for component, values in {
            "u": [1.0, 2.0, 1.0],
            "v": [0.0, 1.0, 0.0],
            "w": [-1.0, 0.0, -1.0],
        }.items():
            h5.create_dataset(
                f"{component}_phase_mean",
                data=np.asarray(values, dtype=np.float64).reshape(shape),
            )
            h5.create_dataset(
                f"{component}_coherent",
                data=np.asarray(values, dtype=np.float64).reshape(shape) - 1.0,
            )
            h5.create_dataset(
                f"{component}_phase_count",
                data=np.array([4, 2, 4], dtype=np.uint32).reshape(shape),
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


def test_apply_harmonic_valid_fraction_requires_all_phase_bins(tmp_path):
    from ptv_flow.postprocess import PhaseAverageVolume

    path = tmp_path / "phase_average.nc"
    import h5py

    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([0.0, 1.0]))
        h5.create_dataset("y", data=np.array([0.0, 1.0]))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("phase", data=np.array([0.0, np.pi]))
        h5.create_dataset("phase_sample_count", data=np.array([4, 4], dtype=np.uint32))
        h5.create_dataset(
            "u_phase_count",
            data=np.array(
                [
                    [[[3, 3], [3, 3]]],
                    [[[3, 1], [3, 3]]],
                ],
                dtype=np.uint32,
            ),
        )
        h5.create_dataset("v_phase_count", data=np.ones((2, 1, 2, 2), dtype=np.uint32))
        h5.create_dataset("w_phase_count", data=np.ones((2, 1, 2, 2), dtype=np.uint32))

    scalar = np.array([[1.0, 2.0], [3.0, 4.0]])
    with PhaseAverageVolume(path) as volume:
        masked, accepted = _apply_harmonic_valid_fraction(
            volume,
            scalar,
            plane_axis="z",
            plane_index=0,
            component="u",
            min_valid_fraction=0.5,
        )

    assert accepted == 3
    assert masked[0, 0] == 1.0
    assert np.isnan(masked[0, 1])


def test_show_phase_voxel_series_saves_file(tmp_path):
    from ptv_flow.postprocess import PhaseAverageVolume
    from ptv_flow.visualize import show_phase_voxel_series

    path = tmp_path / "phase_average.nc"
    output = tmp_path / "phase_voxel.png"
    _write_minimal_phase_average(path)

    with PhaseAverageVolume(path) as volume:
        show_phase_voxel_series(
            volume,
            x_value=1.0,
            y_value=2.0,
            z_value=3.0,
            min_valid_fraction=0.5,
            save=output,
        )

    assert output.exists()
