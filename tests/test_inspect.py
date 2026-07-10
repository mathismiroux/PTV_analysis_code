from __future__ import annotations

import h5py
import numpy as np
import pytest

from ptv_flow.inspect import (
    CellInspection,
    _apply_inspector_valid_mask,
    _rejected_overlay_rgba,
    component_mean_ignoring_zero,
    format_cell_inspection,
    inspect_cell,
    nearest_index,
    validate_average_compatible,
)
from ptv_flow.postprocess import TemporalAverageVolume, temporal_average_volume
from ptv_flow.reader import FlowDataset


def test_component_mean_ignores_exact_zeros():
    mean = component_mean_ignoring_zero(np.array([0.0, 2.0, 4.0, 0.0]))
    assert mean.mean == 3.0
    assert mean.count == 2
    assert mean.accepted

    empty = component_mean_ignoring_zero(np.array([0.0, 0.0]))
    assert np.isnan(empty.mean)
    assert empty.count == 0
    assert not empty.accepted

    rejected = component_mean_ignoring_zero(
        np.array([0.0, 2.0, 4.0, 0.0]), min_valid_count=3
    )
    assert np.isnan(rejected.mean)
    assert rejected.count == 2
    assert not rejected.accepted


def test_nearest_index():
    assert nearest_index(np.array([10.0, 20.0, 30.0]), 26.0) == 2
    assert nearest_index(np.array([10.0, 20.0, 30.0]), 14.0) == 0


def test_apply_inspector_valid_mask_marks_rejected_cells_without_changing_speed():
    speed = np.array([[1.0, 2.0]])
    u = np.array([[3.0, 4.0]])
    v = np.array([[5.0, 6.0]])
    counts = (
        np.array([[10, 8]]),
        np.array([[10, 10]]),
        np.array([[10, 10]]),
    )

    display_speed, display_u, display_v, accepted = _apply_inspector_valid_mask(
        speed,
        u,
        v,
        counts,
        min_valid_count=9,
    )

    np.testing.assert_array_equal(accepted, [[True, False]])
    np.testing.assert_array_equal(display_speed, [[1.0, 2.0]])
    np.testing.assert_array_equal(display_u, [[3.0, 0.0]])
    np.testing.assert_array_equal(display_v, [[5.0, 0.0]])

    overlay = _rejected_overlay_rgba(accepted)
    assert overlay.shape == (1, 2, 4)
    assert overlay[0, 0, 3] == 0.0
    assert overlay[0, 1, 3] > 0.0


def test_inspect_cell_matches_raw_and_average(tiny_flow_path, tmp_path):
    average_path = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, average_path, chunk_size=2)

    with (
        FlowDataset(tiny_flow_path) as flow,
        TemporalAverageVolume(average_path) as average,
        h5py.File(tiny_flow_path, "r") as raw,
    ):
        valid_counts = (raw["u"][:] != 0.0).sum(axis=0)
        z_index, y_index, x_index = np.argwhere(valid_counts > 0)[0]
        cell = inspect_cell(
            flow,
            time_index=1,
            z_index=int(z_index),
            y_index=int(y_index),
            x_index=int(x_index),
            average=average,
        )

        assert cell.raw_u == pytest.approx(raw["u"][1, z_index, y_index, x_index])
        assert cell.raw_v == pytest.approx(raw["v"][1, z_index, y_index, x_index])
        assert cell.raw_w == pytest.approx(raw["w"][1, z_index, y_index, x_index])

        u_series = raw["u"][:, z_index, y_index, x_index]
        u_valid = u_series != 0.0
        assert cell.computed_u.mean == pytest.approx(u_series[u_valid].mean())
        assert cell.computed_u.count == int(u_valid.sum())
        assert cell.average_u is not None
        assert cell.average_u.mean == pytest.approx(cell.computed_u.mean)
        assert cell.average_u.count == cell.computed_u.count
        assert cell.average_u.accepted

        report = format_cell_inspection(
            cell,
            raw_label="tiny_flow.nc",
            min_valid_count=1,
            n_times=flow.n_times,
        )
        assert "Raw value at selected frame" in report
        assert "raw source: tiny_flow.nc" in report
        assert "min valid count: 1 / 4" in report
        assert "Selected-voxel temporal mean computed on demand" in report
        assert "source: raw time series at this one cell only" in report
        assert "accepted" in report
        assert "Value stored in average file" in report


def test_format_cell_inspection_marks_raw_only_mode():
    cell = CellInspection(
        time_index=0,
        z_index=0,
        y_index=0,
        x_index=0,
        time=0.0,
        z=0.0,
        y=0.0,
        x=0.0,
        raw_u=1.0,
        raw_v=2.0,
        raw_w=3.0,
        raw_speed=float(np.sqrt(14.0)),
        computed_u=component_mean_ignoring_zero(np.array([1.0])),
        computed_v=component_mean_ignoring_zero(np.array([2.0])),
        computed_w=component_mean_ignoring_zero(np.array([3.0])),
    )

    report = format_cell_inspection(cell, raw_label="raw.nc")
    assert "raw source: raw.nc" in report
    assert "Average file comparison" in report
    assert "not active" in report
    assert "Value stored in average file" not in report


def test_validate_average_compatible_rejects_shape_mismatch(tiny_flow_path, tmp_path):
    mismatch = tmp_path / "mismatch.nc"
    with h5py.File(mismatch, "w") as h5:
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("u_mean", data=np.zeros((1, 1, 1)))
        h5.create_dataset("v_mean", data=np.zeros((1, 1, 1)))
        h5.create_dataset("w_mean", data=np.zeros((1, 1, 1)))

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(mismatch) as average:
        with pytest.raises(ValueError, match="grid shape"):
            validate_average_compatible(flow, average)
