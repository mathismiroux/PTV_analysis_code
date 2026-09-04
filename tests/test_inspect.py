from __future__ import annotations

import h5py
import numpy as np
import pytest

from ptv_flow.inspect import (
    CellInspection,
    _apply_inspector_valid_mask,
    _coverage_from_counts,
    _filled_image_alpha,
    _rejected_overlay_rgba,
    component_mean_ignoring_zero,
    format_cell_inspection,
    inspect_cell,
    inspect_phase_average_counts,
    nearest_index,
    phase_coverage_for_cell,
    phase_values_for_flow,
    step_index,
    validate_average_compatible,
    validate_interpolated_compatible,
)
from ptv_flow.postprocess import (
    PhaseAverageVolume,
    TemporalAverageVolume,
    phase_average_volume,
    spatio_temporal_interpolate_velocity,
    temporal_average_volume,
)
from ptv_flow.reader import FlowDataset


def test_component_mean_ignores_exact_zeros():
    mean = component_mean_ignoring_zero(
        np.array([0.0, 2.0, 4.0, 0.0]),
        invalid_samples="zero",
    )
    assert mean.mean == 3.0
    assert mean.count == 2
    assert mean.accepted

    empty = component_mean_ignoring_zero(
        np.array([0.0, 0.0]),
        invalid_samples="zero",
    )
    assert np.isnan(empty.mean)
    assert empty.count == 0
    assert not empty.accepted

    rejected = component_mean_ignoring_zero(
        np.array([0.0, 2.0, 4.0, 0.0]),
        min_valid_count=3,
        invalid_samples="zero",
    )
    assert np.isnan(rejected.mean)
    assert rejected.count == 2
    assert not rejected.accepted


def test_component_mean_can_ignore_zero_and_nan():
    mean = component_mean_ignoring_zero(
        np.array([0.0, 2.0, np.nan, 4.0]),
        invalid_samples="zero-or-nan",
    )

    assert mean.mean == 3.0
    assert mean.count == 2
    assert mean.accepted


def test_nearest_index():
    assert nearest_index(np.array([10.0, 20.0, 30.0]), 26.0) == 2
    assert nearest_index(np.array([10.0, 20.0, 30.0]), 14.0) == 0


def test_step_index_clamps_to_available_frames():
    assert step_index(0, -1, 9) == 0
    assert step_index(4, -1, 9) == 3
    assert step_index(4, 1, 9) == 5
    assert step_index(9, 1, 9) == 9


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

    filled_alpha = _filled_image_alpha(np.array([[True, False]]))
    assert filled_alpha.shape == (1, 2)
    assert filled_alpha[0, 0] < 1.0
    assert filled_alpha[0, 1] == 1.0


def test_coverage_from_counts_reports_undercovered_volume():
    counts = (
        np.array([[[3, 2], [3, 3]]]),
        np.array([[[3, 3], [1, 3]]]),
        np.array([[[3, 3], [3, 0]]]),
    )

    coverage = _coverage_from_counts(counts, min_valid_count=3)

    assert coverage.total == 4
    assert coverage.accepted == 1
    assert coverage.rejected == 3
    assert coverage.rejected_fraction == pytest.approx(0.75)


def test_phase_coverage_for_cell_reports_bin_counts(tmp_path):
    raw = tmp_path / "phase_coverage.nc"
    phases = np.array([0.1, 0.2, np.pi + 0.1, np.pi + 0.2])
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(4, dtype=np.float64))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[2.0]]], [[[np.nan]]], [[[4.0]]]]))
        h5.create_dataset("v", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]]]))
        h5.create_dataset("w", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[np.nan]]]]))

    with FlowDataset(raw) as flow:
        coverage = phase_coverage_for_cell(
            flow,
            z_index=0,
            y_index=0,
            x_index=0,
            phases=phases,
            n_phase_bins=2,
            min_valid_fraction=0.8,
        )
        cell = inspect_cell(
            flow,
            time_index=0,
            z_index=0,
            y_index=0,
            x_index=0,
            phases=phases,
            n_phase_bins=2,
            min_valid_fraction=0.8,
        )

    np.testing.assert_array_equal(coverage.sample_counts, [2, 2])
    np.testing.assert_array_equal(coverage.min_valid_counts, [2, 2])
    np.testing.assert_array_equal(coverage.u_counts, [2, 1])
    np.testing.assert_array_equal(coverage.v_counts, [2, 2])
    np.testing.assert_array_equal(coverage.w_counts, [2, 1])
    np.testing.assert_allclose(coverage.u_means, [1.5, 4.0])
    assert cell.phase_coverage is not None
    report = format_cell_inspection(cell)
    assert "Selected-voxel phase coverage" in report
    assert "selected indices: z=0, y=0, x=0" in report
    assert "phase  n  min  u  v  w  ok      u_bar" in report
    assert "1.5" in report
    assert "no" in report


def test_phase_values_for_flow_from_frequency(tiny_flow_path):
    with FlowDataset(tiny_flow_path) as flow:
        phases = phase_values_for_flow(flow, frequency_hz=1.0, phase_offset=0.25)

        expected = (2.0 * np.pi * flow.coordinate("t") + 0.25) % (2.0 * np.pi)
        np.testing.assert_allclose(phases, expected)


def test_inspect_phase_average_counts_reads_final_product(tmp_path):
    raw = tmp_path / "phase_final_source.nc"
    phases = np.array([0.1, 0.2, np.pi + 0.1, np.pi + 0.2])
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(4, dtype=np.float64))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[2.0]]], [[[np.nan]]], [[[4.0]]]]))
        h5.create_dataset("v", data=np.ones((4, 1, 1, 1)))
        h5.create_dataset("w", data=np.ones((4, 1, 1, 1)))

    phase_average = tmp_path / "phase_average.nc"
    with FlowDataset(raw) as flow:
        phase_average_volume(
            flow,
            phase_average,
            n_phase_bins=2,
            phase_signal=phases,
            chunk_size=2,
        )

    with PhaseAverageVolume(phase_average) as volume:
        report = inspect_phase_average_counts(
            volume,
            x_value=0.0,
            y_value=0.0,
            z_value=0.0,
            min_valid_fraction=0.8,
        )

    assert "Phase-average file inspection" in report
    assert "Selected-voxel phase coverage" in report
    assert "selected indices: z=0, y=0, x=0" in report
    assert "phase  n  min  u  v  w  ok      u_bar" in report
    assert "no" in report


def test_inspect_cell_matches_raw_and_average(tiny_flow_path, tmp_path):
    average_path = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, average_path, chunk_size=2)

    with (
        FlowDataset(tiny_flow_path) as flow,
        TemporalAverageVolume(average_path) as average,
        h5py.File(tiny_flow_path, "r") as raw,
    ):
        valid_counts = np.isfinite(raw["u"][:]).sum(axis=0)
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
        u_valid = np.isfinite(u_series)
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
            volume_coverage=_coverage_from_counts(
                (
                    average._file["u_count"][:],
                    average._file["v_count"][:],
                    average._file["w_count"][:],
                ),
                min_valid_count=1,
            ),
            plane_coverage=_coverage_from_counts(
                (
                    average._file["u_count"][z_index, :, :],
                    average._file["v_count"][z_index, :, :],
                    average._file["w_count"][z_index, :, :],
                ),
                min_valid_count=1,
            ),
        )
        assert "Raw value at selected frame" in report
        assert "raw source: tiny_flow.nc" in report
        assert "min valid count: 1 / 4" in report
        assert "Selected-voxel temporal mean computed on demand" in report
        assert "source: raw time series at this one cell only" in report
        assert "accepted" in report
        assert "Value stored in average file" in report
        assert "Average-file coverage" in report
        assert "empty volume:" in report
        assert "empty shown z plane:" in report


def test_inspect_cell_reports_interpolated_values(tmp_path):
    raw = tmp_path / "holes.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]]]))
        h5.create_dataset("v", data=np.array([[[[2.0]]], [[[0.0]]], [[[4.0]]]]))
        h5.create_dataset("w", data=np.array([[[[3.0]]], [[[0.0]]], [[[5.0]]]]))

    interpolated_path = tmp_path / "interpolated.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            interpolated_path,
            axes=("t",),
            zero_mask="vector",
            invalid_samples="zero",
        )

    with FlowDataset(raw) as flow, FlowDataset(interpolated_path) as interpolated:
        validate_interpolated_compatible(flow, interpolated)
        cell = inspect_cell(
            flow,
            time_index=1,
            z_index=0,
            y_index=0,
            x_index=0,
            interpolated=interpolated,
        )

    assert cell.raw_u == 0.0
    assert cell.interpolated_u == 2.0
    assert cell.interpolated_v == 3.0
    assert cell.interpolated_w == 4.0
    assert cell.filled_u
    assert cell.filled_v
    assert cell.filled_w
    report = format_cell_inspection(cell)
    assert "Value stored in interpolated file" in report
    assert "delta=2" in report


def test_inspect_cell_reads_shared_interpolated_filled_mask(tmp_path):
    raw = tmp_path / "holes.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]]]))
        h5.create_dataset("v", data=np.array([[[[2.0]]], [[[0.0]]], [[[4.0]]]]))
        h5.create_dataset("w", data=np.array([[[[3.0]]], [[[0.0]]], [[[5.0]]]]))

    interpolated_path = tmp_path / "interpolated.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            interpolated_path,
            axes=("t",),
            zero_mask="vector",
            invalid_samples="zero",
            store_component_filled_masks=False,
        )

    with (
        FlowDataset(raw) as flow,
        FlowDataset(interpolated_path) as interpolated,
        h5py.File(interpolated_path, "r") as h5,
    ):
        assert "filled_mask" in h5
        assert "u_filled_mask" not in h5
        validate_interpolated_compatible(flow, interpolated)
        cell = inspect_cell(
            flow,
            time_index=1,
            z_index=0,
            y_index=0,
            x_index=0,
            interpolated=interpolated,
        )

    assert cell.filled_u
    assert cell.filled_v
    assert cell.filled_w


def test_format_interpolated_delta_marks_missing_raw_value():
    cell = CellInspection(
        time_index=0,
        z_index=0,
        y_index=0,
        x_index=0,
        time=0.0,
        z=0.0,
        y=0.0,
        x=0.0,
        raw_u=float("nan"),
        raw_v=float("nan"),
        raw_w=float("nan"),
        raw_speed=float("nan"),
        computed_u=component_mean_ignoring_zero(
            np.array([float("nan")]),
            invalid_samples="zero-or-nan",
        ),
        computed_v=component_mean_ignoring_zero(
            np.array([float("nan")]),
            invalid_samples="zero-or-nan",
        ),
        computed_w=component_mean_ignoring_zero(
            np.array([float("nan")]),
            invalid_samples="zero-or-nan",
        ),
        interpolated_u=1.5,
        interpolated_v=-0.1,
        interpolated_w=-0.7,
        interpolated_speed=1.7,
        filled_u=True,
        filled_v=True,
        filled_w=True,
    )

    report = format_cell_inspection(cell)
    assert "delta=n/a, raw missing" in report
    assert "u_mean=nan  count=0  rejected" in report


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
