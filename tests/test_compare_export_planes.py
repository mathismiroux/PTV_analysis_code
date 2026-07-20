from __future__ import annotations

import h5py
import numpy as np

from scripts.compare_export_planes import (
    compare_exports,
    format_run_configuration,
    interpolate_to_grid,
    plane_spec,
    temporal_plane_stats,
)
from ptv_flow.reader import FlowDataset


def test_temporal_plane_stats_matches_source_component_slice(tiny_flow_path):
    with FlowDataset(tiny_flow_path) as flow, h5py.File(tiny_flow_path, "r") as src:
        spec = plane_spec(flow, "y", float(flow.coordinate("y")[2]))
        stats = temporal_plane_stats(flow, "u", spec)
        data = src["u"][:, :, spec.index, :].astype(np.float64)
        valid = data != 0.0
        count = valid.sum(axis=0)
        expected_mean = np.full(data.shape[1:], np.nan)
        np.divide(
            np.where(valid, data, 0.0).sum(axis=0),
            count,
            out=expected_mean,
            where=count > 0,
        )
        expected_std = np.full(data.shape[1:], np.nan)
        fluctuation = data - expected_mean[None, :, :]
        np.divide(
            np.where(valid, fluctuation * fluctuation, 0.0).sum(axis=0),
            count,
            out=expected_std,
            where=count > 0,
        )

    np.testing.assert_allclose(stats.mean, expected_mean, equal_nan=True)
    np.testing.assert_allclose(stats.std, np.sqrt(expected_std), equal_nan=True)
    np.testing.assert_array_equal(stats.count, count)


def test_compare_exports_writes_figure(tiny_flow_path, tmp_path):
    output = tmp_path / "comparison.png"

    compare_exports(
        paths=[tiny_flow_path, tiny_flow_path],
        labels=["a", "b"],
        quantity="speed",
        output=output,
        x_planes=None,
        y_plane=None,
    )

    assert output.exists()


def test_interpolate_to_grid_projects_shifted_regular_grid():
    source_horizontal = np.array([0.0, 1.0, 2.0])
    source_vertical = np.array([10.0, 11.0, 12.0])
    h_grid, v_grid = np.meshgrid(source_horizontal, source_vertical)
    source_field = h_grid + 2.0 * v_grid

    target_horizontal = np.array([0.5, 1.5])
    target_vertical = np.array([10.5, 11.5])
    interpolated = interpolate_to_grid(
        source_horizontal,
        source_vertical,
        source_field,
        target_horizontal,
        target_vertical,
    )

    expected_h, expected_v = np.meshgrid(target_horizontal, target_vertical)
    np.testing.assert_allclose(interpolated, expected_h + 2.0 * expected_v)


def test_compare_exports_difference_writes_figure(tiny_flow_path, tmp_path):
    output = tmp_path / "difference.png"

    compare_exports(
        paths=[tiny_flow_path, tiny_flow_path],
        labels=["reference", "comparison"],
        quantity="u",
        output=output,
        x_planes=None,
        y_plane=None,
        difference=True,
        reference_grid="first",
    )

    assert output.exists()


def test_format_run_configuration_reports_defaults(tiny_flow_path, tmp_path):
    report = format_run_configuration(
        paths=[tiny_flow_path, tiny_flow_path],
        labels=["a", "b"],
        quantity="speed",
        output=tmp_path / "comparison.png",
        x_planes=None,
        y_plane=None,
        invalid_samples="zero",
        difference=False,
        reference_grid="first",
        provided={
            "labels": True,
            "quantity": False,
            "invalid_samples": False,
            "output": True,
            "difference": False,
            "reference_grid": False,
            "x_planes": False,
            "y_plane": False,
        },
    )

    assert "quantity (default)" in report
    assert "invalid_samples (default): zero" in report
    assert "x_planes (default from common overlap)" in report
    assert "y_plane (default from common overlap)" in report
    assert "resolved nearest planes" in report
    assert "index=" in report
