from __future__ import annotations

import h5py
import numpy as np
import pytest

from ptv_flow.reader import FlowDataset


def test_reader_reports_fixture_shape(tiny_flow_path):
    with FlowDataset(tiny_flow_path) as flow:
        assert flow.shape == (4, 3, 5, 6)
        assert flow.n_times == 4
        assert flow.grid_shape == (3, 5, 6)
        assert flow.coordinate("x").shape == (6,)
        assert flow.coordinate("y").shape == (5,)
        assert flow.coordinate("z").shape == (3,)


def test_voxel_size_matches_coordinate_differences(tiny_flow_path):
    with FlowDataset(tiny_flow_path) as flow:
        spacing = flow.voxel_size()
        for name in ("x", "y", "z"):
            expected = np.diff(flow.coordinate(name))
            assert spacing[name]["median"] == pytest.approx(np.median(expected))
            assert spacing[name]["min"] == pytest.approx(expected.min())
            assert spacing[name]["max"] == pytest.approx(expected.max())


def test_read_z_plane_matches_source_slice(tiny_flow_path):
    with FlowDataset(tiny_flow_path) as flow, h5py.File(tiny_flow_path, "r") as src:
        z_index = flow.nearest_z_index(0.0)
        plane = flow.read_z_plane(time_index=2, z_index=z_index)

        assert plane.z_index == z_index
        np.testing.assert_allclose(plane.u, src["u"][2, z_index, :, :])
        np.testing.assert_allclose(plane.v, src["v"][2, z_index, :, :])
        np.testing.assert_allclose(plane.w, src["w"][2, z_index, :, :])
        np.testing.assert_allclose(plane.speed, np.sqrt(plane.u**2 + plane.v**2 + plane.w**2))


def test_reader_rejects_missing_variable(tmp_path):
    broken = tmp_path / "broken.nc"
    with h5py.File(broken, "w") as h5:
        h5.create_dataset("t", data=np.arange(2))
        h5.create_dataset("x", data=np.arange(2))
        h5.create_dataset("y", data=np.arange(2))
        h5.create_dataset("z", data=np.arange(2))
        h5.create_dataset("u", data=np.ones((2, 2, 2, 2)))
        h5.create_dataset("v", data=np.ones((2, 2, 2, 2)))

    with pytest.raises(KeyError, match="w"):
        FlowDataset(broken)
