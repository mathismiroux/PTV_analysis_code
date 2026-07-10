from __future__ import annotations

import h5py
import numpy as np

from ptv_flow.postprocess import (
    TemporalAverageVolume,
    apply_valid_fraction_to_average,
    temporal_average_volume,
)
from ptv_flow.reader import FlowDataset


def _expected_component_mean(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = data != 0.0
    counts = valid.sum(axis=0, dtype=np.uint32)
    means = np.full(data.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64),
        counts,
        out=means,
        where=counts > 0,
    )
    return means, counts


def test_temporal_average_component_mask_matches_fixture(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, output=output, chunk_size=2)

    with h5py.File(tiny_flow_path, "r") as src, h5py.File(output, "r") as out:
        assert out.attrs["zero_mask"] == "component"
        assert set(out.keys()) == {
            "provenance",
            "speed_from_mean",
            "u_count",
            "u_mean",
            "v_count",
            "v_mean",
            "w_count",
            "w_mean",
            "x",
            "y",
            "z",
        }
        assert out.attrs["source_file"].endswith("tiny_flow.nc")
        assert out.attrs["source_file_name"] == "tiny_flow.nc"
        assert "created_utc" in out.attrs
        assert out["provenance"].attrs["source_file"].endswith("tiny_flow.nc")

        for name in ("u", "v", "w"):
            expected_mean, expected_count = _expected_component_mean(src[name][:])
            np.testing.assert_allclose(
                out[f"{name}_mean"][:], expected_mean, equal_nan=True
            )
            np.testing.assert_array_equal(out[f"{name}_count"][:], expected_count)

        expected_speed = np.sqrt(
            out["u_mean"][:] ** 2 + out["v_mean"][:] ** 2 + out["w_mean"][:] ** 2
        )
        np.testing.assert_allclose(
            out["speed_from_mean"][:], expected_speed, equal_nan=True
        )


def test_temporal_average_refuses_to_overwrite_existing_output(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    output.write_text("existing")

    with FlowDataset(tiny_flow_path) as flow:
        try:
            temporal_average_volume(flow, output=output, chunk_size=2)
        except FileExistsError as exc:
            assert "Refusing to overwrite" in str(exc)
        else:
            raise AssertionError("Expected FileExistsError")

    assert output.read_text() == "existing"


def test_temporal_average_overwrite_replaces_existing_output(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    output.write_text("existing")

    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(
            flow,
            output=output,
            chunk_size=2,
            overwrite=True,
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["source_file_name"] == "tiny_flow.nc"
        assert "u_mean" in out


def test_temporal_average_vector_mask_differs_from_component_mask(tmp_path):
    raw = tmp_path / "known_zeros.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("u", data=np.array([[[[0.0]]], [[[2.0]]], [[[4.0]]]]))
        h5.create_dataset("v", data=np.array([[[[1.0]]], [[[0.0]]], [[[5.0]]]]))
        h5.create_dataset("w", data=np.array([[[[1.0]]], [[[3.0]]], [[[0.0]]]]))

    component_output = tmp_path / "component.nc"
    vector_output = tmp_path / "vector.nc"
    with FlowDataset(raw) as flow:
        temporal_average_volume(flow, component_output, chunk_size=1, zero_mask="component")
        temporal_average_volume(flow, vector_output, chunk_size=1, zero_mask="vector")

    with h5py.File(component_output, "r") as component, h5py.File(vector_output, "r") as vector:
        assert component["u_mean"][0, 0, 0] == 3.0
        assert component["v_mean"][0, 0, 0] == 3.0
        assert component["w_mean"][0, 0, 0] == 2.0
        np.testing.assert_array_equal(component["u_count"][:], [[[2]]])
        np.testing.assert_array_equal(component["v_count"][:], [[[2]]])
        np.testing.assert_array_equal(component["w_count"][:], [[[2]]])

        assert vector["u_mean"][0, 0, 0] == 2.0
        assert vector["v_mean"][0, 0, 0] == 2.0
        assert vector["w_mean"][0, 0, 0] == 4.0 / 3.0
        np.testing.assert_array_equal(vector["u_count"][:], [[[3]]])
        np.testing.assert_array_equal(vector["v_count"][:], [[[3]]])
        np.testing.assert_array_equal(vector["w_count"][:], [[[3]]])


def test_temporal_average_min_valid_fraction_discards_sparse_values(tmp_path):
    raw = tmp_path / "sparse.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset(
            "u", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[0.0]]], [[[0.0]]]])
        )
        h5.create_dataset(
            "v", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[0.0]]]])
        )
        h5.create_dataset(
            "w", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]])
        )

    output = tmp_path / "mean.nc"
    with FlowDataset(raw) as flow:
        temporal_average_volume(
            flow,
            output,
            chunk_size=2,
            min_valid_fraction=0.8,
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["min_valid_fraction"] == 0.8
        assert out.attrs["min_valid_count"] == 4
        assert np.isnan(out["u_mean"][0, 0, 0])
        assert out["v_mean"][0, 0, 0] == 2.5
        assert out["w_mean"][0, 0, 0] == 3.0
        np.testing.assert_array_equal(out["u_count"][:], [[[3]]])
        np.testing.assert_array_equal(out["v_count"][:], [[[4]]])
        np.testing.assert_array_equal(out["w_count"][:], [[[5]]])
        assert np.isnan(out["speed_from_mean"][0, 0, 0])


def test_apply_valid_fraction_to_existing_average(tmp_path):
    raw = tmp_path / "sparse.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset(
            "u", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[0.0]]], [[[0.0]]]])
        )
        h5.create_dataset(
            "v", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[0.0]]]])
        )
        h5.create_dataset(
            "w", data=np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]])
        )

    average = tmp_path / "mean.nc"
    filtered = tmp_path / "mean_80.nc"
    with FlowDataset(raw) as flow:
        temporal_average_volume(flow, average, chunk_size=2)

    apply_valid_fraction_to_average(average, filtered, min_valid_fraction=0.8)

    with h5py.File(average, "r") as original, h5py.File(filtered, "r") as out:
        assert original["u_mean"][0, 0, 0] == 2.0
        assert np.isnan(out["u_mean"][0, 0, 0])
        assert out["v_mean"][0, 0, 0] == 2.5
        assert out["w_mean"][0, 0, 0] == 3.0
        assert out.attrs["derived_operation"] == "apply_valid_fraction_to_average"
        assert out.attrs["derived_from_file"].endswith("mean.nc")
        assert out.attrs["min_valid_count"] == 4
        assert out["provenance"].attrs["derived_from_file"].endswith("mean.nc")


def test_temporal_average_volume_reader(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, output=output, chunk_size=4)

    with TemporalAverageVolume(output) as volume:
        z_index = volume.nearest_z_index(0.0)
        plane = volume.read_z_plane(z_index)

        assert volume.grid_shape == (3, 5, 6)
        assert plane["u"].shape == (5, 6)
        assert plane["speed"].shape == (5, 6)
        assert plane["z_index"] == z_index


def test_temporal_average_volume_reads_x_y_z_planes(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, output=output, chunk_size=4)

    with TemporalAverageVolume(output) as volume:
        z_plane = volume.read_plane("z", 1)
        assert z_plane["horizontal_axis"] == "x"
        assert z_plane["vertical_axis"] == "y"
        assert z_plane["vector_horizontal_name"] == "u"
        assert z_plane["vector_vertical_name"] == "v"
        assert z_plane["speed"].shape == (5, 6)

        y_plane = volume.read_plane("y", 2)
        assert y_plane["horizontal_axis"] == "x"
        assert y_plane["vertical_axis"] == "z"
        assert y_plane["vector_horizontal_name"] == "u"
        assert y_plane["vector_vertical_name"] == "w"
        assert y_plane["speed"].shape == (3, 6)

        x_plane = volume.read_plane("x", 3)
        assert x_plane["horizontal_axis"] == "y"
        assert x_plane["vertical_axis"] == "z"
        assert x_plane["vector_horizontal_name"] == "v"
        assert x_plane["vector_vertical_name"] == "w"
        assert x_plane["speed"].shape == (3, 5)
