from __future__ import annotations

import h5py
import numpy as np

from ptv_flow.postprocess import (
    TemporalAverageVolume,
    apply_valid_fraction_to_average,
    reynolds_stresses,
    temporal_average_volume,
    turbulent_kinetic_energy,
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


def _expected_component_prime2(
    data: np.ndarray, mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = (data != 0.0) & np.isfinite(mean)[None, :, :, :]
    counts = valid.sum(axis=0, dtype=np.uint32)
    variances = np.full(data.shape[1:], np.nan, dtype=np.float64)
    fluctuation = data - mean[None, :, :, :]
    np.divide(
        np.where(valid, fluctuation * fluctuation, 0.0).sum(
            axis=0, dtype=np.float64
        ),
        counts,
        out=variances,
        where=counts > 0,
    )
    return variances, counts


def _expected_reynolds_stress(
    data_a: np.ndarray,
    mean_a: np.ndarray,
    data_b: np.ndarray,
    mean_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        (data_a != 0.0)
        & (data_b != 0.0)
        & np.isfinite(mean_a)[None, :, :, :]
        & np.isfinite(mean_b)[None, :, :, :]
    )
    counts = valid.sum(axis=0, dtype=np.uint32)
    stress = np.full(data_a.shape[1:], np.nan, dtype=np.float64)
    product = (data_a - mean_a[None, :, :, :]) * (
        data_b - mean_b[None, :, :, :]
    )
    np.divide(
        np.where(valid, product, 0.0).sum(axis=0, dtype=np.float64),
        counts,
        out=stress,
        where=counts > 0,
    )
    return stress, counts


def test_temporal_average_component_mask_matches_fixture(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, output=output, chunk_size=2)

    with h5py.File(tiny_flow_path, "r") as src, h5py.File(output, "r") as out:
        assert out.attrs["zero_mask"] == "component"
        assert out.attrs["invalid_samples"] == "zero"
        assert set(out.keys()) == {
            "abs_U",
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
        assert out["provenance"].attrs["invalid_samples"] == "zero"

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
        np.testing.assert_allclose(out["abs_U"][:], expected_speed, equal_nan=True)


def test_temporal_average_writes_wake_products_and_case_metadata(
    tiny_flow_path, tmp_path
):
    output = tmp_path / "mean.nc"
    metadata = {
        "case_id": "tiny_static_x3p5d",
        "label": "Tiny static fixture",
        "motion_type": "static",
        "u_inf": 4.0,
        "rotor_diameter": 1.2,
        "rotor_frequency_hz": 8.0,
        "blade_passing_frequency_hz": 24.0,
    }
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(
            flow,
            output=output,
            chunk_size=2,
            u_inf=4.0,
            metadata=metadata,
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["case_id"] == "tiny_static_x3p5d"
        assert out.attrs["label"] == "Tiny static fixture"
        assert out.attrs["motion_type"] == "static"
        assert out.attrs["u_inf"] == 4.0
        assert out["provenance"].attrs["case_id"] == "tiny_static_x3p5d"

        expected_u_over_u_inf = out["u_mean"][:] / 4.0
        expected_wake_deficit = (4.0 - out["u_mean"][:]) / 4.0
        np.testing.assert_allclose(
            out["u_over_u_inf"][:],
            expected_u_over_u_inf,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            out["wake_deficit"][:],
            expected_wake_deficit,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            out["wake_mask_u09"][:],
            out["u_mean"][:] / 4.0 < 0.9,
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


def test_temporal_average_invalid_samples_zero_or_nan(tmp_path):
    raw = tmp_path / "zero_and_nan.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(4, dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset(
            "u",
            data=np.array([[[[0.0]]], [[[2.0]]], [[[np.nan]]], [[[4.0]]]]),
        )
        h5.create_dataset(
            "v",
            data=np.array([[[[1.0]]], [[[0.0]]], [[[np.nan]]], [[[5.0]]]]),
        )
        h5.create_dataset(
            "w",
            data=np.array([[[[1.0]]], [[[3.0]]], [[[0.0]]], [[[np.nan]]]]),
        )

    output = tmp_path / "mean.nc"
    with FlowDataset(raw) as flow:
        temporal_average_volume(
            flow,
            output,
            chunk_size=2,
            invalid_samples="zero-or-nan",
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["invalid_samples"] == "zero-or-nan"
        assert out["u_mean"][0, 0, 0] == 3.0
        assert out["v_mean"][0, 0, 0] == 3.0
        assert out["w_mean"][0, 0, 0] == 2.0
        np.testing.assert_array_equal(out["u_count"][:], [[[2]]])
        np.testing.assert_array_equal(out["v_count"][:], [[[2]]])
        np.testing.assert_array_equal(out["w_count"][:], [[[2]]])


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


def test_turbulent_kinetic_energy_matches_fixture(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    tke_output = tmp_path / "tke.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, average_output, chunk_size=2)

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(average_output) as mean:
        turbulent_kinetic_energy(flow, mean, output=tke_output, chunk_size=2)

    with (
        h5py.File(tiny_flow_path, "r") as src,
        h5py.File(average_output, "r") as mean,
        h5py.File(tke_output, "r") as out,
    ):
        assert out.attrs["operation"] == "turbulent_kinetic_energy"
        assert out.attrs["source_file_name"] == "tiny_flow.nc"
        assert out.attrs["mean_file_name"] == "mean.nc"
        assert out["provenance"].attrs["source_file"].endswith("tiny_flow.nc")
        assert out["provenance"].attrs["mean_file"].endswith("mean.nc")

        expected_prime2 = {}
        for name in ("u", "v", "w"):
            variance, count = _expected_component_prime2(
                src[name][:],
                mean[f"{name}_mean"][:],
            )
            expected_prime2[name] = variance
            np.testing.assert_allclose(
                out[f"{name}_prime2_mean"][:], variance, equal_nan=True
            )
            np.testing.assert_array_equal(out[f"{name}_prime2_count"][:], count)

        expected_tke = 0.5 * (
            expected_prime2["u"] + expected_prime2["v"] + expected_prime2["w"]
        )
        np.testing.assert_allclose(out["tke"][:], expected_tke, equal_nan=True)


def test_turbulent_kinetic_energy_refuses_shape_mismatch(tiny_flow_path, tmp_path):
    mismatch = tmp_path / "mismatch_mean.nc"
    with h5py.File(mismatch, "w") as h5:
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("u_mean", data=np.zeros((1, 1, 1)))
        h5.create_dataset("v_mean", data=np.zeros((1, 1, 1)))
        h5.create_dataset("w_mean", data=np.zeros((1, 1, 1)))

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(mismatch) as mean:
        try:
            turbulent_kinetic_energy(flow, mean, tmp_path / "tke.nc")
        except ValueError as exc:
            assert "grid shape does not match" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_turbulent_kinetic_energy_refuses_missing_mean_provenance(
    tiny_flow_path, tmp_path
):
    mean_path = tmp_path / "mean_without_provenance.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, mean_path, chunk_size=2)
    with h5py.File(mean_path, "r+") as mean:
        del mean.attrs["source_file"]

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(mean_path) as mean:
        try:
            turbulent_kinetic_energy(flow, mean, tmp_path / "tke.nc")
        except ValueError as exc:
            assert "missing 'source_file' provenance" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_turbulent_kinetic_energy_refuses_different_mean_source(
    tiny_flow_path, tmp_path
):
    mean_path = tmp_path / "mean_from_other_raw.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, mean_path, chunk_size=2)
    with h5py.File(mean_path, "r+") as mean:
        mean.attrs["source_file"] = str((tmp_path / "other_raw.nc").resolve())

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(mean_path) as mean:
        try:
            turbulent_kinetic_energy(flow, mean, tmp_path / "tke.nc")
        except ValueError as exc:
            assert "provenance does not match raw file" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_reynolds_stresses_selected_components_match_fixture(
    tiny_flow_path, tmp_path
):
    average_output = tmp_path / "mean.nc"
    stress_output = tmp_path / "reynolds.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, average_output, chunk_size=2)

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(average_output) as mean:
        reynolds_stresses(
            flow,
            mean,
            output=stress_output,
            components=("uv", "ww"),
            chunk_size=2,
        )

    with (
        h5py.File(tiny_flow_path, "r") as src,
        h5py.File(average_output, "r") as mean,
        h5py.File(stress_output, "r") as out,
    ):
        assert out.attrs["operation"] == "reynolds_stresses"
        assert out.attrs["source_file_name"] == "tiny_flow.nc"
        assert out.attrs["mean_file_name"] == "mean.nc"
        assert set(out.keys()) == {
            "provenance",
            "uv_count",
            "uv_reynolds_stress",
            "ww_count",
            "ww_reynolds_stress",
            "x",
            "y",
            "z",
        }

        for component in ("uv", "ww"):
            first, second = component
            expected, count = _expected_reynolds_stress(
                src[first][:],
                mean[f"{first}_mean"][:],
                src[second][:],
                mean[f"{second}_mean"][:],
            )
            np.testing.assert_allclose(
                out[f"{component}_reynolds_stress"][:],
                expected,
                equal_nan=True,
            )
            np.testing.assert_array_equal(out[f"{component}_count"][:], count)


def test_reynolds_stresses_all_components(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    stress_output = tmp_path / "reynolds_all.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, average_output, chunk_size=2)

    with FlowDataset(tiny_flow_path) as flow, TemporalAverageVolume(average_output) as mean:
        reynolds_stresses(flow, mean, output=stress_output, chunk_size=2)

    with h5py.File(stress_output, "r") as out:
        for component in ("uu", "uv", "uw", "vv", "vw", "ww"):
            assert f"{component}_reynolds_stress" in out
            assert f"{component}_count" in out


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
