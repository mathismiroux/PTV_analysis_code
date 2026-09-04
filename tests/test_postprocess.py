from __future__ import annotations

import h5py
import numpy as np

from ptv_flow.postprocess import (
    PhaseAverageVolume,
    TemporalAverageVolume,
    apply_valid_fraction_to_average,
    extract_z_slab,
    phase_average_volume,
    reynolds_stresses,
    spatio_temporal_interpolate_velocity,
    temporal_average_volume,
    turbulent_kinetic_energy,
)
from ptv_flow.reader import FlowDataset
from ptv_flow.validity import valid_component_samples


def _expected_component_mean(
    data: np.ndarray,
    invalid_samples: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_component_samples(data, invalid_samples)
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
    data: np.ndarray,
    mean: np.ndarray,
    invalid_samples: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    valid = valid_component_samples(data, invalid_samples) & np.isfinite(
        mean
    )[None, :, :, :]
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
    invalid_samples: str = "nan",
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        valid_component_samples(data_a, invalid_samples)
        & valid_component_samples(data_b, invalid_samples)
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


def test_phase_average_recovers_phase_bins_and_harmonic(tmp_path):
    raw = tmp_path / "phase_known.nc"
    phases = np.tile(
        np.array([np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0]),
        2,
    )
    u = (10.0 + 2.0 * np.cos(phases) - 3.0 * np.sin(phases)).reshape(8, 1, 1, 1)
    v = (20.0 + np.cos(phases)).reshape(8, 1, 1, 1)
    w = (30.0 - np.sin(phases)).reshape(8, 1, 1, 1)

    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(8, dtype=np.float64) * 0.25)
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("u", data=u)
        h5.create_dataset("v", data=v)
        h5.create_dataset("w", data=w)

    output = tmp_path / "phase_average.nc"
    with FlowDataset(raw) as flow:
        phase_average_volume(
            flow,
            output,
            n_phase_bins=4,
            phase_signal=phases,
            chunk_size=3,
            u_inf=4.0,
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["operation"] == "phase_average_volume"
        assert out.attrs["n_phase_bins"] == 4
        np.testing.assert_array_equal(out["phase_sample_count"][:], [2, 2, 2, 2])
        np.testing.assert_array_equal(out["u_phase_count"][:, 0, 0, 0], [2, 2, 2, 2])
        np.testing.assert_allclose(out["u_phase_mean"][:, 0, 0, 0], u[:, 0, 0, 0][0:4])
        np.testing.assert_allclose(out["u_mean"][0, 0, 0], 10.0)
        np.testing.assert_allclose(
            out["u_coherent"][:, 0, 0, 0],
            u[:, 0, 0, 0][0:4] - 10.0,
        )
        np.testing.assert_allclose(out["u_harmonic_offset"][0, 0, 0], 10.0, atol=1e-12)
        np.testing.assert_allclose(out["u_harmonic_a"][0, 0, 0], 2.0, atol=1e-12)
        np.testing.assert_allclose(out["u_harmonic_b"][0, 0, 0], -3.0, atol=1e-12)
        np.testing.assert_allclose(
            out["u_harmonic_amplitude"][0, 0, 0],
            np.sqrt(13.0),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            out["wake_deficit_phase"][:, 0, 0, 0],
            (4.0 - out["u_phase_mean"][:, 0, 0, 0]) / 4.0,
        )
        np.testing.assert_allclose(
            out["wake_deficit_coherent"][:, 0, 0, 0],
            -out["u_coherent"][:, 0, 0, 0] / 4.0,
        )


def test_first_harmonic_fit_keeps_mean_out_of_amplitude_with_missing_bin():
    from ptv_flow.postprocess import TWO_PI, _first_harmonic_from_phase_means

    phase_centers = (np.arange(4, dtype=np.float64) + 0.5) * TWO_PI / 4.0
    values = 10.0 + 2.0 * np.cos(phase_centers) - 3.0 * np.sin(phase_centers)
    values = values.reshape(4, 1, 1, 1)
    values[2, 0, 0, 0] = np.nan

    offset, a, b, amplitude, phase = _first_harmonic_from_phase_means(
        values,
        phase_centers,
    )

    np.testing.assert_allclose(offset[0, 0, 0], 10.0, atol=1e-12)
    np.testing.assert_allclose(a[0, 0, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(b[0, 0, 0], -3.0, atol=1e-12)
    np.testing.assert_allclose(amplitude[0, 0, 0], np.sqrt(13.0), atol=1e-12)
    assert np.isfinite(phase[0, 0, 0])


def test_phase_average_from_frequency_masks_sparse_bins(tmp_path):
    raw = tmp_path / "phase_sparse.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(4, dtype=np.float64) * 0.25)
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]], [[[4.0]]]]))
        h5.create_dataset("v", data=np.ones((4, 1, 1, 1)))
        h5.create_dataset("w", data=np.ones((4, 1, 1, 1)))

    output = tmp_path / "phase_average.nc"
    with FlowDataset(raw) as flow:
        phase_average_volume(
            flow,
            output,
            n_phase_bins=4,
            frequency_hz=1.0,
            min_valid_fraction=1.0,
            chunk_size=2,
            invalid_samples="zero",
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["phase_source"] == "frequency"
        assert out.attrs["frequency_hz"] == 1.0
        assert np.isnan(out["u_phase_mean"][1, 0, 0, 0])
        assert out["u_phase_mean"][0, 0, 0, 0] == 1.0
        assert out["u_phase_count"][1, 0, 0, 0] == 0


def test_temporal_average_component_mask_matches_fixture(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(flow, output=output, chunk_size=2)

    with h5py.File(tiny_flow_path, "r") as src, h5py.File(output, "r") as out:
        assert out.attrs["zero_mask"] == "component"
        assert out.attrs["invalid_samples"] == "nan"
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
        assert out["provenance"].attrs["invalid_samples"] == "nan"

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
        temporal_average_volume(
            flow,
            component_output,
            chunk_size=1,
            zero_mask="component",
            invalid_samples="zero",
        )
        temporal_average_volume(
            flow,
            vector_output,
            chunk_size=1,
            zero_mask="vector",
            invalid_samples="zero",
        )

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
        assert vector.attrs["count_storage"] == "vector"
        assert "u_count" not in vector
        assert "v_count" not in vector
        assert "w_count" not in vector
        np.testing.assert_array_equal(vector["vector_count"][:], [[[3]]])


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
            invalid_samples="zero",
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
        temporal_average_volume(flow, average, chunk_size=2, invalid_samples="zero")

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


def test_extract_z_slab_keeps_centered_three_voxel_width(tmp_path):
    raw = tmp_path / "full.nc"
    u = np.arange(2 * 5 * 3 * 4, dtype=np.float32).reshape(2, 5, 3, 4)
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0, 1.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        h5.create_dataset("y", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("x", data=np.arange(4, dtype=np.float32))
        h5.create_dataset("u", data=u)
        h5.create_dataset("v", data=u + 100.0)
        h5.create_dataset("w", data=u + 200.0)

    output = tmp_path / "z_slab.nc"
    with FlowDataset(raw) as flow:
        extract_z_slab(flow, output=output, z_center=0.0, z_width=3, chunk_size=1)

    with h5py.File(output, "r") as out:
        assert out.attrs["operation"] == "extract_z_slab"
        assert out.attrs["source_z_start_index"] == 1
        assert out.attrs["source_z_stop_index"] == 4
        assert out.attrs["source_z_center_index"] == 2
        assert out.attrs["z_width_voxels"] == 3
        np.testing.assert_array_equal(out["z"][:], [-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(out["u"][:], u[:, 1:4, :, :])
        np.testing.assert_array_equal(out["v"][:], u[:, 1:4, :, :] + 100.0)
        np.testing.assert_array_equal(out["w"][:], u[:, 1:4, :, :] + 200.0)


def test_extract_z_slab_rejects_even_width(tiny_flow_path, tmp_path):
    with FlowDataset(tiny_flow_path) as flow:
        try:
            extract_z_slab(flow, tmp_path / "slab.nc", z_width=2)
        except ValueError as exc:
            assert "must be odd" in str(exc)
        else:
            raise AssertionError("Expected ValueError")


def test_spatio_temporal_interpolation_fills_holes_and_preserves_data(tmp_path):
    raw = tmp_path / "holes.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        u = np.array(
            [
                [[[1.0, 2.0, 3.0]]],
                [[[2.0, 0.0, 4.0]]],
                [[[3.0, 4.0, 5.0]]],
            ]
        )
        h5.create_dataset("u", data=u)
        h5.create_dataset("v", data=u + 10.0)
        w = u + 20.0
        w[1, 0, 0, 1] = 0.0
        h5.create_dataset("w", data=w)

    output = tmp_path / "interpolated.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=output,
            axes=("t", "x"),
            invalid_samples="zero",
        )

    with h5py.File(output, "r") as out:
        assert out.attrs["operation"] == "spatio_temporal_interpolate_velocity"
        assert out.attrs["method"] == "sequential_linear_interpolation"
        assert out.attrs["u_filled_count"] == 1
        assert out.attrs["w_filled_count"] == 1
        np.testing.assert_allclose(
            out["u"][:, 0, 0, :],
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        )
        np.testing.assert_allclose(out["v"][:], (u + 10.0))
        assert bool(out["u_filled_mask"][1, 0, 0, 1])
        assert not bool(out["u_filled_mask"][0, 0, 0, 0])
        assert bool(out["w_filled_mask"][1, 0, 0, 1])
        np.testing.assert_array_equal(out["x"][:], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out["t"][:], [0.0, 1.0, 2.0])


def test_spatio_temporal_interpolation_vector_mask(tmp_path):
    raw = tmp_path / "vector_holes.nc"
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0, 1.0, 2.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("u", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]]]))
        h5.create_dataset("v", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]]]))
        h5.create_dataset("w", data=np.array([[[[1.0]]], [[[0.0]]], [[[3.0]]]]))

    output = tmp_path / "interpolated.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=output,
            axes=("t",),
            zero_mask="vector",
            invalid_samples="zero",
        )

    with h5py.File(output, "r") as out:
        for name in ("u", "v", "w"):
            np.testing.assert_allclose(out[name][:, 0, 0, 0], [1.0, 2.0, 3.0])
            assert bool(out[f"{name}_filled_mask"][1, 0, 0, 0])


def test_spatio_temporal_interpolation_respects_max_spatial_gap(tmp_path):
    raw = tmp_path / "spatial_gap.nc"
    data = np.array([[[[1.0, 0.0, 0.0, 0.0, 5.0]]]])
    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0], dtype=np.float32))
        h5.create_dataset("x", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("u", data=data)
        h5.create_dataset("v", data=data)
        h5.create_dataset("w", data=data)

    max_two = tmp_path / "max_two.nc"
    max_four = tmp_path / "max_four.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=max_two,
            axes=("x",),
            max_spatial_gap=2,
            invalid_samples="zero",
        )
        spatio_temporal_interpolate_velocity(
            flow,
            output=max_four,
            axes=("x",),
            max_spatial_gap=4,
            invalid_samples="zero",
        )

    with h5py.File(max_two, "r") as two, h5py.File(max_four, "r") as four:
        assert two.attrs["max_spatial_gap"] == 2
        assert four.attrs["max_spatial_gap"] == 4
        assert np.isnan(two["u"][0, 0, 0, 2])
        assert four["u"][0, 0, 0, 2] == 3.0
        assert not bool(two["u_filled_mask"][0, 0, 0, 2])
        assert bool(four["u_filled_mask"][0, 0, 0, 2])


def test_spatio_temporal_interpolation_vector_mode_reuses_hole_mask(tmp_path):
    raw = tmp_path / "shared_holes.nc"
    data = np.ones((3, 3, 3, 3), dtype=float)
    data[1, 1, 1, 1] = 0.0

    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("z", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("y", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("x", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("u", data=data)
        h5.create_dataset("v", data=data * 2.0)
        h5.create_dataset("w", data=data * 3.0)

    output = tmp_path / "interpolated.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=output,
            axes=("t",),
            zero_mask="vector",
            invalid_samples="zero",
        )

    with h5py.File(output, "r") as out:
        np.testing.assert_array_equal(out["u_filled_mask"][:], out["v_filled_mask"][:])
        np.testing.assert_array_equal(out["u_filled_mask"][:], out["w_filled_mask"][:])
        assert bool(out["u_filled_mask"][1, 1, 1, 1])
        assert bool(out["v_filled_mask"][1, 1, 1, 1])
        assert bool(out["w_filled_mask"][1, 1, 1, 1])


def test_spatio_temporal_interpolation_parallel_workers_match_serial(tmp_path):
    raw = tmp_path / "parallel_holes.nc"
    data = np.ones((3, 3, 3, 3), dtype=float)
    data[2, :, :, :] = 3.0
    data[1, 1, 1, 1] = 0.0

    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("z", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("y", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("x", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("u", data=data)
        h5.create_dataset("v", data=data * 2.0)
        h5.create_dataset("w", data=data * 3.0)

    serial = tmp_path / "serial.nc"
    parallel = tmp_path / "parallel.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=serial,
            axes=("t",),
            zero_mask="vector",
            workers=1,
            invalid_samples="zero",
        )
        spatio_temporal_interpolate_velocity(
            flow,
            output=parallel,
            axes=("t",),
            zero_mask="vector",
            workers=3,
            invalid_samples="zero",
        )

    with h5py.File(serial, "r") as one, h5py.File(parallel, "r") as many:
        assert many.attrs["interpolation_workers"] == 3
        for name in ("u", "v", "w"):
            np.testing.assert_allclose(one[name][:], many[name][:], equal_nan=True)
            np.testing.assert_array_equal(
                one[f"{name}_filled_mask"][:],
                many[f"{name}_filled_mask"][:],
            )


def test_spatio_temporal_interpolation_records_multiple_passes(tmp_path):
    raw = tmp_path / "multi_pass.nc"
    data = np.ones((3, 5, 5, 5), dtype=float)
    data[2, :, :, :] = 3.0
    for z_index, y_index, x_index in (
        (1, 1, 1),
        (2, 2, 2),
        (3, 3, 3),
        (3, 3, 1),
    ):
        data[1, z_index, y_index, x_index] = 0.0

    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("z", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("y", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("x", data=np.arange(5, dtype=np.float32))
        h5.create_dataset("u", data=data)
        h5.create_dataset("v", data=data)
        h5.create_dataset("w", data=data)

    two_passes = tmp_path / "two_passes.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=two_passes,
            axes=("t",),
            passes=2,
            overwrite=True,
            invalid_samples="zero",
        )

    with h5py.File(two_passes, "r") as two:
        assert two.attrs["interpolation_passes"] == 2
        assert two["u"][1, 2, 2, 2] == 2.0
        assert two.attrs["u_filled_count"] == 4


def test_spatio_temporal_interpolation_respects_max_temporal_gap(tmp_path):
    raw = tmp_path / "temporal_gap.nc"
    data = np.ones((7, 3, 3, 3), dtype=float)
    data[:, 1, 1, 1] = 0.0
    data[0, 1, 1, 1] = 1.0
    data[6, 1, 1, 1] = 7.0

    with h5py.File(raw, "w") as h5:
        h5.create_dataset("t", data=np.arange(7, dtype=np.float32))
        h5.create_dataset("z", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("y", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("x", data=np.arange(3, dtype=np.float32))
        h5.create_dataset("u", data=data)
        h5.create_dataset("v", data=data)
        h5.create_dataset("w", data=data)

    max_two = tmp_path / "max_two.nc"
    max_three = tmp_path / "max_three.nc"
    with FlowDataset(raw) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=max_two,
            axes=("t",),
            max_temporal_gap=2,
            invalid_samples="zero",
        )
        spatio_temporal_interpolate_velocity(
            flow,
            output=max_three,
            axes=("t",),
            max_temporal_gap=3,
            invalid_samples="zero",
        )

    with h5py.File(max_two, "r") as two, h5py.File(max_three, "r") as three:
        assert two.attrs["max_temporal_gap"] == 2
        assert three.attrs["max_temporal_gap"] == 3
        assert np.isnan(two["u"][3, 1, 1, 1])
        assert three["u"][3, 1, 1, 1] == 4.0
        assert not bool(two["u_filled_mask"][3, 1, 1, 1])
        assert bool(three["u_filled_mask"][3, 1, 1, 1])


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


def test_temporal_average_volume_reader_uses_vector_count(tiny_flow_path, tmp_path):
    output = tmp_path / "mean.nc"
    with FlowDataset(tiny_flow_path) as flow:
        temporal_average_volume(
            flow,
            output=output,
            chunk_size=4,
            zero_mask="vector",
        )

    with h5py.File(output, "r") as h5:
        assert "vector_count" in h5
        assert "u_count" not in h5
    with TemporalAverageVolume(output) as volume:
        plane = volume.read_plane("z", 1)

        assert plane["vector_count"].shape == (5, 6)
        np.testing.assert_array_equal(plane["u_count"], plane["vector_count"])
        np.testing.assert_array_equal(plane["v_count"], plane["vector_count"])
        np.testing.assert_array_equal(plane["w_count"], plane["vector_count"])


def test_phase_average_volume_reads_x_y_z_planes(tiny_flow_path, tmp_path):
    output = tmp_path / "phase_average.nc"
    with FlowDataset(tiny_flow_path) as flow:
        phase_average_volume(
            flow,
            output=output,
            n_phase_bins=2,
            frequency_hz=1.0,
            chunk_size=2,
        )

    with PhaseAverageVolume(output) as volume:
        assert volume.n_phase_bins == 2
        z_plane = volume.read_plane(phase_index=0, axis="z", index=1)
        assert z_plane["horizontal_axis"] == "x"
        assert z_plane["vertical_axis"] == "y"
        assert z_plane["vector_horizontal_name"] == "u"
        assert z_plane["vector_vertical_name"] == "v"
        assert z_plane["speed"].shape == (5, 6)
        assert z_plane["u_count"].shape == (5, 6)

        y_plane = volume.read_plane(phase_index=1, axis="y", index=2)
        assert y_plane["horizontal_axis"] == "x"
        assert y_plane["vertical_axis"] == "z"
        assert y_plane["vector_horizontal_name"] == "u"
        assert y_plane["vector_vertical_name"] == "w"
        assert y_plane["speed"].shape == (3, 6)

        x_plane = volume.read_plane(phase_index=0, axis="x", index=3)
        assert x_plane["horizontal_axis"] == "y"
        assert x_plane["vertical_axis"] == "z"
        assert x_plane["vector_horizontal_name"] == "v"
        assert x_plane["vector_vertical_name"] == "w"
        assert x_plane["speed"].shape == (3, 5)

        coherent_plane = volume.read_plane(
            phase_index=0,
            axis="z",
            index=1,
            field="coherent",
        )
        assert coherent_plane["field"] == "coherent"
        assert coherent_plane["speed"].shape == (5, 6)

        phase_series = volume.read_phase_series_at(
            z_index=1,
            y_index=2,
            x_index=3,
            field="phase_mean",
        )
        assert phase_series["field"] == "phase_mean"
        assert phase_series["phase_degrees"].shape == (2,)
        assert phase_series["u"].shape == (2,)
        assert phase_series["u_count"].shape == (2,)

        harmonic_plane = volume.read_harmonic_plane(
            axis="z",
            index=1,
            component="u",
            quantity="amplitude",
        )
        assert harmonic_plane["horizontal_axis"] == "x"
        assert harmonic_plane["vertical_axis"] == "y"
        assert harmonic_plane["dataset"] == "u_harmonic_amplitude"
        assert harmonic_plane["data"].shape == (5, 6)
