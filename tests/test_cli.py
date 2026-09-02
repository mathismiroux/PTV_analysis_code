from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from ptv_flow.cli import _invalid_samples_from_average, _invalid_samples_from_interpolated
from ptv_flow.postprocess import TemporalAverageVolume
from ptv_flow.reader import FlowDataset


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--temporal-average" in result.stdout
    assert "--average-plane" in result.stdout
    assert "--inspect" in result.stdout
    assert "--average-file" in result.stdout
    assert "--compare-average" in result.stdout
    assert "--interpolated-file" in result.stdout
    assert "--compare-interpolated" in result.stdout
    assert "--max-spatial-gap" in result.stdout
    assert "--interpolation-passes" in result.stdout
    assert "--max-temporal-gap" in result.stdout
    assert "--interpolation-workers" in result.stdout
    assert "--extract-z-slab" in result.stdout
    assert "--z-slab-center" in result.stdout
    assert "--z-slab-width" in result.stdout
    assert "--min-valid-fraction" in result.stdout
    assert "--invalid-samples" in result.stdout
    assert "--quantity" in result.stdout
    assert "--plane" in result.stdout
    assert "--plane-value" in result.stdout
    assert "--overwrite" in result.stdout
    assert "--apply-valid-fraction" in result.stdout
    assert "--tke" in result.stdout
    assert "--reynolds-stress" in result.stdout
    assert "--stress-components" in result.stdout
    assert "--mean-file" in result.stdout
    assert "--case" in result.stdout
    assert "--cases-file" in result.stdout
    assert "--processing-id" in result.stdout
    assert "--postprocess-basic" in result.stdout
    assert "--cases" in result.stdout


def test_cli_rejects_average_file_without_compare_average(tiny_flow_path, tmp_path):
    average_file = tmp_path / "mean.nc"
    average_file.write_bytes(b"not used")

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--inspect",
            "--average-file",
            str(average_file),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--average-file is only used with --compare-average" in (
        result.stdout + result.stderr
    )


def test_cli_rejects_interpolated_file_without_compare_interpolated(
    tiny_flow_path, tmp_path
):
    interpolated_file = tmp_path / "interpolated.nc"
    interpolated_file.write_bytes(b"not used")

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--inspect",
            "--interpolated-file",
            str(interpolated_file),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--interpolated-file is only used with --compare-interpolated" in (
        result.stdout + result.stderr
    )


def test_cli_compare_inspect_uses_average_invalid_samples_metadata(tmp_path):
    average_file = tmp_path / "mean.nc"
    with h5py.File(average_file, "w") as h5:
        h5.attrs["invalid_samples"] = "zero-or-nan"
        h5.create_dataset("x", data=[0.0])
        h5.create_dataset("y", data=[0.0])
        h5.create_dataset("z", data=[0.0])
        h5.create_dataset("u_mean", data=[[[1.0]]])
        h5.create_dataset("v_mean", data=[[[1.0]]])
        h5.create_dataset("w_mean", data=[[[1.0]]])

    with TemporalAverageVolume(average_file) as average:
        assert _invalid_samples_from_average(average, None) == "zero-or-nan"
        assert _invalid_samples_from_average(average, "zero") == "zero"


def test_cli_compare_inspect_uses_interpolated_invalid_samples_metadata(tmp_path):
    interpolated_file = tmp_path / "interpolated.nc"
    with h5py.File(interpolated_file, "w") as h5:
        h5.attrs["invalid_samples"] = "zero-or-nan"
        h5.create_dataset("t", data=[0.0])
        h5.create_dataset("x", data=[0.0])
        h5.create_dataset("y", data=[0.0])
        h5.create_dataset("z", data=[0.0])
        h5.create_dataset("u", data=[[[[1.0]]]])
        h5.create_dataset("v", data=[[[[1.0]]]])
        h5.create_dataset("w", data=[[[[1.0]]]])

    with FlowDataset(interpolated_file) as interpolated:
        assert _invalid_samples_from_interpolated(interpolated, None) == "zero-or-nan"
        assert _invalid_samples_from_interpolated(interpolated, "zero") == "zero"


def test_cli_summarizes_tiny_fixture(tiny_flow_path):
    result = subprocess.run(
        [sys.executable, "main.py", str(tiny_flow_path), "--frame", "0"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Velocity shape: (4, 3, 5, 6)" in result.stdout
    assert "Voxel size" in result.stdout


def test_cli_temporal_average_and_average_plane(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    figure_output = tmp_path / "mean_z0.png"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--temporal-average",
            "--chunk-size",
            "2",
            "--output",
            str(average_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert average_output.exists()

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(average_output),
            "--average-plane",
            "--plane",
            "x",
            "--plane-value",
            "0",
            "--quantity",
            "u",
            "--save",
            str(figure_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert figure_output.exists()


def test_cli_case_temporal_average_uses_default_case_output(tmp_path):
    repo_root = Path(__file__).parents[1]
    output = tmp_path / "outputs" / "tiny_static_x3p5d" / "mean.nc"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(repo_root / "tests" / "data" / "cases.yaml"),
            "--temporal-average",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert output.exists()
    with h5py.File(output, "r") as out:
        assert out.attrs["case_id"] == "tiny_static_x3p5d"
        assert out.attrs["processing_id"] == "tiny_static_x3p5d"
        assert out.attrs["downstream_distance"] == "3.5D"
        assert "wake_deficit" in out
        assert "wake_mask_u09" in out

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(repo_root / "tests" / "data" / "cases.yaml"),
            "--temporal-average",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    second_output = tmp_path / "outputs" / "tiny_static_x3p5d_02" / "mean.nc"
    assert second_output.exists()
    with h5py.File(second_output, "r") as out:
        assert out.attrs["processing_id"] == "tiny_static_x3p5d_02"


def test_cli_case_tke_and_reynolds_use_case_mean_file(tmp_path):
    repo_root = Path(__file__).parents[1]
    cases_file = repo_root / "tests" / "data" / "cases.yaml"
    mean_output = tmp_path / "outputs" / "tiny_static_x3p5d" / "mean.nc"
    tke_output = tmp_path / "outputs" / "tiny_static_x3p5d" / "tke.nc"
    stress_output = (
        tmp_path / "outputs" / "tiny_static_x3p5d" / "reynolds_stresses.nc"
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(cases_file),
            "--temporal-average",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert mean_output.exists()

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(cases_file),
            "--tke",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert tke_output.exists()
    with h5py.File(tke_output, "r") as out:
        assert out.attrs["case_id"] == "tiny_static_x3p5d"
        assert out.attrs["processing_id"] == "tiny_static_x3p5d"
        assert "tke" in out

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(cases_file),
            "--reynolds-stress",
            "--stress-components",
            "uv",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert stress_output.exists()
    with h5py.File(stress_output, "r") as out:
        assert out.attrs["case_id"] == "tiny_static_x3p5d"
        assert out.attrs["processing_id"] == "tiny_static_x3p5d"
        assert "uv_reynolds_stress" in out


def test_cli_case_tke_requires_processing_id_for_ambiguous_mean_files(tmp_path):
    repo_root = Path(__file__).parents[1]
    cases_file = repo_root / "tests" / "data" / "cases.yaml"
    for processing_id in ("tiny_static_x3p5d", "tiny_static_x3p5d_02"):
        mean_dir = tmp_path / "outputs" / processing_id
        mean_dir.mkdir(parents=True)
        (mean_dir / "mean.nc").write_bytes(b"not opened")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(cases_file),
            "--tke",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "--processing-id" in (result.stdout + result.stderr)


def test_cli_case_postprocess_basic_creates_all_basic_products(tmp_path):
    repo_root = Path(__file__).parents[1]
    cases_file = repo_root / "tests" / "data" / "cases.yaml"
    output_dir = tmp_path / "outputs" / "tiny_static_x3p5d"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--case",
            "tiny_static_x3p5d",
            "--cases-file",
            str(cases_file),
            "--postprocess-basic",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert (output_dir / "mean.nc").exists()
    assert (output_dir / "tke.nc").exists()
    assert (output_dir / "reynolds_stresses.nc").exists()
    with h5py.File(output_dir / "reynolds_stresses.nc", "r") as out:
        assert out.attrs["case_id"] == "tiny_static_x3p5d"
        assert "uv_reynolds_stress" in out
        assert "ww_reynolds_stress" in out


def test_cli_cases_postprocess_basic_creates_outputs_for_multiple_cases(tmp_path):
    repo_root = Path(__file__).parents[1]
    velocity = repo_root / "tests" / "data" / "tiny_flow.nc"
    registry = tmp_path / "cases.yaml"
    registry.write_text(
        f"""
cases:
  tiny_static_x0p6d:
    label: Tiny static x/D=0.6
    motion_type: static
    downstream_distance: 0.6D
    u_inf: 4.0
    rotor_diameter: 1.2
    rotor_frequency_hz: 8.0
    blade_passing_frequency_hz: 24.0
    files:
      velocity: "{velocity.as_posix()}"

  tiny_static_x3p5d:
    label: Tiny static x/D=3.5
    motion_type: static
    downstream_distance: 3.5D
    u_inf: 4.0
    rotor_diameter: 1.2
    rotor_frequency_hz: 8.0
    blade_passing_frequency_hz: 24.0
    files:
      velocity: "{velocity.as_posix()}"
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "main.py"),
            "--cases",
            "tiny_static_x0p6d",
            "tiny_static_x3p5d",
            "--cases-file",
            str(registry),
            "--postprocess-basic",
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    for case_id in ("tiny_static_x0p6d", "tiny_static_x3p5d"):
        assert (tmp_path / "outputs" / case_id / "mean.nc").exists()
        assert (tmp_path / "outputs" / case_id / "tke.nc").exists()
        assert (tmp_path / "outputs" / case_id / "reynolds_stresses.nc").exists()


def test_cli_temporal_average_refuses_existing_output(tiny_flow_path, tmp_path):
    output = tmp_path / "existing.nc"
    output.write_text("existing")

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--temporal-average",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite existing output file" in (
        result.stdout + result.stderr
    )
    assert "Traceback" not in (result.stdout + result.stderr)
    assert output.read_text() == "existing"


def test_cli_apply_valid_fraction_to_existing_average(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    filtered_output = tmp_path / "mean_80.nc"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--temporal-average",
            "--chunk-size",
            "2",
            "--output",
            str(average_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(average_output),
            "--apply-valid-fraction",
            "0.8",
            "--output",
            str(filtered_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert filtered_output.exists()


def test_cli_extract_z_slab(tiny_flow_path, tmp_path):
    output = tmp_path / "slab.nc"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--extract-z-slab",
            "--z-slab-center",
            "0",
            "--z-slab-width",
            "3",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with h5py.File(tiny_flow_path, "r") as src, h5py.File(output, "r") as out:
        center_index = int(abs(src["z"][:] - 0.0).argmin())
        np.testing.assert_array_equal(out["z"][:], src["z"][center_index - 1 : center_index + 2])
        assert out["u"].shape == (
            src["t"].shape[0],
            3,
            src["y"].shape[0],
            src["x"].shape[0],
        )


def test_cli_tke_from_temporal_average(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    tke_output = tmp_path / "tke.nc"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--temporal-average",
            "--chunk-size",
            "2",
            "--output",
            str(average_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--tke",
            "--mean-file",
            str(average_output),
            "--chunk-size",
            "2",
            "--output",
            str(tke_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert tke_output.exists()


def test_cli_reynolds_stress_from_temporal_average(tiny_flow_path, tmp_path):
    average_output = tmp_path / "mean.nc"
    stress_output = tmp_path / "reynolds.nc"

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--temporal-average",
            "--chunk-size",
            "2",
            "--output",
            str(average_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            "main.py",
            str(tiny_flow_path),
            "--reynolds-stress",
            "--mean-file",
            str(average_output),
            "--stress-components",
            "uv",
            "ww",
            "--chunk-size",
            "2",
            "--output",
            str(stress_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert stress_output.exists()
