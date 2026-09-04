from __future__ import annotations

import shutil
import subprocess
import sys

import h5py

from scripts.interpolate_velocity_folder import build_parser


def test_interpolate_velocity_folder_defaults_match_selected_parameters(tmp_path):
    args = build_parser().parse_args(
        [str(tmp_path), "--output-root", str(tmp_path / "outputs")]
    )

    assert args.invalid_samples == "nan"
    assert args.zero_mask == "vector"
    assert args.max_temporal_gap == 2
    assert args.max_spatial_gap == 5
    assert args.interpolation_passes == 10
    assert args.interpolation_workers == 3


def test_interpolate_velocity_folder_dry_run(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "interpolated"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/interpolate_velocity_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry-run" in result.stdout
    assert (output_root / "manifest.csv").exists()
    assert not (output_root / "case_a" / "interpolated_velocity.nc").exists()


def test_interpolate_velocity_folder_refuses_existing_output_folder(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "interpolated"
    output_root.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/interpolate_velocity_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to use existing output folder" in (result.stdout + result.stderr)


def test_interpolate_velocity_folder_does_not_accept_overwrite(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/interpolate_velocity_folder.py",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unrecognized arguments: --overwrite" in (result.stdout + result.stderr)


def test_interpolate_velocity_folder_creates_interpolated_files(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "interpolated"

    subprocess.run(
        [
            sys.executable,
            "scripts/interpolate_velocity_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--interpolation-axes",
            "t",
            "--interpolation-passes",
            "1",
            "--invalid-samples",
            "zero-or-nan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = output_root / "case_a" / "interpolated_velocity.nc"
    assert output.exists()
    assert (output_root / "manifest.csv").exists()
    assert (output_root / "manifest.json").exists()
    with h5py.File(output, "r") as h5:
        assert h5.attrs["script_name"] == "interpolate_velocity_folder.py"
        assert h5.attrs["operation"] == "spatio_temporal_interpolate_velocity"
        assert h5.attrs["invalid_samples"] == "zero-or-nan"
        assert h5.attrs["interpolation_passes"] == 1
        assert h5.attrs["filled_mask_storage"] == "shared"
        assert "u" in h5
        assert "v" in h5
        assert "w" in h5
        assert "filled_mask" in h5
        assert "u_filled_mask" not in h5
        assert "v_filled_mask" not in h5
        assert "w_filled_mask" not in h5
        assert "command_line" in h5.attrs
