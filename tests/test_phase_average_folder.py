from __future__ import annotations

import shutil
import subprocess
import sys

import h5py

from scripts.phase_average_folder import build_parser, case_id_for_source


def test_phase_average_folder_defaults(tmp_path):
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )

    assert args.pattern == "SurgeLF*/interpolated_velocity.nc"
    assert args.frequency_hz == 2.0
    assert args.phase_signal is None
    assert args.phase_offset == 0.0
    assert args.n_phase_bins == 32
    assert args.invalid_samples == "nan"
    assert args.zero_mask == "vector"
    assert args.min_valid_fraction == 0.3
    assert args.chunk_size == 50
    assert args.u_inf == 4.0


def test_case_id_for_nested_interpolated_source(tmp_path):
    input_root = tmp_path / "interpolated"
    source = input_root / "case_a" / "interpolated_velocity.nc"

    assert case_id_for_source(input_root, source) == "case_a"


def test_phase_average_folder_dry_run(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "SurgeLF_case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "phase"
    (output_root / "SurgeLF_case_a").mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase_average_folder.py",
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
    assert (output_root / "phase_average_dry_run_manifest.csv").exists()
    assert not (output_root / "phase_average_manifest.csv").exists()
    assert not (output_root / "SurgeLF_case_a" / "phase_average.nc").exists()


def test_phase_average_folder_refuses_existing_phase_average_manifest(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "SurgeLF_case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "phase"
    (output_root / "SurgeLF_case_a").mkdir(parents=True)
    (output_root / "phase_average_manifest.csv").write_text("old\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase_average_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite existing phase-average manifest" in (
        result.stdout + result.stderr
    )


def test_phase_average_folder_skips_missing_output_case_folder(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "SurgeLF_case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "phase"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase_average_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "skipped" in result.stdout
    assert "output case folder does not exist" in result.stdout
    assert (output_root / "phase_average_manifest.csv").exists()
    assert not (output_root / "SurgeLF_case_a" / "phase_average.nc").exists()


def test_phase_average_folder_does_not_overwrite_existing_phase_average(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "SurgeLF_case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "phase"
    case_dir = output_root / "SurgeLF_case_a"
    case_dir.mkdir(parents=True)
    (case_dir / "phase_average.nc").write_bytes(b"old")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/phase_average_folder.py",
            str(input_dir),
            "--output-root",
            str(output_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "exists" in result.stdout
    assert "output exists; choose a new output folder" in result.stdout
    assert (case_dir / "phase_average.nc").read_bytes() == b"old"


def test_phase_average_folder_creates_phase_average_from_nested_inputs(
    tiny_flow_path, tmp_path
):
    input_root = tmp_path / "interpolated"
    case_dir = input_root / "SurgeLF_case_a"
    case_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "phase"
    (output_root / "SurgeLF_case_a").mkdir(parents=True)

    subprocess.run(
        [
            sys.executable,
            "scripts/phase_average_folder.py",
            str(input_root),
            "--output-root",
            str(output_root),
            "--frequency-hz",
            "1.0",
            "--n-phase-bins",
            "4",
            "--chunk-size",
            "2",
            "--invalid-samples",
            "nan",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = output_root / "SurgeLF_case_a" / "phase_average.nc"
    assert output.exists()
    assert (output_root / "phase_average_manifest.csv").exists()
    assert (output_root / "phase_average_manifest.json").exists()
    with h5py.File(output, "r") as h5:
        assert h5.attrs["script_name"] == "phase_average_folder.py"
        assert h5.attrs["operation"] == "phase_average_volume"
        assert h5.attrs["frequency_hz"] == 1.0
        assert h5.attrs["n_phase_bins"] == 4
        assert h5.attrs["chunk_size"] == 2
        assert h5.attrs["zero_mask"] == "vector"
        assert h5.attrs["u_inf"] == 4.0
        assert "phase" in h5
        assert "phase_sample_count" in h5
        assert "u_phase_mean" in h5
        assert "v_phase_mean" in h5
        assert "w_phase_mean" in h5
        assert "u_phase_count" in h5
        assert "u_coherent" in h5
        assert "u_harmonic_amplitude" in h5
        assert "wake_deficit_phase" in h5
        assert "command_line" in h5.attrs
