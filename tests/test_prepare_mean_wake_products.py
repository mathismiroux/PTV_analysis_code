from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import h5py


def test_prepare_mean_wake_products_dry_run(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "paper_mean_wake"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
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
    assert not (output_root / "case_a" / "mean.nc").exists()


def test_prepare_mean_wake_products_requires_output_folder_name(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
            str(input_dir),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires a new output folder name" in (result.stdout + result.stderr)


def test_prepare_mean_wake_products_refuses_existing_output_folder(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "paper_mean_wake"
    output_root.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
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


def test_prepare_mean_wake_products_creates_mean_products(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shutil.copy2(tiny_flow_path, input_dir / "case_a.nc")
    output_root = tmp_path / "outputs" / "paper_mean_wake"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--chunk-size",
            "2",
            "--invalid-samples",
            "zero-or-nan",
            "--min-valid-fraction",
            "0.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = output_root / "case_a" / "mean.nc"
    assert output.exists()
    assert (output_root / "manifest.csv").exists()
    assert (output_root / "manifest.json").exists()
    with h5py.File(output, "r") as h5:
        assert h5.attrs["script_name"] == "prepare_mean_wake_products.py"
        assert h5.attrs["invalid_samples"] == "zero-or-nan"
        assert h5.attrs["chunk_size"] == 2
        assert h5.attrs["min_valid_fraction"] == 0.25
        assert h5.attrs["u_inf"] == 4.0
        assert "u_mean" in h5
        assert "v_mean" in h5
        assert "w_mean" in h5
        assert "abs_U" in h5
        assert "speed_from_mean" in h5
        assert "u_over_u_inf" in h5
        assert "wake_deficit" in h5
        assert "command_line" in h5.attrs
