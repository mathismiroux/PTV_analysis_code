from __future__ import annotations

import subprocess
import sys


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
    assert "--min-valid-fraction" in result.stdout
    assert "--quantity" in result.stdout
    assert "--plane" in result.stdout
    assert "--plane-value" in result.stdout


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
