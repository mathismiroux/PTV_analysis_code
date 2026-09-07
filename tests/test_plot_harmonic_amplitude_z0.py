from __future__ import annotations

import csv
import subprocess
import sys

import h5py
import numpy as np


def _write_phase_average(path, label, amplitude, motion_type="", sparse_cell=False):
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["label"] = label
        if motion_type:
            h5.attrs["motion_type"] = motion_type
        h5.create_dataset("x", data=np.array([0.0, 1.0], dtype=np.float32))
        h5.create_dataset("y", data=np.array([0.0, 1.0], dtype=np.float32))
        h5.create_dataset("z", data=np.array([-0.1, 0.1], dtype=np.float32))
        h5.create_dataset("phase", data=np.array([0.0, np.pi], dtype=np.float64))
        h5.create_dataset("phase_sample_count", data=np.array([4, 4], dtype=np.uint32))

        counts = np.full((2, 2, 2, 2), 4, dtype=np.uint32)
        if sparse_cell:
            counts[1, 0, 0, 1] = 2
        for component in ("u", "v", "w"):
            h5.create_dataset(f"{component}_phase_count", data=counts)
            h5.create_dataset(
                f"{component}_harmonic_amplitude",
                data=np.full((2, 2, 2), amplitude, dtype=np.float64),
            )
            h5.create_dataset(
                f"{component}_harmonic_phase",
                data=np.zeros((2, 2, 2), dtype=np.float64),
            )
            h5.create_dataset(
                f"{component}_harmonic_a",
                data=np.full((2, 2, 2), amplitude, dtype=np.float64),
            )
            h5.create_dataset(
                f"{component}_harmonic_b",
                data=np.zeros((2, 2, 2), dtype=np.float64),
            )
            h5.create_dataset(
                f"{component}_harmonic_offset",
                data=np.zeros((2, 2, 2), dtype=np.float64),
            )
            h5.create_dataset(
                f"{component}_harmonic_r2",
                data=np.full((2, 2, 2), 0.8, dtype=np.float64),
            )


def test_plot_harmonic_amplitude_z0_filters_moving_case_and_writes_outputs(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "Static_1D__b64" / "phase_average.nc",
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )
    _write_phase_average(
        root / "SurgeLF_1D__b64" / "phase_average.nc",
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
        sparse_cell=True,
    )
    output = tmp_path / "plots"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "SurgeLF",
            "--output-folder",
            str(output),
            "--min-valid-fraction",
            "0.5",
            "--vmin",
            "0",
            "--vmax",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    png = output / "SurgeLF_u_harmonic_amplitude_z0_composite.png"
    manifest = output / "SurgeLF_u_harmonic_amplitude_z0_composite.csv"
    assert png.exists()
    assert manifest.exists()
    with manifest.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["source_label"] == "SurgeLF_1D__b64"
    assert row["motion_type"] == "surge"
    assert row["accepted_cells"] == "3"
    assert row["visible_cells"] == "3"
    assert row["color_vmin"] == "0.0"
    assert row["color_vmax"] == "1.0"


def test_plot_harmonic_amplitude_z0_rejects_static_case(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "Static_1D__b64" / "phase_average.nc",
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "Static",
            "--output-folder",
            str(tmp_path / "plots"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Static case(s) rejected" in (
        result.stdout + result.stderr
    )


def test_plot_harmonic_amplitude_z0_reports_static_match_rejected(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "Static_1D__b64" / "phase_average.nc",
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )
    _write_phase_average(
        root / "SurgeLF_1D__b64" / "phase_average.nc",
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "Static",
            "--output-folder",
            str(tmp_path / "plots"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    message = result.stdout + result.stderr
    assert "Static case(s) rejected" in message
    assert "Static_1D__b64" in message


def test_plot_harmonic_amplitude_z0_can_include_static_baseline(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "Static_1D__b64" / "phase_average.nc",
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )
    output = tmp_path / "plots" / "static_u_amp.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "Static",
            "--include-static",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["source_label"] == "Static_1D__b64"
    assert row["motion_type"] == "static"


def test_plot_harmonic_amplitude_z0_discovers_files_not_in_manifest(tmp_path):
    root = tmp_path / "phase"
    static_file = root / "Static_1D__b64" / "phase_average.nc"
    surge_file = root / "SurgeLF_1D__b64" / "phase_average.nc"
    _write_phase_average(
        static_file,
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )
    _write_phase_average(
        surge_file,
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
    )
    with (root / "phase_average_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["status", "output_file"])
        writer.writeheader()
        writer.writerow({"status": "processed", "output_file": str(surge_file)})
    output = tmp_path / "plots" / "static_u_amp.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "Static",
            "--include-static",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["source_label"] == "Static_1D__b64"


def test_plot_harmonic_amplitude_z0_ignores_static_when_unfiltered(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "Static_1D__b64" / "phase_average.nc",
        "Static_1D__b64",
        0.25,
        motion_type="static",
    )
    _write_phase_average(
        root / "SurgeLF_1D__b64" / "phase_average.nc",
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
    )
    output = tmp_path / "plots"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--output-folder",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "SurgeLF_u_harmonic_amplitude_z0_composite.png").exists()


def test_plot_harmonic_amplitude_z0_composites_all_distances_for_case(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "SurgeLF_1D__b64" / "phase_average.nc",
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
    )
    _write_phase_average(
        root / "SurgeLF_2D__b64" / "phase_average.nc",
        "SurgeLF_2D__b64",
        0.75,
        motion_type="surge",
    )
    output = tmp_path / "plots" / "surge_v_amp.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "SurgeLF",
            "--harmonic-component",
            "v",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["source_label"] for row in rows] == [
        "SurgeLF_1D__b64",
        "SurgeLF_2D__b64",
    ]
    assert {row["component"] for row in rows} == {"v"}


def test_plot_harmonic_amplitude_z0_can_plot_r2(tmp_path):
    root = tmp_path / "phase"
    _write_phase_average(
        root / "SurgeLF_1D__b64" / "phase_average.nc",
        "SurgeLF_1D__b64",
        0.5,
        motion_type="surge",
    )
    output = tmp_path / "plots" / "surge_u_r2.png"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_harmonic_amplitude_z0.py",
            str(root),
            "--case",
            "SurgeLF",
            "--harmonic-component",
            "u",
            "--harmonic-quantity",
            "r2",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.exists()
    with output.with_suffix(".csv").open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["harmonic_quantity"] == "r2"
    assert row["color_vmin"] == "0.0"
    assert row["color_vmax"] == "1.0"
