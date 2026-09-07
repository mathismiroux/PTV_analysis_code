from __future__ import annotations

import csv
import subprocess
import sys

import h5py
import numpy as np


def _write_phase_average(path):
    path.parent.mkdir(parents=True)
    phase = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float64)
    zyx_shape = (2, 1, 1)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("phase", data=phase)
        h5.create_dataset("x", data=np.array([0.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("z", data=np.array([0.0, 1.0]))
        for component in ("u", "v", "w"):
            phase_mean = np.empty((phase.size, *zyx_shape), dtype=np.float64)
            fit = 10.0 + 2.0 * np.cos(phase) - 3.0 * np.sin(phase)
            phase_mean[:, 0, 0, 0] = fit
            phase_mean[:, 1, 0, 0] = 4.0
            h5.create_dataset(f"{component}_phase_mean", data=phase_mean)
            h5.create_dataset(f"{component}_harmonic_offset", data=np.full(zyx_shape, 10.0))
            h5.create_dataset(f"{component}_harmonic_a", data=np.full(zyx_shape, 2.0))
            h5.create_dataset(f"{component}_harmonic_b", data=np.full(zyx_shape, -3.0))


def test_add_harmonic_r2_to_phase_average_files_writes_datasets(tmp_path):
    phase_average = tmp_path / "case" / "phase_average.nc"
    _write_phase_average(phase_average)

    subprocess.run(
        [
            sys.executable,
            "scripts/add_harmonic_r2_to_phase_average_files.py",
            str(tmp_path),
            "--components",
            "u,v",
            "--z-chunk-size",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(phase_average, "r") as h5:
        np.testing.assert_allclose(h5["u_harmonic_r2"][0, 0, 0], 1.0)
        np.testing.assert_allclose(h5["v_harmonic_r2"][0, 0, 0], 1.0)
        assert np.isnan(h5["u_harmonic_r2"][1, 0, 0])
        assert "w_harmonic_r2" not in h5
        assert "Coefficient of determination" in h5["u_harmonic_r2"].attrs["description"]

    manifest = tmp_path / "harmonic_r2_manifest.csv"
    assert manifest.exists()
    with manifest.open(newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["status"] == "processed"
    assert row["components_written"] == "u,v"
    assert row["grid_shape_z_y_x"] == "2x1x1"


def test_add_harmonic_r2_to_phase_average_files_skips_existing_without_overwrite(tmp_path):
    phase_average = tmp_path / "case" / "phase_average.nc"
    _write_phase_average(phase_average)
    with h5py.File(phase_average, "a") as h5:
        h5.create_dataset("u_harmonic_r2", data=np.zeros((2, 1, 1), dtype=np.float64))

    subprocess.run(
        [
            sys.executable,
            "scripts/add_harmonic_r2_to_phase_average_files.py",
            str(phase_average),
            "--components",
            "u",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(phase_average, "r") as h5:
        np.testing.assert_array_equal(h5["u_harmonic_r2"][:], np.zeros((2, 1, 1)))


def test_add_harmonic_r2_to_phase_average_files_overwrites_existing(tmp_path):
    phase_average = tmp_path / "case" / "phase_average.nc"
    _write_phase_average(phase_average)
    with h5py.File(phase_average, "a") as h5:
        h5.create_dataset("u_harmonic_r2", data=np.zeros((2, 1, 1), dtype=np.float64))

    subprocess.run(
        [
            sys.executable,
            "scripts/add_harmonic_r2_to_phase_average_files.py",
            str(phase_average),
            "--components",
            "u",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(phase_average, "r") as h5:
        np.testing.assert_allclose(h5["u_harmonic_r2"][0, 0, 0], 1.0)
