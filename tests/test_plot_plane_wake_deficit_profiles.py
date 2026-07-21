from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np

from scripts.plot_plane_wake_deficit_profiles import load_plane_wake_profile


def _write_mean(path, label, x_values, y_values, z_values, u_values):
    path.parent.mkdir(parents=True)
    u = np.asarray(u_values, dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["label"] = label
        h5.attrs["u_inf"] = 4.0
        h5.create_dataset("x", data=np.asarray(x_values, dtype=np.float32))
        h5.create_dataset("y", data=np.asarray(y_values, dtype=np.float32))
        h5.create_dataset("z", data=np.asarray(z_values, dtype=np.float32))
        h5.create_dataset("u_mean", data=u)


def test_plane_profile_excludes_zero_values(tmp_path):
    mean_file = tmp_path / "Static_1D__b64" / "mean.nc"
    _write_mean(
        mean_file,
        "Static_1D__b64",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[0.0], [2.0]]],
    )

    profile = load_plane_wake_profile(
        mean_file,
        plane_axis="z",
        plane_value=0.0,
        axis_coordinate=0.0,
        rotor_diameter=1.0,
        invalid_samples="zero-or-nan",
        u_inf=None,
    )

    assert np.isnan(profile.wake_deficit[0])
    assert profile.wake_deficit[1] == 0.5
    assert profile.valid.tolist() == [False, True]


def test_plot_plane_wake_deficit_profiles_writes_outputs(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 3.0),
    )
    _write_mean(
        mean_root / "Surge_1D__b64" / "mean.nc",
        "Surge_1D__b64",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 2.0),
    )
    output = tmp_path / "figures" / "plane_profiles_001"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_plane_wake_deficit_profiles.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--plane-axis",
            "z",
            "--plane-value",
            "0",
            "--rotor-y",
            "0",
            "--rotor-diameter",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "plane_z0_wake_deficit_profiles.png").exists()
    manifest = output / "plane_z0_wake_deficit_profiles.csv"
    data_csv = output / "plane_z0_wake_deficit_profiles_data.csv"
    assert manifest.exists()
    assert data_csv.exists()
    assert "Surge_1D__b64" in manifest.read_text(encoding="utf-8")
    assert "coordinate_over_d" in data_csv.read_text(encoding="utf-8")


def test_plot_plane_wake_deficit_profiles_refuses_existing_output_folder(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0],
        [0.0],
        [0.0],
        [[[3.0]]],
    )
    output = tmp_path / "figures" / "plane_profiles_001"
    output.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_plane_wake_deficit_profiles.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--rotor-y",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to use existing output folder" in (result.stdout + result.stderr)
