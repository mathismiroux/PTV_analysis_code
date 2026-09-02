from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np

from scripts.plot_radial_wake_deficit import radial_average_wake_deficit


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
        h5.create_dataset("v_mean", data=np.ones_like(u))
        h5.create_dataset("w_mean", data=np.ones_like(u))


def test_radial_wake_deficit_excludes_zero_values(tmp_path):
    mean_file = tmp_path / "Static_1D__b64" / "mean.nc"
    _write_mean(
        mean_file,
        "Static_1D__b64",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[0.0], [2.0]]],
    )

    volume = radial_average_wake_deficit(
        mean_file,
        axis_y=0.0,
        axis_z=0.0,
        radial_bin_width=2.0,
        invalid_samples="zero-or-nan",
        require_all_components=False,
        u_inf=None,
    )

    assert volume.counts[0, 0] == 1
    assert volume.wake_deficit[0, 0] == 0.5


def test_plot_radial_wake_deficit_writes_plot_and_manifest(tmp_path):
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
        mean_root / "Static_1.625D__b64" / "mean.nc",
        "Static_1.625D__b64",
        [0.5, 1.5],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 2.0),
    )
    output = tmp_path / "figures" / "radial_001"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_radial_wake_deficit.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--rotor-y",
            "0",
            "--rotor-z",
            "0",
            "--radial-bin-width",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "Static_radial_wake_deficit.png").exists()
    manifest = output / "Static_radial_wake_deficit.csv"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "nan" in text
    assert "contour_step" in text
    assert "contour_label_step" in text
    assert "Static_1D__b64" in text
    assert "Static_1.625D__b64" in text
    data_csv = output / "Static_radial_wake_deficit_data.csv"
    assert data_csv.exists()
    assert "radial_distance" in data_csv.read_text(encoding="utf-8")


def test_plot_radial_wake_deficit_refuses_existing_output_folder(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0],
        [0.0],
        [0.0],
        [[[3.0]]],
    )
    output = tmp_path / "figures" / "radial_001"
    output.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_radial_wake_deficit.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--rotor-y",
            "0",
            "--rotor-z",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to use existing output folder" in (result.stdout + result.stderr)
