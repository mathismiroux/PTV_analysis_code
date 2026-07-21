from __future__ import annotations

import csv
import subprocess
import sys

import h5py
import numpy as np


def _write_mean(path, label, x_values, y_values, z_values, value):
    path.parent.mkdir(parents=True)
    shape = (len(z_values), len(y_values), len(x_values))
    with h5py.File(path, "w") as h5:
        h5.attrs["label"] = label
        h5.create_dataset("x", data=np.asarray(x_values, dtype=np.float32))
        h5.create_dataset("y", data=np.asarray(y_values, dtype=np.float32))
        h5.create_dataset("z", data=np.asarray(z_values, dtype=np.float32))
        h5.create_dataset("u_over_u_inf", data=np.full(shape, value, dtype=np.float64))
        h5.create_dataset("wake_deficit", data=np.full(shape, 1.0 - value, dtype=np.float64))
        h5.create_dataset("abs_U", data=np.full(shape, value * 4.0, dtype=np.float64))


def test_plot_mean_wake_z0_groups_distances_and_writes_plot(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0, 1.0],
        [0.0, 1.0],
        [-0.2, 0.1],
        0.8,
    )
    _write_mean(
        mean_root / "Static_1.625D__b64" / "mean.nc",
        "Static_1.625D__b64",
        [0.5, 1.5],
        [0.0, 1.0],
        [-0.1, 0.2],
        0.9,
    )
    output = tmp_path / "plots"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_mean_wake_z0.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--z",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "Static_u_over_u_inf_z0.png").exists()
    manifest = output / "Static_u_over_u_inf_z0.csv"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "Static_1D__b64" in text
    assert "Static_1.625D__b64" in text


def test_plot_mean_wake_z0_refuses_existing_output_folder(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0],
        0.8,
    )
    output = tmp_path / "plots"
    output.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_mean_wake_z0.py",
            str(mean_root),
            "--output-folder",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to use existing output folder" in (result.stdout + result.stderr)


def test_plot_mean_wake_z0_uses_one_shared_color_scale_for_all_groups(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0],
        [0.0],
        [0.0],
        0.8,
    )
    _write_mean(
        mean_root / "Surge_1D__b64" / "mean.nc",
        "Surge_1D__b64",
        [0.0],
        [0.0],
        [0.0],
        0.4,
    )
    output = tmp_path / "plots"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_mean_wake_z0.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--vmin",
            "0",
            "--vmax",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Shared color scale: vmin=0, vmax=1" in result.stdout
    limits = set()
    for manifest in (
        output / "Static_u_over_u_inf_z0.csv",
        output / "Surge_u_over_u_inf_z0.csv",
    ):
        with manifest.open(newline="", encoding="utf-8") as fh:
            row = next(csv.DictReader(fh))
        limits.add((row["color_vmin"], row["color_vmax"]))
    assert limits == {("0.0", "1.0")}
