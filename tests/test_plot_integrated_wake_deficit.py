from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np

from scripts.plot_integrated_wake_deficit import integrated_wake_deficit


def _write_mean(path, label, x_values, y_values, z_values, u_values, vector_count=None):
    path.parent.mkdir(parents=True)
    u = np.asarray(u_values, dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["label"] = label
        h5.attrs["u_inf"] = 4.0
        h5.attrs["input_shape_time_z_y_x"] = (4, *u.shape)
        h5.create_dataset("x", data=np.asarray(x_values, dtype=np.float32))
        h5.create_dataset("y", data=np.asarray(y_values, dtype=np.float32))
        h5.create_dataset("z", data=np.asarray(z_values, dtype=np.float32))
        h5.create_dataset("u_mean", data=u)
        h5.create_dataset("v_mean", data=np.ones_like(u))
        h5.create_dataset("w_mean", data=np.ones_like(u))
        if vector_count is not None:
            h5.create_dataset(
                "vector_count",
                data=np.asarray(vector_count, dtype=np.uint32),
            )


def test_integrated_wake_deficit_averages_valid_cross_plane(tmp_path):
    mean_file = tmp_path / "Static_1D" / "mean.nc"
    _write_mean(
        mean_file,
        "Static_1D",
        [0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [[[2.0], [4.0]], [[0.0], [3.0]]],
        vector_count=[[[4], [4]], [[4], [2]]],
    )

    result = integrated_wake_deficit(
        mean_file,
        axis_y=0.0,
        axis_z=0.0,
        integration_radius=None,
        invalid_samples="zero-or-nan",
        require_all_components=False,
        min_valid_fraction=0.75,
        u_inf=None,
    )

    assert result.valid_points == 2
    assert result.wake_deficit == 0.25


def test_plot_integrated_wake_deficit_writes_plot_and_csv(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D" / "mean.nc",
        "Static_1D",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 3.0),
    )
    _write_mean(
        mean_root / "SurgeLF_1D" / "mean.nc",
        "SurgeLF_1D",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 2.0),
    )
    _write_mean(
        mean_root / "SurgeLF_1D_z0" / "mean.nc",
        "SurgeLF_1D_z0",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 1.0),
    )
    output = tmp_path / "figures"

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_integrated_wake_deficit.py",
            str(mean_root),
            "--output-folder",
            str(output),
            "--rotor-y",
            "0",
            "--rotor-z",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (output / "integrated_wake_deficit.png").exists()
    csv_path = output / "integrated_wake_deficit.csv"
    text = csv_path.read_text(encoding="utf-8")
    assert "Static_1D" in text
    assert "SurgeLF_1D" in text
    assert "SurgeLF_1D_z0" not in text
