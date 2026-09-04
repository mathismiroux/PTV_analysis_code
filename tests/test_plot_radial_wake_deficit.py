from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np

from scripts.plot_radial_wake_deficit import (
    discover_radial_mean_files,
    radial_average_wake_deficit,
)


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
        min_valid_fraction=0.0,
        u_inf=None,
    )

    assert volume.counts[0, 0] == 1
    assert volume.wake_deficit[0, 0] == 0.5


def test_radial_wake_deficit_applies_min_valid_fraction(tmp_path):
    mean_file = tmp_path / "Static_1D" / "mean.nc"
    _write_mean(
        mean_file,
        "Static_1D",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[2.0], [4.0]]],
        vector_count=[[[2], [4]]],
    )

    volume = radial_average_wake_deficit(
        mean_file,
        axis_y=0.0,
        axis_z=0.0,
        radial_bin_width=2.0,
        invalid_samples="nan",
        require_all_components=False,
        min_valid_fraction=0.75,
        u_inf=None,
    )

    assert volume.counts[0, 0] == 1
    assert volume.wake_deficit[0, 0] == 0.0


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


def test_discover_radial_mean_files_adds_same_case_sibling_distances(tmp_path):
    mean_root = tmp_path / "means"
    static_1d = mean_root / "Static_1D" / "mean.nc"
    static_2d = mean_root / "Static_2D" / "mean.nc"
    surge_1d = mean_root / "SurgeLF_1D" / "mean.nc"
    static_z0 = mean_root / "Static_2D_z0" / "mean.nc"
    _write_mean(
        static_1d,
        "Static_1D",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[3.0], [3.0]]],
    )
    _write_mean(
        static_2d,
        "Static_2D",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[2.0], [2.0]]],
    )
    _write_mean(
        surge_1d,
        "SurgeLF_1D",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[1.0], [1.0]]],
    )
    _write_mean(
        static_z0,
        "Static_2D_z0",
        [0.0],
        [0.0, 1.0],
        [0.0],
        [[[4.0], [4.0]]],
    )

    paths = discover_radial_mean_files(
        [mean_root / "Static_1D"],
        include_sibling_distances=True,
    )

    assert paths == [static_1d, static_2d]


def test_discover_radial_mean_files_can_include_z0_distances(tmp_path):
    mean_root = tmp_path / "means"
    surge_475d = mean_root / "SurgeLF_4.75D" / "mean.nc"
    surge_475d_z0 = mean_root / "SurgeLF_4.75D_z0" / "mean.nc"
    for path in (surge_475d, surge_475d_z0):
        _write_mean(
            path,
            path.parent.name,
            [0.0],
            [0.0, 1.0],
            [0.0],
            [[[3.0], [3.0]]],
        )

    default_paths = discover_radial_mean_files(
        [mean_root],
        include_sibling_distances=True,
    )
    included_paths = discover_radial_mean_files(
        [mean_root],
        include_sibling_distances=True,
        exclude_patterns=(),
    )

    assert default_paths == [surge_475d]
    assert included_paths == [surge_475d, surge_475d_z0]


def test_plot_radial_wake_deficit_uses_suffix_in_existing_output_folder(tmp_path):
    mean_root = tmp_path / "means"
    _write_mean(
        mean_root / "Static_1D__b64" / "mean.nc",
        "Static_1D__b64",
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        np.full((2, 2, 2), 3.0),
    )
    output = tmp_path / "figures" / "radial_001"
    output.mkdir(parents=True)
    (output / "Static_radial_wake_deficit.png").write_text(
        "existing plot", encoding="utf-8"
    )

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
            "--radial-bin-width",
            "0.5",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert (output / "Static_radial_wake_deficit.png").read_text(
        encoding="utf-8"
    ) == "existing plot"
    assert (output / "Static_radial_wake_deficit_001.png").exists()
    assert (output / "Static_radial_wake_deficit_001.csv").exists()
    assert (output / "Static_radial_wake_deficit_001_data.csv").exists()
