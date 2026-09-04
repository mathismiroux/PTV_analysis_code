from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from ptv_flow.postprocess import spatio_temporal_interpolate_velocity
from ptv_flow.reader import FlowDataset
from scripts.prepare_mean_wake_products import build_parser, case_id_for_source


def test_prepare_mean_wake_products_defaults(tmp_path):
    args = build_parser().parse_args(
        [str(tmp_path), "--output-root", str(tmp_path / "postprocessed")]
    )

    assert args.pattern == "*/interpolated_velocity.nc"
    assert args.invalid_samples == "nan"
    assert args.zero_mask == "vector"
    assert args.chunk_size == 50
    assert args.u_inf == 4.0
    assert args.store_counts


def test_mean_case_id_for_nested_interpolated_source(tmp_path):
    input_root = tmp_path / "interpolated"
    source = input_root / "case_a" / "interpolated_velocity.nc"

    assert case_id_for_source(input_root, source) == "case_a"


def test_prepare_mean_wake_products_dry_run(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "postprocessed"

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
    assert (output_root / "mean_wake_dry_run_manifest.csv").exists()
    assert not (output_root / "mean_wake_manifest.csv").exists()
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


def test_prepare_mean_wake_products_refuses_existing_manifest(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "postprocessed"
    output_root.mkdir(parents=True)
    (output_root / "mean_wake_manifest.csv").write_text("old\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
            str(input_dir),
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Refusing to overwrite existing mean-wake manifest" in (
        result.stdout + result.stderr
    )


def test_prepare_mean_wake_products_does_not_overwrite_existing_mean(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "postprocessed"
    case_output_dir = output_root / "case_a"
    case_output_dir.mkdir(parents=True)
    (case_output_dir / "mean.nc").write_bytes(b"old")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
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
    assert (case_output_dir / "mean.nc").read_bytes() == b"old"


def test_prepare_mean_wake_products_creates_mean_products(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "postprocessed"

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
    assert (output_root / "mean_wake_manifest.csv").exists()
    assert (output_root / "mean_wake_manifest.json").exists()
    with h5py.File(output, "r") as h5:
        assert h5.attrs["script_name"] == "prepare_mean_wake_products.py"
        assert h5.attrs["invalid_samples"] == "zero-or-nan"
        assert h5.attrs["chunk_size"] == 2
        assert h5.attrs["min_valid_fraction"] == 0.25
        assert h5.attrs["u_inf"] == 4.0
        assert h5.attrs["stores_counts"]
        assert h5.attrs["count_storage"] == "vector"
        assert "u_mean" in h5
        assert "v_mean" in h5
        assert "w_mean" in h5
        assert "vector_count" in h5
        assert "u_count" not in h5
        assert "v_count" not in h5
        assert "w_count" not in h5
        assert "abs_U" in h5
        assert "speed_from_mean" in h5
        assert "u_over_u_inf" in h5
        assert "wake_deficit" in h5
        assert "command_line" in h5.attrs


def test_prepare_mean_wake_products_can_skip_counts(tiny_flow_path, tmp_path):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    shutil.copy2(tiny_flow_path, case_input_dir / "interpolated_velocity.nc")
    output_root = tmp_path / "outputs" / "postprocessed"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--chunk-size",
            "2",
            "--no-store-counts",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = output_root / "case_a" / "mean.nc"
    with h5py.File(output, "r") as h5:
        assert not h5.attrs["stores_counts"]
        assert h5.attrs["count_storage"] == "none"
        assert "vector_count" not in h5
        assert "u_count" not in h5
        assert "v_count" not in h5
        assert "w_count" not in h5


def test_prepare_mean_wake_products_copies_interpolated_filled_mask(
    tiny_flow_path, tmp_path
):
    input_dir = tmp_path / "inputs"
    case_input_dir = input_dir / "case_a"
    case_input_dir.mkdir(parents=True)
    interpolated_source = case_input_dir / "interpolated_velocity.nc"
    output_root = tmp_path / "outputs" / "postprocessed"

    with FlowDataset(tiny_flow_path) as flow:
        spatio_temporal_interpolate_velocity(
            flow,
            output=interpolated_source,
            axes=("t",),
            store_component_filled_masks=False,
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_mean_wake_products.py",
            str(input_dir),
            "--output-root",
            str(output_root),
            "--chunk-size",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = output_root / "case_a" / "mean.nc"
    with h5py.File(interpolated_source, "r") as src, h5py.File(output, "r") as out:
        assert out.attrs["stores_filled_mask"]
        assert "filled_mask" in out
        assert "t" in out
        assert "vector_count" in out
        assert "u_count" not in out
        assert out["filled_mask"].shape == src["filled_mask"].shape
        np.testing.assert_array_equal(out["filled_mask"][:], src["filled_mask"][:])
        np.testing.assert_array_equal(out["t"][:], src["t"][:])
