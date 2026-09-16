import csv

import h5py
import numpy as np
import pytest

from scripts.batch_turbulence_intensity import main


@pytest.mark.parametrize("alongside", [False, True])
def test_batch_continues_after_failure_and_uses_adjacent_mean(tmp_path, alongside):
    source = tmp_path / "inputs"
    broken = source / "A_broken"
    broken.mkdir(parents=True)
    (broken / "interpolated_velocity.nc").write_text("invalid")
    case = source / "B_valid"
    case.mkdir()
    with h5py.File(case / "interpolated_velocity.nc", "w") as flow, h5py.File(case / "mean.nc", "w") as mean:
        for key in ("x", "y", "z"):
            flow[key] = mean[key] = [0., 1.]
        flow["t"] = mean["t"] = [0., 1.]
        flow["u"] = np.stack([np.ones((2, 2, 2)), np.full((2, 2, 2), 3.)])
        mean["u_mean"] = np.full((2, 2, 2), 2.)
        mean["u_count"] = np.full((2, 2, 2), 2)
    output = tmp_path / "results"
    arguments = [str(source)] if alongside else [str(source), "--output-root", str(output)]
    assert main(arguments) == 1
    summary = next(source.glob("ti_batch_manifest_*.csv")) if alongside else output / "batch_manifest.csv"
    with summary.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["status"] for row in rows] == ["failed", "processed"]
    assert rows[1]["mean_source"] == str(case / "mean.nc")
    destination = case if alongside else output / "B_valid"
    assert (destination / "turbulence_intensity.png").is_file()
    with np.load(destination / "turbulence_intensity.npz") as data:
        np.testing.assert_allclose(data["ti_percent"], 50.)
    if alongside:
        assert (case / "turbulence_intensity_manifest.json").is_file()
        before = (case / "turbulence_intensity.npz").read_bytes()
        assert main(arguments) == 1
        assert (case / "turbulence_intensity.npz").read_bytes() == before


def test_recursive_dry_run_does_not_write(tmp_path, capsys):
    case = tmp_path / "inputs" / "group" / "Flow_1D"
    case.mkdir(parents=True)
    (case / "interpolated_velocity.nc").touch()
    output = tmp_path / "results"
    assert main([str(tmp_path / "inputs"), "--output-root", str(output),
                 "--recursive", "--case-pattern", "Flow_*", "--dry-run"]) == 0
    assert not output.exists()
    assert "1 cases" in capsys.readouterr().out
