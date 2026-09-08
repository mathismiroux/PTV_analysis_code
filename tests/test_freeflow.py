from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ptv_flow.freeflow import (
    FreeflowSettings, SelectedFlow, characterize_freeflow, integrate_correlation,
    normalized_correlation, pair_moments, uniform_spacing,
)
from scripts.characterize_freeflow import main


def make_flow(path, n=64):
    phase = np.arange(n)[:, None, None, None] * 2*np.pi/n
    x = np.arange(16)[None, None, None, :] * 20.
    y = np.arange(12)[None, None, :, None] * 20.
    z = np.arange(10)[None, :, None, None] * 20.
    shape = (n, 10, 12, 16)
    with h5py.File(path, "w") as f:
        for c, a in (("t", np.arange(n)*.01), ("x", x.ravel()),
                     ("y", y.ravel()), ("z", z.ravel())):
            f[c] = a
        f["u"] = np.broadcast_to(4 + .4*np.cos(phase+x*2*np.pi/120), shape)
        f["v"] = np.broadcast_to(.2*np.sin(phase+y*2*np.pi/160), shape)
        f["w"] = np.broadcast_to(.1*np.cos(phase+z*2*np.pi/200), shape)
    return path


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_fft_pair_moments_match_direct_pairs_with_gaps(axis):
    rng = np.random.default_rng(823)
    f = rng.normal(size=(11, 8, 6))
    f[rng.random(f.shape) < .3] = np.nan
    expected = []
    for k in range(f.shape[axis]):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(0, f.shape[axis]-k), slice(k, None)
        a, b = f[tuple(left)], f[tuple(right)]
        good = np.isfinite(a) & np.isfinite(b)
        a, b = a[good], b[good]
        expected.append([(a*b).sum(), (a*a).sum(), (b*b).sum(), len(a)])
    np.testing.assert_allclose(pair_moments(f, axis, f.shape[axis]-1),
                               np.array(expected).T, atol=1e-10)


def test_zero_interpolation_and_unresolved_are_distinct():
    result = integrate_correlation([0, 1, 2], [1, .5, -.5])
    assert result["cutoff"] == 1.5
    assert result["integral"] == .875
    result = integrate_correlation([0, 1, 2], [1, .5, .25])
    assert result["integral"] is None
    assert result["truncated_integral"] == 1.125
    assert result["status"] == "no_zero_crossing"
    result = integrate_correlation([0, 1, 2], [1, np.nan, -.5])
    assert result["integral"] is None
    assert result["status"] == "insufficient_pairs_before_zero"


def test_zero_variance_and_pair_support():
    rho = normalized_correlation(pair_moments(np.zeros(10), 0, 9), 1, .05)
    assert np.isnan(rho).all()
    f = np.arange(10.) - 4.5
    rho = normalized_correlation(pair_moments(f, 0, 9), 4, .5)
    assert np.isfinite(rho[:6]).all()
    assert np.isnan(rho[6:]).all()


def test_temporal_exponential_covariance_has_expected_scale():
    rng = np.random.default_rng(19)
    noise = rng.normal(size=100000)
    f = np.empty_like(noise)
    f[0] = noise[0]
    for i in range(1, len(f)):
        f[i] = .97*f[i-1] + noise[i]
    rho = normalized_correlation(pair_moments(f-f.mean(), 0, 1000), 100, .05)
    result = integrate_correlation(np.arange(1001)*.01, rho)
    # AR(1) has R(k)=a**k, whose infinite trapezoidal integral is
    # dt*(1+a)/(2*(1-a)). A finite stochastic record has sampling scatter.
    expected = .01*(1+.97)/(2*(1-.97))
    assert result["integral"] == pytest.approx(expected, rel=.15)


def test_selected_flow_units_roi_fill_mask_and_unchanged_input(tmp_path):
    source = make_flow(tmp_path / "raw.nc")
    with h5py.File(source, "a") as f:
        mask = np.zeros(f["u"].shape, dtype=np.uint8)
        mask[3, 1, 2, 4] = 1
        f["filled_mask"] = mask
    s = FreeflowSettings(length_unit="mm", velocity_unit="mm/s", time_unit="s",
                         start=2, stop=10, x_range=(60, 140), y_range=(20, 100))
    with SelectedFlow(source, s) as flow:
        assert flow.shape == (8, 10, 5, 5)
        np.testing.assert_allclose(flow.coordinate("x"), np.arange(3, 8)*.02)
        assert np.isnan(flow.read_frame(1).u[1, 1, 1])
        assert np.isnan(flow.read_frame(1).v[1, 1, 1])
        assert np.nanmax(flow.read_frame(0).u) < .005
    with h5py.File(source) as f:
        assert np.isfinite(f["u"][3, 1, 2, 4])


def test_end_to_end_known_spatial_scales_and_temporal_units(tmp_path):
    source = make_flow(tmp_path / "raw.nc")
    out = characterize_freeflow(source, tmp_path / "result", FreeflowSettings(
        length_unit="mm", time_unit="s", velocity_unit="m/s", temporal=True,
        min_pairs=10, min_valid_fraction=.8, probe_grid=1, chunk_size=7,
        rotor_diameter_m=1.2, max_temporal_lag=32))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["mean_velocity"]["u"] == pytest.approx(4.)
    assert summary["rms"]["u"] == pytest.approx(.4 / np.sqrt(2), rel=1e-6)
    assert summary["mean_tke"] == pytest.approx(.5*(.4**2+.2**2+.1**2)/2, rel=1e-6)
    for component, direction, wavelength in (("u", "x", .12), ("v", "y", .16), ("w", "z", .20)):
        row = next(r for r in summary["scales"] if r["kind"] == "spatial"
                   and r["component"] == component and r["direction"] == direction)
        assert row["status"] == "first_zero_crossing"
        # Trapezoidal quadrature on a coarse grid, compared with analytic cosine integral.
        assert row["length_m"] == pytest.approx(wavelength/(2*np.pi), rel=.10)
        assert row["length_over_D"] == pytest.approx(row["length_m"]/1.2)
    time_row = next(r for r in summary["scales"] if r["kind"] == "temporal" and r["component"] == "u")
    assert time_row["length_m"] == pytest.approx(time_row["integral"]*4.)
    assert (out / "spatial_correlations.png").stat().st_size > 1000
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        characterize_freeflow(source, out)


def test_cli_and_preflight_validation(tmp_path):
    source = make_flow(tmp_path / "raw.nc")
    assert main([str(source), "--output", str(tmp_path/"cli"), "--chunk-size", "16"]) == 0
    with pytest.raises(ValueError, match="time-unit"):
        characterize_freeflow(source, tmp_path / "bad", FreeflowSettings(temporal=True))
    assert not (tmp_path / "bad").exists()
    with h5py.File(source, "a") as f:
        f["x"][3] += 5
    with pytest.raises(ValueError, match="uniformly"):
        characterize_freeflow(source, tmp_path / "irregular")
    assert not (tmp_path / "irregular").exists()
    assert uniform_spacing(np.arange(4000, dtype=np.float32)*.0025, "t") == pytest.approx(.0025, rel=.001)


def test_steady_nonuniform_flow_has_no_turbulent_length(tmp_path):
    source = make_flow(tmp_path / "steady.nc")
    with h5py.File(source, "a") as f:
        for ci, c in enumerate(("u", "v", "w")):
            f[c][:] = np.broadcast_to(.13*(ci+1) + np.arange(16)*.0013, f[c].shape)
    out = characterize_freeflow(source, tmp_path / "steady", FreeflowSettings(
        temporal=True, time_unit="s", velocity_unit="m/s", min_pairs=10, probe_grid=1))
    result = json.loads((out / "summary.json").read_text())
    assert all(r["integral"] is None for r in result["scales"])
