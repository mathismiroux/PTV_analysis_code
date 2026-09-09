from pathlib import Path

import numpy as np

from ptv_flow.postprocess import _first_harmonic_from_phase_means
from scripts.plot_wake_pulsing import is_1p625d_case, reduce_plane, sector_area_weights
from scripts.view_wake_deficit_modulation import harmonic_fit


def test_fixed_support_removes_phase_dependent_missing_cell():
    phase = (np.arange(32) + .5) * 2 * np.pi / 32
    expected = .3 + .06 * np.cos(phase) + .08 * np.sin(phase)
    deficit = np.broadcast_to(expected[:, None, None], (32, 3, 3)).copy()
    deficit[:, 0, 0] = .9
    deficit[0, 0, 0] = np.nan
    counts = np.full_like(deficit, 10)
    counts[0, 0, 0] = 0
    dm, cy, cz, coverage = reduce_plane(deficit, counts, np.full(32, 10),
                                       np.arange(3.), np.arange(3.), [0, 2, 0, 2])
    np.testing.assert_allclose(dm, expected)
    np.testing.assert_allclose(_first_harmonic_from_phase_means(dm, phase)[3], .1)
    np.testing.assert_allclose(coverage, 15 / 16)
    np.testing.assert_allclose(cy, cy[0])
    np.testing.assert_allclose(cz, cz[0])


def test_opposite_local_harmonics_cancel_in_integrated_deficit():
    phase = np.arange(32) * 2 * np.pi / 32
    deficit = np.full((32, 2, 2), .3)
    deficit[:, :, 0] += .1 * np.cos(phase[:, None])
    deficit[:, :, 1] -= .1 * np.cos(phase[:, None])
    dm, cy, _, _ = reduce_plane(deficit, np.ones_like(deficit), np.ones(32),
                                np.arange(2.), np.arange(2.), [0, 1, 0, 1])
    np.testing.assert_allclose(dm, .3)
    assert _first_harmonic_from_phase_means(dm, phase)[3] < 1e-14
    assert np.ptp(cy) > .3


def test_no_common_support_is_nan():
    values = np.full((4, 2, 2), .3)
    counts = np.ones_like(values)
    counts[0] = 0
    dm, cy, cz, coverage = reduce_plane(values, counts, np.ones(4),
                                       np.arange(2.), np.arange(2.), [0, 1, 0, 1])
    assert np.isnan(dm).all() and np.isnan(cy).all() and np.isnan(cz).all()
    assert coverage == 0


def test_sector_area_orientation_and_wrapped_angles():
    c = np.linspace(-2, 2, 81)
    area, requested = sector_area_weights(c, c, 0, 0, 1, [-30, 30])
    np.testing.assert_allclose(area.sum(), np.pi / 6, rtol=.002)
    np.testing.assert_allclose(requested, np.pi / 6)
    assert area[:, c < -.1].sum() == 0
    np.testing.assert_allclose(area, area[::-1], atol=1e-15)
    wrapped, _ = sector_area_weights(c, c, 0, 0, 1, [330, 390])
    np.testing.assert_array_equal(area, wrapped)


def test_sector_excludes_outside_signal_and_counts_unmeasured_area():
    y = np.linspace(0, 1, 41)
    z = np.linspace(0, 1, 41)  # only half of the requested sector is measured
    area, requested = sector_area_weights(y, z, 0, 0, 1, [-30, 30])
    deficit = np.full((8, len(z), len(y)), .3)
    deficit[:, area == 0] = .9
    dm, _, _, coverage = reduce_plane(deficit, np.ones_like(deficit), np.ones(8),
                                       y, z, None, area=area, requested_area=requested)
    np.testing.assert_allclose(dm, .3)
    np.testing.assert_allclose(coverage, .5, atol=.001)


def test_viewer_harmonic_fit_uses_exported_coefficients():
    phase = np.array([0.0, np.pi / 2, np.pi])
    result = {"phase": phase, "offset": np.array([.4]),
              "a": np.array([.1]), "b": np.array([.2])}
    np.testing.assert_allclose(harmonic_fit(result, 0), [.5, .6, .3])


def test_1p625d_case_filter_uses_phase_average_folder_name():
    assert is_1p625d_case(Path("PitchLF_1.625D") / "phase_average.nc")
    assert not is_1p625d_case(Path("PitchLF_2.25D") / "phase_average.nc")
