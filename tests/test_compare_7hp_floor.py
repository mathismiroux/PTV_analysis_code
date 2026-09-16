from pathlib import Path

import numpy as np
import pytest

from scripts.compare_7hp_floor import component_profile, differences, load_samples, profile


def test_continuation_filtering_and_matlab_statistics(tmp_path: Path):
    first, continued = tmp_path / "first.txt", tmp_path / "continued.txt"
    np.savetxt(first, [[100, 100, 0, -2], [100, 100, 0, -4],
                       [100, 103, 0, -99], [100, 100, 0, np.nan]])
    np.savetxt(continued, [[100, 100, 0, 3], [200, 200, 0, -4]])
    samples, audit = load_samples([first, continued])
    assert audit["off_grid_rows"] == 1
    assert audit["nonfinite_rows"] == 1
    assert audit["retained_rows"] == 4
    result = profile(samples, 4, 1.2, 100)
    # abs(mean(u)), not mean(abs(u)); MATLAB std uses N-1.
    assert result[200]["U_m_s"] == pytest.approx(1)
    assert result[200]["std_u_m_s"] == pytest.approx(np.sqrt(13))
    assert result[200]["Iu_percent"] == pytest.approx(100 * np.sqrt(13))
    assert result[200]["z_over_D"] == pytest.approx(1 / 12)


def test_comparison_matches_heights_and_difference_direction():
    floor = profile({0: np.array([-3., -3.]), 100: np.array([-8., -8.])}, 4, 1.2, 100)
    no_floor = profile({0: np.array([-4., -4.]), 200: np.array([-9., -9.])}, 4, 1.2, 100)
    rows = differences(floor, no_floor)
    assert len(rows) == 1
    assert rows[0]["height_mm"] == 0
    assert rows[0]["delta_U_m_s"] == -1
    assert rows[0]["delta_U_percent_of_no_floor"] == -25
    with pytest.raises(ValueError, match="no common"):
        differences({100: floor[100]}, no_floor)


def test_transverse_columns_preserve_sign_and_zero_mean(tmp_path):
    path = tmp_path / "recording.txt"
    np.savetxt(path, [[100, 100, 0, -4, -2, 3], [100, 100, 0, -4, 2, 5]])
    fifth, _ = load_samples([path], column=5)
    sixth, _ = load_samples([path], column=6)
    reversed_sixth, _ = load_samples([path], column=6, sign=-1)
    np.testing.assert_array_equal(fifth[200], [-2, 2])
    np.testing.assert_array_equal(sixth[200], [3, 5])
    p = component_profile(fifth, 4, 1.2, 100)[200]
    assert p["mean_m_s"] == 0
    assert p["std_m_s"] == pytest.approx(np.sqrt(8))
    assert p["std_over_Uref"] == pytest.approx(np.sqrt(8) / 4)
    assert component_profile(reversed_sixth, 4, 1.2, 100)[200]["mean_m_s"] == -4
