import h5py
import numpy as np
import pytest

from scripts.plot_turbulence_intensity import temporal_statistics, intensity, statistics_from_mean


def test_streamed_statistics_and_masking(tmp_path):
    values = np.array([[1., np.nan, 0., 1e8], [3., -999., 0., 1e8 + 2],
                       [5., 2., 0., 1e8 + 4], [7., np.nan, 0., 1e8 + 6]])[:, None, None, :]
    with h5py.File(tmp_path / "velocity.nc", "w") as f:
        u = f.create_dataset("u", data=values)
        u.attrs["missing_value"] = -999.
        for chunk in (1, 3, 10):
            mean, rms, count = temporal_statistics(u, chunk)
            np.testing.assert_allclose(mean.ravel(), [4, 2, 0, 1e8 + 3])
            np.testing.assert_allclose(rms.ravel(), [np.sqrt(5), 0, 0, np.sqrt(5)])
            np.testing.assert_array_equal(count.ravel(), [4, 1, 4, 4])
            local = intensity(mean, rms, count, 4)
            np.testing.assert_allclose(local.ravel()[0], 25 * np.sqrt(5))
            assert np.isnan(local.ravel()[1:3]).all()
            fixed = intensity(mean, rms, count, 4, reference_velocity=2)
            np.testing.assert_allclose(fixed.ravel()[0], 50 * np.sqrt(5))
            assert fixed.ravel()[2] == 0


def test_empty_voxel(tmp_path):
    with h5py.File(tmp_path / "empty.nc", "w") as f:
        u = f.create_dataset("u", data=np.full((3, 1, 1, 1), np.nan))
        mean, rms, count = temporal_statistics(u)
    assert np.isnan(mean).all() and np.isnan(rms).all()
    assert (count == 0).all()


def test_saved_mean_and_vector_mask(tmp_path):
    with h5py.File(tmp_path / "flow.nc", "w") as flow, h5py.File(tmp_path / "mean.nc", "w") as saved:
        for key in ("x", "y", "z"):
            flow[key] = saved[key] = [0.]
        flow["t"] = saved["t"] = [0., 1., 2.]
        flow["u"] = np.array([1., 3., 99.]).reshape(3, 1, 1, 1)
        flow["v"] = np.array([0., 0., np.nan]).reshape(3, 1, 1, 1)
        flow["w"] = np.zeros((3, 1, 1, 1))
        saved["u_mean"] = np.full((1, 1, 1), 2.)
        saved["vector_count"] = np.full((1, 1, 1), 2)
        saved.attrs["zero_mask"] = "vector"
        mean, rms, count = statistics_from_mean(flow, saved, 2)
        np.testing.assert_allclose(mean, 2)
        np.testing.assert_allclose(rms, 1)
        np.testing.assert_array_equal(count, 2)
        saved["vector_count"][:] = 3
        with pytest.raises(ValueError, match="Sample counts differ"):
            statistics_from_mean(flow, saved)
