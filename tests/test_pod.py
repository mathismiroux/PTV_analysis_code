import h5py
import numpy as np
import pytest

from ptv_flow.pod import decompose, run_pod


def test_known_rank_two_energy_and_reconstruction():
    rng = np.random.default_rng(4)
    spatial = np.linalg.qr(rng.normal(size=(24, 2)))[0]
    temporal = rng.normal(size=(40, 2))
    temporal -= temporal.mean(axis=0)
    temporal = np.linalg.qr(temporal)[0]
    matrix = spatial @ np.diag([6., 2.]) @ temporal.T
    ev, vectors, total, residual = decompose(matrix, modes=2)
    np.testing.assert_allclose(ev / total, [.9, .1], atol=1e-12)
    np.testing.assert_allclose(matrix @ vectors @ vectors.T, matrix, atol=1e-12)
    assert residual.max() < 1e-12


def test_volume_fixed_support_and_energy(tmp_path):
    source = tmp_path / 'velocity.nc'
    t = np.arange(8, dtype=float)
    signal = np.sin(2 * np.pi * t / 8)
    with h5py.File(source, 'w') as data:
        for key, value in dict(t=t, z=[0., 1.], y=[0., 1.], x=[0., 1.]).items():
            data[key] = value
        for key, multiplier in zip('uvw', [1., 2., 0.]):
            a = np.broadcast_to((3 + multiplier * signal)[:, None, None, None], (8, 2, 2, 2)).copy()
            a[0, 0, 0, 0] = np.nan
            data[key] = a
    with h5py.File(tmp_path / 'mean.nc', 'w') as saved:
        with h5py.File(source, 'r') as raw:
            for key in ('t', 'z', 'y', 'x'):
                saved[key] = raw[key][:]
            for key in 'uvw':
                saved[key + '_mean'] = np.nanmean(raw[key][:], axis=0)
        saved.attrs['source_file'] = str(source)
    result = run_pod(source, tmp_path / 'pod', modes=2)
    assert result['mean_file'] == str(tmp_path / 'mean.nc')
    assert result['retained_voxels'] == 7
    assert result['total_fluctuation_energy'] == pytest.approx(1.25)
    with h5py.File(tmp_path / 'pod/pod.h5') as data:
        np.testing.assert_allclose(data['energy_fraction'][:], [1.], atol=1e-12)
        modes = np.stack([data['mode_' + k][:] for k in 'uvw'])
        weights = data['spatial_weight'][:]
        assert np.nansum(modes ** 2 * weights) == pytest.approx(1.)
        assert np.isnan(data['mode_u'][0, 0, 0, 0])
    with h5py.File(tmp_path / 'mean.nc', 'a') as saved:
        saved['x'][0] = 99
    with pytest.raises(ValueError, match='coordinates'):
        run_pod(source, tmp_path / 'mismatch')


def test_constant_data_rejected():
    with pytest.raises(ValueError, match='nonzero fluctuation'):
        decompose(np.zeros((12, 8)), modes=2)
