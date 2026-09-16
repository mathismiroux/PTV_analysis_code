import h5py
import numpy as np
import pytest

from scripts.align_velocity_z import align_file


def write_file(path, z, nt):
    with h5py.File(path, 'w') as h:
        for key, values in dict(t=np.arange(nt), z=z, y=[0., 1.], x=[0., 1.]).items():
            h[key] = values
        for factor, key in enumerate('uvw', start=1):
            values = factor*(np.asarray(z)[None, :, None, None] + np.arange(nt)[:, None, None, None])
            h[key] = np.broadcast_to(values, (nt, len(z), 2, 2))
        h['filled_mask'] = np.zeros(h['u'].shape, dtype=bool)


def test_alignment_keeps_time_and_missingness_and_originals(tmp_path):
    source, reference, output = (tmp_path/name for name in ['source.nc', 'reference.nc', 'aligned.nc'])
    write_file(source, [-.5, .5, 1.5], 8)
    write_file(reference, [0., 1.], 9)
    with h5py.File(source, 'a') as h:
        h['u'][0, 0, 0, 0] = np.nan
        h['filled_mask'][1, 0, 0, 0] = True
    before = source.read_bytes()
    result = align_file(source, reference, output)
    assert result['shape'] == [8, 2, 2, 2]
    with h5py.File(output) as h:
        np.testing.assert_array_equal(h['z'][:], [0., 1.])
        np.testing.assert_array_equal(h['t'][:], np.arange(8))
        np.testing.assert_allclose(h['u'][:, :, 1, 1], np.arange(8)[:, None]+np.array([0., 1.]))
        assert all(np.isnan(h[k][0, 0, 0, 0]) for k in 'uvw')
        assert h['filled_mask'][1, 0, 0, 0]
        assert not h['filled_mask'][1, 1, 0, 0]
        assert h['valid_fraction'][0, 0, 0] == .875
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        align_file(source, reference, output)


def test_no_extrapolation(tmp_path):
    source, reference = tmp_path/'source.nc', tmp_path/'reference.nc'
    write_file(source, [0., 1.], 8)
    write_file(reference, [-1., 1.], 8)
    with pytest.raises(ValueError, match='extrapolate'):
        align_file(source, reference, tmp_path/'aligned.nc')
