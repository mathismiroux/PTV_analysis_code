import json

import h5py
import numpy as np
import pytest

from scripts.compare_pod_pair import fit_pair, weighted_gram, run


def test_quadrature_pair_and_masked_orthogonality():
    t = np.arange(400)/100
    a = np.column_stack((2*np.cos(4*np.pi*t), 3*np.sin(4*np.pi*t)))
    _, _, explained, _, difference = fit_pair(t, a, 2.)
    np.testing.assert_allclose(explained, 1.)
    assert difference == pytest.approx(-90.)
    modes = np.zeros((3, 2, 2, 2, 2))
    modes[0, 0] = 1
    modes[1, 1] = 1
    weights = np.ones((2, 2, 2))/7
    weights[0, 0, 0] = 0
    modes[:, :, 0, 0, 0] = np.nan
    gram, _ = weighted_gram(modes, weights)
    np.testing.assert_allclose(gram, np.eye(2))


def test_comparison_and_animation(tmp_path):
    source = tmp_path/'pod.h5'
    t = np.arange(200)/100
    with h5py.File(source, 'w') as f:
        f['t'] = t
        f['temporal_coefficients'] = np.column_stack((np.cos(4*np.pi*t), np.sin(4*np.pi*t)))
        f['energy_fraction'] = [.4, .4]
        f['spatial_weight'] = np.ones((3, 3, 3))/27
        for key in 'xyz':
            f[key] = [-1., 0., 1.]
        for ic, key in enumerate('uvw'):
            modes = np.zeros((2, 3, 3, 3))
            if ic < 2:
                modes[ic] = 1
            f['mode_'+key] = modes
    summary = run(source, tmp_path/'pair', diameter=1., frames=3, z_slices=(0.,), y_slice=0.)
    assert summary['phase_second_minus_first_degrees'] == pytest.approx(-90.)
    assert (tmp_path/'pair/pair_reconstruction.gif').exists()
    assert (tmp_path/'pair/spatial_xz.png').exists()
    assert json.loads((tmp_path/'pair/summary.json').read_text())['pair_energy_fraction'] == pytest.approx(.8)
