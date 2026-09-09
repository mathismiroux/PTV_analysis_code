from pathlib import Path
import csv

import h5py
import numpy as np
import pytest

from ptv_flow.global_pod import (discover_cases, exact_pod, inspect_sources,
                                 interpolation_map, interpolate_values, run_global_pod)


def phase_file(root, distance, x, offset=0.):
    path = root / f'SurgeLF_{distance}D' / 'phase_average_2Hz.nc'
    path.parent.mkdir()
    phase = (np.arange(16) + .5) * 2 * np.pi / 16
    with h5py.File(path, 'w') as f:
        f.attrs.update(frequency_hz=2., phase_source='frequency', phase_offset=offset,
                       invalid_samples='nan', zero_mask='vector')
        f['phase'] = phase
        f['phase_sample_count'] = np.full(16, 100)
        for key, values in dict(z=[0., 1.], y=[0., 1.], x=x).items():
            f[key] = values
        for key, signal in zip('uvw', [3 + np.cos(phase), 1 + .5 * np.sin(phase), np.zeros(16)]):
            f[key + '_phase_mean'] = np.broadcast_to(signal[:, None, None, None], (16, 2, 2, len(x)))
            f[key + '_phase_count'] = np.full((16, 2, 2, len(x)), 95)
    return path


def test_joint_pod_overlap_energy_and_phase(tmp_path):
    a = phase_file(tmp_path, 1, [0., 1., 2.])
    b = phase_file(tmp_path, 2, [1., 2., 3.])
    out = tmp_path / 'global'
    meta = run_global_pod([a, b], out, modes=4, spacing=1., rotor_diameter=1.)
    assert meta['retained_voxels'] == 16
    assert meta['overlap_voxels'] == 8
    assert meta['total_coherent_energy'] == pytest.approx(.3125)
    assert meta['fundamental_energy_fraction'] == pytest.approx(1.)
    assert meta['orthogonality_error'] < 1e-12
    with h5py.File(out / 'global_pod.h5') as f:
        np.testing.assert_allclose(f['energy_fraction'][:], [.8, .2], atol=1e-12)
        np.testing.assert_allclose(f['source_mode_energy_share'][:].sum(0), 1.)
        np.testing.assert_allclose(f['u_fundamental_cos'][:], 1., atol=1e-6)
        for key in 'uvw':
            reconstruction = f['cycle_mean_' + key][:] + np.einsum('pm,mzyx->pzyx', f['phase_coefficients'][:], f['mode_' + key][:])
            np.testing.assert_allclose(reconstruction, f['phase_mean_' + key][:], atol=2e-6)
    assert (out / 'global_mode_01_z0.png').exists()
    with (out / 'overlap_diagnostics.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row['normalized_coherent_rms_difference']) < 1e-12
    assert float(row['fundamental_phase_difference_degrees']) == pytest.approx(0., abs=1e-12)
    assert float(row['fundamental_complex_correlation']) == pytest.approx(1.)


def test_interpolation_affine_and_missing_corner():
    coords = {key: np.array([0., 1.]) for key in ('z', 'y', 'x')}
    target = {key: np.array([-.5, .5, 1.5]) for key in coords}
    z, y, x = np.meshgrid(*coords.values(), indexing='ij')
    data = (2*z + 3*y + x)[None, None]
    support = np.ones((2, 2, 2), dtype=bool)
    index, corners = interpolation_map(coords, target, support)
    assert index.tolist() == [13]
    np.testing.assert_allclose(interpolate_values(data, corners), [[[3.]]])
    support[0, 0, 0] = False
    index, _ = interpolation_map(coords, target, support)
    assert len(index) == 0


def test_reject_mismatched_phase_reference_and_duplicates(tmp_path):
    a = phase_file(tmp_path, 1, [0., 1.])
    b = phase_file(tmp_path, 2, [1., 2.], offset=.3)
    with pytest.raises(ValueError, match='mismatch'):
        inspect_sources([a, b])
    with pytest.raises(ValueError, match='Duplicate'):
        inspect_sources([a, a])
    assert len(discover_cases(tmp_path)['SurgeLF']) == 2
    (b.parent / 'phase_average_other.nc').write_bytes(b.read_bytes())
    with pytest.raises(ValueError, match='Multiple phase files'):
        discover_cases(tmp_path)
    assert discover_cases(tmp_path, selected_case='PitchLF') == {}


def test_no_fill_of_unmeasured_gaps_or_low_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr('ptv_flow.global_pod.plot_global_pod', lambda output: None)
    a = phase_file(tmp_path, 1, [0., 1.])
    b = phase_file(tmp_path, 2, [4., 5.])
    with h5py.File(a, 'a') as f:
        for key in 'uvw':
            f[key + '_phase_count'][:, 0, 0, 0] = 89
    out = tmp_path / 'global'
    run_global_pod([a, b], out, spacing=1.)
    with h5py.File(out / 'global_pod.h5') as f:
        assert not f['support'][0, 0, 0]
        assert not f['measured_footprint'][:, :, 2:4].any()
        assert np.isnan(f['mode_u'][:, :, :, 2:4]).all()


def test_full_phase_spectrum_and_weighted_orthogonality():
    rng = np.random.default_rng(3)
    state = rng.normal(size=(3, 8, 12))
    weights = np.arange(1., 13.); weights /= weights.sum()
    result = exact_pod(state, weights, modes=2)
    assert len(result['spectrum']) == 7
    assert result['spectrum'].sum() == pytest.approx(result['total'])
    assert result['eigenvalue'].sum() < result['total']
    assert result['orthogonality_error'] < 1e-12
