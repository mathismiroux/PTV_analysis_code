import csv
import json

import h5py
import numpy as np
import pytest

from ptv_flow.forcing_response import voxel_budget, run_forcing_response, reduce_energy, resolve_frequency


def signals():
    times = (np.arange(256) + .5) / 64
    phase = 2*np.pi*2*times
    cycles = np.floor(2*times).astype(int)
    data = np.stack([3 + np.cos(phase) + .4 * (-1.)**cycles,
                     1 + .5*np.sin(phase), np.zeros(len(times))])[:, :, None]
    return times, data


def test_energy_partition_known_signal_and_heldout_prediction():
    times, data = signals()
    valid = np.ones((len(times), 1), dtype=bool)
    result = voxel_budget(data, valid, times, 2., phase_bins=32)
    np.testing.assert_allclose(result['total_energy'], .3925, atol=1e-12)
    np.testing.assert_allclose(result['forcing_energy'], .3125, atol=1e-12)
    np.testing.assert_allclose(result['forcing_uniform_cycle_energy'], .3125, atol=1e-12)
    np.testing.assert_allclose(result['phase_locked_energy'], .3125, atol=1e-12)
    np.testing.assert_allclose(result['residual_energy'], .08, atol=1e-12)
    np.testing.assert_allclose(result['cv_phase_prediction_error'], .08*(8/7)**2, atol=1e-12)
    assert result['cv_eligible'].all()


def test_missing_samples_not_zero_filled_and_regression_matches_lstsq():
    times, data = signals()
    data[0, :, 0] -= .4 * (-1.)**np.floor(2*times).astype(int)
    valid = np.ones((len(times), 1), dtype=bool)
    valid[::11] = False
    data[:, ~valid[:, 0], 0] = np.nan
    result = voxel_budget(data, valid, times, 2., phase_bins=32)
    measured = data[:, valid[:, 0], 0]
    expected = .5*np.sum(np.var(measured, axis=1))
    assert result['total_energy'][0] == pytest.approx(expected)
    assert result['forcing_energy'][0] == pytest.approx(expected)
    assert result['forcing_uniform_cycle_energy'][0] == pytest.approx(.3125)
    assert result['phase_locked_energy'][0]+result['residual_energy'][0] == pytest.approx(expected)


def test_cross_validation_matches_brute_force_with_gaps():
    times, data = signals()
    rng = np.random.default_rng(6)
    data += rng.normal(scale=.2, size=data.shape)
    valid = rng.random((len(times), 1)) > .07
    result = voxel_budget(data, valid, times, 2., phase_bins=32)
    cycle = np.floor(2*times).astype(int)
    bins = np.floor((2*times % 1)*32).astype(int)
    errors, baseline = [], []
    for c in np.unique(cycle):
        train = (cycle != c) & valid[:, 0]
        test = (cycle == c) & valid[:, 0]
        for it in np.flatnonzero(test):
            pred = data[:, train & (bins == bins[it]), 0].mean(axis=1)
            errors.append(np.sum((data[:, it, 0]-pred)**2)/2)
            baseline.append(np.sum((data[:, it, 0]-data[:, train, 0].mean(axis=1))**2)/2)
    assert result['cv_phase_prediction_error'][0] == pytest.approx(np.mean(errors))
    assert result['cv_mean_prediction_error'][0] == pytest.approx(np.mean(baseline))


def make_file(tmp_path, case='SurgeLF_3.5D'):
    folder = tmp_path / case
    folder.mkdir()
    source = folder / 'interpolated_velocity.nc'
    times, data = signals()
    with h5py.File(source, 'w') as f:
        f['t'] = times
        f['x'] = [0., .1, .2, .3]
        f['y'] = [-1., 0., 1.]
        f['z'] = [-1., 0., 1.]
        for ic, key in enumerate('uvw'):
            f[key] = np.broadcast_to(data[ic, :, 0, None, None, None], (len(times), 3, 3, 4))
    return source


def test_file_profiles_coverage_slabs_and_protected_output(tmp_path):
    source = make_file(tmp_path)
    output = tmp_path / 'energy'
    summary = run_forcing_response(source, output, rotor_diameter=1., y_bounds_d=(-2., 2.),
                                   z_bounds_d=(-1., 1.), spatial_chunk=2)
    assert summary['frequency_hz'] == 2.
    assert summary['volume_summary']['coverage_fraction'] == pytest.approx(.5)
    assert summary['volume_summary']['total_energy'] == pytest.approx(.3925)
    with (output / 'downstream_slabs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(float(row['coverage_fraction']) == pytest.approx(.5) for row in rows)
    assert (output / 'energy_profiles.png').exists()
    with pytest.raises(FileExistsError):
        run_forcing_response(source, output)


def test_static_has_no_invented_forcing(tmp_path):
    source = make_file(tmp_path, 'Static_3.5D')
    assert resolve_frequency(source)[0] is None
    times, data = signals()
    result = voxel_budget(data, np.ones((len(times), 1), bool), times, None)
    assert np.isfinite(result['total_energy']).all()
    assert np.isnan(result['forcing_energy']).all()


def test_batch_energy_only_reuses_pod_and_combines_profiles(tmp_path):
    from scripts.pod_folder import main
    source = make_file(tmp_path)
    pod_output = source.parent / 'pod_valid90'
    pod_output.mkdir()
    marker = pod_output / 'pod.h5'
    marker.write_bytes(b'existing POD must remain unchanged')
    args = [str(tmp_path), '--pattern', '*/interpolated_velocity.nc', '--output-subfolder',
            'pod_valid90', '--analysis', 'energy', '--rotor-diameter', '1']
    assert main(args) == 0
    assert marker.read_bytes() == b'existing POD must remain unchanged'
    manifest = json.loads((tmp_path/'pod_valid90_energy_manifest.json').read_text())
    assert manifest[0]['energy_status'] == 'complete'
    assert (pod_output/'forcing_response/downstream_slabs.csv').exists()
    assert (tmp_path/'pod_valid90_energy_manifest_downstream_slabs.csv').exists()
    with pytest.raises(SystemExit):
        main(args)


def test_batch_default_runs_both_and_dry_run_does_not_analyze(tmp_path):
    from scripts.pod_folder import main
    source = make_file(tmp_path)
    base = [str(tmp_path), '--pattern', '*/interpolated_velocity.nc', '--rotor-diameter', '1', '--modes', '2']
    assert main(base + ['--output-subfolder', 'planned', '--dry-run']) == 0
    assert not (source.parent/'planned').exists()
    assert main(base + ['--output-subfolder', 'combined']) == 0
    output = source.parent/'combined'
    assert (output/'pod.h5').exists()
    assert (output/'forcing_response/energy_maps.h5').exists()
    manifest = json.loads((tmp_path/'combined_manifest.json').read_text())
    assert manifest[0]['pod_status'] == manifest[0]['energy_status'] == 'complete'


def test_ratio_of_averaged_energies_not_average_of_ratios():
    from ptv_flow.forcing_response import ENERGY_FIELDS
    maps = {key: np.array([1., 1.]) for key in ENERGY_FIELDS}
    maps.update(support=np.ones(2, bool), valid_fraction=np.ones(2),
                total_energy=np.array([1., 9.]), forcing_energy=np.array([1., 0.]))
    row = reduce_energy(maps, np.ones(2), 2.)
    assert row['forcing_fraction'] == pytest.approx(.1)


def test_missing_whole_phase_bin_is_excluded():
    times, data = signals()
    bins = np.floor((2*times % 1)*32).astype(int)
    valid = (bins != 0)[:, None]
    assert valid.mean() > .9
    result = voxel_budget(data, valid, times, 2., phase_bins=32)
    assert not result['eligible'].any()
