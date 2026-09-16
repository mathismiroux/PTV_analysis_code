import pytest
import json
import h5py
import numpy as np

from scripts.compare_pod_band import compare, verify_common_support


def case(group, energy, total):
    return dict(group=group, band_energy=energy, total_fluctuation_energy=total,
                band_fraction_of_total_energy=energy/total,
                spatial_coverage=.4, saved_pod_energy_fraction=.8)


def test_baselines_and_distinct_relative_definitions():
    cases = {('Flow', 1.): case('Flow', 2., 4.),
             ('Static', 1.): case('Static', 6., 20.),
             ('Moving', 1.): case('Moving', 9., 30.)}
    rows = {r['group']:r for r in compare(cases)}
    assert rows['Static']['band_energy_ratio_to_background'] == 3
    assert rows['Static']['band_energy_minus_background'] == 4
    assert rows['Static']['band_energy_change_percent_vs_background'] == 200
    assert rows['Static']['band_fraction_of_total_energy'] == .3
    assert rows['Moving']['band_energy_ratio_to_static'] == 1.5
    cases['Flow', 1.]['band_energy'] = 0
    assert compare(cases)[0]['band_energy_ratio_to_background'] == ''
    del cases['Flow', 1.]
    with pytest.raises(ValueError, match='Missing Flow'):
        compare(cases)


def test_common_support_checks_masks_and_weights(tmp_path):
    cases = {}
    for name in ['Flow', 'Static']:
        pod = tmp_path/f'{name}_pod.h5'
        spectra = tmp_path/f'{name}_spectra.h5'
        with h5py.File(pod, 'w') as h:
            for key in 'xyz':
                h[key] = [0., 1.]
            h['support'] = np.ones((2, 2, 2), dtype=bool)
            h['spatial_weight'] = np.ones((2, 2, 2))/8
        with h5py.File(spectra, 'w') as h:
            h.attrs['metadata_json'] = json.dumps(dict(pod_file=str(pod)))
        cases[name, 1.] = dict(case=name, spectra_file=str(spectra))
    verify_common_support(cases)
    with h5py.File(tmp_path/'Static_pod.h5', 'a') as h:
        h['spatial_weight'][0, 0, 0] = .5
    with pytest.raises(ValueError, match='spatial_weight differs'):
        verify_common_support(cases)
