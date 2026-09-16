import csv
import json

import h5py
import numpy as np
import pytest

from ptv_flow.pod_spectra import calculate_spectra
from scripts.integrate_pod_band import integrate_file, main


def saved_spectra(tmp_path):
    path = tmp_path/'Flow_3.5D'/'pod_valid90_modes100_all'/'modal_spectra_100'/'modal_spectra.h5'
    path.parent.mkdir(parents=True)
    t = np.arange(1000)/100
    coefficients = np.column_stack([np.sin(2*np.pi*f*t) for f in (.5, 1., 2.)])
    result = calculate_spectra(t, coefficients)
    with h5py.File(path, 'w') as h:
        h['frequency'] = result['frequency']
        h['rectangular'] = result['rectangular']
        h['pod_energy_fraction'] = [.2, .3, .1]
        h.attrs['metadata_json'] = json.dumps(dict(sample_rate_hz=100,
            input_pod_metadata=dict(total_fluctuation_energy=10, spatial_coverage=.28)))
    return path


def test_known_band_and_cli(tmp_path):
    path = saved_spectra(tmp_path)
    before = path.read_bytes()
    row = integrate_file(path, 0, 1)
    assert row['band_fraction_of_total_energy'] == pytest.approx(.5)
    assert row['band_energy'] == pytest.approx(5)
    assert row['saved_pod_energy_fraction'] == pytest.approx(.6)
    assert row['last_bin_hz'] == pytest.approx(1)
    assert integrate_file(path, 0, 50)['band_fraction_of_total_energy'] == pytest.approx(.6)
    assert main([str(tmp_path), '--band-hz', '0', '1']) == 0
    with (tmp_path/'pod_energy_0_to_1Hz.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['case'] == 'Flow_3.5D'
    assert float(rows[0]['band_energy']) == pytest.approx(5)
    assert before == path.read_bytes()
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


@pytest.mark.parametrize('band', [(-1, 1), (1, 0), (0, 51), (.01, .02), (0, float('nan'))])
def test_invalid_or_unresolved_band(tmp_path, band):
    with pytest.raises(ValueError):
        integrate_file(saved_spectra(tmp_path), *band)
