import json
import csv

import h5py
import numpy as np
import pytest

from ptv_flow.pod_spectra import calculate_spectra, analyze_pod_file
from scripts.pod_spectra_folder import main


def fixture_pod(tmp_path):
    path = tmp_path/'SurgeLF_3.5D'/'pod_valid90'/'pod.h5'
    path.parent.mkdir(parents=True)
    t = np.arange(1000)/100
    a = np.stack([2*np.sin(2*np.pi*2*t), np.cos(2*np.pi*5*t)], axis=1)
    with h5py.File(path, 'w') as f:
        f['t'] = t
        f['temporal_coefficients'] = a
        f['energy_fraction'] = [.6, .15]
        f.attrs['metadata_json'] = json.dumps(dict(source=str(path.parent.parent/'interpolated_velocity.nc')))
    return path, t, a


def test_parseval_and_known_peak(tmp_path):
    _, t, a = fixture_pod(tmp_path)
    result = calculate_spectra(t, a)
    df = result['frequency'][1]
    np.testing.assert_allclose(result['rectangular'].sum(axis=0)*df, [2., .5], atol=1e-12)
    peaks = result['frequency'][np.argmax(result['hann'], axis=0)]
    np.testing.assert_allclose(peaks, [2., 5.])
    assert result['welch_segments'] == 4


def test_odd_length_and_irregular_sampling():
    t = np.arange(999)/100
    a = np.sin(2*np.pi*3*t)[:, None]
    result = calculate_spectra(t, a)
    assert result['rectangular'].sum()*(100/999) == pytest.approx(a.var())
    t[100] += .002
    with pytest.raises(ValueError, match='Irregular'):
        calculate_spectra(t, a)


def test_batch_preview_and_outputs(tmp_path):
    source, _, _ = fixture_pod(tmp_path)
    before = source.read_bytes()
    assert main([str(tmp_path), '--dry-run']) == 0
    assert not (source.parent/'modal_spectra').exists()
    assert not (tmp_path/'modal_spectra_batch_manifest.json').exists()
    assert main([str(tmp_path)]) == 0
    assert source.read_bytes() == before
    output = source.parent/'modal_spectra'
    assert (output/'mode_frequency_heatmap.png').exists()
    with (output/'forcing_bands.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]['within_mode_energy_fraction']) == pytest.approx(1.)
    assert float(rows[0]['fraction_of_total_pod_energy']) == pytest.approx(.6)
    assert float(rows[1]['within_mode_energy_fraction']) == pytest.approx(0., abs=1e-12)
    with pytest.raises(SystemExit):
        main([str(tmp_path)])
