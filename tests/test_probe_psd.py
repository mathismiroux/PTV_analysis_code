import numpy as np
import pytest

from ptv_flow.probe_psd import load_runs, main, welch_runs


@pytest.mark.parametrize("length", [1000, 1001])
def test_density_normalization_and_peak(length):
    fs = float(length)
    t = np.arange(4*length)/fs
    values = 7 + 2*np.sin(2*np.pi*50*t)
    f, psd, count, unused = welch_runs([values], fs, length, .5)
    assert f[np.argmax(psd)] == pytest.approx(50)
    assert psd.sum() * fs/length == pytest.approx(2, rel=1e-5)
    assert count == (7 if length % 2 == 0 else 6)
    assert unused >= 0


def test_runs_do_not_bridge_gaps_or_files(tmp_path):
    data = np.tile([100., 100., 0., -4., 1., 2.], (30, 1))
    data[10, 5] = np.nan
    data[20, 1] = 103
    path = tmp_path/'rec.txt'
    np.savetxt(path, data)
    runs, audit = load_runs([path, path], 6)
    assert [len(x) for x in runs[200]] == [10, 9, 9, 10, 9, 9]
    assert audit[0]['excluded_rows'] == 2
    with pytest.raises(ValueError, match='No contiguous'):
        welch_runs(runs[200], 100, 16, .5)


def test_command_comparison_outputs(tmp_path):
    data = np.tile([100., 100., 0., -4., 1., 2.], (256, 1))
    data[:, 5] += np.sin(2*np.pi*np.arange(256)/16)
    for rec in (1020, 1028):
        np.savetxt(tmp_path/f'post_rec{rec}.txt', data)
    output = tmp_path/'out'
    main(['1020', '--compare', '1028', '--component', '6', '--fs', '100',
          '--nperseg', '128', '--data-dir', str(tmp_path), '--output', str(output)])
    assert (output/'psd.png').is_file()
    assert (output/'psd.pdf').is_file()
    assert (output/'manifest.json').is_file()
    assert len((output/'summary.csv').read_text().splitlines()) == 3
