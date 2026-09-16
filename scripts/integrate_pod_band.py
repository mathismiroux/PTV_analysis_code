"""Integrate a frequency band from saved POD spectra into a case-comparison CSV."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def integrate_file(path, lower_hz, upper_hz):
    path = Path(path)
    if not np.isfinite([lower_hz, upper_hz]).all() or not 0 <= lower_hz < upper_hz:
        raise ValueError('Require finite band limits with 0 <= lower < upper')
    with h5py.File(path, 'r') as h:
        frequency = h['frequency'][:].astype(float)
        psd = h['rectangular'][:].astype(float)
        weights = h['pod_energy_fraction'][:].astype(float)
        metadata = json.loads(h.attrs['metadata_json'])
    if (frequency.ndim != 1 or len(frequency) < 2
            or not np.isfinite(frequency).all() or frequency[0] != 0):
        raise ValueError('Require a one-sided frequency grid starting at zero')
    df = frequency[1] - frequency[0]
    if df <= 0 or not np.allclose(np.diff(frequency), df, rtol=1e-6, atol=df*1e-8):
        raise ValueError('Require uniformly spaced, increasing frequencies')
    if (weights.ndim != 1 or not weights.size or psd.shape != (len(frequency), len(weights))
            or not np.isfinite(psd).all() or np.any(psd < 0)
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or weights.sum() > 1 + 1e-6):
        raise ValueError('Invalid spectra or POD energy fractions')
    fs = float(metadata['sample_rate_hz'])
    if not np.isfinite(fs) or fs <= 0 or upper_hz > fs/2 + df*1e-5:
        raise ValueError('Band exceeds the Nyquist frequency or sample rate is invalid')
    selected = (frequency >= lower_hz-df*1e-5) & (frequency <= upper_hz+df*1e-5)
    if not selected.any():
        raise ValueError('No Fourier bins inside the requested band')
    variance = psd.sum(axis=0)*df
    if np.any((variance == 0) & (weights > 0)):
        raise ValueError('A mode has positive POD energy but zero spectral variance')
    band_variance = psd[selected].sum(axis=0)*df
    share = np.divide(band_variance, variance, out=np.zeros_like(variance), where=variance > 0)
    fraction = float(np.sum(weights*share))
    pod = metadata['input_pod_metadata']
    total = float(pod['total_fluctuation_energy'])
    if not np.isfinite(total) or total < 0:
        raise ValueError('Invalid total fluctuation energy in POD metadata')
    case = Path(pod['source']).parent.name if pod.get('source') else path.parents[2].name
    return dict(case=case, modes=len(weights), lower_hz=lower_hz, upper_hz=upper_hz,
                first_bin_hz=float(frequency[selected][0]), last_bin_hz=float(frequency[selected][-1]),
                bins=int(selected.sum()), bin_spacing_hz=df,
                saved_pod_energy_fraction=float(weights.sum()),
                band_fraction_of_total_energy=fraction, band_energy=total*fraction,
                total_fluctuation_energy=total, spatial_coverage=pod.get('spatial_coverage', ''),
                spectra_file=str(path.resolve()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--pattern', default='*/pod_valid90_modes100_all/modal_spectra_100/modal_spectra.h5',
                        help='Glob relative to folder selecting one saved spectral-analysis run')
    parser.add_argument('--band-hz', nargs=2, type=float, default=(0., 1.), metavar=('LOW', 'HIGH'))
    parser.add_argument('--output', type=Path, help='CSV destination; defaults to FOLDER/pod_energy_LOW_to_HIGHHz.csv')
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    lo, hi = args.band_hz
    if not np.isfinite([lo, hi]).all() or not 0 <= lo < hi:
        parser.error('Require finite band limits with 0 <= LOW < HIGH')
    output = args.output or args.folder/f'pod_energy_{lo:g}_to_{hi:g}Hz.csv'
    if output.exists():
        parser.error(f'Output already exists: {output}; choose a new --output')
    sources = sorted(p for p in args.folder.glob(args.pattern) if p.is_file())
    if not sources:
        parser.error('No saved spectra matched --pattern; run pod_spectra_folder.bat first')
    rows = []
    for path in sources:
        try:
            rows.append(integrate_file(path, lo, hi))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f'{path}: {exc}; no comparison CSV written')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} cases to {output}')
    print('band_energy is spatially averaged fluctuation kinetic energy per unit mass '
          '(m^2/s^2 for m/s input), on retained POD support and using saved modes only.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
