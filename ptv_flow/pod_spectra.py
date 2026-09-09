"""Spectra of existing local POD coefficients; no velocity fields are read."""
from pathlib import Path
import csv
import json

import h5py
import numpy as np

from .forcing_response import case_label, EXPERIMENT_FREQUENCIES


def periodogram(values, fs, hann=True):
    n = len(values)
    window = .5 - .5*np.cos(2*np.pi*np.arange(n)/n) if hann else np.ones(n)
    centered = values - values.mean(axis=0)
    psd = np.abs(np.fft.rfft(centered*window[:, None], axis=0))**2 / (fs*np.sum(window**2))
    psd[1:] *= 2
    if n % 2 == 0:
        psd[-1] /= 2
    return np.fft.rfftfreq(n, 1/fs), psd


def calculate_spectra(t, coefficients, welch_seconds=4., sampling_tolerance=.001):
    t = np.asarray(t, dtype=float)
    a = np.asarray(coefficients, dtype=float)
    if t.ndim != 1 or a.ndim != 2 or len(t) != len(a) or len(t) < 8 or not a.shape[1]:
        raise ValueError('Require t=(time,) and temporal_coefficients=(time,mode), with at least 8 samples')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)) or np.any(np.diff(t) <= 0):
        raise ValueError('Times must increase and times/coefficients must be finite')
    dt = (t[-1]-t[0])/(len(t)-1)
    jitter = float(np.max(np.abs(np.diff(t)-dt))/dt)
    if not np.isfinite(sampling_tolerance) or not 0 <= sampling_tolerance <= .01:
        raise ValueError('Sampling tolerance must be between 0 and 0.01')
    if jitter > sampling_tolerance:
        raise ValueError(f'Irregular sampling: relative interval deviation {jitter:g}; no automatic resampling')
    if not np.isfinite(welch_seconds) or welch_seconds <= 0:
        raise ValueError('Welch segment length must be positive')
    fs = 1/dt
    frequency, hann = periodogram(a, fs, True)
    _, rectangular = periodogram(a, fs, False)
    nsegment = min(len(t), max(8, int(round(welch_seconds*fs))))
    starts = list(range(0, len(t)-nsegment+1, max(1, nsegment//2)))
    estimates = [periodogram(a[start:start+nsegment], fs, True) for start in starts]
    welch = np.mean([p for _, p in estimates], axis=0)
    return dict(frequency=frequency, hann=hann, rectangular=rectangular,
                welch_frequency=estimates[0][0], welch=welch,
                variance=np.var(a, axis=0), sample_rate_hz=fs,
                relative_interval_deviation=jitter, welch_segment_samples=nsegment,
                welch_segments=len(starts), welch_unused_tail_samples=len(t)-(starts[-1]+nsegment))


def csv_file(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def analyze_pod_file(source, output, *, forcing_frequency_hz=None, band_half_width_hz=.2,
                     harmonics=3, welch_seconds=4., plot_max_hz=20., sampling_tolerance=.001):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}')
    if not np.isfinite(band_half_width_hz) or band_half_width_hz <= 0 or harmonics < 1 or not np.isfinite(plot_max_hz) or plot_max_hz <= 0:
        raise ValueError('Require positive bandwidth, harmonic count and plot frequency limit')
    with h5py.File(source, 'r') as f:
        t, a = f['t'][:], f['temporal_coefficients'][:]
        fractions = f['energy_fraction'][:].astype(float)
        metadata = json.loads(f.attrs.get('metadata_json', '{}'))
    result = calculate_spectra(t, a, welch_seconds, sampling_tolerance)
    if fractions.shape != (a.shape[1],) or not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or fractions.sum() > 1+1e-6:
        raise ValueError('Invalid saved POD energy fractions')
    label = case_label(Path(metadata.get('source', source.parent.parent/'interpolated_velocity.nc')))
    forcing = forcing_frequency_hz
    provenance = 'explicit override'
    if forcing is None:
        forcing = EXPERIMENT_FREQUENCIES.get(label)
        provenance = 'experiment label: LF=2 Hz, HF=5 Hz; Static/unknown has no reference'
    if forcing is not None and (not np.isfinite(forcing) or forcing <= 0):
        raise ValueError('Forcing reference must be positive')
    if forcing is not None and 2*band_half_width_hz >= forcing:
        raise ValueError('Frequency bands must not overlap or include DC; reduce --band-half-width-hz')
    freq, df = result['frequency'], result['sample_rate_hz']/len(t)
    rect = result['rectangular']
    integrated = rect.sum(axis=0)*df
    peak = np.argmax(result['hann'][1:], axis=0)+1
    wpeak = np.argmax(result['welch'][1:], axis=0)+1
    mode_rows = [dict(mode=k+1, pod_energy_fraction=float(fractions[k]),
                      coefficient_variance=float(result['variance'][k]),
                      hann_peak_hz=float(freq[peak[k]]) if integrated[k] > 0 else None,
                      welch_peak_hz=float(result['welch_frequency'][wpeak[k]]) if integrated[k] > 0 else None)
                 for k in range(a.shape[1])]
    band_rows = []
    notes = []
    if result['welch_segments'] < 2:
        notes.append('Welch uses one segment; there is no averaging benefit.')
    if forcing is None:
        notes.append('No forcing reference: spectra and peaks saved; forcing-band table has no rows.')
    else:
        for h in range(1, harmonics+1):
            target = h*forcing
            lo, hi = target-band_half_width_hz, target+band_half_width_hz
            if hi > result['sample_rate_hz']/2:
                notes.append(f'Harmonic {h} omitted: its full band exceeds Nyquist.')
                continue
            # Endpoints have a small tolerance to accommodate float32 time coordinates.
            chosen = (freq >= lo-df*1e-5) & (freq <= hi+df*1e-5)
            if not chosen.any():
                notes.append(f'Harmonic {h} omitted: no Fourier bins inside the band; record too short or band too narrow.')
                continue
            share = np.divide(rect[chosen].sum(axis=0)*df, integrated,
                              out=np.full_like(integrated, np.nan), where=integrated > 0)
            for k in range(a.shape[1]):
                band_rows.append(dict(mode=k+1, harmonic=h, target_hz=target, lower_hz=lo, upper_hz=hi,
                                      first_bin_hz=float(freq[chosen][0]), last_bin_hz=float(freq[chosen][-1]),
                                      bins=int(chosen.sum()), within_mode_energy_fraction=float(share[k]),
                                      fraction_of_total_pod_energy=float(fractions[k]*share[k])))
    summary = dict(pod_file=str(source), case=label, input_pod_metadata=metadata,
                   samples=len(t), modes=a.shape[1], sample_rate_hz=result['sample_rate_hz'],
                   full_record_duration_s=len(t)/result['sample_rate_hz'],
                   full_record_bin_spacing_hz=df, relative_interval_deviation=result['relative_interval_deviation'],
                   sampling_tolerance=sampling_tolerance, welch_requested_seconds=welch_seconds,
                   welch_actual_seconds=result['welch_segment_samples']/result['sample_rate_hz'],
                   welch_segments=result['welch_segments'], welch_unused_tail_samples=result['welch_unused_tail_samples'],
                   welch_bin_spacing_hz=result['sample_rate_hz']/result['welch_segment_samples'],
                   forcing_frequency_hz=forcing, forcing_reference_source=provenance,
                   band_half_width_hz=band_half_width_hz, requested_harmonics=harmonics,
                   saved_pod_energy_fraction=float(fractions.sum()),
                   parseval_max_absolute_error=float(np.max(np.abs(integrated-result['variance']))),
                   plot_max_hz=plot_max_hz,
                   spectral_method='One-sided PSD; de-mean each segment; periodic Hann full record and Welch with 50% overlap',
                   band_method='Full-record rectangular periodogram, exact discrete variance partition; bins selected by centre',
                   band_caution='Includes leakage and broadband energy; not a phase-locked forcing-energy estimate',
                   total_energy_caution='Band fractions multiply saved POD fractions; excludes unsaved modes and inherits POD masking/gap filling',
                   notes=notes)
    output.mkdir(parents=True)
    with h5py.File(output/'modal_spectra.h5', 'w') as f:
        f.attrs['metadata_json'] = json.dumps(summary)
        for key in ('frequency', 'hann', 'rectangular', 'welch_frequency', 'welch', 'variance'):
            f[key] = result[key]
        f['pod_energy_fraction'] = fractions
    csv_file(output/'mode_summary.csv', list(mode_rows[0]), mode_rows)
    csv_file(output/'forcing_bands.csv', ['mode', 'harmonic', 'target_hz', 'lower_hz', 'upper_hz',
             'first_bin_hz', 'last_bin_hz', 'bins', 'within_mode_energy_fraction', 'fraction_of_total_pod_energy'], band_rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_spectra(result, fractions, forcing, plot_max_hz, label, output)
    return summary


def plot_spectra(result, fractions, forcing, limit, label, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    freq, psd = result['frequency'], result['hann']
    maximum = min(limit, result['sample_rate_hz']/2)
    visible = freq <= maximum
    for start in range(0, psd.shape[1], 6):
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        for k, ax in enumerate(axes.ravel(), start=start):
            if k >= psd.shape[1]:
                ax.set_visible(False); continue
            ax.plot(freq, psd[:, k], label='Full-record Hann', linewidth=1)
            ax.plot(result['welch_frequency'], result['welch'][:, k], label='Welch', linewidth=1)
            if forcing is not None:
                ax.axvline(forcing, color='red', linestyle='--', linewidth=.8)
            ax.set(xlim=(0, maximum), title=f'Mode {k+1}: {fractions[k]:.2%} of POD energy',
                   xlabel='Frequency (Hz)', ylabel='Coefficient PSD (velocity units squared/Hz)')
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        fig.suptitle(f'{label}: local POD coefficient spectra')
        fig.tight_layout(); fig.savefig(output/f'mode_spectra_{start+1:02d}_{min(start+6, psd.shape[1]):02d}.png', dpi=160); plt.close(fig)
    # Each row integrates to one over the full Nyquist range, not just the plotted range.
    density = np.divide(psd, psd.sum(axis=0, keepdims=True)*(freq[1]-freq[0]),
                        out=np.zeros_like(psd), where=psd.sum(axis=0, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(12, 6))
    if visible.sum() >= 2:
        artist = ax.pcolormesh(freq[visible], np.arange(1, psd.shape[1]+1), density[visible].T,
                               shading='nearest', cmap='magma')
        fig.colorbar(artist, ax=ax, label='Within-mode normalized Hann PSD (1/Hz)')
    ax.set(xlabel='Frequency (Hz)', ylabel='POD mode', title=f'{label}: within-mode frequency content', xlim=(0, maximum))
    if forcing is not None:
        ax.axvline(forcing, color='cyan', linestyle='--')
    fig.tight_layout(); fig.savefig(output/'mode_frequency_heatmap.png', dpi=180); plt.close(fig)
