"""Local energy budgets from valid observations, without POD truncation or gap fill."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np

from .reader import FlowDataset


EXPERIMENT_FREQUENCIES = {'SurgeLF': 2., 'PitchLF': 2., 'SurgeHF': 5., 'PitchHF': 5., 'Static': None}
ENERGY_FIELDS = ('total_energy', 'forcing_energy', 'forcing_uniform_cycle_energy',
                 'phase_locked_energy', 'residual_energy', 'cv_phase_prediction_error',
                 'cv_mean_prediction_error')


def case_label(source):
    source = Path(source)
    label = source.parent.name if source.stem in ('interpolated_velocity', 'velocity', 'raw') else source.stem
    return re.split(r'_\d+(?:\.\d+)?D', label, maxsplit=1)[0]


def resolve_frequency(source, frequency_hz=None):
    """Explicit override, source metadata, documented experiment mapping, phase metadata."""
    source = Path(source)
    if frequency_hz is not None:
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError('frequency_hz must be positive')
        return float(frequency_hz), 'explicit override'
    label = case_label(source)
    if label == 'Static':
        return None, 'Static case: no forcing reference (use explicit frequency override for a diagnostic fit)'
    with h5py.File(source, 'r') as f:
        stored = f.attrs.get('frequency_hz')
    if stored is not None:
        stored = float(stored)
        if np.isfinite(stored) and stored > 0:
            if label in EXPERIMENT_FREQUENCIES and EXPERIMENT_FREQUENCIES[label] is not None:
                if not np.isclose(stored, EXPERIMENT_FREQUENCIES[label]):
                    raise ValueError(f'{source}: frequency metadata conflicts with experiment case; set --frequency-hz explicitly')
            return stored, 'velocity-file frequency_hz'
    if label in EXPERIMENT_FREQUENCIES:
        return EXPERIMENT_FREQUENCIES[label], 'experiment case convention: LF=2 Hz, HF=5 Hz; Static has no forcing'
    candidates = set()
    for path in source.parent.glob('phase_average*.nc'):
        with h5py.File(path, 'r') as f:
            value = float(f.attrs.get('frequency_hz', np.nan))
            if np.isfinite(value) and value > 0:
                candidates.add(value)
    if len(candidates) == 1:
        return candidates.pop(), 'unique adjacent phase-average frequency'
    if len(candidates) > 1:
        raise ValueError(f'{source}: ambiguous phase frequencies; set --frequency-hz explicitly')
    return None, 'frequency unavailable; total energy only'


def cell_edges(coordinates):
    c = np.asarray(coordinates, dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)) or not np.all(np.diff(c) > 0):
        raise ValueError('Energy profiles require at least two increasing finite coordinates per axis')
    # Integrate only the measured coordinate interval, with half-width end cells.
    return np.r_[c[0], (c[:-1] + c[1:]) / 2, c[-1]]


def clipped_widths(edges, bounds):
    return np.maximum(0., np.minimum(edges[1:], bounds[1]) - np.maximum(edges[:-1], bounds[0]))


def common_bounds(sources, rotor_diameter, rotor_y=0., rotor_z=0., axes=('y', 'z')):
    if not np.isfinite(rotor_diameter) or rotor_diameter <= 0:
        raise ValueError('rotor_diameter must be positive')
    limits = {key: [] for key in axes}
    for source in sources:
        with h5py.File(source, 'r') as f:
            for key, origin in [('y', rotor_y), ('z', rotor_z)]:
                if key not in limits:
                    continue
                c = (f[key][:].astype(float) - origin) / rotor_diameter
                cell_edges(c)
                limits[key].append((float(c[0]), float(c[-1])))
    result = {}
    for key, ranges in limits.items():
        if not ranges:
            raise ValueError('No velocity files for common bounds')
        result[key] = (max(r[0] for r in ranges), min(r[1] for r in ranges))
        if result[key][0] >= result[key][1]:
            raise ValueError(f'No common {key} window; select a compatible subset or supply explicit bounds')
    return result


def group_sums(values, valid, groups, number):
    """values=(component,time,voxel); valid=(time,voxel)."""
    nv = valid.shape[1]
    indices = (groups[:, None] * nv + np.arange(nv)[None]).ravel()
    size = number * nv
    counts = np.bincount(indices, weights=valid.ravel(), minlength=size).reshape(number, nv)
    sums = np.stack([np.bincount(indices, weights=component.ravel(), minlength=size).reshape(number, nv)
                     for component in values])
    return counts, sums


def voxel_budget(data, valid, times, frequency_hz, phase_bins=32, phase_offset=0., min_phase_samples=3):
    """Population variances on observed vectors; phase ANOVA closes exactly.

    Sinusoidal regression includes an intercept. Its explained variance is
    evaluated at observed times; uniform-cycle harmonic energy is separate.
    """
    nt, nv = valid.shape
    n = valid.sum(axis=0).astype(float)
    safe_n = np.maximum(n, 1)
    clean = np.where(valid[None], data, 0.)
    mean = clean.sum(axis=1) / safe_n[None]
    centered = np.where(valid[None], data - mean[:, None], 0.)
    total = .5 * np.sum(centered**2, axis=(0, 1)) / safe_n
    result = {key: np.full(nv, np.nan) for key in ENERGY_FIELDS}
    result.update(total_energy=total, valid_fraction=n / nt, mean=mean,
                  eligible=n >= 3, cv_eligible=np.zeros(nv, dtype=bool),
                  min_phase_count=np.full(nv, np.nan), cycles_observed=np.full(nv, np.nan))
    if frequency_hz is None:
        return result
    phase = 2 * np.pi * frequency_hz * times + phase_offset
    cos, sin = np.cos(phase), np.sin(phase)
    vf = valid.astype(float)
    mc, ms = cos @ vf / safe_n, sin @ vf / safe_n
    gcc = (cos*cos) @ vf / safe_n - mc*mc
    gss = (sin*sin) @ vf / safe_n - ms*ms
    gcs = (cos*sin) @ vf / safe_n - mc*ms
    yc = np.einsum('ctv,t->cv', centered, cos) / safe_n[None]
    ys = np.einsum('ctv,t->cv', centered, sin) / safe_n[None]
    det = gcc*gss - gcs*gcs
    good_fit = det > 1e-10 * np.maximum((gcc+gss)**2, 1e-20)
    denominator = np.where(good_fit, det, np.nan)
    a = (yc*gss - ys*gcs) / denominator
    b = (ys*gcc - yc*gcs) / denominator
    forcing = .5 * np.sum(a*yc + b*ys, axis=0)
    result['forcing_energy'] = np.maximum(forcing, 0.)
    result['forcing_uniform_cycle_energy'] = .25 * np.sum(a*a + b*b, axis=0)
    result['harmonic_cos'], result['harmonic_sin'] = a, b

    bins = np.minimum(np.floor(np.mod(phase, 2*np.pi) * phase_bins / (2*np.pi)).astype(int), phase_bins-1)
    count, sums = group_sums(centered, valid, bins, phase_bins)
    phase_energy = .5 * np.sum(sums*sums / np.maximum(count[None], 1), axis=(0, 1)) / safe_n
    result['phase_locked_energy'] = phase_energy
    result['residual_energy'] = np.maximum(total - phase_energy, 0.)
    result['min_phase_count'] = count.min(axis=0)
    result['eligible'] &= good_fit & (count.min(axis=0) >= min_phase_samples)

    # Leave one whole forcing cycle out; do not treat snapshots as independent.
    cycle_values = np.floor(phase / (2*np.pi)).astype(np.int64)
    _, cycles = np.unique(cycle_values, return_inverse=True)
    ncycles = int(cycles.max()) + 1
    joint = cycles * phase_bins + bins
    group_count, group_sum = group_sums(centered, valid, joint, ncycles * phase_bins)
    group_count = group_count.reshape(ncycles, phase_bins, nv)
    group_sum = group_sum.reshape(3, ncycles, phase_bins, nv)
    result['cycles_observed'] = (group_count.sum(axis=1) > 0).sum(axis=0)
    train_count = count[None] - group_count
    train_sum = sums[:, None] - group_sum
    train_phase = train_sum / np.maximum(train_count[None], 1)
    cv_ok = np.all((group_count == 0) | (train_count > 0), axis=(0, 1)) & (ncycles >= 2)
    base_sse = np.sum(centered**2, axis=(0, 1))
    phase_sse = (base_sse - 2 * np.sum(train_phase*group_sum, axis=(0, 1, 2))
                 + np.sum(train_phase**2 * group_count[None], axis=(0, 1, 2)))
    cycle_count = group_count.sum(axis=1)
    cycle_sum = group_sum.sum(axis=2)
    train_n = n[None] - cycle_count
    train_mean = (centered.sum(axis=1)[:, None] - cycle_sum) / np.maximum(train_n[None], 1)
    mean_sse = (base_sse - 2*np.sum(train_mean*cycle_sum, axis=(0, 1))
                + np.sum(train_mean**2 * cycle_count[None], axis=(0, 1)))
    cv_ok &= np.all((cycle_count == 0) | (train_n > 0), axis=0)
    result['cv_eligible'] = cv_ok & result['eligible']
    result['cv_phase_prediction_error'] = np.where(cv_ok, .5*np.maximum(phase_sse, 0.) / safe_n, np.nan)
    result['cv_mean_prediction_error'] = np.where(cv_ok, .5*np.maximum(mean_sse, 0.) / safe_n, np.nan)
    return result


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def reduce_energy(maps, weights, requested_measure):
    """Ratio of weighted energies. CV uses exactly the same support or is withheld."""
    active = (weights > 0) & maps['support']
    effective = np.where(active, weights, 0.)
    measure = float(effective.sum())
    row = dict(supported_measure=measure, requested_measure=float(requested_measure),
               coverage_fraction=measure / requested_measure if requested_measure > 0 else 0.,
               supported_voxels=int(active.sum()))
    for key in ENERGY_FIELDS:
        values = maps[key]
        row[key] = float(np.sum(values[active]*effective[active]) / measure) if measure and np.all(np.isfinite(values[active])) else float('nan')
    total = row['total_energy']
    row['forcing_fraction'] = row['forcing_energy']/total if total > 0 else float('nan')
    row['phase_locked_fraction'] = row['phase_locked_energy']/total if total > 0 else float('nan')
    row['residual_fraction'] = row['residual_energy']/total if total > 0 else float('nan')
    baseline = row['cv_mean_prediction_error']
    row['cv_phase_predictive_fraction'] = 1-row['cv_phase_prediction_error']/baseline if baseline > 0 else float('nan')
    row['energy_closure_error'] = row['phase_locked_energy']+row['residual_energy']-total
    row['mean_valid_fraction'] = float(np.sum(maps['valid_fraction'][active]*effective[active])/measure) if measure else float('nan')
    return row


def run_forcing_response(source, output, *, min_valid_fraction=.9, frequency_hz=None,
                         phase_bins=32, phase_offset=0., min_phase_samples=3,
                         rotor_diameter=1200., rotor_y=0., rotor_z=0.,
                         y_bounds_d=None, z_bounds_d=None, slab_width_d=.1,
                         slab_origin_d=0., exclude_filled=False, zero_invalid=False,
                         spatial_chunk=256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new forcing-response output folder: {output}')
    if (not 0 < min_valid_fraction <= 1 or not np.isfinite(rotor_diameter) or rotor_diameter <= 0
            or not np.isfinite(slab_width_d) or slab_width_d <= 0 or phase_bins < 3
            or min_phase_samples < 1 or spatial_chunk < 1 or
            not np.all(np.isfinite([phase_offset, rotor_y, rotor_z, slab_origin_d]))):
        raise ValueError('Invalid energy-analysis settings')
    frequency, frequency_source = resolve_frequency(source, frequency_hz)
    with FlowDataset(source) as flow:
        nt, nz, ny, nx = flow.shape
        coordinates = {key: flow.coordinate(key).astype(float) for key in ('t', 'z', 'y', 'x')}
    times = coordinates['t']
    if nt < 3 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError('Need at least three increasing finite sample times in seconds')
    if frequency and frequency >= .5 / np.max(np.diff(times)):
        raise ValueError('Forcing frequency exceeds the sampling limit')
    scaled = {key: (coordinates[key] - origin) / rotor_diameter
              for key, origin in [('x', 0.), ('y', rotor_y), ('z', rotor_z)]}
    edges = {key: cell_edges(values) for key, values in scaled.items()}
    y_bounds = tuple(y_bounds_d) if y_bounds_d is not None else (scaled['y'][0], scaled['y'][-1])
    z_bounds = tuple(z_bounds_d) if z_bounds_d is not None else (scaled['z'][0], scaled['z'][-1])
    for bounds in (y_bounds, z_bounds):
        if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError('Bounds must be finite and increasing')
    dy, dz = clipped_widths(edges['y'], y_bounds), clipped_widths(edges['z'], z_bounds)
    area = dz[:, None] * dy[None]
    requested_area = (y_bounds[1]-y_bounds[0]) * (z_bounds[1]-z_bounds[0])
    if not np.any(area > 0):
        raise ValueError('Requested y-z window does not intersect this volume')
    shape = (nz, ny, nx)
    maps = {key: np.full(shape, np.nan) for key in (*ENERGY_FIELDS, 'valid_fraction', 'min_phase_count', 'cycles_observed')}
    maps['support'] = np.zeros(shape, dtype=bool)
    maps['cv_support'] = np.zeros(shape, dtype=bool)
    harmonic = {key: np.full((3, *shape), np.nan) for key in ('cos', 'sin')} if frequency else {}
    print(f'Energy profiles: {source.parent.name}; frequency={frequency}; using valid observations', flush=True)
    with h5py.File(source, 'r') as f:
        if exclude_filled and 'filled_mask' not in f:
            raise ValueError('--exclude-filled requires filled_mask')
        for iz in range(nz):
            if dz[iz] <= 0:
                continue
            data = np.stack([f[key][:, iz].astype(float) for key in 'uvw']).reshape(3, nt, -1)
            valid = np.all(np.isfinite(data), axis=0)
            for ic, key in enumerate('uvw'):
                for attr in ('_FillValue', 'missing_value'):
                    if attr in f[key].attrs:
                        for value in np.asarray(f[key].attrs[attr]).ravel():
                            valid &= data[ic] != value
            if zero_invalid:
                valid &= np.any(data != 0, axis=0)
            if exclude_filled:
                valid &= f['filled_mask'][:, iz].reshape(nt, -1) == 0
            candidates = np.flatnonzero((valid.mean(axis=0) >= min_valid_fraction)
                                       & np.repeat(dy > 0, nx))
            for start in range(0, len(candidates), spatial_chunk):
                selection = candidates[start:start+spatial_chunk]
                result = voxel_budget(data[:, :, selection], valid[:, selection], times, frequency,
                                      phase_bins, phase_offset, min_phase_samples)
                for key in (*ENERGY_FIELDS, 'valid_fraction', 'min_phase_count', 'cycles_observed'):
                    maps[key][iz].ravel()[selection] = result[key]
                maps['support'][iz].ravel()[selection] = result['eligible']
                maps['cv_support'][iz].ravel()[selection] = result['cv_eligible']
                if frequency:
                    for key in harmonic:
                        for ic in range(3):
                            harmonic[key][ic, iz].ravel()[selection] = result['harmonic_' + key][ic]
            if (iz+1) % 5 == 0:
                print(f'  energy statistics: {iz+1}/{nz} z planes', flush=True)
    if not maps['support'].any():
        raise ValueError('No supported voxels in requested window; review coverage and phase-bin counts')
    for key in ENERGY_FIELDS:
        maps[key][~maps['support']] = np.nan
    native = []
    for ix, x in enumerate(scaled['x']):
        plane = {key: value[:, :, ix] for key, value in maps.items()}
        native.append(dict(x_D=float(x), **reduce_energy(plane, area, requested_area)))
    slab_rows = []
    first = int(np.floor((edges['x'][0]-slab_origin_d)/slab_width_d))
    last = int(np.ceil((edges['x'][-1]-slab_origin_d)/slab_width_d))
    for index in range(first, last):
        lo = slab_origin_d + index * slab_width_d
        hi = lo + slab_width_d
        dx = clipped_widths(edges['x'], (lo, hi))
        observed_width = float(dx.sum())
        if observed_width <= 0:
            continue
        slab_rows.append(dict(x_D=(lo+hi)/2, x_lower_D=lo, x_upper_D=hi,
                              measured_x_lower_D=max(lo, float(edges['x'][0])),
                              measured_x_upper_D=min(hi, float(edges['x'][-1])),
                              axial_coverage_fraction=observed_width/slab_width_d,
                              **reduce_energy(maps, area[:, :, None]*dx[None, None], requested_area*slab_width_d)))
    volume = reduce_energy(maps, area[:, :, None]*np.diff(edges['x'])[None, None],
                           requested_area*(edges['x'][-1]-edges['x'][0]))
    positive_energy = maps['support'] & (maps['total_energy'] > 0)
    closure = (maps['phase_locked_energy']+maps['residual_energy']-maps['total_energy'])[maps['support']]
    metadata = dict(source=str(source), case=case_label(source), frequency_hz=frequency,
                    frequency_source=frequency_source, phase_bins=phase_bins, phase_offset=phase_offset,
                    snapshots=nt, recording_span_s=float(times[-1]-times[0]),
                    approximate_forcing_cycles=float((times[-1]-times[0])*frequency) if frequency else None,
                    min_valid_fraction=min_valid_fraction, min_phase_samples=min_phase_samples,
                    y_bounds_D=list(y_bounds), z_bounds_D=list(z_bounds), rotor_diameter=rotor_diameter,
                    rotor_y=rotor_y, rotor_z=rotor_z, slab_width_D=slab_width_d, slab_origin_D=slab_origin_d,
                    exclude_filled=exclude_filled, zero_invalid=zero_invalid,
                    support='fixed per voxel; valid-vector fraction and every phase-bin count threshold',
                    sample_weighting='equal valid observations at each voxel; population variances',
                    spatial_weighting='clipped cell area/volume within requested window; ratio of averaged energies',
                    missing_policy='no fill; existing interpolated values included unless exclude_filled',
                    forcing_estimator='observed-time variance explained by least-squares intercept+cos+sin',
                    uniform_cycle_harmonic='0.25 * sum_components(cos_coefficient^2+sin_coefficient^2); separate from observed-time budget',
                    phase_estimator='observed-count-weighted phase-bin means; exact in-sample ANOVA partition',
                    residual_definition='within-phase-bin variance, including noise and unresolved waveform variation',
                    cv_definition='leave one forcing cycle out; compare phase-bin prediction error with training-mean prediction error',
                    cv_limitation='predictive check, not a confidence interval; neighbouring cycles may remain dependent',
                    ratio_caution='forcing and phase estimates are separate regressions; forcing energy is not subtracted from phase energy',
                    coverage_caution='common requested window does not guarantee identical supported area between planes/cases',
                    max_absolute_closure_error=float(np.nanmax(np.abs(closure))) if frequency else None,
                    max_forcing_fraction=float(np.nanmax(maps['forcing_energy'][positive_energy]/maps['total_energy'][positive_energy])) if frequency and positive_energy.any() else None,
                    volume_summary=volume)
    output.mkdir(parents=True)
    with h5py.File(output/'energy_maps.h5', 'w') as f:
        f.attrs['metadata_json'] = json.dumps(metadata)
        for key in ('x', 'y', 'z'):
            f[key] = coordinates[key]
            f[key+'_D'] = scaled[key]
        for key, value in maps.items():
            f.create_dataset(key, data=value, compression='gzip')
        for key, value in harmonic.items():
            f.create_dataset('harmonic_'+key, data=value, compression='gzip')
    write_csv(output/'native_planes.csv', native)
    write_csv(output/'downstream_slabs.csv', slab_rows)
    write_csv(output/'volume_energy.csv', [volume])
    (output/'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plot_profiles(native, slab_rows, metadata, output/'energy_profiles.png')
    return metadata


def plot_profiles(native, slabs, metadata, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    x = [r['x_D'] for r in native]
    xs = [r['x_D'] for r in slabs]
    colors = {'total_energy': 'black', 'forcing_energy': 'tab:red',
              'phase_locked_energy': 'tab:blue', 'residual_energy': 'tab:green'}
    for key, color in colors.items():
        if key != 'total_energy' and metadata['frequency_hz'] is None:
            continue
        label = key.replace('_energy', '').replace('_', ' ')
        axes[0].plot(x, [r[key] for r in native], color=color, alpha=.3, linewidth=1)
        axes[0].plot(xs, [r[key] for r in slabs], 'o-', color=color, label=label)
    for key, label, color in [('forcing_fraction', 'Forcing / total', 'tab:red'),
                              ('phase_locked_fraction', 'Phase locked / total', 'tab:blue'),
                              ('cv_phase_predictive_fraction', 'Held-out-cycle predictive fraction', 'tab:purple')]:
        if metadata['frequency_hz'] is not None:
            axes[1].plot(xs, [100*r[key] for r in slabs], 'o-', label=label, color=color)
    axes[2].plot(x, [100*r['coverage_fraction'] for r in native], color='grey', alpha=.5, label='Native plane')
    axes[2].plot(xs, [100*r['coverage_fraction'] for r in slabs], 'o-', label='Slab (includes partial axial extent)')
    axes[0].set_ylabel('Energy (velocity units squared)')
    axes[1].set_ylabel('Fraction (%)')
    axes[1].axhline(0, color='grey', linewidth=.6)
    axes[2].set(xlabel='x/D', ylabel='Requested-region coverage (%)', ylim=(0, 100))
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.lines and ax.get_legend_handles_labels()[0]:
            ax.legend()
    reference = f"{metadata['frequency_hz']:g} Hz" if metadata['frequency_hz'] else 'no forcing reference'
    fig.suptitle(f"{metadata['case']}: {reference}; valid observations; {metadata['slab_width_D']:g}D slabs\nThin lines: native planes. Phase energy is an in-sample estimate; held-out check is not a confidence interval.")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
