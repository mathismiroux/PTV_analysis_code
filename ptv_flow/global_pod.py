"""Joint POD of phase-locked volumes, without inventing inter-run turbulence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np


def case_and_distance(path):
    match = re.fullmatch(r'(.+?)_(\d+(?:\.\d+)?)D(?:__.*)?', Path(path).parent.name)
    if not match:
        raise ValueError(f'Expected a case_distanceD parent folder: {path}')
    return match[1], float(match[2])


def discover_cases(root, pattern='*/phase_average*.nc', selected_case=None):
    cases = {}
    for path in sorted(Path(root).glob(pattern)):
        if path.is_file():
            case, distance = case_and_distance(path)
            if selected_case is not None and case != selected_case:
                continue
            cases.setdefault(case, []).append((distance, path.resolve()))
    result = {}
    for case, entries in cases.items():
        distances = [d for d, _ in entries]
        if len(set(distances)) != len(distances):
            raise ValueError(f'Multiple phase files for a {case} location; narrow --pattern')
        result[case] = [p for _, p in sorted(entries)]
    return result


def inspect_sources(paths):
    infos = []
    for path in paths:
        with h5py.File(path, 'r') as f:
            phase = f['phase'][:].astype(float)
            expected = (np.arange(len(phase)) + .5) * 2 * np.pi / len(phase)
            if len(phase) < 3 or not np.allclose(phase, expected, atol=1e-6, rtol=0):
                raise ValueError(f'{path}: require uniform full-cycle phase-bin centres in radians')
            if f.attrs.get('phase_source') != 'frequency':
                raise ValueError(f'{path}: require the common frequency-based phase reference')
            freq = float(f.attrs['frequency_hz'])
            offset = float(f.attrs['phase_offset'])
            if not np.isfinite(freq) or freq <= 0 or not np.isfinite(offset):
                raise ValueError(f'{path}: invalid forcing frequency/offset')
            coords = {k: f[k][:].astype(float) for k in ('z', 'y', 'x')}
            for k, c in coords.items():
                if len(c) < 2 or not np.all(np.isfinite(c)) or not np.all(np.diff(c) > 0):
                    raise ValueError(f'{path}: {k} must contain increasing finite coordinates')
            shape = (len(phase), *(len(coords[k]) for k in ('z', 'y', 'x')))
            for k in 'uvw':
                if f[k + '_phase_mean'].shape != shape or f[k + '_phase_count'].shape != shape:
                    raise ValueError(f'{path}: inconsistent phase mean/count shapes')
            samples = f['phase_sample_count'][:].astype(float)
            if samples.shape != phase.shape or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
                raise ValueError(f'{path}: empty or invalid phase bins')
            policy = (str(f.attrs.get('invalid_samples', 'unknown')),
                      str(f.attrs.get('zero_mask', 'unknown')))
            if policy[1] != 'vector':
                raise ValueError(f'{path}: require vector validity counts')
            info = dict(path=Path(path), coords=coords, phase=phase, frequency=freq,
                        offset=offset, policy=policy, samples=samples,
                        case=case_and_distance(path)[0])
            if infos:
                ref = infos[0]
                if info['case'] != ref['case']:
                    raise ValueError('A global POD must contain only one forcing case')
                if (phase.shape != ref['phase'].shape or
                        not np.allclose(phase, ref['phase'], atol=1e-6, rtol=0) or
                        not np.isclose(freq, ref['frequency'], atol=1e-8, rtol=0) or
                        abs(np.angle(np.exp(1j * (offset - ref['offset'])))) > 1e-6 or
                        policy != ref['policy']):
                    raise ValueError(f'{path}: phase reference, frequency or validity policy mismatch')
            infos.append(info)
    if not infos:
        raise ValueError('No input volumes')
    if len({i['path'].resolve() for i in infos}) != len(infos):
        raise ValueError('Duplicate input volume')
    return infos


def interpolation_map(coords, target, support):
    """Trilinear mapping; reject targets requiring any unsupported corner.

    No extrapolation and no interpolation across missing spatial cells.
    Returns flattened target indices and source corner indices/weights.
    """
    axes = ('z', 'y', 'x')
    selections = [np.flatnonzero((target[k] >= coords[k][0]) &
                                 (target[k] <= coords[k][-1])) for k in axes]
    if any(len(s) == 0 for s in selections):
        return np.array([], dtype=int), []
    indices = np.meshgrid(*selections, indexing='ij')
    base, frac = [], []
    for k, idx in zip(axes, indices):
        c = coords[k]
        values = target[k][idx.ravel()]
        lo = np.clip(np.searchsorted(c, values, side='right') - 1, 0, len(c) - 2)
        base.append(lo)
        frac.append(np.clip((values - c[lo]) / (c[lo + 1] - c[lo]), 0, 1))
    flat = np.ravel_multi_index(tuple(idx.ravel() for idx in indices),
                               tuple(len(target[k]) for k in axes))
    accepted = np.ones(len(flat), dtype=bool)
    corners = []
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                bits = (dz, dy, dx)
                weight = np.ones(len(flat))
                for j, bit in enumerate(bits):
                    weight *= frac[j] if bit else 1 - frac[j]
                corner = np.ravel_multi_index(tuple(base[j] + bits[j] for j in range(3)), support.shape)
                accepted &= (weight < 1e-12) | support.ravel()[corner]
                corners.append((corner, weight))
    return flat[accepted], [(idx[accepted], w[accepted]) for idx, w in corners]


def interpolate_values(data, corners):
    flat = data.reshape(*data.shape[:-3], -1)
    result = np.zeros((*data.shape[:-3], len(corners[0][0])), dtype=float)
    for indices, weights in corners:
        result += np.where(weights > 1e-12, flat[..., indices], 0) * weights
    return result


def exact_pod(state, weights, modes):
    """state (component, phase, retained voxel); normalized spatial weights."""
    ncomp, nphase, nvox = state.shape
    mean = state.mean(axis=1)
    fluct = state - mean[:, None]
    matrix = fluct.transpose(0, 2, 1).reshape(ncomp * nvox, nphase)
    repeated = np.tile(weights, ncomp)
    weighted = matrix * np.sqrt(repeated[:, None])
    covariance = weighted.T @ weighted / nphase
    ev, vectors = np.linalg.eigh(covariance)
    order = np.argsort(ev)[::-1]
    total = float(np.trace(covariance))
    if not np.isfinite(total) or total <= 0:
        raise ValueError('No nonzero phase-locked fluctuation energy')
    order = order[ev[order] > total * 1e-12]
    spectrum = ev[order]
    order = order[:modes]
    eigenvalue = ev[order]
    temporal = vectors[:, order]
    spatial = matrix @ temporal / np.sqrt(nphase * eigenvalue)[None]
    coefficients = temporal * np.sqrt(nphase * eigenvalue)[None]
    for im in range(len(order)):
        if spatial[np.argmax(np.abs(spatial[:, im])), im] < 0:
            spatial[:, im] *= -1
            coefficients[:, im] *= -1
    gram = spatial.T @ (repeated[:, None] * spatial)
    orthogonality = float(np.max(np.abs(gram - np.eye(len(order)))))
    residual = np.linalg.norm(covariance @ temporal - temporal * eigenvalue, axis=0) / eigenvalue
    return dict(mean=mean, fluct=fluct, spatial=spatial.reshape(ncomp, nvox, -1).transpose(0, 2, 1),
                coefficients=coefficients, eigenvalue=eigenvalue, spectrum=spectrum,
                total=total, orthogonality_error=orthogonality, eigen_residual=residual)


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_global_pod(paths, output, min_valid_fraction=.9, modes=20, spacing=None,
                   rotor_diameter=1200., max_grid_voxels=2000000):
    if not 0 < min_valid_fraction <= 1 or modes < 1 or rotor_diameter <= 0:
        raise ValueError('Require 0 < valid fraction <= 1, positive modes and rotor diameter')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    infos = inspect_sources(paths)
    auto_spacing = max(float(np.median(np.diff(info['coords'][k]))) for info in infos for k in ('z', 'y', 'x'))
    spacing = auto_spacing if spacing is None else float(spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Grid spacing must be positive')
    target = {}
    for k in ('z', 'y', 'x'):
        lo = min(info['coords'][k][0] for info in infos)
        hi = max(info['coords'][k][-1] for info in infos)
        target[k] = lo + np.arange(int(np.floor((hi - lo) / spacing)) + 1) * spacing
    shape = tuple(len(target[k]) for k in ('z', 'y', 'x'))
    nvox = int(np.prod(shape))
    if nvox > max_grid_voxels:
        raise ValueError(f'Grid has {nvox} voxels; increase --spacing or --max-grid-voxels')
    nphase = len(infos[0]['phase'])
    accumulated = np.zeros((3, nphase, nvox), dtype=np.float64)
    contributors = np.zeros(nvox, dtype=np.uint16)
    geometric = np.zeros(nvox, dtype=bool)
    memberships, diagnostics, overlap_rows = [], [], []
    for info in infos:
        path = info['path']
        print(f'Reading existing phase averages: {path.parent.name}', flush=True)
        with h5py.File(path, 'r') as f:
            state = np.stack([f[k + '_phase_mean'][:].astype(float) for k in 'uvw'])
            counts = [f[k + '_phase_count'][:] for k in 'uvw']
            if not all(np.array_equal(counts[0], count) for count in counts[1:]):
                raise ValueError(f'{path}: vector component counts disagree')
            count = counts[0].astype(float)
            samples = info['samples']
            if np.any(~np.isfinite(count)) or np.any(count < 0) or np.any(count > samples[:, None, None, None]):
                raise ValueError(f'{path}: invalid measurement counts')
            valid_fraction = count.sum(axis=0) / samples.sum()
            support = (valid_fraction >= min_valid_fraction) & np.all(np.isfinite(state), axis=(0, 1))
            support &= np.all(count > 0, axis=0)
        footprint, _ = interpolation_map(info['coords'], target, np.ones(support.shape, dtype=bool))
        geometric[footprint] = True
        indices, corners = interpolation_map(info['coords'], target, support)
        member = np.zeros(nvox, dtype=bool)
        member[indices] = True
        memberships.append(member)
        diagnostics.append(dict(source=str(path), native_voxels=support.size,
                                native_retained_voxels=int(support.sum()),
                                native_coverage=float(support.mean()),
                                mapped_voxels=len(indices)))
        if not len(indices):
            raise ValueError(f'{path}: no supported voxels on target grid at selected threshold')
        values = interpolate_values(state, corners)
        overlap = contributors[indices] > 0
        if overlap.any():
            previous = accumulated[:, :, indices[overlap]] / contributors[indices[overlap]][None, None]
            current = values[:, :, overlap]
            difference = current - previous
            coherent_difference = (current - current.mean(axis=1, keepdims=True)
                                   - previous + previous.mean(axis=1, keepdims=True))
            current_fluct = current - current.mean(axis=1, keepdims=True)
            previous_fluct = previous - previous.mean(axis=1, keepdims=True)
            rms_current = float(np.sqrt(np.mean(np.sum(current_fluct**2, axis=0))))
            rms_previous = float(np.sqrt(np.mean(np.sum(previous_fluct**2, axis=0))))
            rms_difference = float(np.sqrt(np.mean(np.sum(coherent_difference**2, axis=0))))
            basis = np.exp(-1j * info['phase']) * 2 / nphase
            hcurrent = np.einsum('cpv,p->cv', current_fluct, basis)
            hprevious = np.einsum('cpv,p->cv', previous_fluct, basis)
            cross = np.sum(hcurrent * np.conj(hprevious))
            norm = float(np.linalg.norm(hcurrent) * np.linalg.norm(hprevious))
            overlap_rows.append(dict(source=str(path), comparison='previous blended volumes',
                                     overlap_voxels=int(overlap.sum()),
                                     velocity_vector_rms_difference=float(np.sqrt(np.mean(np.sum(difference**2, axis=0)))),
                                     coherent_vector_rms_difference=rms_difference,
                                     current_coherent_vector_rms=rms_current,
                                     previous_coherent_vector_rms=rms_previous,
                                     normalized_coherent_rms_difference=rms_difference / np.sqrt((rms_current**2 + rms_previous**2) / 2) if rms_current + rms_previous else 0.,
                                     fundamental_phase_difference_degrees=float(np.degrees(np.angle(cross))) if norm else None,
                                     fundamental_complex_correlation=float(abs(cross) / norm) if norm else None))
        accumulated[:, :, indices] += values
        contributors[indices] += 1
    retained = contributors > 0
    state = accumulated[:, :, retained] / contributors[retained][None, None]
    del accumulated
    weights = np.full(int(retained.sum()), 1. / retained.sum())
    print(f'Joint POD: {len(infos)} volumes, {retained.sum()} grid points, {nphase} phases', flush=True)
    result = exact_pod(state, weights, modes)
    phase = infos[0]['phase']
    freq = infos[0]['frequency']
    # Fundamental q = a*cos(theta) + b*sin(theta) = Re[(a-i*b)*exp(i*theta)].
    harmonic_a = 2 / nphase * np.einsum('cpv,p->cv', result['fluct'], np.cos(phase))
    harmonic_b = 2 / nphase * np.einsum('cpv,p->cv', result['fluct'], np.sin(phase))
    harmonic_energy = float(np.sum(weights * np.sum(harmonic_a**2 + harmonic_b**2, axis=0)) / 4)
    ev, spectrum = result['eigenvalue'], result['spectrum']
    modal_density = np.sum(result['spatial']**2, axis=0) * weights[None]
    contributions = np.stack([(modal_density * member[retained][None] / contributors[retained][None]).sum(axis=1)
                              for member in memberships])
    metadata = dict(case=infos[0]['case'], sources=[str(i['path']) for i in infos],
                    source_count=len(infos), frequency_hz=freq, phase_offset=infos[0]['offset'],
                    phase_reference='2*pi*frequency*t + phase_offset; common platform start phase asserted by user',
                    phase_bins=nphase, min_valid_fraction=min_valid_fraction,
                    validity='>= threshold across original snapshots; all components and phase means finite; positive counts in every phase',
                    missing_policy='excluded; no temporal-mean filling of phase bins',
                    interpolation='trilinear; all nonzero-weight corners supported; no extrapolation',
                    overlap_policy='equal-volume-source averaging with phase-independent weights; one target voxel counted once',
                    mean_policy='equal-phase cycle mean computed from existing phase averages',
                    energy_definition='0.5 * retained-volume mean of phase-locked u_prime^2+v_prime^2+w_prime^2',
                    spatial_weighting='uniform target cell volume divided by retained volume',
                    spacing_source_units=spacing, rotor_diameter_source_units=rotor_diameter,
                    shape_z_y_x=shape, retained_voxels=int(retained.sum()),
                    measured_footprint_voxels=int(geometric.sum()),
                    coverage_of_measured_footprint=float(retained.sum() / geometric.sum()),
                    overlap_voxels=int(np.sum(contributors > 1)),
                    overlap_diagnostics=overlap_rows,
                    retained_volume_source_units_cubed=float(retained.sum() * spacing**3),
                    total_coherent_energy=result['total'] / 2,
                    fundamental_energy=harmonic_energy,
                    fundamental_energy_fraction=harmonic_energy / (result['total'] / 2),
                    saved_modes=len(ev), resolved_rank=len(spectrum),
                    saved_energy_fraction=float(ev.sum() / result['total']),
                    orthogonality_error=result['orthogonality_error'],
                    max_relative_eigen_residual=float(result['eigen_residual'].max()))
    output.mkdir(parents=True)

    def full(values):
        array = np.full((*values.shape[:-1], nvox), np.nan, dtype=np.float32)
        array[..., retained] = values
        return array.reshape(*values.shape[:-1], *shape)

    with h5py.File(output / 'global_pod.h5', 'w') as f:
        f.attrs['metadata_json'] = json.dumps(metadata)
        for key, value in target.items():
            f[key] = value
        f['phase'] = phase
        f['phase_degrees'] = np.degrees(phase)
        f['support'] = retained.reshape(shape)
        f['measured_footprint'] = geometric.reshape(shape)
        f['contributor_count'] = contributors.reshape(shape)
        f['spatial_weight'] = (retained.astype(float) / retained.sum()).reshape(shape)
        f['eigenvalue'] = ev
        f['energy_fraction'] = ev / result['total']
        f['full_spectrum_eigenvalue'] = spectrum
        f['phase_coefficients'] = result['coefficients']
        f['source_mode_energy_share'] = contributions
        f['source_labels'] = np.array([i['path'].parent.name for i in infos], dtype=h5py.string_dtype())
        for ic, key in enumerate('uvw'):
            for name, value in [('mode_' + key, result['spatial'][ic]),
                                ('cycle_mean_' + key, result['mean'][ic]),
                                ('phase_mean_' + key, state[ic]),
                                (key + '_fundamental_cos', harmonic_a[ic]),
                                (key + '_fundamental_sin', harmonic_b[ic])]:
                f.create_dataset(name, data=full(value), compression='gzip')
    write_csv(output / 'modal_energy.csv', ['mode', 'energy', 'energy_fraction', 'cumulative_energy_fraction', 'spatial_mode_saved'],
              [dict(mode=i + 1, energy=value / 2, energy_fraction=value / result['total'],
                    cumulative_energy_fraction=float(spectrum[:i + 1].sum() / result['total']),
                    spatial_mode_saved=i < len(ev)) for i, value in enumerate(spectrum)])
    write_csv(output / 'source_coverage.csv', list(diagnostics[0]), diagnostics)
    write_csv(output / 'overlap_diagnostics.csv', ['source', 'comparison', 'overlap_voxels',
              'velocity_vector_rms_difference', 'coherent_vector_rms_difference',
              'current_coherent_vector_rms', 'previous_coherent_vector_rms',
              'normalized_coherent_rms_difference', 'fundamental_phase_difference_degrees',
              'fundamental_complex_correlation'], overlap_rows)
    write_csv(output / 'source_mode_energy.csv', ['source', 'mode', 'share_of_mode_energy', 'allocated_energy'],
              [dict(source=info['path'].parent.name, mode=im + 1,
                    share_of_mode_energy=contributions[j, im], allocated_energy=contributions[j, im] * ev[im] / 2)
               for j, info in enumerate(infos) for im in range(len(ev))])
    # Exact discrete harmonics of phase coefficients; no repeated cycles invented.
    transform = np.fft.rfft(result['coefficients'], axis=0) / nphase
    harmonic_variance = np.abs(transform)**2
    harmonic_variance[1:] *= 2
    if nphase % 2 == 0:
        harmonic_variance[-1] /= 2
    write_csv(output / 'modal_harmonics.csv', ['mode', 'harmonic', 'frequency_hz', 'fraction_of_mode_energy'],
              [dict(mode=im + 1, harmonic=h, frequency_hz=h * freq,
                    fraction_of_mode_energy=harmonic_variance[h, im] / ev[im])
               for im in range(len(ev)) for h in range(1, len(harmonic_variance))])
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plot_global_pod(output)
    return metadata


def plot_global_pod(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    with h5py.File(output / 'global_pod.h5', 'r') as f:
        meta = json.loads(f.attrs['metadata_json'])
        diameter = meta['rotor_diameter_source_units']
        x, y, z = (f[k][:] / diameter for k in ('x', 'y', 'z'))
        iz = int(np.argmin(np.abs(z)))
        spectrum = f['full_spectrum_eigenvalue'][:]
        fractions = spectrum / (2 * meta['total_coherent_energy'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        index = np.arange(1, len(fractions) + 1)
        axes[0].bar(index, fractions * 100)
        axes[0].set(xlabel='Global mode', ylabel='Phase-locked energy (%)')
        axes[1].plot(index, np.cumsum(fractions) * 100, 'o-')
        axes[1].set(xlabel='Number of modes', ylabel='Cumulative energy (%)', ylim=(0, 101))
        fig.suptitle(f"{meta['case']}: global phase-locked POD")
        fig.tight_layout(); fig.savefig(output / 'modal_energy.png', dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(13, 4))
        count = f['contributor_count'][iz].astype(float)
        count[count == 0] = np.nan
        artist = ax.pcolormesh(x, y, count, shading='auto', vmin=1, vmax=max(2, np.nanmax(count)), cmap='viridis')
        ax.set(xlabel='x/D', ylabel='y/D', aspect='equal', title=f"{meta['case']}: contributing volumes at z/D={z[iz]:.3f}; blank = unsupported")
        fig.colorbar(artist, ax=ax, label='Number of contributing volumes')
        fig.tight_layout(); fig.savefig(output / 'coverage_z0.png', dpi=180); plt.close(fig)
        for im in range(min(6, meta['saved_modes'])):
            fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, sharey=True)
            for ax, key in zip(axes, 'uvw'):
                plane = f['mode_' + key][im, iz]
                lim = max(float(np.nanmax(np.abs(plane))) if np.isfinite(plane).any() else 1., 1e-12)
                artist = ax.pcolormesh(x, y, plane, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
                ax.set(ylabel=f'{key}: y/D', aspect='equal')
                fig.colorbar(artist, ax=ax, label='Normalized spatial mode')
            axes[-1].set_xlabel('x/D')
            fig.suptitle(f"{meta['case']}: global mode {im + 1}, {fractions[im]:.2%} phase-locked energy; z/D={z[iz]:.3f}")
            fig.tight_layout(); fig.savefig(output / f'global_mode_{im + 1:02d}_z0.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        coeff = f['phase_coefficients'][:]
        degrees = f['phase_degrees'][:]
        harm = np.abs(np.fft.rfft(coeff, axis=0) / len(degrees))**2
        harm[1:] *= 2
        if len(degrees) % 2 == 0:
            harm[-1] /= 2
        for im in range(min(4, coeff.shape[1])):
            axes[0].plot(np.r_[degrees, degrees[0]+360], np.r_[coeff[:, im], coeff[0, im]], label=f'Mode {im+1}')
            axes[1].plot(np.arange(1, len(harm))*meta['frequency_hz'], harm[1:, im] / f['eigenvalue'][im] * 100, 'o-', label=f'Mode {im+1}')
        axes[0].set(xlabel='Platform phase (degrees)', ylabel='Phase coefficient (velocity units)')
        axes[1].set(xlabel='Harmonic frequency (Hz)', ylabel='Within-mode energy (%)')
        for ax in axes:
            ax.legend(); ax.grid(alpha=.25)
        fig.suptitle('Phase-locked coefficients and discrete harmonics (not broadband spectra)')
        fig.tight_layout(); fig.savefig(output / 'phase_coefficients_and_harmonics.png', dpi=180); plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, sharey=True)
        a, b = f['u_fundamental_cos'][iz], f['u_fundamental_sin'][iz]
        amplitude = np.hypot(a, b)
        angle = np.degrees(np.arctan2(-b, a))
        angle[amplitude < np.nanmax(amplitude) * .01] = np.nan
        for ax, plane, cmap, label, limits in [(axes[0], amplitude, 'magma', 'u harmonic amplitude (velocity units)', {}),
                                               (axes[1], angle, 'twilight', 'Phase of a - i b (degrees)', dict(vmin=-180, vmax=180))]:
            artist = ax.pcolormesh(x, y, plane, cmap=cmap, shading='auto', **limits)
            ax.set(ylabel='y/D', aspect='equal')
            fig.colorbar(artist, ax=ax, label=label)
        axes[-1].set_xlabel('x/D')
        fig.suptitle(f"{meta['case']}: {meta['frequency_hz']:g} Hz phase-locked streamwise response; phase masked below 1% of slice peak")
        fig.tight_layout(); fig.savefig(output / 'fundamental_u_z0.png', dpi=180); plt.close(fig)
