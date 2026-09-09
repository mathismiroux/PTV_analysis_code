"""Compare two saved local POD modes in space, phase and reconstructed motion."""
from pathlib import Path
import argparse
import json
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.pod_spectra import calculate_spectra


def fit_pair(times, coefficients, frequency):
    theta = 2*np.pi*frequency*np.asarray(times)
    design = np.column_stack((np.ones(len(theta)), np.cos(theta), np.sin(theta)))
    beta, _, rank, _ = np.linalg.lstsq(design, coefficients, rcond=None)
    if rank != 3:
        raise ValueError('Cannot resolve sinusoidal fit from these sample times')
    fluct = coefficients-coefficients.mean(axis=0)
    variance = np.mean(fluct**2, axis=0)
    if np.any(variance <= 0):
        raise ValueError('Both modes require nonzero coefficient variance')
    fit = design@beta
    explained = 1-np.mean((coefficients-fit)**2, axis=0)/variance
    phase = np.angle(beta[1]-1j*beta[2])
    difference = np.angle(np.exp(1j*(phase[1]-phase[0])))
    return beta, variance, explained, np.degrees(phase), float(np.degrees(difference))


def weighted_gram(modes, weights):
    # modes=(component,mode,z,y,x); excluded voxels must not enter the integral.
    supported = weights > 0
    if not np.all(np.isfinite(modes[:, :, supported])):
        raise ValueError('Nonfinite modes on positive-weight support')
    contributions = np.einsum('cmv,cnv,v->cmn', modes[:, :, supported], modes[:, :, supported], weights[supported])
    return contributions.sum(axis=0), contributions


def run(source, output, mode_numbers=(1, 2), frequency=2., diameter=1200.,
        z_slices=(-.15, 0., .15), y_slice=.15, cycles=2., frames=80, fps=20,
        start_time=None, animate=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    if (len(mode_numbers) != 2 or len(set(mode_numbers)) != 2 or min(mode_numbers) < 1
            or not np.all(np.isfinite([frequency, diameter, cycles, y_slice, *z_slices]))
            or frequency <= 0 or diameter <= 0 or cycles <= 0 or frames < 2 or fps < 1 or not z_slices):
        raise ValueError('Require two distinct positive mode numbers and valid positive frequency, diameter, duration and animation settings')
    chosen = np.array(mode_numbers)-1
    with h5py.File(source, 'r') as f:
        if max(chosen) >= f['temporal_coefficients'].shape[1]:
            raise ValueError('Requested mode not saved in this POD')
        t = f['t'][:].astype(float)
        # Index individually so reversed mode order is also accepted by h5py.
        coefficients = np.column_stack([f['temporal_coefficients'][:, k] for k in chosen])
        modes = np.stack([np.stack([f['mode_'+key][k] for k in chosen]) for key in 'uvw'])
        weights = f['spatial_weight'][:].astype(float)
        coordinates = {key: f[key][:].astype(float)/diameter for key in 'xyz'}
        fractions = f['energy_fraction'][:][chosen]
        provenance = json.loads(f.attrs.get('metadata_json', '{}'))
    spectra = calculate_spectra(t, coefficients)
    if frequency >= spectra['sample_rate_hz']/2:
        raise ValueError('Fit frequency must be below Nyquist')
    beta, variance, explained, phases, phase_difference = fit_pair(t, coefficients, frequency)
    gram, component_gram = weighted_gram(modes, weights)
    x, y, z = (coordinates[key] for key in 'xyz')
    if any(value < z[0] or value > z[-1] for value in z_slices) or not y[0] <= y_slice <= y[-1]:
        raise ValueError('Requested slice outside measured coordinate range')
    z_indices = [int(np.argmin(abs(z-value))) for value in z_slices]
    iy = int(np.argmin(abs(y-y_slice)))
    iz = int(np.argmin(abs(z)))
    start = float(t[0]) if start_time is None else float(start_time)
    if not np.isfinite(start) or start < t[0] or start >= t[-1]:
        raise ValueError('Animation start time must be within the recording')
    stop = min(float(t[-1]), start+cycles/frequency)
    output.mkdir(parents=True)

    # Shared component scales across both modes AND every selected slice.
    limits = []
    for ic in range(3):
        values = np.concatenate([modes[ic, :, z_indices].ravel(), modes[ic, :, :, iy, :].ravel()])
        limits.append(max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1., 1e-12))
    def spatial_figure(planes, horizontal, vertical, name, label, horizontal_label, vertical_label):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout='constrained')
        for ic, key in enumerate('uvw'):
            for row in range(2):
                artist = axes[row, ic].pcolormesh(horizontal, vertical, planes[ic, row], cmap='RdBu_r',
                                                vmin=-limits[ic], vmax=limits[ic], shading='auto')
                axes[row, ic].set(title=f'Mode {mode_numbers[row]}: {key}', aspect='equal',
                                 xlabel=horizontal_label, ylabel=vertical_label)
            fig.colorbar(artist, ax=axes[:, ic], label='Normalized spatial mode')
        fig.suptitle(f'{source.parent.parent.name}: matched-scale comparison, {label}')
        fig.savefig(output/name, dpi=170); plt.close(fig)
    for i, zi in enumerate(z_indices):
        spatial_figure(modes[:, :, zi], x, y, f'spatial_xy_{i+1:02d}.png', f'z/D={z[zi]:.3f}', 'x/D', 'y/D')
    spatial_figure(modes[:, :, :, iy, :], x, z, 'spatial_xz.png', f'y/D={y[iy]:.3f}', 'x/D', 'z/D')

    phase_degrees = np.degrees(np.mod(2*np.pi*frequency*t, 2*np.pi))
    normalized = (coefficients-coefficients.mean(axis=0))/np.sqrt(variance)
    theta = np.linspace(0, 2*np.pi, 181)
    harmonic_cycle = (np.cos(theta)[:, None]*beta[1]+np.sin(theta)[:, None]*beta[2])/np.sqrt(variance)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    scatter = axes[0].scatter(*normalized.T, c=phase_degrees, cmap='twilight', vmin=0, vmax=360, s=5, alpha=.5)
    axes[1].plot(*harmonic_cycle.T, color='grey')
    axes[1].scatter(*harmonic_cycle.T, c=np.degrees(theta), cmap='twilight', vmin=0, vmax=360, s=12)
    for ax in axes:
        ax.set(xlabel=f'Mode {mode_numbers[0]} coefficient / RMS', ylabel=f'Mode {mode_numbers[1]} coefficient / RMS', aspect='equal')
        ax.grid(alpha=.25)
    axes[0].set_title('Original coefficient fluctuations')
    axes[1].set_title(f'Fitted {frequency:g} Hz components; phase difference {phase_difference:.1f}°')
    fig.colorbar(scatter, ax=axes, label='Nominal platform phase 2πft (degrees)')
    fig.savefig(output/'coefficient_phase_portrait.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
    selection = (t >= start) & (t <= stop)
    for k in range(2):
        axes[0].plot(t[selection], normalized[selection, k], label=f'Mode {mode_numbers[k]}')
        prediction = (np.cos(2*np.pi*frequency*t[selection])*beta[1, k]+np.sin(2*np.pi*frequency*t[selection])*beta[2, k])/np.sqrt(variance[k])
        axes[0].plot(t[selection], prediction, '--', alpha=.8, label=f'Mode {mode_numbers[k]}: {frequency:g} Hz fit')
        psd = spectra['hann'][:, k]
        density = psd/(psd.sum()*spectra['sample_rate_hz']/len(t))
        axes[1].plot(spectra['frequency'], density, label=f'Mode {mode_numbers[k]}')
    axes[0].set(xlabel='Time (s)', ylabel='Coefficient fluctuation / RMS')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Normalized Hann PSD (1/Hz)', xlim=(0, min(4*frequency, spectra['sample_rate_hz']/2)))
    axes[1].axvline(frequency, color='grey', linestyle=':')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output/'coefficients_and_spectra.png', dpi=180); plt.close(fig)

    if animate:
        display_times = np.linspace(start, stop, frames)
        raw_coefficients = np.column_stack([np.interp(display_times, t, coefficients[:, k]) for k in range(2)])
        fitted_coefficients = np.cos(2*np.pi*frequency*display_times)[:, None]*beta[1]+np.sin(2*np.pi*frequency*display_times)[:, None]*beta[2]
        fields = [np.einsum('tm,cmyx->tcyx', c, modes[:, :, iz]) for c in (raw_coefficients, fitted_coefficients)]
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), layout='constrained')
        artists = []
        for ic, key in enumerate('uvw'):
            values = np.stack([field[:, ic] for field in fields])
            limit = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1., 1e-12)
            row = []
            for col in range(2):
                artist = axes[ic, col].pcolormesh(x, y, fields[col][0, ic], shading='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
                axes[ic, col].set(xlabel='x/D', ylabel=f'{key}: y/D', aspect='equal')
                row.append(artist)
            fig.colorbar(row[0], ax=axes[ic], label='Velocity fluctuation (source units)')
            artists.append(row)
        axes[0, 0].set_title('Two-mode reconstruction from original coefficients')
        axes[0, 1].set_title(f'Two-mode reconstruction: fitted {frequency:g} Hz only')
        title = fig.suptitle('')
        def update(frame):
            for ic in range(3):
                for col in range(2):
                    artists[ic][col].set_array(fields[col][frame, ic].ravel())
            title.set_text(f'{source.parent.parent.name}: modes {mode_numbers}; z/D={z[iz]:.3f}; t={display_times[frame]:.3f} s\nMean flow omitted; colour scales fixed throughout animation')
        animation = FuncAnimation(fig, update, frames=frames)
        animation.save(output/'pair_reconstruction.gif', writer=PillowWriter(fps=fps), dpi=90)
        plt.close(fig)
    summary = dict(pod_file=str(source), source_pod_metadata=provenance, modes=list(mode_numbers),
                   mode_energy_fractions=fractions.tolist(), pair_energy_fraction=float(fractions.sum()),
                   frequency_hz=frequency, fitted_within_mode_variance_fractions=explained.tolist(),
                   fitted_coefficient_phases_degrees=phases.tolist(), phase_second_minus_first_degrees=phase_difference,
                   weighted_spatial_gram=gram.tolist(), component_spatial_gram=component_gram.tolist(),
                   actual_z_slices_D=[float(z[i]) for i in z_indices], actual_y_slice_D=float(y[iy]),
                   rotor_diameter=diameter, animation_written=animate,
                   animation_interval_s=[start, stop], animation_frames=frames, playback_fps=fps,
                   interpretation='Phase difference depends on arbitrary POD signs; equivalent oscillatory pairs can differ by 180 degrees. Near-degenerate modes may rotate within their subspace.',
                   phase_reference='Nominal theta=2*pi*f*t; no independently verified platform phase signal',
                   orthogonality='Weighted 3D vector inner products on saved support; slice similarity need not vanish',
                   animation='Two-mode fluctuations, not full flow; raw coefficients linearly sampled for playback; harmonic fit estimated over entire recording')
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pod_file', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', type=int, nargs=2, default=(1, 2))
    parser.add_argument('--frequency-hz', type=float, required=True)
    parser.add_argument('--rotor-diameter', type=float, default=1200.)
    parser.add_argument('--z-slices-d', type=float, nargs='+', default=(-.15, 0., .15))
    parser.add_argument('--y-slice-d', type=float, default=.15)
    parser.add_argument('--cycles', type=float, default=2.)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--start-time', type=float)
    parser.add_argument('--no-animation', action='store_true')
    args = parser.parse_args()
    result = run(args.pod_file, args.output, args.modes, args.frequency_hz, args.rotor_diameter,
                 args.z_slices_d, args.y_slice_d, args.cycles, args.frames, args.fps, args.start_time, not args.no_animation)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
