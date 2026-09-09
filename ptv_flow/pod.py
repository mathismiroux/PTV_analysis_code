"""Volume-weighted, mean-subtracted snapshot POD on fixed spatial support."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .reader import FlowDataset


def cell_widths(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.ones(1)
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError("Coordinates must be strictly monotonic")
    delta = np.abs(delta)
    return np.r_[delta[0], (delta[:-1] + delta[1:]) / 2, delta[-1]]


def decompose(matrix, modes=20, oversampling=15, iterations=2, seed=0, block=4096):
    """matrix is (weighted spatial DOFs, snapshots), possibly a disk memmap.

    Randomized time subspace; exact trace supplies the energy denominator.
    Returns eigenvalues, temporal eigenvectors, total variance, residuals.
    """
    nspace, nt = matrix.shape
    rank = min(modes + oversampling, nt - 1, nspace)
    if modes < 1 or rank < 1 or iterations < 0 or oversampling < 0:
        raise ValueError("Invalid POD rank or iteration settings")
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((nt, rank)), mode="reduced")[0]
    total = 0.0
    for iteration in range(iterations + 1):
        y = np.zeros_like(q)
        for start in range(0, nspace, block):
            a = np.asarray(matrix[start:start + block], dtype=float)
            y += a.T @ (a @ q)
            if iteration == 0:
                total += float(np.sum(a * a)) / nt
        q = np.linalg.qr(y, mode="reduced")[0]
    if total <= 0 or not np.isfinite(total):
        raise ValueError("No finite nonzero fluctuation energy on retained support")
    c = np.zeros((rank, rank))
    for start in range(0, nspace, block):
        b = np.asarray(matrix[start:start + block], dtype=float) @ q
        c += b.T @ b / nt
    eigenvalues, rotation = np.linalg.eigh(c)
    order = np.argsort(eigenvalues)[::-1]
    order = order[eigenvalues[order] > total * 1e-12][:modes]
    eigenvalues = eigenvalues[order]
    temporal = q @ rotation[:, order]
    action = np.zeros_like(temporal)
    for start in range(0, nspace, block):
        a = np.asarray(matrix[start:start + block], dtype=float)
        action += a.T @ (a @ temporal) / nt
    residuals = np.linalg.norm(action - temporal * eigenvalues, axis=0) / eigenvalues
    return eigenvalues, temporal, total, residuals


def run_pod(source, output, modes=20, min_valid_fraction=1.0,
            exclude_filled=False, zero_invalid=False, iterations=2, seed=0,
            oversampling=15, mean_file=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not 0 < min_valid_fraction <= 1 or modes < 1:
        raise ValueError("Require modes >= 1 and 0 < min_valid_fraction <= 1")
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    with FlowDataset(source) as flow:
        nt, nz, ny, nx = flow.shape
        if nt < 2:
            raise ValueError("POD requires at least two snapshots")
        coords = {k: flow.coordinate(k) for k in ("t", "z", "y", "x")}
    mean_path = Path(mean_file).resolve() if mean_file else source.parent / 'mean.nc'
    existing_mean = None
    if mean_path.exists():
        if exclude_filled or zero_invalid:
            raise ValueError("Existing mean reuse currently requires default finite-vector validity settings")
        with h5py.File(mean_path, 'r') as saved:
            for key, values in coords.items():
                if key not in saved or not np.array_equal(saved[key][:], values):
                    raise ValueError(f"Existing mean {key} coordinates do not match velocity")
            if saved.attrs.get('invalid_samples', 'nan') != 'nan':
                raise ValueError("Existing mean validity policy must be nan")
            recorded_source = saved.attrs.get('source_file')
            if recorded_source and Path(recorded_source).resolve() != source:
                raise ValueError("Existing mean source_file does not match velocity file")
            existing_mean = np.stack([saved[k + '_mean'][:] for k in 'uvw']).astype(float)
            if existing_mean.shape != (3, nz, ny, nx):
                raise ValueError("Existing mean shape does not match velocity grid")
        print(f"Reusing mean: {mean_path}", flush=True)
    elif mean_file:
        raise FileNotFoundError(mean_path)
    output.mkdir(parents=True)
    nvox = nz * ny * nx
    volumes = (cell_widths(coords['z'])[:, None, None]
               * cell_widths(coords['y'])[None, :, None]
               * cell_widths(coords['x'])[None, None, :])
    valid_fraction = np.zeros((nz, ny, nx))
    support = np.zeros((nz, ny, nx), dtype=bool)
    mean = np.full((3, nz, ny, nx), np.nan)
    print(f"Reading {nt} snapshots; building fixed spatial support", flush=True)
    with tempfile.TemporaryDirectory(prefix="pod_", dir=output) as scratch:
        matrix = np.memmap(Path(scratch) / 'fluctuations.dat', mode='w+',
                           dtype='float32', shape=(3 * nvox, nt))
        try:
            with h5py.File(source, 'r') as src:
                for iz in range(nz):
                    data = np.stack([src[k][:, iz].astype(float) for k in 'uvw'])
                    finite = np.all(np.isfinite(data), axis=0)
                    for ic, k in enumerate('uvw'):
                        for attr in ('_FillValue', 'missing_value'):
                            if attr in src[k].attrs:
                                for value in np.asarray(src[k].attrs[attr]).ravel():
                                    finite &= data[ic] != value
                    if zero_invalid:
                        finite &= np.any(data != 0, axis=0)
                    if exclude_filled:
                        if 'filled_mask' not in src:
                            raise ValueError("--exclude-filled requires filled_mask")
                        finite &= src['filled_mask'][:, iz] == 0
                    count = finite.sum(axis=0)
                    valid_fraction[iz] = count / nt
                    keep = (count >= np.ceil(min_valid_fraction * nt)) & (count >= 2)
                    support[iz] = keep
                    if existing_mean is not None:
                        avg = existing_mean[:, iz]
                        keep &= np.all(np.isfinite(avg), axis=0)
                        support[iz] = keep
                    else:
                        avg = np.divide(np.where(finite[None], data, 0).sum(axis=1),
                                        count[None], out=np.zeros((3, ny, nx)),
                                        where=count[None] > 0)
                    mean[:, iz] = np.where(keep[None], avg, np.nan)
                    fluct = np.where(finite[None] & keep[None, None],
                                     data - avg[:, None], 0)
                    for ic in range(3):
                        offset = ic * nvox + iz * ny * nx
                        matrix[offset:offset + ny * nx] = fluct[ic].reshape(nt, -1).T
                    if (iz + 1) % 5 == 0:
                        print(f"  read {iz + 1}/{nz} z planes", flush=True)
            if not support.any():
                raise ValueError("No common valid voxels. Consider interpolating first or explicitly lowering --min-valid-fraction.")
            weights = np.where(support, volumes / volumes[support].sum(), 0)
            sqrt_weights = np.sqrt(np.tile(weights.ravel(), 3))
            for start in range(0, 3 * nvox, 4096):
                matrix[start:start + 4096] *= sqrt_weights[start:start + 4096, None]
            print(f"Retained {support.sum()}/{nvox} voxels; solving POD", flush=True)
            ev, temporal, total, residuals = decompose(
                matrix, modes, oversampling, iterations, seed)
            nm = len(ev)
            spatial = np.full((3 * nvox, nm), np.nan)
            for start in range(0, 3 * nvox, 4096):
                stop = min(start + 4096, 3 * nvox)
                b = np.asarray(matrix[start:stop], dtype=float) @ temporal
                denom = sqrt_weights[start:stop, None] * np.sqrt(nt * ev)[None]
                np.divide(b, denom, out=spatial[start:stop], where=denom > 0)
            coefficients = temporal * np.sqrt(nt * ev)
            for im in range(nm):
                pivot = np.nanargmax(np.abs(spatial[:, im]))
                if spatial[pivot, im] < 0:
                    spatial[:, im] *= -1
                    coefficients[:, im] *= -1
        finally:
            matrix.flush()
            matrix._mmap.close()
    fractions = ev / total
    metadata = dict(source=str(source), snapshots=nt, retained_voxels=int(support.sum()),
                    mean_file=str(mean_path) if existing_mean is not None else None,
                    total_voxels=nvox, spatial_coverage=float(support.mean()),
                    min_valid_fraction=min_valid_fraction, exclude_filled=exclude_filled,
                    zero_invalid=zero_invalid, method='randomized snapshot POD',
                    mean_subtracted=True, spatial_weighting='cell volume / retained volume',
                    time_weighting='equal snapshot weights', missing_policy='voxel temporal mean imputation after support selection',
                    energy_definition='0.5 * spatial mean of u_prime^2 + v_prime^2 + w_prime^2',
                    total_fluctuation_energy=total / 2, saved_modes=nm,
                    saved_energy_fraction=float(fractions.sum()), seed=seed,
                    iterations=iterations, oversampling=oversampling,
                    max_relative_eigen_residual=float(residuals.max()))
    with h5py.File(output / 'pod.h5', 'w') as dst:
        dst.attrs['metadata_json'] = json.dumps(metadata)
        for key, value in coords.items():
            dst[key] = value
        for key, value in dict(support=support, valid_fraction=valid_fraction,
                               spatial_weight=weights, eigenvalue=ev,
                               energy_fraction=fractions, cumulative_energy_fraction=np.cumsum(fractions),
                               temporal_coefficients=coefficients,
                               relative_eigen_residual=residuals).items():
            dst[key] = value
        for ic, key in enumerate('uvw'):
            dst.create_dataset('mean_' + key, data=mean[ic], compression='gzip')
            dst.create_dataset('mode_' + key,
                               data=spatial.reshape(3, nz, ny, nx, nm)[ic].transpose(3, 0, 1, 2),
                               compression='gzip')
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (output / 'modal_energy.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['mode', 'eigenvalue', 'energy', 'energy_fraction', 'cumulative_energy_fraction', 'relative_eigen_residual'])
        for i in range(nm):
            writer.writerow([i + 1, ev[i], ev[i] / 2, fractions[i], fractions[:i + 1].sum(), residuals[i]])
    plot_pod(output)
    return metadata


def plot_pod(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with h5py.File(Path(output) / 'pod.h5') as data:
        fraction = data['energy_fraction'][:]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        indices = np.arange(1, len(fraction) + 1)
        axes[0].bar(indices, fraction * 100)
        axes[0].set(xlabel='Mode', ylabel='Fluctuation energy (%)')
        axes[1].plot(indices, np.cumsum(fraction) * 100, 'o-')
        axes[1].set(xlabel='Number of modes', ylabel='Cumulative energy (%)', ylim=(0, 100))
        fig.tight_layout()
        fig.savefig(Path(output) / 'modal_energy.png', dpi=180)
        plt.close(fig)
        z = data['z'][:]
        iz = int(np.argmin(np.abs(z)))
        for im in range(min(6, len(fraction))):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
            for ax, key in zip(axes, 'uvw'):
                plane = data['mode_' + key][im, iz]
                lim = np.nanmax(np.abs(plane)) if np.isfinite(plane).any() else 1
                lim = max(lim, 1e-12)
                artist = ax.pcolormesh(data['x'][:], data['y'][:], plane,
                                       cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
                ax.set(title=key, xlabel='x (source units)', aspect='equal')
                fig.colorbar(artist, ax=ax)
            axes[0].set_ylabel('y (source units)')
            fig.suptitle(f'Mode {im + 1}: {fraction[im]:.2%} energy; z={z[iz]:g} (source units)')
            fig.tight_layout()
            fig.savefig(Path(output) / f'mode_{im + 1:02d}_z0.png', dpi=180)
            plt.close(fig)
