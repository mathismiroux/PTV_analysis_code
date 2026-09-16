"""Calculate temporal turbulence intensity for one interpolated velocity volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(r"D:\binning64voxel50overlap\outputs_flow\Flow_3.5D\interpolated_velocity.nc")


def statistics_from_mean(flow, mean_file, chunk_size=32):
    """Compute RMS about the saved mean using its finite-sample convention."""
    if mean_file.attrs.get("invalid_samples", "nan") != "nan":
        raise ValueError("Mean file must use invalid_samples='nan'")
    u = flow["u"]
    mean = np.asarray(mean_file["u_mean"][:], dtype=float)
    if mean.shape != u.shape[1:]:
        raise ValueError("Mean and velocity grid shapes differ")
    for key in ("x", "y", "z", "t"):
        if not np.array_equal(mean_file[key][:], flow[key][:]):
            raise ValueError(f"Mean and velocity {key} coordinates differ")
    vector = mean_file.attrs.get("zero_mask", "component") == "vector"
    keys = ("u", "v", "w") if vector else ("u",)
    count_key = "vector_count" if vector else "u_count"
    count = np.zeros(mean.shape, dtype=np.int64)
    squared = np.zeros(mean.shape)
    for start in range(0, u.shape[0], chunk_size):
        values = {}
        for key in keys:
            dataset = flow[key]
            block = np.asarray(dataset[start:start + chunk_size], dtype=float)
            for attr in ("_FillValue", "missing_value"):
                for fill in np.asarray(dataset.attrs.get(attr, [])).ravel():
                    block[block == fill] = np.nan
            values[key] = block
        valid = np.logical_and.reduce([np.isfinite(block) for block in values.values()])
        count += valid.sum(axis=0)
        squared += np.where(valid, (values["u"] - mean) ** 2, 0).sum(axis=0)
    if count_key in mean_file and not np.array_equal(count, mean_file[count_key][:]):
        raise ValueError("Sample counts differ from mean file; check source and masking")
    rms = np.sqrt(np.divide(squared, count, out=np.full(mean.shape, np.nan), where=count > 0))
    return mean, rms, count


def temporal_statistics(dataset, chunk_size=32):
    """Merge block moments in float64; omit nonfinite and declared fill values."""
    shape = dataset.shape[1:]
    count = np.zeros(shape, dtype=np.int64)
    mean = np.zeros(shape)
    m2 = np.zeros(shape)
    for start in range(0, dataset.shape[0], chunk_size):
        values = np.asarray(dataset[start:start + chunk_size], dtype=np.float64)
        valid = np.isfinite(values)
        for attr in ("_FillValue", "missing_value"):
            for fill in np.asarray(dataset.attrs.get(attr, [])).ravel():
                valid &= values != fill
        n = valid.sum(axis=0)
        block_mean = np.divide(np.where(valid, values, 0).sum(axis=0), n,
                               out=np.zeros(shape), where=n > 0)
        deviations = np.where(valid, values - block_mean, 0)
        block_m2 = np.sum(deviations ** 2, axis=0)
        total = count + n
        weight = np.divide(n, total, out=np.zeros(shape), where=total > 0)
        delta = block_mean - mean
        m2 += block_m2 + delta ** 2 * count * weight
        mean += delta * weight
        count = total
    variance = np.divide(m2, count, out=np.full(shape, np.nan), where=count > 0)
    mean[count == 0] = np.nan
    return mean, np.sqrt(np.maximum(variance, 0)), count


def intensity(mean, rms, count, n_times, reference_velocity=None,
              min_valid_fraction=0.8, min_mean_speed=0.01):
    denominator = np.abs(mean) if reference_velocity is None else np.full(mean.shape, reference_velocity)
    valid = (count >= 2) & (count / n_times >= min_valid_fraction)
    valid &= np.isfinite(denominator) & (denominator > min_mean_speed)
    return np.divide(100 * rms, denominator, out=np.full(mean.shape, np.nan), where=valid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="One (time,z,y,x) velocity file, or its containing folder")
    parser.add_argument("--output-folder", type=Path, help="New output directory; default: input folder/turbulence_intensity")
    parser.add_argument("--mean-file", type=Path, help="Saved mean volume; defaults to mean.nc beside the input when present")
    parser.add_argument("--alongside-mean", action="store_true", help="Save TI files directly beside mean.nc; refuse existing TI files")
    parser.add_argument("--reference-velocity", type=float, help="Fixed normalization velocity in input velocity units; otherwise use local |mean(u)|")
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--min-mean-speed", type=float, default=0.01, help="Mask normalization speeds at or below this value, in input velocity units")
    parser.add_argument("--chunk-size", type=int, default=32, help="Frames read at a time")
    parser.add_argument("--coordinate-unit", default="mm", help="Coordinate unit label (no conversion)")
    parser.add_argument("--vmax", type=float, help="Upper color limit in percent; default: maximum finite TI")
    args = parser.parse_args(argv)
    if not (0 <= args.min_valid_fraction <= 1) or not np.isfinite(args.min_mean_speed) or args.min_mean_speed < 0 or args.chunk_size < 1:
        parser.error("Require coverage in [0,1], finite nonnegative speed threshold, and positive chunk size")
    if args.reference_velocity is not None and (not np.isfinite(args.reference_velocity) or args.reference_velocity <= args.min_mean_speed):
        parser.error("Reference velocity must be finite and above --min-mean-speed")
    if args.vmax is not None and (not np.isfinite(args.vmax) or args.vmax <= 0):
        parser.error("--vmax must be finite and positive")
    source = args.input / "interpolated_velocity.nc" if args.input.is_dir() else args.input
    mean_path = args.mean_file or source.parent / "mean.nc"
    use_mean = args.mean_file is not None or mean_path.exists()
    if args.alongside_mean and args.output_folder:
        parser.error("--alongside-mean cannot be combined with --output-folder")
    output = mean_path.parent if args.alongside_mean else (args.output_folder or source.parent / "turbulence_intensity")
    metadata_name = "turbulence_intensity_manifest.json" if args.alongside_mean else "manifest.json"
    if args.alongside_mean:
        for name in ("turbulence_intensity.npz", "turbulence_intensity.png", metadata_name):
            if (output / name).exists():
                parser.error(f"TI output already exists: {output / name}")
    elif output.exists():
        parser.error(f"Output folder already exists: {output}; choose a new --output-folder")
    with h5py.File(source, "r") as f:
        axes = {k: np.asarray(f[k][:], dtype=float) for k in ("x", "y", "z")}
        if any(a.ndim != 1 or len(a) < 1 or not np.all(np.isfinite(a)) for a in axes.values()):
            parser.error("Coordinates must be finite, nonempty 1D arrays")
        u = f["u"]
        if u.ndim != 4 or u.shape[1:] != tuple(len(axes[k]) for k in ("z", "y", "x")) or u.shape[0] < 2:
            parser.error("Expected u(time,z,y,x) with at least two frames and matching coordinates")
        n_times = u.shape[0]
        print(f"Reading {source}: {n_times} frames, grid {u.shape[1:]}", flush=True)
        if use_mean:
            print(f"Using saved mean: {mean_path}", flush=True)
            with h5py.File(mean_path, "r") as mean_file:
                mean, rms, count = statistics_from_mean(f, mean_file, args.chunk_size)
        else:
            mean, rms, count = temporal_statistics(u, args.chunk_size)
    ti = intensity(mean, rms, count, n_times, args.reference_velocity,
                   args.min_valid_fraction, args.min_mean_speed)
    output.mkdir(parents=True, exist_ok=args.alongside_mean)
    np.savez_compressed(output / "turbulence_intensity.npz", **axes,
                        ti_percent=ti, u_mean=mean, u_rms=rms, valid_count=count,
                        valid_fraction=count / n_times)
    iz, iy, ix = (n // 2 for n in ti.shape)
    planes = [("z", "y", ti[:, :, ix].T, f"x = {axes['x'][ix]:g}"),
              ("x", "y", ti[iz], f"z = {axes['z'][iz]:g}"),
              ("x", "z", ti[:, iy, :], f"y = {axes['y'][iy]:g}")]
    finite = ti[np.isfinite(ti)]
    vmax = args.vmax or (max(float(finite.max()), 1e-6) if finite.size else 1)
    fig, panels = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (horizontal, vertical, values, title) in zip(panels, planes):
        plot = ax.pcolormesh(axes[horizontal], axes[vertical], values,
                             shading="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set(xlabel=f"{horizontal} [{args.coordinate_unit}]", ylabel=f"{vertical} [{args.coordinate_unit}]",
               title=f"{title} {args.coordinate_unit}", aspect="equal")
        ax.set_facecolor("#dddddd")
    normalization = "local |mean(u)|" if args.reference_velocity is None else f"Uref={args.reference_velocity:g}"
    fig.suptitle(f"{source.parent.name}: streamwise turbulence intensity\n100 std(u) / {normalization}; coverage ≥ {args.min_valid_fraction:.0%}")
    fig.colorbar(plot, ax=panels, label="TI [%]", shrink=0.8)
    fig.savefig(output / "turbulence_intensity.png", dpi=200)
    plt.close(fig)
    metadata = dict(source=str(source.resolve()), n_times=n_times, component="u", ddof=0,
                    mean_source=str(mean_path.resolve()) if use_mean else "computed from time series",
                    formula="100 * sqrt(mean((u - mean(u))**2)) / normalization",
                    normalization=normalization, reference_velocity=args.reference_velocity,
                    min_valid_fraction=args.min_valid_fraction, min_mean_speed=args.min_mean_speed,
                    retained_voxels=int(finite.size), total_voxels=int(ti.size),
                    plane_indices=dict(x=ix, y=iy, z=iz), coordinate_unit=args.coordinate_unit,
                    notes="Interpolated samples included; zeros retained. Saved mean masking convention used when available, with sample counts checked. Nonfinite and declared fill values excluded. Population RMS about the mean; at least two samples required. Gray indicates masked values. Fluctuations include any coherent unsteadiness.")
    (output / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved maps and full 3D field to {output.resolve()} ({finite.size}/{ti.size} valid voxels)")


if __name__ == "__main__":
    main()
