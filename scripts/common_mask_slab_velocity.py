"""Recompute slab velocities on the intersection of all cases' valid voxels."""
import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np


def align_axis(values, valid, source, target, axis):
    """Linear alignment; every contributing voxel must pass the original mask."""
    if np.array_equal(source, target):
        return values, valid
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("Expected increasing coordinates with at least two points")
    if target.min() < source.min() or target.max() > source.max():
        raise ValueError("Extrapolation is forbidden")
    hi = np.clip(np.searchsorted(source, target, side="right"), 1, len(source)-1)
    lo = hi-1
    weight = (target-source[lo])/(source[hi]-source[lo])
    shape = [1]*values.ndim
    shape[axis] = len(target)
    weight = weight.reshape(shape)
    left, right = np.take(values, lo, axis=axis), np.take(values, hi, axis=axis)
    left_valid, right_valid = np.take(valid, lo, axis=axis), np.take(valid, hi, axis=axis)
    result = np.where(weight == 0, left, np.where(weight == 1, right, (1-weight)*left+weight*right))
    mask = ((weight == 1) | left_valid) & ((weight == 0) | right_valid) & np.isfinite(result)
    return result, mask


def discover_sources(folder):
    groups = {}
    for path in sorted(folder.glob("*/slab_vertical_velocity.csv")):
        metadata = json.loads((path.parent / "manifest.json").read_text())
        for source in metadata["sources"]:
            p = Path(source["source"])
            match = re.fullmatch(r"(.+)_(\d+(?:\.\d+)?)D", p.parent.name)
            if not match:
                raise ValueError(f"Cannot identify volume {p}")
            case, distance = match[1], float(match[2])
            previous = groups.setdefault(distance, {}).get(case)
            if previous is not None and previous.resolve() != p.resolve():
                raise ValueError(f"Multiple sources for {case} {distance}D")
            groups[distance][case] = p
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Figures folder with slab export manifests")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--quantity", choices=("u", "v", "w"), default="v")
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--rotor-diameter", type=float, default=1200)
    parser.add_argument("--reference-case", default="Flow")
    args = parser.parse_args()
    if not 0 <= args.min_valid_fraction < 1 or args.rotor_diameter <= 0:
        parser.error("Invalid coverage threshold or diameter")
    groups = discover_sources(args.input)
    if not groups:
        raise ValueError("No source mean files discovered")
    cases = sorted(set.union(*(set(g) for g in groups.values())))
    for distance, group in groups.items():
        if set(group) != set(cases) or args.reference_case not in group:
            raise ValueError(f"Missing cases at {distance}D; require all {cases}")
    args.output_folder.mkdir(parents=True, exist_ok=False)
    rows_by_case = {c: [] for c in cases}
    sources_by_case = {c: [] for c in cases}
    audits, mask_arrays = [], {}
    for distance, group in sorted(groups.items()):
        loaded = {}
        for case, path in group.items():
            with h5py.File(path) as f:
                coords = [f[a][:].astype(float) for a in ("z", "y", "x")]
                values = f[f"{args.quantity}_mean"][:].astype(float)
                name = "vector_count" if "vector_count" in f else f"{args.quantity}_count"
                n = int(f.attrs["input_shape_time_z_y_x"][0]) if "input_shape_time_z_y_x" in f.attrs else len(f["t"])
                counts = f[name][:]
                if n <= 0 or values.shape != counts.shape or values.shape != tuple(map(len, coords)):
                    raise ValueError(f"Invalid dimensions/counts: {path}")
                valid = np.isfinite(values) & (counts/n > args.min_valid_fraction)
            loaded[case] = (coords, values, valid)
            sources_by_case[case].append(dict(source=str(path.resolve()), count_dataset=name, total_samples=n))
        reference = loaded[args.reference_case][0]
        target = []
        for axis in range(3):
            lower = max(v[0][axis].min() for v in loaded.values())
            upper = min(v[0][axis].max() for v in loaded.values())
            target.append(reference[axis][(reference[axis] >= lower) & (reference[axis] <= upper)])
            if not target[-1].size:
                raise ValueError(f"No common spatial extent at {distance}D")
        aligned, masks, alignment = {}, {}, {}
        for case, (coords, values, valid) in loaded.items():
            alignment[case] = []
            for axis, (source, dest) in enumerate(zip(coords, target)):
                if not np.array_equal(source, dest):
                    alignment[case].append("zyx"[axis])
                values, valid = align_axis(values, valid, source, dest, axis)
            aligned[case], masks[case] = values, valid
        common = np.logical_and.reduce([masks[c] for c in cases])
        retained = common.sum(axis=(0, 1))
        slab_size = common.shape[0]*common.shape[1]
        x = target[2]
        for case in cases:
            sums = np.where(common, aligned[case], 0).sum(axis=(0, 1))
            means = np.divide(sums, retained, out=np.full(sums.shape, np.nan), where=retained > 0)
            for xi, mean, count in zip(x, means, retained):
                rows_by_case[case].append(dict(volume=f"{case}_{distance:g}D", x=xi,
                    x_over_d=xi/args.rotor_diameter, **{f"slab_mean_{args.quantity}_m_s": mean},
                    retained_voxels=int(count), total_slab_voxels=slab_size,
                    retained_slab_fraction=float(count/slab_size)))
        key = f"volume_{distance:g}D"
        mask_arrays[key+"_mask"] = common
        for a, coord in zip("zyx", target):
            mask_arrays[key+"_"+a] = coord
        central = (x >= x.min()+.2*np.ptp(x)) & (x <= x.max()-.2*np.ptp(x))
        audit = dict(distance=distance, grid_shape=list(common.shape),
                     aligned_axes=alignment, common_voxels=int(common.sum()),
                     individual_valid_voxels={c: int(masks[c].sum()) for c in cases},
                     central60_min_voxels_per_slab=int(retained[central].min()),
                     central60_max_voxels_per_slab=int(retained[central].max()),
                     central60_empty_slabs=int((retained[central]==0).sum()))
        audits.append(audit)
        print(json.dumps(audit))
    direction = {"u": "streamwise", "v": "vertical", "w": "lateral"}[args.quantity]
    method = ("Strict temporal coverage plus finite component mean, intersected across all cases at each distance. "
              "Flow reference grid restricted to common physical extent. Linear coordinate alignment only where grids differ; "
              "all contributing source voxels must pass coverage and finite-value masks. No extrapolation. "
              "Equal voxel weights; shared mask per slab; empty slabs are NaN.")
    for case, rows in rows_by_case.items():
        folder = args.output_folder / case
        folder.mkdir()
        with (folder / f"slab_{direction}_velocity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (folder / "manifest.json").write_text(json.dumps(dict(
            min_valid_fraction=args.min_valid_fraction, comparison="strictly greater than",
            rotor_diameter=args.rotor_diameter, quantity=args.quantity, sources=sources_by_case[case],
            common_mask_cases=cases, method=method), indent=2))
    np.savez_compressed(args.output_folder / "common_masks.npz", **mask_arrays)
    (args.output_folder / "manifest.json").write_text(json.dumps(dict(
        cases=cases, quantity=args.quantity, threshold=args.min_valid_fraction,
        reference_case=args.reference_case, volumes=audits, method=method), indent=2))


if __name__ == "__main__":
    main()
