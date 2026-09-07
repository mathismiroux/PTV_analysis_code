from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Iterable

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


COMPONENTS = ("u", "v", "w")


@dataclass(frozen=True)
class ManifestRow:
    path: str
    status: str
    reason: str
    components_written: str
    components_skipped: str
    grid_shape_z_y_x: str
    n_phase_bins: int | None


def discover_phase_average_files(root_or_file: Path) -> list[Path]:
    if root_or_file.is_file():
        return [root_or_file]
    return sorted(root_or_file.rglob("phase_average.nc"))


def harmonic_r2_chunk(
    phase: np.ndarray,
    phase_mean: np.ndarray,
    offset: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    fit = (
        offset[None, :, :, :]
        + a[None, :, :, :] * np.cos(phase)[:, None, None, None]
        + b[None, :, :, :] * np.sin(phase)[:, None, None, None]
    )
    valid = np.isfinite(phase_mean) & np.isfinite(fit)
    valid_count = valid.sum(axis=0)

    observed_sum = np.where(valid, phase_mean, 0.0).sum(axis=0)
    observed_mean = np.divide(
        observed_sum,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    residual = np.where(valid, phase_mean - fit, 0.0)
    centered = np.where(valid, phase_mean - observed_mean[None, :, :, :], 0.0)
    ss_residual = np.sum(residual * residual, axis=0)
    ss_total = np.sum(centered * centered, axis=0)

    r2 = np.full(valid_count.shape, np.nan, dtype=np.float64)
    accepted = (valid_count >= 3) & (ss_total > 0.0)
    r2[accepted] = 1.0 - ss_residual[accepted] / ss_total[accepted]
    return r2


def component_inputs_exist(h5: h5py.File, component: str) -> bool:
    return all(
        name in h5
        for name in (
            f"{component}_phase_mean",
            f"{component}_harmonic_offset",
            f"{component}_harmonic_a",
            f"{component}_harmonic_b",
        )
    )


def add_component_r2(
    h5: h5py.File,
    component: str,
    z_chunk_size: int,
    overwrite: bool,
) -> str:
    dataset_name = f"{component}_harmonic_r2"
    if dataset_name in h5:
        if not overwrite:
            return "exists"
        del h5[dataset_name]

    phase = h5["phase"][:].astype(np.float64)
    phase_mean_dataset = h5[f"{component}_phase_mean"]
    output = h5.create_dataset(
        dataset_name,
        shape=phase_mean_dataset.shape[1:],
        dtype=np.float64,
        compression="gzip",
        compression_opts=4,
    )
    output.attrs["description"] = (
        "Coefficient of determination for the stored first-harmonic fit "
        "against finite phase-bin means."
    )
    output.attrs["source_phase_mean"] = f"{component}_phase_mean"
    output.attrs["source_fit"] = (
        f"{component}_harmonic_offset + {component}_harmonic_a*cos(phase) + "
        f"{component}_harmonic_b*sin(phase)"
    )

    nz = phase_mean_dataset.shape[1]
    for z_start in range(0, nz, z_chunk_size):
        z_stop = min(z_start + z_chunk_size, nz)
        z_slice = slice(z_start, z_stop)
        output[z_slice, :, :] = harmonic_r2_chunk(
            phase,
            phase_mean_dataset[:, z_slice, :, :].astype(np.float64),
            h5[f"{component}_harmonic_offset"][z_slice, :, :].astype(np.float64),
            h5[f"{component}_harmonic_a"][z_slice, :, :].astype(np.float64),
            h5[f"{component}_harmonic_b"][z_slice, :, :].astype(np.float64),
        )
    return "written"


def process_file(
    path: Path,
    components: Iterable[str],
    z_chunk_size: int,
    overwrite: bool,
    dry_run: bool,
) -> ManifestRow:
    try:
        mode = "r" if dry_run else "r+"
        with h5py.File(path, mode) as h5:
            if "phase" not in h5:
                return ManifestRow(
                    str(path),
                    "skipped",
                    "missing phase dataset",
                    "",
                    ",".join(components),
                    "",
                    None,
                )
            grid_shape = ""
            if "u_phase_mean" in h5:
                grid_shape = "x".join(str(value) for value in h5["u_phase_mean"].shape[1:])
            written = []
            skipped = []
            for component in components:
                if not component_inputs_exist(h5, component):
                    skipped.append(f"{component}:missing-input")
                    continue
                if f"{component}_harmonic_r2" in h5 and not overwrite:
                    skipped.append(f"{component}:exists")
                    continue
                if dry_run:
                    written.append(component)
                else:
                    result = add_component_r2(
                        h5,
                        component,
                        z_chunk_size=z_chunk_size,
                        overwrite=overwrite,
                    )
                    if result == "written":
                        written.append(component)
                    else:
                        skipped.append(f"{component}:{result}")
            status = "processed" if written and not dry_run else "dry-run" if written else "skipped"
            reason = "ok" if written and not dry_run else "would update" if written else "nothing to do"
            return ManifestRow(
                str(path),
                status,
                reason,
                ",".join(written),
                ",".join(skipped),
                grid_shape,
                int(h5["phase"].shape[0]),
            )
    except OSError as exc:
        return ManifestRow(
            str(path),
            "skipped",
            f"cannot open as HDF5/NetCDF: {exc}",
            "",
            ",".join(components),
            "",
            None,
        )


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def parse_components(value: str) -> tuple[str, ...]:
    if value == "all":
        return COMPONENTS
    components = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in components if item not in COMPONENTS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"components must be 'all' or a comma-separated subset of {COMPONENTS}"
        )
    if not components:
        raise argparse.ArgumentTypeError("at least one component is required")
    return components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add u/v/w_harmonic_r2 datasets to existing phase_average.nc files "
            "without rerunning phase averaging."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("phase_average_root_or_file", type=Path)
    parser.add_argument("--components", type=parse_components, default=COMPONENTS)
    parser.add_argument("--z-chunk-size", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing *_harmonic_r2 datasets",
    )
    parser.add_argument("--dry-run", action="store_true", help="show work without editing files")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV manifest path; defaults beside the root/file",
    )
    return parser


def default_manifest_path(root_or_file: Path, dry_run: bool) -> Path:
    name = "harmonic_r2_dry_run_manifest.csv" if dry_run else "harmonic_r2_manifest.csv"
    if root_or_file.is_file():
        return root_or_file.with_name(name)
    return root_or_file / name


def main() -> None:
    args = build_parser().parse_args()
    if args.z_chunk_size <= 0:
        raise SystemExit("--z-chunk-size must be positive")

    paths = discover_phase_average_files(args.phase_average_root_or_file)
    if not paths:
        raise SystemExit(f"No phase_average.nc files found in {args.phase_average_root_or_file}")

    rows = [
        process_file(
            path,
            components=args.components,
            z_chunk_size=args.z_chunk_size,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for path in paths
    ]
    manifest = args.manifest or default_manifest_path(
        args.phase_average_root_or_file,
        args.dry_run,
    )
    write_manifest(manifest, rows)
    processed = sum(row.status == "processed" for row in rows)
    dry_runs = sum(row.status == "dry-run" for row in rows)
    skipped = sum(row.status == "skipped" for row in rows)
    print(
        f"Found {len(paths)} phase_average.nc file(s): "
        f"processed={processed}, dry-run={dry_runs}, skipped={skipped}.",
        flush=True,
    )
    print(f"Wrote manifest: {manifest.resolve()}", flush=True)


if __name__ == "__main__":
    main()
