from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.postprocess import temporal_average_volume
from ptv_flow.reader import FlowDataset
from ptv_flow.validity import INVALID_SAMPLE_MODES


REQUIRED_RAW_DATASETS = ("t", "x", "y", "z", "u", "v", "w")


@dataclass(frozen=True)
class MeanWakeSettings:
    input_folder: str
    output_root: str
    pattern: str
    invalid_samples: str
    zero_mask: str
    min_valid_fraction: float
    chunk_size: int
    u_inf: float | None
    overwrite: bool
    dry_run: bool
    command_line: str


@dataclass(frozen=True)
class ManifestRow:
    source_file: str
    output_file: str
    status: str
    reason: str
    file_size_bytes: int
    n_times: int | None
    nz: int | None
    ny: int | None
    nx: int | None
    invalid_samples: str
    zero_mask: str
    min_valid_fraction: float
    chunk_size: int
    u_inf: float | None


def _safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "case"


def default_output_root(input_folder: Path, output_name: str) -> Path:
    return input_folder.resolve().parent / "outputs" / output_name


def resolve_output_root(input_folder: Path, args: argparse.Namespace) -> Path:
    if args.output_root is None and args.output_name is None:
        raise SystemExit(
            "This analysis requires a new output folder name. "
            "Pass --output-name NAME, for example --output-name mean_wake_figure1."
        )
    if args.output_root is not None:
        return args.output_root.resolve()
    return default_output_root(input_folder, args.output_name)


def refuse_existing_output_root(output_root: Path) -> None:
    if output_root.exists():
        raise SystemExit(
            f"Refusing to use existing output folder: {output_root}. "
            "Choose a new --output-name or remove/archive the folder yourself."
        )


def is_raw_velocity_timeseries(path: Path) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as h5:
            missing = [name for name in REQUIRED_RAW_DATASETS if name not in h5]
            if missing:
                return False, f"missing datasets: {', '.join(missing)}"
            u_shape = h5["u"].shape
            if len(u_shape) != 4:
                return False, f"u is not 4D time,z,y,x; shape={u_shape}"
            for component in ("v", "w"):
                if h5[component].shape != u_shape:
                    return False, f"{component} shape {h5[component].shape} != u shape {u_shape}"
            if h5["t"].shape[0] != u_shape[0]:
                return False, "t length does not match velocity time dimension"
            if h5["z"].shape[0] != u_shape[1]:
                return False, "z length does not match velocity z dimension"
            if h5["y"].shape[0] != u_shape[2]:
                return False, "y length does not match velocity y dimension"
            if h5["x"].shape[0] != u_shape[3]:
                return False, "x length does not match velocity x dimension"
    except OSError as exc:
        return False, f"cannot open as HDF5/NetCDF: {exc}"
    return True, "ok"


def _raw_shape(path: Path) -> tuple[int, int, int, int] | tuple[None, None, None, None]:
    try:
        with h5py.File(path, "r") as h5:
            shape = h5["u"].shape
            return int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    except Exception:
        return None, None, None, None


def discover_input_files(input_folder: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_folder.glob(pattern) if path.is_file())


def _metadata(settings: MeanWakeSettings, source_file: Path) -> dict[str, str | float]:
    values: dict[str, str | float] = {
        "script_name": Path(__file__).name,
        "command_line": settings.command_line,
        "batch_input_folder": settings.input_folder,
        "batch_output_root": settings.output_root,
        "batch_pattern": settings.pattern,
        "case_id": _safe_name(source_file),
        "label": source_file.stem,
    }
    if settings.u_inf is not None:
        values["u_inf"] = float(settings.u_inf)
    return values


def process_file(
    source: Path,
    output: Path,
    settings: MeanWakeSettings,
) -> ManifestRow:
    valid, reason = is_raw_velocity_timeseries(source)
    n_times, nz, ny, nx = _raw_shape(source)
    if not valid:
        return ManifestRow(
            source_file=str(source),
            output_file=str(output),
            status="skipped",
            reason=reason,
            file_size_bytes=source.stat().st_size,
            n_times=n_times,
            nz=nz,
            ny=ny,
            nx=nx,
            invalid_samples=settings.invalid_samples,
            zero_mask=settings.zero_mask,
            min_valid_fraction=settings.min_valid_fraction,
            chunk_size=settings.chunk_size,
            u_inf=settings.u_inf,
        )
    if output.exists() and not settings.overwrite:
        return ManifestRow(
            source_file=str(source),
            output_file=str(output),
            status="exists",
            reason="output exists; pass --overwrite to replace",
            file_size_bytes=source.stat().st_size,
            n_times=n_times,
            nz=nz,
            ny=ny,
            nx=nx,
            invalid_samples=settings.invalid_samples,
            zero_mask=settings.zero_mask,
            min_valid_fraction=settings.min_valid_fraction,
            chunk_size=settings.chunk_size,
            u_inf=settings.u_inf,
        )
    if settings.dry_run:
        return ManifestRow(
            source_file=str(source),
            output_file=str(output),
            status="dry-run",
            reason="would process",
            file_size_bytes=source.stat().st_size,
            n_times=n_times,
            nz=nz,
            ny=ny,
            nx=nx,
            invalid_samples=settings.invalid_samples,
            zero_mask=settings.zero_mask,
            min_valid_fraction=settings.min_valid_fraction,
            chunk_size=settings.chunk_size,
            u_inf=settings.u_inf,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with FlowDataset(source) as flow:
        temporal_average_volume(
            flow,
            output=output,
            chunk_size=settings.chunk_size,
            zero_mask=settings.zero_mask,
            min_valid_fraction=settings.min_valid_fraction,
            overwrite=settings.overwrite,
            u_inf=settings.u_inf,
            metadata=_metadata(settings, source),
            invalid_samples=settings.invalid_samples,
        )
    return ManifestRow(
        source_file=str(source),
        output_file=str(output),
        status="processed",
        reason="ok",
        file_size_bytes=source.stat().st_size,
        n_times=n_times,
        nz=nz,
        ny=ny,
        nx=nx,
        invalid_samples=settings.invalid_samples,
        zero_mask=settings.zero_mask,
        min_valid_fraction=settings.min_valid_fraction,
        chunk_size=settings.chunk_size,
        u_inf=settings.u_inf,
    )


def write_manifest(output_root: Path, settings: MeanWakeSettings, rows: Iterable[ManifestRow]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    csv_path = output_root / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        if row_list:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(row_list[0])))
            writer.writeheader()
            for row in row_list:
                writer.writerow(asdict(row))
    json_path = output_root / "manifest.json"
    json_path.write_text(
        json.dumps(
            {
                "settings": asdict(settings),
                "files": [asdict(row) for row in row_list],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote manifest: {csv_path.resolve()}", flush=True)
    print(f"Wrote manifest: {json_path.resolve()}", flush=True)


def prepare_mean_wake_products(args: argparse.Namespace) -> list[ManifestRow]:
    input_folder = args.input_folder.resolve()
    output_root = resolve_output_root(input_folder, args)
    refuse_existing_output_root(output_root)
    settings = MeanWakeSettings(
        input_folder=str(input_folder),
        output_root=str(output_root),
        pattern=args.pattern,
        invalid_samples=args.invalid_samples,
        zero_mask=args.zero_mask,
        min_valid_fraction=args.min_valid_fraction,
        chunk_size=args.chunk_size,
        u_inf=args.u_inf,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        command_line=" ".join(sys.argv),
    )
    files = discover_input_files(input_folder, args.pattern)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No files matched {args.pattern!r} in {input_folder}")

    print(f"Input folder: {input_folder}", flush=True)
    print(f"Output root:  {output_root}", flush=True)
    print(f"Files found:  {len(files)}", flush=True)
    print(f"Dry run:      {args.dry_run}", flush=True)

    rows = []
    for source in files:
        case_dir = output_root / _safe_name(source)
        output = case_dir / "mean.nc"
        print(f"\n{source.name} -> {output}", flush=True)
        row = process_file(source, output, settings)
        print(f"  {row.status}: {row.reason}", flush=True)
        rows.append(row)
    write_manifest(output_root, settings, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare reusable 3D mean wake products for every raw velocity "
            "NetCDF file in a folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_folder", type=Path, help="folder containing raw .nc files")
    parser.add_argument("--pattern", default="*.nc", help="glob pattern for input files")
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "new folder name under ../outputs relative to input_folder; "
            "required unless --output-root is used"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "explicit new output folder; required unless --output-name is used "
            "and must not already exist"
        ),
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="zero",
        help="raw samples to exclude from statistics",
    )
    parser.add_argument(
        "--zero-mask",
        choices=("component", "vector"),
        default="component",
        help="component or vector validity mask for exact-zero invalid samples",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help="minimum fraction of valid time samples required for each mean value",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="time frames per chunk")
    parser.add_argument(
        "--u-inf",
        type=float,
        default=4.0,
        help="free-stream velocity used for wake recovery products",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing mean.nc files")
    parser.add_argument("--dry-run", action="store_true", help="show work without processing")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N matching files, useful for testing",
    )
    return parser


def main() -> None:
    prepare_mean_wake_products(build_parser().parse_args())


if __name__ == "__main__":
    main()
