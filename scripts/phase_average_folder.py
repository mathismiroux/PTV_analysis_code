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

from ptv_flow.postprocess import phase_average_volume
from ptv_flow.reader import FlowDataset
from ptv_flow.validity import INVALID_SAMPLE_MODES


REQUIRED_RAW_DATASETS = ("t", "x", "y", "z", "u", "v", "w")
GENERIC_VOLUME_NAMES = {"interpolated_velocity", "velocity", "raw"}
PHASE_MANIFEST_CSV = "phase_average_manifest.csv"
PHASE_MANIFEST_JSON = "phase_average_manifest.json"
PHASE_DRY_RUN_MANIFEST_CSV = "phase_average_dry_run_manifest.csv"
PHASE_DRY_RUN_MANIFEST_JSON = "phase_average_dry_run_manifest.json"


@dataclass(frozen=True)
class PhaseAverageSettings:
    input_folder: str
    output_root: str
    pattern: str
    frequency_hz: float | None
    phase_signal: str | None
    phase_offset: float
    n_phase_bins: int
    invalid_samples: str
    zero_mask: str
    min_valid_fraction: float
    chunk_size: int
    u_inf: float | None
    require_existing_case_folder: bool
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
    phase_source: str
    frequency_hz: float | None
    phase_signal: str | None
    phase_offset: float
    n_phase_bins: int
    invalid_samples: str
    zero_mask: str
    min_valid_fraction: float
    chunk_size: int
    u_inf: float | None


def _safe_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "case"


def case_id_for_source(input_folder: Path, source: Path) -> str:
    try:
        relative = source.resolve().relative_to(input_folder.resolve())
    except ValueError:
        relative = source.name
    if isinstance(relative, Path) and source.stem in GENERIC_VOLUME_NAMES and source.parent != input_folder:
        return _safe_text(source.parent.name)
    if isinstance(relative, Path):
        return _safe_text(source.stem)
    return _safe_text(str(relative))


def default_output_root(input_folder: Path, output_name: str) -> Path:
    return input_folder.resolve().parent / "outputs" / output_name


def resolve_output_root(input_folder: Path, args: argparse.Namespace) -> Path:
    if args.output_root is None and args.output_name is None:
        raise SystemExit(
            "This analysis requires a new output folder name. "
            "Pass --output-name NAME, for example --output-name phase_average_001."
        )
    if args.output_root is not None:
        return args.output_root.resolve()
    return default_output_root(input_folder, args.output_name)


def manifest_paths(output_root: Path, dry_run: bool) -> tuple[Path, Path]:
    if dry_run:
        return (
            output_root / PHASE_DRY_RUN_MANIFEST_CSV,
            output_root / PHASE_DRY_RUN_MANIFEST_JSON,
        )
    return output_root / PHASE_MANIFEST_CSV, output_root / PHASE_MANIFEST_JSON


def prepare_output_root(output_root: Path, allow_existing: bool, dry_run: bool) -> None:
    if output_root.exists() and not allow_existing:
        raise SystemExit(
            f"Refusing to use existing output folder: {output_root}. "
            "Choose a new --output-name or remove/archive the folder yourself."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = manifest_paths(output_root, dry_run)
    existing_manifests = [
        path.name
        for path in (csv_path, json_path)
        if path.exists()
    ]
    if existing_manifests:
        raise SystemExit(
            "Refusing to overwrite existing phase-average manifest file(s) in "
            f"{output_root}: {', '.join(existing_manifests)}."
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


def _stored_frequency_hz(source: Path) -> float | None:
    try:
        with h5py.File(source, "r") as h5:
            value = h5.attrs.get("frequency_hz")
    except OSError:
        return None
    if value is None:
        return None
    frequency = float(value)
    return frequency if frequency > 0.0 else None


def _phase_source(
    source: Path,
    settings: PhaseAverageSettings,
) -> tuple[float | None, Path | None, str]:
    if settings.phase_signal is not None:
        return None, Path(settings.phase_signal), str(Path(settings.phase_signal).resolve())
    if settings.frequency_hz is not None:
        return settings.frequency_hz, None, f"frequency_hz={settings.frequency_hz:g}"
    stored = _stored_frequency_hz(source)
    if stored is not None:
        return stored, None, f"source metadata frequency_hz={stored:g}"
    return None, None, "missing phase source"


def _metadata(
    settings: PhaseAverageSettings,
    source_file: Path,
    case_id: str,
    phase_source: str,
) -> dict[str, str | float]:
    values: dict[str, str | float] = {
        "script_name": Path(__file__).name,
        "command_line": settings.command_line,
        "batch_input_folder": settings.input_folder,
        "batch_output_root": settings.output_root,
        "batch_pattern": settings.pattern,
        "case_id": case_id,
        "label": case_id,
        "batch_phase_source": phase_source,
    }
    if settings.u_inf is not None:
        values["u_inf"] = float(settings.u_inf)
    return values


def _manifest_row(
    source: Path,
    output: Path,
    settings: PhaseAverageSettings,
    status: str,
    reason: str,
    phase_source: str,
    frequency_hz: float | None = None,
    phase_signal: Path | None = None,
) -> ManifestRow:
    n_times, nz, ny, nx = _raw_shape(source)
    return ManifestRow(
        source_file=str(source),
        output_file=str(output),
        status=status,
        reason=reason,
        file_size_bytes=source.stat().st_size,
        n_times=n_times,
        nz=nz,
        ny=ny,
        nx=nx,
        phase_source=phase_source,
        frequency_hz=frequency_hz,
        phase_signal=str(phase_signal) if phase_signal is not None else None,
        phase_offset=settings.phase_offset,
        n_phase_bins=settings.n_phase_bins,
        invalid_samples=settings.invalid_samples,
        zero_mask=settings.zero_mask,
        min_valid_fraction=settings.min_valid_fraction,
        chunk_size=settings.chunk_size,
        u_inf=settings.u_inf,
    )


def process_file(
    source: Path,
    output: Path,
    settings: PhaseAverageSettings,
    case_id: str,
) -> ManifestRow:
    valid, reason = is_raw_velocity_timeseries(source)
    if not valid:
        return _manifest_row(source, output, settings, "skipped", reason, "not resolved")

    if settings.require_existing_case_folder and not output.parent.exists():
        return _manifest_row(
            source,
            output,
            settings,
            "skipped",
            f"output case folder does not exist: {output.parent}",
            "not resolved",
        )

    frequency_hz, phase_signal, phase_source = _phase_source(source, settings)
    if frequency_hz is None and phase_signal is None:
        return _manifest_row(
            source,
            output,
            settings,
            "skipped",
            "pass --frequency-hz or --phase-signal, or store frequency_hz metadata",
            phase_source,
        )
    if phase_signal is not None and not phase_signal.exists():
        return _manifest_row(
            source,
            output,
            settings,
            "skipped",
            f"phase signal does not exist: {phase_signal}",
            phase_source,
            frequency_hz=frequency_hz,
            phase_signal=phase_signal,
        )
    if output.exists():
        return _manifest_row(
            source,
            output,
            settings,
            "exists",
            "output exists; choose a new output folder",
            phase_source,
            frequency_hz=frequency_hz,
            phase_signal=phase_signal,
        )
    if settings.dry_run:
        return _manifest_row(
            source,
            output,
            settings,
            "dry-run",
            "would process",
            phase_source,
            frequency_hz=frequency_hz,
            phase_signal=phase_signal,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with FlowDataset(source) as flow:
        phase_average_volume(
            flow,
            output=output,
            n_phase_bins=settings.n_phase_bins,
            frequency_hz=frequency_hz,
            phase_signal=phase_signal,
            phase_offset=settings.phase_offset,
            chunk_size=settings.chunk_size,
            zero_mask=settings.zero_mask,
            min_valid_fraction=settings.min_valid_fraction,
            overwrite=False,
            u_inf=settings.u_inf,
            metadata=_metadata(settings, source, case_id, phase_source),
            invalid_samples=settings.invalid_samples,
        )
    return _manifest_row(
        source,
        output,
        settings,
        "processed",
        "ok",
        phase_source,
        frequency_hz=frequency_hz,
        phase_signal=phase_signal,
    )


def write_manifest(
    output_root: Path,
    settings: PhaseAverageSettings,
    rows: Iterable[ManifestRow],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    csv_path, json_path = manifest_paths(output_root, settings.dry_run)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        if row_list:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(row_list[0])))
            writer.writeheader()
            for row in row_list:
                writer.writerow(asdict(row))
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


def phase_average_folder(args: argparse.Namespace) -> list[ManifestRow]:
    input_folder = args.input_folder.resolve()
    output_root = resolve_output_root(input_folder, args)
    prepare_output_root(
        output_root,
        allow_existing=args.output_root is not None,
        dry_run=args.dry_run,
    )
    settings = PhaseAverageSettings(
        input_folder=str(input_folder),
        output_root=str(output_root),
        pattern=args.pattern,
        frequency_hz=args.frequency_hz,
        phase_signal=str(args.phase_signal.resolve()) if args.phase_signal is not None else None,
        phase_offset=args.phase_offset,
        n_phase_bins=args.n_phase_bins,
        invalid_samples=args.invalid_samples,
        zero_mask=args.zero_mask,
        min_valid_fraction=args.min_valid_fraction,
        chunk_size=args.chunk_size,
        u_inf=args.u_inf,
        require_existing_case_folder=args.output_root is not None,
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
    used_case_ids: set[str] = set()
    for source in files:
        case_id = case_id_for_source(input_folder, source)
        if case_id in used_case_ids:
            raise SystemExit(
                f"Duplicate case id {case_id!r} from input file {source}. "
                "Use a narrower --pattern or rename input folders/files."
            )
        used_case_ids.add(case_id)
        case_dir = output_root / case_id
        output = case_dir / "phase_average.nc"
        print(f"\n{source.name} -> {output}", flush=True)
        row = process_file(source, output, settings, case_id)
        print(f"  {row.status}: {row.reason}", flush=True)
        rows.append(row)
    write_manifest(output_root, settings, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute phase-average products for every raw/interpolated velocity "
            "NetCDF file in a folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_folder", type=Path, help="folder containing velocity .nc files")
    parser.add_argument(
        "--pattern",
        default="SurgeLF*/interpolated_velocity.nc",
        help="glob pattern for input files",
    )
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
            "explicit output folder containing per-case subfolders; required "
            "unless --output-name is used"
        ),
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--frequency-hz",
        type=float,
        default=2.0,
        help=(
            "imposed motion frequency for all files; use 2 Hz for SurgeLF"
        ),
    )
    phase_group.add_argument(
        "--phase-signal",
        type=Path,
        default=None,
        help=(
            "one phase value per time step, shared by all files; text, .npy, "
            "or HDF5 with phase/phase_signal/phi"
        ),
    )
    parser.add_argument(
        "--phase-offset",
        type=float,
        default=0.0,
        help="phase offset in radians added when phase is inferred from frequency",
    )
    parser.add_argument(
        "--n-phase-bins",
        type=int,
        default=32,
        help="number of equal phase bins over one cycle",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help="raw samples to exclude from phase averages",
    )
    parser.add_argument(
        "--zero-mask",
        choices=("component", "vector"),
        default="vector",
        help="component or vector validity mask for exact-zero invalid samples",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.3,
        help="minimum valid fraction required in each phase bin",
    )
    parser.add_argument("--chunk-size", type=int, default=50, help="time frames per chunk")
    parser.add_argument(
        "--u-inf",
        type=float,
        default=4.0,
        help="free-stream velocity used for phase wake-deficit products",
    )
    parser.add_argument("--dry-run", action="store_true", help="show work without processing")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N matching files, useful for testing",
    )
    return parser


def main() -> None:
    phase_average_folder(build_parser().parse_args())


if __name__ == "__main__":
    main()
