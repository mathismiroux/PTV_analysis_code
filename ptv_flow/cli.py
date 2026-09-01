from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

from ptv_flow.cases import DEFAULT_CASES_FILE, FlowCase, load_case
from ptv_flow.inspect import inspect_flow_gui
from ptv_flow.postprocess import (
    REYNOLDS_STRESS_COMPONENTS,
    TemporalAverageVolume,
    apply_valid_fraction_to_average,
    reynolds_stresses,
    spatio_temporal_interpolate_velocity,
    temporal_average_volume,
    turbulent_kinetic_energy,
)
from ptv_flow.reader import DEFAULT_FILE, FlowDataset
from ptv_flow.visualize import animate_z_plane, show_temporal_average_plane
from ptv_flow.validity import INVALID_SAMPLE_MODES


def print_stats(stats: dict[str, float]) -> None:
    print(f"\nFrame {int(stats['time_index'])} at t={stats['time']:.6g}")
    for key, value in stats.items():
        if key in {"time_index", "time"}:
            continue
        print(f"{key:>12}: {value:.6g}")


def _case_metadata(flow_case: FlowCase, processing_id: str) -> dict[str, str | float]:
    metadata = flow_case.metadata_attributes()
    metadata["processing_id"] = processing_id
    return metadata


def _case_mean_output(
    flow_case: FlowCase,
    overwrite: bool,
    processing_id: str | None = None,
) -> Path:
    if processing_id is not None:
        return flow_case.processing_output_dir(processing_id) / "mean.nc"
    return flow_case.default_output_path("mean.nc", unique=not overwrite)


def _compute_case_temporal_average(
    flow_case: FlowCase,
    flow: FlowDataset,
    output: Path,
    chunk_size: int,
    zero_mask: str,
    min_valid_fraction: float,
    overwrite: bool,
    invalid_samples: str,
) -> Path:
    metadata = _case_metadata(flow_case, output.parent.name)
    return temporal_average_volume(
        flow,
        output=output,
        chunk_size=chunk_size,
        zero_mask=zero_mask,
        min_valid_fraction=min_valid_fraction,
        overwrite=overwrite,
        u_inf=flow_case.u_inf,
        metadata=metadata,
        invalid_samples=invalid_samples,
    )


def _compute_case_tke(
    flow_case: FlowCase,
    flow: FlowDataset,
    mean_file: Path,
    output: Path,
    chunk_size: int,
    zero_mask: str,
    overwrite: bool,
    invalid_samples: str,
) -> Path:
    metadata = _case_metadata(flow_case, mean_file.parent.name)
    with TemporalAverageVolume(mean_file) as mean:
        return turbulent_kinetic_energy(
            flow,
            mean,
            output=output,
            chunk_size=chunk_size,
            zero_mask=zero_mask,
            overwrite=overwrite,
            metadata=metadata,
            invalid_samples=invalid_samples,
        )


def _compute_case_interpolated_velocity(
    flow_case: FlowCase,
    flow: FlowDataset,
    output: Path,
    axes: list[str],
    min_spatial_neighbors: int,
    passes: int,
    max_temporal_gap: int | None,
    workers: int,
    zero_mask: str,
    overwrite: bool,
    invalid_samples: str,
) -> Path:
    metadata = _case_metadata(flow_case, output.parent.name)
    return spatio_temporal_interpolate_velocity(
        flow,
        output=output,
        axes=axes,
        min_spatial_neighbors=min_spatial_neighbors,
        passes=passes,
        max_temporal_gap=max_temporal_gap,
        workers=workers,
        zero_mask=zero_mask,
        overwrite=overwrite,
        metadata=metadata,
        invalid_samples=invalid_samples,
    )


def _compute_case_reynolds_stresses(
    flow_case: FlowCase,
    flow: FlowDataset,
    mean_file: Path,
    output: Path,
    components: list[str],
    chunk_size: int,
    zero_mask: str,
    overwrite: bool,
    invalid_samples: str,
) -> Path:
    metadata = _case_metadata(flow_case, mean_file.parent.name)
    with TemporalAverageVolume(mean_file) as mean:
        return reynolds_stresses(
            flow,
            mean,
            output=output,
            components=components,
            chunk_size=chunk_size,
            zero_mask=zero_mask,
            overwrite=overwrite,
            metadata=metadata,
            invalid_samples=invalid_samples,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and visualize 3D PTV velocity fields from NetCDF4 files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_FILE,
        help=f"path to the .nc file, default: {DEFAULT_FILE}; ignored with --case",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="case id from --cases-file for paper postprocessing",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="one or more case ids from --cases-file for batch postprocessing",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help=f"YAML case registry, default: {DEFAULT_CASES_FILE}",
    )
    parser.add_argument(
        "--processing-id",
        type=str,
        default=None,
        help=(
            "specific processing output folder for --case workflows, for "
            "example static_x3p5d_02"
        ),
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="time index to inspect without loading the full file",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="visualize the x-y velocity field at the selected z plane",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="open an interactive cell inspector for raw values and averages",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="z plane to visualize; the nearest available z value is used",
    )
    parser.add_argument(
        "--plane",
        choices=("x", "y", "z"),
        default="z",
        help="slice axis for --average-plane",
    )
    parser.add_argument(
        "--plane-value",
        type=float,
        default=None,
        help=(
            "coordinate value for --average-plane; defaults to --z for z "
            "planes and 0 for x/y planes"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=40.0,
        help="animation frame rate",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="first time index for animation",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="one-past-last time index for animation; default uses all frames",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="time-index stride for animation",
    )
    parser.add_argument(
        "--quiver-step",
        type=int,
        default=3,
        help="plot one arrow every N grid points in x and y",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional output path; .gif works with the default dependencies",
    )
    parser.add_argument(
        "--temporal-average",
        action="store_true",
        help="compute a temporal average volume, excluding configured invalid samples",
    )
    parser.add_argument(
        "--interpolate-velocity",
        action="store_true",
        help=(
            "fill holes in the raw velocity time series using non-physics-informed "
            "spatio-temporal interpolation"
        ),
    )
    parser.add_argument(
        "--interpolation-axes",
        nargs="+",
        choices=("t", "z", "y", "x"),
        default=["t", "z", "y", "x"],
        help=(
            "axis order used by --interpolate-velocity; linear interpolation is "
            "applied sequentially"
        ),
    )
    parser.add_argument(
        "--min-interpolation-neighbors",
        type=int,
        default=6,
        help=(
            "minimum number of valid 3D corner neighbors required around a "
            "hole; default 6 means at least 6 out of the 8 surrounding "
            "spatial cells at the same time"
        ),
    )
    parser.add_argument(
        "--interpolation-passes",
        type=int,
        default=1,
        help=(
            "maximum number of interpolation passes; later passes may use "
            "values filled by earlier passes"
        ),
    )
    parser.add_argument(
        "--max-temporal-gap",
        type=int,
        default=None,
        help=(
            "maximum frame-index distance to the valid bracketing samples used "
            "for temporal interpolation; omit for no temporal distance limit"
        ),
    )
    parser.add_argument(
        "--interpolation-workers",
        type=int,
        default=1,
        help=(
            "number of velocity components to interpolate concurrently; use 3 "
            "to process u, v, and w in parallel if memory allows"
        ),
    )
    parser.add_argument(
        "--postprocess-basic",
        action="store_true",
        help="for case workflows, compute mean.nc, tke.nc, and reynolds_stresses.nc",
    )
    parser.add_argument(
        "--average-plane",
        action="store_true",
        help="visualize one x, y, or z plane from a temporal-average output file",
    )
    parser.add_argument(
        "--apply-valid-fraction",
        type=float,
        default=None,
        help="apply a valid-fraction cutoff to an existing temporal-average file",
    )
    parser.add_argument(
        "--tke",
        action="store_true",
        help="compute turbulent kinetic energy from a raw file and a mean file",
    )
    parser.add_argument(
        "--reynolds-stress",
        action="store_true",
        help="compute Reynolds stress components from a raw file and a mean file",
    )
    parser.add_argument(
        "--stress-components",
        nargs="+",
        choices=(*REYNOLDS_STRESS_COMPONENTS, "all"),
        default=["all"],
        help=(
            "Reynolds stress components for --reynolds-stress: uu uv uw "
            "vv vw ww or all"
        ),
    )
    parser.add_argument(
        "--quantity",
        choices=("speed", "u", "v", "w"),
        default="speed",
        help="scalar quantity to plot for --average-plane",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output path for postprocessing results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow postprocessing to replace an existing --output file",
    )
    parser.add_argument(
        "--average-file",
        type=Path,
        default=None,
        help="temporal-average file to compare in --inspect mode",
    )
    parser.add_argument(
        "--interpolated-file",
        type=Path,
        default=None,
        help="interpolated raw-style velocity file to compare in --inspect mode",
    )
    parser.add_argument(
        "--mean-file",
        type=Path,
        default=None,
        help="temporal-average velocity file used by --tke or --reynolds-stress",
    )
    parser.add_argument(
        "--compare-average",
        action="store_true",
        help="compare raw inspector values with --average-file",
    )
    parser.add_argument(
        "--compare-interpolated",
        action="store_true",
        help=(
            "compare raw inspector values with --interpolated-file and show "
            "filled cells as transparent"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="number of time steps to read at once for postprocessing",
    )
    parser.add_argument(
        "--zero-mask",
        choices=("component", "vector"),
        default="component",
        help=(
            "component ignores zeros independently for u/v/w; vector ignores "
            "only samples where u, v, and w are all exactly zero"
        ),
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default=None,
        help=(
            "raw samples to exclude from statistics: zero ignores exact zeros "
            "(default unless --inspect --compare-average can read a different "
            "policy from --average-file), nan ignores NaN/inf values, "
            "zero-or-nan ignores both, none excludes nothing"
        ),
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "discard averaged values with fewer than this fraction of valid "
            "time samples; use 0.8 for an 80 percent cutoff"
        ),
    )
    return parser


def _invalid_samples_from_average(
    average: TemporalAverageVolume,
    explicit_invalid_samples: str | None,
) -> str:
    if explicit_invalid_samples is not None:
        return explicit_invalid_samples
    stored = average._file.attrs.get("invalid_samples")
    if stored is None:
        return "zero"
    stored_text = stored.decode() if isinstance(stored, bytes) else str(stored)
    if stored_text not in INVALID_SAMPLE_MODES:
        print(
            f"Average file records unsupported invalid_samples={stored_text!r}; "
            "falling back to 'zero'.",
            flush=True,
        )
        return "zero"
    print(
        f"Using invalid_samples={stored_text!r} from average file metadata.",
        flush=True,
    )
    return stored_text


def _invalid_samples_from_interpolated(
    interpolated: FlowDataset,
    explicit_invalid_samples: str | None,
) -> str:
    if explicit_invalid_samples is not None:
        return explicit_invalid_samples
    stored = interpolated._file.attrs.get("invalid_samples")
    if stored is None:
        return "zero"
    stored_text = stored.decode() if isinstance(stored, bytes) else str(stored)
    if stored_text not in INVALID_SAMPLE_MODES:
        print(
            f"Interpolated file records unsupported invalid_samples={stored_text!r}; "
            "falling back to 'zero'.",
            flush=True,
        )
        return "zero"
    print(
        f"Using invalid_samples={stored_text!r} from interpolated file metadata.",
        flush=True,
    )
    return stored_text


def main() -> None:
    args = build_parser().parse_args()
    invalid_samples = args.invalid_samples or "zero"
    if args.case is not None and args.cases is not None:
        raise SystemExit("Use either --case or --cases, not both.")

    if args.postprocess_basic:
        case_ids = args.cases or ([args.case] if args.case is not None else None)
        if not case_ids:
            raise SystemExit("--postprocess-basic requires --case or --cases.")
        if args.output is not None:
            raise SystemExit("--postprocess-basic uses case output folders; omit --output.")
        if args.mean_file is not None:
            raise SystemExit("--postprocess-basic computes mean.nc; omit --mean-file.")
        if args.processing_id is not None and len(case_ids) > 1:
            raise SystemExit("--processing-id can only be used with one --case.")

        for case_id in case_ids:
            flow_case = load_case(case_id, args.cases_file)
            flow_case.validate_for_temporal_average()
            raw_path = flow_case.require_velocity()
            with FlowDataset(raw_path) as flow:
                mean_output = _case_mean_output(
                    flow_case,
                    overwrite=args.overwrite,
                    processing_id=args.processing_id,
                )
                print(
                    f"Postprocessing basic products for case {case_id!r} "
                    f"into {mean_output.parent}",
                    flush=True,
                )
                try:
                    mean_file = _compute_case_temporal_average(
                        flow_case=flow_case,
                        flow=flow,
                        output=mean_output,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        min_valid_fraction=args.min_valid_fraction,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                    _compute_case_tke(
                        flow_case=flow_case,
                        flow=flow,
                        mean_file=mean_file,
                        output=mean_file.parent / "tke.nc",
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                    _compute_case_reynolds_stresses(
                        flow_case=flow_case,
                        flow=flow,
                        mean_file=mean_file,
                        output=mean_file.parent / "reynolds_stresses.nc",
                        components=args.stress_components,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                except FileExistsError as exc:
                    raise SystemExit(str(exc)) from exc
        return

    flow_case = None
    raw_path = args.path
    if args.case is not None:
        flow_case = load_case(args.case, args.cases_file)
        if args.temporal_average:
            flow_case.validate_for_temporal_average()
        if args.interpolate_velocity:
            flow_case.validate_for_temporal_average()
        raw_path = flow_case.require_velocity()

    if args.apply_valid_fraction is not None:
        if args.output is None:
            raise SystemExit("--apply-valid-fraction requires --output.")
        try:
            apply_valid_fraction_to_average(
                source=args.path,
                output=args.output,
                min_valid_fraction=args.apply_valid_fraction,
                overwrite=args.overwrite,
            )
        except FileExistsError as exc:
            raise SystemExit(str(exc)) from exc
        return

    if args.average_plane:
        plane_value = args.z if args.plane_value is None and args.plane == "z" else args.plane_value
        if plane_value is None:
            plane_value = 0.0
        with TemporalAverageVolume(args.path) as volume:
            show_temporal_average_plane(
                volume,
                plane_axis=args.plane,
                plane_value=plane_value,
                quantity=args.quantity,
                quiver_step=args.quiver_step,
                save=args.save,
                min_valid_fraction=args.min_valid_fraction,
            )
        return

    with FlowDataset(raw_path) as flow:
        if args.interpolate_velocity:
            output = args.output
            if output is None:
                if flow_case is not None:
                    output = flow_case.default_output_path(
                        "interpolated_velocity.nc",
                        unique=not args.overwrite,
                    )
                else:
                    output = Path("outputs") / f"{raw_path.stem}_interpolated.nc"
            try:
                if flow_case is not None:
                    _compute_case_interpolated_velocity(
                        flow_case=flow_case,
                        flow=flow,
                        output=output,
                        axes=args.interpolation_axes,
                        min_spatial_neighbors=args.min_interpolation_neighbors,
                        passes=args.interpolation_passes,
                        max_temporal_gap=args.max_temporal_gap,
                        workers=args.interpolation_workers,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                else:
                    spatio_temporal_interpolate_velocity(
                        flow,
                        output=output,
                        axes=args.interpolation_axes,
                        min_spatial_neighbors=args.min_interpolation_neighbors,
                        passes=args.interpolation_passes,
                        max_temporal_gap=args.max_temporal_gap,
                        workers=args.interpolation_workers,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.reynolds_stress:
            if args.mean_file is None and flow_case is None:
                raise SystemExit("--reynolds-stress requires --mean-file.")
            mean_file = args.mean_file
            if mean_file is None and flow_case is not None:
                mean_file = flow_case.resolve_existing_product(
                    "mean.nc",
                    processing_id=args.processing_id,
                )
            output = args.output
            if output is None:
                if flow_case is not None:
                    output = mean_file.parent / "reynolds_stresses.nc"
                else:
                    output = Path("outputs") / f"{raw_path.stem}_reynolds_stresses.nc"
            try:
                if flow_case is not None:
                    _compute_case_reynolds_stresses(
                        flow_case,
                        flow,
                        mean_file,
                        output=output,
                        components=args.stress_components,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                else:
                    with TemporalAverageVolume(mean_file) as mean:
                        reynolds_stresses(
                            flow,
                            mean,
                            output=output,
                            components=args.stress_components,
                            chunk_size=args.chunk_size,
                            zero_mask=args.zero_mask,
                            overwrite=args.overwrite,
                            invalid_samples=invalid_samples,
                        )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.tke:
            if args.mean_file is None and flow_case is None:
                raise SystemExit("--tke requires --mean-file.")
            mean_file = args.mean_file
            if mean_file is None and flow_case is not None:
                mean_file = flow_case.resolve_existing_product(
                    "mean.nc",
                    processing_id=args.processing_id,
                )
            output = args.output
            if output is None:
                if flow_case is not None:
                    output = mean_file.parent / "tke.nc"
                else:
                    output = Path("outputs") / f"{raw_path.stem}_tke.nc"
            try:
                if flow_case is not None:
                    _compute_case_tke(
                        flow_case,
                        flow,
                        mean_file,
                        output=output,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                else:
                    with TemporalAverageVolume(mean_file) as mean:
                        turbulent_kinetic_energy(
                            flow,
                            mean,
                            output=output,
                            chunk_size=args.chunk_size,
                            zero_mask=args.zero_mask,
                            overwrite=args.overwrite,
                            invalid_samples=invalid_samples,
                        )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.temporal_average:
            output = args.output
            if output is None:
                if flow_case is not None:
                    output = flow_case.default_output_path(
                        "mean.nc",
                        unique=not args.overwrite,
                    )
                else:
                    output = Path("outputs") / f"{raw_path.stem}_temporal_mean.nc"
            try:
                if flow_case is not None:
                    _compute_case_temporal_average(
                        flow_case=flow_case,
                        flow=flow,
                        output=output,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        min_valid_fraction=args.min_valid_fraction,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
                else:
                    temporal_average_volume(
                        flow,
                        output=output,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        min_valid_fraction=args.min_valid_fraction,
                        overwrite=args.overwrite,
                        invalid_samples=invalid_samples,
                    )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.animate:
            animate_z_plane(
                flow,
                z_value=args.z,
                fps=args.fps,
                start=args.start,
                stop=args.stop,
                step=args.step,
                quiver_step=args.quiver_step,
                save=args.save,
            )
        elif args.inspect:
            if args.average_file is not None and not args.compare_average:
                raise SystemExit(
                    "--average-file is only used with --compare-average. "
                    "Use --inspect alone for raw-only inspection."
                )
            if args.compare_average and args.average_file is None:
                raise SystemExit("--compare-average requires --average-file.")
            if args.interpolated_file is not None and not args.compare_interpolated:
                raise SystemExit(
                    "--interpolated-file is only used with --compare-interpolated."
                )
            if args.compare_interpolated and args.interpolated_file is None:
                raise SystemExit("--compare-interpolated requires --interpolated-file.")
            if not args.compare_average and not args.compare_interpolated:
                inspect_flow_gui(
                    flow,
                    initial_frame=args.frame,
                    initial_z=args.z,
                    quiver_step=args.quiver_step,
                    min_valid_fraction=args.min_valid_fraction,
                    invalid_samples=invalid_samples,
                )
            else:
                average_context = (
                    TemporalAverageVolume(args.average_file)
                    if args.compare_average
                    else nullcontext(None)
                )
                interpolated_context = (
                    FlowDataset(args.interpolated_file)
                    if args.compare_interpolated
                    else nullcontext(None)
                )
                with average_context as average, interpolated_context as interpolated:
                    inspect_invalid_samples = invalid_samples
                    if average is not None:
                        inspect_invalid_samples = _invalid_samples_from_average(
                            average,
                            args.invalid_samples,
                        )
                    elif interpolated is not None:
                        inspect_invalid_samples = _invalid_samples_from_interpolated(
                            interpolated,
                            args.invalid_samples,
                        )
                    inspect_flow_gui(
                        flow,
                        average=average,
                        interpolated=interpolated,
                        initial_frame=args.frame,
                        initial_z=args.z,
                        quiver_step=args.quiver_step,
                        min_valid_fraction=args.min_valid_fraction,
                        invalid_samples=inspect_invalid_samples,
                    )
        else:
            print(flow.describe())
            print_stats(flow.frame_stats(args.frame))
