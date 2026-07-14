from __future__ import annotations

import argparse
from pathlib import Path

from ptv_flow.inspect import inspect_flow_gui
from ptv_flow.postprocess import (
    REYNOLDS_STRESS_COMPONENTS,
    TemporalAverageVolume,
    apply_valid_fraction_to_average,
    reynolds_stresses,
    temporal_average_volume,
    turbulent_kinetic_energy,
)
from ptv_flow.reader import DEFAULT_FILE, FlowDataset
from ptv_flow.visualize import animate_z_plane, show_temporal_average_plane


def print_stats(stats: dict[str, float]) -> None:
    print(f"\nFrame {int(stats['time_index'])} at t={stats['time']:.6g}")
    for key, value in stats.items():
        if key in {"time_index", "time"}:
            continue
        print(f"{key:>12}: {value:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and visualize 3D PTV velocity fields from NetCDF4 files."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_FILE,
        help=f"path to the .nc file, default: {DEFAULT_FILE}",
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
        help="compute a temporal average volume, ignoring exact-zero values",
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
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "discard averaged values with fewer than this fraction of valid "
            "time samples; use 0.8 for an 80 percent cutoff"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
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

    with FlowDataset(args.path) as flow:
        if args.reynolds_stress:
            if args.mean_file is None:
                raise SystemExit("--reynolds-stress requires --mean-file.")
            output = args.output
            if output is None:
                output = Path("outputs") / f"{args.path.stem}_reynolds_stresses.nc"
            try:
                with TemporalAverageVolume(args.mean_file) as mean:
                    reynolds_stresses(
                        flow,
                        mean,
                        output=output,
                        components=args.stress_components,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                    )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.tke:
            if args.mean_file is None:
                raise SystemExit("--tke requires --mean-file.")
            output = args.output
            if output is None:
                output = Path("outputs") / f"{args.path.stem}_tke.nc"
            try:
                with TemporalAverageVolume(args.mean_file) as mean:
                    turbulent_kinetic_energy(
                        flow,
                        mean,
                        output=output,
                        chunk_size=args.chunk_size,
                        zero_mask=args.zero_mask,
                        overwrite=args.overwrite,
                    )
            except FileExistsError as exc:
                raise SystemExit(str(exc)) from exc
        elif args.temporal_average:
            output = args.output
            if output is None:
                output = Path("outputs") / f"{args.path.stem}_temporal_mean.nc"
            try:
                temporal_average_volume(
                    flow,
                    output=output,
                    chunk_size=args.chunk_size,
                    zero_mask=args.zero_mask,
                    min_valid_fraction=args.min_valid_fraction,
                    overwrite=args.overwrite,
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
            if not args.compare_average:
                inspect_flow_gui(
                    flow,
                    initial_frame=args.frame,
                    initial_z=args.z,
                    quiver_step=args.quiver_step,
                    min_valid_fraction=args.min_valid_fraction,
                )
            else:
                with TemporalAverageVolume(args.average_file) as average:
                    inspect_flow_gui(
                        flow,
                        average=average,
                        initial_frame=args.frame,
                        initial_z=args.z,
                        quiver_step=args.quiver_step,
                        min_valid_fraction=args.min_valid_fraction,
                    )
        else:
            print(flow.describe())
            print_stats(flow.frame_stats(args.frame))
