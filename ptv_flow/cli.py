from __future__ import annotations

import argparse
from pathlib import Path

from ptv_flow.reader import DEFAULT_FILE, FlowDataset
from ptv_flow.visualize import animate_z_plane


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
        "--z",
        type=float,
        default=0.0,
        help="z plane to visualize; the nearest available z value is used",
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with FlowDataset(args.path) as flow:
        if args.animate:
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
        else:
            print(flow.describe())
            print_stats(flow.frame_stats(args.frame))
