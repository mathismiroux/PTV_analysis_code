"""Characterize one free-flow export, without modifying input data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.freeflow import FreeflowSettings, characterize_freeflow


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Velocity .nc file, or folder containing interpolated_velocity.nc")
    p.add_argument("--output", type=Path, required=True, help="New output directory (existing folders refused)")
    p.add_argument("--length-unit", choices=("native", "m", "mm"), default="native")
    p.add_argument("--time-unit", choices=("native", "s", "ms"), default="native")
    p.add_argument("--velocity-unit", choices=("native", "m/s", "mm/s"), default="native")
    for axis in ("x", "y", "z"):
        p.add_argument(f"--{axis}-range", nargs=2, type=float, metavar=("MIN", "MAX"), help="Inclusive bounds in INPUT units")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, help="Exclusive frame index")
    p.add_argument("--chunk-size", type=int, default=8)
    p.add_argument("--min-valid-fraction", type=float, default=.5)
    p.add_argument("--min-pairs", type=int, default=100)
    p.add_argument("--min-pair-fraction", type=float, default=.05)
    p.add_argument("--max-spatial-lag", type=int, help="Maximum separation in grid intervals")
    p.add_argument("--max-temporal-lag", type=int, help="Maximum lag in frames (default N/4)")
    p.add_argument("--invalid-samples", choices=("nan", "zero-or-nan"), default="nan")
    p.add_argument("--include-filled", action="store_true", help="Include samples marked as interpolated")
    p.add_argument("--temporal", action="store_true", help="Assert time-resolved sampling and compute point autocorrelations")
    p.add_argument("--probe-grid", type=int, default=2, help="Interior probe locations per spatial axis")
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--rotor-diameter-m", type=float)
    p.add_argument("--convection-speed-m-s", type=float, help="Otherwise use positive local mean u when velocity units are known")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    options = vars(args).copy()
    source, output = options.pop("source"), options.pop("output")
    if source.is_dir():
        source = source / "interpolated_velocity.nc"
    try:
        result = characterize_freeflow(source, output, FreeflowSettings(**options))
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Free-flow characterization written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
