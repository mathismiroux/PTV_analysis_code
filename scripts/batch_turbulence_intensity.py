"""Run TI analysis for case subfolders containing interpolated_velocity.nc."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Parent folder containing case/volume subfolders")
    parser.add_argument("--output-root", type=Path, help="Optional new results directory; default: save TI files beside each mean.nc")
    parser.add_argument("--case-pattern", default="*", help="Case folder glob, e.g. Flow_* or Static_*")
    parser.add_argument("--recursive", action="store_true", help="Find cases at any depth; preserve relative paths")
    parser.add_argument("--dry-run", action="store_true", help="List cases without creating outputs")
    parser.add_argument("--reference-velocity", type=float)
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--min-mean-speed", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--coordinate-unit", default="mm")
    parser.add_argument("--vmax", type=float, help="Shared color scale maximum in percent")
    args = parser.parse_args(argv)
    root = args.input.resolve()
    output = args.output_root.resolve() if args.output_root else None
    if not root.is_dir():
        parser.error(f"Input directory does not exist: {root}")
    if output is not None and output.exists():
        parser.error(f"Output already exists; choose a new --output-root: {output}")
    pattern = "**/interpolated_velocity.nc" if args.recursive else "*/interpolated_velocity.nc"
    sources = sorted(p for p in root.glob(pattern) if p.is_file() and p.parent.match(args.case_pattern))
    if not sources:
        parser.error("No matching case subfolders with interpolated_velocity.nc")
    forwarded = []
    for key in ("reference_velocity", "min_valid_fraction", "min_mean_speed", "chunk_size", "coordinate_unit", "vmax"):
        value = getattr(args, key)
        if value is not None:
            forwarded.extend(["--" + key.replace("_", "-"), str(value)])
    script = Path(__file__).with_name("plot_turbulence_intensity.py")
    if args.dry_run:
        for source in sources:
            mean = source.with_name("mean.nc")
            destination = output / source.parent.relative_to(root) if output else source.parent
            print(f"{source.parent.relative_to(root)} -> {destination} "
                  f"({'saved mean.nc' if mean.is_file() else 'mean computed from time series'})")
        print(f"{len(sources)} cases; no files written")
        return 0
    if output:
        output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    manifest_path = output / "batch_manifest.csv" if output else root / f"ti_batch_manifest_{run_id}.csv"
    failures = 0
    with manifest_path.open("x", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=["case", "source", "mean_source", "output", "status", "returncode", "log"])
        writer.writeheader()
        for index, source in enumerate(sources, 1):
            relative = source.parent.relative_to(root)
            destination = output / relative if output else source.parent
            log = destination.parent / (destination.name + ".log") if output else destination / f"ti_{run_id}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            target_args = ["--output-folder", str(destination)] if output else ["--alongside-mean"]
            command = [sys.executable, str(script), str(source), *target_args, *forwarded]
            print(f"[{index}/{len(sources)}] {relative}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            status = "failed" if result.returncode else "processed"
            failures += bool(result.returncode)
            mean = source.with_name("mean.nc")
            writer.writerow(dict(case=str(relative), source=str(source),
                                 mean_source=str(mean) if mean.is_file() else "computed from time series",
                                 output=str(destination), status=status, returncode=result.returncode, log=str(log)))
            manifest.flush()
            print(f"  {status}; log: {log}", flush=True)
    print(f"Finished: {len(sources) - failures} processed, {failures} failed. Summary: {manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
