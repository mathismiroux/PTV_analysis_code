"""Combine all saved TI volumes of one case using a single color scale."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def load_volumes(root, case):
    volumes = []
    for folder in root.iterdir():
        if not folder.is_dir() or not folder.name.lower().startswith(case.lower() + "_"):
            continue
        candidates = [folder / "turbulence_intensity.npz",
                      folder / "turbulence_intensity_saved_mean" / "turbulence_intensity.npz",
                      folder / "turbulence_intensity" / "turbulence_intensity.npz"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            print(f"Skipping {folder.name}: no saved TI field", flush=True)
            continue
        with np.load(path) as data:
            axes = {key: np.asarray(data[key], dtype=float) for key in ("x", "y", "z")}
            ti = np.asarray(data["ti_percent"], dtype=float)
        if any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in axes.values()):
            raise ValueError(f"Invalid coordinates: {path}")
        if ti.shape != tuple(len(axes[k]) for k in ("z", "y", "x")):
            raise ValueError(f"TI shape does not match coordinates: {path}")
        volumes.append(dict(name=folder.name, axes=axes, ti=ti, source=path))
    if not volumes:
        raise ValueError(f"No saved TI volumes for case {case!r} in {root}")
    return sorted(volumes, key=lambda v: float(np.mean(v["axes"]["x"])))


def build_figure(volumes, case, vmax=None, coordinate_unit="mm"):
    planes = [v["ti"][v["ti"].shape[0] // 2] for v in volumes]
    finite_maxima = [float(p[np.isfinite(p)].max()) for p in planes if np.isfinite(p).any()]
    limit = vmax if vmax is not None else max([1e-6, *finite_maxima])
    norm = Normalize(vmin=0, vmax=limit)
    fig, ax = plt.subplots(figsize=(18, 5), layout="constrained")
    for volume, plane in zip(volumes, planes):
        axes, ti = volume["axes"], volume["ti"]
        iz = ti.shape[0] // 2
        plot = ax.pcolormesh(axes["x"], axes["y"], plane,
                             shading="auto", cmap="viridis", norm=norm)
        ax.text(float(np.mean(axes["x"])), 1.02,
                f"{volume['name']}\nz = {axes['z'][iz]:g} {coordinate_unit}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.set(xlabel=f"Downstream x [{coordinate_unit}]", ylabel=f"y [{coordinate_unit}]",
           aspect="equal")
    ax.set_facecolor("#dddddd")
    fig.suptitle(f"{case}: streamwise turbulence intensity — central z planes", fontsize=14)
    clipped = any(value > limit for value in finite_maxima)
    fig.colorbar(plot, ax=ax, label="TI [%]", shrink=0.8,
                 extend="max" if clipped else "neither")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Parent directory containing the case/volume folders")
    parser.add_argument("--case", default="Flow", help="Case prefix, e.g. Flow, Static, SurgeLF")
    parser.add_argument("--output", type=Path, help="Output PNG or PDF; default: INPUT/CASE_TI_combined.png")
    parser.add_argument("--vmax", type=float, help="Shared maximum TI in percent; default: maximum across displayed central-z planes")
    parser.add_argument("--coordinate-unit", default="mm", help="Coordinate label; no conversion")
    args = parser.parse_args(argv)
    if args.vmax is not None and (not np.isfinite(args.vmax) or args.vmax <= 0):
        parser.error("--vmax must be finite and positive")
    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.case)
    output = args.output or args.input / f"{safe_case}_TI_combined.png"
    if output.suffix.lower() not in (".png", ".pdf"):
        parser.error("Output must have .png or .pdf extension")
    volumes = load_volumes(args.input, args.case)
    fig = build_figure(volumes, args.case, args.vmax, args.coordinate_unit)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {len(volumes)} volumes with one color scale: {output.resolve()}")


if __name__ == "__main__":
    main()
