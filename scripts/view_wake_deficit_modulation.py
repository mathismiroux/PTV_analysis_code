"""Interactively inspect wake-deficit modulation from one phase_average.nc file.

Choose a file at launch, then use the x/D slider to inspect the area-mean
wake-deficit cycle and its fitted first harmonic. The region and validity
rules match plot_wake_pulsing.py: by default, a rotor-centred circular sector
with R=D/2 and angles -25 to 25 degrees from +y is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_wake_pulsing import DEFAULT_INPUT, calculate, sector_area_weights

# plot_wake_pulsing deliberately uses Agg for saved batch figures. Switch this
# standalone viewer back to a windowed backend after importing its helpers.
try:
    matplotlib.use("TkAgg", force=True)
except ImportError as exc:  # pragma: no cover - depends on local Python build
    raise SystemExit("The interactive viewer requires the Tk GUI backend.") from exc
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def choose_file(initial_directory: Path) -> Path:
    """Open a native file picker only when no input path was supplied."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover - depends on local Python build
        raise SystemExit("Pass a phase_average.nc path because tkinter is unavailable.") from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        initialdir=initial_directory,
        title="Select a phase-average volume",
        filetypes=[("Phase-average NetCDF", "phase_average.nc"), ("NetCDF/HDF5", "*.nc"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected:
        raise SystemExit("No file selected.")
    return Path(selected)


def load_result(path: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    if not path.is_file():
        raise ValueError(f"File does not exist: {path}")
    with h5py.File(path) as h5:
        for name in ("x", "y", "z", "phase", "u_phase_mean", "u_phase_count", "phase_sample_count"):
            if name not in h5:
                raise ValueError(f"{path} is not a compatible phase-average file; missing {name!r}")
        bounds = [float(h5["y"][0]), float(h5["y"][-1]), float(h5["z"][0]), float(h5["z"][-1])]
    sector = dict(
        axis_y=args.rotor_y,
        axis_z=args.rotor_z,
        radius=args.radius if args.radius is not None else args.rotor_diameter / 2,
        angles=args.sector_angles,
    )
    # Validate before loading the potentially large phase-average arrays.
    sector_area_weights(np.arange(2.0), np.arange(2.0), **sector)
    result = calculate(
        path, bounds, args.rotor_diameter, args.min_valid_fraction,
        args.min_coverage, args.edge_slices, args.u_inf, sector,
    )
    return result, sector


def harmonic_fit(result: dict, index: int) -> np.ndarray:
    return (
        result["offset"][index]
        + result["a"][index] * np.cos(result["phase"])
        + result["b"][index] * np.sin(result["phase"])
    )


def run_viewer(result: dict, sector: dict) -> None:
    x = result["x"]
    phase_degrees = np.degrees(result["phase"])
    valid_indices = np.flatnonzero(np.isfinite(result["amplitude"]))
    initial = int(valid_indices[0]) if valid_indices.size else 0

    figure = plt.figure(figsize=(11, 7.5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=[4, 1.65], width_ratios=[3.2, 1])
    cycle_axis = figure.add_subplot(grid[0, 0])
    amplitude_axis = figure.add_subplot(grid[0, 1])
    slider_axis = figure.add_subplot(grid[1, :])
    slider_axis.set_position([.16, .075, .68, .035])

    observed, = cycle_axis.plot([], [], "o-", color="#2678b2", label=r"$D_m(x,phi)$")
    fitted, = cycle_axis.plot([], [], "--", color="#d95f02", linewidth=2, label="First-harmonic fit")
    amplitude_curve, = amplitude_axis.plot(x, result["amplitude"], color="#555555", linewidth=1.5)
    selected_amplitude, = amplitude_axis.plot([], [], "o", color="#d95f02", markersize=7)
    info = cycle_axis.text(.02, .98, "", transform=cycle_axis.transAxes, va="top",
                           bbox={"facecolor": "white", "alpha": .88, "edgecolor": "0.75"})

    cycle_axis.set(xlabel="Saved phase φ [degrees]", ylabel=r"Area-mean deficit $D_m$", xlim=(0, 360),
                   xticks=[0, 90, 180, 270, 360])
    cycle_axis.grid(alpha=.25)
    cycle_axis.legend(loc="upper right")
    amplitude_axis.set(xlabel=r"$x/D$", ylabel=r"$A_D$", title="Modulation amplitude")
    amplitude_axis.grid(alpha=.25)
    amplitude_axis.set_ylim(bottom=0)

    slider = Slider(slider_axis, r"Downstream distance $x/D$", 0, x.size - 1,
                    valinit=initial, valstep=1, valfmt="%0.0f")

    def update(value: float) -> None:
        index = int(round(value))
        slider.valtext.set_text(f"{x[index]:.3f}")
        values = result["dm"][:, index]
        fit = harmonic_fit(result, index)
        observed.set_data(phase_degrees, values)
        fitted.set_data(phase_degrees, fit)
        selected_amplitude.set_data([x[index]], [result["amplitude"][index]])
        finite = np.r_[values[np.isfinite(values)], fit[np.isfinite(fit)]]
        if finite.size:
            padding = max(.01, .08 * np.ptp(finite))
            cycle_axis.set_ylim(float(finite.min() - padding), float(finite.max() + padding))
        accepted = np.isfinite(result["amplitude"][index])
        info.set_text(
            f"x/D = {x[index]:.3f}\n"
            f"A_D = {result['amplitude'][index]:.4f}\n"
            f"Mean deficit = {result['offset'][index]:.4f}\n"
            f"Harmonic R² = {result['r2'][index]:.3f}\n"
            f"Sector coverage = {result['coverage'][index]:.1%}\n"
            + ("" if accepted else "\nExcluded by coverage/validity threshold")
        )
        cycle_axis.set_title(f"{result['label']} — wake-deficit modulation at x/D = {x[index]:.3f}")
        figure.canvas.draw_idle()

    slider.on_changed(update)
    geometry = (f"Sector: centre (y,z)=({sector['axis_y']:g}, {sector['axis_z']:g}), "
                f"R={sector['radius']:g}, angles {sector['angles'][0]:g} to {sector['angles'][1]:g}° from +y")
    figure.suptitle(f"{Path(result['path']).name} | {geometry}", fontsize=11)
    update(initial)
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path, nargs="?", help="phase_average.nc file; omit to choose it in a dialog")
    parser.add_argument("--rotor-diameter", type=float, default=1200)
    parser.add_argument("--rotor-y", type=float, default=0)
    parser.add_argument("--rotor-z", type=float, default=0)
    parser.add_argument("--radius", type=float, help="Sector radius; default is D/2")
    parser.add_argument("--sector-angles", type=float, nargs=2, default=[-25, 25], metavar=("START", "STOP"))
    parser.add_argument("--min-valid-fraction", type=float, default=.5)
    parser.add_argument("--min-coverage", type=float, default=.6)
    parser.add_argument("--edge-slices", type=int, default=1)
    parser.add_argument("--u-inf", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.file is None:
        args.file = choose_file(DEFAULT_INPUT)
    if args.rotor_diameter <= 0 or args.edge_slices < 0:
        raise SystemExit("--rotor-diameter must be positive and --edge-slices nonnegative")
    if not 0 <= args.min_valid_fraction <= 1 or not 0 <= args.min_coverage <= 1:
        raise SystemExit("Validity fractions must lie in [0, 1]")
    try:
        result, sector = load_result(args.file, args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    run_viewer(result, sector)


if __name__ == "__main__":
    main()
