"""Plot phase-dependent, area-mean wake deficit and its first harmonic.

D_m = integral(1-u_phase/U_inf) dA / integral(dA), on a measured
circular sector with support fixed across all phases at each x.
A_D is the fitted sinusoidal semi-amplitude of D_m (not the mean of
local amplitudes). The positive-deficit centroid describes only the
observed window, which need not contain the entire wake.

Run: python scripts/plot_wake_pulsing.py
Coordinates and --rotor-diameter use the same units (mm in these files).
The saved phase reference/frequency is used without reinterpretation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.postprocess import _first_harmonic_from_phase_means
from scripts.plot_harmonic_amplitude_z0 import discover_phase_average_files, parse_case_distance
from scripts.plot_mean_wake_z0 import nearest_index
from scripts.plot_plane_wake_deficit_profiles import _case_style
from scripts.plot_radial_wake_deficit import _infer_u_inf

DEFAULT_INPUT = Path(r"D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10")


def is_1p625d_case(path: Path) -> bool:
    """Identify a 1.625D volume from its standard phase-average folder name."""
    _, _, distance = parse_case_distance(path.parent.name)
    return distance is not None and np.isclose(distance, 1.625)


def cell_widths(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size < 2 or not np.all(np.diff(coordinates) > 0):
        raise ValueError("Area averaging requires at least two increasing y and z coordinates")
    edges = np.r_[coordinates[0], (coordinates[1:] + coordinates[:-1]) / 2, coordinates[-1]]
    return np.diff(edges)


def sector_area_weights(y, z, axis_y, axis_z, radius, angles, subdivisions=16):
    """Cell/sector overlap by midpoint quadrature; angle 0 is +y, +90 is +z.

    Uses the same rotor-centred radius as radial_average_wake_deficit.
    Coverage is measured against the entire requested sector, including
    portions outside the field of view. Angles increase from start to end;
    use e.g. (330, 390) for a sector crossing zero.
    """
    start, stop = angles
    if not np.all(np.isfinite([axis_y, axis_z, radius, start, stop])) or radius <= 0 or not 0 < stop-start <= 360:
        raise ValueError("Sector requires a positive radius and an angular span in (0, 360]")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    dy, dz = cell_widths(y), cell_widths(z)
    ey = np.r_[y[0], (y[1:] + y[:-1]) / 2, y[-1]]
    ez = np.r_[z[0], (z[1:] + z[:-1]) / 2, z[-1]]
    fraction = np.zeros((len(z), len(y)))
    for fy in (np.arange(subdivisions) + .5) / subdivisions:
        yy = ey[:-1] + fy * dy - axis_y
        for fz in (np.arange(subdivisions) + .5) / subdivisions:
            zz = ez[:-1] + fz * dz - axis_z
            rr = np.hypot(yy[None, :], zz[:, None])
            theta = np.degrees(np.arctan2(zz[:, None], yy[None, :]))
            fraction += (rr <= radius) & ((theta - start) % 360 <= stop-start)
    area = np.outer(dz, dy) * fraction / subdivisions**2
    requested_area = .5 * radius**2 * np.radians(stop-start)
    return area, requested_area


def reduce_plane(deficit, counts, samples, y, z, bounds, min_valid_fraction=0.3,
                 area=None, requested_area=None):
    """Return D_m, positive-deficit centroids, and fixed-support coverage."""
    yy, zz = np.meshgrid(y, z)
    # Clip cell areas to the requested physical window, including boundary cells.
    def clipped_widths(c, lo, hi):
        cell_widths(c)  # validate coordinate ordering
        edges = np.r_[c[0], (c[1:] + c[:-1]) / 2, c[-1]]
        return np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
    if area is None:
        area = np.outer(clipped_widths(z, *bounds[2:]), clipped_widths(y, *bounds[:2]))
    if requested_area is None:
        requested_area = area.sum()
    fractions = np.divide(counts, samples[:, None, None],
                          out=np.zeros_like(counts, dtype=float), where=samples[:, None, None] > 0)
    support = np.all(np.isfinite(deficit) & (counts > 0) & (fractions >= min_valid_fraction), axis=0)
    weights = area * support
    accepted_area = weights.sum()
    coverage = float(min(1, accepted_area / requested_area)) if requested_area else 0.0
    empty = np.full(deficit.shape[0], np.nan)
    if accepted_area == 0:
        return empty.copy(), empty.copy(), empty.copy(), coverage
    safe = np.where(support[None], deficit, 0.0)
    dm = np.sum(safe * weights, axis=(1, 2)) / accepted_area
    positive = np.maximum(safe, 0) * weights
    mass = positive.sum(axis=(1, 2))
    cy = np.divide((positive * yy).sum(axis=(1, 2)), mass, out=empty.copy(), where=mass > 0)
    cz = np.divide((positive * zz).sum(axis=(1, 2)), mass, out=empty.copy(), where=mass > 0)
    return dm, cy, cz, coverage


def calculate(path, bounds, diameter, min_valid_fraction, min_coverage, edge_slices, u_inf=None, sector=None):
    with h5py.File(path, "r") as f:
        label = str(f.attrs.get("label", path.parent.name))
        case, _, _ = parse_case_distance(label)
        velocity = _infer_u_inf(f, u_inf)
        if not np.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"{path}: U_inf must be finite and positive")
        x, y, z, phase = [np.asarray(f[k][:], float) for k in ("x", "y", "z", "phase")]
        samples = f["phase_sample_count"][:]
        # Read one volume at a time; avoid repeatedly decompressing HDF5 chunks.
        deficit = 1 - f["u_phase_mean"][:].astype(float) / velocity
        counts = f["u_phase_count"][:]
        dm = np.full((len(phase), len(x)), np.nan)
        cy, cz = dm.copy(), dm.copy()
        coverage = np.zeros(len(x))
        area, requested_area = (None, None) if sector is None else sector_area_weights(y, z, **sector)
        for i in range(edge_slices, len(x) - edge_slices):
            d, yc, zc, coverage[i] = reduce_plane(
                deficit[..., i], counts[..., i], samples, y, z, bounds, min_valid_fraction,
                area, requested_area)
            if coverage[i] >= min_coverage:
                dm[:, i], cy[:, i], cz[:, i] = d, yc / diameter, zc / diameter
        offset, a, b, amplitude, lag = _first_harmonic_from_phase_means(dm, phase)
        ay = _first_harmonic_from_phase_means(cy, phase)[3]
        az = _first_harmonic_from_phase_means(cz, phase)[3]
        fit = offset + np.cos(phase[:, None]) * a + np.sin(phase[:, None]) * b
        residual = np.sum((dm - fit) ** 2, axis=0)
        total = np.sum((dm - offset) ** 2, axis=0)
        r2 = np.divide(residual, total, out=np.full_like(total, np.nan), where=total > 1e-24)
        return dict(path=str(path), label=label, case=case, frequency=float(f.attrs.get("frequency_hz", np.nan)),
                    phase_source=str(f.attrs.get("phase_source", "unknown")),
                    phase_offset=float(f.attrs.get("phase_offset", 0)), u_inf=velocity,
                    x=x / diameter, phase=phase, dm=dm, cy=cy, cz=cz, coverage=coverage,
                    offset=offset, a=a, b=b, amplitude=amplitude, lag=lag, r2=1-r2, ay=ay, az=az)


def write_csvs(results, output):
    shared = ["case", "source_file", "frequency_hz", "phase_source", "phase_offset_rad", "u_inf", "x_over_D", "coverage"]
    with (output / "wake_pulsing_summary.csv").open("w", newline="", encoding="utf-8") as sf, \
         (output / "wake_pulsing_phase.csv").open("w", newline="", encoding="utf-8") as pf:
        summary, phases = csv.writer(sf), csv.writer(pf)
        summary.writerow(shared + ["mean_deficit", "A_D", "harmonic_a", "harmonic_b", "harmonic_phase_rad", "harmonic_r2", "A_y_over_D", "A_z_over_D"])
        phases.writerow(shared + ["phase_degrees", "D_m", "centroid_y_over_D", "centroid_z_over_D"])
        for r in results:
            for i, x in enumerate(r["x"]):
                base = [r["case"], r["path"], r["frequency"], r["phase_source"], r["phase_offset"], r["u_inf"], x, r["coverage"][i]]
                summary.writerow(base + [r[k][i] for k in ("offset", "amplitude", "a", "b", "lag", "r2", "ay", "az")])
                for j, phase in enumerate(r["phase"]):
                    phases.writerow(base + [np.degrees(phase)] + [r[k][j, i] for k in ("dm", "cy", "cz")])


def plot_results(results, output, locations, bounds, min_harmonic_r2, sector=None):
    cases = sorted({r["case"] for r in results})
    # Use the shared case palette from the wake-profile line plots.
    colors = {case: _case_style(case, index)["color"] for index, case in enumerate(cases)}
    for centroid in (False, True):
        fig = plt.figure(figsize=(13, 8), layout="constrained")
        gs = fig.add_gridspec(len(locations), 2, width_ratios=[1.25, 1])
        right = fig.add_subplot(gs[:, 1])
        for row, target in enumerate(locations):
            ax = fig.add_subplot(gs[row, 0])
            for case in cases:
                candidates = [(abs(r["x"][i] - target), r, i)
                              for r in results if r["case"] == case and r["x"][0] <= target <= r["x"][-1]
                              for i in [nearest_index(r["x"], target)]
                              if np.isfinite(r["amplitude"][i]) and r["r2"][i] > min_harmonic_r2]
                if not candidates:
                    continue
                _, r, i = min(candidates, key=lambda item: item[0])
                degrees = np.degrees(r["phase"])
                for key, style in ([("cy", "-"), ("cz", "--")] if centroid else [("dm", "-")]):
                    values = r[key][:, i]
                    if centroid:
                        values = values - np.mean(values)
                    ax.plot(np.r_[degrees[-1]-360, degrees, degrees[0]+360],
                            np.r_[values[-1], values, values[0]], style, color=colors[case], linewidth=1.5,
                            label=f'{case} ({r["x"][i]:.3f}D)' if key != "cz" else None)
            ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360],
                   ylabel=r"$\Delta c/D$" if centroid else r"$D_m(x,\phi)$",
                   title=f"Selected x/D = {target:g} (actual positions in legend)")
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7, ncol=2)
            else:
                ax.text(.5, .5, f"No planes pass R² > {min_harmonic_r2:g}", transform=ax.transAxes, ha="center")
            if row == len(locations)-1:
                ax.set_xlabel(r"Saved phase $\phi$ [degrees]")
        for case in cases:
            for n, r in enumerate(sorted([r for r in results if r["case"] == case], key=lambda r: r["x"][0])):
                for key, style in ([("ay", "-"), ("az", "--")] if centroid else [("amplitude", "-")]):
                    visible = np.where(r["r2"] > min_harmonic_r2, r[key], np.nan)
                    right.plot(r["x"], visible, style, color=colors[case], linewidth=1.7,
                               label=case if n == 0 and key != "az" else None)
        right.set(xlabel=r"$x/D$", ylabel=r"$A_c/D$" if centroid else r"$A_D$", ylim=(0, None),
                  title=fr"First-harmonic semi-amplitude ($R^2 > {min_harmonic_r2:g}$)")
        right.grid(alpha=.2)
        right.legend()
        region = "circular sector" if sector else "measured window"
        title = "Observed positive-deficit centroid (solid: vertical y; dashed: lateral z)" if centroid else f"Wake deficit pulsing — area mean over the {region}"
        frequencies = ", ".join(f"{v:g}" for v in sorted({r["frequency"] for r in results}))
        geometry = (f"window y=[{bounds[0]:.1f}, {bounds[1]:.1f}], z=[{bounds[2]:.1f}, {bounds[3]:.1f}]"
                    if sector is None else f'sector centre (y,z)=({sector["axis_y"]:g},{sector["axis_z"]:g}), R={sector["radius"]:g}, angles {sector["angles"][0]:g} to {sector["angles"][1]:g} deg from +y')
        fig.suptitle(title + f"\nSaved phase averaging: {frequencies} Hz; {geometry}; displayed only for $R^2 > {min_harmonic_r2:g}$", fontsize=11)
        name = "wake_centroid_pulsing" if centroid else "wake_deficit_pulsing"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}.{extension}", dpi=200)
        plt.close(fig)


def plot_sector(sector, bounds, output):
    """Show the actual requested geometry in the measured y-z coordinates."""
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.add_patch(Rectangle((bounds[2], bounds[0]), bounds[3]-bounds[2], bounds[1]-bounds[0],
                           fill=False, linestyle="--", edgecolor="gray", label="Common field of view"))
    ax.add_patch(Wedge((sector["axis_z"], sector["axis_y"]), sector["radius"],
                      90-sector["angles"][1], 90-sector["angles"][0],
                      facecolor="tab:blue", alpha=.35, label="Averaging sector"))
    ax.plot(sector["axis_z"], sector["axis_y"], "k+", label="Rotor centre")
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set(xlabel="Lateral z [input coordinate units]", ylabel="Vertical y [input coordinate units]",
           title=f'R = {sector["radius"]:g}; angles {sector["angles"][0]:g} to {sector["angles"][1]:g} degrees from +y')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(output / "sector_window.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output-folder", type=Path, default=Path("outputs/wake_pulsing_sector"))
    parser.add_argument("--rotor-diameter", type=float, default=1200, help="Same coordinate units as input")
    parser.add_argument("--locations", type=float, nargs="+", default=[1, 2.5, 3.75], help="Selected x/D")
    parser.add_argument("--bounds", type=float, nargs=4, metavar=("YMIN", "YMAX", "ZMIN", "ZMAX"), help="Default: intersection of measured y-z extents across all files")
    parser.add_argument("--rotor-y", "--axis-y", type=float, default=0, help="Sector centre y, as in radial deficit")
    parser.add_argument("--rotor-z", "--axis-z", type=float, default=0, help="Sector centre z, as in radial deficit")
    parser.add_argument("--radius", type=float, help="Sector outer radius; default D/2")
    parser.add_argument("--sector-angles", type=float, nargs=2, default=[-25, 25], metavar=("START", "STOP"), help="Degrees from +y toward +z; default -25 25")
    parser.add_argument("--window", choices=["sector", "rectangle"], default="sector")
    parser.add_argument("--min-valid-fraction", type=float, default=.3)
    parser.add_argument("--min-coverage", type=float, default=.6, help="Minimum area fraction valid at every phase (default: 0.6); actual coverage is exported")
    parser.add_argument("--min-harmonic-r2", type=float, default=.9,
                        help="Show phase traces and harmonic amplitudes only above this first-harmonic R² (default: 0.9)")
    parser.add_argument("--edge-slices", type=int, default=1)
    parser.add_argument("--u-inf", type=float)
    parser.add_argument("--include-static", action="store_true", help="Include the arbitrary-phase static baseline")
    parser.add_argument("--exclude-1p625d", action="store_true",
                        help="Discard all 1.625D phase-average volumes before calculation")
    args = parser.parse_args()
    sector = None
    if args.window == "sector":
        if args.bounds is not None:
            parser.error("--bounds applies only to --window rectangle")
        sector = dict(axis_y=args.rotor_y, axis_z=args.rotor_z,
                      radius=args.radius if args.radius is not None else args.rotor_diameter/2,
                      angles=args.sector_angles)
        try:
            sector_area_weights(np.arange(2.), np.arange(2.), **sector)
        except ValueError as exc:
            parser.error(str(exc))
    if not np.isfinite(args.rotor_diameter) or args.rotor_diameter <= 0 or args.edge_slices < 0:
        parser.error("Diameter must be finite and positive; edge-slices must be nonnegative")
    if not 0 <= args.min_valid_fraction <= 1 or not 0 <= args.min_coverage <= 1 or not 0 <= args.min_harmonic_r2 <= 1:
        parser.error("Fractions and --min-harmonic-r2 must lie in [0, 1]")
    files = discover_phase_average_files(args.input)
    # Old manifests can point to copies outside the requested data directory.
    if args.input.is_dir():
        files = [p for p in files if p.is_relative_to(args.input.resolve())]
    files = [p for p in files if args.include_static or "static" not in p.parent.name.lower()]
    if args.exclude_1p625d:
        files = [p for p in files if not is_1p625d_case(p)]
    if not files:
        parser.error(f"No phase_average.nc files found in {args.input}")
    extents = []
    for path in files:
        with h5py.File(path) as f:
            extents.append([f["y"][0], f["y"][-1], f["z"][0], f["z"][-1]])
    extents = np.array(extents)
    common = [extents[:, 0].max(), extents[:, 1].min(), extents[:, 2].max(), extents[:, 3].min()]
    bounds = args.bounds or common
    if not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        parser.error("The integration window must have finite, positive width and height")
    if bounds[0] < common[0] or bounds[1] > common[1] or bounds[2] < common[2] or bounds[3] > common[3]:
        parser.error("Requested window is not contained in all measured planes")
    results = []
    for path in files:
        result = calculate(path, bounds, args.rotor_diameter, args.min_valid_fraction,
                           args.min_coverage, args.edge_slices, args.u_inf, sector)
        print(f'{result["label"]}: {np.isfinite(result["amplitude"]).sum()}/{len(result["x"])} accepted x slices; saved frequency {result["frequency"]:g} Hz', flush=True)
        results.append(result)
    if not any(np.isfinite(r["amplitude"]).any() for r in results):
        parser.error("No planes pass the coverage and phase-validity thresholds")
    args.output_folder.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings["bounds"] = [float(v) for v in bounds] if sector is None else None
    settings["sector"] = sector
    settings["definition"] = "Area mean of signed deficit on support fixed across phases at each x; support may differ between x and cases. Centroid uses positive deficit in the selected region. Sector coverage is relative to the full geometric sector, including unmeasured area; boundary cell areas use 16x16 midpoint quadrature."
    (args.output_folder / "calculation_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    write_csvs(results, args.output_folder)
    plot_results(results, args.output_folder, args.locations, bounds, args.min_harmonic_r2, sector)
    if sector is not None:
        plot_sector(sector, common, args.output_folder)
    print(f"Plots and calculations saved to {args.output_folder.resolve()}")


if __name__ == "__main__":
    main()
