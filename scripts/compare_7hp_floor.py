"""Compare 7-hole-probe profiles and velocity distributions with/without a floor.

Port of computeVelo7HPv2.m and profiles_7HP.m: height is column 1 + 2
(mm), streamwise velocity is column 4 (m/s), U = abs(mean(u)), and
Iu = std(u, ddof=1)/U. Only heights on the 100 mm grid are retained.
Defaults Uref=4 m/s, D=1.2 m, and height offset=100 mm follow profiles_7HP.m.

Pair labels and the 1061 -> 1063 continuation were checked against sheet PD
of testdiary7HP.xlsx. No turbine is present in the 1060/1061 pair.
Distributions are shown separately at common measured heights; samples at
different heights are never pooled. Histograms use -u (positive downstream),
NOT abs(u), preserving any reverse-flow samples. Profiles use abs(mean(u))
exactly as in MATLAB. Iu is a fluctuation measure, not an uncertainty interval.

Additional *_col4/5/6_* files compare all three components using SIGNED means
and sample standard deviations in m/s. Column 4 is negated for downstream
positive; columns 5 and 6 retain the recorded signs. Column 6 is suspected to
be vertical, but the supplied MATLAB scripts do not establish this mapping.
Both transverse columns are therefore labelled by column number by default.
Use --vertical-column 6 (or 5) once confirmed, and --vertical-sign -1 if needed.
Transverse fluctuations are not divided by their near-zero mean velocities.
Existing unsuffixed plots retain the original MATLAB streamwise calculation.

Run from the repository root: python scripts/compare_7hp_floor.py
Dependencies: numpy, matplotlib (already in this project's requirements).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = (
    Path.home() / "OneDrive - Delft University of Technology" / "Desktop" / "7HP_data"
)
PAIRS = (
    (1020, 1027, "Static turbine, x/D = 5"),
    (1060, 1061, "No turbine, x/D = 3"),
)


def load_samples(paths: list[Path], column: int = 4, sign: int = 1) -> tuple[dict[float, np.ndarray], dict]:
    """Merge continuation samples by height; reject transitions and nonfinite rows."""
    pieces: dict[float, list[np.ndarray]] = {}
    audit = {"files": [str(p.resolve()) for p in paths], "rows": 0,
             "nonfinite_rows": 0, "off_grid_rows": 0}
    for path in paths:
        data = np.loadtxt(path, usecols=(0, 1, column - 1), ndmin=2)
        data[:, 2] *= sign
        audit["rows"] += len(data)
        finite = np.isfinite(data).all(axis=1)
        audit["nonfinite_rows"] += int((~finite).sum())
        data = data[finite]
        height = data[:, 0] + data[:, 1]
        snapped = np.round(height / 100) * 100
        on_grid = np.isclose(height, snapped, rtol=0, atol=1e-6)
        audit["off_grid_rows"] += int((~on_grid).sum())
        for h in np.unique(snapped[on_grid]):
            pieces.setdefault(float(h), []).append(data[on_grid & (snapped == h), 2])
    samples = {h: np.concatenate(pieces[h]) for h in sorted(pieces)}
    if not samples:
        raise ValueError(f"No valid samples on the 100 mm height grid in {paths}")
    audit["retained_rows"] = sum(len(v) for v in samples.values())
    return samples, audit


def profile(samples: dict[float, np.ndarray], uref: float, diameter: float,
            offset: float, absolute_mean: bool = True) -> dict[float, dict]:
    rows = {}
    for h, u in sorted(samples.items()):
        mean = float(np.mean(u))
        if absolute_mean:
            mean = abs(mean)
        std = float(np.std(u, ddof=1)) if len(u) > 1 else float("nan")
        rows[h] = {
            "height_mm": h, "z_over_D": (h - offset) / (1000 * diameter),
            "n": len(u), "U_m_s": mean, "U_over_Uref": mean / uref,
            "std_u_m_s": std, "Iu_percent": 100 * std / mean if mean else float("nan"),
        }
    return rows


def component_profile(samples: dict, uref: float, diameter: float, offset: float) -> dict:
    """Signed means and sample standard deviations; no division by transverse mean."""
    rows = {}
    for h, values in samples.items():
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        rows[h] = {"height_mm": h, "z_over_D": (h - offset) / (1000 * diameter),
                   "n": len(values), "mean_m_s": mean, "mean_over_Uref": mean / uref,
                   "std_m_s": std, "std_over_Uref": std / uref}
    return rows


def compare_component(output: Path, name: str, title: str, samples: list[dict],
                      labels: list[str], args: argparse.Namespace) -> dict:
    profiles = [component_profile(s, args.u_ref, args.diameter, args.height_offset_mm)
                for s in samples]
    common = sorted(profiles[0].keys() & profiles[1].keys())
    if not common:
        raise ValueError(f"{name}: no common measured heights")
    delta = []
    for h in common:
        f, n = (p[h] for p in profiles)
        delta.append({"height_mm": h, "z_over_D": f["z_over_D"],
                      "floor_mean_m_s": f["mean_m_s"], "no_floor_mean_m_s": n["mean_m_s"],
                      "delta_mean_m_s": f["mean_m_s"] - n["mean_m_s"],
                      "delta_std_m_s": f["std_m_s"] - n["std_m_s"]})
    for condition, prof in zip(("floor", "no_floor"), profiles):
        write_csv(output / f"{name}_{condition}_profile.csv", list(prof.values()))
    write_csv(output / f"{name}_differences.csv", delta)
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True, layout="constrained")
    for prof, label, color in zip(profiles, labels, ("#cc6232", "#246aa5")):
        rows = list(prof.values())
        for ax, key in zip(axes[:2], ("mean_m_s", "std_m_s")):
            ax.plot([r[key] for r in rows], [r["z_over_D"] for r in rows],
                    "o-", color=color, label=label)
    for ax, key in zip(axes[2:], ("delta_mean_m_s", "delta_std_m_s")):
        ax.plot([r[key] for r in delta], [r["z_over_D"] for r in delta], "o-", color="#5a416b")
    for ax, label in zip(axes, ("Signed mean (m/s)", "Fluctuation std (m/s)",
                                "Floor − no floor: mean (m/s)", "Floor − no floor: std (m/s)")):
        ax.set_xlabel(label)
        ax.axvline(0, color="0.5", linewidth=.8)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("z / D")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}_profiles.{extension}", dpi=180)
    plt.close(fig)

    ncols = min(3, len(common))
    nrows = (len(common) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4*ncols, 3.2*nrows),
                             squeeze=False, layout="constrained")
    histograms = []
    for ax, h in zip(axes.flat, common):
        values = [s[h] for s in samples]
        edges = np.histogram_bin_edges(np.concatenate(values), bins=args.bins)
        densities = []
        for v, label, color in zip(values, labels, ("#cc6232", "#246aa5")):
            density, _ = np.histogram(v, bins=edges, density=True)
            densities.append(density)
            ax.stairs(density, edges, label=label, color=color)
        for i in range(args.bins):
            histograms.append({"height_mm": h, "bin_left_m_s": edges[i],
                               "bin_right_m_s": edges[i+1], "floor_density": densities[0][i],
                               "no_floor_density": densities[1][i]})
        ax.set_title(f"h = {h:g} mm; z/D = {profiles[0][h]['z_over_D']:.3f}")
        ax.set_xlabel("Signed velocity (m/s)")
        ax.set_ylabel("Probability density (s/m)")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(common):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title + " — distributions at common heights")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}_distributions.{extension}", dpi=180)
    plt.close(fig)
    write_csv(output / f"{name}_histograms.csv", histograms)
    changes = [r["delta_mean_m_s"] for r in delta]
    return {"comparison": name, "common_heights": len(common),
            "equal_height_mean_delta_m_s": float(np.mean(changes)),
            "min_delta_m_s": min(changes), "max_delta_m_s": max(changes)}


def differences(floor: dict, no_floor: dict) -> list[dict]:
    """Use only matching measured heights, without interpolation/extrapolation."""
    rows = []
    for h in sorted(floor.keys() & no_floor.keys()):
        f, n = floor[h], no_floor[h]
        delta = f["U_m_s"] - n["U_m_s"]
        rows.append({
            "height_mm": h, "z_over_D": f["z_over_D"],
            "floor_U_m_s": f["U_m_s"], "no_floor_U_m_s": n["U_m_s"],
            "delta_U_m_s": delta,
            "delta_U_percent_of_no_floor": 100 * delta / n["U_m_s"] if n["U_m_s"] else float("nan"),
            "delta_U_over_Uref": f["U_over_Uref"] - n["U_over_Uref"],
            "delta_Iu_percentage_points": f["Iu_percent"] - n["Iu_percent"],
        })
    if not rows:
        raise ValueError("Recordings have no common measured heights")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(output: Path, pair_name: str, title: str, samples: list[dict],
              profiles: list[dict], delta: list[dict], labels: list[str],
              uref: float, bins: int) -> None:
    colors = ("#cc6232", "#246aa5")
    fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True, layout="constrained")
    for prof, label, color in zip(profiles, labels, colors):
        rows = list(prof.values())
        z = [r["z_over_D"] for r in rows]
        for ax, key in zip(axes[:2], ("U_over_Uref", "Iu_percent")):
            ax.plot([r[key] for r in rows], z, "o-", label=label, color=color)
    for ax, key in zip(axes[2:], ("delta_U_over_Uref", "delta_Iu_percentage_points")):
        ax.plot([r[key] for r in delta], [r["z_over_D"] for r in delta], "o-", color="#5a416b")
        ax.axvline(0, color="0.5", linewidth=1)
    for ax, label in zip(axes, ("U / Uref", "Iu (%)", "Floor − no floor: ΔU / Uref",
                                "Floor − no floor: ΔIu (pp)")):
        ax.set_xlabel(label)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("z / D")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{pair_name}_profiles.{extension}", dpi=180)
    plt.close(fig)

    common = [r["height_mm"] for r in delta]
    ncols = min(3, len(common))
    nrows = (len(common) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows),
                             squeeze=False, layout="constrained")
    histogram_rows = []
    for ax, h in zip(axes.flat, common):
        values = [-s[h] / uref for s in samples]
        # Identical edges and unit-integral PDFs within each height comparison.
        edges = np.histogram_bin_edges(np.concatenate(values), bins=bins)
        densities = []
        for v, label, color in zip(values, labels, colors):
            density, _ = np.histogram(v, bins=edges, density=True)
            densities.append(density)
            ax.stairs(density, edges, label=label, color=color, linewidth=1.5)
        for i in range(bins):
            histogram_rows.append({"height_mm": h, "bin_left_minus_u_over_Uref": edges[i],
                                   "bin_right_minus_u_over_Uref": edges[i + 1],
                                   "floor_density": densities[0][i],
                                   "no_floor_density": densities[1][i]})
        ax.set_title(f"h = {h:g} mm; z/D = {profiles[0][h]['z_over_D']:.3f}")
        ax.set_xlabel("−u / Uref (positive downstream)")
        ax.set_ylabel("Probability density")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(common):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title + " — distributions at common heights")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{pair_name}_distributions.{extension}", dpi=180)
    plt.close(fig)
    write_csv(output / f"{pair_name}_histograms.csv", histogram_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT / "Data7HPpost",
                        help="Directory containing post_recNNNN.txt")
    parser.add_argument("--output", type=Path, default=Path("outputs/7hp_floor_comparison/three_components"))
    parser.add_argument("--u-ref", type=float, default=4.0, help="Reference velocity in m/s")
    parser.add_argument("--diameter", type=float, default=1.2, help="Reference diameter in m")
    parser.add_argument("--height-offset-mm", type=float, default=100.0)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--vertical-column", type=int, choices=(5, 6),
                        help="Identify the vertical component; otherwise transverse columns remain unlabelled")
    parser.add_argument("--vertical-sign", type=int, choices=(-1, 1), default=1,
                        help="Multiplier for vertical velocity (default preserves recorded sign)")
    parser.add_argument("--no-continuation", action="store_true",
                        help="Exclude 1063; 1061 alone contains only h=1200 mm")
    args = parser.parse_args()
    if args.vertical_column is None and args.vertical_sign != 1:
        parser.error("--vertical-sign requires --vertical-column")
    if (not np.isfinite([args.u_ref, args.diameter, args.height_offset_mm]).all()
            or args.u_ref <= 0 or args.diameter <= 0 or args.bins < 1):
        parser.error("Uref and diameter must be finite and positive; offset finite; bins >= 1")
    prepared = []
    manifest = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "metadata_source": "testdiary7HP.xlsx, sheet PD, rows 18, 23, 58, 59",
                "method": "See script docstring. Differences are floor minus no floor at common heights.",
                "recordings": {}}
    for floor, no_floor, title in PAIRS:
        samples, profiles, labels = [], [], []
        for rec, condition in ((floor, "Floor"), (no_floor, "No floor")):
            ids = [rec, 1063] if rec == 1061 and not args.no_continuation else [rec]
            paths = [args.data_dir / f"post_rec{i}.txt" for i in ids]
            for path in paths:
                if not path.is_file():
                    parser.error(f"Missing recording: {path}. Set --data-dir to Data7HPpost.")
            data, audit = load_samples(paths)
            manifest["recordings"][str(rec)] = audit
            samples.append(data)
            profiles.append(profile(data, args.u_ref, args.diameter, args.height_offset_mm))
            labels.append(f"{condition}: {' + '.join(map(str, ids))}")
        delta = differences(*profiles)
        prepared.append((floor, no_floor, title, samples, profiles, delta, labels))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for floor, no_floor, title, samples, profiles, delta, labels in prepared:
        name = f"{floor}_vs_{no_floor}"
        for rec, prof in zip((floor, no_floor), profiles):
            write_csv(args.output / f"{rec}_profile.csv", list(prof.values()))
        write_csv(args.output / f"{name}_differences.csv", delta)
        plot_pair(args.output, name, title, samples, profiles, delta, labels, args.u_ref, args.bins)
        change = np.array([r["delta_U_m_s"] for r in delta])
        summary.append({"pair": name, "case": title, "common_heights": len(delta),
                        "equal_height_mean_delta_U_m_s": float(change.mean()),
                        "min_delta_U_m_s": float(change.min()),
                        "max_delta_U_m_s": float(change.max())})
        print(f"{name}: {len(delta)} common heights; equal-height mean delta U = {change.mean():+.4f} m/s")
    write_csv(args.output / "summary.csv", summary)
    component_summary = []
    manifest["components"] = {}
    for column in (4, 5, 6):
        key = f"col{column}"
        if column == 4:
            description, sign = "Streamwise (negative of recorded column 4)", -1
        elif column == args.vertical_column:
            description, sign = f"Vertical (column {column}, sign x{args.vertical_sign:+d})", args.vertical_sign
        else:
            description, sign = f"Transverse column {column} (recorded sign)", 1
        manifest["components"][key] = {"column": column, "sign": sign, "label": description}
        for floor, no_floor, title in PAIRS:
            samples, labels = [], []
            for rec, condition in ((floor, "Floor"), (no_floor, "No floor")):
                ids = [rec, 1063] if rec == 1061 and not args.no_continuation else [rec]
                data, audit = load_samples([args.data_dir / f"post_rec{i}.txt" for i in ids], column, sign)
                manifest["recordings"][f"{rec}_{key}"] = audit
                samples.append(data)
                labels.append(f"{condition}: {' + '.join(map(str, ids))}")
            name = f"{floor}_vs_{no_floor}_{key}"
            result = compare_component(args.output, name, title + " — " + description, samples, labels, args)
            component_summary.append(result)
            print(f"{name}: equal-height signed mean delta = {result['equal_height_mean_delta_m_s']:+.4f} m/s")
    write_csv(args.output / "component_summary.csv", component_summary)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved figures, CSV tables, and input audit to {args.output.resolve()}")


if __name__ == "__main__":
    main()
