"""Combine exported slab vertical velocities with one curve/legend per case."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Figures folder containing exported slab CSVs")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--central-fraction", type=float, default=1.0,
                        help="Fraction of each volume's downstream extent to retain")
    parser.add_argument("--mask-description", default="",
                        help="Additional plot label describing the averaging mask")
    args = parser.parse_args()
    if not 0 < args.central_fraction <= 1:
        parser.error("--central-fraction must be in (0, 1]")
    records, sources, settings = {}, [], None
    for path in sorted(args.input.glob("*/slab_vertical_velocity.csv")):
        metadata = json.loads((path.parent / "manifest.json").read_text())
        current = (metadata["min_valid_fraction"], metadata["comparison"], metadata["rotor_diameter"])
        if metadata.get("quantity", "v") != "v":
            raise ValueError(f"Not a vertical velocity export: {path}")
        if settings is not None and current != settings:
            raise ValueError(f"Inconsistent coverage or diameter settings in {path}")
        settings = current
        sources.append(str(path.resolve()))
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                match = re.fullmatch(r"(.+)_\d+(?:\.\d+)?D", row["volume"])
                if not match:
                    raise ValueError(f"Cannot identify case from {row['volume']}")
                item = dict(case=match[1], volume=row["volume"], x_over_d=float(row["x_over_d"]),
                            slab_mean_v_m_s=float(row["slab_mean_v_m_s"]),
                            retained_voxels=int(row["retained_voxels"]),
                            retained_slab_fraction=float(row["retained_slab_fraction"]))
                key = (item["volume"], item["x_over_d"])
                if key in records and not np.isclose(records[key]["slab_mean_v_m_s"], item["slab_mean_v_m_s"], equal_nan=True):
                    raise ValueError(f"Conflicting duplicate sample: {key}")
                records[key] = item
    if not records:
        raise ValueError("No vertical-velocity slab exports found")
    ranges = {}
    for row in records.values():
        ranges.setdefault(row["volume"], []).append(row["x_over_d"])
    selection = []
    for volume, x in ranges.items():
        low, high = min(x), max(x)
        margin = (high-low) * (1-args.central_fraction) / 2
        selection.append(dict(volume=volume, full_x_min=low, full_x_max=high,
                              retained_x_min=low+margin, retained_x_max=high-margin))
    bounds = {r["volume"]: (r["retained_x_min"], r["retained_x_max"]) for r in selection}
    records = {key: row for key, row in records.items()
               if bounds[row["volume"]][0] <= row["x_over_d"] <= bounds[row["volume"]][1]}
    if not records:
        raise ValueError("No slab positions inside selected central fraction")
    cases = sorted({r["case"] for r in records.values()}, key=lambda c: (0 if c=="Flow" else 1 if c=="Static" else 2, c))
    colors = ["#666666", "#111111", "#0072b2", "#56b4e9", "#d55e00", "#cc79a7"]
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    interactive = go.Figure()
    for i, case in enumerate(cases):
        color = colors[i % len(colors)]
        groups = {}
        for row in records.values():
            if row["case"] == case:
                groups.setdefault(row["volume"], []).append(row)
        xx, yy, hover = [], [], []
        for j, (volume, rows) in enumerate(sorted(groups.items(), key=lambda kv: min(r["x_over_d"] for r in kv[1]))):
            rows.sort(key=lambda r: r["x_over_d"])
            x = [r["x_over_d"] for r in rows]
            y = [r["slab_mean_v_m_s"] for r in rows]
            ax.plot(x, y, color=color, lw=1.7, label=case if j==0 else None,
                    linestyle="--" if case=="Flow" else "-")
            xx.extend(x+[None]); yy.extend(y+[None])
            hover.extend([[volume, r["retained_voxels"], r["retained_slab_fraction"]] for r in rows])
            hover.append(["", None, None])
        interactive.add_trace(go.Scatter(
            x=xx, y=yy, name=case, legendgroup=case, mode="lines", connectgaps=False,
            line=dict(color=color, width=2, dash="dash" if case=="Flow" else "solid"),
            customdata=hover,
            hovertemplate="%{customdata[0]}<br>x/D=%{x:.3f}<br>Mean v=%{y:.5f} m/s"
                          "<br>Retained voxels=%{customdata[1]}<br>Retained slab fraction=%{customdata[2]:.1%}<extra>%{fullData.name}</extra>"))
    title = f"Slab-averaged vertical velocity by case · voxel temporal coverage > {settings[0]:.0%}"
    if args.central_fraction < 1:
        title += f" · central {args.central_fraction:.0%} of each volume"
    if args.mask_description:
        title += "\n" + args.mask_description
    ax.set(title=title, xlabel="Downstream $x/D$", ylabel="Slab mean vertical velocity [m/s]")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.grid(alpha=0.2)
    ax.legend(ncol=len(cases), loc="upper center", bbox_to_anchor=(0.5, -0.13))
    args.output_folder.mkdir(parents=True, exist_ok=False)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_folder / f"vertical_velocity_by_case.{ext}", dpi=220)
    plt.close(fig)
    interactive.add_hline(y=0, line_color="gray", line_width=1)
    interactive.update_layout(title=title.replace("\n", "<br>"), template="plotly_white",
                              xaxis_title="Downstream x/D", yaxis_title="Slab mean vertical velocity [m/s]",
                              legend=dict(orientation="h", y=-0.15), margin=dict(b=100))
    interactive.write_html(args.output_folder / "vertical_velocity_by_case.html", include_plotlyjs=True,
                           config=dict(scrollZoom=True, displaylogo=False))
    with (args.output_folder / "vertical_velocity_by_case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(records.values()))))
        writer.writeheader()
        writer.writerows(sorted(records.values(), key=lambda r: (cases.index(r["case"]), r["volume"], r["x_over_d"])))
    (args.output_folder / "manifest.json").write_text(json.dumps(dict(
        sources=sources, cases=cases, min_valid_fraction=settings[0], comparison=settings[1],
        rotor_diameter=settings[2], samples=len(records),
        central_fraction=args.central_fraction, volume_selection=selection,
        mask_description=args.mask_description,
        method="Existing slab averages grouped by case. Separate volume segments; gaps and overlaps preserved."), indent=2))
    print(f"Combined {len(records)} slab samples from {len(sources)} exports into {len(cases)} cases: {', '.join(cases)}")


if __name__ == "__main__":
    main()
