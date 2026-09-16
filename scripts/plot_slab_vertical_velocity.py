"""Plot spatial mean u, v, or w per x slab from temporal mean volumes."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def slab_means(velocity, counts, n_times, threshold):
    if n_times <= 0 or counts.shape != velocity.shape:
        raise ValueError("Invalid sample count or incompatible count shape")
    valid = np.isfinite(velocity) & (counts / n_times > threshold)
    retained = valid.sum(axis=(0, 1))
    total = np.where(valid, velocity, 0).sum(axis=(0, 1))
    mean = np.divide(total, retained, out=np.full(total.shape, np.nan), where=retained > 0)
    return mean, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--case-pattern", default="Flow*")
    parser.add_argument("--quantity", choices=("u", "v", "w"), default="v",
                        help="Velocity component: u streamwise, v vertical, w lateral")
    parser.add_argument("--min-valid-fraction", type=float, default=0.8)
    parser.add_argument("--rotor-diameter", type=float, default=1200,
                        help="Diameter in coordinate units (mm for these files)")
    args = parser.parse_args()
    direction = {"u": "streamwise", "v": "vertical", "w": "lateral"}[args.quantity]
    dataset = f"{args.quantity}_mean"
    output_stem = f"slab_{direction}_velocity"
    ylabel = f"Slab mean {direction} velocity {args.quantity} [m/s]"
    if not 0 <= args.min_valid_fraction < 1 or args.rotor_diameter <= 0:
        parser.error("Require coverage fraction in [0, 1) and a positive diameter")
    volumes, audit = [], []
    for path in args.input.glob(f"{args.case_pattern}/mean.nc"):
        if path.parent.name.endswith("_z0"):
            continue
        with h5py.File(path) as f:
            velocity = f[dataset][:].astype(float)
            count_name = "vector_count" if "vector_count" in f else f"{args.quantity}_count"
            if count_name not in f:
                raise ValueError(f"Missing {args.quantity}-velocity sample counts: {path}")
            n_times = (int(f.attrs["input_shape_time_z_y_x"][0])
                       if "input_shape_time_z_y_x" in f.attrs else len(f["t"]))
            mean, retained = slab_means(velocity, f[count_name][:], n_times, args.min_valid_fraction)
            x = f["x"][:].astype(float)
            if velocity.shape != (len(f["z"]), len(f["y"]), len(x)):
                raise ValueError(f"Unexpected velocity dimensions: {path}")
            total_voxels = velocity.shape[0] * velocity.shape[1]
        volumes.append((path.parent.name, x, mean, retained, total_voxels))
        audit.append(dict(source=str(path.resolve()), total_samples=n_times, count_dataset=count_name))
    if not volumes:
        raise ValueError("No matching mean.nc volumes found")
    volumes.sort(key=lambda v: float(np.mean(v[1])))
    args.output_folder.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    interactive, rows = go.Figure(), []
    colors = ["#4477aa", "#ee7733", "#228833", "#aa3377", "#6655aa"]
    for i, (name, x, mean, retained, total_voxels) in enumerate(volumes):
        xd, color = x / args.rotor_diameter, colors[i % len(colors)]
        ax.plot(xd, mean, "-o", color=color, ms=2.5, lw=1.5, label=name.replace("_", " "))
        interactive.add_trace(go.Scatter(
            x=xd, y=mean, mode="lines+markers", name=name,
            line=dict(color=color), marker=dict(size=4), connectgaps=False,
            customdata=np.column_stack([retained, retained / total_voxels]),
            hovertemplate="x/D=%{x:.3f}<br>Mean " + args.quantity + "=%{y:.5f} m/s"
                          "<br>Retained voxels=%{customdata[0]:.0f}"
                          "<br>Retained slab fraction=%{customdata[1]:.1%}<extra>%{fullData.name}</extra>"))
        for xi, di, vi, ni in zip(x, xd, mean, retained):
            rows.append(dict(volume=name, x=xi, x_over_d=di, **{f"slab_mean_{args.quantity}_m_s": vi},
                             retained_voxels=int(ni), total_slab_voxels=total_voxels,
                             retained_slab_fraction=ni / total_voxels))
    title = f"{args.case_pattern} cases: slab-averaged {direction} velocity\nVoxel temporal coverage > {args.min_valid_fraction:.0%}"
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set(xlabel="Downstream $x/D$", ylabel=ylabel, title=title)
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_folder / f"{output_stem}.{ext}", dpi=220)
    plt.close(fig)
    interactive.add_hline(y=0, line_dash="dash", line_color="black")
    interactive.update_layout(title=title.replace("\n", "<br>"), template="plotly_white",
                              xaxis_title="Downstream x/D", yaxis_title=ylabel,
                              hovermode="closest", legend=dict(orientation="h"))
    interactive.write_html(args.output_folder / f"{output_stem}.html", include_plotlyjs=True,
                           config=dict(scrollZoom=True, displaylogo=False))
    with (args.output_folder / f"{output_stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output_folder / "manifest.json").write_text(json.dumps(dict(
        min_valid_fraction=args.min_valid_fraction, comparison="strictly greater than",
        rotor_diameter=args.rotor_diameter, sources=audit, quantity=args.quantity,
        method=f"Arithmetic spatial mean of finite {dataset} over qualifying y-z voxels at each x."
               " Equal weight per voxel; all slab positions retained; no smoothing or joining between volumes."
               " Empty slabs have NaN means."), indent=2))
    print(f"Saved {len(volumes)} volume curves and {len(rows)} slabs in {args.output_folder}")


if __name__ == "__main__":
    main()
