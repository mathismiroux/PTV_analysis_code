"""Fit individual and pooled lines to interior measured wake-center traces."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fit_line(x, y):
    if len(x) < 3 or np.ptp(x) == 0:
        return dict(n=len(x), x_min=float(x.min()) if len(x) else float("nan"),
                    x_max=float(x.max()) if len(x) else float("nan"),
                    slope=float("nan"), intercept=float("nan"),
                    r_squared=float("nan"), rmse=float("nan"),
                    status="unavailable: fewer than three finite center points")
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope*x + intercept)
    ss = np.sum((y-y.mean())**2)
    return dict(n=len(x), x_min=float(x.min()), x_max=float(x.max()),
                slope=float(slope), intercept=float(intercept),
                r_squared=float(1-np.sum(residual**2)/ss) if ss > 0 else None,
                rmse=float(np.sqrt(np.mean(residual**2))), status="fitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--edge-fraction", type=float, default=0.2)
    parser.add_argument("--case-label", default="Static wake center")
    args = parser.parse_args()
    if not 0 <= args.edge_fraction < 0.5:
        parser.error("--edge-fraction must be between 0 and 0.5")
    source_manifest = json.loads((args.input.parent / "manifest.json").read_text())
    groups = {}
    with args.input.open(newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault(row["volume"], []).append(row)
    prepared, fits, samples = [], [], []
    for name, rows in groups.items():
        rows.sort(key=lambda r: float(r["x_over_d"]))
        x = np.array([float(r["x_over_d"]) for r in rows])
        y = np.array([float(r["measured_center_y_over_d"]) for r in rows])
        low, high = x.min()+args.edge_fraction*np.ptp(x), x.max()-args.edge_fraction*np.ptp(x)
        selected = (x >= low) & (x <= high) & np.isfinite(y)
        result = dict(volume=name, **fit_line(x[selected], y[selected]))
        fits.append(result)
        prepared.append((name, x, y, selected, result))
        for xi, yi, keep in zip(x, y, selected):
            samples.append(dict(volume=name, x_over_d=xi, center_y_over_d=yi,
                                included=bool(keep), local_fit=result["slope"]*xi+result["intercept"]))
    pooled_x = np.concatenate([x[keep] for _, x, _, keep, _ in prepared])
    pooled_y = np.concatenate([y[keep] for _, _, y, keep, _ in prepared])
    global_fit = dict(volume="Global", **fit_line(pooled_x, pooled_y))
    fits.append(global_fit)
    args.output_folder.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, layout="constrained")
    interactive = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Individual volume fits", "Global pooled fit"))
    colors = ["#4477aa", "#ee7733", "#228833", "#aa3377", "#6655aa"]
    for (name, x, y, keep, result), color in zip(prepared, colors):
        predicted = result["slope"]*x[keep]+result["intercept"]
        for panel, ax in enumerate(axes, start=1):
            ax.plot(x, y, color="#bdbdbd", lw=1, zorder=1)
            ax.plot(x[keep], y[keep], ".", color=color, ms=5, label=name.replace("_", " "))
            interactive.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#bdbdbd", width=1),
                                  showlegend=False, name=f"{name} full trace"), row=panel, col=1)
            interactive.add_trace(go.Scatter(x=x[keep], y=y[keep], mode="markers", marker=dict(color=color, size=5),
                                  legendgroup=name, showlegend=panel==1, name=name), row=panel, col=1)
        axes[0].plot(x[keep], predicted, "--", color=color, lw=2)
        interactive.add_trace(go.Scatter(x=x[keep], y=predicted, mode="lines", line=dict(color=color, dash="dash"),
                              legendgroup=name, showlegend=False,
                              name=f"{name}: slope={result['slope']:.5f}; R²={result['r_squared']:.3f}"), row=1, col=1)
    gx = np.array([pooled_x.min(), pooled_x.max()])
    gy = global_fit["slope"]*gx+global_fit["intercept"]
    label = f"Global: y/D = {global_fit['slope']:.5f}(x/D) + {global_fit['intercept']:.5f}; R² = {global_fit['r_squared']:.3f}"
    axes[1].plot(gx, gy, "--", color="black", lw=2, label=label)
    interactive.add_trace(go.Scatter(x=gx, y=gy, mode="lines", line=dict(color="black", dash="dash"),
                          name=label), row=2, col=1)
    for ax, title in zip(axes, ("Individual volume fits", "Global fit to all retained points")):
        ax.set_title(title)
        ax.set_ylabel("Measured center $y/D$")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=9)
    axes[1].legend(handles=[axes[1].lines[-1]], fontsize=9)
    axes[1].set_xlabel("Downstream $x/D$")
    threshold = source_manifest.get("min_valid_fraction")
    title = f"{args.case_label} · temporal coverage > {threshold:.0%} · central {1-2*args.edge_fraction:.0%} of each volume"
    missing = [f["volume"] for f in fits if f["status"] != "fitted"]
    if missing:
        title += "\nNo fit (undefined positive-deficit center): " + ", ".join(missing)
    fig.suptitle(title + "\nGray: full traces; colored points: retained; dashed: linear fits", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(args.output_folder / f"wake_center_linear_fits.{ext}", dpi=200)
    plt.close(fig)
    interactive.update_layout(title=title, template="plotly_white", height=850,
                              legend=dict(orientation="h", y=-0.12), margin=dict(b=150))
    interactive.update_xaxes(title_text="Downstream x/D", row=2, col=1)
    interactive.update_yaxes(title_text="Measured center y/D")
    interactive.write_html(args.output_folder / "wake_center_linear_fits.html", include_plotlyjs=True,
                           config=dict(displaylogo=False, scrollZoom=True))
    for filename, rows in (("linear_fits.csv", fits), ("fit_samples.csv", samples)):
        with (args.output_folder / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    # Show dependence on the chosen edge exclusion without optimizing it for R².
    sensitivity = []
    for edge in (0.15, 0.2, 0.25):
        xx, yy = [], []
        for name, x, y, _, _ in prepared:
            keep = (x >= x.min()+edge*np.ptp(x)) & (x <= x.max()-edge*np.ptp(x)) & np.isfinite(y)
            sensitivity.append(dict(edge_fraction=edge, volume=name, **fit_line(x[keep], y[keep])))
            xx.extend(x[keep]); yy.extend(y[keep])
        sensitivity.append(dict(edge_fraction=edge, volume="Global", **fit_line(np.array(xx), np.array(yy))))
    (args.output_folder / "manifest.json").write_text(json.dumps(dict(
        source=str(args.input.resolve()), edge_fraction=args.edge_fraction,
        method="Ordinary least squares; equal weight per retained downstream point. Global fit pools all volumes."
               " No interpolation across gaps. R² is descriptive; spatial samples are correlated.",
        source_coverage=threshold, fits=fits, edge_sensitivity=sensitivity), indent=2))
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
