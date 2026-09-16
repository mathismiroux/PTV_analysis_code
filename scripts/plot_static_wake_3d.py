"""Build a standalone interactive 3D view of exported static wake slices.

Requires plotly (python -m pip install plotly).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Folder containing planes.npz and manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = json.loads((args.input / "manifest.json").read_text())
    fig = go.Figure()
    with np.load(args.input / "planes.npz") as data:
        for plane in sorted(metadata["planes"], key=lambda p: p["used_x_over_d"]):
            volume = Path(plane["source"]).parent.name
            key = f"{volume}_slice_{plane['x_index']}"
            deficit = data[f"{key}_deficit"]
            y, z = data[f"{key}_y_over_d"], data[f"{key}_z_over_d"]
            zz, yy = np.meshgrid(z, y)
            downstream = plane["used_x_over_d"]
            xx = np.full_like(yy, downstream)
            # Plotly's vertical display axis carries physical y; its horizontal
            # cross-stream axis carries physical z, matching the 2D figures.
            yy = np.where(np.isfinite(deficit), yy, np.nan)
            fig.add_trace(go.Surface(
                x=xx, y=zz, z=yy, surfacecolor=deficit,
                colorscale="Viridis", cmin=0, cmax=0.7,
                showscale=len(fig.data) == 0,
                colorbar=dict(title="Wake deficit", thickness=18, len=0.65),
                name=f"{volume}: {downstream:.3f}D", legendgroup=volume,
                showlegend=True, connectgaps=False,
                lighting=dict(ambient=1, diffuse=0, specular=0, fresnel=0),
                hovertemplate=(f"{volume}<br>x/D={downstream:.3f}"
                               "<br>z/D=%{y:.3f}<br>y/D=%{z:.3f}"
                               "<br>Deficit=%{surfacecolor:.3f}<extra></extra>")))
            contour_fig, ax = plt.subplots()
            contour = ax.contour(z, y, np.ma.masked_invalid(deficit),
                                 levels=[0.1], corner_mask=False)
            cx, cy, cz = [], [], []
            for segment in contour.allsegs[0]:
                if len(segment) < 2:
                    continue
                cx.extend([downstream] * len(segment) + [None])
                cy.extend(segment[:, 0].tolist() + [None])
                cz.extend(segment[:, 1].tolist() + [None])
            plt.close(contour_fig)
            # Draw on both sides to keep the contour visible when rotating.
            for offset in (-0.001, 0.001):
                fig.add_trace(go.Scatter3d(
                    x=[v + offset if v is not None else None for v in cx],
                    y=cy, z=cz, mode="lines", line=dict(color="#ff3030", width=6),
                    legendgroup=volume, showlegend=False, name="0.1 deficit contour",
                    hovertemplate="Deficit = 0.1<extra></extra>"))
    surfaces = [i for i, trace in enumerate(fig.data) if trace.type == "surface"]
    fig.update_layout(
        title=dict(text="Static wake — two interior slices per volume"
                   "<br><sup>Red: 0.1 deficit contour · Drag to rotate · Scroll to zoom · "
                   "Right-drag to pan · Click legend to toggle a volume</sup>", x=0.03),
        template="plotly_white", margin=dict(l=0, r=20, t=100, b=35),
        scene=dict(xaxis_title="Downstream x/D", yaxis_title="Lateral z/D",
                   zaxis_title="Vertical y/D", aspectmode="data",
                   camera=dict(eye=dict(x=1.35, y=1.65, z=0.95), up=dict(x=0, y=0, z=1))),
        legend=dict(groupclick="togglegroup", x=0.01, y=0.99,
                    bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
        updatemenus=[dict(type="buttons", direction="right", x=0.35, y=1.07,
                         buttons=[dict(label="Solid slices", method="restyle",
                                       args=[{"opacity": 1}, surfaces]),
                                  dict(label="Translucent slices", method="restyle",
                                       args=[{"opacity": 0.45}, surfaces]),
                                  dict(label="Contours only", method="restyle",
                                       args=[{"opacity": 0}, surfaces])])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.output, include_plotlyjs=True, full_html=True,
                   config=dict(scrollZoom=True, displaylogo=False, responsive=True),
                   default_height="100vh")
    print(f"Saved {len(surfaces)} slices from {len(set(t.legendgroup for t in fig.data))} volumes: {args.output}")


if __name__ == "__main__":
    main()
