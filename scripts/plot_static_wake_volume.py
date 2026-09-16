"""Interactive full-volume static wake isosurfaces (plotly, scikit-image)."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes


def coverage_mask(h5, threshold):
    """Strict temporal coverage threshold, using counts stored with the mean."""
    count_name = "vector_count" if "vector_count" in h5 else "u_count"
    if count_name not in h5:
        raise ValueError(f"Missing valid sample counts in {h5.filename}")
    if "input_shape_time_z_y_x" in h5.attrs:
        n_times = int(h5.attrs["input_shape_time_z_y_x"][0])
    elif "t" in h5:
        n_times = len(h5["t"])
    else:
        raise ValueError(f"Missing total sample count in {h5.filename}")
    if n_times <= 0:
        raise ValueError("Total sample count must be positive")
    count = h5[count_name][:]
    return count / n_times > threshold, count_name, n_times


def surface_mesh(field, level):
    """Retain only triangles inside cells whose eight corners are measured."""
    valid = np.isfinite(field)
    if not valid.any() or not np.nanmin(field) < level < np.nanmax(field):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices, faces, _, _ = marching_cubes(
        np.where(valid, field, -100).astype(np.float32), level=level,
        allow_degenerate=False)
    cells = np.floor(vertices[faces].mean(axis=1)).astype(int)
    cells = np.minimum(cells, np.array(field.shape) - 2)
    keep = np.ones(len(faces), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                keep &= valid[cells[:, 0]+dz, cells[:, 1]+dy, cells[:, 2]+dx]
    faces = faces[keep]
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--rotor-diameter", type=float, default=1200)
    parser.add_argument("--case-pattern", default="Static*", help="Volume folder pattern")
    parser.add_argument("--case-label", default="Static wake")
    parser.add_argument("--min-valid-fraction", type=float, default=None,
                        help="Require strictly greater temporal coverage than this fraction")
    args = parser.parse_args()
    if args.min_valid_fraction is not None and not 0 <= args.min_valid_fraction < 1:
        parser.error("--min-valid-fraction must be in [0, 1)")
    volumes = []
    coverage_audit = []
    for path in args.input.glob(f"{args.case_pattern}/mean.nc"):
        if path.parent.name.endswith("_z0"):
            continue
        with h5py.File(path) as f:
            axes = [f[a][:].astype(float) / args.rotor_diameter for a in ("z", "y", "x")]
            deficit = 1 - f["u_mean"][:].astype(float) / float(f.attrs["u_inf"])
            if args.min_valid_fraction is not None:
                keep, count_name, n_times = coverage_mask(f, args.min_valid_fraction)
                if keep.shape != deficit.shape:
                    raise ValueError(f"Count shape does not match mean in {path}")
                before = int(np.isfinite(deficit).sum())
                deficit[~keep] = np.nan
                coverage_audit.append(dict(source=str(path), count_dataset=count_name,
                                           total_samples=n_times, finite_before=before,
                                           retained_voxels=int(np.isfinite(deficit).sum())))
        volumes.append((path, axes, deficit))
    volumes.sort(key=lambda v: np.mean(v[1][2]))
    if not volumes:
        raise ValueError("No static mean files found")
    args.output_folder.mkdir(parents=True, exist_ok=False)
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]],
                        row_heights=[0.77, 0.23], vertical_spacing=0.07)
    colors = ["#4477aa", "#ee7733", "#228833", "#aa3377", "#6655aa"]
    rows, meshes, audit = [], [], []
    for number, (path, (z, y, x), deficit) in enumerate(volumes):
        name, color = path.parent.name, colors[number % len(colors)]
        for level, shade, opacity in [(0.1, "#ef4444", 0.28),
                                       (0.3, "#22b8a0", 0.38), (0.5, "#f4cc42", 0.65)]:
            vertices, faces = surface_mesh(deficit, level)
            if not len(faces):
                continue
            coords = [np.interp(vertices[:, k], np.arange(len(a)), a)
                      for k, a in enumerate((z, y, x))]
            meshes.append(len(fig.data))
            fig.add_trace(go.Mesh3d(
                x=coords[2], y=coords[0], z=coords[1],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=shade, opacity=opacity, flatshading=False,
                name=f"{name} · deficit {level:g}", legendgroup=name,
                visible=True if level == 0.1 else "legendonly",
                lighting=dict(ambient=0.65, diffuse=0.65, specular=0.1),
                hovertemplate=f"{name} · deficit {level:g}<br>"
                              "x/D=%{x:.3f}<br>y/D=%{z:.3f}<br>z/D=%{y:.3f}<extra></extra>"),
                          row=1, col=1)
            audit.append(dict(source=str(path), level=level, triangles=len(faces)))
        # This is a measured-domain diagnostic, not a fitted whole-wake axis.
        zz, yy = np.meshgrid(z, y, indexing="ij")
        area = np.abs(np.gradient(z)[:, None] * np.gradient(y)[None, :])
        weights = np.where(np.isfinite(deficit), np.maximum(deficit, 0), 0) * area[:, :, None]
        total = weights.sum(axis=(0, 1))
        cy = np.divide((weights * yy[:, :, None]).sum(axis=(0, 1)), total,
                       out=np.full_like(total, np.nan), where=total > 0)
        cz = np.divide((weights * zz[:, :, None]).sum(axis=(0, 1)), total,
                       out=np.full_like(total, np.nan), where=total > 0)
        coverage = np.isfinite(deficit).mean(axis=(0, 1))
        fig.add_trace(go.Scatter3d(x=x, y=cz, z=cy, mode="lines", line=dict(color=color, width=6),
                      name=f"{name} · measured center", legendgroup=f"center_{name}",
                      hovertemplate="Measured-deficit center<br>x/D=%{x:.3f}<br>y/D=%{z:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=cy, mode="lines", line=dict(color=color, width=2),
                      name=name, legendgroup=f"center_{name}", showlegend=False,
                      customdata=coverage,
                      hovertemplate="x/D=%{x:.3f}<br>Measured center y/D=%{y:.3f}"
                                    "<br>Retained spatial grid fraction=%{customdata:.0%}<extra></extra>"), row=2, col=1)
        for xi, yi, zi, cov in zip(x, cy, cz, coverage):
            rows.append(dict(volume=name, x_over_d=xi, measured_center_y_over_d=yi,
                             measured_center_z_over_d=zi, valid_grid_fraction=cov))
    oblique = dict(eye=dict(x=1.25, y=1.8, z=0.8), up=dict(x=0, y=0, z=1))
    side = dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1),
                projection=dict(type="orthographic"))
    fig.update_layout(
        title=dict(text=args.case_label + " — full measured volumes"
                   + (f" · temporal coverage &gt; {args.min_valid_fraction:.0%}"
                      if args.min_valid_fraction is not None else "") +
                   "<br><sup>Red: deficit 0.1 · Other levels in legend · Open edges mark measurement limits</sup>", x=0.02),
        template="plotly_white", margin=dict(l=85, r=30, t=120, b=100),
        scene=dict(xaxis_title="Downstream x/D", yaxis_title="Lateral z/D",
                   zaxis_title="Vertical y/D", aspectmode="data", camera=oblique),
        legend=dict(font=dict(size=10), groupclick="toggleitem"),
        updatemenus=[dict(type="buttons", direction="right", x=0, y=1.09,
                         buttons=[dict(label="3D view", method="relayout", args=[{"scene.camera": oblique}]),
                                  dict(label="Side view (vertical motion)", method="relayout", args=[{"scene.camera": side}])]),
                     dict(type="buttons", direction="right", x=0.52, y=1.09,
                         buttons=[dict(label="0.1 surface", method="restyle",
                                       args=[{"visible": [True if a["level"] == 0.1 else "legendonly" for a in audit]}, meshes]),
                                  dict(label="All levels", method="restyle", args=[{"visible": True}, meshes])])],
        annotations=[dict(text="Measured-deficit center only: changing coverage and wake truncation can shift this trace."
                         "<br>Gaps and overlaps are kept; no extrapolation, smoothing, or joining between volumes.",
                         x=0, y=-0.13, xref="paper", yref="paper", showarrow=False, xanchor="left")])
    fig.update_xaxes(title_text="Downstream x/D", row=2, col=1)
    fig.update_yaxes(title_text="Measured center y/D", row=2, col=1)
    fig.write_html(args.output_folder / "static_wake_volume.html", include_plotlyjs=True,
                   config=dict(scrollZoom=True, displaylogo=False, responsive=True), default_height="100vh")
    with (args.output_folder / "measured_wake_center.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_folder / "manifest.json").write_text(json.dumps(dict(
        rotor_diameter=args.rotor_diameter, surfaces=audit,
        min_valid_fraction=args.min_valid_fraction, coverage_comparison="strictly greater than",
        coverage_audit=coverage_audit,
        center_method="Area-weighted centroid of positive deficit over finite measured voxels; not whole-wake center."), indent=2))
    print(f"Saved {len(volumes)} full volumes, {len(meshes)} surfaces, {len(rows)} center samples")


if __name__ == "__main__":
    main()
