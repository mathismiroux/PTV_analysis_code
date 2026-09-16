"""Interactive 3D view of an exported Q field (one frame, no gradients recomputed)."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import webbrowser

import h5py
import numpy as np


DEFAULT_FOLDER = Path(r"D:\binning64voxel75overlap_z0")


def surface_mesh(axes, q, level):
    """Extract triangles only in cells whose eight corners are valid."""
    from skimage.measure import marching_cubes

    finite = np.isfinite(q)
    valid_cells = np.ones(tuple(n - 1 for n in q.shape), dtype=bool)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                valid_cells &= finite[dz:dz + q.shape[0] - 1,
                                      dy:dy + q.shape[1] - 1,
                                      dx:dx + q.shape[2] - 1]
    filled = np.where(finite, q, level - max(abs(level), 1.0))
    if not filled.min() < level < filled.max():
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    vertices, faces, _, _ = marching_cubes(filled, level=level, allow_degenerate=False)
    cells = np.floor(vertices[faces].mean(axis=1)).astype(int)
    cells = np.clip(cells, 0, np.array(valid_cells.shape) - 1)
    faces = faces[valid_cells[tuple(cells.T)]]
    xyz = np.column_stack([np.interp(vertices[:, dim], np.arange(len(axis)), axis)
                           for dim, axis in zip((2, 1, 0), axes)])
    return xyz, faces


def read_frame(path, variable, frame, coordinate_unit, hole_mask):
    with h5py.File(path, "r") as f:
        raw = f[variable]
        if raw.ndim not in (3, 4):
            raise ValueError(f"Expected (t,z,y,x) or (z,y,x), got {raw.shape}")
        n_frames = raw.shape[0] if raw.ndim == 4 else 1
        if not 0 <= frame < n_frames:
            raise ValueError(f"Frame must be between 0 and {n_frames - 1}")

        def read(dataset):
            data = np.asarray(dataset[frame] if dataset.ndim == 4 else dataset[:], dtype=float)
            for attr in ("_FillValue", "missing_value"):
                for value in np.asarray(dataset.attrs.get(attr, [])).ravel():
                    data[data == value] = np.nan
            return data

        q = read(raw)
        axes = [np.asarray(f[key][:], dtype=float) for key in ("x", "y", "z")]
        if any(a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a))
               or not (np.all(np.diff(a) > 0) or np.all(np.diff(a) < 0)) for a in axes):
            raise ValueError("3D rendering requires at least two monotonic coordinates per axis")
        if q.shape != tuple(len(a) for a in reversed(axes)):
            raise ValueError("Q dimensions do not match (z,y,x) coordinates")
        if hole_mask != "none":
            velocities = [read(f[key]) for key in ("u", "v", "w")]
            invalid = ~np.logical_and.reduce([np.isfinite(v) for v in velocities])
            if hole_mask == "velocity-zero-or-nan":
                invalid |= np.logical_and.reduce([v == 0 for v in velocities])
            q[invalid] = np.nan
        scale = 0.001 if coordinate_unit == "mm" else 1.0
        return [a * scale for a in axes], q


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    p.add_argument("--case", default="Static", help="Case prefix, e.g. Static or SurgeLF (case insensitive)")
    p.add_argument("--distance", default="1D", help="Downstream station, e.g. 1D, 1.625D, 3.5D")
    p.add_argument("--all-distances", action="store_true", help="Combine every station of the case using exported coordinates and one shared Q threshold")
    p.add_argument("--file", type=Path, help="Explicit file; overrides folder/case/distance selection")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--variable", default="qcri")
    p.add_argument("--coordinate-unit", choices=("mm", "m"), default="mm")
    p.add_argument("--rotor-diameter", type=float, default=1.2, help="D in metres")
    p.add_argument("--u-inf", type=float, default=4.0, help="Reference velocity in m/s")
    p.add_argument("--threshold", type=float, help="Positive isosurface level in Q*=Q D^2/U_inf^2")
    p.add_argument("--percentile", type=float, default=90, help="Percentile of positive valid Q used if threshold omitted")
    p.add_argument("--hole-mask", choices=("none", "velocity-nan", "velocity-zero-or-nan"), default="velocity-zero-or-nan")
    p.add_argument("--output", type=Path, help="Output HTML path")
    p.add_argument("--no-open", action="store_true", help="Save without opening the browser")
    p.add_argument("--gif", type=Path, help="Save an animated GIF instead of HTML")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--stop-frame", type=int, help="Exclusive final frame; defaults to the shortest recording")
    p.add_argument("--frame-step", type=int, default=1)
    p.add_argument("--fps", type=float, default=15, help="GIF playback frames per second")
    p.add_argument("--elevation", type=float, default=20, help="GIF camera elevation in degrees")
    p.add_argument("--azimuth", type=float, default=-70, help="GIF camera azimuth in degrees")
    return p


def select_files(args):
    if args.file is not None:
        if args.all_distances:
            raise ValueError("--file and --all-distances cannot be used together")
        return [args.file]
    pattern = re.compile(rf"^{re.escape(args.case)}_(\d+(?:\.\d+)?)D__", re.IGNORECASE)
    matches = []
    for path in args.folder.glob("*.nc"):
        match = pattern.match(path.name)
        if match and (args.all_distances or path.name.lower().startswith(f"{args.case}_{args.distance}__".lower())):
            matches.append((float(match[1]), path))
    matches.sort()
    if not matches:
        raise ValueError(f"No matching files for {args.case} in {args.folder}")
    if len({distance for distance, _ in matches}) != len(matches):
        raise ValueError("Multiple exports found for a station; select a folder with one export per station or use --file")
    return [path for _, path in matches]


def build_figure(volumes, level, diameter, title):
    import plotly.graph_objects as go

    fig = go.Figure()
    surfaces, points = [], []
    colors = ("#168aad", "#e68a2e", "#37985a", "#b34e8f", "#805acb")
    maximum = max(float(np.nanmax(q)) for _, _, q in volumes if np.isfinite(q).any())
    for index, (path, axes, q) in enumerate(volumes):
        axes = [a / diameter for a in axes]
        z, y, x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
        vertices, faces = surface_mesh(axes, q, level)
        selected = np.isfinite(q) & (q >= level)
        label = path.stem.split("__")[0]
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0].tolist(), y=vertices[:, 1].tolist(), z=vertices[:, 2].tolist(),
            i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
            color=colors[index % len(colors)], opacity=1, name=label,
            legendgroup=label, showlegend=True, visible=bool(len(faces)),
        ))
        fig.add_trace(go.Scatter3d(
            x=x[selected].tolist(), y=y[selected].tolist(), z=z[selected].tolist(),
            mode="markers", name=label, legendgroup=label, visible=not bool(len(faces)),
            marker=dict(size=3, color=q[selected].tolist(), colorscale="Viridis",
                        cmin=level, cmax=maximum, showscale=index == 0,
                        colorbar=dict(title="Q*"), opacity=0.85),
        ))
        surfaces.extend([bool(len(faces)), not bool(len(faces))])
        points.extend([False, True])
        print(f"Source: {path}\nGrid (z,y,x): {q.shape}; valid cells: {np.isfinite(q).sum()}\n"
              f"Surface triangles: {len(faces)}; points above threshold: {selected.sum()}")
    fig.update_layout(title=title, scene=dict(xaxis_title="x/D", yaxis_title="y/D",
                      zaxis_title="z/D", aspectmode="data"),
                      margin=dict(l=0, r=0, b=0, t=110),
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.12,
                          buttons=[dict(label="Isosurfaces (points if absent)", method="update", args=[dict(visible=surfaces)]),
                                   dict(label="Points above threshold", method="update", args=[dict(visible=points)])])])
    return fig


def animate_gif(paths, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if args.frame_step <= 0 or not np.isfinite(args.fps) or not 0 < args.fps <= 100:
        raise ValueError("Frame step must be positive and FPS must be in (0,100]")
    lengths = []
    times = []
    for path in paths:
        with h5py.File(path, "r") as f:
            raw = f[args.variable]
            lengths.append(raw.shape[0] if raw.ndim == 4 else 1)
            times.append(np.asarray(f["t"][:], dtype=float) if "t" in f else None)
    stop = args.stop_frame if args.stop_frame is not None else min(lengths)
    if not 0 <= args.start_frame < stop <= min(lengths):
        raise ValueError(f"Require 0 <= start-frame < stop-frame <= {min(lengths)} (stop exclusive)")
    frames = range(args.start_frame, stop, args.frame_step)

    def load_frame(frame):
        result = []
        for path in paths:
            axes, q = read_frame(path, args.variable, frame, args.coordinate_unit, args.hole_mask)
            result.append((path, [a / args.rotor_diameter for a in axes],
                           q * args.rotor_diameter**2 / args.u_inf**2))
        return result

    first = load_frame(frames[0])
    positive = np.concatenate([q[np.isfinite(q) & (q > 0)] for _, _, q in first])
    if args.threshold is None and not positive.size:
        raise ValueError("First animation frame has no positive Q; provide --threshold or change --start-frame")
    level = args.threshold if args.threshold is not None else float(np.percentile(positive, args.percentile))
    if not np.isfinite(level) or level <= 0:
        raise ValueError("Threshold must be finite and positive")
    bounds = [(min(a[d].min() for _, a, _ in first), max(a[d].max() for _, a, _ in first)) for d in range(3)]
    colors = ("#168aad", "#e68a2e", "#37985a", "#b34e8f", "#805acb")
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0.08, top=0.82)
    legend = [Patch(color=colors[i % len(colors)], label=p.stem.split("__")[0]) for i, p in enumerate(paths)]
    fig.legend(handles=legend, loc="lower center", ncol=min(5, len(paths)))
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=args.fps)
    print(f"Rendering {len(frames)} frames at {args.fps:g} FPS; fixed Q*={level:.6g}", flush=True)
    try:
        with writer.saving(fig, str(args.gif), dpi=100):
            for count, frame in enumerate(frames):
                volumes = first if count == 0 else load_frame(frame)
                ax.clear()
                ax.set(xlim=bounds[0], ylim=bounds[2], zlim=bounds[1],
                       xlabel="x/D", ylabel="z/D", zlabel="y/D")
                ax.set_box_aspect([bounds[d][1] - bounds[d][0] for d in (0, 2, 1)], zoom=1.9)
                ax.set_yticks([float(np.mean(bounds[2]))])
                ax.view_init(elev=args.elevation, azim=args.azimuth)
                for i, (_, axes, q) in enumerate(volumes):
                    vertices, faces = surface_mesh(axes, q, level)
                    color = colors[i % len(colors)]
                    if len(faces):
                        # The zoomed wide slab extends beyond mplot3d's square
                        # axes patch. Clipping there hides the end stations.
                        ax.add_collection3d(Poly3DCollection(vertices[faces][:, :, [0, 2, 1]],
                                                           facecolor=color, edgecolor="none",
                                                           clip_on=False))
                    else:
                        iz, iy, ix = np.where(np.isfinite(q) & (q >= level))
                        ax.scatter(axes[0][ix], axes[2][iz], axes[1][iy], color=color, s=3, clip_on=False)
                elapsed = []
                for p, t in zip(paths, times):
                    if t is not None and len(t) > frame:
                        elapsed.append(f"{p.stem.split('__')[0]}: {t[frame] - t[frames[0]]:.4f}s")
                fig.suptitle(f"Q*={level:.4g} | frame {frame} | playback {args.fps:g} FPS\n"
                             + ("Separate recordings played together (not synchronized)\n" if len(paths) > 1 else "")
                             + "Elapsed from animation start: " + ("; ".join(elapsed) or "timestamps unavailable"), fontsize=10)
                writer.grab_frame()
                if count % 10 == 0 or count == len(frames) - 1:
                    print(f"Rendered {count + 1}/{len(frames)} (source frame {frame})", flush=True)
    finally:
        plt.close(fig)
    print(f"Saved: {args.gif.resolve()}", flush=True)
    if not args.no_open:
        webbrowser.open(args.gif.resolve().as_uri())


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not all(np.isfinite(v) and v > 0 for v in (args.rotor_diameter, args.u_inf)):
            raise ValueError("Rotor diameter and reference velocity must be finite and positive")
        if not 0 < args.percentile < 100:
            raise ValueError("Percentile must be strictly between 0 and 100")
        paths = select_files(args)
        if args.gif is not None:
            animate_gif(paths, args)
            return
        volumes = []
        for path in paths:
            axes, q = read_frame(path, args.variable, args.frame, args.coordinate_unit, args.hole_mask)
            q *= args.rotor_diameter**2 / args.u_inf**2
            volumes.append((path, axes, q))
        positive = np.concatenate([q[np.isfinite(q) & (q > 0)] for _, _, q in volumes])
        if not positive.size:
            raise ValueError("Selected frame has no valid positive Q")
        level = args.threshold if args.threshold is not None else float(np.percentile(positive, args.percentile))
        if not np.isfinite(level) or not 0 < level < positive.max():
            raise ValueError(f"Threshold must be positive and below maximum Q*={positive.max():.6g}")
        name = f"{args.case}_all_distances" if args.all_distances else paths[0].stem
        title = f"{name} | frame {args.frame} | Q*={level:.4g}"
        fig = build_figure(volumes, level, args.rotor_diameter, title)
        output = args.output or Path("outputs") / "q_criterion" / f"{name}_frame{args.frame}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output), include_plotlyjs=True)
        print(f"Q*=Q D^2/U_inf^2; shared threshold: {level:.6g}\nSaved: {output.resolve()}")
        if any(q.shape[0] <= 3 for _, _, q in volumes):
            print("Only three or fewer z planes: this is a thin slab, not a full wake volume.")
        if len(volumes) > 1:
            print("Stations use exported coordinates without stitching; equal frame indices do not establish synchronized acquisitions.")
        if not args.no_open:
            webbrowser.open(output.resolve().as_uri())
    except (OSError, KeyError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
