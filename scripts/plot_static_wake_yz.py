"""Plot two interior y-z slices per static mean volume."""
from pathlib import Path
import argparse
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--rotor-diameter", type=float, default=1200)
    args = parser.parse_args()
    volumes = []
    for path in sorted(args.input.glob("Static*/mean.nc")):
        if path.parent.name.endswith("_z0"):
            continue
        with h5py.File(path) as f:
            x = f["x"][:].astype(float)
            if np.any(np.diff(x) <= 0):
                raise ValueError(f"Expected increasing x: {path}")
            volumes.append((path, x))
    if not volumes or args.rotor_diameter <= 0:
        raise ValueError("Need static mean volumes and positive rotor diameter")
    args.output_folder.mkdir(parents=True, exist_ok=False)
    volumes.sort(key=lambda item: float(np.mean(item[1])))
    fig, axes = plt.subplots(2, len(volumes), figsize=(3 * len(volumes), 9),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    records, arrays = [], {}
    selections = [(axes[row, col], path, x, fraction)
                  for row, fraction in enumerate((1/3, 2/3))
                  for col, (path, x) in enumerate(volumes)]
    for ax, path, x, fraction in selections:
        requested = float(x[0] + fraction * (x[-1] - x[0]))
        index = int(np.argmin(np.abs(x - requested)))
        used = float(x[index])
        with h5py.File(path) as f:
            u = f["u_mean"][:, :, index].astype(float)
            uinf = float(f.attrs["u_inf"])
            deficit = np.ma.masked_invalid((uinf-u)/uinf).T
            y, z = f["y"][:] / args.rotor_diameter, f["z"][:] / args.rotor_diameter
        mesh = ax.pcolormesh(z, y, deficit, cmap="viridis", vmin=0, vmax=0.7,
                             shading="nearest", rasterized=True)
        contour = ax.contour(z, y, deficit, levels=[0.1], colors=["#ff3030"],
                             linewidths=2, corner_mask=False)
        contour.set_path_effects([pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()])
        title = f"{path.parent.name.replace('_', ' ')}\n$x/D = {used/args.rotor_diameter:.3f}$"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Lateral $z/D$")
        ax.set_aspect("equal")
        ax.set_facecolor("#eeeeee")
        records.append(dict(requested_x_over_d=requested/args.rotor_diameter,
                            used_x_over_d=used/args.rotor_diameter,
                            volume_fraction=fraction, x_index=index,
                            source=str(path),
                            u_inf=uinf, finite_voxels=int(deficit.count()),
                            contour_segments=sum(len(s)>1 for s in contour.allsegs[0])))
        key = f"{path.parent.name}_slice_{index}"
        arrays[f"{key}_deficit"] = deficit.filled(np.nan)
        arrays[f"{key}_y_over_d"] = y
        arrays[f"{key}_z_over_d"] = z
    for ax in axes[:, 0]:
        ax.set_ylabel("Vertical $y/D$")
    axes[0, 0].set_xlim(-0.42, 0.42)
    axes[0, 0].set_ylim(-0.24, 0.79)
    fig.colorbar(mesh, ax=axes, shrink=0.75, pad=0.015, extend="both",
                 label=r"Wake deficit $(U_\infty-\overline{u})/U_\infty$")
    fig.suptitle("Static wake — two interior slices per volume\n"
                 "Top: one-third of volume extent; bottom: two-thirds", fontsize=15)
    fig.legend(handles=[Line2D([], [], color="#ff3030", lw=2, label="0.1 deficit contour")],
               loc="lower center", bbox_to_anchor=(0.5, 0.025), frameon=False)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.9))
    for extension in ("png", "pdf"):
        fig.savefig(args.output_folder / f"static_wake_yz.{extension}", dpi=220)
    plt.close(fig)
    np.savez_compressed(args.output_folder / "planes.npz", **arrays)
    metadata = dict(rotor_diameter=args.rotor_diameter,
                    method="Nearest measured slices to one-third and two-thirds of each volume's x extent. No interpolation or smoothing. NaNs masked; gray denotes unmeasured coverage.",
                    planes=records)
    (args.output_folder / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
