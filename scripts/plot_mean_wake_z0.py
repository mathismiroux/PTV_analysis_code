from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MeanPlane:
    case_key: str
    distance_label: str
    distance_value: float | None
    source_label: str
    path: Path
    x: np.ndarray
    y: np.ndarray
    z_value: float
    values: np.ndarray

    @property
    def x_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.x)), float(np.nanmax(self.x))

    @property
    def y_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.y)), float(np.nanmax(self.y))


def require_new_output_folder(output_folder: Path) -> None:
    if output_folder.exists():
        raise SystemExit(
            f"Refusing to use existing output folder: {output_folder}. "
            "Choose a new --output-folder for this plot run."
        )


def discover_mean_files(mean_products_folder: Path) -> list[Path]:
    manifest = mean_products_folder / "manifest.csv"
    if manifest.exists():
        paths = []
        with manifest.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "processed" and row.get("output_file"):
                    paths.append(Path(row["output_file"]))
        return [path for path in paths if path.exists()]
    return sorted(mean_products_folder.rglob("mean.nc"))


def parse_case_distance(label: str) -> tuple[str, str, float | None]:
    match = re.search(r"(?P<case>.+?)_(?P<distance>\d+(?:\.\d+)?)D(?:__|_|$)", label)
    if match:
        distance_value = float(match.group("distance"))
        return match.group("case"), f"{distance_value:g}D", distance_value
    fallback = label.split("__", 1)[0]
    return fallback, "unknown", None


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values - target)))


def load_mean_plane(path: Path, quantity: str, z_value: float) -> MeanPlane:
    with h5py.File(path, "r") as h5:
        if quantity not in h5:
            raise ValueError(
                f"{quantity!r} not found in {path}. Available datasets: "
                f"{', '.join(h5.keys())}"
            )
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        z = h5["z"][:]
        z_index = nearest_index(z, z_value)
        return MeanPlane(
            case_key=case_key,
            distance_label=distance_label,
            distance_value=distance_value,
            source_label=label,
            path=path,
            x=h5["x"][:].astype(np.float64),
            y=h5["y"][:].astype(np.float64),
            z_value=float(z[z_index]),
            values=h5[quantity][z_index, :, :].astype(np.float64),
        )


def group_planes(planes: list[MeanPlane]) -> dict[str, list[MeanPlane]]:
    groups: dict[str, list[MeanPlane]] = {}
    for plane in planes:
        groups.setdefault(plane.case_key, []).append(plane)
    for case_key, items in groups.items():
        groups[case_key] = sorted(
            items,
            key=lambda item: (
                float("inf") if item.distance_value is None else item.distance_value,
                item.x_range[0],
                item.source_label,
            ),
        )
    return groups


def _centres_covered_by_extents(
    x: np.ndarray,
    y: np.ndarray,
    extents: list[tuple[float, float, float, float]],
) -> np.ndarray:
    if not extents:
        return np.zeros((y.size, x.size), dtype=bool)
    xx, yy = np.meshgrid(x, y)
    covered = np.zeros(xx.shape, dtype=bool)
    for xmin, xmax, ymin, ymax in extents:
        covered |= (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    return covered


def _quantity_label(quantity: str) -> str:
    labels = {
        "u_over_u_inf": r"$\overline{u}/U_\infty$",
        "wake_deficit": r"$(U_\infty-\overline{u})/U_\infty$",
        "abs_U": r"$|\overline{U}|$",
        "speed_from_mean": r"$|\overline{U}|$",
        "u_mean": r"$\overline{u}$",
        "v_mean": r"$\overline{v}$",
        "w_mean": r"$\overline{w}$",
    }
    return labels.get(quantity, quantity)


def color_limits_for_planes(
    planes: list[MeanPlane],
    quantity: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    if (vmin is None) != (vmax is None):
        raise ValueError("--vmin and --vmax must be provided together")
    if vmin is not None and vmax is not None:
        if not vmin < vmax:
            raise ValueError("--vmin must be smaller than --vmax")
        return float(vmin), float(vmax)

    finite_arrays = [
        plane.values[np.isfinite(plane.values)].ravel()
        for plane in planes
        if np.isfinite(plane.values).any()
    ]
    if not finite_arrays:
        raise ValueError("No finite values found for shared color scale")
    values = np.concatenate(finite_arrays)
    if quantity in {"u_mean", "v_mean", "w_mean"}:
        limit = float(np.nanmax(np.abs(values)))
        return -limit, limit
    return (
        float(np.nanpercentile(values, 1.0)),
        float(np.nanpercentile(values, 99.0)),
    )


def plot_group(
    case_key: str,
    planes: list[MeanPlane],
    output_folder: Path,
    quantity: str,
    z_requested: float,
    cmap_name: str,
    color_limits: tuple[float, float],
) -> Path:
    vmin, vmax = color_limits

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    ax.set_facecolor("#f0f0f0")
    overlap_cmap = plt.get_cmap("Greys").copy()
    overlap_cmap.set_bad((1, 1, 1, 0))

    covered_extents: list[tuple[float, float, float, float]] = []
    last_image = None
    report_rows = []
    for plane in planes:
        overlap = _centres_covered_by_extents(plane.x, plane.y, covered_extents)
        visible_values = plane.values.copy()
        visible_values[overlap] = np.nan
        extent = (*plane.x_range, *plane.y_range)
        last_image = ax.imshow(
            np.ma.masked_invalid(visible_values),
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
        )
        if overlap.any():
            ax.imshow(
                np.ma.masked_where(~overlap, overlap.astype(float)),
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                cmap=overlap_cmap,
                alpha=0.22,
                zorder=3,
            )
        ax.plot(
            [plane.x_range[0], plane.x_range[1], plane.x_range[1], plane.x_range[0], plane.x_range[0]],
            [plane.y_range[0], plane.y_range[0], plane.y_range[1], plane.y_range[1], plane.y_range[0]],
            color="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )
        ax.text(
            0.5 * (plane.x_range[0] + plane.x_range[1]),
            plane.y_range[1] - 0.025 * (plane.y_range[1] - plane.y_range[0]),
            plane.distance_label,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            zorder=5,
        )
        covered_extents.append((*plane.x_range, *plane.y_range))
        report_rows.append(
            {
                "case_key": case_key,
                "source_label": plane.source_label,
                "distance_label": plane.distance_label,
                "z_used": plane.z_value,
                "color_vmin": vmin,
                "color_vmax": vmax,
                "overlap_cells_masked": int(overlap.sum()),
                "visible_cells": int(np.isfinite(visible_values).sum()),
                "path": str(plane.path),
            }
        )

    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=ax)
    cbar.set_label(_quantity_label(quantity))
    ax.set_title(
        f"{case_key}: z={z_requested:g} plane mean wake composite",
        pad=12,
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.01,
        0.01,
        "Downstream overlap is shaded; upstream values are kept.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    output = output_folder / f"{case_key}_{quantity}_z{z_requested:g}.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    write_group_manifest(output.with_suffix(".csv"), report_rows)
    return output


def write_group_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot z-plane mean wake composites from prepared mean.nc products, "
            "grouped by case and ordered by downstream distance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mean_products_folder", type=Path)
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="new folder where plot PNGs and plot manifests are written",
    )
    parser.add_argument("--z", type=float, default=0.0, help="requested z plane")
    parser.add_argument(
        "--quantity",
        default="u_over_u_inf",
        choices=(
            "u_over_u_inf",
            "wake_deficit",
            "abs_U",
            "speed_from_mean",
            "u_mean",
            "v_mean",
            "w_mean",
        ),
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="manual shared colorbar minimum; must be used with --vmax",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="manual shared colorbar maximum; must be used with --vmin",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_folder = args.output_folder.resolve()
    require_new_output_folder(output_folder)
    output_folder.mkdir(parents=True)

    mean_files = discover_mean_files(args.mean_products_folder)
    if not mean_files:
        raise SystemExit(f"No mean.nc files found in {args.mean_products_folder}")
    planes = [load_mean_plane(path, args.quantity, args.z) for path in mean_files]
    try:
        color_limits = color_limits_for_planes(
            planes,
            quantity=args.quantity,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    groups = group_planes(planes)
    print(
        f"Found {len(mean_files)} mean files in {len(groups)} group(s). "
        f"Shared color scale: vmin={color_limits[0]:.6g}, "
        f"vmax={color_limits[1]:.6g}.",
        flush=True,
    )
    for case_key, items in groups.items():
        output = plot_group(
            case_key,
            items,
            output_folder=output_folder,
            quantity=args.quantity,
            z_requested=args.z,
            cmap_name=args.cmap,
            color_limits=color_limits,
        )
        print(f"Saved plot: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
