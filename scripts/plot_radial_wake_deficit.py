from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.validity import INVALID_SAMPLE_MODES, valid_component_samples
from scripts.plot_mean_wake_z0 import (
    discover_mean_files,
    parse_case_distance,
    require_new_output_folder,
)


class RadialWakeVolume:
    def __init__(
        self,
        case_key: str,
        distance_label: str,
        distance_value: float | None,
        source_label: str,
        path: Path,
        x: np.ndarray,
        radial_centres: np.ndarray,
        wake_deficit: np.ndarray,
        counts: np.ndarray,
    ) -> None:
        self.case_key = case_key
        self.distance_label = distance_label
        self.distance_value = distance_value
        self.source_label = source_label
        self.path = path
        self.x = x
        self.radial_centres = radial_centres
        self.wake_deficit = wake_deficit
        self.counts = counts

    @property
    def x_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.x)), float(np.nanmax(self.x))


def _radial_edges(max_radius: float, radial_bin_width: float) -> np.ndarray:
    if radial_bin_width <= 0:
        raise ValueError("radial_bin_width must be positive")
    return np.arange(0.0, max_radius + radial_bin_width, radial_bin_width)


def _infer_u_inf(h5: h5py.File, explicit_u_inf: float | None) -> float:
    if explicit_u_inf is not None:
        return explicit_u_inf
    if "u_inf" not in h5.attrs:
        raise ValueError("Missing u_inf metadata; pass --u-inf explicitly")
    return float(h5.attrs["u_inf"])


def _valid_volume_mask(
    h5: h5py.File,
    invalid_samples: str,
    require_all_components: bool,
) -> np.ndarray:
    component_masks = [
        valid_component_samples(h5[f"{component}_mean"][:], invalid_samples)
        for component in ("u", "v", "w")
    ]
    finite_masks = [np.isfinite(h5[f"{component}_mean"][:]) for component in ("u", "v", "w")]
    masks = [valid & finite for valid, finite in zip(component_masks, finite_masks)]
    if require_all_components:
        return np.logical_and.reduce(masks)
    return masks[0]


def radial_average_wake_deficit(
    path: Path,
    axis_y: float,
    axis_z: float,
    radial_bin_width: float,
    invalid_samples: str,
    require_all_components: bool,
    u_inf: float | None,
) -> RadialWakeVolume:
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        u = h5["u_mean"][:].astype(np.float64)
        free_stream = _infer_u_inf(h5, u_inf)
        valid = _valid_volume_mask(h5, invalid_samples, require_all_components)
        wake_deficit = (free_stream - u) / free_stream
        wake_deficit[~valid] = np.nan

    yy, zz = np.meshgrid(y, z)
    radius = np.sqrt((yy - axis_y) ** 2 + (zz - axis_z) ** 2)
    edges = _radial_edges(float(np.nanmax(radius)), radial_bin_width)
    radial_centres = 0.5 * (edges[:-1] + edges[1:])
    radial_values = np.full((radial_centres.size, x.size), np.nan, dtype=np.float64)
    counts = np.zeros((radial_centres.size, x.size), dtype=np.uint32)

    for radial_index in range(radial_centres.size):
        annulus = (radius >= edges[radial_index]) & (radius < edges[radial_index + 1])
        if not annulus.any():
            continue
        for x_index in range(x.size):
            values = wake_deficit[:, :, x_index][annulus]
            values = values[np.isfinite(values)]
            counts[radial_index, x_index] = values.size
            if values.size:
                radial_values[radial_index, x_index] = float(np.mean(values))

    return RadialWakeVolume(
        case_key=case_key,
        distance_label=distance_label,
        distance_value=distance_value,
        source_label=label,
        path=path,
        x=x,
        radial_centres=radial_centres,
        wake_deficit=radial_values,
        counts=counts,
    )


def group_radial_volumes(
    volumes: list[RadialWakeVolume],
) -> dict[str, list[RadialWakeVolume]]:
    grouped_indices: dict[str, list[int]] = {}
    for index, volume in enumerate(volumes):
        grouped_indices.setdefault(volume.case_key, []).append(index)
    return {
        case_key: sorted(
            [volumes[index] for index in indices],
            key=lambda item: (
                float("inf") if item.distance_value is None else item.distance_value,
                item.x_range[0],
                item.source_label,
            ),
        )
        for case_key, indices in grouped_indices.items()
    }


def _x_covered(x: np.ndarray, covered_ranges: list[tuple[float, float]]) -> np.ndarray:
    covered = np.zeros(x.shape, dtype=bool)
    for xmin, xmax in covered_ranges:
        covered |= (x >= xmin) & (x <= xmax)
    return covered


def plot_radial_group(
    case_key: str,
    volumes: list[RadialWakeVolume],
    output_folder: Path,
    axis_y: float,
    axis_z: float,
    radial_bin_width: float,
    invalid_samples: str,
    require_all_components: bool,
    cmap_name: str,
    contour_step: float,
    contour_label_step: float,
) -> Path:
    finite_arrays = [
        volume.wake_deficit[np.isfinite(volume.wake_deficit)] for volume in volumes
    ]
    finite_arrays = [values for values in finite_arrays if values.size]
    if not finite_arrays:
        raise ValueError(f"No finite radial wake-deficit values for {case_key}")
    finite_values = np.concatenate(finite_arrays)
    if finite_values.size == 0:
        raise ValueError(f"No finite radial wake-deficit values for {case_key}")
    vmin = float(np.nanpercentile(finite_values, 1.0))
    vmax = float(np.nanpercentile(finite_values, 99.0))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    overlap_cmap = plt.get_cmap("Greys").copy()
    overlap_cmap.set_bad((1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(12, 6.2), constrained_layout=True)
    ax.set_facecolor("#f0f0f0")
    covered_ranges: list[tuple[float, float]] = []
    last_image = None
    manifest_rows = []
    data_rows = []
    for volume in volumes:
        overlap = _x_covered(volume.x, covered_ranges)
        visible = volume.wake_deficit.copy()
        visible[:, overlap] = np.nan
        radial_min = 0.0
        radial_max = float(volume.radial_centres[-1] + 0.5 * radial_bin_width)
        extent = (
            float(np.nanmin(volume.x)),
            float(np.nanmax(volume.x)),
            radial_min,
            radial_max,
        )
        last_image = ax.imshow(
            np.ma.masked_invalid(visible),
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
            overlap_image = np.tile(overlap, (volume.radial_centres.size, 1))
            ax.imshow(
                np.ma.masked_where(~overlap_image, overlap_image.astype(float)),
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                cmap=overlap_cmap,
                alpha=0.22,
                zorder=3,
            )
        if contour_step > 0:
            contour_min = np.floor(vmin / contour_step) * contour_step
            contour_max = np.ceil(vmax / contour_step) * contour_step
            levels = np.arange(
                contour_min,
                contour_max + 0.5 * contour_step,
                contour_step,
            )
            if levels.size:
                xx, rr = np.meshgrid(volume.x, volume.radial_centres)
                contours = ax.contour(
                    xx,
                    rr,
                    visible,
                    levels=levels,
                    colors="black",
                    linewidths=0.45,
                    alpha=0.55,
                    zorder=4,
                )
                if contour_label_step > 0:
                    label_levels = [
                        level
                        for level in contours.levels
                        if np.isclose(
                            level / contour_label_step,
                            np.round(level / contour_label_step),
                            atol=1e-9,
                        )
                    ]
                    if label_levels:
                        ax.clabel(
                            contours,
                            levels=label_levels,
                            inline=True,
                            fmt="%.1f",
                            fontsize=7,
                            colors="black",
                        )
        ax.plot(
            [
                volume.x_range[0],
                volume.x_range[1],
                volume.x_range[1],
                volume.x_range[0],
                volume.x_range[0],
            ],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=5,
        )
        ax.text(
            0.5 * (volume.x_range[0] + volume.x_range[1]),
            extent[3] - 0.04 * (extent[3] - extent[2]),
            volume.distance_label,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            zorder=6,
        )
        covered_ranges.append(volume.x_range)
        manifest_rows.append(
            {
                "case_key": case_key,
                "source_label": volume.source_label,
                "distance_label": volume.distance_label,
                "axis_y": axis_y,
                "axis_z": axis_z,
                "radial_bin_width": radial_bin_width,
                "invalid_samples": invalid_samples,
                "require_all_components": require_all_components,
                "contour_step": contour_step,
                "contour_label_step": contour_label_step,
                "overlap_x_columns_masked": int(overlap.sum()),
                "visible_cells": int(np.isfinite(visible).sum()),
                "path": str(volume.path),
            }
        )
        for radial_index, radius in enumerate(volume.radial_centres):
            for x_index, x_value in enumerate(volume.x):
                data_rows.append(
                    {
                        "case_key": case_key,
                        "source_label": volume.source_label,
                        "distance_label": volume.distance_label,
                        "x": float(x_value),
                        "radial_distance": float(radius),
                        "wake_deficit": volume.wake_deficit[radial_index, x_index],
                        "count": int(volume.counts[radial_index, x_index]),
                        "masked_as_downstream_overlap": bool(overlap[x_index]),
                    }
                )

    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=ax)
    cbar.set_label(r"$(U_\infty-\overline{u})/U_\infty$")
    ax.set_title(f"{case_key}: radial mean wake deficit", pad=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("radial distance in vertical-lateral plane [mm]")
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
    output = output_folder / f"{case_key}_radial_wake_deficit.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    write_manifest(output.with_suffix(".csv"), manifest_rows)
    write_manifest(
        output_folder / f"{case_key}_radial_wake_deficit_data.csv",
        data_rows,
    )
    return output


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot radial wake-deficit profiles from prepared 3D mean wake products."
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
    parser.add_argument(
        "--axis-y",
        "--rotor-y",
        dest="axis_y",
        type=float,
        required=True,
        help="rotor-axis y coordinate, i.e. vertical coordinate",
    )
    parser.add_argument(
        "--axis-z",
        "--rotor-z",
        dest="axis_z",
        type=float,
        required=True,
        help="rotor-axis z coordinate, i.e. lateral/right-hand-rule coordinate",
    )
    parser.add_argument(
        "--radial-bin-width",
        type=float,
        default=25.0,
        help="radial averaging bin width, in the same units as y/z",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="zero-or-nan",
        help="mean velocity samples excluded before radial averaging",
    )
    parser.add_argument(
        "--require-all-components",
        action="store_true",
        help="require u_mean, v_mean, and w_mean to be valid at a voxel",
    )
    parser.add_argument(
        "--u-inf",
        type=float,
        default=None,
        help="free-stream velocity; default reads u_inf from each mean.nc",
    )
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap")
    parser.add_argument(
        "--contour-step",
        type=float,
        default=0.05,
        help="wake-deficit contour spacing; use 0 to disable contours",
    )
    parser.add_argument(
        "--contour-label-step",
        type=float,
        default=0.1,
        help="spacing for numeric contour labels; use 0 to disable labels",
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
    volumes = [
        radial_average_wake_deficit(
            path,
            axis_y=args.axis_y,
            axis_z=args.axis_z,
            radial_bin_width=args.radial_bin_width,
            invalid_samples=args.invalid_samples,
            require_all_components=args.require_all_components,
            u_inf=args.u_inf,
        )
        for path in mean_files
    ]
    groups = group_radial_volumes(volumes)
    print(f"Found {len(mean_files)} mean files in {len(groups)} group(s).", flush=True)
    for case_key, items in groups.items():
        output = plot_radial_group(
            case_key,
            items,
            output_folder=output_folder,
            axis_y=args.axis_y,
            axis_z=args.axis_z,
            radial_bin_width=args.radial_bin_width,
            invalid_samples=args.invalid_samples,
            require_all_components=args.require_all_components,
            cmap_name=args.cmap,
            contour_step=args.contour_step,
            contour_label_step=args.contour_label_step,
        )
        print(f"Saved plot: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
