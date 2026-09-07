from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.validity import INVALID_SAMPLE_MODES
from scripts.plot_mean_wake_z0 import nearest_index, parse_case_distance
from scripts.plot_radial_wake_deficit import (
    DEFAULT_EXCLUDE_PATTERNS,
    _infer_u_inf,
    _valid_volume_mask,
    discover_radial_mean_files,
)


@dataclass(frozen=True)
class IntegratedWakeDeficit:
    case_key: str
    distance_label: str
    distance_value: float | None
    plot_distance_d: float | None
    source_label: str
    path: Path
    x_used: float
    z_used: float | None
    wake_deficit: float
    valid_points: int
    integration_radius: float | None
    reduction: str


def _plane_valid_mask(
    y: np.ndarray,
    z: np.ndarray,
    axis_y: float,
    axis_z: float,
    integration_radius: float | None,
    z_plane: float | None,
) -> tuple[np.ndarray, float | None, str]:
    if z_plane is not None:
        z_index = nearest_index(z, z_plane)
        mask = np.zeros((z.size, y.size), dtype=bool)
        mask[z_index, :] = True
        if integration_radius is not None:
            mask[z_index, :] &= np.abs(y - axis_y) <= integration_radius
        return mask, float(z[z_index]), "z-plane"

    yy, zz = np.meshgrid(y, z)
    mask = np.ones((z.size, y.size), dtype=bool)
    if integration_radius is not None:
        radius = np.sqrt((yy - axis_y) ** 2 + (zz - axis_z) ** 2)
        mask &= radius <= integration_radius
    return mask, None, "area"


def _center_x_index(x: np.ndarray) -> int:
    return nearest_index(x, 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x))))


def integrated_wake_deficit(
    path: Path,
    axis_y: float,
    axis_z: float,
    integration_radius: float | None,
    invalid_samples: str,
    require_all_components: bool,
    min_valid_fraction: float,
    u_inf: float | None,
    x_value: float | None = None,
    z_plane: float | None = None,
) -> IntegratedWakeDeficit:
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        free_stream = _infer_u_inf(h5, u_inf)
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        u = h5["u_mean"][:].astype(np.float64)
        valid = _valid_volume_mask(
            h5,
            invalid_samples,
            require_all_components,
            min_valid_fraction,
        )

    x_index = _center_x_index(x) if x_value is None else nearest_index(x, x_value)
    wake_deficit = (free_stream - u[:, :, x_index]) / free_stream
    plane_mask, z_used, reduction = _plane_valid_mask(
        y,
        z,
        axis_y,
        axis_z,
        integration_radius,
        z_plane,
    )
    plane_valid = valid[:, :, x_index] & plane_mask & np.isfinite(wake_deficit)

    values = wake_deficit[plane_valid]
    integrated_value = float(np.mean(values)) if values.size else np.nan
    return IntegratedWakeDeficit(
        case_key=case_key,
        distance_label=distance_label,
        distance_value=distance_value,
        plot_distance_d=distance_value,
        source_label=label,
        path=path,
        x_used=float(x[x_index]),
        z_used=z_used,
        wake_deficit=integrated_value,
        valid_points=int(values.size),
        integration_radius=integration_radius,
        reduction=reduction,
    )


def integrated_wake_deficit_slices(
    path: Path,
    axis_y: float,
    axis_z: float,
    integration_radius: float | None,
    invalid_samples: str,
    require_all_components: bool,
    min_valid_fraction: float,
    u_inf: float | None,
    rotor_diameter: float,
    min_plane_valid_fraction: float,
    edge_slices_to_drop: int,
    z_plane: float | None,
) -> list[IntegratedWakeDeficit]:
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        free_stream = _infer_u_inf(h5, u_inf)
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        u = h5["u_mean"][:].astype(np.float64)
        valid = _valid_volume_mask(
            h5,
            invalid_samples,
            require_all_components,
            min_valid_fraction,
        )

    plane_mask, z_used, reduction = _plane_valid_mask(
        y,
        z,
        axis_y,
        axis_z,
        integration_radius,
        z_plane,
    )

    slice_values = []
    slice_counts = []
    for x_index in range(x.size):
        if x_index < edge_slices_to_drop or x_index >= x.size - edge_slices_to_drop:
            slice_values.append(np.nan)
            slice_counts.append(0)
            continue
        wake_deficit = (free_stream - u[:, :, x_index]) / free_stream
        plane_valid = valid[:, :, x_index] & plane_mask & np.isfinite(wake_deficit)
        values = wake_deficit[plane_valid]
        slice_values.append(float(np.mean(values)) if values.size else np.nan)
        slice_counts.append(int(values.size))

    max_count = max(slice_counts, default=0)
    min_count = int(np.ceil(min_plane_valid_fraction * max_count))

    results = []
    for x_used, integrated_value, valid_points in zip(x, slice_values, slice_counts):
        if valid_points < min_count:
            integrated_value = np.nan
        results.append(
            IntegratedWakeDeficit(
                case_key=case_key,
                distance_label=distance_label,
                distance_value=distance_value,
                plot_distance_d=float(x_used) / rotor_diameter,
                source_label=label,
                path=path,
                x_used=float(x_used),
                z_used=z_used,
                wake_deficit=integrated_value,
                valid_points=valid_points,
                integration_radius=integration_radius,
                reduction=reduction,
            )
        )
    return results


def group_by_case(
    values: list[IntegratedWakeDeficit],
) -> dict[str, list[IntegratedWakeDeficit]]:
    groups: dict[str, list[IntegratedWakeDeficit]] = {}
    for value in values:
        groups.setdefault(value.case_key, []).append(value)
    return {
        case_key: sorted(
            items,
            key=lambda item: (
                float("inf")
                if item.plot_distance_d is None
                else item.plot_distance_d,
                float("inf") if item.distance_value is None else item.distance_value,
                item.source_label,
            ),
        )
        for case_key, items in sorted(groups.items())
    }


def _case_style(index: int) -> dict[str, object]:
    colors = ["black", "#2f6bff", "#ff2f6b", "#00a676", "#f28e2b"]
    markers = ["o", "s", "^", "D", "v"]
    return {
        "color": colors[index % len(colors)],
        "marker": markers[index % len(markers)],
        "linewidth": 1.8,
        "markersize": 5,
    }


def _finite_xy(items: list[IntegratedWakeDeficit]) -> tuple[np.ndarray, np.ndarray]:
    xy = [
        (item.plot_distance_d, item.wake_deficit)
        for item in items
        if item.plot_distance_d is not None and np.isfinite(item.wake_deficit)
    ]
    if not xy:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    xy = sorted(xy)
    x_values = np.asarray([x for x, _ in xy], dtype=np.float64)
    y_values = np.asarray([y for _, y in xy], dtype=np.float64)
    return x_values, y_values


def _continuous_line_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size < 2:
        return x_values, y_values
    x_dense = np.linspace(float(x_values[0]), float(x_values[-1]), 300)
    return x_dense, np.interp(x_dense, x_values, y_values)


def missing_distances_by_case(
    values: list[IntegratedWakeDeficit],
) -> dict[str, list[float]]:
    groups = group_by_case(values)
    all_distances = sorted(
        {
            item.distance_value
            for item in values
            if item.distance_value is not None
        }
    )
    missing: dict[str, list[float]] = {}
    for case_key, items in groups.items():
        present = {
            item.distance_value
            for item in items
            if item.distance_value is not None
        }
        case_missing = [
            distance
            for distance in all_distances
            if distance not in present
        ]
        if case_missing:
            missing[case_key] = case_missing
    return missing


def available_output_paths(output_folder: Path) -> tuple[Path, Path]:
    stem = "integrated_wake_deficit"
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:03d}"
        output = output_folder / f"{stem}{suffix}.png"
        data = output_folder / f"{stem}{suffix}.csv"
        if not output.exists() and not data.exists():
            return output, data
    raise FileExistsError(
        f"Could not find an available output name for {stem} in {output_folder}"
    )


def plot_integrated_wake_deficit(
    values: list[IntegratedWakeDeficit],
    output_folder: Path,
    min_valid_fraction: float,
    min_plane_valid_fraction: float,
    edge_slices_to_drop: int,
    z_plane: float | None,
    invalid_samples: str,
    require_all_components: bool,
) -> Path:
    groups = group_by_case(values)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rows = []
    for index, (case_key, items) in enumerate(groups.items()):
        x_values, y_values = _finite_xy(items)
        line_x, line_y = _continuous_line_xy(x_values, y_values)
        style = _case_style(index)
        ax.plot(
            line_x,
            line_y,
            label=case_key,
            color=style["color"],
            linewidth=style["linewidth"],
        )
        ax.plot(
            x_values,
            y_values,
            linestyle="none",
            color=style["color"],
            marker=style["marker"],
            markersize=2.4 if x_values.size > 30 else style["markersize"],
        )
        for item in items:
            rows.append(
                {
                    "case_key": item.case_key,
                    "source_label": item.source_label,
                    "distance_label": item.distance_label,
                    "distance_d": item.distance_value,
                    "plot_distance_d": item.plot_distance_d,
                    "x_used": item.x_used,
                    "z_used": item.z_used,
                    "wake_deficit": item.wake_deficit,
                    "valid_points": item.valid_points,
                    "integration_radius": item.integration_radius,
                    "reduction": item.reduction,
                    "z_plane": z_plane,
                    "min_valid_fraction": min_valid_fraction,
                    "min_plane_valid_fraction": min_plane_valid_fraction,
                    "edge_slices_to_drop": edge_slices_to_drop,
                    "invalid_samples": invalid_samples,
                    "require_all_components": require_all_components,
                    "path": str(item.path),
                }
            )

    ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
    ax.grid(True, color="0.78", linewidth=0.6)
    ax.set_xlabel("distance [D]")
    if z_plane is None:
        ax.set_ylabel(r"area-averaged $(U_\infty-\overline{u})/U_\infty$")
    else:
        ax.set_ylabel(
            rf"$z={z_plane:g}$ plane-averaged "
            r"$(U_\infty-\overline{u})/U_\infty$"
        )
    ax.set_title("Integrated wake deficit")
    ax.legend(frameon=True)
    output, data = available_output_paths(output_folder)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    with data.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one area-averaged wake-deficit value per downstream distance."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mean_inputs",
        nargs="+",
        type=Path,
        help=(
            "one or more mean-product roots, distance folders containing "
            "mean.nc, or explicit mean.nc files"
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help=(
            "folder where the plot PNG and CSV are written; existing files are "
            "not overwritten"
        ),
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
        "--integration-radius",
        type=float,
        default=None,
        help=(
            "maximum radius around the rotor axis included in the cross-plane "
            "average; omit to use the full measured plane"
        ),
    )
    parser.add_argument(
        "--rotor-diameter",
        type=float,
        default=None,
        help="shortcut for --integration-radius equal to half this diameter",
    )
    parser.add_argument(
        "--x-value",
        type=float,
        default=None,
        help="streamwise coordinate to sample; default uses each volume centre",
    )
    parser.add_argument(
        "--z-plane",
        type=float,
        default=None,
        help=(
            "use only the nearest z plane to this coordinate before averaging; "
            "pass 0 for the centre plane instead of full cross-plane averaging"
        ),
    )
    parser.add_argument(
        "--all-x-slices",
        action="store_true",
        help=(
            "integrate every streamwise x slice in each volume and plot it at "
            "x / rotor_diameter; requires --rotor-diameter"
        ),
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help="mean velocity samples excluded before averaging",
    )
    parser.add_argument(
        "--require-all-components",
        action="store_true",
        help="require u_mean, v_mean, and w_mean to be valid at a voxel",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "minimum fraction of raw time samples required at each voxel before "
            "it contributes to the cross-plane average"
        ),
    )
    parser.add_argument(
        "--min-plane-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "with --all-x-slices, drop slice averages whose valid cross-plane "
            "point count is below this fraction of the best-covered slice in "
            "the same volume"
        ),
    )
    parser.add_argument(
        "--edge-slices-to-drop",
        type=int,
        default=2,
        help=(
            "with --all-x-slices, drop this many streamwise slices from both "
            "the start and end of each volume"
        ),
    )
    parser.add_argument(
        "--u-inf",
        type=float,
        default=None,
        help="free-stream velocity; default reads u_inf from each mean.nc",
    )
    parser.add_argument(
        "--no-sibling-distances",
        action="store_true",
        help=(
            "when an input is one distance folder or mean.nc file, use only "
            "that file instead of the same-case sibling distance folders"
        ),
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=list(DEFAULT_EXCLUDE_PATTERNS),
        help=(
            "folder/file glob pattern to exclude from discovered mean files; "
            "use more than once for multiple patterns"
        ),
    )
    parser.add_argument(
        "--include-z0",
        action="store_true",
        help="include folders such as *_z0 that are excluded by default",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise SystemExit("--min-valid-fraction must be between 0 and 1.")
    if args.integration_radius is not None and args.rotor_diameter is not None:
        raise SystemExit("Use either --integration-radius or --rotor-diameter.")
    if args.all_x_slices and args.rotor_diameter is None:
        raise SystemExit("--all-x-slices requires --rotor-diameter.")
    if not 0.0 <= args.min_plane_valid_fraction <= 1.0:
        raise SystemExit("--min-plane-valid-fraction must be between 0 and 1.")
    if args.edge_slices_to_drop < 0:
        raise SystemExit("--edge-slices-to-drop must be non-negative.")
    integration_radius = args.integration_radius
    if args.rotor_diameter is not None:
        integration_radius = 0.5 * args.rotor_diameter

    output_folder = args.output_folder.resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    mean_files = discover_radial_mean_files(
        args.mean_inputs,
        include_sibling_distances=not args.no_sibling_distances,
        exclude_patterns=()
        if args.include_z0
        else tuple(args.exclude_pattern or ()),
    )
    if not mean_files:
        inputs = ", ".join(str(path) for path in args.mean_inputs)
        raise SystemExit(f"No mean.nc files found for input(s): {inputs}")

    try:
        if args.all_x_slices:
            values = [
                value
                for path in mean_files
                for value in integrated_wake_deficit_slices(
                    path,
                    axis_y=args.axis_y,
                    axis_z=args.axis_z,
                    integration_radius=integration_radius,
                    invalid_samples=args.invalid_samples,
                    require_all_components=args.require_all_components,
                    min_valid_fraction=args.min_valid_fraction,
                    u_inf=args.u_inf,
                    rotor_diameter=args.rotor_diameter,
                    min_plane_valid_fraction=args.min_plane_valid_fraction,
                    edge_slices_to_drop=args.edge_slices_to_drop,
                    z_plane=args.z_plane,
                )
            ]
        else:
            values = [
                integrated_wake_deficit(
                    path,
                    axis_y=args.axis_y,
                    axis_z=args.axis_z,
                    integration_radius=integration_radius,
                    invalid_samples=args.invalid_samples,
                    require_all_components=args.require_all_components,
                    min_valid_fraction=args.min_valid_fraction,
                    u_inf=args.u_inf,
                    x_value=args.x_value,
                    z_plane=args.z_plane,
                )
                for path in mean_files
            ]
        output = plot_integrated_wake_deficit(
            values,
            output_folder=output_folder,
            min_valid_fraction=args.min_valid_fraction,
            min_plane_valid_fraction=args.min_plane_valid_fraction,
            edge_slices_to_drop=(
                args.edge_slices_to_drop if args.all_x_slices else 0
            ),
            z_plane=args.z_plane,
            invalid_samples=args.invalid_samples,
            require_all_components=args.require_all_components,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Found {len(mean_files)} mean files.", flush=True)
    for case_key, distances in missing_distances_by_case(values).items():
        labels = ", ".join(f"{distance:g}D" for distance in distances)
        print(f"Missing distances for {case_key}: {labels}", flush=True)
    print(f"Saved plot: {output.resolve()}", flush=True)
    print(f"Saved data: {output.with_suffix('.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
