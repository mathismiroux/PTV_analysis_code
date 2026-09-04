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
    source_label: str
    path: Path
    x_used: float
    wake_deficit: float
    valid_points: int
    integration_radius: float | None


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
    plane_valid = valid[:, :, x_index] & np.isfinite(wake_deficit)
    if integration_radius is not None:
        yy, zz = np.meshgrid(y, z)
        radius = np.sqrt((yy - axis_y) ** 2 + (zz - axis_z) ** 2)
        plane_valid &= radius <= integration_radius

    values = wake_deficit[plane_valid]
    integrated_value = float(np.mean(values)) if values.size else np.nan
    return IntegratedWakeDeficit(
        case_key=case_key,
        distance_label=distance_label,
        distance_value=distance_value,
        source_label=label,
        path=path,
        x_used=float(x[x_index]),
        wake_deficit=integrated_value,
        valid_points=int(values.size),
        integration_radius=integration_radius,
    )


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
    invalid_samples: str,
    require_all_components: bool,
) -> Path:
    groups = group_by_case(values)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    rows = []
    for index, (case_key, items) in enumerate(groups.items()):
        x_values = [
            item.distance_value for item in items if item.distance_value is not None
        ]
        y_values = [
            item.wake_deficit for item in items if item.distance_value is not None
        ]
        ax.plot(x_values, y_values, label=case_key, **_case_style(index))
        for item in items:
            rows.append(
                {
                    "case_key": item.case_key,
                    "source_label": item.source_label,
                    "distance_label": item.distance_label,
                    "distance_d": item.distance_value,
                    "x_used": item.x_used,
                    "wake_deficit": item.wake_deficit,
                    "valid_points": item.valid_points,
                    "integration_radius": item.integration_radius,
                    "min_valid_fraction": min_valid_fraction,
                    "invalid_samples": invalid_samples,
                    "require_all_components": require_all_components,
                    "path": str(item.path),
                }
            )

    ax.axhline(0.0, color="0.45", linestyle=":", linewidth=0.8)
    ax.grid(True, color="0.78", linewidth=0.6)
    ax.set_xlabel("distance [D]")
    ax.set_ylabel(r"area-averaged $(U_\infty-\overline{u})/U_\infty$")
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
            )
            for path in mean_files
        ]
        output = plot_integrated_wake_deficit(
            values,
            output_folder=output_folder,
            min_valid_fraction=args.min_valid_fraction,
            invalid_samples=args.invalid_samples,
            require_all_components=args.require_all_components,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Found {len(mean_files)} mean files.", flush=True)
    print(f"Saved plot: {output.resolve()}", flush=True)
    print(f"Saved data: {output.with_suffix('.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
