from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import ceil
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
    nearest_index,
    parse_case_distance,
    require_new_output_folder,
)


@dataclass(frozen=True)
class PlaneWakeProfile:
    case_key: str
    distance_label: str
    distance_value: float | None
    source_label: str
    path: Path
    plane_axis: str
    plane_requested: float
    plane_used: float
    x_used: float
    profile_coordinate: np.ndarray
    profile_coordinate_over_d: np.ndarray
    wake_deficit: np.ndarray
    valid: np.ndarray


def _infer_u_inf(h5: h5py.File, explicit_u_inf: float | None) -> float:
    if explicit_u_inf is not None:
        return explicit_u_inf
    if "u_inf" not in h5.attrs:
        raise ValueError("Missing u_inf metadata; pass --u-inf explicitly")
    return float(h5.attrs["u_inf"])


def _center_x_index(x: np.ndarray) -> int:
    return nearest_index(x, 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x))))


def _profile_from_plane(
    h5: h5py.File,
    plane_axis: str,
    plane_value: float,
    x_value: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = h5["x"][:].astype(np.float64)
    y = h5["y"][:].astype(np.float64)
    z = h5["z"][:].astype(np.float64)
    u = h5["u_mean"][:].astype(np.float64)

    x_index = _center_x_index(x) if x_value is None else nearest_index(x, x_value)
    if plane_axis == "z":
        plane_index = nearest_index(z, plane_value)
        return y, u[plane_index, :, x_index], float(z[plane_index]), float(x[x_index])
    if plane_axis == "y":
        plane_index = nearest_index(y, plane_value)
        return z, u[:, plane_index, x_index], float(y[plane_index]), float(x[x_index])
    raise ValueError("plane_axis must be 'z' or 'y'")


def load_plane_wake_profile(
    path: Path,
    plane_axis: str,
    plane_value: float,
    axis_coordinate: float,
    rotor_diameter: float,
    invalid_samples: str,
    u_inf: float | None,
    x_value: float | None = None,
) -> PlaneWakeProfile:
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        free_stream = _infer_u_inf(h5, u_inf)
        coordinate, u_profile, plane_used, x_used = _profile_from_plane(
            h5,
            plane_axis=plane_axis,
            plane_value=plane_value,
            x_value=x_value,
        )

    valid = valid_component_samples(u_profile, invalid_samples) & np.isfinite(u_profile)
    wake_deficit = (free_stream - u_profile) / free_stream
    wake_deficit[~valid] = np.nan
    coordinate_over_d = (coordinate - axis_coordinate) / rotor_diameter
    return PlaneWakeProfile(
        case_key=case_key,
        distance_label=distance_label,
        distance_value=distance_value,
        source_label=label,
        path=path,
        plane_axis=plane_axis,
        plane_requested=plane_value,
        plane_used=plane_used,
        x_used=x_used,
        profile_coordinate=coordinate,
        profile_coordinate_over_d=coordinate_over_d,
        wake_deficit=wake_deficit,
        valid=valid,
    )


def group_profiles_by_distance(
    profiles: list[PlaneWakeProfile],
) -> dict[str, list[PlaneWakeProfile]]:
    groups: dict[str, list[PlaneWakeProfile]] = {}
    for profile in profiles:
        groups.setdefault(profile.distance_label, []).append(profile)
    return {
        key: sorted(items, key=lambda item: (item.case_key, item.source_label))
        for key, items in sorted(
            groups.items(),
            key=lambda item: (
                float("inf")
                if item[1][0].distance_value is None
                else item[1][0].distance_value,
                item[0],
            ),
        )
    }


def _case_style(case_key: str, index: int) -> dict[str, object]:
    colors = [
        "black",
        "#2f6bff",
        "#ff2f6b",
        "#00a676",
        "#f28e2b",
        "#7f3c8d",
    ]
    linestyles = ["-", "-", "--", "-.", ":"]
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[(index // len(colors)) % len(linestyles)],
        "linewidth": 1.6,
    }


def plot_profile_grid(
    profiles: list[PlaneWakeProfile],
    output_folder: Path,
    plane_axis: str,
    plane_value: float,
    axis_coordinate: float,
    rotor_diameter: float,
    invalid_samples: str,
) -> Path:
    distance_groups = group_profiles_by_distance(profiles)
    case_keys = sorted({profile.case_key for profile in profiles})
    style_by_case = {
        case_key: _case_style(case_key, index) for index, case_key in enumerate(case_keys)
    }
    finite_arrays = [
        profile.wake_deficit[np.isfinite(profile.wake_deficit)]
        for profile in profiles
        if np.isfinite(profile.wake_deficit).any()
    ]
    if not finite_arrays:
        raise ValueError("No finite wake-deficit values found")
    finite_values = np.concatenate(finite_arrays)
    xmin = float(np.nanpercentile(finite_values, 1.0))
    xmax = float(np.nanpercentile(finite_values, 99.0))
    margin = 0.08 * max(xmax - xmin, 1e-12)
    xmin -= margin
    xmax += margin

    panel_count = len(distance_groups)
    columns = min(4, panel_count)
    rows = ceil(panel_count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.0 * columns, 3.2 * rows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    data_rows = []
    manifest_rows = []

    for ax, (distance_label, items) in zip(axes_array, distance_groups.items()):
        for profile in items:
            ax.plot(
                profile.wake_deficit,
                profile.profile_coordinate_over_d,
                label=profile.case_key,
                **style_by_case[profile.case_key],
            )
            manifest_rows.append(
                {
                    "case_key": profile.case_key,
                    "source_label": profile.source_label,
                    "distance_label": profile.distance_label,
                    "plane_axis": profile.plane_axis,
                    "plane_requested": profile.plane_requested,
                    "plane_used": profile.plane_used,
                    "x_used": profile.x_used,
                    "axis_coordinate": axis_coordinate,
                    "rotor_diameter": rotor_diameter,
                    "invalid_samples": invalid_samples,
                    "valid_points": int(np.isfinite(profile.wake_deficit).sum()),
                    "path": str(profile.path),
                }
            )
            for coordinate, coordinate_over_d, deficit, valid in zip(
                profile.profile_coordinate,
                profile.profile_coordinate_over_d,
                profile.wake_deficit,
                profile.valid,
            ):
                data_rows.append(
                    {
                        "case_key": profile.case_key,
                        "source_label": profile.source_label,
                        "distance_label": profile.distance_label,
                        "coordinate": float(coordinate),
                        "coordinate_over_d": float(coordinate_over_d),
                        "wake_deficit": deficit,
                        "valid": bool(valid),
                        "plane_used": profile.plane_used,
                        "x_used": profile.x_used,
                    }
                )
        ax.axhline(0.0, color="0.45", linestyle="--", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="0.45", linestyle=":", linewidth=0.8, zorder=0)
        ax.grid(True, color="0.75", linewidth=0.55, alpha=0.75)
        ax.set_title(distance_label)
        ax.set_xlim(xmin, xmax)

    for ax in axes_array[panel_count:]:
        ax.axis("off")

    label_axis = "y" if plane_axis == "z" else "z"
    for ax in axes_array[:panel_count]:
        ax.set_xlabel(r"$(U_\infty-\overline{u})/U_\infty$")
    for ax in axes_array[::columns]:
        ax.set_ylabel(f"({label_axis} - {label_axis}_rotor) / D")

    handles, labels = axes_array[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )
    output = output_folder / f"plane_{plane_axis}{plane_value:g}_wake_deficit_profiles.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(output.with_suffix(".csv"), manifest_rows)
    write_csv(
        output_folder / f"plane_{plane_axis}{plane_value:g}_wake_deficit_profiles_data.csv",
        data_rows,
    )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot wake-deficit profiles from one selected plane of each prepared "
            "3D mean wake product."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mean_products_folder", type=Path)
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="new folder where plot PNGs and CSVs are written",
    )
    parser.add_argument(
        "--plane-axis",
        choices=("z", "y"),
        default="z",
        help=(
            "constant coordinate used to select the plane; z gives a vertical "
            "y profile, y gives a lateral z profile"
        ),
    )
    parser.add_argument(
        "--plane-value",
        type=float,
        default=0.0,
        help="requested plane coordinate; nearest available plane is used",
    )
    parser.add_argument(
        "--x-value",
        type=float,
        default=None,
        help="streamwise coordinate for the profile; default uses each volume centre",
    )
    parser.add_argument(
        "--axis-coordinate",
        type=float,
        default=None,
        help=(
            "legacy rotor-axis coordinate along the profile direction; prefer "
            "--rotor-y for z planes and --rotor-z for y planes"
        ),
    )
    parser.add_argument(
        "--rotor-y",
        type=float,
        default=None,
        help="rotor-axis y coordinate, i.e. vertical coordinate, used for z-plane profiles",
    )
    parser.add_argument(
        "--rotor-z",
        type=float,
        default=None,
        help="rotor-axis z coordinate, i.e. lateral coordinate, used for y-plane profiles",
    )
    parser.add_argument(
        "--rotor-diameter",
        type=float,
        default=1200.0,
        help="rotor diameter in the same units as the file coordinates",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help="mean velocity samples excluded before plotting",
    )
    parser.add_argument(
        "--u-inf",
        type=float,
        default=None,
        help="free-stream velocity; default reads u_inf from each mean.nc",
    )
    return parser


def resolve_axis_coordinate(args: argparse.Namespace) -> float:
    if args.plane_axis == "z":
        if args.rotor_y is not None:
            return float(args.rotor_y)
        if args.axis_coordinate is not None:
            return float(args.axis_coordinate)
        raise SystemExit("For --plane-axis z, pass --rotor-y.")
    if args.rotor_z is not None:
        return float(args.rotor_z)
    if args.axis_coordinate is not None:
        return float(args.axis_coordinate)
    raise SystemExit("For --plane-axis y, pass --rotor-z.")


def main() -> None:
    args = build_parser().parse_args()
    output_folder = args.output_folder.resolve()
    require_new_output_folder(output_folder)
    output_folder.mkdir(parents=True)

    mean_files = discover_mean_files(args.mean_products_folder)
    if not mean_files:
        raise SystemExit(f"No mean.nc files found in {args.mean_products_folder}")
    axis_coordinate = resolve_axis_coordinate(args)

    profiles = [
        load_plane_wake_profile(
            path,
            plane_axis=args.plane_axis,
            plane_value=args.plane_value,
            axis_coordinate=axis_coordinate,
            rotor_diameter=args.rotor_diameter,
            invalid_samples=args.invalid_samples,
            u_inf=args.u_inf,
            x_value=args.x_value,
        )
        for path in mean_files
    ]
    output = plot_profile_grid(
        profiles,
        output_folder=output_folder,
        plane_axis=args.plane_axis,
        plane_value=args.plane_value,
        axis_coordinate=axis_coordinate,
        rotor_diameter=args.rotor_diameter,
        invalid_samples=args.invalid_samples,
    )
    print(f"Found {len(mean_files)} mean files.", flush=True)
    print(f"Saved plot: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
