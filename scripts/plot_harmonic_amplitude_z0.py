from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.postprocess import PhaseAverageVolume  # noqa: E402


@dataclass(frozen=True)
class HarmonicPlane:
    case_key: str
    distance_label: str
    distance_value: float | None
    source_label: str
    motion_type: str
    path: Path
    plane_axis: str
    plane_value: float
    horizontal_axis: str
    vertical_axis: str
    horizontal: np.ndarray
    vertical: np.ndarray
    values: np.ndarray
    accepted_cells: int | None
    min_valid_fraction_values: np.ndarray | None


def discover_phase_average_files(root_or_file: Path) -> list[Path]:
    if root_or_file.is_file():
        return [root_or_file]

    paths = []
    manifest = root_or_file / "phase_average_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "processed" and row.get("output_file"):
                    paths.append(Path(row["output_file"]))

    paths.extend(root_or_file.rglob("phase_average.nc"))
    return sorted({path.resolve() for path in paths if path.exists()})


def parse_case_distance(label: str) -> tuple[str, str, float | None]:
    match = re.search(r"(?P<case>.+?)_(?P<distance>\d+(?:\.\d+)?)D(?:__|_|$)", label)
    if match:
        distance_value = float(match.group("distance"))
        return match.group("case"), f"{distance_value:g}D", distance_value
    fallback = label.split("__", 1)[0]
    return fallback, "unknown", None


def label_for_path(path: Path, volume: PhaseAverageVolume) -> str:
    return str(volume._file.attrs.get("label", path.parent.name))


def motion_type_for_path(path: Path, volume: PhaseAverageVolume) -> str:
    return str(volume._file.attrs.get("motion_type", "")).strip()


def phase_count_stack_for_plane(
    volume: PhaseAverageVolume,
    plane_axis: str,
    plane_index: int,
    component: str,
) -> np.ndarray:
    count_name = f"{component}_phase_count"
    if count_name not in volume._file:
        raise KeyError(f"Missing expected phase-count dataset: {count_name}")
    if plane_axis == "z":
        return volume._file[count_name][:, plane_index, :, :]
    if plane_axis == "y":
        return volume._file[count_name][:, :, plane_index, :]
    if plane_axis == "x":
        return volume._file[count_name][:, :, :, plane_index]
    raise ValueError("plane_axis must be one of 'x', 'y', or 'z'")


def apply_min_phase_valid_fraction_mask(
    volume: PhaseAverageVolume,
    values: np.ndarray,
    plane_axis: str,
    plane_index: int,
    component: str,
    min_valid_fraction: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("--min-valid-fraction must be between 0 and 1")

    phase_counts = phase_count_stack_for_plane(
        volume,
        plane_axis=plane_axis,
        plane_index=plane_index,
        component=component,
    ).astype(np.float64)
    sample_counts = volume._file["phase_sample_count"][:].astype(np.float64)
    fractions = np.divide(
        phase_counts,
        sample_counts[:, None, None],
        out=np.zeros_like(phase_counts, dtype=np.float64),
        where=sample_counts[:, None, None] > 0,
    )
    min_fraction = np.nanmin(fractions, axis=0)
    accepted = min_fraction > min_valid_fraction
    masked = np.where(accepted, values, np.nan)
    return masked, int(np.count_nonzero(accepted)), min_fraction


def load_harmonic_plane(
    path: Path,
    plane_axis: str,
    plane_value: float,
    component: str,
    harmonic_quantity: str,
    min_valid_fraction: float,
) -> HarmonicPlane:
    with PhaseAverageVolume(path) as volume:
        source_label = label_for_path(path, volume)
        motion_type = motion_type_for_path(path, volume)
        case_key, distance_label, distance_value = parse_case_distance(source_label)
        plane_index = volume.nearest_index(plane_axis, plane_value)
        plane = volume.read_harmonic_plane(
            axis=plane_axis,
            index=plane_index,
            component=component,
            quantity=harmonic_quantity,
        )
        values = np.asarray(plane["data"], dtype=np.float64)
        if harmonic_quantity == "phase":
            values = np.degrees(values)

        accepted_cells = None
        min_valid_fraction_values = None
        if min_valid_fraction > 0.0:
            values, accepted_cells, min_valid_fraction_values = (
                apply_min_phase_valid_fraction_mask(
                    volume,
                    values,
                    plane_axis=plane_axis,
                    plane_index=plane_index,
                    component=component,
                    min_valid_fraction=min_valid_fraction,
                )
            )

        return HarmonicPlane(
            case_key=case_key,
            distance_label=distance_label,
            distance_value=distance_value,
            source_label=source_label,
            motion_type=motion_type,
            path=path,
            plane_axis=plane_axis,
            plane_value=float(plane["value"]),
            horizontal_axis=str(plane["horizontal_axis"]),
            vertical_axis=str(plane["vertical_axis"]),
            horizontal=np.asarray(plane["horizontal"], dtype=np.float64),
            vertical=np.asarray(plane["vertical"], dtype=np.float64),
            values=values,
            accepted_cells=accepted_cells,
            min_valid_fraction_values=min_valid_fraction_values,
        )


def is_static_plane(plane: HarmonicPlane) -> bool:
    if plane.motion_type.casefold() == "static":
        return True
    static_pattern = re.compile(r"(^|[_\W])static([_\W]|$)", flags=re.IGNORECASE)
    return any(
        static_pattern.search(text) is not None
        for text in (plane.case_key, plane.source_label, plane.path.parent.name)
    )


def require_moving_planes(
    planes: list[HarmonicPlane],
    include_static: bool = False,
) -> list[HarmonicPlane]:
    if include_static:
        return planes
    moving = [plane for plane in planes if not is_static_plane(plane)]
    if moving:
        return moving

    rejected = ", ".join(sorted({plane.source_label for plane in planes}))
    raise SystemExit(
        "Phase-coherent harmonic maps are only available for moving cases "
        "such as SurgeLF, surge, pitch, or wave. Static case(s) rejected: "
        f"{rejected}"
    )


def filter_planes(planes: list[HarmonicPlane], case_filter: str | None) -> list[HarmonicPlane]:
    if case_filter is None:
        return planes
    needle = case_filter.casefold()
    filtered = [
        plane
        for plane in planes
        if needle in plane.case_key.casefold()
        or needle in plane.source_label.casefold()
        or needle in plane.path.parent.name.casefold()
    ]
    if not filtered:
        choices = ", ".join(sorted({plane.case_key for plane in planes}))
        raise SystemExit(
            f"No moving phase-average files matched --case {case_filter!r}. "
            f"Found moving cases: {choices}"
        )
    return filtered


def ordered_planes(planes: list[HarmonicPlane]) -> list[HarmonicPlane]:
    return sorted(
        planes,
        key=lambda plane: (
            float("inf") if plane.distance_value is None else plane.distance_value,
            float(np.nanmin(plane.horizontal)),
            plane.source_label,
            str(plane.path),
        ),
    )


def choose_one_plane(planes: list[HarmonicPlane]) -> HarmonicPlane:
    if len(planes) == 1:
        return planes[0]
    choices = "\n".join(
        f"  - {plane.source_label}: {plane.path}" for plane in ordered_planes(planes)
    )
    raise SystemExit(
        "More than one matching phase_average.nc file was found. "
        "Pass --composite to plot all matches in one figure, use --case to "
        "narrow it down, or pass one phase_average.nc file directly.\n"
        f"{choices}"
    )


def default_cmap(harmonic_quantity: str) -> str:
    if harmonic_quantity == "r2":
        return "viridis"
    if harmonic_quantity == "phase":
        return "twilight"
    if harmonic_quantity == "amplitude":
        return "magma"
    return "coolwarm"


def colorbar_label(component: str, harmonic_quantity: str) -> str:
    if harmonic_quantity == "r2":
        return f"{component} harmonic R2"
    if harmonic_quantity == "phase":
        return f"{component} harmonic phase [deg]"
    return f"{component} harmonic {harmonic_quantity}"


def color_limits_for_values(
    values: list[np.ndarray],
    harmonic_quantity: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    if (vmin is None) != (vmax is None):
        raise SystemExit("--vmin and --vmax must be provided together")
    if vmin is not None and vmax is not None:
        if not vmin < vmax:
            raise SystemExit("--vmin must be smaller than --vmax")
        return float(vmin), float(vmax)
    if harmonic_quantity == "r2":
        return 0.0, 1.0
    if harmonic_quantity == "phase":
        return -180.0, 180.0
    if harmonic_quantity in {"a", "b", "offset"}:
        finite_arrays = [item[np.isfinite(item)] for item in values if np.isfinite(item).any()]
        if not finite_arrays:
            return None, None
        finite = np.concatenate(finite_arrays)
        if finite.size:
            limit = float(np.nanmax(np.abs(finite)))
            if limit > 0.0:
                return -limit, limit
    return None, None


def color_limits(
    values: np.ndarray,
    harmonic_quantity: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    return color_limits_for_values([values], harmonic_quantity, vmin, vmax)


def output_path_for(
    output: Path | None,
    output_folder: Path | None,
    plane: HarmonicPlane,
    component: str,
    harmonic_quantity: str,
    requested_plane_value: float,
) -> Path:
    if output is not None and output_folder is not None:
        raise SystemExit("Use either --output or --output-folder, not both")
    if output is not None:
        return output
    folder = output_folder if output_folder is not None else Path("outputs") / "figures"
    return (
        folder
        / f"{plane.case_key}_{component}_harmonic_{harmonic_quantity}_"
        f"{plane.plane_axis}{requested_plane_value:g}.png"
    )


def composite_output_path_for(
    output: Path | None,
    output_folder: Path | None,
    case_key: str,
    plane_axis: str,
    component: str,
    harmonic_quantity: str,
    requested_plane_value: float,
) -> Path:
    if output is not None and output_folder is not None:
        raise SystemExit("Use either --output or --output-folder, not both")
    if output is not None:
        return output
    folder = output_folder if output_folder is not None else Path("outputs") / "figures"
    return (
        folder
        / f"{case_key}_{component}_harmonic_{harmonic_quantity}_"
        f"{plane_axis}{requested_plane_value:g}_composite.png"
    )


def write_manifest(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def min_visible_voxel_fraction(plane: HarmonicPlane) -> float | str:
    if plane.min_valid_fraction_values is None:
        return ""
    visible = np.isfinite(plane.values)
    if not visible.any():
        return ""
    return float(np.nanmin(plane.min_valid_fraction_values[visible]))


def _centres_covered_by_extents(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    extents: list[tuple[float, float, float, float]],
) -> np.ndarray:
    if not extents:
        return np.zeros((vertical.size, horizontal.size), dtype=bool)
    hh, vv = np.meshgrid(horizontal, vertical)
    covered = np.zeros(hh.shape, dtype=bool)
    for hmin, hmax, vmin, vmax in extents:
        covered |= (hh >= hmin) & (hh <= hmax) & (vv >= vmin) & (vv <= vmax)
    return covered


def plot_harmonic_plane(
    plane: HarmonicPlane,
    component: str,
    harmonic_quantity: str,
    requested_plane_value: float,
    output: Path,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
    dpi: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_invalid(plane.values),
        extent=(
            float(np.nanmin(plane.horizontal)),
            float(np.nanmax(plane.horizontal)),
            float(np.nanmin(plane.vertical)),
            float(np.nanmax(plane.vertical)),
        ),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label(component, harmonic_quantity))
    ax.set_xlabel(f"{plane.horizontal_axis} [mm]")
    ax.set_ylabel(f"{plane.vertical_axis} [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        title
        or (
            f"{plane.source_label}: {component} harmonic {harmonic_quantity}, "
            f"{plane.plane_axis}={plane.plane_value:.6g} "
            f"(nearest to {requested_plane_value:g})"
        )
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def plot_harmonic_composite(
    planes: list[HarmonicPlane],
    component: str,
    harmonic_quantity: str,
    requested_plane_value: float,
    output: Path,
    cmap_name: str,
    vmin: float | None,
    vmax: float | None,
    title: str | None,
    dpi: int,
) -> list[dict[str, object]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    ordered = ordered_planes(planes)
    first = ordered[0]
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    ax.set_facecolor("#f0f0f0")

    covered_extents: list[tuple[float, float, float, float]] = []
    last_image = None
    rows = []
    for plane in ordered:
        if (
            plane.horizontal_axis != first.horizontal_axis
            or plane.vertical_axis != first.vertical_axis
        ):
            raise SystemExit("All composite planes must use the same plotted axes")

        h_range = (float(np.nanmin(plane.horizontal)), float(np.nanmax(plane.horizontal)))
        v_range = (float(np.nanmin(plane.vertical)), float(np.nanmax(plane.vertical)))
        overlap = _centres_covered_by_extents(
            plane.horizontal,
            plane.vertical,
            covered_extents,
        )
        visible_values = plane.values.copy()
        visible_values[overlap] = np.nan
        extent = (*h_range, *v_range)

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
        ax.plot(
            [h_range[0], h_range[1], h_range[1], h_range[0], h_range[0]],
            [v_range[0], v_range[0], v_range[1], v_range[1], v_range[0]],
            color="black",
            linewidth=0.8,
            alpha=0.7,
            zorder=4,
        )
        ax.text(
            0.5 * (h_range[0] + h_range[1]),
            v_range[1] - 0.025 * (v_range[1] - v_range[0]),
            plane.distance_label,
            ha="center",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
            zorder=5,
        )
        covered_extents.append((*h_range, *v_range))
        rows.append(
            manifest_row(
                plane,
                component=component,
                harmonic_quantity=harmonic_quantity,
                requested_plane_value=requested_plane_value,
                min_valid_fraction="",
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                output=output,
                overlap_cells_masked=int(overlap.sum()),
                visible_values=visible_values,
            )
        )

    assert last_image is not None
    cbar = fig.colorbar(last_image, ax=ax)
    cbar.set_label(colorbar_label(component, harmonic_quantity))
    ax.set_xlabel(f"{first.horizontal_axis} [mm]")
    ax.set_ylabel(f"{first.vertical_axis} [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        title
        or (
            f"{first.case_key}: {component} harmonic {harmonic_quantity}, "
            f"{first.plane_axis}={requested_plane_value:g} composite"
        )
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return rows


def manifest_row(
    plane: HarmonicPlane,
    component: str,
    harmonic_quantity: str,
    requested_plane_value: float,
    min_valid_fraction: float | str,
    vmin: float | None,
    vmax: float | None,
    cmap_name: str,
    output: Path,
    overlap_cells_masked: int = 0,
    visible_values: np.ndarray | None = None,
) -> dict[str, object]:
    plotted_values = plane.values if visible_values is None else visible_values
    return {
        "case_key": plane.case_key,
        "source_label": plane.source_label,
        "distance_label": plane.distance_label,
        "motion_type": plane.motion_type,
        "plane_axis": plane.plane_axis,
        "plane_value_requested": requested_plane_value,
        "plane_value_used": plane.plane_value,
        "component": component,
        "harmonic_quantity": harmonic_quantity,
        "min_valid_fraction": min_valid_fraction,
        "accepted_cells": "" if plane.accepted_cells is None else plane.accepted_cells,
        "overlap_cells_masked": overlap_cells_masked,
        "visible_cells": int(np.isfinite(plotted_values).sum()),
        "min_visible_voxel_fraction": min_visible_voxel_fraction(plane),
        "color_vmin": "" if vmin is None else vmin,
        "color_vmax": "" if vmax is None else vmax,
        "cmap": cmap_name,
        "source_file": str(plane.path),
        "output_file": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one phase-coherent first-harmonic map from phase_average.nc, "
            "defaulting to the z=0 u-amplitude plane."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("phase_average_root_or_file", type=Path)
    parser.add_argument("--case", default=None, help="moving case/label substring to plot, e.g. SurgeLF")
    parser.add_argument("--plane", choices=("x", "y", "z"), default="z")
    parser.add_argument("--plane-value", type=float, default=0.0)
    parser.add_argument(
        "--component",
        "--harmonic-component",
        dest="component",
        choices=("u", "v", "w"),
        default="u",
    )
    parser.add_argument(
        "--harmonic-quantity",
        choices=("amplitude", "phase", "a", "b", "offset", "r2"),
        default="amplitude",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        help=(
            "mask cells whose lowest phase-bin valid-sample fraction is not "
            "above this threshold for the selected component; use 0 to disable"
        ),
    )
    parser.add_argument("--cmap", default=None, help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--output", type=Path, default=None, help="PNG file to write")
    parser.add_argument("--output-folder", type=Path, default=None, help="folder for auto-named PNG and CSV")
    parser.add_argument(
        "--single",
        action="store_true",
        help="require exactly one matching plane instead of compositing all matched distances",
    )
    parser.add_argument(
        "--include-static",
        action="store_true",
        help=(
            "allow static phase_average.nc files to be plotted as an explicit "
            "baseline/noise comparison"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = discover_phase_average_files(args.phase_average_root_or_file)
    if not paths:
        raise SystemExit(f"No phase_average.nc files found in {args.phase_average_root_or_file}")

    planes = [
        load_harmonic_plane(
            path,
            plane_axis=args.plane,
            plane_value=args.plane_value,
            component=args.component,
            harmonic_quantity=args.harmonic_quantity,
            min_valid_fraction=args.min_valid_fraction,
        )
        for path in paths
    ]
    matched_planes = filter_planes(planes, args.case)
    selected_planes = require_moving_planes(
        matched_planes,
        include_static=args.include_static,
    )
    cmap_name = args.cmap or default_cmap(args.harmonic_quantity)

    if args.single or args.phase_average_root_or_file.is_file():
        plane = choose_one_plane(selected_planes)
        vmin, vmax = color_limits(plane.values, args.harmonic_quantity, args.vmin, args.vmax)
        output = output_path_for(
            args.output,
            args.output_folder,
            plane,
            component=args.component,
            harmonic_quantity=args.harmonic_quantity,
            requested_plane_value=args.plane_value,
        )
        plot_harmonic_plane(
            plane,
            component=args.component,
            harmonic_quantity=args.harmonic_quantity,
            requested_plane_value=args.plane_value,
            output=output,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            title=args.title,
            dpi=args.dpi,
        )
        rows = [
            manifest_row(
                plane,
                component=args.component,
                harmonic_quantity=args.harmonic_quantity,
                requested_plane_value=args.plane_value,
                min_valid_fraction=args.min_valid_fraction,
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
                output=output,
            )
        ]
    else:
        case_keys = sorted({plane.case_key for plane in selected_planes})
        if len(case_keys) > 1:
            raise SystemExit(
                "Composite plotting needs one moving case. Pass --case to select one of: "
                f"{', '.join(case_keys)}"
            )
        vmin, vmax = color_limits_for_values(
            [plane.values for plane in selected_planes],
            args.harmonic_quantity,
            args.vmin,
            args.vmax,
        )
        output = composite_output_path_for(
            args.output,
            args.output_folder,
            case_key=case_keys[0],
            plane_axis=args.plane,
            component=args.component,
            harmonic_quantity=args.harmonic_quantity,
            requested_plane_value=args.plane_value,
        )
        rows = plot_harmonic_composite(
            selected_planes,
            component=args.component,
            harmonic_quantity=args.harmonic_quantity,
            requested_plane_value=args.plane_value,
            output=output,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            title=args.title,
            dpi=args.dpi,
        )
        for row in rows:
            row["min_valid_fraction"] = args.min_valid_fraction

    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved plot: {output.resolve()}", flush=True)
    print(f"Saved manifest: {output.with_suffix('.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
