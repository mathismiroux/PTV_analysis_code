from __future__ import annotations

import argparse
import csv
import fnmatch
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.validity import INVALID_SAMPLE_MODES, valid_component_samples
from scripts.plot_mean_wake_z0 import parse_case_distance


REQUIRED_MEAN_DATASETS = frozenset(
    ("x", "y", "z", "u_mean", "v_mean", "w_mean")
)
DEFAULT_EXCLUDE_PATTERNS = ("*_z0",)


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


def _is_radial_mean_file(path: Path) -> bool:
    if not path.exists() or path.name != "mean.nc":
        return False
    try:
        with h5py.File(path, "r") as h5:
            return REQUIRED_MEAN_DATASETS.issubset(h5.keys())
    except OSError:
        return False


def _mean_case_key(path: Path) -> str:
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
    case_key, _, _ = parse_case_distance(label)
    return case_key


def _is_excluded(path: Path, exclude_patterns: tuple[str, ...]) -> bool:
    if not exclude_patterns:
        return False
    return any(
        fnmatch.fnmatch(part, pattern)
        for part in path.parts
        for pattern in exclude_patterns
    )


def _sibling_distance_mean_files(
    path: Path,
    exclude_patterns: tuple[str, ...],
) -> list[Path]:
    mean_file = path if path.is_file() else path / "mean.nc"
    if not _is_radial_mean_file(mean_file) or _is_excluded(
        mean_file,
        exclude_patterns,
    ):
        return []
    case_key = _mean_case_key(mean_file)
    parent = mean_file.parent.parent
    siblings = []
    for candidate in parent.glob("*/mean.nc"):
        if (
            _is_radial_mean_file(candidate)
            and not _is_excluded(candidate, exclude_patterns)
            and _mean_case_key(candidate) == case_key
        ):
            siblings.append(candidate)
    return sorted(siblings)


def _discover_mean_files_under(
    path: Path,
    exclude_patterns: tuple[str, ...],
) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        candidate
        for candidate in path.rglob("mean.nc")
        if _is_radial_mean_file(candidate)
        and not _is_excluded(candidate, exclude_patterns)
    )


def discover_radial_mean_files(
    inputs: list[Path],
    include_sibling_distances: bool,
    exclude_patterns: tuple[str, ...] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            if input_path.name != "mean.nc":
                continue
            if _is_excluded(input_path, exclude_patterns):
                continue
            if include_sibling_distances:
                paths.extend(
                    _sibling_distance_mean_files(input_path, exclude_patterns)
                )
            else:
                paths.append(input_path)
            continue

        direct_mean = input_path / "mean.nc"
        if _is_radial_mean_file(direct_mean):
            if _is_excluded(direct_mean, exclude_patterns):
                continue
            if include_sibling_distances:
                paths.extend(
                    _sibling_distance_mean_files(input_path, exclude_patterns)
                )
            else:
                paths.append(direct_mean)
            continue

        paths.extend(_discover_mean_files_under(input_path, exclude_patterns))

    unique: dict[str, Path] = {}
    for path in paths:
        if path.exists():
            unique[str(path.resolve())] = path
    return sorted(unique.values())


def _valid_volume_mask(
    h5: h5py.File,
    invalid_samples: str,
    require_all_components: bool,
    min_valid_fraction: float,
) -> np.ndarray:
    component_masks = [
        valid_component_samples(h5[f"{component}_mean"][:], invalid_samples)
        for component in ("u", "v", "w")
    ]
    finite_masks = [
        np.isfinite(h5[f"{component}_mean"][:]) for component in ("u", "v", "w")
    ]
    masks = [valid & finite for valid, finite in zip(component_masks, finite_masks)]
    if min_valid_fraction > 0.0:
        masks = [
            mask
            & _valid_count_mask(
                h5,
                component,
                min_valid_fraction,
            )
            for mask, component in zip(masks, ("u", "v", "w"))
        ]
    if require_all_components:
        return np.logical_and.reduce(masks)
    return masks[0]


def _n_times_from_mean_file(h5: h5py.File) -> int:
    if "input_shape_time_z_y_x" in h5.attrs:
        return int(h5.attrs["input_shape_time_z_y_x"][0])
    if "t" in h5:
        return len(h5["t"])
    raise ValueError(
        "Missing input_shape_time_z_y_x metadata or t coordinate; cannot apply "
        "a valid-fraction threshold to this mean file."
    )


def _valid_count_mask(
    h5: h5py.File,
    component: str,
    min_valid_fraction: float,
) -> np.ndarray:
    n_times = _n_times_from_mean_file(h5)
    min_valid_count = int(np.ceil(min_valid_fraction * n_times))
    if "vector_count" in h5:
        count = h5["vector_count"][:]
    else:
        count_name = f"{component}_count"
        if count_name not in h5:
            raise ValueError(
                f"Missing {count_name!r} or shared 'vector_count'; cannot "
                "apply a valid-fraction threshold to this mean file."
            )
        count = h5[count_name][:]
    return count >= min_valid_count


def radial_average_wake_deficit(
    path: Path,
    axis_y: float,
    axis_z: float,
    radial_bin_width: float,
    invalid_samples: str,
    require_all_components: bool,
    min_valid_fraction: float,
    u_inf: float | None,
) -> RadialWakeVolume:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    with h5py.File(path, "r") as h5:
        label = str(h5.attrs.get("label", path.parent.name))
        case_key, distance_label, distance_value = parse_case_distance(label)
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        u = h5["u_mean"][:].astype(np.float64)
        free_stream = _infer_u_inf(h5, u_inf)
        valid = _valid_volume_mask(
            h5,
            invalid_samples,
            require_all_components,
            min_valid_fraction,
        )
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


def available_radial_output_paths(
    output_folder: Path,
    case_key: str,
) -> tuple[Path, Path, Path]:
    stem = f"{case_key}_radial_wake_deficit"
    for index in range(1000):
        suffix = "" if index == 0 else f"_{index:03d}"
        candidate_stem = f"{stem}{suffix}"
        output = output_folder / f"{candidate_stem}.png"
        manifest = output_folder / f"{candidate_stem}.csv"
        data = output_folder / f"{candidate_stem}_data.csv"
        if not output.exists() and not manifest.exists() and not data.exists():
            return output, manifest, data
    raise FileExistsError(
        f"Could not find an available output name for {stem} in {output_folder}"
    )


def plot_radial_group(
    case_key: str,
    volumes: list[RadialWakeVolume],
    output_folder: Path,
    axis_y: float,
    axis_z: float,
    radial_bin_width: float,
    invalid_samples: str,
    require_all_components: bool,
    min_valid_fraction: float,
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
                "min_valid_fraction": min_valid_fraction,
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
    output, manifest, data_output = available_radial_output_paths(
        output_folder,
        case_key,
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)
    write_manifest(manifest, manifest_rows)
    write_manifest(data_output, data_rows)
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
            "folder where plot PNGs and plot manifests are written; existing "
            "files are not overwritten"
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
        "--radial-bin-width",
        type=float,
        default=25.0,
        help="radial averaging bin width, in the same units as y/z",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help="mean velocity samples excluded before radial averaging",
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
            "it contributes to the radial average"
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
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        raise SystemExit("--min-valid-fraction must be between 0 and 1.")
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
        volumes = [
            radial_average_wake_deficit(
                path,
                axis_y=args.axis_y,
                axis_z=args.axis_z,
                radial_bin_width=args.radial_bin_width,
                invalid_samples=args.invalid_samples,
                require_all_components=args.require_all_components,
                min_valid_fraction=args.min_valid_fraction,
                u_inf=args.u_inf,
            )
            for path in mean_files
        ]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
            min_valid_fraction=args.min_valid_fraction,
            cmap_name=args.cmap,
            contour_step=args.contour_step,
            contour_label_step=args.contour_label_step,
        )
        print(f"Saved plot: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
