from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.reader import FlowDataset
from ptv_flow.validity import INVALID_SAMPLE_MODES, valid_component_samples


@dataclass(frozen=True)
class PlaneSpec:
    axis: str
    index: int
    value: float
    horizontal_axis: str
    vertical_axis: str

    @property
    def label(self) -> str:
        return f"{self.axis}={self.value:.6g}"


@dataclass(frozen=True)
class PlaneStats:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    spec: PlaneSpec


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - value)))


def default_x_plane_values(flow: FlowDataset) -> tuple[float, float]:
    x = flow.coordinate("x")
    return (
        float(x.min() + (x.max() - x.min()) / 3.0),
        float(x.min() + 2.0 * (x.max() - x.min()) / 3.0),
    )


def centered_coordinate(flow: FlowDataset, axis: str) -> float:
    values = flow.coordinate(axis)
    return float(0.5 * (values.min() + values.max()))


def common_coordinate_range(flows: list[FlowDataset], axis: str) -> tuple[float, float]:
    mins = [float(flow.coordinate(axis).min()) for flow in flows]
    maxs = [float(flow.coordinate(axis).max()) for flow in flows]
    lower = max(mins)
    upper = min(maxs)
    if lower >= upper:
        raise ValueError(f"No overlapping {axis!r} coordinate range")
    return lower, upper


def default_common_x_plane_values(flows: list[FlowDataset]) -> tuple[float, float]:
    lower, upper = common_coordinate_range(flows, "x")
    return (
        float(lower + (upper - lower) / 3.0),
        float(lower + 2.0 * (upper - lower) / 3.0),
    )


def centered_common_coordinate(flows: list[FlowDataset], axis: str) -> float:
    lower, upper = common_coordinate_range(flows, axis)
    return float(0.5 * (lower + upper))


def plane_spec(flow: FlowDataset, axis: str, value: float) -> PlaneSpec:
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    values = flow.coordinate(axis)
    index = nearest_index(values, value)
    if axis == "x":
        horizontal_axis = "y"
        vertical_axis = "z"
    elif axis == "y":
        horizontal_axis = "x"
        vertical_axis = "z"
    else:
        horizontal_axis = "x"
        vertical_axis = "y"
    return PlaneSpec(
        axis=axis,
        index=index,
        value=float(values[index]),
        horizontal_axis=horizontal_axis,
        vertical_axis=vertical_axis,
    )


def _component_plane(data: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return data[:, :, :, index]
    if axis == "y":
        return data[:, :, index, :]
    if axis == "z":
        return data[:, index, :, :]
    raise ValueError("axis must be one of 'x', 'y', or 'z'")


def _quantity_plane(flow: FlowDataset, quantity: str, spec: PlaneSpec) -> np.ndarray:
    if quantity in {"u", "v", "w"}:
        return _component_plane(flow._file[quantity], spec.axis, spec.index).astype(
            np.float64
        )
    if quantity == "speed":
        u = _component_plane(flow._file["u"], spec.axis, spec.index)
        v = _component_plane(flow._file["v"], spec.axis, spec.index)
        w = _component_plane(flow._file["w"], spec.axis, spec.index)
        return np.sqrt(u * u + v * v + w * w).astype(np.float64)
    raise ValueError("quantity must be one of 'speed', 'u', 'v', or 'w'")


def temporal_plane_stats(
    flow: FlowDataset,
    quantity: str,
    spec: PlaneSpec,
    invalid_samples: str = "nan",
) -> PlaneStats:
    data = _quantity_plane(flow, quantity, spec)
    valid = valid_component_samples(data, invalid_samples)
    count = valid.sum(axis=0, dtype=np.uint32)

    mean = np.full(data.shape[1:], np.nan, dtype=np.float64)
    sums = np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64)
    np.divide(sums, count, out=mean, where=count > 0)

    variance = np.full(data.shape[1:], np.nan, dtype=np.float64)
    fluctuation = data - mean[None, :, :]
    sum_squares = np.where(valid, fluctuation * fluctuation, 0.0).sum(
        axis=0, dtype=np.float64
    )
    np.divide(sum_squares, count, out=variance, where=count > 0)

    return PlaneStats(
        mean=mean,
        std=np.sqrt(variance),
        count=count,
        horizontal=flow.coordinate(spec.horizontal_axis),
        vertical=flow.coordinate(spec.vertical_axis),
        spec=spec,
    )


def _shared_limits(
    stats_by_file: list[list[PlaneStats]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    means = np.concatenate(
        [
            stats.mean[np.isfinite(stats.mean)].ravel()
            for file_stats in stats_by_file
            for stats in file_stats
        ]
    )
    stds = np.concatenate(
        [
            stats.std[np.isfinite(stats.std)].ravel()
            for file_stats in stats_by_file
            for stats in file_stats
        ]
    )
    return (
        (float(np.nanmin(means)), float(np.nanmax(means))),
        (float(np.nanmin(stds)), float(np.nanmax(stds))),
    )


def plot_export_plane_comparison(
    stats_by_file: list[list[PlaneStats]],
    labels: list[str],
    quantity: str,
    output: Path,
) -> Path:
    n_files = len(stats_by_file)
    n_planes = len(stats_by_file[0])
    mean_limits, std_limits = _shared_limits(stats_by_file)

    fig, axes = plt.subplots(
        n_files,
        n_planes * 2,
        figsize=(4.0 * n_planes * 2, 3.4 * n_files),
        squeeze=False,
        constrained_layout=True,
    )
    mean_image = None
    std_image = None
    for row, (label, file_stats) in enumerate(zip(labels, stats_by_file)):
        for plane_index, stats in enumerate(file_stats):
            extent = (
                float(stats.horizontal.min()),
                float(stats.horizontal.max()),
                float(stats.vertical.min()),
                float(stats.vertical.max()),
            )
            mean_ax = axes[row, plane_index * 2]
            std_ax = axes[row, plane_index * 2 + 1]
            mean_image = mean_ax.imshow(
                stats.mean,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                vmin=mean_limits[0],
                vmax=mean_limits[1],
            )
            std_image = std_ax.imshow(
                stats.std,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                vmin=std_limits[0],
                vmax=std_limits[1],
            )
            mean_ax.set_title(f"{stats.spec.label} mean")
            std_ax.set_title(f"{stats.spec.label} std")
            for ax in (mean_ax, std_ax):
                ax.set_xlabel(stats.spec.horizontal_axis)
                ax.set_ylabel(stats.spec.vertical_axis)
            mean_ax.text(
                0.02,
                0.98,
                label,
                transform=mean_ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.45, "linewidth": 0},
            )

    if mean_image is not None:
        fig.colorbar(mean_image, ax=axes[:, 0::2], shrink=0.85, label=f"{quantity} mean")
    if std_image is not None:
        fig.colorbar(std_image, ax=axes[:, 1::2], shrink=0.85, label=f"{quantity} std")
    fig.suptitle(f"Export comparison: temporal mean and std of {quantity}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _ascending(values: np.ndarray, field: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if values[0] <= values[-1]:
        return values, field
    return values[::-1], np.flip(field, axis=axis)


def interpolate_to_grid(
    source_horizontal: np.ndarray,
    source_vertical: np.ndarray,
    source_field: np.ndarray,
    target_horizontal: np.ndarray,
    target_vertical: np.ndarray,
) -> np.ndarray:
    source_horizontal, source_field = _ascending(source_horizontal, source_field, axis=1)
    source_vertical, source_field = _ascending(source_vertical, source_field, axis=0)
    target_horizontal = np.asarray(target_horizontal)
    target_vertical = np.asarray(target_vertical)

    intermediate = np.full(
        (source_field.shape[0], target_horizontal.size),
        np.nan,
        dtype=np.float64,
    )
    for row_index, row in enumerate(source_field):
        valid = np.isfinite(row)
        if valid.sum() >= 2:
            intermediate[row_index, :] = np.interp(
                target_horizontal,
                source_horizontal[valid],
                row[valid],
                left=np.nan,
                right=np.nan,
            )

    result = np.full(
        (target_vertical.size, target_horizontal.size),
        np.nan,
        dtype=np.float64,
    )
    for column_index in range(intermediate.shape[1]):
        column = intermediate[:, column_index]
        valid = np.isfinite(column)
        if valid.sum() >= 2:
            result[:, column_index] = np.interp(
                target_vertical,
                source_vertical[valid],
                column[valid],
                left=np.nan,
                right=np.nan,
            )
    return result


def _crop_to_overlap(stats: PlaneStats, other: PlaneStats) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_min = max(float(stats.horizontal.min()), float(other.horizontal.min()))
    h_max = min(float(stats.horizontal.max()), float(other.horizontal.max()))
    v_min = max(float(stats.vertical.min()), float(other.vertical.min()))
    v_max = min(float(stats.vertical.max()), float(other.vertical.max()))
    if h_min >= h_max or v_min >= v_max:
        raise ValueError(f"No overlapping plane domain for {stats.spec.label}")
    h_mask = (stats.horizontal >= h_min) & (stats.horizontal <= h_max)
    v_mask = (stats.vertical >= v_min) & (stats.vertical <= v_max)
    return (
        stats.horizontal[h_mask],
        stats.vertical[v_mask],
        np.ix_(v_mask, h_mask),
        np.array([h_min, h_max, v_min, v_max], dtype=float),
    )


def plot_difference_comparison(
    reference_stats: list[PlaneStats],
    comparison_stats: list[PlaneStats],
    labels: list[str],
    quantity: str,
    output: Path,
) -> Path:
    n_planes = len(reference_stats)
    fig, axes = plt.subplots(
        2,
        n_planes * 3,
        figsize=(4.0 * n_planes * 3, 7.0),
        squeeze=False,
        constrained_layout=True,
    )
    for plane_index, (reference, comparison) in enumerate(
        zip(reference_stats, comparison_stats)
    ):
        target_horizontal, target_vertical, indexer, extent_values = _crop_to_overlap(
            reference, comparison
        )
        extent = tuple(extent_values)
        ref_mean = reference.mean[indexer]
        ref_std = reference.std[indexer]
        comp_mean = interpolate_to_grid(
            comparison.horizontal,
            comparison.vertical,
            comparison.mean,
            target_horizontal,
            target_vertical,
        )
        comp_std = interpolate_to_grid(
            comparison.horizontal,
            comparison.vertical,
            comparison.std,
            target_horizontal,
            target_vertical,
        )
        mean_diff = comp_mean - ref_mean
        std_diff = comp_std - ref_std

        panels = (
            (0, ref_mean, f"{labels[0]} mean"),
            (0, comp_mean, f"{labels[1]} mean on {labels[0]} grid"),
            (0, mean_diff, f"{labels[1]} - {labels[0]} mean"),
            (1, ref_std, f"{labels[0]} std"),
            (1, comp_std, f"{labels[1]} std on {labels[0]} grid"),
            (1, std_diff, f"{labels[1]} - {labels[0]} std"),
        )
        mean_limit = _symmetric_or_data_limits([ref_mean, comp_mean])
        std_limit = _symmetric_or_data_limits([ref_std, comp_std])
        mean_diff_limit = _symmetric_or_data_limits([mean_diff], symmetric=True)
        std_diff_limit = _symmetric_or_data_limits([std_diff], symmetric=True)
        images = []
        for offset, (row, field, title) in enumerate(panels):
            ax = axes[row, plane_index * 3 + offset % 3]
            if row == 0 and offset % 3 < 2:
                limits = mean_limit
                cmap = "viridis"
            elif row == 1 and offset % 3 < 2:
                limits = std_limit
                cmap = "magma"
            elif row == 0:
                limits = mean_diff_limit
                cmap = "coolwarm"
            else:
                limits = std_diff_limit
                cmap = "coolwarm"
            image = ax.imshow(
                field,
                extent=extent,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            images.append(image)
            ax.set_title(f"{reference.spec.label}\n{title}")
            ax.set_xlabel(reference.spec.horizontal_axis)
            ax.set_ylabel(reference.spec.vertical_axis)
            fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(f"Export difference on common overlap: {quantity}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _symmetric_or_data_limits(
    fields: list[np.ndarray], symmetric: bool = False
) -> tuple[float, float]:
    values = np.concatenate([field[np.isfinite(field)].ravel() for field in fields])
    if values.size == 0:
        return 0.0, 1.0
    if symmetric:
        limit = float(np.nanmax(np.abs(values)))
        return -limit, limit
    return float(np.nanmin(values)), float(np.nanmax(values))


def compare_exports(
    paths: list[Path],
    labels: list[str],
    quantity: str,
    output: Path,
    x_planes: tuple[float, float] | None,
    y_plane: float | None,
    invalid_samples: str = "nan",
    difference: bool = False,
    reference_grid: str = "first",
) -> Path:
    stats_by_file: list[list[PlaneStats]] = []
    reference_x_planes = x_planes
    reference_y_plane = y_plane
    with _open_flows(paths) as flows:
        if reference_x_planes is None:
            reference_x_planes = default_common_x_plane_values(flows)
        if reference_y_plane is None:
            reference_y_plane = centered_common_coordinate(flows, "y")
        for flow in flows:
            if reference_x_planes is None:
                reference_x_planes = default_x_plane_values(flow)
            if reference_y_plane is None:
                reference_y_plane = centered_coordinate(flow, "y")
            specs = [
                plane_spec(flow, "x", reference_x_planes[0]),
                plane_spec(flow, "x", reference_x_planes[1]),
                plane_spec(flow, "y", reference_y_plane),
            ]
            stats_by_file.append(
                [
                    temporal_plane_stats(
                        flow, quantity, spec, invalid_samples=invalid_samples
                    )
                    for spec in specs
                ]
            )
    if difference:
        if len(paths) != 2:
            raise ValueError("--difference requires exactly two input files")
        if reference_grid == "first":
            return plot_difference_comparison(
                stats_by_file[0], stats_by_file[1], labels, quantity, output
            )
        if reference_grid == "second":
            return plot_difference_comparison(
                stats_by_file[1], stats_by_file[0], labels[::-1], quantity, output
            )
        raise ValueError("reference_grid must be 'first' or 'second'")
    return plot_export_plane_comparison(stats_by_file, labels, quantity, output)


def _source_label(provided: bool, default_note: str = "default") -> str:
    return "provided" if provided else default_note


def format_run_configuration(
    paths: list[Path],
    labels: list[str],
    quantity: str,
    output: Path,
    x_planes: tuple[float, float] | None,
    y_plane: float | None,
    invalid_samples: str,
    difference: bool,
    reference_grid: str,
    provided: dict[str, bool],
) -> str:
    lines = ["Export plane comparison configuration:"]
    lines.append("  paths (provided):")
    for path in paths:
        lines.append(f"    - {path}")
    lines.extend(
        [
            f"  labels ({_source_label(provided['labels'], 'default from file stems')}): "
            f"{', '.join(labels)}",
            f"  quantity ({_source_label(provided['quantity'])}): {quantity}",
            f"  invalid_samples ({_source_label(provided['invalid_samples'])}): "
            f"{invalid_samples}",
            f"  output ({_source_label(provided['output'])}): {output}",
            f"  difference ({_source_label(provided['difference'])}): {difference}",
            f"  reference_grid ({_source_label(provided['reference_grid'])}): "
            f"{reference_grid}",
        ]
    )

    with _open_flows(paths) as flows:
        resolved_x_planes = x_planes or default_common_x_plane_values(flows)
        resolved_y_plane = y_plane
        if resolved_y_plane is None:
            resolved_y_plane = centered_common_coordinate(flows, "y")

        lines.extend(
            [
                (
                    "  x_planes "
                    f"({_source_label(provided['x_planes'], 'default from common overlap')}): "
                    f"{resolved_x_planes[0]:.9g}, {resolved_x_planes[1]:.9g}"
                ),
                (
                    "  y_plane "
                    f"({_source_label(provided['y_plane'], 'default from common overlap')}): "
                    f"{resolved_y_plane:.9g}"
                ),
                "  resolved nearest planes:",
            ]
        )
        for label, flow in zip(labels, flows):
            specs = [
                plane_spec(flow, "x", resolved_x_planes[0]),
                plane_spec(flow, "x", resolved_x_planes[1]),
                plane_spec(flow, "y", resolved_y_plane),
            ]
            lines.append(f"    {label}:")
            for spec in specs:
                lines.append(
                    f"      - requested {spec.axis}, nearest {spec.label}, "
                    f"index={spec.index}, plot axes={spec.horizontal_axis}/{spec.vertical_axis}"
                )
    return "\n".join(lines)


class _open_flows:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.flows: list[FlowDataset] = []

    def __enter__(self) -> list[FlowDataset]:
        self.flows = [FlowDataset(path) for path in self.paths]
        return self.flows

    def __exit__(self, *exc_info: object) -> None:
        for flow in self.flows:
            flow.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visually compare temporal mean and standard deviation on matching "
            "planes from multiple raw velocity exports."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="raw .nc export files")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="display labels; defaults to file stems",
    )
    parser.add_argument(
        "--quantity",
        choices=("speed", "u", "v", "w"),
        default="speed",
        help="quantity to compare",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help=(
            "raw samples to exclude from statistics: zero ignores exact zeros, "
            "nan ignores NaN/inf values, zero-or-nan ignores both, none excludes nothing"
        ),
    )
    parser.add_argument(
        "--x-planes",
        nargs=2,
        type=float,
        default=None,
        metavar=("X1", "X2"),
        help="two x-plane coordinate values; defaults to one-third and two-thirds of x range",
    )
    parser.add_argument(
        "--y-plane",
        type=float,
        default=None,
        help="y coordinate for the streamwise-vertical plane; defaults to center of y range",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/export_plane_comparison.png"),
        help="output figure path",
    )
    parser.add_argument(
        "--difference",
        action="store_true",
        help=(
            "for exactly two files, project the second export onto the reference "
            "grid over the common overlap and plot differences"
        ),
    )
    parser.add_argument(
        "--reference-grid",
        choices=("first", "second"),
        default="first",
        help="which file supplies the grid in --difference mode",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = args.labels or [path.stem for path in args.paths]
    if len(labels) != len(args.paths):
        raise SystemExit("--labels must have the same number of entries as paths")
    provided = {
        "labels": args.labels is not None,
        "quantity": _flag_was_provided("--quantity"),
        "invalid_samples": _flag_was_provided("--invalid-samples"),
        "output": _flag_was_provided("--output"),
        "difference": _flag_was_provided("--difference"),
        "reference_grid": _flag_was_provided("--reference-grid"),
        "x_planes": args.x_planes is not None,
        "y_plane": args.y_plane is not None,
    }
    x_planes = tuple(args.x_planes) if args.x_planes is not None else None
    print(
        format_run_configuration(
            paths=args.paths,
            labels=labels,
            quantity=args.quantity,
            output=args.output,
            x_planes=x_planes,
            y_plane=args.y_plane,
            invalid_samples=args.invalid_samples,
            difference=args.difference,
            reference_grid=args.reference_grid,
            provided=provided,
        ),
        flush=True,
    )
    output = compare_exports(
        paths=args.paths,
        labels=labels,
        quantity=args.quantity,
        output=args.output,
        x_planes=x_planes,
        y_plane=args.y_plane,
        invalid_samples=args.invalid_samples,
        difference=args.difference,
        reference_grid=args.reference_grid,
    )
    print(f"Saved comparison figure to: {output.resolve()}")


def _flag_was_provided(flag: str) -> bool:
    prefix = f"{flag}="
    return any(arg == flag or arg.startswith(prefix) for arg in sys.argv[1:])


if __name__ == "__main__":
    main()
