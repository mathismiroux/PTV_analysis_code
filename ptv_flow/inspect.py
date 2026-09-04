from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np

from ptv_flow.postprocess import PhaseAverageVolume, TemporalAverageVolume
from ptv_flow.reader import FlowDataset
from ptv_flow.validity import valid_component_samples
from ptv_flow.visualize import _draw_xy_vector_plane

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ComponentMean:
    mean: float
    count: int
    accepted: bool = True


@dataclass(frozen=True)
class CellInspection:
    time_index: int
    z_index: int
    y_index: int
    x_index: int
    time: float
    z: float
    y: float
    x: float
    raw_u: float
    raw_v: float
    raw_w: float
    raw_speed: float
    computed_u: ComponentMean
    computed_v: ComponentMean
    computed_w: ComponentMean
    average_u: ComponentMean | None = None
    average_v: ComponentMean | None = None
    average_w: ComponentMean | None = None
    interpolated_u: float | None = None
    interpolated_v: float | None = None
    interpolated_w: float | None = None
    interpolated_speed: float | None = None
    filled_u: bool | None = None
    filled_v: bool | None = None
    filled_w: bool | None = None
    phase_coverage: "PhaseCoverage | None" = None


@dataclass(frozen=True)
class CoverageStats:
    total: int
    accepted: int
    rejected: int

    @property
    def rejected_fraction(self) -> float:
        if self.total == 0:
            return float("nan")
        return self.rejected / self.total


@dataclass(frozen=True)
class PhaseCoverage:
    phase_degrees: np.ndarray
    sample_counts: np.ndarray
    min_valid_counts: np.ndarray
    u_counts: np.ndarray
    v_counts: np.ndarray
    w_counts: np.ndarray
    u_means: np.ndarray | None = None
    v_means: np.ndarray | None = None
    w_means: np.ndarray | None = None
    z_index: int | None = None
    y_index: int | None = None
    x_index: int | None = None

    @property
    def n_phase_bins(self) -> int:
        return int(self.phase_degrees.size)


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - value)))


def step_index(current: int, offset: int, upper: int) -> int:
    return int(np.clip(current + offset, 0, upper))


def _average_count_at(
    average: TemporalAverageVolume,
    component: str,
    z_index: int,
    y_index: int,
    x_index: int,
) -> int:
    count_name = f"{component}_count"
    if count_name in average._file:
        return int(average._file[count_name][z_index, y_index, x_index])
    if "vector_count" in average._file:
        return int(average._file["vector_count"][z_index, y_index, x_index])
    return -1


def component_mean_ignoring_zero(
    series: np.ndarray, min_valid_count: int = 1, invalid_samples: str = "nan"
) -> ComponentMean:
    valid = valid_component_samples(series, invalid_samples)
    count = int(valid.sum())
    if count == 0 or count < min_valid_count:
        return ComponentMean(mean=float("nan"), count=count, accepted=False)
    return ComponentMean(mean=float(series[valid].mean()), count=count, accepted=True)


def _valid_counts_for_z_plane(
    flow: FlowDataset, z_index: int, invalid_samples: str = "nan"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        valid_component_samples(
            flow._file["u"][:, z_index, :, :], invalid_samples
        ).sum(axis=0),
        valid_component_samples(
            flow._file["v"][:, z_index, :, :], invalid_samples
        ).sum(axis=0),
        valid_component_samples(
            flow._file["w"][:, z_index, :, :], invalid_samples
        ).sum(axis=0),
    )


def _apply_inspector_valid_mask(
    speed: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    counts: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_valid_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if min_valid_count <= 1:
        accepted = np.ones(speed.shape, dtype=bool)
        return speed, u, v, accepted

    u_count, v_count, w_count = counts
    accepted = (
        (u_count >= min_valid_count)
        & (v_count >= min_valid_count)
        & (w_count >= min_valid_count)
    )
    display_speed = speed
    display_u = np.where(accepted, u, 0.0)
    display_v = np.where(accepted, v, 0.0)
    return display_speed, display_u, display_v, accepted


def _rejected_overlay_rgba(accepted: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*accepted.shape, 4), dtype=float)
    overlay[~accepted] = (0.95, 0.10, 0.10, 0.38)
    return overlay


def _coverage_from_counts(
    counts: tuple[np.ndarray, np.ndarray, np.ndarray],
    min_valid_count: int,
) -> CoverageStats:
    u_count, v_count, w_count = counts
    accepted = (
        (u_count >= min_valid_count)
        & (v_count >= min_valid_count)
        & (w_count >= min_valid_count)
    )
    accepted_count = int(np.count_nonzero(accepted))
    total = int(accepted.size)
    return CoverageStats(
        total=total,
        accepted=accepted_count,
        rejected=total - accepted_count,
    )


def _load_phase_signal(path: str) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.loadtxt(path)


def phase_values_for_flow(
    flow: FlowDataset,
    frequency_hz: float | None = None,
    phase_signal: str | None = None,
    phase_offset: float = 0.0,
) -> np.ndarray | None:
    if phase_signal is not None:
        phases = _load_phase_signal(phase_signal)
    elif frequency_hz is not None:
        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive")
        phases = TWO_PI * float(frequency_hz) * flow.coordinate("t") + phase_offset
    else:
        return None

    phases = np.asarray(phases, dtype=np.float64)
    if phases.ndim == 2:
        if 1 in phases.shape:
            phases = phases.reshape(-1)
        else:
            phases = phases[:, -1]
    phases = phases.reshape(-1)
    if phases.size != flow.n_times:
        raise ValueError(
            "phase signal length must match the raw time dimension: "
            f"{phases.size} != {flow.n_times}"
        )
    return phases % TWO_PI


def phase_coverage_for_cell(
    flow: FlowDataset,
    z_index: int,
    y_index: int,
    x_index: int,
    phases: np.ndarray,
    n_phase_bins: int,
    min_valid_fraction: float = 0.0,
) -> PhaseCoverage:
    """Return finite-sample phase coverage at one spatial cell.

    The coverage table intentionally counts non-NaN/non-infinite values at the
    selected voxel. It is not affected by zero-masking options used elsewhere.
    """

    if n_phase_bins <= 0:
        raise ValueError("n_phase_bins must be positive")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phases.size != flow.n_times:
        raise ValueError(
            "phase signal length must match the raw time dimension: "
            f"{phases.size} != {flow.n_times}"
        )

    phase_width = TWO_PI / float(n_phase_bins)
    phase_indices = np.floor((phases % TWO_PI) / phase_width).astype(np.int64)
    sample_counts = np.bincount(phase_indices, minlength=n_phase_bins).astype(np.uint32)
    min_valid_counts = np.ceil(
        min_valid_fraction * sample_counts.astype(np.float64)
    ).astype(np.uint32)
    component_counts = []
    component_means = []
    for name in ("u", "v", "w"):
        series = flow._file[name][:, z_index, y_index, x_index]
        valid = np.isfinite(series)
        bin_counts = np.bincount(
            phase_indices,
            weights=valid.astype(np.uint8),
            minlength=n_phase_bins,
        ).astype(np.uint32)
        bin_sums = np.bincount(
            phase_indices,
            weights=np.where(valid, series, 0.0),
            minlength=n_phase_bins,
        ).astype(np.float64)
        bin_means = np.full(n_phase_bins, np.nan, dtype=np.float64)
        np.divide(bin_sums, bin_counts, out=bin_means, where=bin_counts > 0)
        component_counts.append(bin_counts)
        component_means.append(bin_means)
    phase_degrees = np.degrees((np.arange(n_phase_bins) + 0.5) * phase_width)
    return PhaseCoverage(
        phase_degrees=phase_degrees,
        sample_counts=sample_counts,
        min_valid_counts=min_valid_counts,
        u_counts=component_counts[0],
        v_counts=component_counts[1],
        w_counts=component_counts[2],
        u_means=component_means[0],
        v_means=component_means[1],
        w_means=component_means[2],
        z_index=z_index,
        y_index=y_index,
        x_index=x_index,
    )


def _average_count_volumes(
    average: TemporalAverageVolume,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not all(f"{name}_count" in average._file for name in ("u", "v", "w")):
        return None
    return (
        average._file["u_count"][:],
        average._file["v_count"][:],
        average._file["w_count"][:],
    )


def _format_coverage(prefix: str, coverage: CoverageStats) -> str:
    percentage = 100.0 * coverage.rejected_fraction
    return (
        f"  {prefix}: empty={coverage.rejected}/{coverage.total} "
        f"({percentage:.1f}%), accepted={coverage.accepted}"
    )


def _format_phase_coverage(coverage: PhaseCoverage) -> list[str]:
    has_means = (
        coverage.u_means is not None
        and coverage.v_means is not None
        and coverage.w_means is not None
    )
    lines = [
        "",
        "Selected-voxel phase coverage",
        f"  bins: {coverage.n_phase_bins}",
        "  source: all frames at the selected x/y/z cell",
    ]
    if (
        coverage.z_index is not None
        and coverage.y_index is not None
        and coverage.x_index is not None
    ):
        lines.append(
            f"  selected indices: z={coverage.z_index}, "
            f"y={coverage.y_index}, x={coverage.x_index}"
        )
    lines.append(
        (
            "  phase  n  min  u  v  w  ok      u_bar      v_bar      w_bar"
            if has_means
            else "  phase  n  min  u  v  w  ok"
        )
    )
    for index, (phase, samples, minimum, u_count, v_count, w_count) in enumerate(
        zip(
            coverage.phase_degrees,
            coverage.sample_counts,
            coverage.min_valid_counts,
            coverage.u_counts,
            coverage.v_counts,
            coverage.w_counts,
        )
    ):
        ok = u_count >= minimum and v_count >= minimum and w_count >= minimum
        line = (
            f"  {phase:5.1f} {samples:2d} {minimum:4d} "
            f"{u_count:2d} {v_count:2d} {w_count:2d}  {'yes' if ok else 'no'}"
        )
        if has_means:
            line += (
                f"  {coverage.u_means[index]:9.4g}"
                f"  {coverage.v_means[index]:9.4g}"
                f"  {coverage.w_means[index]:9.4g}"
            )
        lines.append(line)
    return lines


def inspect_phase_average_counts(
    volume: PhaseAverageVolume,
    x_value: float,
    y_value: float,
    z_value: float,
    min_valid_fraction: float = 0.0,
) -> str:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    x_index = volume.nearest_index("x", x_value)
    y_index = volume.nearest_index("y", y_value)
    z_index = volume.nearest_index("z", z_value)
    counts = volume.phase_counts_at(z_index, y_index, x_index)
    sample_counts = counts["phase_sample_count"].astype(np.uint32)
    min_valid_counts = np.maximum(
        np.ceil(min_valid_fraction * sample_counts.astype(np.float64)).astype(np.uint32),
        1,
    )
    coverage = PhaseCoverage(
        phase_degrees=counts["phase_degrees"],
        sample_counts=sample_counts,
        min_valid_counts=min_valid_counts,
        u_counts=counts["u_phase_count"].astype(np.uint32),
        v_counts=counts["v_phase_count"].astype(np.uint32),
        w_counts=counts["w_phase_count"].astype(np.uint32),
        u_means=(
            volume._file["u_phase_mean"][:, z_index, y_index, x_index]
            if "u_phase_mean" in volume._file
            else None
        ),
        v_means=(
            volume._file["v_phase_mean"][:, z_index, y_index, x_index]
            if "v_phase_mean" in volume._file
            else None
        ),
        w_means=(
            volume._file["w_phase_mean"][:, z_index, y_index, x_index]
            if "w_phase_mean" in volume._file
            else None
        ),
        z_index=z_index,
        y_index=y_index,
        x_index=x_index,
    )

    lines = [
        "Phase-average file inspection",
        f"  source: {volume.path}",
        f"  min valid fraction: {min_valid_fraction:g}",
        "",
        "Selected cell",
        f"  indices: z={z_index}, y={y_index}, x={x_index}",
        f"  coords:  z={volume.coordinate('z')[z_index]:.6g}, "
        f"y={volume.coordinate('y')[y_index]:.6g}, "
        f"x={volume.coordinate('x')[x_index]:.6g}",
    ]
    lines.extend(_format_phase_coverage(coverage))
    return "\n".join(lines)


def _filled_image_alpha(filled: np.ndarray, filled_alpha: float = 0.45) -> np.ndarray:
    alpha = np.ones(filled.shape, dtype=float)
    alpha[filled] = filled_alpha
    return alpha


def validate_average_compatible(
    flow: FlowDataset, average: TemporalAverageVolume
) -> None:
    """Raise if a temporal-average file cannot be compared with a raw file."""

    if average.grid_shape != flow.grid_shape:
        raise ValueError(
            "Average file grid shape does not match raw file grid shape: "
            f"{average.grid_shape} != {flow.grid_shape}. "
            f"raw={flow.path}, average={average.path}"
        )

    for name in ("x", "y", "z"):
        raw_values = flow.coordinate(name)
        average_values = average.coordinate(name)
        if raw_values.shape != average_values.shape or not np.allclose(
            raw_values, average_values, equal_nan=True
        ):
            raise ValueError(
                f"Average file {name!r} coordinates do not match raw file. "
                f"raw={flow.path}, average={average.path}"
            )


def validate_interpolated_compatible(
    flow: FlowDataset, interpolated: FlowDataset
) -> None:
    """Raise if an interpolated file cannot be compared with a raw file."""

    if interpolated.shape != flow.shape:
        raise ValueError(
            "Interpolated file shape does not match raw file shape: "
            f"{interpolated.shape} != {flow.shape}. "
            f"raw={flow.path}, interpolated={interpolated.path}"
        )

    for name in ("t", "x", "y", "z"):
        raw_values = flow.coordinate(name)
        interpolated_values = interpolated.coordinate(name)
        if raw_values.shape != interpolated_values.shape or not np.allclose(
            raw_values, interpolated_values, equal_nan=True
        ):
            raise ValueError(
                f"Interpolated file {name!r} coordinates do not match raw file. "
                f"raw={flow.path}, interpolated={interpolated.path}"
            )

    has_shared_mask = "filled_mask" in interpolated._file
    has_component_masks = all(
        f"{name}_filled_mask" in interpolated._file for name in ("u", "v", "w")
    )
    if not has_shared_mask and not has_component_masks:
        raise ValueError(
            "Interpolated file is missing filled-mask datasets. Expected either "
            "'filled_mask' or 'u_filled_mask', 'v_filled_mask', and 'w_filled_mask'."
        )


def _filled_mask_dataset(interpolated: FlowDataset, component: str):
    name = f"{component}_filled_mask"
    if name in interpolated._file:
        return interpolated._file[name]
    return interpolated._file["filled_mask"]


def _interpolated_filled_plane(
    interpolated: FlowDataset,
    time_index: int,
    z_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = tuple(
        _filled_mask_dataset(interpolated, name)[time_index, z_index, :, :].astype(bool)
        for name in ("u", "v", "w")
    )
    any_filled = np.logical_or.reduce(masks)
    return masks[0], masks[1], masks[2], any_filled


def inspect_cell(
    flow: FlowDataset,
    time_index: int,
    z_index: int,
    y_index: int,
    x_index: int,
    average: TemporalAverageVolume | None = None,
    interpolated: FlowDataset | None = None,
    min_valid_count: int = 1,
    invalid_samples: str = "nan",
    phases: np.ndarray | None = None,
    n_phase_bins: int = 16,
    min_valid_fraction: float = 0.0,
) -> CellInspection:
    if time_index < 0:
        time_index += flow.n_times
    if not 0 <= time_index < flow.n_times:
        raise IndexError(f"time_index must be in [0, {flow.n_times - 1}]")

    z_len, y_len, x_len = flow.grid_shape
    if not 0 <= z_index < z_len:
        raise IndexError(f"z_index must be in [0, {z_len - 1}]")
    if not 0 <= y_index < y_len:
        raise IndexError(f"y_index must be in [0, {y_len - 1}]")
    if not 0 <= x_index < x_len:
        raise IndexError(f"x_index must be in [0, {x_len - 1}]")

    raw_u = float(flow._file["u"][time_index, z_index, y_index, x_index])
    raw_v = float(flow._file["v"][time_index, z_index, y_index, x_index])
    raw_w = float(flow._file["w"][time_index, z_index, y_index, x_index])

    interpolated_u = None
    interpolated_v = None
    interpolated_w = None
    interpolated_speed = None
    filled_u = None
    filled_v = None
    filled_w = None
    if interpolated is not None:
        interpolated_u = float(interpolated._file["u"][time_index, z_index, y_index, x_index])
        interpolated_v = float(interpolated._file["v"][time_index, z_index, y_index, x_index])
        interpolated_w = float(interpolated._file["w"][time_index, z_index, y_index, x_index])
        interpolated_speed = float(
            np.sqrt(
                interpolated_u * interpolated_u
                + interpolated_v * interpolated_v
                + interpolated_w * interpolated_w
            )
        )
        filled_u = bool(
            _filled_mask_dataset(interpolated, "u")[time_index, z_index, y_index, x_index]
        )
        filled_v = bool(
            _filled_mask_dataset(interpolated, "v")[time_index, z_index, y_index, x_index]
        )
        filled_w = bool(
            _filled_mask_dataset(interpolated, "w")[time_index, z_index, y_index, x_index]
        )

    average_u = None
    average_v = None
    average_w = None
    if average is not None:
        u_count = _average_count_at(average, "u", z_index, y_index, x_index)
        v_count = _average_count_at(average, "v", z_index, y_index, x_index)
        w_count = _average_count_at(average, "w", z_index, y_index, x_index)
        average_u = ComponentMean(
            mean=(
                float(average._file["u_mean"][z_index, y_index, x_index])
                if u_count < 0 or u_count >= min_valid_count
                else float("nan")
            ),
            count=u_count,
            accepted=u_count < 0 or u_count >= min_valid_count,
        )
        average_v = ComponentMean(
            mean=(
                float(average._file["v_mean"][z_index, y_index, x_index])
                if v_count < 0 or v_count >= min_valid_count
                else float("nan")
            ),
            count=v_count,
            accepted=v_count < 0 or v_count >= min_valid_count,
        )
        average_w = ComponentMean(
            mean=(
                float(average._file["w_mean"][z_index, y_index, x_index])
                if w_count < 0 or w_count >= min_valid_count
                else float("nan")
            ),
            count=w_count,
            accepted=w_count < 0 or w_count >= min_valid_count,
        )

    phase_coverage = None
    if phases is not None:
        phase_coverage = phase_coverage_for_cell(
            flow,
            z_index=z_index,
            y_index=y_index,
            x_index=x_index,
            phases=phases,
            n_phase_bins=n_phase_bins,
            min_valid_fraction=min_valid_fraction,
        )

    return CellInspection(
        time_index=time_index,
        z_index=z_index,
        y_index=y_index,
        x_index=x_index,
        time=float(flow._file["t"][time_index]),
        z=float(flow._file["z"][z_index]),
        y=float(flow._file["y"][y_index]),
        x=float(flow._file["x"][x_index]),
        raw_u=raw_u,
        raw_v=raw_v,
        raw_w=raw_w,
        raw_speed=float(np.sqrt(raw_u * raw_u + raw_v * raw_v + raw_w * raw_w)),
        computed_u=component_mean_ignoring_zero(
            flow._file["u"][:, z_index, y_index, x_index],
            min_valid_count=min_valid_count,
            invalid_samples=invalid_samples,
        ),
        computed_v=component_mean_ignoring_zero(
            flow._file["v"][:, z_index, y_index, x_index],
            min_valid_count=min_valid_count,
            invalid_samples=invalid_samples,
        ),
        computed_w=component_mean_ignoring_zero(
            flow._file["w"][:, z_index, y_index, x_index],
            min_valid_count=min_valid_count,
            invalid_samples=invalid_samples,
        ),
        average_u=average_u,
        average_v=average_v,
        average_w=average_w,
        interpolated_u=interpolated_u,
        interpolated_v=interpolated_v,
        interpolated_w=interpolated_w,
        interpolated_speed=interpolated_speed,
        filled_u=filled_u,
        filled_v=filled_v,
        filled_w=filled_w,
        phase_coverage=phase_coverage,
    )


def _format_mean(name: str, value: ComponentMean) -> str:
    status = "accepted" if value.accepted else "rejected"
    return f"  {name}_mean={value.mean:.9g}  count={value.count}  {status}"


def _format_delta(interpolated: float, raw: float) -> str:
    if not np.isfinite(raw):
        return "n/a, raw missing"
    return f"{interpolated - raw:.9g}"


def format_cell_inspection(
    cell: CellInspection,
    raw_label: str = "raw file",
    min_valid_count: int = 1,
    n_times: int | None = None,
    volume_coverage: CoverageStats | None = None,
    plane_coverage: CoverageStats | None = None,
    coverage_source: str = "average file counts",
) -> str:
    lines = [
        "Inspector mode",
        f"  raw source: {raw_label}",
        f"  min valid count: {min_valid_count}"
        + (f" / {n_times}" if n_times is not None else ""),
        "",
        "Selected cell",
        f"  indices: t={cell.time_index}, z={cell.z_index}, "
        f"y={cell.y_index}, x={cell.x_index}",
        f"  coords:  t={cell.time:.6g}, z={cell.z:.6g}, "
        f"y={cell.y:.6g}, x={cell.x:.6g}",
        "",
        "Raw value at selected frame",
        f"  u={cell.raw_u:.9g}",
        f"  v={cell.raw_v:.9g}",
        f"  w={cell.raw_w:.9g}",
        f"  speed={cell.raw_speed:.9g}",
        "",
        "Selected-voxel temporal mean computed on demand",
        "  source: raw time series at this one cell only",
        _format_mean("u", cell.computed_u),
        _format_mean("v", cell.computed_v),
        _format_mean("w", cell.computed_w),
    ]
    if cell.average_u is not None and cell.average_v is not None and cell.average_w is not None:
        lines.extend(
            [
                "",
                "Value stored in average file",
                _format_mean("u", cell.average_u),
                _format_mean("v", cell.average_v),
                _format_mean("w", cell.average_w),
            ]
        )
        lines.extend(["", "Average-file coverage", f"  source: {coverage_source}"])
        if volume_coverage is None:
            lines.append("  empty volume: unavailable; count datasets missing")
        else:
            lines.append(_format_coverage("empty volume", volume_coverage))
        if plane_coverage is not None:
            lines.append(_format_coverage("empty shown z plane", plane_coverage))
    else:
        lines.extend(
            [
                "",
                "Average file comparison",
                "  not active",
            ]
        )
    if cell.phase_coverage is not None:
        lines.extend(_format_phase_coverage(cell.phase_coverage))
    else:
        lines.extend(
            [
                "",
                "Phase coverage",
                "  not active",
            ]
        )
    if (
        cell.interpolated_u is not None
        and cell.interpolated_v is not None
        and cell.interpolated_w is not None
        and cell.interpolated_speed is not None
    ):
        lines.extend(
            [
                "",
                "Value stored in interpolated file",
                f"  u={cell.interpolated_u:.9g}  "
                f"delta={_format_delta(cell.interpolated_u, cell.raw_u)}  "
                f"filled={cell.filled_u}",
                f"  v={cell.interpolated_v:.9g}  "
                f"delta={_format_delta(cell.interpolated_v, cell.raw_v)}  "
                f"filled={cell.filled_v}",
                f"  w={cell.interpolated_w:.9g}  "
                f"delta={_format_delta(cell.interpolated_w, cell.raw_w)}  "
                f"filled={cell.filled_w}",
                f"  speed={cell.interpolated_speed:.9g}  "
                f"delta={_format_delta(cell.interpolated_speed, cell.raw_speed)}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Interpolated file comparison",
                "  not active",
            ]
        )
    return "\n".join(lines)


def inspect_flow_gui(
    flow: FlowDataset,
    average: TemporalAverageVolume | None = None,
    interpolated: FlowDataset | None = None,
    initial_frame: int = 0,
    initial_z: float = 0.0,
    quiver_step: int = 3,
    min_valid_fraction: float = 0.0,
    invalid_samples: str = "nan",
    phase_frequency_hz: float | None = None,
    phase_signal: str | None = None,
    phase_offset: float = 0.0,
    n_phase_bins: int = 16,
) -> None:
    """Interactive visual inspection of raw values and temporal means."""

    if average is not None:
        validate_average_compatible(flow, average)
    if interpolated is not None:
        validate_interpolated_compatible(flow, interpolated)
    raw_label = flow.path.name
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    frame_index = int(np.clip(initial_frame, 0, flow.n_times - 1))
    z_index = flow.nearest_z_index(initial_z)
    x_values = flow.coordinate("x")
    y_values = flow.coordinate("y")
    selected_y_index = len(y_values) // 2
    selected_x_index = len(x_values) // 2

    first = (interpolated or flow).read_z_plane(frame_index, z_index)
    fig = plt.figure(figsize=(13, 8), constrained_layout=False)
    ax = fig.add_axes((0.06, 0.20, 0.58, 0.72))
    text_ax = fig.add_axes((0.68, 0.20, 0.29, 0.72))
    prev_frame_ax = fig.add_axes((0.10, 0.112, 0.055, 0.04))
    frame_ax = fig.add_axes((0.18, 0.115, 0.39, 0.03))
    next_frame_ax = fig.add_axes((0.585, 0.112, 0.055, 0.04))
    valid_ax = fig.add_axes((0.18, 0.07, 0.39, 0.03))

    image, quiver, q_slice = _draw_xy_vector_plane(
        ax=ax,
        horizontal=first.x,
        vertical=first.y,
        scalar=first.speed,
        vector_horizontal=first.u,
        vector_vertical=first.v,
        quiver_step=quiver_step,
        horizontal_label="x",
        vertical_label="y",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Velocity magnitude")
    rejected_overlay = ax.imshow(
        _rejected_overlay_rgba(np.ones_like(first.speed, dtype=bool)),
        extent=image.get_extent(),
        origin="lower",
        interpolation="nearest",
        zorder=image.get_zorder() + 0.5,
    )
    image.set_alpha(np.ones_like(first.speed, dtype=float))
    marker = ax.plot(
        [x_values[selected_x_index]],
        [y_values[selected_y_index]],
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2,
    )[0]
    title = ax.set_title("")

    text_ax.axis("off")
    info = text_ax.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        transform=text_ax.transAxes,
    )

    frame_slider = Slider(
        frame_ax,
        "Frame",
        0,
        flow.n_times - 1,
        valinit=frame_index,
        valstep=1,
    )
    previous_frame_button = Button(prev_frame_ax, "<")
    next_frame_button = Button(next_frame_ax, ">")
    valid_slider = Slider(
        valid_ax,
        "min valid frac",
        0.0,
        1.0,
        valinit=min_valid_fraction,
        valstep=0.01,
    )

    def current_min_valid_count() -> int:
        return max(int(np.ceil(float(valid_slider.val) * flow.n_times)), 1)

    counts_for_fixed_z = _valid_counts_for_z_plane(
        flow, z_index, invalid_samples=invalid_samples
    )
    phases = phase_values_for_flow(
        flow,
        frequency_hz=phase_frequency_hz,
        phase_signal=phase_signal,
        phase_offset=phase_offset,
    )
    average_count_volumes = (
        _average_count_volumes(average) if average is not None else None
    )

    def refresh() -> None:
        nonlocal frame_index
        frame_index = int(frame_slider.val)
        min_valid_count = current_min_valid_count()
        display_flow = interpolated or flow
        plane = display_flow.read_z_plane(frame_index, z_index)
        display_speed, display_u, display_v, accepted = _apply_inspector_valid_mask(
            plane.speed,
            plane.u,
            plane.v,
            counts_for_fixed_z,
            min_valid_count,
        )
        volume_coverage = None
        plane_coverage = None
        if average_count_volumes is not None:
            volume_coverage = _coverage_from_counts(
                average_count_volumes,
                min_valid_count,
            )
            plane_coverage = _coverage_from_counts(
                tuple(count[z_index, :, :] for count in average_count_volumes),
                min_valid_count,
            )
        image.set_data(display_speed)
        rejected_overlay.set_data(_rejected_overlay_rgba(accepted))
        if interpolated is not None:
            _, _, _, any_filled = _interpolated_filled_plane(
                interpolated,
                frame_index,
                z_index,
            )
            image.set_alpha(_filled_image_alpha(any_filled))
        else:
            image.set_alpha(np.ones_like(accepted, dtype=float))
        quiver.set_UVC(display_u[q_slice], display_v[q_slice])
        marker.set_data([x_values[selected_x_index]], [y_values[selected_y_index]])
        title.set_text(
            f"{'Interpolated' if interpolated is not None else 'Raw'} frame={frame_index}, "
            f"t={plane.time:.6g}, "
            f"z={plane.z_value:.6g} (z_index={z_index}), "
            f"selected y/x={selected_y_index}/{selected_x_index}, "
            f"shown accepted={int(accepted.sum())}/{accepted.size}"
            + (
                f", avg empty volume={100.0 * volume_coverage.rejected_fraction:.1f}%"
                if volume_coverage is not None
                else ""
            )
            + (
                f", transparent filled cells={int(any_filled.sum())}/{any_filled.size}"
                if interpolated is not None
                else ""
            )
        )
        cell = inspect_cell(
            flow,
            time_index=frame_index,
            z_index=z_index,
            y_index=selected_y_index,
            x_index=selected_x_index,
            average=average,
            interpolated=interpolated,
            min_valid_count=min_valid_count,
            invalid_samples=invalid_samples,
            phases=phases,
            n_phase_bins=n_phase_bins,
            min_valid_fraction=float(valid_slider.val),
        )
        info.set_text(
            format_cell_inspection(
                cell,
                raw_label=raw_label,
                min_valid_count=min_valid_count,
                n_times=flow.n_times,
                volume_coverage=volume_coverage,
                plane_coverage=plane_coverage,
            )
        )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal selected_y_index, selected_x_index
        if event.inaxes is not ax and not ax.bbox.contains(event.x, event.y):
            return
        if event.xdata is None or event.ydata is None:
            x_data, y_data = ax.transData.inverted().transform((event.x, event.y))
        else:
            x_data, y_data = event.xdata, event.ydata
        if not (
            np.nanmin(x_values) <= x_data <= np.nanmax(x_values)
            and np.nanmin(y_values) <= y_data <= np.nanmax(y_values)
        ):
            return
        next_x_index = nearest_index(x_values, x_data)
        next_y_index = nearest_index(y_values, y_data)
        if (
            next_x_index == selected_x_index
            and next_y_index == selected_y_index
        ):
            return
        selected_x_index = next_x_index
        selected_y_index = next_y_index
        refresh()

    def step_frame(offset: int) -> None:
        current = int(frame_slider.val)
        next_frame = step_index(current, offset, flow.n_times - 1)
        if next_frame != current:
            frame_slider.set_val(next_frame)

    frame_slider.on_changed(lambda _value: refresh())
    valid_slider.on_changed(lambda _value: refresh())
    previous_frame_button.on_clicked(lambda _event: step_frame(-1))
    next_frame_button.on_clicked(lambda _event: step_frame(1))
    fig.canvas.mpl_connect("button_press_event", on_click)

    print(f"Reading NetCDF file: {flow.path.resolve()}")
    if average is not None:
        print(f"Reading average file: {average.path.resolve()}")
    if interpolated is not None:
        print(f"Reading interpolated file: {interpolated.path.resolve()}")
    else:
        print("No interpolated file comparison active.")
    if average is None:
        print("Raw-only average inspection: no average file comparison active.")
    print(
        f"Inspector min_valid_fraction={min_valid_fraction:g} "
        f"(initial min_count={current_min_valid_count()})."
    )
    if phases is not None:
        print(
            f"Phase coverage active with n_phase_bins={n_phase_bins}.",
            flush=True,
        )
    print("Click a cell to inspect raw values and selected-voxel means.")
    refresh()
    plt.show()
