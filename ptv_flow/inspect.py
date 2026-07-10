from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from ptv_flow.postprocess import TemporalAverageVolume
from ptv_flow.reader import FlowDataset
from ptv_flow.visualize import _draw_xy_vector_plane


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


def nearest_index(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - value)))


def component_mean_ignoring_zero(
    series: np.ndarray, min_valid_count: int = 1
) -> ComponentMean:
    valid = series != 0.0
    count = int(valid.sum())
    if count == 0 or count < min_valid_count:
        return ComponentMean(mean=float("nan"), count=count, accepted=False)
    return ComponentMean(mean=float(series[valid].mean()), count=count, accepted=True)


def _valid_counts_for_z_plane(
    flow: FlowDataset, z_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        (flow._file["u"][:, z_index, :, :] != 0.0).sum(axis=0),
        (flow._file["v"][:, z_index, :, :] != 0.0).sum(axis=0),
        (flow._file["w"][:, z_index, :, :] != 0.0).sum(axis=0),
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


def inspect_cell(
    flow: FlowDataset,
    time_index: int,
    z_index: int,
    y_index: int,
    x_index: int,
    average: TemporalAverageVolume | None = None,
    min_valid_count: int = 1,
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

    average_u = None
    average_v = None
    average_w = None
    if average is not None:
        average_u = ComponentMean(
            mean=(
                float(average._file["u_mean"][z_index, y_index, x_index])
                if "u_count" not in average._file
                or int(average._file["u_count"][z_index, y_index, x_index])
                >= min_valid_count
                else float("nan")
            ),
            count=(
                int(average._file["u_count"][z_index, y_index, x_index])
                if "u_count" in average._file
                else -1
            ),
            accepted=(
                "u_count" not in average._file
                or int(average._file["u_count"][z_index, y_index, x_index])
                >= min_valid_count
            ),
        )
        average_v = ComponentMean(
            mean=(
                float(average._file["v_mean"][z_index, y_index, x_index])
                if "v_count" not in average._file
                or int(average._file["v_count"][z_index, y_index, x_index])
                >= min_valid_count
                else float("nan")
            ),
            count=(
                int(average._file["v_count"][z_index, y_index, x_index])
                if "v_count" in average._file
                else -1
            ),
            accepted=(
                "v_count" not in average._file
                or int(average._file["v_count"][z_index, y_index, x_index])
                >= min_valid_count
            ),
        )
        average_w = ComponentMean(
            mean=(
                float(average._file["w_mean"][z_index, y_index, x_index])
                if "w_count" not in average._file
                or int(average._file["w_count"][z_index, y_index, x_index])
                >= min_valid_count
                else float("nan")
            ),
            count=(
                int(average._file["w_count"][z_index, y_index, x_index])
                if "w_count" in average._file
                else -1
            ),
            accepted=(
                "w_count" not in average._file
                or int(average._file["w_count"][z_index, y_index, x_index])
                >= min_valid_count
            ),
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
        ),
        computed_v=component_mean_ignoring_zero(
            flow._file["v"][:, z_index, y_index, x_index],
            min_valid_count=min_valid_count,
        ),
        computed_w=component_mean_ignoring_zero(
            flow._file["w"][:, z_index, y_index, x_index],
            min_valid_count=min_valid_count,
        ),
        average_u=average_u,
        average_v=average_v,
        average_w=average_w,
    )


def _format_mean(name: str, value: ComponentMean) -> str:
    status = "accepted" if value.accepted else "rejected"
    return f"  {name}_mean={value.mean:.9g}  count={value.count}  {status}"


def format_cell_inspection(
    cell: CellInspection,
    raw_label: str = "raw file",
    min_valid_count: int = 1,
    n_times: int | None = None,
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
    else:
        lines.extend(
            [
                "",
                "Average file comparison",
                "  not active",
            ]
        )
    return "\n".join(lines)


def inspect_flow_gui(
    flow: FlowDataset,
    average: TemporalAverageVolume | None = None,
    initial_frame: int = 0,
    initial_z: float = 0.0,
    quiver_step: int = 3,
    min_valid_fraction: float = 0.0,
) -> None:
    """Interactive visual inspection of raw values and temporal means."""

    if average is not None:
        validate_average_compatible(flow, average)
    raw_label = flow.path.name
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    frame_index = int(np.clip(initial_frame, 0, flow.n_times - 1))
    z_index = flow.nearest_z_index(initial_z)
    x_values = flow.coordinate("x")
    y_values = flow.coordinate("y")
    selected_y_index = len(y_values) // 2
    selected_x_index = len(x_values) // 2

    first = flow.read_z_plane(frame_index, z_index)
    fig = plt.figure(figsize=(13, 8), constrained_layout=False)
    ax = fig.add_axes((0.06, 0.20, 0.58, 0.72))
    text_ax = fig.add_axes((0.68, 0.20, 0.29, 0.72))
    frame_ax = fig.add_axes((0.18, 0.115, 0.39, 0.03))
    z_ax = fig.add_axes((0.18, 0.07, 0.39, 0.03))
    valid_ax = fig.add_axes((0.18, 0.025, 0.39, 0.03))

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
    z_slider = Slider(
        z_ax,
        "z index",
        0,
        flow.grid_shape[0] - 1,
        valinit=z_index,
        valstep=1,
    )
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

    counts_by_z_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def current_counts() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if z_index not in counts_by_z_index:
            counts_by_z_index[z_index] = _valid_counts_for_z_plane(flow, z_index)
        return counts_by_z_index[z_index]

    def refresh() -> None:
        nonlocal frame_index, z_index
        frame_index = int(frame_slider.val)
        z_index = int(z_slider.val)
        min_valid_count = current_min_valid_count()
        plane = flow.read_z_plane(frame_index, z_index)
        display_speed, display_u, display_v, accepted = _apply_inspector_valid_mask(
            plane.speed,
            plane.u,
            plane.v,
            current_counts(),
            min_valid_count,
        )
        image.set_data(display_speed)
        rejected_overlay.set_data(_rejected_overlay_rgba(accepted))
        quiver.set_UVC(display_u[q_slice], display_v[q_slice])
        marker.set_data([x_values[selected_x_index]], [y_values[selected_y_index]])
        title.set_text(
            f"Raw frame={frame_index}, t={plane.time:.6g}, "
            f"z={plane.z_value:.6g} (z_index={z_index}), "
            f"shown accepted={int(accepted.sum())}/{accepted.size}"
        )
        cell = inspect_cell(
            flow,
            time_index=frame_index,
            z_index=z_index,
            y_index=selected_y_index,
            x_index=selected_x_index,
            average=average,
            min_valid_count=min_valid_count,
        )
        info.set_text(
            format_cell_inspection(
                cell,
                raw_label=raw_label,
                min_valid_count=min_valid_count,
                n_times=flow.n_times,
            )
        )
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        nonlocal selected_y_index, selected_x_index
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        selected_x_index = nearest_index(x_values, event.xdata)
        selected_y_index = nearest_index(y_values, event.ydata)
        refresh()

    frame_slider.on_changed(lambda _value: refresh())
    z_slider.on_changed(lambda _value: refresh())
    valid_slider.on_changed(lambda _value: refresh())
    fig.canvas.mpl_connect("button_press_event", on_click)

    print(f"Reading NetCDF file: {flow.path.resolve()}")
    if average is not None:
        print(f"Reading average file: {average.path.resolve()}")
    else:
        print("Raw-only inspection: no average file comparison active.")
    print(
        f"Inspector min_valid_fraction={min_valid_fraction:g} "
        f"(initial min_count={current_min_valid_count()})."
    )
    print("Click a cell to inspect raw values and on-demand selected-voxel means.")
    refresh()
    plt.show()
