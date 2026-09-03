from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np

from ptv_flow.postprocess import PhaseAverageVolume, TemporalAverageVolume
from ptv_flow.reader import FlowDataset


def _draw_xy_vector_plane(
    ax: plt.Axes,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    scalar: np.ndarray,
    vector_horizontal: np.ndarray,
    vector_vertical: np.ndarray,
    quiver_step: int,
    animated: bool = False,
    cmap: str = "viridis",
    horizontal_label: str = "x",
    vertical_label: str = "y",
):
    if quiver_step <= 0:
        raise ValueError("quiver_step must be positive")

    x_grid, y_grid = np.meshgrid(horizontal, vertical)
    q_slice = (slice(None, None, quiver_step), slice(None, None, quiver_step))
    image = ax.imshow(
        scalar,
        extent=(
            horizontal.min(),
            horizontal.max(),
            vertical.min(),
            vertical.max(),
        ),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        animated=animated,
    )
    quiver = ax.quiver(
        x_grid[q_slice],
        y_grid[q_slice],
        vector_horizontal[q_slice],
        vector_vertical[q_slice],
        color="white",
        pivot="mid",
        scale=None,
        width=0.003,
        animated=animated,
    )
    ax.set_xlabel(horizontal_label)
    ax.set_ylabel(vertical_label)
    ax.set_aspect("equal", adjustable="box")
    return image, quiver, q_slice


def _temporal_average_quantity(
    plane: dict[str, np.ndarray | float | int], quantity: str
) -> tuple[np.ndarray, str, str]:
    if quantity == "speed":
        return plane["speed"], "Mean 3D velocity magnitude", "viridis"
    if quantity == "u":
        return plane["u"], "Mean u velocity", "coolwarm"
    if quantity == "v":
        return plane["v"], "Mean v velocity", "coolwarm"
    if quantity == "w":
        return plane["w"], "Mean w velocity", "coolwarm"
    raise ValueError("quantity must be one of 'speed', 'u', 'v', or 'w'")


def _phase_average_quantity(
    plane: dict[str, np.ndarray | float | int | str],
    quantity: str,
) -> tuple[np.ndarray, str, str]:
    field = str(plane.get("field", "phase_mean"))
    prefix = "Phase-mean" if field == "phase_mean" else "Coherent"
    if quantity == "speed":
        return plane["speed"], f"{prefix} 3D velocity magnitude", "viridis"
    if quantity == "u":
        return plane["u"], f"{prefix} u velocity", "coolwarm"
    if quantity == "v":
        return plane["v"], f"{prefix} v velocity", "coolwarm"
    if quantity == "w":
        return plane["w"], f"{prefix} w velocity", "coolwarm"
    raise ValueError("quantity must be one of 'speed', 'u', 'v', or 'w'")


def _min_valid_count(volume: TemporalAverageVolume, min_valid_fraction: float) -> int:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    if min_valid_fraction == 0.0:
        return 1
    if "input_shape_time_z_y_x" not in volume._file.attrs:
        raise KeyError(
            "Missing input_shape_time_z_y_x metadata; cannot apply "
            "min_valid_fraction while visualizing."
        )
    n_times = int(volume._file.attrs["input_shape_time_z_y_x"][0])
    return int(np.ceil(min_valid_fraction * n_times))


def _apply_plane_valid_fraction(
    plane: dict[str, np.ndarray | float | int],
    scalar: np.ndarray,
    quantity: str,
    min_valid_count: int,
) -> np.ndarray:
    if min_valid_count <= 1:
        return scalar

    scalar = scalar.astype(float, copy=True)
    if quantity == "speed":
        counts = [plane.get(f"{name}_count") for name in ("u", "v", "w")]
        if all(count is not None for count in counts):
            valid = np.ones(scalar.shape, dtype=bool)
            for count in counts:
                valid &= count >= min_valid_count
            scalar[~valid] = np.nan
        return scalar

    count = plane.get(f"{quantity}_count")
    if count is not None:
        scalar[count < min_valid_count] = np.nan
    return scalar


def animate_z_plane(
    flow: FlowDataset,
    z_value: float = 0.0,
    fps: float = 40.0,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    quiver_step: int = 3,
    save: Path | None = None,
) -> None:
    """Animate velocity magnitude and in-plane vectors at the nearest z plane."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if quiver_step <= 0:
        raise ValueError("quiver_step must be positive")

    z_index = flow.nearest_z_index(z_value)
    stop = flow.n_times if stop is None else min(stop, flow.n_times)
    frame_indices = list(range(start, stop, step))
    if not frame_indices:
        raise ValueError("No frames selected for animation")

    first = flow.read_z_plane(frame_indices[0], z_index)
    interval_ms = 1000.0 / fps

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image, quiver, q_slice = _draw_xy_vector_plane(
        ax=ax,
        horizontal=first.x,
        vertical=first.y,
        scalar=first.speed,
        vector_horizontal=first.u,
        vector_vertical=first.v,
        quiver_step=quiver_step,
        animated=True,
        horizontal_label="x",
        vertical_label="y",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Velocity magnitude")

    title = ax.set_title("")

    def update(frame_index: int):
        plane = flow.read_z_plane(frame_index, z_index)
        image.set_data(plane.speed)
        quiver.set_UVC(plane.u[q_slice], plane.v[q_slice])
        title.set_text(
            f"z={plane.z_value:.6g} (nearest to {z_value:.6g}), "
            f"frame={plane.time_index}, t={plane.time:.6g}"
        )
        return image, quiver, title

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )

    if save is not None:
        if save.suffix.lower() != ".gif":
            raise ValueError(
                "Only .gif saving is available by default. Use --save output.gif, "
                "or install ffmpeg and extend the writer configuration for MP4."
            )
        save.parent.mkdir(parents=True, exist_ok=True)
        print(f"Reading NetCDF file: {flow.path.resolve()}")
        animation.save(save, writer="pillow", fps=fps)
        print(f"Saved animation to {save}")
    else:
        print(
            f"Reading NetCDF file: {flow.path.resolve()}\n"
            f"Animating {len(frame_indices)} frames at {fps:g} fps on "
            f"z={first.z_value:.6g} (nearest to requested z={z_value:g})."
        )
        plt.show()


def show_temporal_average_plane(
    volume: TemporalAverageVolume,
    plane_axis: str = "z",
    plane_value: float = 0.0,
    quantity: str = "speed",
    quiver_step: int = 3,
    save: Path | None = None,
    min_valid_fraction: float = 0.0,
) -> None:
    """Show one x, y, or z plane from a temporal-average volume."""

    plane_index = volume.nearest_index(plane_axis, plane_value)
    plane = volume.read_plane(plane_axis, plane_index)
    scalar, colorbar_label, cmap = _temporal_average_quantity(plane, quantity)
    min_valid_count = _min_valid_count(volume, min_valid_fraction)
    scalar = _apply_plane_valid_fraction(
        plane,
        scalar,
        quantity=quantity,
        min_valid_count=min_valid_count,
    )

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image, _, _ = _draw_xy_vector_plane(
        ax=ax,
        horizontal=plane["horizontal"],
        vertical=plane["vertical"],
        scalar=scalar,
        vector_horizontal=plane["vector_horizontal"],
        vector_vertical=plane["vector_vertical"],
        quiver_step=quiver_step,
        cmap=cmap,
        horizontal_label=plane["horizontal_axis"],
        vertical_label=plane["vertical_axis"],
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    if quantity in {"u", "v", "w"}:
        finite = scalar[np.isfinite(scalar)]
        if finite.size:
            limit = float(np.nanmax(np.abs(finite)))
            if limit > 0:
                image.set_clim(-limit, limit)
    ax.set_title(
        f"Temporal mean {quantity}, {plane_axis}={plane['value']:.6g} "
        f"(nearest to {plane_value:.6g})"
    )

    print(f"Reading postprocessed file: {volume.path.resolve()}")
    print(
        f"Showing quantity={quantity}, {plane_axis}={plane['value']:.6g} "
        f"({plane_axis}_index={plane['index']}, "
        f"nearest to requested {plane_axis}={plane_value:g})."
    )
    if min_valid_fraction > 0.0:
        print(
            f"Display mask: min_valid_fraction={min_valid_fraction:g} "
            f"(min_count={min_valid_count})."
        )

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"Saved plane visualization to {save}")
    else:
        plt.show()


def _phase_plane_min_valid_counts(
    volume: PhaseAverageVolume,
    min_valid_fraction: float,
) -> np.ndarray:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")
    counts = volume._file["phase_sample_count"][:].astype(np.float64)
    return np.maximum(np.ceil(min_valid_fraction * counts), 1).astype(np.uint32)


def _apply_phase_plane_valid_fraction(
    plane: dict[str, np.ndarray | float | int | str],
    scalar: np.ndarray,
    quantity: str,
    min_valid_count: int,
) -> np.ndarray:
    if min_valid_count <= 1:
        return scalar.copy()
    if quantity in {"u", "v", "w"}:
        count = plane[f"{quantity}_count"]
        return np.where(count >= min_valid_count, scalar, np.nan)
    accepted = (
        (plane["u_count"] >= min_valid_count)
        & (plane["v_count"] >= min_valid_count)
        & (plane["w_count"] >= min_valid_count)
    )
    return np.where(accepted, scalar, np.nan)


def _phase_count_stack_for_plane(
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


def _apply_harmonic_valid_fraction(
    volume: PhaseAverageVolume,
    scalar: np.ndarray,
    plane_axis: str,
    plane_index: int,
    component: str,
    min_valid_fraction: float,
) -> tuple[np.ndarray, int]:
    min_counts = _phase_plane_min_valid_counts(volume, min_valid_fraction)
    counts = _phase_count_stack_for_plane(
        volume,
        plane_axis=plane_axis,
        plane_index=plane_index,
        component=component,
    )
    accepted = np.all(counts >= min_counts[:, None, None], axis=0)
    masked = np.where(accepted, scalar, np.nan)
    return masked, int(np.count_nonzero(accepted))


def show_phase_voxel_series(
    volume: PhaseAverageVolume,
    x_value: float,
    y_value: float,
    z_value: float,
    field: str = "phase_mean",
    min_valid_fraction: float = 0.0,
    save: Path | None = None,
) -> None:
    """Plot phase-locked velocity components at the nearest stored voxel."""

    x_index = volume.nearest_index("x", x_value)
    y_index = volume.nearest_index("y", y_value)
    z_index = volume.nearest_index("z", z_value)
    series = volume.read_phase_series_at(
        z_index=z_index,
        y_index=y_index,
        x_index=x_index,
        field=field,
    )
    min_counts = _phase_plane_min_valid_counts(volume, min_valid_fraction)
    phase_degrees = np.asarray(series["phase_degrees"], dtype=np.float64)
    phase_closed = np.r_[phase_degrees, phase_degrees[0] + 360.0]

    component_styles = {
        "u": ("#1f77b4", "o"),
        "v": ("#d62728", "s"),
        "w": ("#2ca02c", "^"),
    }
    fig, (ax_velocity, ax_counts) = plt.subplots(
        2,
        1,
        figsize=(9.5, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )

    accepted_by_component: dict[str, np.ndarray] = {}
    for component, (color, marker) in component_styles.items():
        values = np.asarray(series[component], dtype=np.float64)
        counts = np.asarray(series[f"{component}_count"], dtype=np.uint32)
        accepted = counts >= min_counts
        accepted_by_component[component] = accepted
        plotted = np.where(accepted, values, np.nan)
        plotted_closed = np.r_[plotted, plotted[0]]
        ax_velocity.plot(
            phase_closed,
            plotted_closed,
            color=color,
            marker=marker,
            linewidth=1.7,
            label=component,
        )
        rejected = ~accepted & np.isfinite(values)
        if rejected.any():
            ax_velocity.scatter(
                phase_degrees[rejected],
                values[rejected],
                color=color,
                marker="x",
                s=42,
                alpha=0.9,
            )

    count_width = 360.0 / max(volume.n_phase_bins, 1) / 4.5
    offsets = {"u": -count_width, "v": 0.0, "w": count_width}
    for component, (color, _) in component_styles.items():
        counts = np.asarray(series[f"{component}_count"], dtype=np.uint32)
        ax_counts.bar(
            phase_degrees + offsets[component],
            counts,
            width=count_width,
            color=color,
            alpha=0.72,
            label=f"{component} count",
        )
    ax_counts.step(
        phase_closed,
        np.r_[min_counts, min_counts[0]],
        where="mid",
        color="black",
        linewidth=1.1,
        label="min count",
    )

    title_prefix = "Phase-mean" if field == "phase_mean" else "Coherent"
    ax_velocity.set_title(
        f"{title_prefix} velocity at "
        f"x={series['x']:.6g}, y={series['y']:.6g}, z={series['z']:.6g}"
    )
    ax_velocity.set_ylabel("velocity")
    ax_velocity.grid(True, color="0.82", linewidth=0.6)
    ax_velocity.legend(loc="best", ncol=3)
    ax_counts.set_xlabel("phase [deg]")
    ax_counts.set_ylabel("valid samples")
    ax_counts.set_xlim(float(phase_degrees[0]), float(phase_degrees[0]) + 360.0)
    ax_counts.grid(True, axis="y", color="0.82", linewidth=0.6)
    ax_counts.legend(loc="best", ncol=4)

    print(f"Reading phase-average file: {volume.path.resolve()}")
    print(
        f"Showing {field} voxel phase series at nearest voxel "
        f"x={series['x']:.6g} (index {x_index}), "
        f"y={series['y']:.6g} (index {y_index}), "
        f"z={series['z']:.6g} (index {z_index})."
    )
    if min_valid_fraction > 0.0:
        summary = ", ".join(
            f"{component}: {int(mask.sum())}/{mask.size} bins"
            for component, mask in accepted_by_component.items()
        )
        print(f"Accepted bins with min_valid_fraction={min_valid_fraction:g}: {summary}.")
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"Saved phase voxel series to {save}")
    else:
        plt.show()


def show_phase_average_plane_gui(
    volume: PhaseAverageVolume,
    plane_axis: str = "z",
    plane_value: float = 0.0,
    quantity: str = "speed",
    field: str = "phase_mean",
    quiver_step: int = 3,
    min_valid_fraction: float = 0.0,
) -> None:
    """Interactively show phase-averaged planes, one phase bin at a time."""

    plane_index = volume.nearest_index(plane_axis, plane_value)
    phase_index = 0
    min_valid_counts = _phase_plane_min_valid_counts(volume, min_valid_fraction)
    first = volume.read_plane(
        phase_index=phase_index,
        axis=plane_axis,
        index=plane_index,
        field=field,
    )
    scalar, colorbar_label, cmap = _phase_average_quantity(first, quantity)
    scalar = _apply_phase_plane_valid_fraction(
        first,
        scalar,
        quantity=quantity,
        min_valid_count=int(min_valid_counts[phase_index]),
    )

    fig = plt.figure(figsize=(11, 8), constrained_layout=False)
    ax = fig.add_axes((0.08, 0.20, 0.78, 0.72))
    prev_ax = fig.add_axes((0.17, 0.105, 0.06, 0.04))
    slider_ax = fig.add_axes((0.26, 0.112, 0.43, 0.03))
    next_ax = fig.add_axes((0.72, 0.105, 0.06, 0.04))

    image, quiver, q_slice = _draw_xy_vector_plane(
        ax=ax,
        horizontal=first["horizontal"],
        vertical=first["vertical"],
        scalar=scalar,
        vector_horizontal=first["vector_horizontal"],
        vector_vertical=first["vector_vertical"],
        quiver_step=quiver_step,
        cmap=cmap,
        horizontal_label=first["horizontal_axis"],
        vertical_label=first["vertical_axis"],
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    title = ax.set_title("")
    phase_slider = Slider(
        slider_ax,
        "Phase bin",
        0,
        volume.n_phase_bins - 1,
        valinit=phase_index,
        valstep=1,
    )
    previous_button = Button(prev_ax, "<")
    next_button = Button(next_ax, ">")

    def set_symmetric_clim(values: np.ndarray) -> None:
        if quantity not in {"u", "v", "w"}:
            return
        finite = values[np.isfinite(values)]
        if finite.size:
            limit = float(np.nanmax(np.abs(finite)))
            if limit > 0.0:
                image.set_clim(-limit, limit)

    set_symmetric_clim(scalar)

    def refresh() -> None:
        current_phase = int(phase_slider.val)
        plane = volume.read_plane(
            phase_index=current_phase,
            axis=plane_axis,
            index=plane_index,
            field=field,
        )
        scalar_data, _, _ = _phase_average_quantity(plane, quantity)
        display_scalar = _apply_phase_plane_valid_fraction(
            plane,
            scalar_data,
            quantity=quantity,
            min_valid_count=int(min_valid_counts[current_phase]),
        )
        image.set_data(display_scalar)
        quiver.set_UVC(
            plane["vector_horizontal"][q_slice],
            plane["vector_vertical"][q_slice],
        )
        set_symmetric_clim(display_scalar)
        title.set_text(
            f"{field} {quantity}, phase={plane['phase_degrees']:.1f} deg "
            f"(bin {current_phase}/{volume.n_phase_bins - 1}), "
            f"{plane_axis}={plane['value']:.6g}, "
            f"min_count={int(min_valid_counts[current_phase])}"
        )
        fig.canvas.draw_idle()

    def step_phase(offset: int) -> None:
        current = int(phase_slider.val)
        next_phase = int(np.clip(current + offset, 0, volume.n_phase_bins - 1))
        if next_phase != current:
            phase_slider.set_val(next_phase)

    phase_slider.on_changed(lambda _value: refresh())
    previous_button.on_clicked(lambda _event: step_phase(-1))
    next_button.on_clicked(lambda _event: step_phase(1))

    print(f"Reading phase-average file: {volume.path.resolve()}")
    print(
        f"Showing field={field}, quantity={quantity}, "
        f"{plane_axis}={first['value']:.6g} "
        f"({plane_axis}_index={first['index']}, nearest to requested "
        f"{plane_axis}={plane_value:g}), n_phase_bins={volume.n_phase_bins}."
    )
    if min_valid_fraction > 0.0:
        print(f"Display mask: min_valid_fraction={min_valid_fraction:g}.")
    refresh()
    plt.show()


def show_harmonic_plane(
    volume: PhaseAverageVolume,
    plane_axis: str = "z",
    plane_value: float = 0.0,
    component: str = "u",
    harmonic_quantity: str = "amplitude",
    save: Path | None = None,
    min_valid_fraction: float = 0.0,
) -> None:
    """Show one harmonic-fit map from a phase-average volume."""

    plane_index = volume.nearest_index(plane_axis, plane_value)
    plane = volume.read_harmonic_plane(
        axis=plane_axis,
        index=plane_index,
        component=component,
        quantity=harmonic_quantity,
    )
    data = plane["data"]
    if harmonic_quantity == "phase":
        scalar = np.degrees(data)
        colorbar_label = f"{component} harmonic phase [deg]"
        cmap = "twilight"
        vmin, vmax = -180.0, 180.0
    elif harmonic_quantity == "amplitude":
        scalar = data
        colorbar_label = f"{component} harmonic amplitude"
        cmap = "magma"
        vmin = vmax = None
    else:
        scalar = data
        colorbar_label = f"{component} harmonic {harmonic_quantity}"
        cmap = "coolwarm"
        finite = scalar[np.isfinite(scalar)]
        if finite.size:
            limit = float(np.nanmax(np.abs(finite)))
            vmin, vmax = (-limit, limit) if limit > 0.0 else (None, None)
        else:
            vmin = vmax = None

    accepted_cells = None
    if min_valid_fraction > 0.0:
        scalar, accepted_cells = _apply_harmonic_valid_fraction(
            volume,
            scalar,
            plane_axis=plane_axis,
            plane_index=plane_index,
            component=component,
            min_valid_fraction=min_valid_fraction,
        )

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(
        scalar,
        extent=(
            plane["horizontal"].min(),
            plane["horizontal"].max(),
            plane["vertical"].min(),
            plane["vertical"].max(),
        ),
        origin="lower",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xlabel(str(plane["horizontal_axis"]))
    ax.set_ylabel(str(plane["vertical_axis"]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{plane['dataset']}, {plane_axis}={plane['value']:.6g} "
        f"(nearest to {plane_value:.6g})"
    )

    print(f"Reading phase-average file: {volume.path.resolve()}")
    print(
        f"Showing harmonic {harmonic_quantity} for component={component}, "
        f"{plane_axis}={plane['value']:.6g} "
        f"({plane_axis}_index={plane['index']}, nearest to requested "
        f"{plane_axis}={plane_value:g})."
    )
    if accepted_cells is not None:
        print(
            f"Display mask: min_valid_fraction={min_valid_fraction:g}; "
            f"accepted {accepted_cells}/{scalar.size} cells with all phase bins "
            f"valid for {component}."
        )
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"Saved harmonic plane visualization to {save}")
    else:
        plt.show()


def show_temporal_average_z_plane(
    volume: TemporalAverageVolume,
    z_value: float = 0.0,
    quantity: str = "speed",
    quiver_step: int = 3,
    save: Path | None = None,
    min_valid_fraction: float = 0.0,
) -> None:
    """Backward-compatible wrapper for z-plane visualization."""

    show_temporal_average_plane(
        volume=volume,
        plane_axis="z",
        plane_value=z_value,
        quantity=quantity,
        quiver_step=quiver_step,
        save=save,
        min_valid_fraction=min_valid_fraction,
    )
