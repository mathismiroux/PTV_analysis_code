from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from ptv_flow.postprocess import TemporalAverageVolume
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
