from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from ptv_flow.reader import FlowDataset


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
    x_grid, y_grid = np.meshgrid(first.x, first.y)
    q_slice = (slice(None, None, quiver_step), slice(None, None, quiver_step))
    interval_ms = 1000.0 / fps

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(
        first.speed,
        extent=(first.x.min(), first.x.max(), first.y.min(), first.y.max()),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        animated=True,
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Velocity magnitude")

    quiver = ax.quiver(
        x_grid[q_slice],
        y_grid[q_slice],
        first.u[q_slice],
        first.v[q_slice],
        color="white",
        pivot="mid",
        scale=None,
        width=0.003,
        animated=True,
    )
    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

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
