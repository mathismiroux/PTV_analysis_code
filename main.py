from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


DEFAULT_FILE = Path("Static_3.5D__b128f.nc")
VELOCITY_COMPONENTS = ("u", "v", "w")
COORDINATES = ("t", "z", "y", "x")


@dataclass(frozen=True)
class FlowFrame:
    """Velocity field at one time index."""

    time_index: int
    time: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v + self.w * self.w)


@dataclass(frozen=True)
class FlowPlane:
    """Velocity field on one z plane at one time index."""

    time_index: int
    time: float
    z_index: int
    z_value: float
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v + self.w * self.w)

    @property
    def in_plane_speed(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v)


class FlowDataset:
    """Lazy reader for the PTV NetCDF/HDF5 velocity time series.

    The velocity arrays in this file are stored as (time, z, y, x). Reading a
    single frame returns three arrays of shape (z, y, x), which is small enough
    to work with interactively while avoiding a full multi-GB load.
    """

    def __init__(self, path: str | Path = DEFAULT_FILE) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Could not find NetCDF file: {self.path}")

        self._file = h5py.File(self.path, "r")
        self._validate()

    def __enter__(self) -> "FlowDataset":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def _validate(self) -> None:
        missing = [
            name
            for name in (*COORDINATES, *VELOCITY_COMPONENTS)
            if name not in self._file
        ]
        if missing:
            raise KeyError(f"Missing expected variable(s): {', '.join(missing)}")

        shapes = {name: self._file[name].shape for name in VELOCITY_COMPONENTS}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"Velocity components have inconsistent shapes: {shapes}")

        expected_shape = (
            self._file["t"].shape[0],
            self._file["z"].shape[0],
            self._file["y"].shape[0],
            self._file["x"].shape[0],
        )
        if next(iter(shapes.values())) != expected_shape:
            raise ValueError(
                "Velocity shape does not match coordinate lengths: "
                f"{next(iter(shapes.values()))} != {expected_shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._file["u"].shape

    @property
    def dtype(self) -> np.dtype:
        return self._file["u"].dtype

    @property
    def n_times(self) -> int:
        return self.shape[0]

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self.shape[1:]

    def coordinate(self, name: str) -> np.ndarray:
        if name not in COORDINATES:
            raise ValueError(f"Unknown coordinate {name!r}; use one of {COORDINATES}")
        return self._file[name][:]

    def nearest_z_index(self, z_value: float) -> int:
        z = self.coordinate("z")
        return int(np.nanargmin(np.abs(z - z_value)))

    def read_frame(self, time_index: int) -> FlowFrame:
        if time_index < 0:
            time_index += self.n_times
        if not 0 <= time_index < self.n_times:
            raise IndexError(f"time_index must be in [0, {self.n_times - 1}]")

        return FlowFrame(
            time_index=time_index,
            time=float(self._file["t"][time_index]),
            x=self.coordinate("x"),
            y=self.coordinate("y"),
            z=self.coordinate("z"),
            u=self._file["u"][time_index, :, :, :],
            v=self._file["v"][time_index, :, :, :],
            w=self._file["w"][time_index, :, :, :],
        )

    def read_z_plane(self, time_index: int, z_index: int) -> FlowPlane:
        if time_index < 0:
            time_index += self.n_times
        if not 0 <= time_index < self.n_times:
            raise IndexError(f"time_index must be in [0, {self.n_times - 1}]")
        if z_index < 0:
            z_index += self.grid_shape[0]
        if not 0 <= z_index < self.grid_shape[0]:
            raise IndexError(f"z_index must be in [0, {self.grid_shape[0] - 1}]")

        z = self.coordinate("z")
        return FlowPlane(
            time_index=time_index,
            time=float(self._file["t"][time_index]),
            z_index=z_index,
            z_value=float(z[z_index]),
            x=self.coordinate("x"),
            y=self.coordinate("y"),
            u=self._file["u"][time_index, z_index, :, :],
            v=self._file["v"][time_index, z_index, :, :],
            w=self._file["w"][time_index, z_index, :, :],
        )

    def iter_frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterable[FlowFrame]:
        stop = self.n_times if stop is None else min(stop, self.n_times)
        for time_index in range(start, stop, step):
            yield self.read_frame(time_index)

    def describe(self) -> str:
        lines = [
            f"File: {self.path}",
            f"Velocity variables: {', '.join(VELOCITY_COMPONENTS)}",
            f"Velocity shape: {self.shape} = (time, z, y, x)",
            f"Velocity dtype: {self.dtype}",
            f"Number of images/time steps: {self.n_times}",
            f"Grid shape per image: {self.grid_shape} = (z, y, x)",
        ]

        for name in COORDINATES:
            values = self.coordinate(name)
            lines.append(
                f"{name}: length={values.size}, min={values.min():.6g}, "
                f"max={values.max():.6g}"
            )

        return "\n".join(lines)

    def frame_stats(self, time_index: int) -> dict[str, float]:
        frame = self.read_frame(time_index)
        speed = frame.speed
        return {
            "time_index": float(frame.time_index),
            "time": frame.time,
            "u_min": float(np.nanmin(frame.u)),
            "u_max": float(np.nanmax(frame.u)),
            "v_min": float(np.nanmin(frame.v)),
            "v_max": float(np.nanmax(frame.v)),
            "w_min": float(np.nanmin(frame.w)),
            "w_max": float(np.nanmax(frame.w)),
            "speed_mean": float(np.nanmean(speed)),
            "speed_max": float(np.nanmax(speed)),
        }


def print_stats(stats: dict[str, float]) -> None:
    print(f"\nFrame {int(stats['time_index'])} at t={stats['time']:.6g}")
    for key, value in stats.items():
        if key in {"time_index", "time"}:
            continue
        print(f"{key:>12}: {value:.6g}")


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
                "Only .gif saving is available in this environment. "
                "Use --save output.gif, or install ffmpeg for MP4 export."
            )
        writer = "pillow"
        animation.save(save, writer=writer, fps=fps)
        print(f"Saved animation to {save}")
    else:
        print(
            f"Animating {len(frame_indices)} frames at {fps:g} fps on "
            f"z={first.z_value:.6g} (nearest to requested z={z_value:g})."
        )
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a 3D velocity time series from a NetCDF4/HDF5 .nc file."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_FILE,
        help=f"path to the .nc file, default: {DEFAULT_FILE}",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="time index to inspect without loading the full file",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="visualize the x-y velocity field at the selected z plane",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=0.0,
        help="z plane to visualize; the nearest available z value is used",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=40.0,
        help="animation frame rate",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="first time index for animation",
    )
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="one-past-last time index for animation; default uses all frames",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="time-index stride for animation",
    )
    parser.add_argument(
        "--quiver-step",
        type=int,
        default=3,
        help="plot one arrow every N grid points in x and y",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="optional output path; .gif works with the installed pillow writer",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with FlowDataset(args.path) as flow:
        if args.animate:
            animate_z_plane(
                flow,
                z_value=args.z,
                fps=args.fps,
                start=args.start,
                stop=args.stop,
                step=args.step,
                quiver_step=args.quiver_step,
                save=args.save,
            )
        else:
            print(flow.describe())
            print_stats(flow.frame_stats(args.frame))


if __name__ == "__main__":
    main()
