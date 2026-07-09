from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
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

    The velocity arrays in these files are stored as (time, z, y, x). Reading a
    single frame or plane avoids loading the full multi-GB dataset into memory.
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

    def voxel_size(self) -> dict[str, dict[str, float]]:
        """Return grid spacing statistics for x, y, and z coordinates."""

        spacing = {}
        for name in ("x", "y", "z"):
            values = self.coordinate(name)
            diffs = np.diff(values)
            spacing[name] = {
                "median": float(np.nanmedian(diffs)),
                "min": float(np.nanmin(diffs)),
                "max": float(np.nanmax(diffs)),
            }
        return spacing

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

        spacing = self.voxel_size()
        lines.append(
            "Voxel size, median dx/dy/dz: "
            f"{spacing['x']['median']:.6g} / "
            f"{spacing['y']['median']:.6g} / "
            f"{spacing['z']['median']:.6g}"
        )
        lines.append(
            "Voxel spacing ranges: "
            f"dx=[{spacing['x']['min']:.6g}, {spacing['x']['max']:.6g}], "
            f"dy=[{spacing['y']['min']:.6g}, {spacing['y']['max']:.6g}], "
            f"dz=[{spacing['z']['min']:.6g}, {spacing['z']['max']:.6g}]"
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
