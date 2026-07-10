from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import uuid

import h5py
import numpy as np

from ptv_flow.reader import COORDINATES, VELOCITY_COMPONENTS, FlowDataset


class TemporalAverageVolume:
    """Reader for temporal-average output files created by this package."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Could not find postprocessed file: {self.path}")
        self._file = h5py.File(self.path, "r")
        self._validate()

    def __enter__(self) -> "TemporalAverageVolume":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def _validate(self) -> None:
        required = ("x", "y", "z", "u_mean", "v_mean", "w_mean")
        missing = [name for name in required if name not in self._file]
        if missing:
            raise KeyError(f"Missing expected variable(s): {', '.join(missing)}")

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return self._file["u_mean"].shape

    def coordinate(self, name: str) -> np.ndarray:
        if name not in ("x", "y", "z"):
            raise ValueError("name must be one of 'x', 'y', or 'z'")
        return self._file[name][:]

    def nearest_z_index(self, z_value: float) -> int:
        return self.nearest_index("z", z_value)

    def nearest_index(self, axis: str, value: float) -> int:
        values = self.coordinate(axis)
        return int(np.nanargmin(np.abs(values - value)))

    def read_z_plane(self, z_index: int) -> dict[str, np.ndarray | float | int]:
        return self.read_plane("z", z_index)

    def read_plane(
        self, axis: str, index: int
    ) -> dict[str, np.ndarray | float | int | str]:
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of 'x', 'y', or 'z'")

        axis_to_dim = {"z": 0, "y": 1, "x": 2}
        dim = axis_to_dim[axis]
        max_index = self.grid_shape[dim] - 1
        if index < 0:
            index += self.grid_shape[dim]
        if not 0 <= index <= max_index:
            raise IndexError(f"{axis}_index must be in [0, {max_index}]")

        values = self.coordinate(axis)
        if axis == "z":
            horizontal_axis = "x"
            vertical_axis = "y"
            vector_horizontal_name = "u"
            vector_vertical_name = "v"
            u = self._file["u_mean"][index, :, :]
            v = self._file["v_mean"][index, :, :]
            w = self._file["w_mean"][index, :, :]
            speed = (
                self._file["speed_from_mean"][index, :, :]
                if "speed_from_mean" in self._file
                else np.sqrt(u * u + v * v + w * w)
            )
        elif axis == "y":
            horizontal_axis = "x"
            vertical_axis = "z"
            vector_horizontal_name = "u"
            vector_vertical_name = "w"
            u = self._file["u_mean"][:, index, :]
            v = self._file["v_mean"][:, index, :]
            w = self._file["w_mean"][:, index, :]
            speed = (
                self._file["speed_from_mean"][:, index, :]
                if "speed_from_mean" in self._file
                else np.sqrt(u * u + v * v + w * w)
            )
        else:
            horizontal_axis = "y"
            vertical_axis = "z"
            vector_horizontal_name = "v"
            vector_vertical_name = "w"
            u = self._file["u_mean"][:, :, index]
            v = self._file["v_mean"][:, :, index]
            w = self._file["w_mean"][:, :, index]
            speed = (
                self._file["speed_from_mean"][:, :, index]
                if "speed_from_mean" in self._file
                else np.sqrt(u * u + v * v + w * w)
            )

        vector_horizontal = {"u": u, "v": v, "w": w}[vector_horizontal_name]
        vector_vertical = {"u": u, "v": v, "w": w}[vector_vertical_name]
        speed = (
            speed
            if isinstance(speed, np.ndarray)
            else np.asarray(speed)
        )
        return {
            "axis": axis,
            "index": index,
            "value": float(values[index]),
            f"{axis}_index": index,
            f"{axis}_value": float(values[index]),
            "horizontal_axis": horizontal_axis,
            "vertical_axis": vertical_axis,
            "horizontal": self.coordinate(horizontal_axis),
            "vertical": self.coordinate(vertical_axis),
            "vector_horizontal_name": vector_horizontal_name,
            "vector_vertical_name": vector_vertical_name,
            "vector_horizontal": vector_horizontal,
            "vector_vertical": vector_vertical,
            "u": u,
            "v": v,
            "w": w,
            "speed": speed,
        }


def temporal_average_volume(
    flow: FlowDataset,
    output: Path,
    chunk_size: int = 50,
    zero_mask: str = "component",
    min_valid_fraction: float = 0.0,
    overwrite: bool = False,
) -> Path:
    """Compute temporal mean velocity fields while ignoring exact zeros.

    Parameters
    ----------
    flow:
        Open velocity dataset.
    output:
        HDF5/NetCDF-style output file containing coordinates, means, counts,
        and speed magnitude computed from the mean vector.
    chunk_size:
        Number of time steps read per chunk.
    zero_mask:
        ``component`` ignores zeros independently for u, v, and w. ``vector``
        ignores a sample only when u, v, and w are all exactly zero.
    min_valid_fraction:
        Minimum fraction of time steps that must be valid at a voxel. Means
        with fewer valid samples are set to NaN. Use 0.8 for an 80% cutoff.
    overwrite:
        Replace an existing output file. By default existing files are refused.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output}. "
            "Choose a new --output path or pass --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f"{output.name}.tmp-{uuid.uuid4().hex}"
    )
    min_valid_count = int(np.ceil(min_valid_fraction * flow.n_times))

    sums = {
        name: np.zeros(flow.grid_shape, dtype=np.float64)
        for name in VELOCITY_COMPONENTS
    }
    counts = {
        name: np.zeros(flow.grid_shape, dtype=np.uint32)
        for name in VELOCITY_COMPONENTS
    }

    print(f"Reading NetCDF file: {flow.path.resolve()}", flush=True)
    print(
        f"Computing temporal average over {flow.n_times} time steps, "
        f"grid={flow.grid_shape}, zero_mask={zero_mask}, "
        f"chunk_size={chunk_size}, min_valid_fraction={min_valid_fraction:g} "
        f"(min_count={min_valid_count})",
        flush=True,
    )

    start_time = perf_counter()
    for start in range(0, flow.n_times, chunk_size):
        stop = min(start + chunk_size, flow.n_times)
        if zero_mask == "vector":
            chunks = {
                name: flow._file[name][start:stop, :, :, :]
                for name in VELOCITY_COMPONENTS
            }
            valid = np.logical_not(
                (chunks["u"] == 0.0) & (chunks["v"] == 0.0) & (chunks["w"] == 0.0)
            )
            for name, data in chunks.items():
                sums[name] += np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64)
                counts[name] += valid.sum(axis=0, dtype=np.uint32)
        else:
            for name in VELOCITY_COMPONENTS:
                data = flow._file[name][start:stop, :, :, :]
                valid = data != 0.0
                sums[name] += np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64)
                counts[name] += valid.sum(axis=0, dtype=np.uint32)

        elapsed = perf_counter() - start_time
        print(
            f"Processed time steps {start:4d}-{stop - 1:4d} / "
            f"{flow.n_times - 1} ({elapsed:.1f} s)",
            flush=True,
        )

    means = {}
    for name in VELOCITY_COMPONENTS:
        means[name] = np.full(flow.grid_shape, np.nan, dtype=np.float64)
        enough_data = (counts[name] > 0) & (counts[name] >= min_valid_count)
        np.divide(sums[name], counts[name], out=means[name], where=enough_data)

    speed_from_mean = np.sqrt(
        means["u"] * means["u"] + means["v"] * means["v"] + means["w"] * means["w"]
    )

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "temporal_average_volume_ignore_exact_zeros"
            out.attrs["zero_mask"] = zero_mask
            out.attrs["chunk_size"] = chunk_size
            out.attrs["min_valid_fraction"] = min_valid_fraction
            out.attrs["min_valid_count"] = min_valid_count
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["zero_mask"] = zero_mask
            provenance.attrs["chunk_size"] = chunk_size
            provenance.attrs["min_valid_fraction"] = min_valid_fraction
            provenance.attrs["min_valid_count"] = min_valid_count

            for name in COORDINATES:
                if name == "t":
                    continue
                out.create_dataset(name, data=flow.coordinate(name))

            for name in VELOCITY_COMPONENTS:
                out.create_dataset(
                    f"{name}_mean",
                    data=means[name],
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    f"{name}_count",
                    data=counts[name],
                    compression="gzip",
                    compression_opts=4,
                )
            out.create_dataset(
                "speed_from_mean",
                data=speed_from_mean,
                compression="gzip",
                compression_opts=4,
            )
        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    print(f"Saved temporal average to: {output.resolve()}", flush=True)
    return output
