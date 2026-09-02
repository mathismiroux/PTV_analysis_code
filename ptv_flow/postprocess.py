from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence
import uuid

import h5py
import numpy as np

from ptv_flow.reader import COORDINATES, VELOCITY_COMPONENTS, FlowDataset
from ptv_flow.validity import (
    valid_component_samples,
    valid_vector_samples,
    validate_invalid_samples,
)

REYNOLDS_STRESS_COMPONENTS = ("uu", "uv", "uw", "vv", "vw", "ww")
INTERPOLATION_AXES = ("t", "z", "y", "x")
_AXIS_TO_DIM = {"t": 0, "z": 1, "y": 2, "x": 3}


def _prepare_output_path(output: Path, overwrite: bool) -> tuple[Path, Path]:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output}. "
            "Choose a new --output path or pass --overwrite."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f"{output.name}.tmp-{uuid.uuid4().hex}")
    return output, temporary_output


def _normalize_interpolation_axes(axes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(axis.lower() for axis in axes)
    invalid = [axis for axis in normalized if axis not in INTERPOLATION_AXES]
    if invalid:
        raise ValueError(
            "Unknown interpolation axis/axes: "
            f"{', '.join(invalid)}. Choose from {', '.join(INTERPOLATION_AXES)}."
        )
    return tuple(dict.fromkeys(normalized))


def _interpolate_along_axis(
    data: np.ndarray,
    axis: int,
    coordinates: np.ndarray,
    eligible: np.ndarray,
    max_index_gap: int | None = None,
    max_bracket_span: int | None = None,
) -> np.ndarray:
    moved = np.moveaxis(data, axis, 0)
    moved_eligible = np.moveaxis(eligible, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    flat_eligible = moved_eligible.reshape(moved_eligible.shape[0], -1)
    valid = np.isfinite(flat)
    candidate = flat_eligible & ~valid
    if not np.any(candidate):
        return data

    axis_indices = np.arange(flat.shape[0])[:, None]
    left_indices = np.where(valid, axis_indices, -1)
    left_indices = np.maximum.accumulate(left_indices, axis=0)
    right_indices = np.where(valid, axis_indices, flat.shape[0])
    right_indices = np.minimum.accumulate(right_indices[::-1, :], axis=0)[::-1, :]
    bracketed = (
        candidate
        & (left_indices >= 0)
        & (right_indices < flat.shape[0])
        & (left_indices != right_indices)
    )
    if max_index_gap is not None:
        bracketed &= (axis_indices - left_indices <= max_index_gap) & (
            right_indices - axis_indices <= max_index_gap
        )
    if max_bracket_span is not None:
        bracketed &= right_indices - left_indices <= max_bracket_span
    if not np.any(bracketed):
        return data

    filled = flat.copy()
    fill_rows, fill_columns = np.nonzero(bracketed)
    left = left_indices[fill_rows, fill_columns]
    right = right_indices[fill_rows, fill_columns]
    left_coordinates = coordinates[left]
    right_coordinates = coordinates[right]
    coordinate_span = right_coordinates - left_coordinates
    nonzero_span = coordinate_span != 0.0
    fill_rows = fill_rows[nonzero_span]
    fill_columns = fill_columns[nonzero_span]
    left = left[nonzero_span]
    right = right[nonzero_span]
    left_coordinates = left_coordinates[nonzero_span]
    coordinate_span = coordinate_span[nonzero_span]
    fractions = (coordinates[fill_rows] - left_coordinates) / coordinate_span
    left_values = flat[left, fill_columns]
    right_values = flat[right, fill_columns]
    filled[fill_rows, fill_columns] = left_values + (
        right_values - left_values
    ) * fractions
    return np.moveaxis(filled.reshape(moved.shape), 0, axis)


def _interpolate_component_passes(
    data: np.ndarray,
    hole_mask: np.ndarray,
    coordinates: Mapping[str, np.ndarray],
    axes: Sequence[str],
    passes: int,
    max_temporal_gap: int | None,
    max_spatial_gap: int | None,
) -> np.ndarray:
    for _pass_index in range(passes):
        missing = hole_mask & ~np.isfinite(data)
        if not np.any(missing):
            break

        filled_before = int(np.count_nonzero(hole_mask & np.isfinite(data)))
        for axis_name in axes:
            data = _interpolate_along_axis(
                data,
                axis=_AXIS_TO_DIM[axis_name],
                coordinates=coordinates[axis_name],
                eligible=missing,
                max_index_gap=max_temporal_gap if axis_name == "t" else None,
                max_bracket_span=max_spatial_gap if axis_name != "t" else None,
            )
        filled_after = int(np.count_nonzero(hole_mask & np.isfinite(data)))
        if filled_after == filled_before:
            break
    return data


def _interpolate_component_result(
    name: str,
    data: np.ndarray,
    hole_mask: np.ndarray,
    before_missing: int,
    coordinates: Mapping[str, np.ndarray],
    axes: Sequence[str],
    passes: int,
    max_temporal_gap: int | None,
    max_spatial_gap: int | None,
) -> tuple[str, np.ndarray, np.ndarray, int, int, int]:
    original_data = data.copy()
    data[hole_mask] = np.nan
    data = _interpolate_component_passes(
        data=data,
        hole_mask=hole_mask,
        coordinates=coordinates,
        axes=axes,
        passes=passes,
        max_temporal_gap=max_temporal_gap,
        max_spatial_gap=max_spatial_gap,
    )

    filled_mask = hole_mask & np.isfinite(data)
    untouched_nonfinite = (~hole_mask) & (~np.isfinite(original_data))
    data[untouched_nonfinite] = original_data[untouched_nonfinite]
    remaining_mask = hole_mask & ~np.isfinite(data)
    return (
        name,
        data,
        filled_mask,
        before_missing,
        int(np.count_nonzero(filled_mask)),
        int(np.count_nonzero(remaining_mask)),
    )


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
            u_count = self._file["u_count"][index, :, :] if "u_count" in self._file else None
            v_count = self._file["v_count"][index, :, :] if "v_count" in self._file else None
            w_count = self._file["w_count"][index, :, :] if "w_count" in self._file else None
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
            u_count = self._file["u_count"][:, index, :] if "u_count" in self._file else None
            v_count = self._file["v_count"][:, index, :] if "v_count" in self._file else None
            w_count = self._file["w_count"][:, index, :] if "w_count" in self._file else None
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
            u_count = self._file["u_count"][:, :, index] if "u_count" in self._file else None
            v_count = self._file["v_count"][:, :, index] if "v_count" in self._file else None
            w_count = self._file["w_count"][:, :, index] if "w_count" in self._file else None

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
            "u_count": u_count,
            "v_count": v_count,
            "w_count": w_count,
            "speed": speed,
        }


def temporal_average_volume(
    flow: FlowDataset,
    output: Path,
    chunk_size: int = 50,
    zero_mask: str = "component",
    min_valid_fraction: float = 0.0,
    overwrite: bool = False,
    u_inf: float | None = None,
    metadata: Mapping[str, str | float] | None = None,
    invalid_samples: str = "zero",
) -> Path:
    """Compute temporal mean velocity fields while excluding invalid samples.

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
    u_inf:
        Free-stream velocity used to store wake-deficit products.
    metadata:
        Optional case metadata stored as file attributes.
    invalid_samples:
        Which raw samples are excluded: ``zero``, ``nan``, ``zero-or-nan``,
        or ``none``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")
    validate_invalid_samples(invalid_samples)
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    output, temporary_output = _prepare_output_path(output, overwrite)
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
        f"invalid_samples={invalid_samples}, "
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
            valid = valid_vector_samples(chunks, invalid_samples)
            for name, data in chunks.items():
                sums[name] += np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64)
                counts[name] += valid.sum(axis=0, dtype=np.uint32)
        else:
            for name in VELOCITY_COMPONENTS:
                data = flow._file[name][start:stop, :, :, :]
                valid = valid_component_samples(data, invalid_samples)
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
    wake_deficit = None
    wake_mask_u09 = None
    u_over_u_inf = None
    if u_inf is not None:
        if u_inf == 0.0:
            raise ValueError("u_inf must be non-zero")
        u_over_u_inf = means["u"] / float(u_inf)
        wake_deficit = (float(u_inf) - means["u"]) / float(u_inf)
        wake_mask_u09 = u_over_u_inf < 0.9

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            if metadata is not None:
                for key, value in metadata.items():
                    out.attrs[key] = value
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "temporal_average_volume"
            out.attrs["zero_mask"] = zero_mask
            out.attrs["invalid_samples"] = invalid_samples
            out.attrs["chunk_size"] = chunk_size
            out.attrs["min_valid_fraction"] = min_valid_fraction
            out.attrs["min_valid_count"] = min_valid_count
            out.attrs["input_shape_time_z_y_x"] = flow.shape
            if u_inf is not None:
                out.attrs["u_inf"] = float(u_inf)

            provenance = out.create_group("provenance")
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["zero_mask"] = zero_mask
            provenance.attrs["invalid_samples"] = invalid_samples
            provenance.attrs["chunk_size"] = chunk_size
            provenance.attrs["min_valid_fraction"] = min_valid_fraction
            provenance.attrs["min_valid_count"] = min_valid_count
            if metadata is not None:
                for key, value in metadata.items():
                    provenance.attrs[key] = value
            if u_inf is not None:
                provenance.attrs["u_inf"] = float(u_inf)

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
            out.create_dataset(
                "abs_U",
                data=speed_from_mean,
                compression="gzip",
                compression_opts=4,
            )
            if (
                wake_deficit is not None
                and wake_mask_u09 is not None
                and u_over_u_inf is not None
            ):
                out.create_dataset(
                    "u_over_u_inf",
                    data=u_over_u_inf,
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    "wake_deficit",
                    data=wake_deficit,
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    "wake_mask_u09",
                    data=wake_mask_u09,
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


def apply_valid_fraction_to_average(
    source: Path,
    output: Path,
    min_valid_fraction: float,
    overwrite: bool = False,
) -> Path:
    """Apply a valid-count threshold to an existing temporal-average file."""

    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1")

    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Could not find postprocessed file: {source}")

    output, temporary_output = _prepare_output_path(output, overwrite)
    print(f"Reading postprocessed file: {source.resolve()}", flush=True)

    try:
        with h5py.File(source, "r") as src, h5py.File(temporary_output, "w") as out:
            if "input_shape_time_z_y_x" not in src.attrs:
                raise KeyError(
                    "Missing input_shape_time_z_y_x metadata; cannot infer "
                    "number of time steps for valid-fraction filtering."
                )
            n_times = int(src.attrs["input_shape_time_z_y_x"][0])
            min_valid_count = int(np.ceil(min_valid_fraction * n_times))

            for key, value in src.attrs.items():
                out.attrs[key] = value
            out.attrs["derived_from_file"] = str(source.resolve())
            out.attrs["derived_operation"] = "apply_valid_fraction_to_average"
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["min_valid_fraction"] = min_valid_fraction
            out.attrs["min_valid_count"] = min_valid_count

            for name in COORDINATES:
                if name == "t":
                    continue
                src.copy(name, out)

            means = {}
            for name in VELOCITY_COMPONENTS:
                count_name = f"{name}_count"
                mean_name = f"{name}_mean"
                if count_name not in src or mean_name not in src:
                    raise KeyError(f"Missing {mean_name!r} or {count_name!r}")

                counts = src[count_name][:]
                mean = src[mean_name][:].astype(np.float64, copy=True)
                mean[counts < min_valid_count] = np.nan
                means[name] = mean

                out.create_dataset(
                    mean_name,
                    data=mean,
                    compression="gzip",
                    compression_opts=4,
                )
                src.copy(count_name, out)

            speed_from_mean = np.sqrt(
                means["u"] * means["u"]
                + means["v"] * means["v"]
                + means["w"] * means["w"]
            )
            out.create_dataset(
                "speed_from_mean",
                data=speed_from_mean,
                compression="gzip",
                compression_opts=4,
            )

            provenance = out.create_group("provenance")
            if "provenance" in src:
                for key, value in src["provenance"].attrs.items():
                    provenance.attrs[f"parent_{key}"] = value
            provenance.attrs["derived_from_file"] = str(source.resolve())
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["derived_operation"]
            provenance.attrs["min_valid_fraction"] = min_valid_fraction
            provenance.attrs["min_valid_count"] = min_valid_count

        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(
        f"Applied min_valid_fraction={min_valid_fraction:g} "
        f"(min_count={min_valid_count})",
        flush=True,
    )
    print(f"Saved filtered average to: {output.resolve()}", flush=True)
    return output


def z_slab_indices(
    z_values: np.ndarray,
    center: float,
    width: int,
) -> tuple[int, int, int]:
    """Return ``(start, stop, center_index)`` for a centered z slab."""

    if width <= 0:
        raise ValueError("z slab width must be positive")
    if width % 2 == 0:
        raise ValueError("z slab width must be odd so it has a center plane")

    center_index = int(np.nanargmin(np.abs(z_values - center)))
    half_width = width // 2
    start = center_index - half_width
    stop = center_index + half_width + 1
    if start < 0 or stop > z_values.size:
        raise ValueError(
            "Requested z slab does not fit in the available z range: "
            f"center={center:g}, nearest_index={center_index}, width={width}, "
            f"available_indices=[0, {z_values.size - 1}]"
        )
    return start, stop, center_index


def extract_z_slab(
    flow: FlowDataset,
    output: Path,
    z_center: float = 0.0,
    z_width: int = 3,
    chunk_size: int = 50,
    overwrite: bool = False,
    metadata: Mapping[str, str | float] | None = None,
) -> Path:
    """Write a raw-style velocity file containing a centered z slab."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output, temporary_output = _prepare_output_path(output, overwrite)
    z_values = flow.coordinate("z")
    z_start, z_stop, z_center_index = z_slab_indices(
        z_values,
        center=z_center,
        width=z_width,
    )

    print(f"Reading NetCDF file: {flow.path.resolve()}", flush=True)
    print(
        f"Extracting z slab centered near {z_center:g}: "
        f"indices={z_start}:{z_stop}, "
        f"nearest_center_z={z_values[z_center_index]:.6g}, "
        f"shape={(flow.n_times, z_width, flow.grid_shape[1], flow.grid_shape[2])}",
        flush=True,
    )

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            if metadata is not None:
                for key, value in metadata.items():
                    out.attrs[key] = value
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "extract_z_slab"
            out.attrs["input_shape_time_z_y_x"] = flow.shape
            out.attrs["output_shape_time_z_y_x"] = (
                flow.n_times,
                z_width,
                flow.grid_shape[1],
                flow.grid_shape[2],
            )
            out.attrs["requested_z_center"] = float(z_center)
            out.attrs["nearest_z_center"] = float(z_values[z_center_index])
            out.attrs["source_z_center_index"] = z_center_index
            out.attrs["source_z_start_index"] = z_start
            out.attrs["source_z_stop_index"] = z_stop
            out.attrs["z_width_voxels"] = z_width

            provenance = out.create_group("provenance")
            if metadata is not None:
                for key, value in metadata.items():
                    provenance.attrs[key] = value
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["requested_z_center"] = float(z_center)
            provenance.attrs["nearest_z_center"] = float(z_values[z_center_index])
            provenance.attrs["source_z_center_index"] = z_center_index
            provenance.attrs["source_z_start_index"] = z_start
            provenance.attrs["source_z_stop_index"] = z_stop
            provenance.attrs["z_width_voxels"] = z_width

            out.create_dataset("t", data=flow.coordinate("t"))
            out.create_dataset("z", data=z_values[z_start:z_stop])
            out.create_dataset("y", data=flow.coordinate("y"))
            out.create_dataset("x", data=flow.coordinate("x"))

            for name in VELOCITY_COMPONENTS:
                dataset = out.create_dataset(
                    name,
                    shape=(flow.n_times, z_width, flow.grid_shape[1], flow.grid_shape[2]),
                    dtype=flow._file[name].dtype,
                    compression="gzip",
                    compression_opts=4,
                )
                for start in range(0, flow.n_times, chunk_size):
                    stop = min(start + chunk_size, flow.n_times)
                    dataset[start:stop, :, :, :] = flow._file[name][
                        start:stop,
                        z_start:z_stop,
                        :,
                        :,
                    ]

        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"Saved z slab to: {output.resolve()}", flush=True)
    return output


def spatio_temporal_interpolate_velocity(
    flow: FlowDataset,
    output: Path,
    axes: Sequence[str] = INTERPOLATION_AXES,
    passes: int = 1,
    max_temporal_gap: int | None = None,
    max_spatial_gap: int | None = None,
    workers: int = 1,
    zero_mask: str = "component",
    overwrite: bool = False,
    metadata: Mapping[str, str | float] | None = None,
    invalid_samples: str = "zero",
) -> Path:
    """Fill holes in raw velocity fields with sequential linear interpolation.

    Interpolation is purely data-driven. Invalid samples are converted to NaN,
    then each requested axis is filled with one-dimensional linear interpolation
    in sequence. Original valid samples are preserved exactly.
    """

    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")
    validate_invalid_samples(invalid_samples)
    if passes <= 0:
        raise ValueError("passes must be positive")
    if max_temporal_gap is not None and max_temporal_gap <= 0:
        raise ValueError("max_temporal_gap must be positive when provided")
    if max_spatial_gap is not None and max_spatial_gap <= 0:
        raise ValueError("max_spatial_gap must be positive when provided")
    if workers <= 0:
        raise ValueError("workers must be positive")
    workers = min(workers, len(VELOCITY_COMPONENTS))
    selected_axes = _normalize_interpolation_axes(axes)

    output, temporary_output = _prepare_output_path(output, overwrite)
    coordinates = {name: flow.coordinate(name) for name in INTERPOLATION_AXES}
    fill_counts: dict[str, int] = {}
    remaining_counts: dict[str, int] = {}

    print(f"Reading NetCDF file: {flow.path.resolve()}", flush=True)
    print(
        "Interpolating velocity fields, "
        f"shape={flow.shape}, axes={','.join(selected_axes)}, "
        f"passes={passes}, "
        f"max_temporal_gap={max_temporal_gap if max_temporal_gap is not None else 'none'}, "
        f"max_spatial_gap={max_spatial_gap if max_spatial_gap is not None else 'none'}, "
        f"workers={workers}, "
        f"zero_mask={zero_mask}, invalid_samples={invalid_samples}",
        flush=True,
    )

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            if metadata is not None:
                for key, value in metadata.items():
                    out.attrs[key] = value
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "spatio_temporal_interpolate_velocity"
            out.attrs["method"] = "sequential_linear_interpolation"
            out.attrs["interpolation_axes"] = np.array(
                selected_axes,
                dtype=h5py.string_dtype(),
            )
            out.attrs["interpolation_passes"] = passes
            out.attrs["max_temporal_gap"] = (
                -1 if max_temporal_gap is None else max_temporal_gap
            )
            out.attrs["max_spatial_gap"] = (
                -1 if max_spatial_gap is None else max_spatial_gap
            )
            out.attrs["interpolation_workers"] = workers
            out.attrs["zero_mask"] = zero_mask
            out.attrs["invalid_samples"] = invalid_samples
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            if metadata is not None:
                for key, value in metadata.items():
                    provenance.attrs[key] = value
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["method"] = out.attrs["method"]
            provenance.attrs["interpolation_axes"] = out.attrs["interpolation_axes"]
            provenance.attrs["interpolation_passes"] = passes
            provenance.attrs["max_temporal_gap"] = out.attrs["max_temporal_gap"]
            provenance.attrs["max_spatial_gap"] = out.attrs["max_spatial_gap"]
            provenance.attrs["interpolation_workers"] = workers
            provenance.attrs["zero_mask"] = zero_mask
            provenance.attrs["invalid_samples"] = invalid_samples

            for name in COORDINATES:
                out.create_dataset(name, data=flow.coordinate(name))

            vector_valid = None
            shared_hole_mask = None
            shared_missing_count = None
            if zero_mask == "vector":
                chunks = {
                    name: flow._file[name][:] for name in VELOCITY_COMPONENTS
                }
                vector_valid = valid_vector_samples(chunks, invalid_samples)
                shared_hole_mask = ~vector_valid
                shared_missing_count = int(np.count_nonzero(shared_hole_mask))
                print(
                    f"Shared vector hole mask: holes={shared_missing_count}",
                    flush=True,
                )
            else:
                chunks = {}

            component_tasks = []
            for name in VELOCITY_COMPONENTS:
                data = (
                    chunks[name].astype(np.float64, copy=True)
                    if zero_mask == "vector"
                    else flow._file[name][:].astype(np.float64, copy=True)
                )
                valid = (
                    vector_valid
                    if zero_mask == "vector"
                    else valid_component_samples(data, invalid_samples)
                )
                hole_mask = shared_hole_mask if zero_mask == "vector" else ~valid
                data[hole_mask] = np.nan
                if shared_missing_count is not None:
                    before_missing = shared_missing_count
                else:
                    before_missing = int(np.count_nonzero(hole_mask))

                component_tasks.append(
                    (
                        name,
                        data,
                        hole_mask,
                        before_missing,
                        coordinates,
                        selected_axes,
                        passes,
                        max_temporal_gap,
                        max_spatial_gap,
                    )
                )

            if workers == 1:
                component_results = [
                    _interpolate_component_result(*task)
                    for task in component_tasks
                ]
            else:
                task_columns = tuple(zip(*component_tasks))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    component_results = list(
                        executor.map(
                            _interpolate_component_result,
                            *task_columns,
                        )
                    )

            for name, data, filled_mask, before_missing, filled_count, remaining_count in (
                component_results
            ):
                fill_counts[name] = int(np.count_nonzero(filled_mask))
                remaining_counts[name] = remaining_count

                out.create_dataset(
                    name,
                    data=data,
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    f"{name}_filled_mask",
                    data=filled_mask,
                    compression="gzip",
                    compression_opts=4,
                )
                print(
                    f"Interpolated {name}: holes={before_missing}, "
                    f"filled={filled_count}, "
                    f"remaining={remaining_count}",
                    flush=True,
                )

            out.attrs["u_filled_count"] = fill_counts["u"]
            out.attrs["v_filled_count"] = fill_counts["v"]
            out.attrs["w_filled_count"] = fill_counts["w"]
            out.attrs["u_remaining_missing_count"] = remaining_counts["u"]
            out.attrs["v_remaining_missing_count"] = remaining_counts["v"]
            out.attrs["w_remaining_missing_count"] = remaining_counts["w"]

        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"Saved interpolated velocity file to: {output.resolve()}", flush=True)
    return output


def _validate_mean_compatible(flow: FlowDataset, mean: TemporalAverageVolume) -> None:
    if mean.grid_shape != flow.grid_shape:
        raise ValueError(
            "Mean file grid shape does not match raw file grid shape: "
            f"{mean.grid_shape} != {flow.grid_shape}. "
            f"raw={flow.path}, mean={mean.path}"
        )

    for name in ("x", "y", "z"):
        raw_values = flow.coordinate(name)
        mean_values = mean.coordinate(name)
        if raw_values.shape != mean_values.shape or not np.allclose(
            raw_values, mean_values, equal_nan=True
        ):
            raise ValueError(
                f"Mean file {name!r} coordinates do not match raw file. "
                f"raw={flow.path}, mean={mean.path}"
            )

    mean_source = mean._file.attrs.get("source_file")
    if mean_source is None:
        raise ValueError(
            "Mean file is missing 'source_file' provenance, so postprocessing cannot "
            f"verify that it was created from the raw file: {flow.path}"
        )
    if isinstance(mean_source, bytes):
        mean_source = mean_source.decode()

    raw_source_path = flow.path.resolve()
    mean_source_path = Path(str(mean_source)).resolve()
    if mean_source_path != raw_source_path:
        raise ValueError(
            "Mean file provenance does not match raw file. "
            f"raw={raw_source_path}, mean_source={mean_source_path}, "
            f"mean_file={mean.path}"
        )


def turbulent_kinetic_energy(
    flow: FlowDataset,
    mean: TemporalAverageVolume,
    output: Path,
    chunk_size: int = 50,
    zero_mask: str = "component",
    overwrite: bool = False,
    metadata: Mapping[str, str | float] | None = None,
    invalid_samples: str = "zero",
) -> Path:
    """Compute turbulent kinetic energy from raw velocities and mean velocities.

    The output stores component fluctuation variances and
    ``tke = 0.5 * (u_prime2_mean + v_prime2_mean + w_prime2_mean)``.
    Raw values are ignored using the requested zero mask and invalid-sample
    mode.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")
    validate_invalid_samples(invalid_samples)

    _validate_mean_compatible(flow, mean)
    output, temporary_output = _prepare_output_path(output, overwrite)

    mean_fields = {
        name: mean._file[f"{name}_mean"][:].astype(np.float64)
        for name in VELOCITY_COMPONENTS
    }
    sum_squares = {
        name: np.zeros(flow.grid_shape, dtype=np.float64)
        for name in VELOCITY_COMPONENTS
    }
    counts = {
        name: np.zeros(flow.grid_shape, dtype=np.uint32)
        for name in VELOCITY_COMPONENTS
    }

    print(f"Reading NetCDF file: {flow.path.resolve()}", flush=True)
    print(f"Reading mean velocity file: {mean.path.resolve()}", flush=True)
    print(
        f"Computing turbulent kinetic energy over {flow.n_times} time steps, "
        f"grid={flow.grid_shape}, zero_mask={zero_mask}, "
        f"invalid_samples={invalid_samples}, chunk_size={chunk_size}",
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
            vector_valid = valid_vector_samples(chunks, invalid_samples)
            for name, data in chunks.items():
                valid = vector_valid & np.isfinite(mean_fields[name])[None, :, :, :]
                fluctuation = data - mean_fields[name][None, :, :, :]
                sum_squares[name] += np.where(valid, fluctuation * fluctuation, 0.0).sum(
                    axis=0, dtype=np.float64
                )
                counts[name] += valid.sum(axis=0, dtype=np.uint32)
        else:
            for name in VELOCITY_COMPONENTS:
                data = flow._file[name][start:stop, :, :, :]
                valid = valid_component_samples(data, invalid_samples) & np.isfinite(
                    mean_fields[name]
                )[None, :, :, :]
                fluctuation = data - mean_fields[name][None, :, :, :]
                sum_squares[name] += np.where(valid, fluctuation * fluctuation, 0.0).sum(
                    axis=0, dtype=np.float64
                )
                counts[name] += valid.sum(axis=0, dtype=np.uint32)

        elapsed = perf_counter() - start_time
        print(
            f"Processed time steps {start:4d}-{stop - 1:4d} / "
            f"{flow.n_times - 1} ({elapsed:.1f} s)",
            flush=True,
        )

    variances = {}
    for name in VELOCITY_COMPONENTS:
        variances[name] = np.full(flow.grid_shape, np.nan, dtype=np.float64)
        np.divide(
            sum_squares[name],
            counts[name],
            out=variances[name],
            where=counts[name] > 0,
        )
    tke = 0.5 * (variances["u"] + variances["v"] + variances["w"])

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            mean_path = mean.path.resolve()
            if metadata is not None:
                for key, value in metadata.items():
                    out.attrs[key] = value
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["mean_file"] = str(mean_path)
            out.attrs["mean_file_name"] = mean_path.name
            out.attrs["mean_file_parent"] = str(mean_path.parent)
            out.attrs["mean_file_size_bytes"] = mean_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "turbulent_kinetic_energy"
            out.attrs["formula"] = "0.5 * (mean(u_prime^2) + mean(v_prime^2) + mean(w_prime^2))"
            out.attrs["zero_mask"] = zero_mask
            out.attrs["invalid_samples"] = invalid_samples
            out.attrs["chunk_size"] = chunk_size
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            if metadata is not None:
                for key, value in metadata.items():
                    provenance.attrs[key] = value
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["mean_file"] = str(mean_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["zero_mask"] = zero_mask
            provenance.attrs["invalid_samples"] = invalid_samples
            provenance.attrs["chunk_size"] = chunk_size

            for name in COORDINATES:
                if name == "t":
                    continue
                out.create_dataset(name, data=flow.coordinate(name))

            for name in VELOCITY_COMPONENTS:
                out.create_dataset(
                    f"{name}_prime2_mean",
                    data=variances[name],
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    f"{name}_prime2_count",
                    data=counts[name],
                    compression="gzip",
                    compression_opts=4,
                )
            out.create_dataset("tke", data=tke, compression="gzip", compression_opts=4)
        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"Saved turbulent kinetic energy to: {output.resolve()}", flush=True)
    return output


def _normalize_reynolds_stress_components(
    components: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    normalized = tuple(component.lower() for component in components)
    if "all" in normalized:
        if len(normalized) > 1:
            raise ValueError("Use either 'all' or explicit Reynolds stress components")
        return REYNOLDS_STRESS_COMPONENTS

    invalid = [
        component
        for component in normalized
        if component not in REYNOLDS_STRESS_COMPONENTS
    ]
    if invalid:
        raise ValueError(
            "Unknown Reynolds stress component(s): "
            f"{', '.join(invalid)}. Choose from "
            f"{', '.join(REYNOLDS_STRESS_COMPONENTS)} or all."
        )
    return tuple(dict.fromkeys(normalized))


def reynolds_stresses(
    flow: FlowDataset,
    mean: TemporalAverageVolume,
    output: Path,
    components: tuple[str, ...] | list[str] = ("all",),
    chunk_size: int = 50,
    zero_mask: str = "component",
    overwrite: bool = False,
    metadata: Mapping[str, str | float] | None = None,
    invalid_samples: str = "zero",
) -> Path:
    """Compute selected Reynolds stress components from raw and mean velocities.

    The stored fields are temporal means of fluctuation products, for example
    ``uv_reynolds_stress = mean(u_prime * v_prime)``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")
    validate_invalid_samples(invalid_samples)

    selected_components = _normalize_reynolds_stress_components(components)
    _validate_mean_compatible(flow, mean)
    output, temporary_output = _prepare_output_path(output, overwrite)

    mean_fields = {
        name: mean._file[f"{name}_mean"][:].astype(np.float64)
        for name in VELOCITY_COMPONENTS
    }
    sums = {
        component: np.zeros(flow.grid_shape, dtype=np.float64)
        for component in selected_components
    }
    counts = {
        component: np.zeros(flow.grid_shape, dtype=np.uint32)
        for component in selected_components
    }

    print(f"Reading NetCDF file: {flow.path.resolve()}", flush=True)
    print(f"Reading mean velocity file: {mean.path.resolve()}", flush=True)
    print(
        f"Computing Reynolds stresses {', '.join(selected_components)} over "
        f"{flow.n_times} time steps, grid={flow.grid_shape}, "
        f"zero_mask={zero_mask}, invalid_samples={invalid_samples}, "
        f"chunk_size={chunk_size}",
        flush=True,
    )

    start_time = perf_counter()
    for start in range(0, flow.n_times, chunk_size):
        stop = min(start + chunk_size, flow.n_times)
        chunks = {
            name: flow._file[name][start:stop, :, :, :]
            for name in VELOCITY_COMPONENTS
        }
        fluctuations = {
            name: chunks[name] - mean_fields[name][None, :, :, :]
            for name in VELOCITY_COMPONENTS
        }
        finite_mean = {
            name: np.isfinite(mean_fields[name])[None, :, :, :]
            for name in VELOCITY_COMPONENTS
        }
        if zero_mask == "vector":
            vector_valid = valid_vector_samples(chunks, invalid_samples)
            valid_by_component = {
                name: vector_valid & finite_mean[name] for name in VELOCITY_COMPONENTS
            }
        else:
            valid_by_component = {
                name: valid_component_samples(chunks[name], invalid_samples)
                & finite_mean[name]
                for name in VELOCITY_COMPONENTS
            }

        for component in selected_components:
            first, second = component
            valid = valid_by_component[first] & valid_by_component[second]
            product = fluctuations[first] * fluctuations[second]
            sums[component] += np.where(valid, product, 0.0).sum(
                axis=0, dtype=np.float64
            )
            counts[component] += valid.sum(axis=0, dtype=np.uint32)

        elapsed = perf_counter() - start_time
        print(
            f"Processed time steps {start:4d}-{stop - 1:4d} / "
            f"{flow.n_times - 1} ({elapsed:.1f} s)",
            flush=True,
        )

    stresses = {}
    for component in selected_components:
        stresses[component] = np.full(flow.grid_shape, np.nan, dtype=np.float64)
        np.divide(
            sums[component],
            counts[component],
            out=stresses[component],
            where=counts[component] > 0,
        )

    try:
        with h5py.File(temporary_output, "w") as out:
            source_path = flow.path.resolve()
            mean_path = mean.path.resolve()
            if metadata is not None:
                for key, value in metadata.items():
                    out.attrs[key] = value
            out.attrs["source_file"] = str(source_path)
            out.attrs["source_file_name"] = source_path.name
            out.attrs["source_file_parent"] = str(source_path.parent)
            out.attrs["source_file_size_bytes"] = source_path.stat().st_size
            out.attrs["mean_file"] = str(mean_path)
            out.attrs["mean_file_name"] = mean_path.name
            out.attrs["mean_file_parent"] = str(mean_path.parent)
            out.attrs["mean_file_size_bytes"] = mean_path.stat().st_size
            out.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            out.attrs["created_by"] = "ptv-flow"
            out.attrs["operation"] = "reynolds_stresses"
            out.attrs["formula"] = "mean(component_a_prime * component_b_prime)"
            out.attrs["reynolds_stress_components"] = np.array(
                selected_components, dtype=h5py.string_dtype()
            )
            out.attrs["zero_mask"] = zero_mask
            out.attrs["invalid_samples"] = invalid_samples
            out.attrs["chunk_size"] = chunk_size
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            if metadata is not None:
                for key, value in metadata.items():
                    provenance.attrs[key] = value
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["mean_file"] = str(mean_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["components"] = np.array(
                selected_components, dtype=h5py.string_dtype()
            )
            provenance.attrs["zero_mask"] = zero_mask
            provenance.attrs["invalid_samples"] = invalid_samples
            provenance.attrs["chunk_size"] = chunk_size

            for name in COORDINATES:
                if name == "t":
                    continue
                out.create_dataset(name, data=flow.coordinate(name))

            for component in selected_components:
                out.create_dataset(
                    f"{component}_reynolds_stress",
                    data=stresses[component],
                    compression="gzip",
                    compression_opts=4,
                )
                out.create_dataset(
                    f"{component}_count",
                    data=counts[component],
                    compression="gzip",
                    compression_opts=4,
                )
        temporary_output.replace(output)
    except Exception:
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print(f"Saved Reynolds stresses to: {output.resolve()}", flush=True)
    return output
