from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import uuid

import h5py
import numpy as np

from ptv_flow.reader import COORDINATES, VELOCITY_COMPONENTS, FlowDataset

REYNOLDS_STRESS_COMPONENTS = ("uu", "uv", "uw", "vv", "vw", "ww")


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
) -> Path:
    """Compute turbulent kinetic energy from raw velocities and mean velocities.

    The output stores component fluctuation variances and
    ``tke = 0.5 * (u_prime2_mean + v_prime2_mean + w_prime2_mean)``.
    Exact-zero raw values are ignored using the requested zero mask.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")

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
        f"grid={flow.grid_shape}, zero_mask={zero_mask}, chunk_size={chunk_size}",
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
            vector_valid = np.logical_not(
                (chunks["u"] == 0.0) & (chunks["v"] == 0.0) & (chunks["w"] == 0.0)
            )
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
                valid = (data != 0.0) & np.isfinite(mean_fields[name])[None, :, :, :]
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
            out.attrs["chunk_size"] = chunk_size
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["mean_file"] = str(mean_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["zero_mask"] = zero_mask
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
) -> Path:
    """Compute selected Reynolds stress components from raw and mean velocities.

    The stored fields are temporal means of fluctuation products, for example
    ``uv_reynolds_stress = mean(u_prime * v_prime)``.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if zero_mask not in {"component", "vector"}:
        raise ValueError("zero_mask must be 'component' or 'vector'")

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
        f"zero_mask={zero_mask}, chunk_size={chunk_size}",
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
            vector_valid = np.logical_not(
                (chunks["u"] == 0.0) & (chunks["v"] == 0.0) & (chunks["w"] == 0.0)
            )
            valid_by_component = {
                name: vector_valid & finite_mean[name] for name in VELOCITY_COMPONENTS
            }
        else:
            valid_by_component = {
                name: (chunks[name] != 0.0) & finite_mean[name]
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
            out.attrs["chunk_size"] = chunk_size
            out.attrs["input_shape_time_z_y_x"] = flow.shape

            provenance = out.create_group("provenance")
            provenance.attrs["source_file"] = str(source_path)
            provenance.attrs["mean_file"] = str(mean_path)
            provenance.attrs["created_utc"] = out.attrs["created_utc"]
            provenance.attrs["operation"] = out.attrs["operation"]
            provenance.attrs["components"] = np.array(
                selected_components, dtype=h5py.string_dtype()
            )
            provenance.attrs["zero_mask"] = zero_mask
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
