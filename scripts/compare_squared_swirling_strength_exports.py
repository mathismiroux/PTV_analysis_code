from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import h5py
import matplotlib

if "--inspect-z" not in sys.argv and "--inspect-z-compare" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np


COORDINATE_ALIASES = {
    "x": ("x", "coord_x", "position_x", "pos_x", "xcoordinate", "coordinatex"),
    "y": ("y", "coord_y", "position_y", "pos_y", "ycoordinate", "coordinatey"),
    "z": ("z", "coord_z", "position_z", "pos_z", "zcoordinate", "coordinatez"),
}
SWIRL_ALIASES = (
    "lambda_ci_squared",
    "lambda_ci^2",
    "lambdaci2",
    "lambdaci_squared",
    "lambdaci_sq",
    "lambdaci^2",
    "lambdaci",
    "lambdasquared",
    "lambdaci2",
    "lambdaci_squared",
    "lambdaci2",
    "lambdaci2",
    "swirling_strength_squared",
    "swir",
    "swirl2",
    "swirling2",
    "lambda2",
    "lambda_2",
    "lambda_ci_sq",
)
VELOCITY_ALIASES = {
    "u": ("u", "ux", "velocity_u", "velocityx", "velocity_x", "u_velocity"),
    "v": ("v", "uy", "velocity_v", "velocityy", "velocity_y", "v_velocity"),
    "w": ("w", "uz", "velocity_w", "velocityz", "velocity_z", "w_velocity"),
}


@dataclass(frozen=True)
class FieldTable:
    path: Path
    columns: dict[str, np.ndarray]
    column_map: dict[str, str]

    @property
    def x(self) -> np.ndarray:
        return self.columns[self.column_map["x"]]

    @property
    def y(self) -> np.ndarray:
        return self.columns[self.column_map["y"]]

    @property
    def z(self) -> np.ndarray:
        return self.columns[self.column_map["z"]]

    @property
    def swirl_squared(self) -> np.ndarray:
        return self.columns[self.column_map["lambda_ci_squared"]]


@dataclass(frozen=True)
class GridInspection:
    label: str
    n_points: int
    nx: int
    ny: int
    nz: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    dx: float
    dy: float
    dz: float
    dx_over_d: float
    dy_over_d: float
    dz_over_d: float
    structured: bool
    uniform: bool
    nan_count: int
    nan_percent: float
    duplicate_points: int
    missing_points: int
    warnings: list[str]


@dataclass(frozen=True)
class StructuredField:
    label: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    swirl_star: np.ndarray
    u: np.ndarray | None = None


@dataclass(frozen=True)
class Component:
    file_label: str
    threshold_label: str
    component_id: int
    size_cells: int
    centroid_x: float
    centroid_y: float
    centroid_z: float
    weighted_centroid_x: float
    weighted_centroid_y: float
    weighted_centroid_z: float
    peak_x: float
    peak_y: float
    peak_z: float
    peak: float
    integrated: float
    bbox_x_min: float
    bbox_x_max: float
    bbox_y_min: float
    bbox_y_max: float
    bbox_z_min: float
    bbox_z_max: float
    core_diameter_m: float
    cells_across_core: float
    touches_boundary: bool


@dataclass(frozen=True)
class PlaneComponent:
    component_id: int
    size_cells: int
    weighted_centroid_x: float
    weighted_centroid_y: float
    peak_x: float
    peak_y: float
    peak: float
    integrated: float
    touches_boundary: bool


@dataclass(frozen=True)
class VortexFileView:
    path: Path
    label: str
    h5: h5py.File
    variable: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    n_frames: int


def _normalise_name(name: str) -> str:
    cleaned = name.strip().strip('"').strip("'")
    cleaned = cleaned.replace("^2", "2")
    return re.sub(r"[^a-z0-9]+", "", cleaned.lower())


def _split_header(line: str) -> list[str]:
    if '"' in line or "'" in line:
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', line)
        values = [a or b for a, b in quoted]
        if values:
            return values
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    if ";" in line:
        return [part.strip() for part in line.split(";") if part.strip()]
    return [part.strip() for part in line.split() if part.strip()]


def _infer_delimiter(line: str) -> str | None:
    if "," in line:
        return ","
    if ";" in line:
        return ";"
    if "\t" in line:
        return "\t"
    return None


def _looks_numeric(line: str) -> bool:
    try:
        [float(part) for part in re.split(r"[,\s;]+", line.strip()) if part]
    except ValueError:
        return False
    return True


def _tecplot_variables(lines: list[str]) -> list[str] | None:
    for line in lines[:50]:
        if line.strip().lower().startswith("variables"):
            return _split_header(line.split("=", 1)[-1])
    return None


def _apply_swirl_sign(values: np.ndarray, swirl_sign: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values.astype(np.float64)
    if swirl_sign == "positive":
        signed = values.astype(np.float64)
    elif swirl_sign == "negative":
        signed = -values.astype(np.float64)
    elif swirl_sign == "auto":
        has_positive = np.nanmax(finite) > 0.0
        has_negative = np.nanmin(finite) < 0.0
        signed = -values.astype(np.float64) if has_negative and not has_positive else values.astype(np.float64)
    else:
        raise ValueError("swirl_sign must be 'auto', 'positive', or 'negative'")
    signed[(signed < 0.0) & np.isfinite(signed)] = 0.0
    return signed


def _apply_hole_mask(columns: dict[str, np.ndarray], hole_mask: str) -> None:
    if hole_mask == "none":
        return
    if hole_mask == "velocity-zero":
        if not all(name in columns for name in ("u", "v", "w")):
            raise ValueError("--hole-mask velocity-zero requires u, v, and w")
        holes = (columns["u"] == 0.0) & (columns["v"] == 0.0) & (columns["w"] == 0.0)
    elif hole_mask == "velocity-nan":
        if not all(name in columns for name in ("u", "v", "w")):
            raise ValueError("--hole-mask velocity-nan requires u, v, and w")
        holes = ~(
            np.isfinite(columns["u"])
            & np.isfinite(columns["v"])
            & np.isfinite(columns["w"])
        )
    elif hole_mask == "velocity-zero-or-nan":
        if not all(name in columns for name in ("u", "v", "w")):
            raise ValueError("--hole-mask velocity-zero-or-nan requires u, v, and w")
        holes = (
            (columns["u"] == 0.0)
            & (columns["v"] == 0.0)
            & (columns["w"] == 0.0)
        ) | ~(
            np.isfinite(columns["u"])
            & np.isfinite(columns["v"])
            & np.isfinite(columns["w"])
        )
    else:
        raise ValueError(
            "hole_mask must be 'none', 'velocity-zero', 'velocity-nan', "
            "or 'velocity-zero-or-nan'"
        )
    columns["lambda_ci_squared"] = columns["lambda_ci_squared"].copy()
    columns["lambda_ci_squared"][holes] = np.nan


def load_field_file(
    path: Path,
    frame_index: int = 0,
    swirl_variable: str | None = None,
    swirl_sign: str = "auto",
    hole_mask: str = "none",
) -> FieldTable:
    """Load a table-like export and infer the required columns."""

    print(f"Loading field file: {path.resolve()}", flush=True)
    if path.suffix.lower() in {".nc", ".h5", ".hdf5"}:
        return _load_hdf5_field_file(
            path,
            frame_index=frame_index,
            swirl_variable=swirl_variable,
            swirl_sign=swirl_sign,
            hole_mask=hole_mask,
        )

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    variables = _tecplot_variables(lines)
    numeric_lines: list[str] = []
    header: list[str] | None = variables
    delimiter: str | None = None

    if variables is not None:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith(("title", "variables", "zone", "#", "%")):
                continue
            if _looks_numeric(stripped):
                delimiter = delimiter or _infer_delimiter(stripped)
                numeric_lines.append(stripped)
    else:
        header_seen = False
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            lower = stripped.lower()
            if lower.startswith(("title", "variables", "zone")):
                continue
            if not header_seen:
                if _looks_numeric(stripped):
                    raise ValueError(
                        f"{path} has numeric data but no identifiable column header. "
                        "Available columns cannot be inferred safely."
                    )
                header = _split_header(stripped)
                delimiter = _infer_delimiter(stripped)
                header_seen = True
                continue
            if _looks_numeric(stripped):
                delimiter = delimiter or _infer_delimiter(stripped)
                numeric_lines.append(stripped)

    if header is None or not numeric_lines:
        raise ValueError(f"Could not find a header and numeric data in {path}")

    array = np.genfromtxt(numeric_lines, delimiter=delimiter, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[1] != len(header):
        raise ValueError(
            f"Column count mismatch in {path}: header has {len(header)} columns, "
            f"data has {array.shape[1]} columns. Header={header}"
        )

    columns = {name: array[:, i] for i, name in enumerate(header)}
    column_map = infer_columns(list(columns))
    swirl_name = column_map["lambda_ci_squared"]
    columns["lambda_ci_squared"] = _apply_swirl_sign(columns[swirl_name], swirl_sign)
    for component in ("u", "v", "w"):
        if component in column_map:
            columns[component] = columns[column_map[component]]
    _apply_hole_mask(columns, hole_mask)
    column_map["lambda_ci_squared"] = "lambda_ci_squared"
    return FieldTable(path=path, columns=columns, column_map=column_map)


def _load_hdf5_field_file(
    path: Path,
    frame_index: int,
    swirl_variable: str | None,
    swirl_sign: str,
    hole_mask: str,
) -> FieldTable:
    with h5py.File(path, "r") as h5:
        names = list(h5.keys())
        if swirl_variable is None:
            column_map = infer_columns(names)
            swirl_variable = column_map["lambda_ci_squared"]
        elif swirl_variable not in h5:
            raise ValueError(
                f"Variable {swirl_variable!r} not found. Available variables: "
                f"{', '.join(names)}"
            )
        for coord in ("x", "y", "z"):
            if coord not in h5:
                raise ValueError(f"Coordinate {coord!r} not found in {path}")
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        raw = h5[swirl_variable]
        if raw.ndim == 4:
            if not 0 <= frame_index < raw.shape[0]:
                raise ValueError(f"frame_index must be in [0, {raw.shape[0] - 1}]")
            swirl = raw[frame_index, :, :, :].astype(np.float64).transpose(2, 1, 0)
        elif raw.ndim == 3:
            swirl = raw[:, :, :].astype(np.float64).transpose(2, 1, 0)
        else:
            raise ValueError(
                f"Expected {swirl_variable!r} to be 3D or 4D, got shape {raw.shape}"
            )
        columns = {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "lambda_ci_squared": _apply_swirl_sign(swirl.ravel(), swirl_sign),
        }
        for component in ("u", "v", "w"):
            if component not in h5:
                continue
            data = h5[component]
            if data.ndim == 4:
                values = data[frame_index, :, :, :].astype(np.float64).transpose(2, 1, 0)
            elif data.ndim == 3:
                values = data[:, :, :].astype(np.float64).transpose(2, 1, 0)
            else:
                continue
            columns[component] = values.ravel()
        _apply_hole_mask(columns, hole_mask)
    column_map = {
        "x": "x",
        "y": "y",
        "z": "z",
        "lambda_ci_squared": "lambda_ci_squared",
    }
    for component in ("u", "v", "w"):
        if component in columns:
            column_map[component] = component
    return FieldTable(path=path, columns=columns, column_map=column_map)


def infer_columns(columns: list[str]) -> dict[str, str]:
    normalised = {_normalise_name(name): name for name in columns}
    result: dict[str, str] = {}

    for target, aliases in COORDINATE_ALIASES.items():
        for alias in aliases:
            if _normalise_name(alias) in normalised:
                result[target] = normalised[_normalise_name(alias)]
                break

    for alias in SWIRL_ALIASES:
        key = _normalise_name(alias)
        if key in normalised:
            result["lambda_ci_squared"] = normalised[key]
            break

    for target, aliases in VELOCITY_ALIASES.items():
        for alias in aliases:
            key = _normalise_name(alias)
            if key in normalised:
                result[target] = normalised[key]
                break

    required = ("x", "y", "z", "lambda_ci_squared")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(
            "Could not infer required columns "
            f"{missing}. Available columns are: {', '.join(columns)}"
        )
    return result


def nondimensionalize_swirl_squared(
    lambda_ci_squared: np.ndarray, rotor_diameter: float, u_inf: float
) -> np.ndarray:
    return lambda_ci_squared * rotor_diameter**2 / u_inf**2


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    return np.unique(values[np.isfinite(values)])


def _spacing(values: np.ndarray) -> tuple[float, bool, list[str]]:
    warnings: list[str] = []
    if values.size < 2:
        return float("nan"), True, warnings
    diffs = np.diff(values)
    if np.any(diffs <= 0):
        warnings.append("coordinate values are not strictly increasing")
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return float("nan"), True, warnings
    typical = float(np.median(positive))
    uniform = bool(np.allclose(positive, typical, rtol=1e-5, atol=1e-10))
    if not uniform:
        warnings.append("nonuniform coordinate spacing detected")
    return typical, uniform, warnings


def inspect_grid(
    field: FieldTable,
    label: str,
    rotor_diameter: float,
    coordinate_unit: str,
) -> GridInspection:
    scale = 0.001 if coordinate_unit == "mm" else 1.0
    x = field.x * scale
    y = field.y * scale
    z = field.z * scale
    swirl = field.swirl_squared
    xu = _unique_sorted(x)
    yu = _unique_sorted(y)
    zu = _unique_sorted(z)
    xyz = np.column_stack((x, y, z))
    unique_xyz = np.unique(xyz, axis=0)
    duplicate_points = int(xyz.shape[0] - unique_xyz.shape[0])
    expected = int(xu.size * yu.size * zu.size)
    missing_points = int(max(expected - unique_xyz.shape[0], 0))
    structured = duplicate_points == 0 and expected == xyz.shape[0]

    dx, ux, wx = _spacing(xu)
    dy, uy, wy = _spacing(yu)
    dz, uz, wz = _spacing(zu)
    warnings = wx + wy + wz
    if duplicate_points:
        warnings.append(f"{duplicate_points} duplicate coordinate points detected")
    if missing_points:
        warnings.append(f"{missing_points} missing structured-grid coordinate points")
    if not structured:
        warnings.append("grid is not a complete structured grid")

    nan_count = int(np.isnan(swirl).sum())
    return GridInspection(
        label=label,
        n_points=int(x.size),
        nx=int(xu.size),
        ny=int(yu.size),
        nz=int(zu.size),
        x_range=(float(np.nanmin(x)), float(np.nanmax(x))),
        y_range=(float(np.nanmin(y)), float(np.nanmax(y))),
        z_range=(float(np.nanmin(z)), float(np.nanmax(z))),
        dx=dx,
        dy=dy,
        dz=dz,
        dx_over_d=float(dx / rotor_diameter),
        dy_over_d=float(dy / rotor_diameter),
        dz_over_d=float(dz / rotor_diameter),
        structured=structured,
        uniform=bool(ux and uy and uz),
        nan_count=nan_count,
        nan_percent=float(100.0 * nan_count / max(swirl.size, 1)),
        duplicate_points=duplicate_points,
        missing_points=missing_points,
        warnings=warnings,
    )


def reshape_structured_grid(
    field: FieldTable,
    label: str,
    rotor_diameter: float,
    u_inf: float,
    coordinate_unit: str,
) -> StructuredField:
    scale = 0.001 if coordinate_unit == "mm" else 1.0
    x = field.x * scale
    y = field.y * scale
    z = field.z * scale
    xu = _unique_sorted(x)
    yu = _unique_sorted(y)
    zu = _unique_sorted(z)
    shape = (xu.size, yu.size, zu.size)
    values = np.full(shape, np.nan, dtype=np.float64)
    u_values = np.full(shape, np.nan, dtype=np.float64) if "u" in field.column_map else None
    xi = np.searchsorted(xu, x)
    yi = np.searchsorted(yu, y)
    zi = np.searchsorted(zu, z)
    swirl_star = nondimensionalize_swirl_squared(
        field.swirl_squared.astype(np.float64), rotor_diameter, u_inf
    )
    values[xi, yi, zi] = swirl_star
    if u_values is not None:
        u_values[xi, yi, zi] = field.columns[field.column_map["u"]] / u_inf

    reconstructed = values[xi, yi, zi]
    finite = np.isfinite(swirl_star)
    if not np.allclose(reconstructed[finite], swirl_star[finite], equal_nan=True):
        raise ValueError(f"Structured-grid reshape check failed for {label}")
    return StructuredField(label=label, x=xu, y=yu, z=zu, swirl_star=values, u=u_values)


def _grid_spacing_score(grid: StructuredField) -> float:
    spacings = []
    for axis in (grid.x, grid.y, grid.z):
        diffs = np.diff(axis)
        spacings.append(float(np.nanmedian(diffs[diffs > 0])))
    return float(np.nanmean(spacings))


def _linear_indices(values: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    upper = np.searchsorted(values, points, side="right")
    upper = np.clip(upper, 1, values.size - 1)
    lower = upper - 1
    denom = values[upper] - values[lower]
    weight = np.divide(points - values[lower], denom, out=np.zeros_like(points), where=denom != 0)
    inside = (points >= values[0]) & (points <= values[-1])
    return lower, upper, np.where(inside, weight, np.nan)


def interpolate_to_common_grid(
    source: StructuredField, target: StructuredField
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate source swirl onto target coordinates over the shared domain."""

    x_mask = (target.x >= source.x[0]) & (target.x <= source.x[-1])
    y_mask = (target.y >= source.y[0]) & (target.y <= source.y[-1])
    z_mask = (target.z >= source.z[0]) & (target.z <= source.z[-1])
    tx = target.x[x_mask]
    ty = target.y[y_mask]
    tz = target.z[z_mask]
    if tx.size == 0 or ty.size == 0 or tz.size == 0:
        raise ValueError("No overlapping grid points for interpolation")

    xi0, xi1, xw = _linear_indices(source.x, tx)
    yi0, yi1, yw = _linear_indices(source.y, ty)
    zi0, zi1, zw = _linear_indices(source.z, tz)
    output = np.empty((tx.size, ty.size, tz.size), dtype=np.float64)

    for i, (a0, a1, wa) in enumerate(zip(xi0, xi1, xw)):
        c00 = (1 - wa) * source.swirl_star[a0, :, :] + wa * source.swirl_star[a1, :, :]
        for j, (b0, b1, wb) in enumerate(zip(yi0, yi1, yw)):
            c0 = (1 - wb) * c00[b0, :] + wb * c00[b1, :]
            output[i, j, :] = (1 - zw) * c0[zi0] + zw * c0[zi1]

    target_subset = target.swirl_star[np.ix_(x_mask, y_mask, z_mask)]
    return output, target_subset


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    aa = a[mask] - np.mean(a[mask])
    bb = b[mask] - np.mean(b[mask])
    denom = np.sqrt(np.sum(aa * aa) * np.sum(bb * bb))
    return float(np.sum(aa * bb) / denom) if denom > 0 else float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return _pearson(_rankdata(a[mask]), _rankdata(b[mask]))


def compare_swirl_fields(
    fine_on_coarse: np.ndarray, coarse: np.ndarray
) -> dict[str, float]:
    mask = np.isfinite(fine_on_coarse) & np.isfinite(coarse)
    if not np.any(mask):
        raise ValueError("No finite overlapping values to compare")
    a = fine_on_coarse[mask]
    b = coarse[mask]
    diff = b - a
    rms = float(np.sqrt(np.mean(diff * diff)))
    reference_rms = float(np.sqrt(np.mean(a * a)))
    metrics = {
        "overlap_points": int(mask.sum()),
        "mean_absolute_error": float(np.mean(np.abs(diff))),
        "rms_error": rms,
        "normalized_rms_error": rms / reference_rms if reference_rms > 0 else float("nan"),
        "maximum_absolute_error": float(np.max(np.abs(diff))),
        "pearson_correlation": _pearson(a, b),
        "spearman_correlation": _spearman(a, b),
        "relative_difference_maximum": _relative_difference(np.nanmax(b), np.nanmax(a)),
    }
    for percentile in (90.0, 95.0, 97.5, 99.0):
        metrics[f"relative_difference_p{percentile:g}"] = _relative_difference(
            float(np.nanpercentile(b, percentile)),
            float(np.nanpercentile(a, percentile)),
        )
    return metrics


def _relative_difference(value: float, reference: float) -> float:
    return (value - reference) / reference if reference != 0 else float("nan")


def plot_histograms(a: StructuredField, b: StructuredField, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(a.swirl_star[np.isfinite(a.swirl_star)].ravel(), bins=80, alpha=0.55, label=a.label)
    ax.hist(b.swirl_star[np.isfinite(b.swirl_star)].ravel(), bins=80, alpha=0.55, label=b.label)
    ax.set_xlabel(r"$\lambda_{ci}^{2*}$")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _nearest(values: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(values - value)))


def _imshow_xz(ax: plt.Axes, grid: StructuredField, data_xz: np.ndarray, title: str) -> None:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(
        np.ma.masked_invalid(data_xz.T),
        origin="lower",
        aspect="auto",
        extent=(grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]),
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    plt.colorbar(image, ax=ax)


def plot_centre_plane(
    a: StructuredField,
    b: StructuredField,
    fine_on_coarse: np.ndarray,
    coarse_overlap: np.ndarray,
    output: Path,
    y_value: float = 0.0,
) -> None:
    ay = _nearest(a.y, y_value)
    by = _nearest(b.y, y_value)
    oy = _nearest(b.y[(b.y >= a.y[0]) & (b.y <= a.y[-1])], y_value)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    _imshow_xz(axes[0], a, a.swirl_star[:, ay, :], f"{a.label}: y={a.y[ay]:.4g} m")
    _imshow_xz(axes[1], b, b.swirl_star[:, by, :], f"{b.label}: y={b.y[by]:.4g} m")
    diff_grid = StructuredField(
        label=b.label,
        x=b.x[(b.x >= a.x[0]) & (b.x <= a.x[-1])],
        y=b.y[(b.y >= a.y[0]) & (b.y <= a.y[-1])],
        z=b.z[(b.z >= a.z[0]) & (b.z <= a.z[-1])],
        swirl_star=coarse_overlap,
    )
    _imshow_xz(
        axes[2],
        diff_grid,
        coarse_overlap[:, oy, :] - fine_on_coarse[:, oy, :],
        f"{b.label} - {a.label}",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_y_projection(
    a: StructuredField,
    b: StructuredField,
    fine_on_coarse: np.ndarray,
    coarse_overlap: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    _imshow_xz(axes[0, 0], a, np.nanmax(a.swirl_star, axis=1), f"{a.label} max over y")
    _imshow_xz(axes[0, 1], b, np.nanmax(b.swirl_star, axis=1), f"{b.label} max over y")
    diff_grid = StructuredField(
        label=b.label,
        x=b.x[(b.x >= a.x[0]) & (b.x <= a.x[-1])],
        y=b.y[(b.y >= a.y[0]) & (b.y <= a.y[-1])],
        z=b.z[(b.z >= a.z[0]) & (b.z <= a.z[-1])],
        swirl_star=coarse_overlap,
    )
    _imshow_xz(
        axes[0, 2],
        diff_grid,
        np.nanmax(coarse_overlap, axis=1) - np.nanmax(fine_on_coarse, axis=1),
        "max projection difference",
    )

    a_int = np.trapz(a.swirl_star, a.y, axis=1)
    b_int = np.trapz(b.swirl_star, b.y, axis=1)
    _imshow_xz(axes[1, 0], a, a_int, f"{a.label} integral over y")
    _imshow_xz(axes[1, 1], b, b_int, f"{b.label} integral over y")
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_u_centre_plane(a: StructuredField, b: StructuredField, output: Path) -> None:
    if a.u is None or b.u is None:
        return
    ay = _nearest(a.y, 0.0)
    by = _nearest(b.y, 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _imshow_xz(axes[0], a, a.u[:, ay, :], f"{a.label}: u/U_inf")
    _imshow_xz(axes[1], b, b.u[:, by, :], f"{b.label}: u/U_inf")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_vortex_core_overview(
    grid: StructuredField,
    components: list[Component],
    threshold_label: str,
    threshold: float,
    output: Path,
) -> None:
    y_index = _nearest(grid.y, 0.0)
    projection = np.full((grid.x.size, grid.z.size), np.nan, dtype=np.float64)
    for i in range(grid.x.size):
        for k in range(grid.z.size):
            column = grid.swirl_star[i, :, k]
            if np.isfinite(column).any():
                projection[i, k] = np.nanmax(column)
    centre = grid.swirl_star[:, y_index, :]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#d9d9d9")

    for ax, data, title in (
        (
            axes[0],
            projection,
            rf"max over y, $\lambda_{{ci}}^{{2*}} \geq$ {threshold_label}",
        ),
        (axes[1], centre, f"centre plane y={grid.y[y_index]:.4g} m"),
    ):
        image = ax.imshow(
            np.ma.masked_invalid(data.T),
            origin="lower",
            aspect="auto",
            extent=(grid.x[0], grid.x[-1], grid.z[0], grid.z[-1]),
            cmap=cmap,
        )
        ax.contour(
            grid.x,
            grid.z,
            data.T,
            levels=[threshold],
            colors="cyan",
            linewidths=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        plt.colorbar(image, ax=ax, label=r"$\lambda_{ci}^{2*}$")

    dominant = sorted(components, key=lambda item: item.integrated, reverse=True)[:20]
    for component in dominant:
        for ax in axes:
            ax.scatter(
                component.weighted_centroid_x,
                component.weighted_centroid_z,
                s=max(25.0, min(180.0, component.size_cells * 3.0)),
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
            )
            ax.scatter(
                component.peak_x,
                component.peak_z,
                s=18,
                marker="x",
                c="cyan",
                linewidths=1.2,
            )

    fig.suptitle(
        (
            f"{grid.label}: detected vortex-core candidates "
            f"({len(components)} components, top 20 shown)"
        )
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _thresholds(values: np.ndarray) -> list[tuple[str, float]]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    maximum = float(np.nanmax(finite)) if finite.size else float("nan")
    thresholds = [(f"{int(frac * 100)}pct_max", frac * maximum) for frac in (0.2, 0.3, 0.4)]
    thresholds.extend(
        [(f"p{p:g}", float(np.nanpercentile(finite, p))) for p in (90.0, 95.0, 97.5, 99.0)]
    )
    return thresholds


def threshold_value(values: np.ndarray, threshold: str) -> tuple[str, float]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise ValueError("No positive finite lambda_ci^2 values are available")
    if threshold.endswith("pct-max"):
        fraction = float(threshold.removesuffix("pct-max")) / 100.0
        return threshold.replace("-", "_"), fraction * float(np.nanmax(finite))
    if threshold.startswith("p"):
        percentile = float(threshold[1:])
        return threshold, float(np.nanpercentile(finite, percentile))
    raise ValueError("threshold must look like '40pct-max' or 'p99'")


def _touches_boundary(indices: np.ndarray, shape: tuple[int, int, int]) -> bool:
    return bool(
        np.any(indices[:, 0] == 0)
        or np.any(indices[:, 0] == shape[0] - 1)
        or np.any(indices[:, 1] == 0)
        or np.any(indices[:, 1] == shape[1] - 1)
        or np.any(indices[:, 2] == 0)
        or np.any(indices[:, 2] == shape[2] - 1)
    )


def label_vortex_components(
    grid: StructuredField,
    threshold: float,
    threshold_label: str,
    min_component_size: int,
    remove_boundary_components: bool,
) -> list[Component]:
    mask = np.isfinite(grid.swirl_star) & (grid.swirl_star >= threshold)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[Component] = []
    component_id = 0
    spacings = [
        float(np.nanmedian(np.diff(axis))) if axis.size > 1 else 0.0
        for axis in (grid.x, grid.y, grid.z)
    ]
    cell_volume = float(np.prod([spacing for spacing in spacings if spacing > 0]))
    typical_spacing = float(np.nanmean([spacing for spacing in spacings if spacing > 0]))

    for start in zip(*np.where(mask & ~visited)):
        if visited[start]:
            continue
        queue: deque[tuple[int, int, int]] = deque([start])
        visited[start] = True
        cells: list[tuple[int, int, int]] = []
        while queue:
            cell = queue.popleft()
            cells.append(cell)
            i, j, k = cell
            for neighbour in (
                (i - 1, j, k),
                (i + 1, j, k),
                (i, j - 1, k),
                (i, j + 1, k),
                (i, j, k - 1),
                (i, j, k + 1),
            ):
                ni, nj, nk = neighbour
                if (
                    0 <= ni < mask.shape[0]
                    and 0 <= nj < mask.shape[1]
                    and 0 <= nk < mask.shape[2]
                    and mask[neighbour]
                    and not visited[neighbour]
                ):
                    visited[neighbour] = True
                    queue.append(neighbour)

        indices = np.asarray(cells, dtype=int)
        touches = _touches_boundary(indices, mask.shape)
        if indices.shape[0] < min_component_size:
            continue
        if remove_boundary_components and touches:
            continue

        weights = grid.swirl_star[indices[:, 0], indices[:, 1], indices[:, 2]]
        xs = grid.x[indices[:, 0]]
        ys = grid.y[indices[:, 1]]
        zs = grid.z[indices[:, 2]]
        peak_index = int(np.nanargmax(weights))
        total_weight = float(np.nansum(weights))
        if total_weight > 0:
            weighted = (
                float(np.nansum(weights * xs) / total_weight),
                float(np.nansum(weights * ys) / total_weight),
                float(np.nansum(weights * zs) / total_weight),
            )
        else:
            weighted = (float(np.nanmean(xs)), float(np.nanmean(ys)), float(np.nanmean(zs)))
        equivalent_volume = indices.shape[0] * cell_volume
        core_diameter = (
            (6.0 * equivalent_volume / np.pi) ** (1.0 / 3.0)
            if equivalent_volume > 0
            else float("nan")
        )
        components.append(
            Component(
                file_label=grid.label,
                threshold_label=threshold_label,
                component_id=component_id,
                size_cells=int(indices.shape[0]),
                centroid_x=float(np.nanmean(xs)),
                centroid_y=float(np.nanmean(ys)),
                centroid_z=float(np.nanmean(zs)),
                weighted_centroid_x=weighted[0],
                weighted_centroid_y=weighted[1],
                weighted_centroid_z=weighted[2],
                peak_x=float(xs[peak_index]),
                peak_y=float(ys[peak_index]),
                peak_z=float(zs[peak_index]),
                peak=float(np.nanmax(weights)),
                integrated=total_weight * cell_volume,
                bbox_x_min=float(np.nanmin(xs)),
                bbox_x_max=float(np.nanmax(xs)),
                bbox_y_min=float(np.nanmin(ys)),
                bbox_y_max=float(np.nanmax(ys)),
                bbox_z_min=float(np.nanmin(zs)),
                bbox_z_max=float(np.nanmax(zs)),
                core_diameter_m=float(core_diameter),
                cells_across_core=float(core_diameter / typical_spacing)
                if typical_spacing > 0
                else float("nan"),
                touches_boundary=touches,
            )
        )
        component_id += 1
    return components


def _touches_plane_boundary(indices: np.ndarray, shape: tuple[int, int]) -> bool:
    return bool(
        np.any(indices[:, 0] == 0)
        or np.any(indices[:, 0] == shape[0] - 1)
        or np.any(indices[:, 1] == 0)
        or np.any(indices[:, 1] == shape[1] - 1)
    )


def label_plane_components(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    min_component_size: int,
    remove_boundary_components: bool,
) -> list[PlaneComponent]:
    mask = np.isfinite(values) & (values >= threshold)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[PlaneComponent] = []
    component_id = 0
    dx = float(np.nanmedian(np.diff(x))) if x.size > 1 else 1.0
    dy = float(np.nanmedian(np.diff(y))) if y.size > 1 else 1.0
    cell_area = dx * dy

    for start in zip(*np.where(mask & ~visited)):
        if visited[start]:
            continue
        queue: deque[tuple[int, int]] = deque([start])
        visited[start] = True
        cells: list[tuple[int, int]] = []
        while queue:
            i, j = queue.popleft()
            cells.append((i, j))
            for neighbour in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                ni, nj = neighbour
                if (
                    0 <= ni < mask.shape[0]
                    and 0 <= nj < mask.shape[1]
                    and mask[neighbour]
                    and not visited[neighbour]
                ):
                    visited[neighbour] = True
                    queue.append(neighbour)

        indices = np.asarray(cells, dtype=int)
        touches = _touches_plane_boundary(indices, mask.shape)
        if indices.shape[0] < min_component_size:
            continue
        if remove_boundary_components and touches:
            continue

        weights = values[indices[:, 0], indices[:, 1]]
        xs = x[indices[:, 1]]
        ys = y[indices[:, 0]]
        peak_index = int(np.nanargmax(weights))
        total_weight = float(np.nansum(weights))
        if total_weight > 0:
            weighted_x = float(np.nansum(weights * xs) / total_weight)
            weighted_y = float(np.nansum(weights * ys) / total_weight)
        else:
            weighted_x = float(np.nanmean(xs))
            weighted_y = float(np.nanmean(ys))
        components.append(
            PlaneComponent(
                component_id=component_id,
                size_cells=int(indices.shape[0]),
                weighted_centroid_x=weighted_x,
                weighted_centroid_y=weighted_y,
                peak_x=float(xs[peak_index]),
                peak_y=float(ys[peak_index]),
                peak=float(np.nanmax(weights)),
                integrated=total_weight * cell_area,
                touches_boundary=touches,
            )
        )
        component_id += 1
    return components


def threshold_sensitivity(
    grid: StructuredField,
    min_component_size: int,
    remove_boundary_components: bool,
) -> list[Component]:
    components: list[Component] = []
    for label, threshold in _thresholds(grid.swirl_star):
        print(f"{grid.label}: threshold {label} = {threshold:.6g}", flush=True)
        components.extend(
            label_vortex_components(
                grid,
                threshold,
                label,
                min_component_size,
                remove_boundary_components,
            )
        )
    return components


def compare_components(
    components_a: list[Component],
    components_b: list[Component],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    by_threshold_a = _components_by_threshold(components_a)
    by_threshold_b = _components_by_threshold(components_b)
    for threshold in sorted(set(by_threshold_a) & set(by_threshold_b)):
        a_sorted = sorted(by_threshold_a[threshold], key=lambda c: c.integrated, reverse=True)
        b_available = sorted(by_threshold_b[threshold], key=lambda c: c.integrated, reverse=True)
        for comp_a in a_sorted:
            if not b_available:
                break
            distances = [
                _centroid_distance(comp_a, comp_b)
                for comp_b in b_available
            ]
            best_index = int(np.nanargmin(distances))
            comp_b = b_available.pop(best_index)
            rows.append(
                {
                    "threshold": threshold,
                    "component_a": comp_a.component_id,
                    "component_b": comp_b.component_id,
                    "centroid_displacement_m": distances[best_index],
                    "size_ratio_b_over_a": comp_b.size_cells / comp_a.size_cells,
                    "peak_ratio_b_over_a": _ratio(comp_b.peak, comp_a.peak),
                    "integrated_ratio_b_over_a": _ratio(comp_b.integrated, comp_a.integrated),
                    "cells_across_core_a": comp_a.cells_across_core,
                    "cells_across_core_b": comp_b.cells_across_core,
                }
            )
    return rows


def _components_by_threshold(components: Iterable[Component]) -> dict[str, list[Component]]:
    grouped: dict[str, list[Component]] = {}
    for component in components:
        grouped.setdefault(component.threshold_label, []).append(component)
    return grouped


def _centroid_distance(a: Component, b: Component) -> float:
    return float(
        np.sqrt(
            (a.weighted_centroid_x - b.weighted_centroid_x) ** 2
            + (a.weighted_centroid_y - b.weighted_centroid_y) ** 2
            + (a.weighted_centroid_z - b.weighted_centroid_z) ** 2
        )
    )


def _ratio(value: float, reference: float) -> float:
    return value / reference if reference != 0 else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _inspection_rows(inspections: list[GridInspection]) -> list[dict[str, object]]:
    rows = []
    for item in inspections:
        rows.append(
            {
                "label": item.label,
                "n_points": item.n_points,
                "nx": item.nx,
                "ny": item.ny,
                "nz": item.nz,
                "x_min_m": item.x_range[0],
                "x_max_m": item.x_range[1],
                "y_min_m": item.y_range[0],
                "y_max_m": item.y_range[1],
                "z_min_m": item.z_range[0],
                "z_max_m": item.z_range[1],
                "dx_m": item.dx,
                "dy_m": item.dy,
                "dz_m": item.dz,
                "dx_over_d": item.dx_over_d,
                "dy_over_d": item.dy_over_d,
                "dz_over_d": item.dz_over_d,
                "structured": item.structured,
                "uniform": item.uniform,
                "nan_count": item.nan_count,
                "nan_percent": item.nan_percent,
                "duplicate_points": item.duplicate_points,
                "missing_points": item.missing_points,
                "warnings": "; ".join(item.warnings),
            }
        )
    return rows


def _component_rows(components: list[Component]) -> list[dict[str, object]]:
    return [component.__dict__ for component in components]


def _recommendation(
    coarse_inspection: GridInspection,
    metrics: dict[str, float],
    matches: list[dict[str, float | int | str]],
    components_b: list[Component],
    rotor_diameter: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    score = 0
    max_spacing = max(coarse_inspection.dx_over_d, coarse_inspection.dy_over_d, coarse_inspection.dz_over_d)
    if max_spacing > 0.08:
        score += 2
        reasons.append(f"coarse grid spacing is large: max(dx,dy,dz)/D={max_spacing:.3g}")
    elif max_spacing > 0.04:
        score += 1
        reasons.append(f"coarse grid spacing is moderate: max(dx,dy,dz)/D={max_spacing:.3g}")

    nrmse = metrics.get("normalized_rms_error", float("nan"))
    correlation = metrics.get("pearson_correlation", float("nan"))
    p99 = abs(metrics.get("relative_difference_p99", float("nan")))
    if np.isfinite(nrmse) and nrmse > 0.35:
        score += 2
        reasons.append(f"normalized RMS error is high: {nrmse:.3g}")
    elif np.isfinite(nrmse) and nrmse > 0.2:
        score += 1
        reasons.append(f"normalized RMS error is moderate: {nrmse:.3g}")
    if np.isfinite(correlation) and correlation < 0.75:
        score += 2
        reasons.append(f"Pearson correlation is low: {correlation:.3g}")
    elif np.isfinite(correlation) and correlation < 0.9:
        score += 1
        reasons.append(f"Pearson correlation is moderate: {correlation:.3g}")
    if np.isfinite(p99) and p99 > 0.35:
        score += 1
        reasons.append(f"99th percentile differs by {p99:.1%}")

    displacements = [
        float(row["centroid_displacement_m"])
        for row in matches
        if np.isfinite(float(row["centroid_displacement_m"]))
    ]
    if displacements and np.nanmedian(displacements) > 0.05 * rotor_diameter:
        score += 1
        reasons.append(
            f"median matched-centroid displacement is {np.nanmedian(displacements):.3g} m"
        )

    core_cells = [c.cells_across_core for c in components_b if np.isfinite(c.cells_across_core)]
    if core_cells and np.nanmedian(core_cells) < 4.0:
        score += 2
        reasons.append(
            f"median detected core spans only {np.nanmedian(core_cells):.2g} cells"
        )
    elif core_cells and np.nanmedian(core_cells) < 6.0:
        score += 1
        reasons.append(
            f"median detected core spans {np.nanmedian(core_cells):.2g} cells"
        )

    if not reasons:
        reasons.append("main grid, field-comparison, and component metrics look stable")

    if score >= 5:
        return "Export B is too coarse for vortex detection", reasons
    if score >= 2:
        return "Export B is suitable only for qualitative visualization", reasons
    return "Export B is suitable for quantitative vortex detection", reasons


def write_summary_report(
    output: Path,
    inspections: list[GridInspection],
    metrics: dict[str, float],
    matches: list[dict[str, float | int | str]],
    components_b: list[Component],
    rotor_diameter: float,
) -> None:
    recommendation, reasons = _recommendation(
        inspections[1], metrics, matches, components_b, rotor_diameter
    )
    report = {
        "recommendation": recommendation,
        "reasons": reasons,
        "comparison_metrics": metrics,
        "grid_quality": [row for row in _inspection_rows(inspections)],
        "component_matches": matches[:20],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output.with_suffix(".txt")).write_text(
        recommendation + "\n\n" + "\n".join(f"- {reason}" for reason in reasons) + "\n",
        encoding="utf-8",
    )


def write_detection_report(
    output: Path,
    inspection: GridInspection,
    components: list[Component],
    threshold_label: str,
    threshold: float,
    source_url: str,
) -> None:
    dominant = sorted(components, key=lambda item: item.integrated, reverse=True)[:10]
    report = {
        "methodology_reference": source_url,
        "method": (
            "Use exported squared swirling strength as the vortex detector, "
            "following the swirling-strength criterion idea that vortex cores "
            "correspond to regions with nonzero lambda_ci. This script does not "
            "recompute lambda_ci from velocity gradients."
        ),
        "threshold_label": threshold_label,
        "threshold_value": threshold,
        "grid_quality": _inspection_rows([inspection])[0],
        "n_components": len(components),
        "dominant_components": [component.__dict__ for component in dominant],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_detection_only(args: argparse.Namespace) -> None:
    source_url = "https://guilindner.github.io/VortexFitting/methodology.html#swirling-strength-criterion"
    field = load_field_file(
        args.file_a,
        frame_index=args.frame,
        swirl_variable=args.swirl_variable,
        swirl_sign=args.swirl_sign,
        hole_mask=args.hole_mask,
    )
    inspection = inspect_grid(
        field, args.label_a, args.rotor_diameter, args.coordinate_unit
    )
    print(
        f"{inspection.label}: {inspection.n_points} points, "
        f"grid={inspection.nx}x{inspection.ny}x{inspection.nz}, "
        f"structured={inspection.structured}, uniform={inspection.uniform}",
        flush=True,
    )
    for warning in inspection.warnings:
        print(f"  warning: {warning}", flush=True)
    if not inspection.structured:
        raise SystemExit("The input must be a complete structured grid for vortex detection.")

    grid = reshape_structured_grid(
        field, args.label_a, args.rotor_diameter, args.u_inf, args.coordinate_unit
    )
    all_components = threshold_sensitivity(
        grid, args.min_component_size, args.remove_boundary_components
    )
    write_csv(args.output_folder / "grid_quality.csv", _inspection_rows([inspection]))
    write_csv(args.output_folder / "detected_components.csv", _component_rows(all_components))

    threshold_label, threshold = threshold_value(grid.swirl_star, args.core_threshold)
    overview_components = label_vortex_components(
        grid,
        threshold,
        threshold_label,
        args.min_component_size,
        args.remove_boundary_components,
    )
    plot_vortex_core_overview(
        grid,
        overview_components,
        threshold_label,
        threshold,
        args.output_folder / "vortex_core_overview.png",
    )
    write_detection_report(
        args.output_folder / "vortex_core_report.json",
        inspection,
        overview_components,
        threshold_label,
        threshold,
        source_url,
    )
    print(
        f"Saved vortex-core detection outputs to: {args.output_folder.resolve()}",
        flush=True,
    )


def _read_swirl_z_plane(
    h5: h5py.File,
    variable: str,
    frame_index: int,
    z_index: int,
    rotor_diameter: float,
    u_inf: float,
    swirl_sign: str,
    hole_mask: str,
) -> np.ndarray:
    raw = h5[variable]
    if raw.ndim == 4:
        plane = raw[frame_index, z_index, :, :].astype(np.float64)
    elif raw.ndim == 3:
        plane = raw[z_index, :, :].astype(np.float64)
    else:
        raise ValueError(f"Expected {variable!r} to be 3D or 4D, got {raw.shape}")
    values = nondimensionalize_swirl_squared(
        _apply_swirl_sign(plane, swirl_sign), rotor_diameter, u_inf
    )
    if hole_mask == "none":
        return values
    if not all(name in h5 for name in ("u", "v", "w")):
        raise ValueError(f"--hole-mask {hole_mask} requires u, v, and w")
    if raw.ndim == 4:
        u = h5["u"][frame_index, z_index, :, :]
        v = h5["v"][frame_index, z_index, :, :]
        w = h5["w"][frame_index, z_index, :, :]
    else:
        u = h5["u"][z_index, :, :]
        v = h5["v"][z_index, :, :]
        w = h5["w"][z_index, :, :]
    if hole_mask == "velocity-zero":
        holes = (u == 0.0) & (v == 0.0) & (w == 0.0)
    elif hole_mask == "velocity-nan":
        holes = ~(np.isfinite(u) & np.isfinite(v) & np.isfinite(w))
    elif hole_mask == "velocity-zero-or-nan":
        holes = ((u == 0.0) & (v == 0.0) & (w == 0.0)) | ~(
            np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
        )
    else:
        raise ValueError(f"Unsupported hole mask: {hole_mask}")
    values = values.copy()
    values[holes] = np.nan
    return values


def _open_vortex_file_view(
    path: Path,
    label: str,
    swirl_variable: str | None,
    coordinate_unit: str,
) -> VortexFileView:
    h5 = h5py.File(path, "r")
    try:
        variable = swirl_variable
        if variable is None:
            variable = infer_columns(list(h5.keys()))["lambda_ci_squared"]
        if variable not in h5:
            raise ValueError(
                f"Variable {variable!r} not found in {path}. Available variables: "
                f"{', '.join(h5.keys())}"
            )
        for coord in ("x", "y", "z"):
            if coord not in h5:
                raise ValueError(f"Coordinate {coord!r} not found in {path}")
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        if coordinate_unit == "mm":
            x = x * 0.001
            y = y * 0.001
            z = z * 0.001
        raw = h5[variable]
        n_frames = raw.shape[0] if raw.ndim == 4 else 1
        return VortexFileView(
            path=path,
            label=label,
            h5=h5,
            variable=variable,
            x=x,
            y=y,
            z=z,
            n_frames=n_frames,
        )
    except Exception:
        h5.close()
        raise


def _close_vortex_file_views(views: Iterable[VortexFileView]) -> None:
    for view in views:
        view.h5.close()


def inspect_vortex_z_plane(args: argparse.Namespace) -> None:
    path = args.file_a
    with h5py.File(path, "r") as h5:
        variable = args.swirl_variable
        if variable is None:
            variable = infer_columns(list(h5.keys()))["lambda_ci_squared"]
        if variable not in h5:
            raise SystemExit(
                f"Variable {variable!r} not found. Available variables: {', '.join(h5.keys())}"
            )
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        if args.coordinate_unit == "mm":
            x = x * 0.001
            y = y * 0.001
            z = z * 0.001
        raw = h5[variable]
        n_frames = raw.shape[0] if raw.ndim == 4 else 1
        frame_index = int(np.clip(args.frame, 0, n_frames - 1))
        z_index = int(np.nanargmin(np.abs(z - args.z_value)))

        first = _read_swirl_z_plane(
            h5,
            variable,
            frame_index,
            z_index,
            args.rotor_diameter,
            args.u_inf,
            args.swirl_sign,
            args.hole_mask,
        )
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#d9d9d9")
        fig = plt.figure(figsize=(13, 8), constrained_layout=False)
        ax = fig.add_axes((0.07, 0.22, 0.62, 0.70))
        text_ax = fig.add_axes((0.73, 0.22, 0.24, 0.70))
        frame_ax = fig.add_axes((0.20, 0.13, 0.44, 0.03))
        z_ax = fig.add_axes((0.20, 0.085, 0.44, 0.03))
        threshold_ax = fig.add_axes((0.20, 0.04, 0.44, 0.03))

        image = ax.imshow(
            np.ma.masked_invalid(first),
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], y[0], y[-1]),
            cmap=cmap,
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(r"$\lambda_{ci}^{2*}$")
        contour_holder: list[object] = []
        centroid_artist = ax.scatter([], [], s=70, facecolors="none", edgecolors="white", linewidths=1.3)
        peak_artist = ax.scatter([], [], s=28, marker="x", c="cyan", linewidths=1.3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
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
            frame_ax, "Frame", 0, n_frames - 1, valinit=frame_index, valstep=1
        )
        z_slider = Slider(z_ax, "z index", 0, z.size - 1, valinit=z_index, valstep=1)
        threshold_slider = Slider(
            threshold_ax,
            "percentile",
            80.0,
            99.9,
            valinit=float(args.inspect_percentile),
            valstep=0.1,
        )

        def refresh() -> None:
            nonlocal contour_holder
            frame = int(frame_slider.val)
            zi = int(z_slider.val)
            plane = _read_swirl_z_plane(
                h5,
                variable,
                frame,
                zi,
                args.rotor_diameter,
                args.u_inf,
                args.swirl_sign,
                args.hole_mask,
            )
            finite_positive = plane[np.isfinite(plane) & (plane > 0.0)]
            threshold = (
                float(np.nanpercentile(finite_positive, float(threshold_slider.val)))
                if finite_positive.size
                else float("nan")
            )
            components = (
                label_plane_components(
                    plane,
                    x,
                    y,
                    threshold,
                    args.min_component_size,
                    args.remove_boundary_components,
                )
                if np.isfinite(threshold)
                else []
            )
            image.set_data(np.ma.masked_invalid(plane))
            if np.isfinite(plane).any():
                image.set_clim(0.0, float(np.nanpercentile(plane[np.isfinite(plane)], 99.5)))
            for contour in contour_holder:
                contour.remove()
            contour_holder = []
            if np.isfinite(threshold) and np.nanmax(plane) >= threshold:
                contour_set = ax.contour(
                    x,
                    y,
                    plane,
                    levels=[threshold],
                    colors="cyan",
                    linewidths=0.8,
                )
                contour_holder = [contour_set]
            dominant = sorted(components, key=lambda item: item.integrated, reverse=True)[:30]
            if dominant:
                centroid_artist.set_offsets(
                    np.column_stack(
                        (
                            [component.weighted_centroid_x for component in dominant],
                            [component.weighted_centroid_y for component in dominant],
                        )
                    )
                )
                peak_artist.set_offsets(
                    np.column_stack(
                        (
                            [component.peak_x for component in dominant],
                            [component.peak_y for component in dominant],
                        )
                    )
                )
            else:
                centroid_artist.set_offsets(np.empty((0, 2)))
                peak_artist.set_offsets(np.empty((0, 2)))
            ax.set_title(
                f"{path.name}: frame={frame}, z={z[zi]:.6g} m, "
                f"threshold=p{threshold_slider.val:.1f}"
            )
            largest = max((component.size_cells for component in components), default=0)
            info.set_text(
                "\n".join(
                    [
                        "Vortex z-plane inspector",
                        f"source: {path.name}",
                        f"variable: {variable}",
                        f"swirl_sign: {args.swirl_sign}",
                        f"hole_mask: {args.hole_mask}",
                        f"frame: {frame} / {n_frames - 1}",
                        f"z index: {zi} / {z.size - 1}",
                        f"z: {z[zi]:.9g} m",
                        f"threshold: {threshold:.9g}",
                        f"components: {len(components)}",
                        f"largest size: {largest} cells",
                        "",
                        "white circles: weighted centroids",
                        "cyan x: local peak in component",
                        "gray: masked/NaN cells",
                    ]
                )
            )
            fig.canvas.draw_idle()

        frame_slider.on_changed(lambda _value: refresh())
        z_slider.on_changed(lambda _value: refresh())
        threshold_slider.on_changed(lambda _value: refresh())
        print(f"Reading vortex field: {path.resolve()}")
        print(
            f"Using variable={variable}, swirl_sign={args.swirl_sign}, "
            f"hole_mask={args.hole_mask}."
        )
        print("Use sliders to move through frames, z planes, and percentile threshold.")
        refresh()
        plt.show()


def inspect_vortex_z_plane_comparison(args: argparse.Namespace) -> None:
    if args.file_b is None:
        raise SystemExit("--inspect-z-compare requires file_b.")

    views = [
        _open_vortex_file_view(
            args.file_a, args.label_a, args.swirl_variable, args.coordinate_unit
        ),
        _open_vortex_file_view(
            args.file_b, args.label_b, args.swirl_variable, args.coordinate_unit
        ),
    ]
    try:
        n_frames = min(view.n_frames for view in views)
        frame_index = int(np.clip(args.frame, 0, n_frames - 1))
        z_min = max(float(view.z.min()) for view in views)
        z_max = min(float(view.z.max()) for view in views)
        if z_min >= z_max:
            raise SystemExit("The two files have no overlapping z range.")
        z_initial = float(np.clip(args.z_value, z_min, z_max))

        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#d9d9d9")
        fig = plt.figure(figsize=(15, 8), constrained_layout=False)
        axes = [fig.add_axes((0.06, 0.22, 0.39, 0.70)), fig.add_axes((0.51, 0.22, 0.39, 0.70))]
        text_ax = fig.add_axes((0.91, 0.22, 0.08, 0.70))
        frame_ax = fig.add_axes((0.20, 0.13, 0.52, 0.03))
        z_ax = fig.add_axes((0.20, 0.085, 0.52, 0.03))
        threshold_ax = fig.add_axes((0.20, 0.04, 0.52, 0.03))

        artists = []
        first_planes = []
        first_z_indices = []
        for view in views:
            zi = int(np.nanargmin(np.abs(view.z - z_initial)))
            first_z_indices.append(zi)
            first_planes.append(
                _read_swirl_z_plane(
                    view.h5,
                    view.variable,
                    frame_index,
                    zi,
                    args.rotor_diameter,
                    args.u_inf,
                    args.swirl_sign,
                    args.hole_mask,
                )
            )
        finite_positive = np.concatenate(
            [
                plane[np.isfinite(plane) & (plane > 0.0)]
                for plane in first_planes
                if np.any(np.isfinite(plane) & (plane > 0.0))
            ]
            or [np.array([1.0])]
        )
        initial_vmax = float(np.nanpercentile(finite_positive, args.color_percentile))

        for ax, view, plane in zip(axes, views, first_planes):
            image = ax.imshow(
                np.ma.masked_invalid(plane),
                origin="lower",
                aspect="auto",
                extent=(view.x[0], view.x[-1], view.y[0], view.y[-1]),
                cmap=cmap,
                vmin=0.0,
                vmax=initial_vmax,
            )
            centroid_artist = ax.scatter(
                [], [], s=70, facecolors="none", edgecolors="white", linewidths=1.3
            )
            peak_artist = ax.scatter([], [], s=28, marker="x", c="cyan", linewidths=1.3)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            artists.append(
                {
                    "image": image,
                    "centroid": centroid_artist,
                    "peak": peak_artist,
                    "contours": [],
                }
            )
        cbar = fig.colorbar(artists[1]["image"], ax=axes)
        cbar.set_label(r"$\lambda_{ci}^{2*}$")
        text_ax.axis("off")
        info = text_ax.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8,
            transform=text_ax.transAxes,
        )
        frame_slider = Slider(
            frame_ax, "Frame", 0, n_frames - 1, valinit=frame_index, valstep=1
        )
        z_slider = Slider(z_ax, "z [m]", z_min, z_max, valinit=z_initial)
        threshold_slider = Slider(
            threshold_ax,
            "percentile",
            80.0,
            99.9,
            valinit=float(args.inspect_percentile),
            valstep=0.1,
        )

        def refresh() -> None:
            frame = int(frame_slider.val)
            z_value = float(z_slider.val)
            percentile = float(threshold_slider.val)
            panels = []
            shared_values = []
            for view in views:
                zi = int(np.nanargmin(np.abs(view.z - z_value)))
                plane = _read_swirl_z_plane(
                    view.h5,
                    view.variable,
                    frame,
                    zi,
                    args.rotor_diameter,
                    args.u_inf,
                    args.swirl_sign,
                    args.hole_mask,
                )
                finite_positive = plane[np.isfinite(plane) & (plane > 0.0)]
                threshold = (
                    float(np.nanpercentile(finite_positive, percentile))
                    if finite_positive.size
                    else float("nan")
                )
                components = (
                    label_plane_components(
                        plane,
                        view.x,
                        view.y,
                        threshold,
                        args.min_component_size,
                        args.remove_boundary_components,
                    )
                    if np.isfinite(threshold)
                    else []
                )
                if finite_positive.size:
                    shared_values.append(finite_positive)
                panels.append((view, zi, plane, threshold, components))

            if args.color_vmax is not None:
                vmax = args.color_vmax
            elif shared_values:
                vmax = float(
                    np.nanpercentile(np.concatenate(shared_values), args.color_percentile)
                )
            else:
                vmax = 1.0

            info_lines = [
                "Compare z-plane",
                f"frame: {frame}",
                f"requested z: {z_value:.6g} m",
                f"threshold: p{percentile:.1f}",
                f"vmax: {vmax:.4g}",
                f"hole_mask: {args.hole_mask}",
                "",
            ]
            for ax, artist, (view, zi, plane, threshold, components) in zip(
                axes, artists, panels
            ):
                artist["image"].set_data(np.ma.masked_invalid(plane))
                artist["image"].set_clim(0.0, vmax)
                for contour in artist["contours"]:
                    contour.remove()
                artist["contours"] = []
                if np.isfinite(threshold) and np.nanmax(plane) >= threshold:
                    artist["contours"] = [
                        ax.contour(
                            view.x,
                            view.y,
                            plane,
                            levels=[threshold],
                            colors="cyan",
                            linewidths=0.8,
                        )
                    ]
                dominant = sorted(
                    components, key=lambda item: item.integrated, reverse=True
                )[:30]
                if dominant:
                    artist["centroid"].set_offsets(
                        np.column_stack(
                            (
                                [component.weighted_centroid_x for component in dominant],
                                [component.weighted_centroid_y for component in dominant],
                            )
                        )
                    )
                    artist["peak"].set_offsets(
                        np.column_stack(
                            (
                                [component.peak_x for component in dominant],
                                [component.peak_y for component in dominant],
                            )
                        )
                    )
                else:
                    artist["centroid"].set_offsets(np.empty((0, 2)))
                    artist["peak"].set_offsets(np.empty((0, 2)))
                largest = max((component.size_cells for component in components), default=0)
                ax.set_title(
                    f"{view.label}: z={view.z[zi]:.6g} m, "
                    f"{len(components)} components"
                )
                info_lines.extend(
                    [
                        view.label,
                        f"  z: {view.z[zi]:.6g}",
                        f"  threshold: {threshold:.4g}",
                        f"  components: {len(components)}",
                        f"  largest: {largest}",
                        "",
                    ]
                )
            info.set_text("\n".join(info_lines))
            fig.canvas.draw_idle()

        frame_slider.on_changed(lambda _value: refresh())
        z_slider.on_changed(lambda _value: refresh())
        threshold_slider.on_changed(lambda _value: refresh())
        print("Reading vortex comparison files:")
        for view in views:
            print(f"  {view.label}: {view.path.resolve()}")
        print(
            f"Using variable={views[0].variable}, swirl_sign={args.swirl_sign}, "
            f"hole_mask={args.hole_mask}."
        )
        print("Use sliders to compare frame, physical z plane, and percentile threshold.")
        refresh()
        plt.show()
    finally:
        _close_vortex_file_views(views)


def animate_vortex_z_plane(args: argparse.Namespace) -> Path:
    if args.save is None:
        raise SystemExit("--animate-z requires --save output.gif")
    if args.save.suffix.lower() != ".gif":
        raise SystemExit("--animate-z currently saves GIF files; use --save output.gif")

    path = args.file_a
    with h5py.File(path, "r") as h5:
        variable = args.swirl_variable
        if variable is None:
            variable = infer_columns(list(h5.keys()))["lambda_ci_squared"]
        if variable not in h5:
            raise SystemExit(
                f"Variable {variable!r} not found. Available variables: {', '.join(h5.keys())}"
            )
        x = h5["x"][:].astype(np.float64)
        y = h5["y"][:].astype(np.float64)
        z = h5["z"][:].astype(np.float64)
        if args.coordinate_unit == "mm":
            x = x * 0.001
            y = y * 0.001
            z = z * 0.001
        raw = h5[variable]
        n_frames = raw.shape[0] if raw.ndim == 4 else 1
        start = max(args.start, 0)
        stop = n_frames if args.stop is None else min(args.stop, n_frames)
        if args.step <= 0:
            raise SystemExit("--step must be positive")
        frames = list(range(start, stop, args.step))
        if not frames:
            raise SystemExit("No frames selected for animation")
        z_index = int(np.nanargmin(np.abs(z - args.z_value)))

        color_vmax = args.color_vmax
        if color_vmax is None:
            percentile_values = []
            print(
                f"Computing fixed color scale from {len(frames)} selected frames "
                f"(p{args.color_percentile:g}).",
                flush=True,
            )
            for frame in frames:
                plane = _read_swirl_z_plane(
                    h5,
                    variable,
                    frame,
                    z_index,
                    args.rotor_diameter,
                    args.u_inf,
                    args.swirl_sign,
                    args.hole_mask,
                )
                finite_positive = plane[np.isfinite(plane) & (plane > 0.0)]
                if finite_positive.size:
                    percentile_values.append(
                        float(np.nanpercentile(finite_positive, args.color_percentile))
                    )
            color_vmax = max(percentile_values) if percentile_values else 1.0

        first = _read_swirl_z_plane(
            h5,
            variable,
            frames[0],
            z_index,
            args.rotor_diameter,
            args.u_inf,
            args.swirl_sign,
            args.hole_mask,
        )
        positive = first[np.isfinite(first) & (first > 0.0)]
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("#d9d9d9")
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        image = ax.imshow(
            np.ma.masked_invalid(first),
            origin="lower",
            aspect="auto",
            extent=(x[0], x[-1], y[0], y[-1]),
            cmap=cmap,
            vmin=0.0,
            vmax=color_vmax,
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(r"$\lambda_{ci}^{2*}$")
        centroid_artist = ax.scatter(
            [], [], s=70, facecolors="none", edgecolors="white", linewidths=1.3
        )
        peak_artist = ax.scatter([], [], s=28, marker="x", c="cyan", linewidths=1.3)
        text = ax.text(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="white",
            fontsize=9,
            family="monospace",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        contour_holder: list[object] = []

        def update(frame: int) -> list[object]:
            nonlocal contour_holder
            plane = _read_swirl_z_plane(
                h5,
                variable,
                frame,
                z_index,
                args.rotor_diameter,
                args.u_inf,
                args.swirl_sign,
                args.hole_mask,
            )
            finite_positive = plane[np.isfinite(plane) & (plane > 0.0)]
            threshold = (
                float(np.nanpercentile(finite_positive, args.inspect_percentile))
                if finite_positive.size
                else float("nan")
            )
            components = (
                label_plane_components(
                    plane,
                    x,
                    y,
                    threshold,
                    args.min_component_size,
                    args.remove_boundary_components,
                )
                if np.isfinite(threshold)
                else []
            )
            image.set_data(np.ma.masked_invalid(plane))
            for contour in contour_holder:
                contour.remove()
            contour_holder = []
            if np.isfinite(threshold) and np.nanmax(plane) >= threshold:
                contour_holder = [
                    ax.contour(
                        x,
                        y,
                        plane,
                        levels=[threshold],
                        colors="cyan",
                        linewidths=0.8,
                    )
                ]
            dominant = sorted(components, key=lambda item: item.integrated, reverse=True)[:30]
            if dominant:
                centroid_artist.set_offsets(
                    np.column_stack(
                        (
                            [component.weighted_centroid_x for component in dominant],
                            [component.weighted_centroid_y for component in dominant],
                        )
                    )
                )
                peak_artist.set_offsets(
                    np.column_stack(
                        (
                            [component.peak_x for component in dominant],
                            [component.peak_y for component in dominant],
                        )
                    )
                )
            else:
                centroid_artist.set_offsets(np.empty((0, 2)))
                peak_artist.set_offsets(np.empty((0, 2)))
            largest = max((component.size_cells for component in components), default=0)
            ax.set_title(
                f"{args.label_a}: z={z[z_index]:.6g} m, "
                f"threshold=p{args.inspect_percentile:g}"
            )
            text.set_text(
                "\n".join(
                    [
                        f"frame: {frame}",
                        f"components: {len(components)}",
                        f"largest: {largest} cells",
                        f"threshold: {threshold:.4g}",
                        f"color max: {color_vmax:.4g}",
                    ]
                )
            )
            return [image, centroid_artist, peak_artist, text]

        animation = FuncAnimation(fig, update, frames=frames, interval=1000.0 / args.fps)
        args.save.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Saving {len(frames)}-frame GIF for {path.name} at z={z[z_index]:.6g} m "
            f"to {args.save.resolve()}",
            flush=True,
        )
        animation.save(args.save, writer="pillow", fps=args.fps)
        plt.close(fig)
    print(f"Saved GIF to: {args.save.resolve()}", flush=True)
    return args.save


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two table-like exports containing squared swirling strength "
            "lambda_ci^2 without recomputing it from velocity gradients."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_a", type=Path, help="reference/finer export, or single file with --detect-only")
    parser.add_argument("file_b", nargs="?", type=Path, help="coarser export")
    parser.add_argument("--label-a", default="file_A", help="display label for file A")
    parser.add_argument("--label-b", default="file_B", help="display label for file B")
    parser.add_argument("--rotor-diameter", type=float, default=1.2, help="D in metres")
    parser.add_argument("--u-inf", type=float, default=4.0, help="inflow velocity in m/s")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="time index to read when the input variable is a 4D NetCDF field",
    )
    parser.add_argument(
        "--swirl-variable",
        default=None,
        help="name of the squared swirling-strength variable; inferred if omitted",
    )
    parser.add_argument(
        "--swirl-sign",
        choices=("auto", "positive", "negative"),
        default="auto",
        help=(
            "sign convention for the stored squared swirling-strength field; "
            "use negative when the export stores -lambda_ci^2"
        ),
    )
    parser.add_argument(
        "--hole-mask",
        choices=("none", "velocity-zero", "velocity-nan", "velocity-zero-or-nan"),
        default="none",
        help="mask detector values where velocity indicates missing data",
    )
    parser.add_argument(
        "--use-davis-values",
        action="store_true",
        help=(
            "do not mask zeros or velocity holes; use the exported detector "
            "values exactly, apart from optional sign conversion and nondimensionalization"
        ),
    )
    parser.add_argument(
        "--coordinate-unit",
        choices=("m", "mm"),
        default="mm",
        help="unit used by x, y, z coordinates in the input files",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("outputs/swirl_export_comparison"),
        help="folder where plots, CSV files, and reports are written",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=5,
        help="minimum number of grid cells for a detected vortex component",
    )
    parser.add_argument(
        "--remove-boundary-components",
        action="store_true",
        help="discard detected vortex components that touch the domain boundary",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="detect and plot vortex-core candidates for file_a only",
    )
    parser.add_argument(
        "--inspect-z",
        action="store_true",
        help="open an interactive z-plane vortex detector for file_a",
    )
    parser.add_argument(
        "--inspect-z-compare",
        action="store_true",
        help="open a side-by-side interactive z-plane vortex detector for file_a and file_b",
    )
    parser.add_argument(
        "--animate-z",
        action="store_true",
        help="save a z-plane GIF with vortex detections for file_a",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=0.0,
        help="z coordinate for --inspect-z or --animate-z, after coordinate-unit conversion",
    )
    parser.add_argument(
        "--inspect-percentile",
        type=float,
        default=99.0,
        help="percentile threshold for --inspect-z or --animate-z",
    )
    parser.add_argument("--fps", type=float, default=40.0, help="GIF frame rate")
    parser.add_argument("--start", type=int, default=0, help="first frame for --animate-z")
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="one-past-last frame for --animate-z; default uses the whole series",
    )
    parser.add_argument("--step", type=int, default=1, help="frame stride for --animate-z")
    parser.add_argument("--save", type=Path, default=None, help="output GIF path for --animate-z")
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help=(
            "percentile used to choose a fixed GIF color maximum across the "
            "selected time range"
        ),
    )
    parser.add_argument(
        "--color-vmax",
        type=float,
        default=None,
        help="manual fixed colorbar maximum for --animate-z",
    )
    parser.add_argument(
        "--core-threshold",
        default="p99",
        help="threshold used for the overview plot, e.g. p99 or 40pct-max",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.use_davis_values:
        args.hole_mask = "none"
    args.output_folder.mkdir(parents=True, exist_ok=True)

    if args.inspect_z:
        inspect_vortex_z_plane(args)
        return
    if args.inspect_z_compare:
        inspect_vortex_z_plane_comparison(args)
        return
    if args.animate_z:
        animate_vortex_z_plane(args)
        return
    if args.detect_only:
        run_detection_only(args)
        return
    if args.file_b is None:
        raise SystemExit("file_b is required unless --detect-only is used.")

    field_a = load_field_file(
        args.file_a,
        frame_index=args.frame,
        swirl_variable=args.swirl_variable,
        swirl_sign=args.swirl_sign,
        hole_mask=args.hole_mask,
    )
    field_b = load_field_file(
        args.file_b,
        frame_index=args.frame,
        swirl_variable=args.swirl_variable,
        swirl_sign=args.swirl_sign,
        hole_mask=args.hole_mask,
    )
    inspections = [
        inspect_grid(field_a, args.label_a, args.rotor_diameter, args.coordinate_unit),
        inspect_grid(field_b, args.label_b, args.rotor_diameter, args.coordinate_unit),
    ]
    for inspection in inspections:
        print(
            f"{inspection.label}: {inspection.n_points} points, "
            f"grid={inspection.nx}x{inspection.ny}x{inspection.nz}, "
            f"structured={inspection.structured}, uniform={inspection.uniform}",
            flush=True,
        )
        for warning in inspection.warnings:
            print(f"  warning: {warning}", flush=True)
    write_csv(args.output_folder / "grid_quality.csv", _inspection_rows(inspections))

    if not all(inspection.structured for inspection in inspections):
        raise SystemExit("Both files must be complete structured grids for this comparison.")

    grid_a = reshape_structured_grid(
        field_a, args.label_a, args.rotor_diameter, args.u_inf, args.coordinate_unit
    )
    grid_b = reshape_structured_grid(
        field_b, args.label_b, args.rotor_diameter, args.u_inf, args.coordinate_unit
    )
    if _grid_spacing_score(grid_a) > _grid_spacing_score(grid_b):
        print("File A appears coarser than file B; interpolation still uses A as reference.", flush=True)

    fine_on_coarse, coarse_overlap = interpolate_to_common_grid(grid_a, grid_b)
    metrics = compare_swirl_fields(fine_on_coarse, coarse_overlap)
    write_csv(args.output_folder / "comparison_metrics.csv", [metrics])

    plot_histograms(grid_a, grid_b, args.output_folder / "histogram_lambda_ci_squared_star.png")
    plot_centre_plane(
        grid_a,
        grid_b,
        fine_on_coarse,
        coarse_overlap,
        args.output_folder / "centre_plane_y0.png",
    )
    plot_y_projection(
        grid_a,
        grid_b,
        fine_on_coarse,
        coarse_overlap,
        args.output_folder / "y_projections.png",
    )
    plot_u_centre_plane(grid_a, grid_b, args.output_folder / "centre_plane_u_over_uinf.png")

    components_a = threshold_sensitivity(
        grid_a, args.min_component_size, args.remove_boundary_components
    )
    components_b = threshold_sensitivity(
        grid_b, args.min_component_size, args.remove_boundary_components
    )
    write_csv(args.output_folder / "detected_components.csv", _component_rows(components_a + components_b))
    matches = compare_components(components_a, components_b)
    write_csv(args.output_folder / "component_matches.csv", matches)
    write_summary_report(
        args.output_folder / "recommendation.json",
        inspections,
        metrics,
        matches,
        components_b,
        args.rotor_diameter,
    )
    print(f"Saved squared-swirling-strength comparison to: {args.output_folder.resolve()}")


if __name__ == "__main__":
    main()
