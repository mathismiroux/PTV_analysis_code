from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.reader import FlowDataset, VELOCITY_COMPONENTS
from ptv_flow.validity import INVALID_SAMPLE_MODES, valid_component_samples


@dataclass(frozen=True)
class AxisSelection:
    name: str
    center: float
    half_width: float
    indices: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ComponentStats:
    mean: np.ndarray
    fluctuation_variance: np.ndarray
    count: np.ndarray


def _axis_selection(
    flow: FlowDataset, name: str, center: float, half_width: float
) -> AxisSelection:
    values = flow.coordinate(name)
    indices = np.where((values >= center - half_width) & (values <= center + half_width))[0]
    if indices.size == 0:
        nearest = int(np.nanargmin(np.abs(values - center)))
        raise ValueError(
            f"No {name} coordinates found in [{center - half_width:g}, "
            f"{center + half_width:g}] mm. Nearest value is "
            f"{values[nearest]:g} at index {nearest}."
        )
    return AxisSelection(
        name=name,
        center=center,
        half_width=half_width,
        indices=indices,
        values=values[indices],
    )


def _centered_x(flow: FlowDataset) -> float:
    x = flow.coordinate("x")
    return float(0.5 * (np.nanmin(x) + np.nanmax(x)))


def _read_component_cube(
    flow: FlowDataset,
    component: str,
    z_indices: np.ndarray,
    y_indices: np.ndarray,
    x_indices: np.ndarray,
) -> np.ndarray:
    data = flow._file[component][
        :,
        z_indices.min() : z_indices.max() + 1,
        y_indices.min() : y_indices.max() + 1,
        x_indices.min() : x_indices.max() + 1,
    ]
    return np.asarray(data[:, :, :, :], dtype=np.float64)


def _component_stats(data: np.ndarray, invalid_samples: str = "nan") -> ComponentStats:
    valid = valid_component_samples(data, invalid_samples)
    count = valid.sum(axis=0, dtype=np.uint32)
    mean = np.full(data.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.where(valid, data, 0.0).sum(axis=0, dtype=np.float64),
        count,
        out=mean,
        where=count > 0,
    )

    fluctuation = data - mean[None, :, :, :]
    variance = np.full(data.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.where(valid, fluctuation * fluctuation, 0.0).sum(axis=0, dtype=np.float64),
        count,
        out=variance,
        where=count > 0,
    )
    return ComponentStats(mean=mean, fluctuation_variance=variance, count=count)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _format_stats(label: str, stats: dict[str, float]) -> str:
    return (
        f"{label:<28} mean={stats['mean']:.6g}  p95={stats['p95']:.6g}  "
        f"p99={stats['p99']:.6g}  max={stats['max']:.6g}"
    )


def compare_precision(
    double_path: Path,
    single_path: Path,
    y_center: float,
    z_center: float,
    half_width: float,
    x_center: float | None,
    invalid_samples: str = "nan",
) -> str:
    lines: list[str] = []
    with FlowDataset(double_path) as double, FlowDataset(single_path) as single:
        if double.shape != single.shape:
            raise ValueError(f"Shape mismatch: {double.shape} != {single.shape}")

        for name in ("x", "y", "z"):
            a = double.coordinate(name)
            b = single.coordinate(name)
            if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1e-2):
                raise ValueError(f"Coordinate mismatch on {name!r}")

        x_center = _centered_x(double) if x_center is None else x_center
        x_sel = _axis_selection(double, "x", x_center, half_width)
        y_sel = _axis_selection(double, "y", y_center, half_width)
        z_sel = _axis_selection(double, "z", z_center, half_width)

        lines.extend(
            [
                "Precision comparison on fluctuation quantities",
                f"double file: {double.path.resolve()}",
                f"single file: {single.path.resolve()}",
                f"double dtype: {double.dtype}, single dtype: {single.dtype}",
                f"time steps: {double.n_times}",
                f"invalid samples: {invalid_samples}",
                "",
                "Selected cube:",
                (
                    f"  x center={x_center:.6g} mm, range="
                    f"[{x_center - half_width:.6g}, {x_center + half_width:.6g}], "
                    f"indices={x_sel.indices.tolist()}, values={x_sel.values.tolist()}"
                ),
                (
                    f"  y center={y_center:.6g} mm, range="
                    f"[{y_center - half_width:.6g}, {y_center + half_width:.6g}], "
                    f"indices={y_sel.indices.tolist()}, values={y_sel.values.tolist()}"
                ),
                (
                    f"  z center={z_center:.6g} mm, range="
                    f"[{z_center - half_width:.6g}, {z_center + half_width:.6g}], "
                    f"indices={z_sel.indices.tolist()}, values={z_sel.values.tolist()}"
                ),
                (
                    "  cube grid points: "
                    f"{z_sel.indices.size} x {y_sel.indices.size} x {x_sel.indices.size} "
                    f"= {z_sel.indices.size * y_sel.indices.size * x_sel.indices.size}"
                ),
                "",
            ]
        )

        double_stats = {}
        single_stats = {}
        double_tke_terms = []
        single_tke_terms = []

        for component in VELOCITY_COMPONENTS:
            double_data = _read_component_cube(
                double, component, z_sel.indices, y_sel.indices, x_sel.indices
            )
            single_data = _read_component_cube(
                single, component, z_sel.indices, y_sel.indices, x_sel.indices
            )
            d_stats = _component_stats(double_data, invalid_samples=invalid_samples)
            s_stats = _component_stats(single_data, invalid_samples=invalid_samples)
            double_stats[component] = d_stats
            single_stats[component] = s_stats
            double_tke_terms.append(d_stats.fluctuation_variance)
            single_tke_terms.append(s_stats.fluctuation_variance)

            both_valid = valid_component_samples(
                double_data, invalid_samples
            ) & valid_component_samples(single_data, invalid_samples)
            double_fluct = double_data - d_stats.mean[None, :, :, :]
            single_fluct = single_data - s_stats.mean[None, :, :, :]
            fluct_abs_diff = np.abs(double_fluct - single_fluct)[both_valid]
            rms_fluct = np.sqrt(np.nanmean(double_fluct[both_valid] ** 2))
            rel_fluct_diff = fluct_abs_diff / rms_fluct if rms_fluct > 0 else fluct_abs_diff

            variance_abs_diff = np.abs(
                d_stats.fluctuation_variance - s_stats.fluctuation_variance
            )
            typical_variance = np.nanmean(d_stats.fluctuation_variance)
            variance_rel_diff = (
                variance_abs_diff / typical_variance
                if typical_variance > 0
                else variance_abs_diff
            )

            lines.extend(
                [
                    f"{component} fluctuation precision:",
                    _format_stats(
                        f"  abs({component}'_64 - {component}'_32)",
                        _finite_stats(fluct_abs_diff),
                    ),
                    _format_stats(
                        "  rel to RMS fluctuation",
                        _finite_stats(rel_fluct_diff),
                    ),
                    _format_stats(
                        f"  abs({component}'^2 mean diff)",
                        _finite_stats(variance_abs_diff),
                    ),
                    _format_stats(
                        "  rel to mean variance",
                        _finite_stats(variance_rel_diff),
                    ),
                    (
                        "  valid-count difference max="
                        f"{int(np.max(np.abs(d_stats.count.astype(int) - s_stats.count.astype(int))))}"
                    ),
                    "",
                ]
            )

        double_tke = 0.5 * sum(double_tke_terms)
        single_tke = 0.5 * sum(single_tke_terms)
        tke_abs_diff = np.abs(double_tke - single_tke)
        typical_tke = np.nanmean(double_tke)
        tke_rel_diff = tke_abs_diff / typical_tke if typical_tke > 0 else tke_abs_diff
        lines.extend(
            [
                "TKE precision:",
                _format_stats("  abs(k64 - k32)", _finite_stats(tke_abs_diff)),
                _format_stats("  rel to mean k64", _finite_stats(tke_rel_diff)),
                f"  mean k64={typical_tke:.6g}",
                f"  mean k32={np.nanmean(single_tke):.6g}",
            ]
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare double- and single-precision fluctuation/TKE quantities "
            "inside a wake shear-layer volume."
        )
    )
    parser.add_argument("--double", type=Path, default=Path("Static_3.5D__b128.nc"))
    parser.add_argument("--single", type=Path, default=Path("Static_3.5D__b128f.nc"))
    parser.add_argument("--y-center", type=float, default=600.0)
    parser.add_argument("--z-center", type=float, default=0.0)
    parser.add_argument("--x-center", type=float, default=None)
    parser.add_argument(
        "--half-width",
        type=float,
        default=90.0,
        help="half-width in mm; default gives a 180 mm cube",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default="nan",
        help=(
            "raw samples to exclude from statistics: zero ignores exact zeros, "
            "nan ignores NaN/inf values, zero-or-nan ignores both, none excludes nothing"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = compare_precision(
        double_path=args.double,
        single_path=args.single,
        y_center=args.y_center,
        z_center=args.z_center,
        half_width=args.half_width,
        x_center=args.x_center,
        invalid_samples=args.invalid_samples,
    )
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
