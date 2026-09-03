from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ptv_flow.postprocess import _phase_bin_indices  # noqa: E402
from ptv_flow.reader import FlowDataset  # noqa: E402
from ptv_flow.validity import INVALID_SAMPLE_MODES, valid_component_samples, valid_vector_samples  # noqa: E402
from scripts.plot_phase_voxel_convergence import (  # noqa: E402
    COMPONENTS,
    cumulative_means_by_cycle,
    nearest_index,
    parse_phase_degrees,
    phase_signal_from_metadata,
    source_file_from_phase_average,
)


def _read_metadata_defaults(path: Path) -> tuple[str, str]:
    with h5py.File(path, "r") as h5:
        invalid_samples = str(h5.attrs.get("invalid_samples", "nan"))
        zero_mask = str(h5.attrs.get("zero_mask", "component"))
    return invalid_samples, zero_mask


def _valid_masks(values: np.ndarray, invalid_samples: str, zero_mask: str) -> np.ndarray:
    if zero_mask == "vector":
        valid = valid_vector_samples(
            {component: values[index] for index, component in enumerate(COMPONENTS)},
            invalid_samples,
        )
        return np.vstack([valid, valid, valid])
    return np.vstack(
        [
            valid_component_samples(values[index], invalid_samples)
            for index in range(len(COMPONENTS))
        ]
    )


def _phase_bin_stats(
    values: np.ndarray,
    valid: np.ndarray,
    phase_indices: np.ndarray,
    n_phase_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.full((len(COMPONENTS), n_phase_bins), np.nan, dtype=np.float64)
    stds = np.full_like(means, np.nan)
    counts = np.zeros((len(COMPONENTS), n_phase_bins), dtype=np.uint32)
    for component_index in range(len(COMPONENTS)):
        for phase_bin in range(n_phase_bins):
            selected = (phase_indices == phase_bin) & valid[component_index]
            samples = values[component_index, selected]
            counts[component_index, phase_bin] = samples.size
            if samples.size:
                means[component_index, phase_bin] = float(np.mean(samples))
            if samples.size > 1:
                stds[component_index, phase_bin] = float(np.std(samples, ddof=1))
    return means, stds, counts


def _rms_quality(
    values: np.ndarray,
    valid: np.ndarray,
    phase_indices: np.ndarray,
    phase_means: np.ndarray,
    temporal_means: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for component_index, component in enumerate(COMPONENTS):
        selected = valid[component_index] & np.isfinite(
            phase_means[component_index, phase_indices]
        )
        if not np.any(selected):
            rows.append(
                {
                    "component": component,
                    "coherent_rms": np.nan,
                    "residual_rms": np.nan,
                    "coherent_to_residual": np.nan,
                    "samples": 0,
                }
            )
            continue
        fitted_phase = phase_means[component_index, phase_indices[selected]]
        coherent = fitted_phase - temporal_means[component_index]
        residual = values[component_index, selected] - fitted_phase
        residual_rms = float(np.sqrt(np.mean(residual * residual)))
        coherent_rms = float(np.sqrt(np.mean(coherent * coherent)))
        rows.append(
            {
                "component": component,
                "coherent_rms": coherent_rms,
                "residual_rms": residual_rms,
                "coherent_to_residual": coherent_rms / residual_rms
                if residual_rms > 0.0
                else np.inf,
                "samples": int(selected.sum()),
            }
        )
    return rows


def _harmonic_quality(
    phase_centers: np.ndarray,
    phase_means: np.ndarray,
    harmonic_offset: np.ndarray,
    harmonic_a: np.ndarray,
    harmonic_b: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for component_index, component in enumerate(COMPONENTS):
        values = phase_means[component_index]
        valid = np.isfinite(values)
        fitted = (
            harmonic_offset[component_index]
            + harmonic_a[component_index] * np.cos(phase_centers)
            + harmonic_b[component_index] * np.sin(phase_centers)
        )
        if valid.sum() < 3 or not np.isfinite(fitted[valid]).all():
            r2 = np.nan
            residual_rms = np.nan
        else:
            residual = values[valid] - fitted[valid]
            centered = values[valid] - float(np.mean(values[valid]))
            ss_total = float(np.sum(centered * centered))
            ss_error = float(np.sum(residual * residual))
            r2 = 1.0 - ss_error / ss_total if ss_total > 0.0 else np.nan
            residual_rms = float(np.sqrt(np.mean(residual * residual)))
        rows.append(
            {
                "component": component,
                "harmonic_amplitude": float(
                    np.sqrt(harmonic_a[component_index] ** 2 + harmonic_b[component_index] ** 2)
                ),
                "harmonic_phase_deg": float(
                    np.degrees(np.arctan2(-harmonic_b[component_index], harmonic_a[component_index]))
                ),
                "harmonic_r2": r2,
                "harmonic_residual_rms": residual_rms,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assess_phase_quality(
    phase_average_file: Path,
    raw_file: Path,
    x_value: float,
    y_value: float,
    z_value: float,
    selected_phases_deg: np.ndarray,
    invalid_samples: str,
    zero_mask: str,
    min_valid_fraction: float,
    frequency_hz: float | None,
    phase_signal: Path | None,
    phase_offset: float | None,
    output: Path,
) -> Path:
    with h5py.File(phase_average_file, "r") as h5:
        n_phase_bins = int(h5.attrs.get("n_phase_bins", h5["phase"].shape[0]))
        phase_centers = h5["phase"][:].astype(np.float64)
        phase_degrees = h5["phase_degrees"][:] if "phase_degrees" in h5 else np.degrees(phase_centers)
        phase_sample_count = h5["phase_sample_count"][:].astype(np.float64)

    with FlowDataset(raw_file) as raw:
        x = raw.coordinate("x")
        y = raw.coordinate("y")
        z = raw.coordinate("z")
        x_index = nearest_index(x, x_value)
        y_index = nearest_index(y, y_value)
        z_index = nearest_index(z, z_value)
        phases, cycles, _, phase_source = phase_signal_from_metadata(
            phase_average_file,
            raw,
            explicit_phase_signal=phase_signal,
            explicit_frequency_hz=frequency_hz,
            explicit_phase_offset=phase_offset,
        )
        phase_indices = _phase_bin_indices(phases, n_phase_bins)
        values = np.vstack(
            [
                raw._file[component][:, z_index, y_index, x_index].astype(np.float64)
                for component in COMPONENTS
            ]
        )

    valid = _valid_masks(values, invalid_samples, zero_mask)
    raw_bin_means, raw_bin_stds, raw_bin_counts = _phase_bin_stats(
        values,
        valid,
        phase_indices,
        n_phase_bins,
    )
    with h5py.File(phase_average_file, "r") as h5:
        phase_means = np.vstack(
            [
                h5[f"{component}_phase_mean"][:, z_index, y_index, x_index]
                for component in COMPONENTS
            ]
        )
        temporal_means = np.asarray(
            [h5[f"{component}_mean"][z_index, y_index, x_index] for component in COMPONENTS],
            dtype=np.float64,
        )
        harmonic_offset = np.asarray(
            [
                h5[f"{component}_harmonic_offset"][z_index, y_index, x_index]
                if f"{component}_harmonic_offset" in h5
                else h5[f"{component}_mean"][z_index, y_index, x_index]
                for component in COMPONENTS
            ],
            dtype=np.float64,
        )
        harmonic_a = np.asarray(
            [h5[f"{component}_harmonic_a"][z_index, y_index, x_index] for component in COMPONENTS],
            dtype=np.float64,
        )
        harmonic_b = np.asarray(
            [h5[f"{component}_harmonic_b"][z_index, y_index, x_index] for component in COMPONENTS],
            dtype=np.float64,
        )

    target_bins = np.asarray(
        [nearest_index(phase_degrees % 360.0, phase) for phase in selected_phases_deg],
        dtype=np.int64,
    )
    cycle_numbers, convergence, convergence_counts = cumulative_means_by_cycle(
        values,
        valid,
        phase_indices,
        cycles,
        target_bins,
    )
    min_counts = np.ceil(min_valid_fraction * phase_sample_count).astype(np.uint32)
    accepted = raw_bin_counts >= min_counts[None, :]
    standard_error = raw_bin_stds / np.sqrt(raw_bin_counts)

    rms_rows = _rms_quality(values, valid, phase_indices, phase_means, temporal_means)
    harmonic_rows = _harmonic_quality(
        phase_centers,
        phase_means,
        harmonic_offset,
        harmonic_a,
        harmonic_b,
    )

    colors = {"u": "#1f77b4", "v": "#d62728", "w": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_phase, ax_coverage, ax_convergence, ax_summary = axes.ravel()

    for component_index, component in enumerate(COMPONENTS):
        yerr = standard_error[component_index]
        ax_phase.errorbar(
            phase_degrees,
            phase_means[component_index],
            yerr=yerr,
            color=colors[component],
            marker="o",
            linewidth=1.5,
            capsize=2.5,
            label=component,
        )
        fitted = (
            harmonic_offset[component_index]
            + harmonic_a[component_index] * np.cos(phase_centers)
            + harmonic_b[component_index] * np.sin(phase_centers)
        )
        ax_phase.plot(phase_degrees, fitted, color=colors[component], linestyle="--", linewidth=1.0)
        ax_coverage.plot(
            phase_degrees,
            raw_bin_counts[component_index] / np.maximum(phase_sample_count, 1.0),
            color=colors[component],
            marker="o",
            linewidth=1.4,
            label=component,
        )

    ax_coverage.axhline(min_valid_fraction, color="black", linestyle="--", linewidth=1.0)
    ax_coverage.set_ylim(-0.03, 1.05)
    ax_coverage.set_ylabel("valid fraction per bin")
    ax_coverage.set_xlabel("phase [deg]")
    ax_coverage.grid(True, color="0.82", linewidth=0.6)
    ax_coverage.legend(loc="best", ncol=3)

    phase_order = 0
    phase_bin = int(target_bins[phase_order])
    for component_index, component in enumerate(COMPONENTS):
        ax_convergence.plot(
            cycle_numbers,
            convergence[phase_order, component_index],
            color=colors[component],
            marker="o",
            markersize=3.0,
            linewidth=1.3,
            label=component,
        )
        ax_convergence.axhline(
            phase_means[component_index, phase_bin],
            color=colors[component],
            linestyle="--",
            linewidth=0.9,
        )
    ax_convergence.set_title(f"Convergence at {phase_degrees[phase_bin]:.1f} deg bin")
    ax_convergence.set_xlabel("completed cycle index")
    ax_convergence.set_ylabel("cumulative phase mean")
    ax_convergence.grid(True, color="0.82", linewidth=0.6)
    ax_convergence.legend(loc="best", ncol=3)

    ax_phase.set_title("Phase-bin means with standard-error bars")
    ax_phase.set_xlabel("phase [deg]")
    ax_phase.set_ylabel("velocity")
    ax_phase.grid(True, color="0.82", linewidth=0.6)
    ax_phase.legend(loc="best", ncol=3)

    summary_lines = [
        f"nearest voxel:",
        f"x={float(x[x_index]):.6g}  y={float(y[y_index]):.6g}  z={float(z[z_index]):.6g}",
        f"phase source: {phase_source}",
        f"invalid: {invalid_samples}, zero-mask: {zero_mask}",
        "",
        "accepted phase bins:",
    ]
    for component_index, component in enumerate(COMPONENTS):
        summary_lines.append(
            f"{component}: {int(accepted[component_index].sum())}/{n_phase_bins}"
        )
    summary_lines.append("")
    summary_lines.append("coherent RMS / residual RMS:")
    for row in rms_rows:
        summary_lines.append(f"{row['component']}: {float(row['coherent_to_residual']):.3g}")
    summary_lines.append("")
    summary_lines.append("first-harmonic R2:")
    for row in harmonic_rows:
        summary_lines.append(f"{row['component']}: {float(row['harmonic_r2']):.3g}")
    ax_summary.axis("off")
    ax_summary.text(0.0, 1.0, "\n".join(summary_lines), va="top", family="monospace")

    fig.suptitle("Phase-average quality check at one voxel", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    rows = []
    for component_index, component in enumerate(COMPONENTS):
        for phase_bin, phase_deg in enumerate(phase_degrees):
            rows.append(
                {
                    "kind": "phase_bin",
                    "component": component,
                    "phase_bin": phase_bin,
                    "phase_deg": float(phase_deg),
                    "mean": phase_means[component_index, phase_bin],
                    "raw_recomputed_mean": raw_bin_means[component_index, phase_bin],
                    "std": raw_bin_stds[component_index, phase_bin],
                    "standard_error": standard_error[component_index, phase_bin],
                    "count": int(raw_bin_counts[component_index, phase_bin]),
                    "required_count": int(min_counts[phase_bin]),
                    "accepted": bool(accepted[component_index, phase_bin]),
                }
            )
    for row in rms_rows:
        rows.append({"kind": "rms", **row})
    for row in harmonic_rows:
        rows.append({"kind": "harmonic", **row})
    write_csv(output.with_suffix(".csv"), rows)
    print(f"Saved phase-quality plot: {output.resolve()}", flush=True)
    print(f"Saved phase-quality data: {output.with_suffix('.csv').resolve()}", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assess phase-average quality at one voxel by rereading the raw time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("phase_average_file", type=Path)
    parser.add_argument("--raw-file", type=Path, default=None)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--phases-deg", default="0,90,180,270")
    parser.add_argument("--invalid-samples", choices=INVALID_SAMPLE_MODES, default=None)
    parser.add_argument("--zero-mask", choices=("component", "vector"), default=None)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--phase-signal", type=Path, default=None)
    parser.add_argument("--phase-offset", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "phase_quality.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    defaults_invalid, defaults_zero = _read_metadata_defaults(args.phase_average_file)
    raw_file = args.raw_file or source_file_from_phase_average(args.phase_average_file)
    assess_phase_quality(
        phase_average_file=args.phase_average_file,
        raw_file=raw_file,
        x_value=args.x,
        y_value=args.y,
        z_value=args.z,
        selected_phases_deg=parse_phase_degrees(args.phases_deg),
        invalid_samples=args.invalid_samples or defaults_invalid,
        zero_mask=args.zero_mask or defaults_zero,
        min_valid_fraction=args.min_valid_fraction,
        frequency_hz=args.frequency_hz,
        phase_signal=args.phase_signal,
        phase_offset=args.phase_offset,
        output=args.output,
    )


if __name__ == "__main__":
    main()
