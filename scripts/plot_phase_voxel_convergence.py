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

from ptv_flow.postprocess import (  # noqa: E402
    TWO_PI,
    _load_phase_signal,
    _phase_bin_indices,
    _phase_from_frequency,
)
from ptv_flow.reader import FlowDataset  # noqa: E402
from ptv_flow.validity import (  # noqa: E402
    INVALID_SAMPLE_MODES,
    valid_component_samples,
    valid_vector_samples,
)


COMPONENTS = ("u", "v", "w")


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values - target)))


def parse_phase_degrees(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one phase angle is required")
    return np.asarray(values, dtype=np.float64) % 360.0


def source_file_from_phase_average(path: Path) -> Path:
    with h5py.File(path, "r") as h5:
        source = h5.attrs.get("source_file")
    if source is None:
        raise ValueError("phase-average file does not contain source_file metadata")
    return Path(str(source))


def phase_signal_from_metadata(
    phase_average_file: Path,
    raw: FlowDataset,
    explicit_phase_signal: Path | None,
    explicit_frequency_hz: float | None,
    explicit_phase_offset: float | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    times = raw.coordinate("t").astype(np.float64)
    with h5py.File(phase_average_file, "r") as h5:
        stored_frequency = float(h5.attrs.get("frequency_hz", -1.0))
        stored_offset = float(h5.attrs.get("phase_offset", 0.0))
        stored_source = str(h5.attrs.get("phase_source", ""))

    phase_offset = stored_offset if explicit_phase_offset is None else explicit_phase_offset
    frequency_hz = (
        stored_frequency if explicit_frequency_hz is None else explicit_frequency_hz
    )
    if explicit_phase_signal is not None:
        unwrapped = np.unwrap(_load_phase_signal(explicit_phase_signal))
        source = str(explicit_phase_signal)
    elif frequency_hz > 0.0:
        cycles = float(frequency_hz) * times + phase_offset / TWO_PI
        unwrapped = TWO_PI * cycles
        source = f"frequency_hz={frequency_hz:g}"
    elif stored_source and stored_source not in {"frequency", "array"}:
        unwrapped = np.unwrap(_load_phase_signal(stored_source))
        source = stored_source
    else:
        raise ValueError(
            "Could not reconstruct phase. Pass --frequency-hz or --phase-signal."
        )

    if unwrapped.shape[0] != raw.n_times:
        raise ValueError(
            "phase signal length must match raw time dimension: "
            f"{unwrapped.shape[0]} != {raw.n_times}"
        )
    return unwrapped % TWO_PI, np.floor(unwrapped / TWO_PI).astype(np.int64), phase_offset, source


def cumulative_means_by_cycle(
    values: np.ndarray,
    valid: np.ndarray,
    phase_indices: np.ndarray,
    cycle_indices: np.ndarray,
    target_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cycles = np.arange(int(cycle_indices.min()), int(cycle_indices.max()) + 1)
    means = np.full((target_bins.size, COMPONENTS.__len__(), cycles.size), np.nan)
    counts = np.zeros((target_bins.size, COMPONENTS.__len__(), cycles.size), dtype=np.uint32)

    for phase_order, phase_bin in enumerate(target_bins):
        phase_mask = phase_indices == phase_bin
        for component_index in range(values.shape[0]):
            cumulative_sum = 0.0
            cumulative_count = 0
            for output_index, cycle in enumerate(cycles):
                selected = phase_mask & (cycle_indices == cycle) & valid[component_index]
                if np.any(selected):
                    cumulative_sum += float(values[component_index, selected].sum())
                    cumulative_count += int(selected.sum())
                counts[phase_order, component_index, output_index] = cumulative_count
                if cumulative_count:
                    means[phase_order, component_index, output_index] = (
                        cumulative_sum / cumulative_count
                    )
    return cycles, means, counts


def cumulative_temporal_means_by_cycle(
    values: np.ndarray,
    valid: np.ndarray,
    cycle_indices: np.ndarray,
) -> np.ndarray:
    cycles = np.arange(int(cycle_indices.min()), int(cycle_indices.max()) + 1)
    means = np.full((values.shape[0], cycles.size), np.nan)
    for component_index in range(values.shape[0]):
        cumulative_sum = 0.0
        cumulative_count = 0
        for output_index, cycle in enumerate(cycles):
            selected = (cycle_indices == cycle) & valid[component_index]
            if np.any(selected):
                cumulative_sum += float(values[component_index, selected].sum())
                cumulative_count += int(selected.sum())
            if cumulative_count:
                means[component_index, output_index] = cumulative_sum / cumulative_count
    return means


def reference_series(
    phase_average_file: Path,
    z_index: int,
    y_index: int,
    x_index: int,
    field: str,
) -> dict[str, np.ndarray]:
    suffix = "phase_mean" if field == "phase_mean" else "coherent"
    with h5py.File(phase_average_file, "r") as h5:
        return {
            "phase_degrees": h5["phase_degrees"][:]
            if "phase_degrees" in h5
            else np.degrees(h5["phase"][:]),
            **{
                component: h5[f"{component}_{suffix}"][:, z_index, y_index, x_index]
                for component in COMPONENTS
            },
        }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence(
    phase_average_file: Path,
    raw_file: Path,
    x_value: float,
    y_value: float,
    z_value: float,
    phases_deg: np.ndarray,
    field: str,
    invalid_samples: str,
    zero_mask: str,
    min_valid_fraction: float,
    frequency_hz: float | None,
    phase_signal: Path | None,
    phase_offset: float | None,
    output: Path,
) -> Path:
    with h5py.File(phase_average_file, "r") as phase_h5:
        n_phase_bins = int(phase_h5.attrs.get("n_phase_bins", phase_h5["phase"].shape[0]))
        phase_centres_deg = (
            phase_h5["phase_degrees"][:]
            if "phase_degrees" in phase_h5
            else np.degrees(phase_h5["phase"][:])
        )

    with FlowDataset(raw_file) as raw:
        x_index = nearest_index(raw.coordinate("x"), x_value)
        y_index = nearest_index(raw.coordinate("y"), y_value)
        z_index = nearest_index(raw.coordinate("z"), z_value)
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

    if zero_mask == "vector":
        vector_valid = valid_vector_samples(
            {component: values[index] for index, component in enumerate(COMPONENTS)},
            invalid_samples,
        )
        valid = np.vstack([vector_valid, vector_valid, vector_valid])
    else:
        valid = np.vstack(
            [
                valid_component_samples(values[index], invalid_samples)
                for index in range(len(COMPONENTS))
            ]
        )

    target_bins = np.asarray(
        [nearest_index(phase_centres_deg % 360.0, phase) for phase in phases_deg],
        dtype=np.int64,
    )
    cycle_numbers, means, counts = cumulative_means_by_cycle(
        values,
        valid,
        phase_indices,
        cycles,
        target_bins,
    )
    if field == "coherent":
        temporal_means = cumulative_temporal_means_by_cycle(values, valid, cycles)
        means = means - temporal_means[None, :, :]
    references = reference_series(phase_average_file, z_index, y_index, x_index, field)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes_flat = axes.ravel()
    colors = {"u": "#1f77b4", "v": "#d62728", "w": "#2ca02c"}
    markers = {"u": "o", "v": "s", "w": "^"}
    rows = []

    for phase_order, ax in enumerate(axes_flat[: target_bins.size]):
        phase_bin = int(target_bins[phase_order])
        phase_label = float(phase_centres_deg[phase_bin])
        for component_index, component in enumerate(COMPONENTS):
            ydata = means[phase_order, component_index]
            count_data = counts[phase_order, component_index]
            ax.plot(
                cycle_numbers,
                ydata,
                color=colors[component],
                marker=markers[component],
                markersize=3.5,
                linewidth=1.4,
                label=component,
            )
            ax.axhline(
                float(references[component][phase_bin]),
                color=colors[component],
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
            )
            for cycle, mean, count in zip(cycle_numbers, ydata, count_data):
                rows.append(
                    {
                        "requested_phase_deg": float(phases_deg[phase_order]),
                        "phase_bin": phase_bin,
                        "phase_bin_center_deg": phase_label,
                        "cycle": int(cycle),
                        "component": component,
                        "mean": mean,
                        "count": int(count),
                    }
                )
        max_possible = np.bincount(
            cycles[phase_indices == phase_bin] - cycle_numbers[0],
            minlength=cycle_numbers.size,
        ).cumsum()
        min_required = np.ceil(min_valid_fraction * max_possible)
        if min_valid_fraction > 0.0:
            ax_count = ax.twinx()
            ax_count.step(
                cycle_numbers,
                min_required,
                color="0.45",
                linewidth=0.9,
                alpha=0.65,
                label="min count",
            )
            ax_count.set_ylabel("required count", color="0.35")
            ax_count.tick_params(axis="y", colors="0.35")
        ax.set_title(f"{phase_label:.1f} deg bin")
        ax.grid(True, color="0.82", linewidth=0.6)
        ax.set_ylabel("velocity" if field == "phase_mean" else "coherent velocity")

    for ax in axes_flat[target_bins.size :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3)
    for ax in axes_flat[: target_bins.size]:
        ax.set_xlabel("completed cycle index")

    fig.suptitle(
        f"{field} convergence at nearest voxel "
        f"x={x_value:g}, y={y_value:g}, z={z_value:g}\n"
        f"source={raw_file.name}, phase={phase_source}",
        fontsize=11,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_csv(output.with_suffix(".csv"), rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot true cycle-by-cycle convergence of phase-locked u/v/w at one "
            "voxel by rereading the raw time series."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("phase_average_file", type=Path)
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=None,
        help="raw/interpolated time-series file; default uses phase_average source_file metadata",
    )
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument(
        "--phases-deg",
        default="0,90,180,270",
        help="comma-separated requested motion phases",
    )
    parser.add_argument(
        "--field",
        choices=("phase_mean", "coherent"),
        default="phase_mean",
        help="plot cumulative phase means or cumulative coherent fluctuations",
    )
    parser.add_argument(
        "--invalid-samples",
        choices=INVALID_SAMPLE_MODES,
        default=None,
        help="invalid-sample policy; default reads phase_average metadata or uses nan",
    )
    parser.add_argument(
        "--zero-mask",
        choices=("component", "vector"),
        default=None,
        help="zero-mask policy; default reads phase_average metadata or uses component",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help="draw required cumulative count guide for each selected phase bin",
    )
    parser.add_argument("--frequency-hz", type=float, default=None)
    parser.add_argument("--phase-signal", type=Path, default=None)
    parser.add_argument("--phase-offset", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "phase_voxel_convergence.png",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_average_file = args.phase_average_file
    raw_file = args.raw_file or source_file_from_phase_average(phase_average_file)
    with h5py.File(phase_average_file, "r") as h5:
        invalid_samples = args.invalid_samples or str(h5.attrs.get("invalid_samples", "nan"))
        zero_mask = args.zero_mask or str(h5.attrs.get("zero_mask", "component"))
    output = plot_convergence(
        phase_average_file=phase_average_file,
        raw_file=raw_file,
        x_value=args.x,
        y_value=args.y,
        z_value=args.z,
        phases_deg=parse_phase_degrees(args.phases_deg),
        field=args.field,
        invalid_samples=invalid_samples,
        zero_mask=zero_mask,
        min_valid_fraction=args.min_valid_fraction,
        frequency_hz=args.frequency_hz,
        phase_signal=args.phase_signal,
        phase_offset=args.phase_offset,
        output=args.output,
    )
    print(f"Saved convergence plot: {output.resolve()}", flush=True)
    print(f"Saved convergence data: {output.with_suffix('.csv').resolve()}", flush=True)


if __name__ == "__main__":
    main()
