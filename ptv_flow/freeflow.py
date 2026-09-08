"""Free-flow statistics and finite-domain correlation estimates.

See docs/freeflow_workflow.md for definitions, assumptions and references.
The selected reader reuses the existing mean/TKE/stress implementations.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from .postprocess import (
    TemporalAverageVolume, reynolds_stresses, temporal_average_volume,
    turbulent_kinetic_energy,
)
from .reader import FlowDataset, VELOCITY_COMPONENTS
from .validity import valid_vector_samples


@dataclass(frozen=True)
class FreeflowSettings:
    length_unit: str = "native"
    time_unit: str = "native"
    velocity_unit: str = "native"
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None
    z_range: tuple[float, float] | None = None
    start: int = 0
    stop: int | None = None
    chunk_size: int = 8
    min_valid_fraction: float = 0.5
    min_pairs: int = 100
    min_pair_fraction: float = 0.05
    max_spatial_lag: int | None = None
    max_temporal_lag: int | None = None
    invalid_samples: str = "nan"
    include_filled: bool = False
    temporal: bool = False
    probe_grid: int = 2
    blocks: int = 4
    rotor_diameter_m: float | None = None
    convection_speed_m_s: float | None = None


class _VelocityView:
    def __init__(self, dataset, selection, scale, filled):
        self.dataset = dataset
        self.selection = selection
        self.scale = scale
        self.filled = filled
        self.shape = tuple(s.stop - s.start for s in selection)
        self.dtype = np.dtype("float64")

    def __getitem__(self, key):
        keys = key if isinstance(key, tuple) else (key,)
        keys = keys + (slice(None),) * (4 - len(keys))
        mapped = []
        for k, s, n in zip(keys, self.selection, self.shape):
            if isinstance(k, slice):
                a, b, step = k.indices(n)
                mapped.append(slice(s.start + a, s.start + b, step))
            else:
                i = int(k)
                i = i + n if i < 0 else i
                if not 0 <= i < n:
                    raise IndexError(i)
                mapped.append(s.start + i)
        mapped = tuple(mapped)
        values = np.asarray(self.dataset[mapped], dtype=np.float64) * self.scale
        for mask in self.filled:
            values = np.where(mask[mapped] != 0, np.nan, values)
        return values


class SelectedFlow(FlowDataset):
    """Lazy, contiguous ROI/time selection with consistent units and fill masking."""

    def __init__(self, path, settings: FreeflowSettings):
        super().__init__(path)
        self._source = self._file
        try:
            scales = {
                "length": {"native": 1., "m": 1., "mm": 0.001}[settings.length_unit],
                "time": {"native": 1., "s": 1., "ms": 0.001}[settings.time_unit],
                "velocity": {"native": 1., "m/s": 1., "mm/s": 0.001}[settings.velocity_unit],
            }
            stop = self.n_times if settings.stop is None else settings.stop
            if not 0 <= settings.start < stop <= self.n_times or stop - settings.start < 4:
                raise ValueError("Select at least four frames within the input time range")
            selection = [slice(settings.start, stop)]
            coordinates = {"t": self.coordinate("t")[settings.start:stop] * scales["time"]}
            for axis in ("z", "y", "x"):
                coord = self.coordinate(axis).astype(float)
                if not np.all(np.isfinite(coord)):
                    raise ValueError(f"{axis} coordinates must be finite")
                if len(coord) > 1 and not (np.all(np.diff(coord) > 0) or np.all(np.diff(coord) < 0)):
                    raise ValueError(f"{axis} coordinates must be strictly monotone")
                bounds = getattr(settings, axis + "_range")
                indices = np.arange(len(coord)) if bounds is None else np.flatnonzero(
                    (coord >= bounds[0]) & (coord <= bounds[1]))
                if not len(indices):
                    raise ValueError(f"Empty {axis} selection; bounds use input coordinate units")
                selection.append(slice(int(indices[0]), int(indices[-1]) + 1))
                coordinates[axis] = coord[indices] * scales["length"]
            masks = []
            if not settings.include_filled:
                for key in ("filled_mask", "u_filled_mask", "v_filled_mask", "w_filled_mask"):
                    if key in self._source:
                        if self._source[key].shape != self._source["u"].shape:
                            raise ValueError(f"Unexpected {key} shape")
                        masks.append(self._source[key])
            self.selection = selection
            self.mask_names = [m.name for m in masks]
            self.source_attrs = {k: str(v) for k, v in self._source.attrs.items()}
            self._file = dict(coordinates)
            self._file.update({c: _VelocityView(self._source[c], selection, scales["velocity"], masks)
                               for c in VELOCITY_COMPONENTS})
        except Exception:
            self._source.close()
            raise

    def close(self):
        self._source.close()


def uniform_spacing(values, name):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    delta = np.diff(values)
    spacing = float(np.median(delta))
    # Accommodate rounded coordinates and float32 timestamps in DaVis exports.
    if not np.all(np.isfinite(values)) or spacing == 0 or not np.allclose(
        delta, spacing, rtol=0.002, atol=abs(spacing) * 1e-5
    ):
        raise ValueError(f"{name} must be uniformly spaced for lag-index correlations")
    return abs(spacing)


def pair_moments(fluctuations, axis, max_lag):
    """FFT sums [ab, a², b², count] over valid endpoint pairs, no wraparound.

    Missing observations retain their positions. Means are removed before this
    function; no lag-dependent re-centring or missing-value interpolation occurs.
    """
    f = np.asarray(fluctuations, dtype=float)
    valid = np.isfinite(f)
    f = np.where(valid, f, 0.)
    n = f.shape[axis]
    nfft = 1 << (2 * n - 1).bit_length()
    ff = np.fft.rfft(f, n=nfft, axis=axis)
    mm = np.fft.rfft(valid.astype(float), n=nfft, axis=axis)
    qq = np.fft.rfft(f * f, n=nfft, axis=axis)
    reduction = tuple(i for i in range(f.ndim) if i != axis)

    def reduce(product):
        # Linearity lets us pool spectra before the inverse FFT. This avoids
        # constructing four full lag volumes for every component/direction.
        spectrum = product.sum(axis=reduction)
        return np.fft.irfft(spectrum, n=nfft)[:max_lag + 1]

    return np.stack([reduce(np.conj(ff) * ff), reduce(np.conj(qq) * mm),
                     reduce(np.conj(mm) * qq), np.rint(reduce(np.conj(mm) * mm))])


def normalized_correlation(moments, min_pairs, min_pair_fraction):
    numerator, aa, bb, count = moments
    denom = np.sqrt(np.maximum(aa, 0) * np.maximum(bb, 0))
    rho = np.full(count.shape, np.nan)
    supported = (count >= max(min_pairs, count[0] * min_pair_fraction)) & (denom > 0)
    np.divide(numerator, denom, out=rho, where=supported)
    return np.clip(rho, -1., 1.)


def integrate_correlation(lag, rho):
    """First positive lobe, interpolating the zero; unresolved lengths are null."""
    lag, rho = np.asarray(lag, float), np.asarray(rho, float)
    if lag.ndim != 1 or lag.shape != rho.shape or not len(lag):
        raise ValueError("lag and rho must be nonempty matching vectors")
    if lag[0] != 0 or np.any(np.diff(lag) <= 0):
        raise ValueError("lags must start at zero and increase")
    result = dict(integral=None, truncated_integral=None, cutoff=None,
                  status="insufficient_pairs_or_zero_variance")
    if not np.isfinite(rho[0]) or rho[0] <= 0:
        return result
    area = 0.
    for j in range(1, len(lag)):
        if not np.isfinite(rho[j]):
            result.update(truncated_integral=area, cutoff=float(lag[j-1]),
                          status="insufficient_pairs_before_zero")
            return result
        if rho[j] <= 0:
            zero = lag[j-1] + (lag[j] - lag[j-1]) * rho[j-1] / (rho[j-1] - rho[j])
            area += .5 * rho[j-1] * (zero - lag[j-1])
            result.update(integral=float(area), truncated_integral=float(area),
                          cutoff=float(zero), status="first_zero_crossing")
            return result
        area += .5 * (rho[j-1] + rho[j]) * (lag[j] - lag[j-1])
    result.update(truncated_integral=float(area), cutoff=float(lag[-1]),
                  status="no_zero_crossing")
    return result


def _finite_mean(a):
    a = np.asarray(a)
    return float(np.mean(a[np.isfinite(a)])) if np.any(np.isfinite(a)) else None


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_settings(s):
    if s.chunk_size < 1 or s.min_pairs < 1 or s.blocks < 2 or s.probe_grid < 1:
        raise ValueError("chunk size, pair count and probe grid must be positive; blocks >= 2")
    if not 0 <= s.min_valid_fraction <= 1 or not 0 < s.min_pair_fraction <= 1:
        raise ValueError("Validity fractions must be in [0, 1], pair fraction in (0, 1]")
    if s.invalid_samples not in ("nan", "zero-or-nan"):
        raise ValueError("Use nan or zero-or-nan; nonfinite samples must be excluded")
    for value in (s.max_spatial_lag, s.max_temporal_lag):
        if value is not None and value < 1:
            raise ValueError("Maximum lags must be positive")
    for value in (s.rotor_diameter_m, s.convection_speed_m_s):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Rotor diameter and convection speed must be finite and positive")
    if s.temporal and s.time_unit == "native":
        raise ValueError("Temporal analysis requires explicit --time-unit s or ms")
    if s.convection_speed_m_s is not None and not s.temporal:
        raise ValueError("Convection speed requires temporal analysis")


def characterize_freeflow(source, output, settings=FreeflowSettings()):
    """Write a new, self-contained analysis directory; refuse an existing one."""
    _validate_settings(settings)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")
    s = settings
    with SelectedFlow(source, s) as flow:
        spacing = {a: uniform_spacing(flow.coordinate(a), a) for a in ("x", "y", "z")}
        dt = None
        if s.temporal:
            times = flow.coordinate("t")
            if not np.all(np.diff(times) > 0):
                raise ValueError("Temporal analysis requires increasing timestamps")
            dt = uniform_spacing(times, "t")
        if s.blocks > flow.n_times // 2:
            raise ValueError("Select enough frames for at least two per diagnostic block")
        output.mkdir(parents=True)
        manifest = dict(source_file=str(flow.path.resolve()),
                        source_size_bytes=flow.path.stat().st_size,
                        source_mtime_ns=flow.path.stat().st_mtime_ns,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        settings=asdict(s), shape=flow.shape, source_attributes=flow.source_attrs,
                        excluded_fill_masks=flow.mask_names, status="running")
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        metadata = {"workflow": "freeflow", "selection_and_units": json.dumps(asdict(s)),
                    "excluded_fill_masks": ",".join(flow.mask_names),
                    "coordinate_units": "m" if s.length_unit != "native" else "native_length",
                    "velocity_units": "m/s" if s.velocity_unit != "native" else "native_velocity",
                    "time_units": "s" if s.time_unit != "native" else "native_time"}
        common = dict(chunk_size=s.chunk_size, zero_mask="vector",
                      invalid_samples=s.invalid_samples, metadata=metadata)
        temporal_average_volume(flow, output / "mean.nc",
                                min_valid_fraction=s.min_valid_fraction, **common)
        with TemporalAverageVolume(output / "mean.nc") as mean:
            turbulent_kinetic_energy(flow, mean, output / "tke.nc", **common)
            reynolds_stresses(flow, mean, output / "reynolds_stresses.nc", **common)
            means = {c: mean._file[c + "_mean"][:] for c in VELOCITY_COMPONENTS}
            counts = mean._file["vector_count"][:]
        with h5py.File(output / "tke.nc") as f:
            variance = {c: f[c + "_prime2_mean"][:] for c in VELOCITY_COMPONENTS}
            tke = f["tke"][:]
        # Constant fields can retain roundoff after mean subtraction. Do not
        # interpret that residual as physical correlation at machine precision.
        fluctuating = {c: variance[c] > (64 * np.finfo(float).eps * np.abs(means[c]))**2
                       for c in VELOCITY_COMPONENTS}
        good = np.logical_and.reduce([np.isfinite(means[c]) for c in VELOCITY_COMPONENTS])
        if not np.any(good):
            raise ValueError("No voxels meet min_valid_fraction; inspect coverage or choose another ROI")
        length_label = "m" if s.length_unit != "native" else "native_length"
        velocity_label = "m/s" if s.velocity_unit != "native" else "native_velocity"
        with h5py.File(output / "statistics.h5", "w") as f:
            for a in ("x", "y", "z"):
                f.create_dataset(a, data=flow.coordinate(a)).attrs["units"] = length_label
            f.create_dataset("valid_fraction", data=counts / flow.n_times)
            for c in VELOCITY_COMPONENTS:
                rms = np.sqrt(variance[c])
                f.create_dataset(c + "_rms", data=rms).attrs["units"] = velocity_label
                intensity = np.full(flow.grid_shape, np.nan)
                np.divide(rms, np.abs(means["u"]), out=intensity,
                          where=np.isfinite(means["u"]) & (np.abs(means["u"]) > 1e-12))
                f.create_dataset(c + "_TI", data=intensity).attrs["units"] = "fraction"
        profiles = []
        for axis, dim in (("x", 2), ("y", 1), ("z", 0)):
            for i, coord in enumerate(flow.coordinate(axis)):
                sl = [slice(None)] * 3
                sl[dim] = i
                sl = tuple(sl)
                row = dict(axis=axis, coordinate=float(coord), coordinate_unit=length_label,
                           valid_fraction=_finite_mean(counts[sl] / flow.n_times),
                           tke=_finite_mean(tke[sl]))
                uref = _finite_mean(means["u"][sl])
                for c in VELOCITY_COMPONENTS:
                    var = _finite_mean(variance[c][sl])
                    rms = None if var is None else float(np.sqrt(var))
                    row[c + "_mean"] = _finite_mean(means[c][sl])
                    row[c + "_rms"] = rms
                    row[c + "_TI"] = rms / abs(uref) if rms is not None and uref and abs(uref) > 1e-12 else None
                profiles.append(row)
        _write_csv(output / "profiles.csv", profiles)

        limits = {a: min(flow.grid_shape[d] - 1, s.max_spatial_lag or flow.grid_shape[d] - 1)
                  for a, d in (("x", 2), ("y", 1), ("z", 0))}
        accum = {(c, a): np.zeros((4, limits[a] + 1)) for c in VELOCITY_COMPONENTS
                 for a in ("x", "y", "z") if spacing[a] is not None}
        probe_indices = []
        if s.temporal:
            # Fixed grid locations; gaps remain gaps. No ROI-averaged signal is used.
            axes = [np.unique(np.linspace(0, n-1, s.probe_grid+2, dtype=int)[1:-1])
                    for n in flow.grid_shape]
            probe_indices = [tuple(int(v) for v in p) for p in np.array(
                np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T if good[tuple(p)]]
        probes = np.full((flow.n_times, len(probe_indices), 3), np.nan)
        block_edges = np.linspace(0, flow.n_times, s.blocks + 1, dtype=int)
        # Per-block local means and variances distinguish drift from spatial shear.
        block_sum = np.zeros((s.blocks, 3, *flow.grid_shape))
        block_square = np.zeros_like(block_sum)
        block_count = np.zeros((s.blocks, *flow.grid_shape), dtype=np.int64)
        for start in range(0, flow.n_times, s.chunk_size):
            stop = min(start + s.chunk_size, flow.n_times)
            raw = {c: flow._file[c][start:stop, :, :, :] for c in VELOCITY_COMPONENTS}
            valid = valid_vector_samples(raw, s.invalid_samples) & good[None]
            for b in range(s.blocks):
                lo, hi = max(start, block_edges[b]), min(stop, block_edges[b+1])
                if lo >= hi:
                    continue
                take = slice(lo-start, hi-start)
                block_count[b] += valid[take].sum(axis=0)
                for ci, c in enumerate(VELOCITY_COMPONENTS):
                    # Center on the full-record local mean for numerical stability.
                    f = np.where(valid[take], raw[c][take] - means[c], 0.)
                    block_sum[b, ci] += f.sum(axis=0)
                    block_square[b, ci] += (f*f).sum(axis=0)
            for ci, c in enumerate(VELOCITY_COMPONENTS):
                fluct = np.where(valid & fluctuating[c][None], raw[c] - means[c][None], np.nan)
                for axis, dim in (("x", 3), ("y", 2), ("z", 1)):
                    if (c, axis) in accum:
                        accum[c, axis] += pair_moments(fluct, dim, limits[axis])
                for pi, p in enumerate(probe_indices):
                    probes[start:stop, pi, ci] = np.where(valid[(slice(None), *p)],
                                                         raw[c][(slice(None), *p)], np.nan)
            if start == 0 or stop == flow.n_times or start // 200 != stop // 200:
                print(f"Correlations: {stop}/{flow.n_times} frames", flush=True)
        block_rows = []
        for b in range(s.blocks):
            count = block_count[b]
            valid_block = count >= max(2, s.min_valid_fraction * (block_edges[b+1]-block_edges[b]))
            for ci, c in enumerate(VELOCITY_COMPONENTS):
                delta = np.full(flow.grid_shape, np.nan)
                second = np.full(flow.grid_shape, np.nan)
                np.divide(block_sum[b, ci], count, out=delta, where=valid_block)
                np.divide(block_square[b, ci], count, out=second, where=valid_block)
                var = _finite_mean(np.maximum(second - delta**2, 0))
                block_rows.append(dict(block=b, start_frame=int(block_edges[b]+s.start),
                                       stop_frame_exclusive=int(block_edges[b+1]+s.start), component=c,
                                       eligible_voxels=int(valid_block.sum()),
                                       mean=_finite_mean(means[c]+delta),
                                       rms=None if var is None else float(np.sqrt(var))))
        _write_csv(output / "stationarity_blocks.csv", block_rows)
        curves, scales = [], []
        for c in VELOCITY_COMPONENTS:
            for axis in ("x", "y", "z"):
                key = (c, axis)
                if key not in accum:
                    scales.append(dict(kind="spatial", component=c, direction=axis, probe=None,
                                       integral=None, truncated_integral=None, cutoff=None,
                                       status="singleton_axis", unit=length_label,
                                       length_m=None, length_over_D=None))
                    continue
                lag = np.arange(limits[axis]+1) * spacing[axis]
                rho = normalized_correlation(accum[key], s.min_pairs, s.min_pair_fraction)
                result = integrate_correlation(lag, rho)
                length = result["integral"] if s.length_unit != "native" else None
                scales.append(dict(kind="spatial", component=c, direction=axis, probe=None,
                                   **result, unit=length_label, length_m=length,
                                   length_over_D=length/s.rotor_diameter_m if length is not None and s.rotor_diameter_m else None))
                for j in range(len(lag)):
                    curves.append(dict(kind="spatial", component=c, direction=axis, probe=None,
                                       lag=float(lag[j]), lag_unit=length_label,
                                       correlation=float(rho[j]) if np.isfinite(rho[j]) else None,
                                       pairs=int(round(accum[key][3, j]))))
        probe_rows = []
        temporal_checks = []
        for pi, p in enumerate(probe_indices):
            probe_rows.append(dict(probe=pi, z_index=p[0], y_index=p[1], x_index=p[2],
                                   x=float(flow.coordinate("x")[p[2]]),
                                   y=float(flow.coordinate("y")[p[1]]),
                                   z=float(flow.coordinate("z")[p[0]])))
            for ci, c in enumerate(VELOCITY_COMPONENTS):
                series = probes[:, pi, ci]
                limit = min(s.max_temporal_lag or flow.n_times//4, flow.n_times-1)
                fluct = series-means[c][p] if fluctuating[c][p] else np.full_like(series, np.nan)
                moments = pair_moments(fluct, 0, limit)
                rho = normalized_correlation(moments, s.min_pairs, s.min_pair_fraction)
                lag = np.arange(limit+1) * dt
                result = integrate_correlation(lag, rho)
                speed = s.convection_speed_m_s
                if speed is None and s.velocity_unit != "native" and means["u"][p] > 0:
                    speed = float(means["u"][p])
                length = result["integral"] * speed if result["integral"] is not None and speed else None
                scales.append(dict(kind="temporal", component=c, direction="x_advection", probe=pi,
                                   **result, unit="s", length_m=length,
                                   length_over_D=length/s.rotor_diameter_m if length is not None and s.rotor_diameter_m else None))
                temporal_checks.append(dict(probe=pi, component=c, convection_speed_m_s=speed,
                                            sigma_u_over_U=float(np.sqrt(variance["u"][p])/abs(means["u"][p])) if means["u"][p] else None,
                                            duration_over_integral=(flow.n_times-1)*dt/result["integral"] if result["integral"] else None,
                                            integral_over_dt=result["integral"]/dt if result["integral"] else None))
                for j in range(len(lag)):
                    curves.append(dict(kind="temporal", component=c, direction="x_advection", probe=pi,
                                       lag=float(lag[j]), lag_unit="s",
                                       correlation=float(rho[j]) if np.isfinite(rho[j]) else None,
                                       pairs=int(round(moments[3, j]))))
        _write_csv(output / "correlations.csv", curves)
        _write_csv(output / "integral_scales.csv", scales)
        _write_csv(output / "probes.csv", probe_rows)
        _write_csv(output / "temporal_checks.csv", temporal_checks)
        if s.temporal:
            with h5py.File(output / "probe_timeseries.h5", "w") as f:
                f.create_dataset("t", data=flow.coordinate("t")).attrs["units"] = "s"
                f.create_dataset("velocity", data=probes).attrs["units"] = velocity_label
                f.attrs["dimension_order"] = "time, probe, component(u,v,w)"
        summary = dict(eligible_voxels=int(good.sum()), total_voxels=int(good.size),
                       mean_valid_fraction=float(np.mean(counts/flow.n_times)),
                       mean_velocity={c: _finite_mean(means[c]) for c in VELOCITY_COMPONENTS},
                       rms={c: float(np.sqrt(_finite_mean(variance[c]))) for c in VELOCITY_COMPONENTS},
                       mean_tke=_finite_mean(tke), velocity_unit=velocity_label,
                       length_unit=length_label, temporal_probes=len(probe_indices), scales=scales)
        warnings = ["Pooled spatial correlations assume approximate homogeneity within the selected ROI.",
                    "A first-zero estimate is a finite-record convention, not a convergence guarantee.",
                    "Pair counts are support counts, not independent sample counts or confidence intervals."]
        if "interpol" in flow.source_attrs.get("operation", ""):
            warnings.append("Input was interpolated; compare with the original export. Fill exclusion cannot undo earlier binning/filtering.")
        if s.length_unit == "native" or s.velocity_unit == "native":
            warnings.append("Some units are unconfirmed; native values must not be interpreted as SI.")
        if s.temporal:
            warnings.append("Temporal mode assumes time-resolved sampling; all U*T lengths describe streamwise advection, including v and w.")
            if not probe_indices:
                warnings.append("No automatic probes met the coverage threshold; select a smaller ROI or change probe-grid.")
        summary["warnings"] = warnings
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        _plot_diagnostics(output, curves, profiles, block_rows, length_label, velocity_label)
        manifest["status"] = "complete"
        manifest["products"] = sorted(p.name for p in output.iterdir() if p.is_file())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return output


def _plot_diagnostics(output, curves, profiles, blocks, length_label, velocity_label):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig = Figure(figsize=(12, 9), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 3)
    for ci, c in enumerate(VELOCITY_COMPONENTS):
        for ai, axis in enumerate(("x", "y", "z")):
            rows = [r for r in curves if r["kind"] == "spatial" and r["component"] == c and r["direction"] == axis]
            ax = axes[ci, ai]
            ax.plot([r["lag"] for r in rows], [np.nan if r["correlation"] is None else r["correlation"] for r in rows])
            ax.axhline(0, color="0.5", lw=.7)
            ax.set(xlabel=f"{axis} separation [{length_label}]", ylabel=f"R_{c}{c}", ylim=(-1.05, 1.05))
            ax.grid(alpha=.2)
    fig.suptitle("Spatial correlations: positive-lobe integration; inspect domain truncation")
    fig.savefig(output / "spatial_correlations.png", dpi=160)
    fig = Figure(figsize=(15, 4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 4)
    vertical = [r for r in profiles if r["axis"] == "y"]
    for c in VELOCITY_COMPONENTS:
        axes[0].plot([r[c+"_mean"] for r in vertical], [r["coordinate"] for r in vertical], label=c)
        axes[1].plot([r[c+"_TI"] for r in vertical], [r["coordinate"] for r in vertical], label=c)
        rows = [r for r in blocks if r["component"] == c]
        axes[2].plot([r["block"] for r in rows], [r["rms"] for r in rows], "o-", label=c)
    axes[0].set(xlabel=f"Mean velocity [{velocity_label}]", ylabel=f"Vertical y [{length_label}]")
    axes[1].set(xlabel="Plane RMS / |plane mean u|", ylabel=f"Vertical y [{length_label}]")
    axes[1].xaxis.set_major_locator(MaxNLocator(nbins=4))
    axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    axes[2].set(xlabel="Time block", ylabel=f"RMS [{velocity_label}]")
    axes[3].plot([r["valid_fraction"] for r in vertical], [r["coordinate"] for r in vertical])
    axes[3].set(xlabel="Mean valid fraction", ylabel=f"Vertical y [{length_label}]", xlim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    for ax in axes[:3]:
        ax.legend()
    fig.savefig(output / "flow_diagnostics.png", dpi=160)
    temporal = [r for r in curves if r["kind"] == "temporal"]
    if temporal:
        fig = Figure(figsize=(12, 4), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        for c, ax in zip(VELOCITY_COMPONENTS, axes):
            for probe in sorted({r["probe"] for r in temporal}):
                rows = [r for r in temporal if r["probe"] == probe and r["component"] == c]
                ax.plot([r["lag"] for r in rows],
                        [np.nan if r["correlation"] is None else r["correlation"] for r in rows],
                        label=f"Probe {probe}")
            ax.axhline(0, color="0.5", lw=.7)
            ax.set(xlabel="Time lag [s]", ylabel=f"R_{c}{c}", ylim=(-1.05, 1.05))
            ax.legend(fontsize="small")
            ax.grid(alpha=.2)
        fig.savefig(output / "temporal_correlations.png", dpi=160)
