# Free-flow characterization for floating-wind wake experiments

This workflow computes mean flow, turbulence intensity, TKE, Reynolds stresses,
and directional integral-scale estimates from the existing `(time, z, y, x)`
velocity exports. It reuses `FlowDataset`, `temporal_average_volume`,
`turbulent_kinetic_energy`, `reynolds_stresses`, and vector validity handling.
The new calculation lives in `ptv_flow/freeflow.py`; its command is
`scripts/characterize_freeflow.py`.

The intended input is a **no-turbine free-flow acquisition**. A static-turbine wake
can exercise the software, but its statistics do not establish the free-flow
conditions. The supplied `Static_1.625D` dataset is treated as a trial dataset.

## Run the workflow

From the repository root, with the existing dependencies installed:

```powershell
python scripts/characterize_freeflow.py "D:\path\to\freeflow.nc" `
  --output outputs/freeflow_baseline `
  --length-unit mm --velocity-unit m/s --time-unit s
```

Use a new output directory each time. The workflow refuses an existing one and
does not modify the source. A folder argument selects its
`interpolated_velocity.nc`; a file argument accepts any compatible raw export.

For **confirmed time-resolved** measurements, add:

```powershell
  --temporal --probe-grid 2 --rotor-diameter-m 1.2
```

The example diameter is the current case-registry value; set it to the diameter
appropriate for the experiment. This optional parameter only supplies `L/D`.
Input units are explicitly selected. Known units are converted to metres,
seconds, and metres per second in all products. Without unit arguments, spatial
analysis remains in native units; it does not silently assume SI. Temporal
analysis requires a known time unit. Converting a time scale to metres also
requires known velocity units or an explicit `--convection-speed-m-s`.

To restrict the measurement region or remove a startup interval:

```powershell
python scripts/characterize_freeflow.py "D:\path\to\freeflow.nc" `
  --output outputs/freeflow_central_region `
  --length-unit mm --velocity-unit m/s --time-unit s `
  --x-range 1600 2200 --y-range 0 700 --z-range -300 300 `
  --start 400 --stop 3600 --temporal
```

Spatial bounds are inclusive and always use **input coordinate units**. `--start`
is inclusive; `--stop` is exclusive. The same region, time interval and validity
rules apply to means, stresses and correlations. Selections are lazy views;
the workflow does not create a duplicate velocity volume. Grid coordinates may
ascend or descend, but must be nearly uniformly spaced. Temporal mode also
requires uniformly spaced, increasing timestamps. No resampling is performed.

## Scientific interpretation

The integral scale adds information beyond turbulence intensity. In a controlled
wind-farm LES study, varying inflow integral scales while maintaining velocity
and TI changed wake development and power production. This motivates recording
scales alongside TI when comparing floating-wind experiments; it does not imply
the study's numerical effects transfer to this experiment.
[Hodgson, Troldborg & Andersen, *Renewable Energy*, 2025](https://doi.org/10.1016/j.renene.2024.121804).

The coordinate convention is **x streamwise, y vertical, z lateral**, with velocity
components u, v, w respectively. File storage order does not alter this convention.

At each voxel, the selected-record temporal mean is subtracted:

\[
u_i'(\mathbf{x},t)=u_i(\mathbf{x},t)-\overline{u_i}(\mathbf{x}).
\]

The workflow reports all nine same-component spatial correlations: u, v and w
each separated along x, y and z. For a separation r, it pools valid endpoint
pairs over time and locations within the selected region:

\[
\widehat R_{ii}^{(j)}(r)=
\frac{\sum_{P(r)}u_i'(\mathbf{x},t)u_i'(\mathbf{x}+r\mathbf e_j,t)}
{\sqrt{\sum_{P(r)}u_i'(\mathbf{x},t)^2
             \sum_{P(r)}u_i'(\mathbf{x}+r\mathbf e_j,t)^2}}.
\]

This is an explicitly chosen **pair-normalized pooled estimator**. It is bounded
by ±1 and gives R(0)=1 for nonzero variance. For homogeneous stationary flow it
approaches the usual covariance normalized by variance. With inhomogeneity or
lag-dependent missingness it can differ: contributing locations and endpoint
energies change with separation. It weights energetic, well-sampled locations
more strongly than an equal average of per-voxel correlation coefficients.
Use smaller regions if mean/RMS profiles show substantial spatial variation.
It is not an estimator of the entire two-point tensor in an inhomogeneous wake.
Component variances at the level of floating-point mean-subtraction roundoff
are excluded from correlations rather than interpreted as turbulent motion.

For every supported curve, the reported operational scale is

\[
\widehat L_{ii}^{(j)}=\int_0^{r_0}\widehat R_{ii}^{(j)}(r)\,dr,
\]

where r0 is its first zero crossing. Integration uses trapezoids and linearly
interpolates the crossing. This first-zero convention also appears in the
wind-energy study cited above. The theoretical infinite-lag integral and this
finite positive-lobe estimate need not coincide, especially with negative lobes
or coherent oscillations.

If no supported crossing occurs, **the integral estimate is blank/null**.
`truncated_integral` records the area only up to the last contiguous supported
lag. It is neither a converged length nor a rigorous lower bound. The code does
not fit an unobserved tail. Active-grid correlations can make ordinary integral
estimation difficult, including cases without a useful zero crossing; alternate
estimators require their own validation.
[Mora & Obligado, original research preprint, 2020](https://arxiv.org/abs/2005.06055).

### Temporal mode

Temporal correlations use fixed voxel signals at a regular interior probe grid,
with probes failing the full-record coverage threshold omitted. `--probe-grid 2`
requests up to eight probes; `--probe-grid 1` requests one interior probe. Exact
coordinates and indices are exported. A spatially averaged velocity time series
is not substituted for a point measurement. Missing timestamps remain in their
original positions; pairs with invalid endpoints are excluded without compressing
the timeline or filling gaps.

Time scales T use the same estimator and zero-crossing convention. Under Taylor's
frozen-flow assumption the workflow optionally reports L = Uc T. Uc is the
positive local mean streamwise velocity, or the supplied convection speed.
**All three temporal components yield streamwise-advection lengths**; the v and w
time scales do not measure vertical or lateral spatial lengths. The relation
between temporal and spatial correlations in this wind-energy context is given
by [Hodgson et al.](https://doi.org/10.1016/j.renene.2024.121804).

`temporal_checks.csv` provides sigma_u/|mean u|, T/dt, record-duration/T, and the
actual convection speed. Inspect these alongside correlation decay, block
variation and direct spatial lengths; none independently validates frozen flow.
Uniform timestamps alone do not establish that the acquisition resolves eddy
evolution. No temporal scales are calculated unless `--temporal` is supplied.

### Missing vectors and interpolation

The default excludes any vector with a nonfinite component. True zero components
remain valid. For exports using an all-zero vector as an invalid sentinel, use
`--invalid-samples zero-or-nan`. The validity mask is shared by all components,
means and stresses so that cross-component statistics use consistent support.

When available, `filled_mask` or component `*_filled_mask` datasets are excluded
by default; their union excludes the vector. `--include-filled` permits a
sensitivity run on the interpolated data. Exclusion only uses recorded masks;
it cannot recover unrecorded interpolation, particle-scale information or
features removed by voxel averaging. Finite field of view limits the observable
large scales, and measurement filtering limits the smaller scales. The precise
transfer function of this PTV binning pipeline has not been established here.
[Smits, *Journal of Fluid Mechanics*, 2022](https://doi.org/10.1017/jfm.2022.83)
discusses these measurement limits for PIV.

## Products and checks

| Product | Meaning |
|---|---|
| `mean.nc`, `tke.nc`, `reynolds_stresses.nc` | Existing statistical products, using the selected region and units |
| `statistics.h5` | Coordinates, vector coverage, component RMS and voxel TI |
| `profiles.csv` | Profiles along x, y, z, spatially averaged over each remaining plane |
| `correlations.csv` | Every lag, correlation coefficient and valid endpoint-pair count |
| `integral_scales.csv` | Nine spatial estimates and optional probe time scales, status, cutoff, truncated area and L/D |
| `stationarity_blocks.csv` | Contiguous time-block mean and RMS, calculated from local voxel moments |
| `spatial_correlations.png`, `flow_diagnostics.png` | Spatial decay, vertical profiles, block RMS and coverage plots |
| `temporal_correlations.png` | Optional fixed-probe temporal decay curves |
| `probes.csv`, `probe_timeseries.h5`, `temporal_checks.csv` | Optional point locations, masked signals and temporal diagnostics |
| `summary.json`, `manifest.json` | Machine-readable summary, settings, source metadata, units and completion state |

RMS uses population second moments, consistent with the existing functions.
Voxel TI is sigma_i/|local mean u| as a fraction. Profile TI is
sqrt(plane-average local variance)/|plane-average mean u|, and consequently
differs from the plane average of voxel TI. Spatial variation of the mean is
excluded from the turbulent variance. No isotropy assumption is used to infer
unmeasured components or substitute one length direction for another.

Default controls are engineering choices, not literature-certified thresholds:

- At least 50% valid snapshots per voxel (`--min-valid-fraction`).
- At least 100 pairs and 5% of zero-lag pairs at each lag (`--min-pairs`,
  `--min-pair-fraction`). Integration stops at the first unsupported lag.
- Spatial lags extend to the selected-domain span, optionally capped with
  `--max-spatial-lag` in grid intervals. Temporal lags default to N/4 frames,
  optionally changed with `--max-temporal-lag`.
- Four contiguous blocks provide a stationarity diagnostic. Block RMS removes
  each block's local mean. Block coverage can change, so also compare their
  `eligible_voxels` counts.

Before using a scale in a paper, inspect the curves, spatial profiles, coverage
and block statistics; repeat with sensible ROI, interval, lag-limit and coverage
choices. Compare original and interpolated exports. Pair counts share space and
time and are **not independent sample counts**. A crossing is not proof of
convergence, and this workflow does not provide calibrated confidence intervals,
automatic stationarity certification, spectral fits or dissipation estimates.
Replicate acquisitions and a justified temporal block-resampling study are
appropriate follow-up for uncertainty in the actual free-flow experiment.

## Verification and provenance

`tests/test_freeflow.py` compares FFT endpoint moments against explicit pair sums
with missing data in every spatial direction, checks interpolated zero crossings
and unresolved curves, exercises selection/unit/mask handling, and runs the
workflow on synthetic fields with analytic cosine correlation lengths. The
synthetic fields verify numerics, not the physical assumptions for experimental
data. A long synthetic AR(1) record additionally checks temporal integration
against its known exponential covariance scale, allowing finite-record scatter.
Tests also exercise the command, overwrite refusal, steady nonuniform flow,
zero variance and nonuniform-grid rejection.

Research reviewed on 7 September 2026. The central methodological references are
linked beside their claims. The finite-pair normalization, defaults, output
design and coverage rules above are implementation decisions made for this
workspace; they are not presented as a published standard.
