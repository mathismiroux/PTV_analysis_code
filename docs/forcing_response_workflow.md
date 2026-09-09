# Local forcing-response energy budgets

This analysis answers how much local velocity fluctuation energy is explained
by the forcing frequency and by a repeatable phase-dependent waveform. It reads
the existing velocity time series directly; it neither stitches recordings nor
uses a truncated POD reconstruction. No raw interpolation is rerun. Existing
interpolated values remain included unless `--exclude-filled` is specified.

## Run commands

New local PODs and energy profiles, beside each existing `mean.nc`:

```powershell
.\scripts\pod_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-subfolder pod_energy_valid90 --analysis both --modes 20 --iterations 4 --min-valid-fraction 0.90 --slab-width-d 0.1
```

Add only energy profiles to already completed `pod_valid90` directories:

```powershell
.\scripts\pod_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-subfolder pod_valid90 --analysis energy --min-valid-fraction 0.90 --slab-width-d 0.1
```

Use a new subfolder name if the target products already exist. `--analysis
energy` permits an existing POD directory but refuses to overwrite its energy
subfolder. `--dry-run` records output paths, inferred frequencies and requested
bounds without running either calculation. Each selected stage is attempted
independently: a POD failure does not prevent trying energy analysis. Failures
are recorded in the manifest and produce a nonzero batch exit code.

For one volume, independent of the batch:

```powershell
python scripts/analyze_forcing_response.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\SurgeLF_3.5D\interpolated_velocity.nc" --output outputs/forcing_surgeLF_3p5D --min-valid-fraction 0.90 --slab-width-d 0.1
```

## Spatial regions and outputs

The batch automatically uses the intersection of all matched velocity files'
y-z coordinate intervals as the **requested** energy-analysis window. This
choice is reported before computation and saved in every summary. Change it
explicitly with `--y-bounds-d MIN MAX --z-bounds-d MIN MAX`. Coordinates are
relative to `--rotor-y` and `--rotor-z` (both zero by default). `--rotor-diameter`
defaults to 1200 in source units (mm for these measurements). The standalone
command defaults to its own volume's coordinate limits instead. x/D uses the
source x origin; no streamwise origin shift is applied.

The POD itself retains its original full-volume domain. Consequently, its
mean-filled energy and these valid-observation, restricted-region energies are
different estimates and should not be directly equated.

Each case's output directory contains `forcing_response/` with:

- `native_planes.csv`: energy budget and coverage at each native x-plane.
- `downstream_slabs.csv`: clipped cell-volume averages in slabs anchored at
  `--slab-origin-d` (default zero), width `--slab-width-d` (default 0.1D).
- `volume_energy.csv`: the budget over that volume's supported requested window.
- `energy_profiles.png`: energy, energy ratios and coverage versus x/D. Thin
  lines are native-plane estimates; marked lines are slab estimates.
- `energy_maps.h5`: voxel energies, support, valid fractions, phase-bin minimum
  counts, observed-cycle counts and three-component harmonic coefficients.
- `summary.json`: source, settings, estimator definitions and volume budget.

The batch also writes a combined `*_manifest_downstream_slabs.csv` beside its
manifest. It preserves case and source columns; overlapping recordings remain
separate rows, with no phase alignment or merging.

All three components are used. Spatial weights are cell areas or volumes,
clipped to the requested region. Boundary cells terminate at the first/last
measured coordinate, so boundary weights are half-width. Slabs crossing the
source x boundaries are partial; `axial_coverage_fraction` reports this. Slab
coverage is supported volume divided by the entire requested slab volume,
including its unmeasured axial portion. Native coverage is supported area
divided by requested area. Measures use D squared or D cubed, respectively.

Energy ratios are ratios of spatially averaged energies, not averages of local
ratios. A common requested rectangle does **not** ensure identical supported
cross-sectional areas. Coverage is always shown, with no hidden threshold that
discards low-coverage planes. Check it before interpreting downstream trends.
Repeating the analysis with 0.05D or 0.2D slabs requires a new output name.

## Valid-observation estimators

At a voxel, the validity mask requires a finite u-v-w vector and excludes any
declared fill values. Optional all-zero-vector and interpolation-mask exclusions
follow the CLI settings. The voxel must have at least the selected fraction of
valid snapshots. Missing snapshots are **not filled**. All energies at that
voxel use the same observed sample set and population normalization 1/N.

The temporal mean is calculated from those same observations. This calculation
is necessary for the energy accounting and may differ from an existing mean
file if its sample-validity policy differs. The POD stage still reuses the
existing `mean.nc` as before.

For observed velocity vectors q_i and their mean qbar:

`E_total = (1 / (2N)) sum_i |q_i - qbar|^2`.

**Forcing-frequency energy:** independently regress each velocity component on
`1, cos(2*pi*f*t + offset), sin(2*pi*f*t + offset)` using the actual observed
times. `forcing_energy` is half the summed observed-time variance of the fitted
prediction. Its ratio to total energy is the explained-variance fraction and
lies between zero and one, up to numerical roundoff. A constant phase offset
does not change that fit's amplitude or explained energy.

`forcing_uniform_cycle_energy = sum_components(a^2+b^2)/4` is also exported.
This is the fitted harmonic's energy over a uniformly sampled cycle. With
uneven phase sampling it differs from the observed-time explained energy and
is **not** used as the bounded forcing/total ratio.

**Phase-locked and residual energies:** place observed samples into phase bins
(`--phase-bins 32`, `--phase-offset 0` by default). Require at least
`--min-phase-samples 3` valid observations in every bin at the voxel. Compute
the bin means from the observed vectors. Weight their energy relative to the
overall mean by observed bin counts. The residual is the within-bin variance.
The finite-sample identity is exactly

`E_total = E_phase_locked + E_residual`.

This count-weighted in-sample partition differs from the equal-phase global
POD previously used. Residual energy contains turbulence, measurement noise,
run variability and finite-bin waveform variation. The continuous harmonic
fit and the piecewise-constant phase-bin fit are separate regressions:
forcing energy can slightly exceed phase-bin energy. Do not calculate
"higher-harmonic energy" by simply subtracting those columns.

## Forcing-frequency selection

`--frequency-hz` explicitly overrides every matched recording. Without it,
Static cases have no forcing reference and export total energy/coverage only;
periodic fields and ratios are NaN, not invented zeros. For other cases:

1. Use positive `frequency_hz` metadata on the velocity file, checking it
   against known experiment labels.
2. Otherwise use the documented experiment convention: SurgeLF/PitchLF = 2 Hz,
   SurgeHF/PitchHF = 5 Hz.
3. For unknown labels, use a unique adjacent phase-average frequency. Multiple
   different candidates are an error; absent frequency gives total-only results.

The convention avoids accidentally selecting an old 2 Hz phase product in a
PitchHF folder containing both 2 Hz and 5 Hz products. The chosen frequency and
its provenance are recorded. Existing phase-average fields are not reused for
the energy partition because the raw observed samples are needed for consistent
weights, residual energy and cross-validation. Time coordinates must be seconds.

## Finite-cycle diagnostic and remaining limitations

Phase averages estimated on the same data can absorb random fluctuations.
Therefore, each observed forcing cycle is held out in turn. Predict its samples
using phase-bin means from the other cycles and calculate
`cv_phase_prediction_error`. Also predict using the temporal mean from the
other cycles and calculate `cv_mean_prediction_error` on the same held-out data.

`cv_phase_predictive_fraction = 1 - cv_phase_prediction_error / cv_mean_prediction_error`.

This is a held-out predictive score, not an additive energy component or a
confidence interval. Negative values are retained: the phase model then
predicts worse than the training-mean baseline. If a training fold has no data
for a required phase bin, CV is unavailable at that voxel. A spatially averaged
CV score is withheld when any supported contributing voxel lacks valid CV,
rather than silently changing the averaging support.

This diagnostic does not establish independent experimental repeatability or
remove temporal dependence between successive cycles. Partial first/last
cycles are kept as observed groups. No startup transient removal, cycle-block
bootstrap confidence intervals or measurement-noise correction is performed.
Phase-dependent missingness, interpolation smoothing and limited records can
bias the estimates. Inspect coverage and perform sensitivity/convergence checks
before using energy differences as journal-paper conclusions.

## Tests

```powershell
python -m pytest tests/test_forcing_response.py tests/test_pod.py -q
```

Validation includes known energy partitions, missing-sample sinusoidal fits,
leave-one-cycle-out results against a direct reference calculation, slab
integration, coverage, Static handling and adding energy profiles to existing
POD directories without changing their POD files.
