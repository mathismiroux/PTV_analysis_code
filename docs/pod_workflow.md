# Spatial POD of velocity volumes

For Flow/Static/moving-case comparisons on identical spatial support, see the
[common-mask preparation and review workflow](pod_common_mask_workflow.md).
It previews the intersection per downstream volume before an explicit POD run.

Run one case using the existing interpolated snapshots and the `mean.nc` beside
them (automatically detected):

```powershell
python scripts/pod_volume.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\SurgeLF_3.5D\interpolated_velocity.nc" --output outputs/pod_surgeLF_3p5D_existing_mean --modes 20 --iterations 4
```

Choose a new output directory for each run. `--mean-file PATH` selects a specific
existing mean. The script checks coordinates, snapshot times, shape and available
source metadata before reuse. Without an existing mean, it calculates one while
reading the velocity. No interpolation or phase averaging is repeated.

## Batch processing

The batch now runs **both local snapshot POD and downstream forcing-response
energy analysis** by default. Its valid-snapshot threshold defaults to 90%.
`scripts/pod_volume.py` remains a POD-only command with a 100% default threshold.
See [the energy-estimator guide](forcing_response_workflow.md) for definitions,
sampling assumptions, coverage and the held-out-cycle diagnostic.

To keep each volume's POD results beside its existing `mean.nc`, use
`--output-subfolder` instead of `--output-root`:

```powershell
.\scripts\pod_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-subfolder pod_valid90 --modes 20 --iterations 4 --min-valid-fraction 0.90
```

For example, this writes `SurgeLF_3.5D/pod_valid90/pod.h5` and its plots,
alongside `SurgeLF_3.5D/mean.nc`. The batch manifest is
`pod_valid90_manifest.json` in the input root. A dry run writes
`pod_valid90_dry_run_manifest.json` and does not create case output folders.
Existing results are protected: choose another subfolder name for a new run.
This option requires at most one matching velocity file per case directory.

Each case also gets a `forcing_response/` subdirectory with native x-plane
profiles, 0.1D slab profiles, voxel energy maps, a volume energy budget and a
profile plot. The energy estimates use valid observations rather than POD's
mean-filled gaps or its truncated modes. The batch defaults to the intersection
of the requested files' y-z coordinate ranges for the energy profiles; it does
not change the POD spatial domain. Actual supported coverage remains
case/position dependent and is reported.

To add energy profiles when `pod_valid90` already contains completed POD files:

```powershell
.\scripts\pod_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-subfolder pod_valid90 --analysis energy --min-valid-fraction 0.90 --slab-width-d 0.1
```

This preserves existing POD products and writes an independent
`pod_valid90_energy_manifest.json`. Existing `forcing_response` outputs are
protected. `--analysis pod` selects only the original POD stage.

For a folder containing case subdirectories with existing interpolated velocity
and mean files:

```powershell
python scripts/pod_folder.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-root outputs/pod_all_cases --modes 20 --iterations 4
```

The Windows batch launcher accepts the same arguments:

```powershell
.\scripts\pod_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/interpolated_velocity.nc" --output-root outputs/pod_all_cases --modes 20 --iterations 4
```

Use `--pattern "*.nc"` for a flat folder of velocity exports, or
`--pattern "**/interpolated_velocity.nc"` for deeper nesting. Each source gets
its own output directory preserving its relative path. Each case automatically
uses its own adjacent mean; do not pass a shared `--mean-file` for different cases.
Add `--dry-run` with a fresh output root to write a manifest without calculating
POD. Derived mean/phase files lacking a velocity time series are skipped.
Failures are recorded in `manifest.json`; remaining cases continue.

## Definition and interpretation

The state combines u, v, w over the full 3D volume. Subtract the temporal mean,
then weight each voxel by the square root of its cell volume divided by the
total retained volume. Equal weights are used for snapshots. With weighted
matrix X (spatial DOFs by time), snapshot covariance is X.T X / N.
Eigenvalues measure summed velocity variance; energy is eigenvalue / 2.
Each energy fraction divides by the **total** fluctuation variance, calculated
directly from X, including energy beyond the saved modes.

The solver uses a reproducible randomized time subspace, oversampling and power
iterations to avoid a full decomposition of multi-GB data. Leading modes are
approximations. `relative_eigen_residual` measures how well each computed
eigenpair satisfies the full covariance equation. Increase `--iterations` or
`--oversampling` if residuals are too large for your purpose. Twenty saved modes
do not necessarily describe all physically relevant structures, and POD energy
ranking alone does not identify a structure's physical mechanism or frequency.

The standalone POD's default support includes only voxels with finite velocity vectors in **every**
snapshot. Interpolated values are included. This gives a fixed support without
filling the remaining gaps. Check `spatial_coverage` and the saved support mask:
the energy describes that retained region, not unmeasured parts of the volume.
An explicit `--min-valid-fraction 0.95`, for example, retains more voxels but
replaces remaining missing samples with their temporal mean (zero fluctuation),
which can bias energies and modes. `--zero-invalid` treats all-zero vectors as
missing; `--exclude-filled` excludes values marked by `filled_mask`. These two
options currently require input without an adjacent existing mean, because its
validity policy may differ. Reused means should have been made from the same,
unchanged velocity file.

## Outputs

- `pod.h5`: full 3D `mode_u/v/w` arrays `(mode,z,y,x)`, `mean_u/v/w`,
  `temporal_coefficients` `(time,mode)`, coordinates, fixed support, spatial
  weights, valid fractions, eigenvalues, energy fractions and residuals.
- `modal_energy.csv`, `modal_energy.png`: individual and cumulative fractions.
- `mode_01_z0.png` through `mode_06_z0.png`: u/v/w slices nearest z=0.
- `summary.json`: input provenance, coverage, numerical settings and energy.

The reconstruction is mean_component + sum over modes of
temporal_coefficients[t, mode] * mode_component[mode,z,y,x]. Spatial modes are
normalized with the saved spatial weights and are dimensionless; coefficients
have the source velocity units. Mode sign is arbitrary. Coordinates retain the
source units. Excluded voxels are NaN in the mean and modes. A temporary local
float32 disk matrix uses about 12 * N_time * N_voxels bytes and is removed after
decomposition; covariance calculations use float64. Abruptly interrupted runs
may leave this temporary directory for manual cleanup.

## Comparing valid-snapshot thresholds

Increasing `--min-valid-fraction` retains fewer spatial voxels; it does not add
time snapshots. To assess sensitivity, keep the source, mean, mode count, seed,
oversampling and iteration count fixed, and use a fresh output directory at
each threshold. The SurgeLF 2.25D and 3.5D sensitivity runs use 20 modes,
4 iterations, oversampling 15 and seed 0 at 90%, 92% and 95%. Existing 90%
results are reused from `pod_energy_valid90`; new results use `pod_valid92`
and `pod_valid95` beside the processed files.

After computing the runs, create the comparison:

```powershell
python scripts/compare_pod_thresholds.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --cases SurgeLF_2.25D SurgeLF_3.5D --output outputs/pod_threshold_comparison
```

It checks matching solver settings and nested spatial support, then exports
individual/cumulative modal-energy plots, spatial coverage, CSV tables of the
first 1/2/5/10/20 modes' fractions, absolute spatially averaged fluctuation
energy and eigenpair residuals. Use a new comparison output directory if that
one already exists. No POD is recomputed by the comparison script.

Because each threshold changes the spatial region and normalization volume,
these results assess workflow sensitivity, not a physical change in the flow.
Modal rank does not establish a match between physical structures across runs;
near-degenerate modes may rotate. Saved high-order eigenpairs are approximate,
so small differences should be assessed alongside their numerical residuals.
