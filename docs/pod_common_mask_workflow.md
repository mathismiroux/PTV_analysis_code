# Common spatial support for local POD comparisons

Use `scripts/pod_common_mask.py` to compare Flow, Static, Pitch and Surge on one
fixed support at each downstream volume. The two commands deliberately separate
mask preparation from execution: `prepare` never calls the POD solver.

## Prepare and review

From the project root, using the existing paths on disk:

```powershell
python scripts/pod_common_mask.py prepare `
  --flow-folder "D:\binning64voxel50overlap\outputs_flow" `
  --cases-folder "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" `
  --pattern "*/interpolated_velocity.nc" `
  --min-valid-fraction 0.80 `
  --modes 100 --iterations 4 --oversampling 15 --seed 0 `
  --output-subfolder pod_common_valid80_modes100 `
  --output "outputs/pod_common_valid80_review"
```

Choose a new preview directory and a new POD output subfolder if those names
already exist. Preparation scans the velocity files in time chunks, then writes:

- `review.md`: group membership and common-mask coverage.
- `review.html`: standalone interactive viewer with volume, plane and slice
  controls; inspect the full saved 3D mask in a browser without a server.
- `coverage.csv`: retained counts and volume fractions per downstream location.
- `volume_*D_mask.png`: common-mask slices and spatial coverage diagnostics.
- `volume_*D_mask.npz`: exact target coordinates, common support, minimum temporal
  availability across cases, settings and source provenance.
- `plan.json`: solver settings, input fingerprints and every proposed output path.

Groups are identified from `Case_DISTANCE D` directory names without the space,
for example `PitchLF_3.5D`. Every case discovered in either root must be present
at every included downstream location. Missing cases and duplicate inputs are
errors, so group membership cannot change silently. No velocity or prior POD
files are modified during preparation. Empty intersections produce a blocked
preview that cannot be executed.

## Meaning of the mask

At each downstream location, use the Flow grid (`--reference-case Flow`) cropped
to the coordinate ranges shared by all cases. Align other grids to it with
linear interpolation only where their nodes differ; never extrapolate. A target
velocity vector is valid only when **all native vectors contributing to its
interpolation** are valid. Matching nodes are used directly.

Retain a target voxel when it has a valid vector in **at least 80% of snapshots
in every case**, and at least two valid samples. The 80% threshold is temporal
availability, not a requirement to retain 80% of the spatial volume. Reported
preview spatial coverage is relative to the common target grid. The NPZ also
contains the minimum temporal availability across cases for mask inspection.

Default validity requires finite u, v and w and excludes declared fill values.
Values previously interpolated in `interpolated_velocity.nc` count as available.
`--exclude-filled` additionally excludes `filled_mask` samples when that dataset
exists; `--zero-invalid` treats all-zero vectors as missing. Both options apply
identically to preparation and POD execution.

In the current dataset, PitchHF/PitchLF at 1D need z alignment to the Flow grid.
This extra interpolation can affect small-scale fluctuations and is reported in
`plan.json`. Coordinate and velocity unit declarations are checked when present;
physical registration and missing unit metadata still require experiment knowledge.

## Run only after reviewing the preview

To save aligned Pitch 1D velocity copies **without running POD**, use:

```powershell
python scripts/align_velocity_z.py `
  "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\PitchHF_1D\interpolated_velocity.nc" `
  "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\PitchLF_1D\interpolated_velocity.nc" `
  --reference "D:\binning64voxel50overlap\outputs_flow\Flow_1D\interpolated_velocity.nc" `
  --output-root "D:\binning64voxel50overlap\outputs\pitch_1D_flow_grid"
```

This creates `PitchHF_1D/interpolated_velocity.nc` and
`PitchLF_1D/interpolated_velocity.nc` under the new root. It preserves each
source's timestamps, requires matching x/y coordinates, and linearly aligns z
without extrapolation. A missing contributor makes the target vector missing.
`filled_mask` propagates upstream gap-filling flags; `grid_interpolated_z`
separately identifies planes introduced by grid alignment. Original files are
preserved. Existing roots are protected; interrupted writes retain a `.partial`
file instead of presenting an incomplete file as a finished result.

The existing common-mask plan still references the original input files and
performs the same alignment during its own read stage. Creating these separate
copies does not start or modify that plan.

```powershell
python scripts/pod_common_mask.py run "outputs/pod_common_valid80_review/plan.json"
```

To execute only the 1D volume (all its cases), append `--volumes 1`.
Other locations can be selected later, for example `--volumes 1.625 2.25 3.5 4.75`.
Each selection gets its own run manifest. Existing case outputs remain protected.

The run uses the saved settings and masks. It checks the discovered inputs,
source sizes/modification times, mask SHA-256 hashes and existing output paths
before starting. A changed source, changed mask, incomplete preview or existing
output requires a new preparation/output location. The run manifest records
success/failure; a failed execution stops rather than silently dropping a case.

Each case gets `pod_common_valid80_modes100/pod.h5` beside its interpolated
velocity file. All cases at a given location have identical output coordinates,
support and volume-normalized spatial weights. Temporal means are recomputed
separately from each case's valid aligned samples, rather than reusing `mean.nc`.
Independent recordings keep their own snapshot times; they are not synchronized.

Remaining temporal gaps are filled with that voxel's mean (zero fluctuation),
as in the original POD. **This implements common support, not gappy POD or
validated missing-data reconstruction.** The spectral-analysis and band-integration
scripts accept these standard `pod.h5` outputs; select the new POD folder in
their `--pattern` options.

The single-volume and existing batch POD commands also accept `--spatial-mask`
pointing to a prepared NPZ. Its source membership, validity settings and source
fingerprint must match, and `--mean-file` cannot be combined with it. Prefer the
saved-plan runner to avoid accidentally assigning the wrong mask to a volume.

Tests:

```powershell
python -m pytest tests/test_pod_common_mask.py tests/test_pod.py -q
```
