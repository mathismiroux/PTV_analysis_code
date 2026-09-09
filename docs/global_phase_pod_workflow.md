# Global phase-locked POD

This workflow combines **existing phase-average files**, then computes one exact
snapshot POD per forcing case. It does not run interpolation of raw snapshots,
temporal averaging, phase averaging, or independent-volume POD again.

## Run one full case

From the project root:

```powershell
python scripts/global_phase_pod.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --case SurgeLF --output-root outputs/global_phase_pod_valid90 --min-valid-fraction 0.90 --modes 20
```

The Windows launcher accepts the same arguments:

```powershell
.\scripts\global_phase_pod.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --case SurgeLF --output-root outputs/global_phase_pod_next --min-valid-fraction 0.90
```

Each run needs a new output root; the source files and earlier outputs are never
overwritten. Results appear in a case subdirectory, e.g. `SurgeLF/`.
Use `--dry-run` to check input metadata and write a manifest without computing
the global grid or POD. `--case` matches the complete forcing-case prefix in
folders named `SurgeLF_1D`, `SurgeLF_1.625D`, etc. All matching downstream
positions, including 1.625D, are included.

Use `--all-cases` instead of `--case` to process each discovered forcing case
independently. Static files without a valid forcing frequency are not suitable
for this workflow. If more than one phase file is found at the same location,
discovery refuses to choose silently. Narrow `--pattern`, for example:

```powershell
python scripts/global_phase_pod.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --case PitchHF --pattern "*/phase_average_5Hz.nc" --output-root outputs/global_pitchHF_valid90
```

The existing data folder contains both 2 Hz and 5 Hz phase products for some
PitchHF locations. Select the product appropriate to the experiment explicitly.
Files belonging to other case groups do not affect a single-case run.

## Synchronization and meaning of energy

The experimental assumption is that every acquisition starts at the same
platform position **and velocity**, with consistent trigger timing, and the
mean periodic flow response is repeatable between runs. The program checks
that all inputs share the same positive forcing frequency, uniform full-cycle
phase bins, frequency-based phase source, phase offset modulo 2 pi, and sample
validity policy. It cannot verify hardware trigger timing or repeatability from
these averaged files alone.

The phase convention is `theta = 2*pi*frequency*t + phase_offset`. No per-volume
phase shift is fitted: downstream propagation delays are preserved. Uniform
phase bins receive equal weight, regardless of differing sample counts. These
are phase-conditioned ensemble means, not simultaneously measured turbulence.

At each retained global voxel, subtract the **equal-phase cycle mean of the
stitched phase averages**. This reuses the phase files directly; the independent
`mean.nc` is not needed because its sample-weighted temporal mean can differ
slightly from this cycle mean. The resulting decomposition describes the
repeatable phase-locked response. It does not estimate the global instantaneous
turbulent covariance or total turbulent fluctuation energy.

For phase fluctuations q, the energy is

`E = 0.5 * mean_over_phase(sum_over_voxels(w * (u'^2 + v'^2 + w'^2)))`,

where the spatial weights sum to one. The exact phase-snapshot covariance is
only 32 by 32 for the existing inputs. Every non-negligible eigenvalue is saved
in the energy table; `--modes` controls how many full 3D modes are exported.
Fractions use **all** resolved phase-locked energy as their denominator.

## Coverage and spatial stitching

- The default `--min-valid-fraction 0.90` uses stored vector measurement counts:
  total valid samples across bins divided by total samples across bins. It is
  **not** a requirement that each individual phase bin be 90% complete.
- All three velocity components must also have finite phase means and positive
  counts in **every** phase bin. Stored means already reflect any validity mask
  used when making the phase-average files. No missing phase bins are filled
  with a temporal mean, and no data are recovered from discarded raw samples.
- A regular isotropic global grid uses the largest median input grid spacing.
  `--spacing VALUE` overrides it in source coordinate units. This assumes
  trilinear spatial variation between supported neighbouring measurements and
  can smooth small-scale structures. Output grid centres stay within source
  coordinate ranges. The grid does not represent a conservative finite-volume
  remapping; edge sampling and interpolation can slightly change energies.
- Trilinear interpolation requires every source corner with nonzero weight to
  be supported. No extrapolation is used. Unmeasured downstream gaps and holes
  remain unsupported, with NaN means and modes. Thus, the global POD spans the
  **measured supported union**, not all points in the bounding box.
- Overlaps are blended equally between contributing volume estimates, using
  spatial weights fixed across all phases. Each target voxel enters the global
  energy integral exactly once. Equal averaging treats the estimates as equally
  trustworthy; inspect disagreement before interpreting seams as flow physics.
- Target voxels use equal cell volumes `spacing**3`. Coverage is reported
  against the target-grid union of input coordinate boxes, excluding gaps
  between measurement boxes. Source coverage and target coverage differ
  because interpolation requires supported corners.

The memory guard defaults to two million target voxels; the primary phase
accumulator alone uses `3 * N_phase * N_voxels * 8` bytes. The actual peak is
higher while reading/interpolating and decomposing. Use coarser spacing for
larger domains or fewer available resources.

## Outputs

- `global_pod.h5`: coordinates; phase; cycle and phase means; 3D u/v/w modes;
  common phase coefficients; full energy spectrum; support and contributor
  count; fundamental cosine/sine fields; source energy allocations.
- `modal_energy.csv/png`: global individual and cumulative energy fractions.
- `global_mode_01_z0.png` through mode 6: full-case u/v/w mode slices nearest
  z=0, with unmeasured gaps left blank.
- `phase_coefficients_and_harmonics.png`, `modal_harmonics.csv`: shared
  coefficients and their **discrete forcing harmonics**. They are not broadband
  frequency spectra of the original recordings. Once phase-averaged, the data
  contain only integer forcing harmonics by construction; a peak at the
  fundamental alone does not establish predominance over unconditioned
  turbulence.
- `fundamental_u_z0.png`: streamwise fundamental amplitude and phase.
  With `u' = a*cos(theta) + b*sin(theta)`, the complex harmonic is `a-i*b`.
  Its phase is `atan2(-b,a)`; values wrap at +/-180 degrees. Phase is hidden in
  the plot where amplitude is below 1% of that slice's peak. All u/v/w cosine
  and sine fields remain available in HDF5. The summary gives the fundamental's
  energy across all three components and all retained grid points.
- `coverage_z0.png`, `source_coverage.csv`: spatial support diagnostics.
- `overlap_diagnostics.csv`: vector RMS differences between each incoming
  volume and previously blended overlapping volumes, for both the full phase
  means and their cycle-mean-subtracted response. Also includes each response's
  RMS, RMS difference normalized by their quadratic-mean RMS, and an aggregate
  fundamental phase difference and complex correlation magnitude. The latter
  compares the full complex harmonic vectors over the overlap; it is a spatial
  similarity measure, not statistical magnitude-squared coherence. A value near
  one with a nonzero phase difference can indicate a timing mismatch, but can
  also reflect run-to-run or registration differences. No phase correction is
  applied automatically. These are checks on agreement,
  not statistical confidence intervals or corrections to trigger timing.
- `source_mode_energy.csv`: each volume's allocated share of every global
  mode. Overlap energy is divided equally among contributors, so shares sum to
  one. This is a bookkeeping allocation, not an independent local POD energy.
- `summary.json`, root `manifest.json`: inputs, assumptions, numerical
  checks and completion status.

Spatial modes are dimensionless under the normalized spatial weighting;
phase coefficients retain velocity units. Reconstruction is
`cycle_mean_component + sum_mode(coefficient[phase, mode] * mode_component)`.
Modes are stored `(mode,z,y,x)` and phase means `(phase,z,y,x)`. Plot coordinates
use x/D and y/D; `--rotor-diameter` defaults to 1200 in the source units (mm for
these experimental files). No coordinate origin shift is applied.

## Validation

```powershell
python -m pytest tests/test_global_pod.py tests/test_pod.py -q
```

Tests cover overlap counting, known modal energies, field reconstruction,
orthogonality, fundamental extraction, missing-corner rejection, affine spatial
interpolation, phase-reference mismatches and complete-spectrum normalization.

## First SurgeLF validation run

Results with the extended overlap diagnostics are in
`outputs/global_phase_pod_valid90_checked/SurgeLF`. The run includes five
downstream volumes and retains 55,612 target voxels (28.61% of the target-grid
measurement footprint). Modes 1 and 2 contain 48.00% and 45.60% of phase-locked
energy; the fundamental contains 93.46%.

Treat this stitched result as provisional: the two supported joins show
aggregate fundamental phase differences of about -62 and +38 degrees, with
coherent vector RMS differences near 0.34 m/s. Their complex harmonic spatial
correlations are high (0.92 and 0.97), but their phases/amplitudes disagree.
These are the same overlapping physical regions, so the discrepancy should be
checked against trigger timing, coordinate registration and run repeatability.
The workflow does not remove it by fitting a phase shift. There is no overlap
check available across the unmeasured gaps to the farther downstream volumes.
