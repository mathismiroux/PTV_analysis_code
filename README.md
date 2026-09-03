# PTV Flow Visualization

Small Python tools for reading and visualizing 3D PTV velocity fields stored in
NetCDF4/HDF5 `.nc` files.

The expected file layout is:

```text
t                 (time)
z, y, x           (grid coordinates)
u, v, w           (velocity components)
u/v/w shape       (time, z, y, x)
```

The reader is lazy: it reads one frame or one `z` plane at a time instead of
loading the full multi-GB dataset into memory.

## Coordinate System

The physical coordinate convention used throughout this repository is:

```text
x: streamwise direction
y: vertical direction
z: lateral direction completing the right-handed coordinate system
```

The NetCDF velocity arrays are still stored in memory/file order as
`(time, z, y, x)`. That storage order does not change the physical meaning of
the coordinates.

For wake plots:

- `--rotor-y` is the vertical rotor-axis coordinate.
- `--rotor-z` is the lateral rotor-axis coordinate.
- `--plane-axis z` selects a constant lateral plane and plots vertical profiles.
- `--plane-axis y` selects a constant vertical plane and plots lateral profiles.

## Quick Start

From a fresh clone, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that the default local file can be read:

```powershell
python main.py
```

Read a specific file:

```powershell
python main.py "U:\FWExp data\DATA_2025_02_OJF__FW_PTV\postprocessed PTV data v2\Static_3.5D__b96.nc"
```

## Development Setup

Install the development dependencies to run tests and pre-commit hooks:

```powershell
python -m pip install -r requirements-dev.txt
```

Run the test suite manually:

```powershell
python -m pytest
```

Install the pre-commit hook once per clone:

```powershell
pre-commit install
```

After that, `python -m py_compile ...` and `python -m pytest` run automatically
before each commit.

The tests use `tests\data\tiny_flow.nc`, a 13 KB fixture extracted from the
local `Static_3.5D__b128f.nc` file. To regenerate it:

```powershell
python scripts\extract_test_fixture.py
```

## Command Reference

All commands use the same basic form:

```powershell
python main.py "path\to\file.nc" [options]
```

If no path is provided, the code uses `Static_3.5D__b128f.nc` in the repository
root.

### Case Registry Workflow

For paper postprocessing, use `cases.yaml` as the source of truth for case
metadata and file paths:

```powershell
python main.py --case static_x3p5d --cases-file cases.yaml --temporal-average
```

This reads the velocity file from the selected case and writes the default
case output:

```text
outputs\static_x3p5d\mean.nc
```

The YAML key, for example `static_x3p5d`, is the unique case selected at the
CLI and the default output folder name. Include `downstream_distance` in that
key when needed to distinguish repeated static, surge, or pitch exports. If
that output product already exists, the next default run writes to
`static_x3p5d_02`, `static_x3p5d_03`, and so on, unless `--overwrite` is
passed.

Case paths in `cases.yaml` are resolved relative to the registry file. Cases
whose data paths are not available yet can remain in the registry with `null`
file entries; they only fail when you try to process that specific case.

To compute the first complete statistical product set for one case:

```powershell
python main.py --case static_x3p5d --postprocess-basic
```

This writes:

```text
outputs\static_x3p5d\mean.nc
outputs\static_x3p5d\tke.nc
outputs\static_x3p5d\reynolds_stresses.nc
```

For several cases:

```powershell
python main.py --cases static_x0p6d static_x3p5d --postprocess-basic
```

`--postprocess-basic` computes the temporal mean first, then uses that exact
`mean.nc` to compute TKE and Reynolds stresses in the same output folder.

### Adding A New Case Safely

When you copy a new DaVis/exported `.nc` file into the project, do this before
running postprocessing:

1. Put the raw file somewhere stable.

   Good options are the repository root for quick local work, or a data folder
   outside Git. Large `.nc` files should not be committed.

2. Choose a unique `case_id`.

   The `case_id` is the YAML key in `cases.yaml`, the CLI name, and the default
   output folder. Include the downstream distance and any export/binned-data
   distinction needed to avoid ambiguity, for example:

   ```text
   static_x0p6d
   static_x3p5d_b96
   surge_st06_x1p5d_b128
   pitch_st15_x3p5d_davis_v2
   ```

3. Add or update the entry in `cases.yaml`.

   Minimal example:

   ```yaml
   cases:
     static_x3p5d_b96:
       label: "Static baseline, x/D=3.5, b96"
       motion_type: static
       downstream_distance: 3.5D
       frequency_hz: null
       reduced_frequency: null
       amplitude: null
       u_inf: 4.0
       rotor_diameter: 1.2
       rotor_frequency_hz: 8.0
       blade_passing_frequency_hz: 24.0
       files:
         velocity: "path/to/Static_3.5D__b96.nc"
         vorticity: null
         q_criterion: null
         phase_signal: null
   ```

   Relative paths are resolved relative to `cases.yaml`. Absolute paths are also
   allowed.

4. Check that the raw file can be read.

   ```powershell
   python main.py --case static_x3p5d_b96
   ```

   This prints the file shape, coordinate ranges, voxel size, and one frame
   summary. If the path or variables are wrong, fix this before postprocessing.

5. Run the basic postprocessing.

   ```powershell
   python main.py --case static_x3p5d_b96 --postprocess-basic
   ```

   This creates:

   ```text
   outputs\static_x3p5d_b96\mean.nc
   outputs\static_x3p5d_b96\tke.nc
   outputs\static_x3p5d_b96\reynolds_stresses.nc
   ```

6. If you rerun intentionally, decide whether you want a new folder or an
   overwrite.

   By default, existing products are not overwritten. A second run creates:

   ```text
   outputs\static_x3p5d_b96_02\mean.nc
   ```

   To replace the existing product instead:

   ```powershell
   python main.py --case static_x3p5d_b96 --postprocess-basic --overwrite
   ```

7. Use `--processing-id` when several processed versions exist.

   ```powershell
   python main.py --case static_x3p5d_b96 --processing-id static_x3p5d_b96_02 --tke
   ```

   This prevents accidentally using the wrong `mean.nc`.

### Inspect A Raw Time-Series File

Print file metadata, grid dimensions, coordinate ranges, voxel size, and one
frame summary:

```powershell
python main.py "path\to\raw_file.nc"
python main.py "path\to\raw_file.nc" --frame 100
```

Options:

- `--frame N`: choose the time index used for the printed velocity statistics.

### Animate A Raw Time-Series File

Animate one `z` plane from the raw 4D time series:

```powershell
python main.py "path\to\raw_file.nc" --animate --z 0 --fps 40
```

The visualization shows velocity magnitude as the color field and in-plane
`(u, v)` velocity vectors as arrows. GIF export works with the default
dependencies; MP4 export needs FFmpeg.

Options:

- `--z VALUE`: requested `z` plane. The nearest available plane is used.
- `--fps VALUE`: animation frame rate.
- `--start N`: first time index to animate.
- `--stop N`: one-past-last time index to animate.
- `--step N`: time-index stride. Use `--step 2` to animate every other frame.
- `--quiver-step N`: arrow spacing. Larger values draw fewer arrows.
- `--save path.gif`: save a GIF instead of opening an interactive window.

### Inspect Raw Values And Averages

Open an interactive inspection GUI for a raw time-series file:

```powershell
python main.py "path\to\raw_file.nc" --inspect --z 0 --frame 0
python main.py "path\to\raw_file.nc" --inspect --z 0 --frame 0 --min-valid-fraction 0.8
```

In the GUI, use the sliders to choose the frame and minimum valid fraction.
The `z` plane is fixed when the GUI opens from the `--z` command-line value.
Click a cell in the slice to inspect:

- the selected indices and coordinates
- raw `u`, `v`, `w`, and speed at the selected frame
- selected-voxel temporal mean computed on demand from the raw time series
- selected-voxel phase coverage, when phase information is available

The inspector does not compute a full averaged volume. It only reads the time
series for the one clicked voxel, which is why this value appears quickly.
The minimum valid fraction can be changed interactively in the GUI. The
selected-voxel means update immediately and are shown as accepted or rejected
using the corresponding valid-count threshold. In the displayed slice, the
raw frame colors stay unchanged, rejected cells are marked with a translucent
red overlay, and their arrows are suppressed. The side panel still reports the
raw value at the selected frame. Passing
`--min-valid-fraction` sets the initial slider value.

To check whether phase averaging is well populated at one location, pass the
motion frequency or use a case that already defines `frequency_hz`:

```powershell
python main.py --case surge_st06_x3p5d --inspect --z 0 --n-phase-bins 16 --min-valid-fraction 0.8
python main.py "path\to\raw_file.nc" --inspect --frequency-hz 2.0 --n-phase-bins 16
```

The side panel then shows one row per phase bin with the number of raw samples
in that bin, the minimum valid count required by the current valid-fraction
slider, the finite/non-NaN `u`, `v`, and `w` counts at the clicked voxel, and
the phase-averaged `u_bar`, `v_bar`, and `w_bar` values. This makes it visible
when a point has acceptable temporal coverage overall but poor coverage in one
or more phase bins.

If you already computed `phase_average.nc`, inspect its stored phase-counts
directly without reopening the raw file:

```powershell
python main.py outputs\surge_st06_x3p5d\phase_average.nc --inspect-phase-average --x 0 --y 0 --z 0 --min-valid-fraction 0.8
```

This prints the same per-bin count table for the nearest stored grid point.

If you already computed a temporal-average output file, pass it too:

```powershell
python main.py "path\to\raw_file.nc" --inspect --z 0 --frame 0 --compare-average --average-file outputs\temporal_mean.nc
```

With `--compare-average --average-file`, the side panel also shows the stored
`u_mean`, `v_mean`, `w_mean`, and counts from the postprocessed file so you can
compare them with the value computed from the raw series. It also reports the
proportion of the full averaged volume, plus the currently shown `z` plane,
that is empty/under-covered at the current minimum valid fraction. A voxel is
counted as empty when any velocity component has fewer valid samples than the
current threshold.

The comparison file must have the same grid shape and `x`, `y`, `z`
coordinates as the raw file. This prevents accidentally comparing, for example,
a `b128` raw file with a `b96` temporal average.

### Prepare Mean Wake Products For Figures

Before plotting paper figures, create reusable 3D mean wake files for every raw
velocity NetCDF file in a folder. This keeps expensive averaging separate from
plotting:

```powershell
python scripts\prepare_mean_wake_products.py "D:\my_case_exports" --output-name mean_wake_figure1_dryrun --dry-run
python scripts\prepare_mean_wake_products.py "D:\my_case_exports" --output-name mean_wake_figure1_001 --invalid-samples zero-or-nan --min-valid-fraction 0.8 --chunk-size 50
```

You must provide either `--output-name` or `--output-root`. The output folder
must not already exist; this analysis refuses to write into an existing folder,
including with `--overwrite`. Choose a new run name for every processing
attempt.

With `--output-name`, outputs are written next to the input folder:

```text
D:\
  my_case_exports\
    case_a.nc
    case_b.nc
  outputs\
    mean_wake_figure1\
      case_a\mean.nc
      case_b\mean.nc
      manifest.csv
      manifest.json
```

Each `mean.nc` contains `x`, `y`, `z`, `u_mean`, `v_mean`, `w_mean`,
`speed_from_mean`, `abs_U`, `u_count`, `v_count`, `w_count`, `u_over_u_inf`,
`wake_deficit`, and `wake_mask_u09` when `--u-inf` is provided. File attributes
and the manifest record the source file, command line, invalid-sample policy,
zero-mask mode, chunk size, minimum valid fraction, and overwrite setting.

Useful options:

- `--pattern "*.nc"`: choose which files in the input folder are processed.
- `--output-name NAME`: required unless `--output-root` is used; write to
  `..\outputs\NAME` relative to the input folder. `NAME` must be new.
- `--output-root path`: explicitly choose the output folder. The folder must
  be new.
- `--dry-run`: write the manifest and show what would be processed.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose which raw samples are
  excluded.
- `--zero-mask {component,vector}`: choose independent component masking or
  whole-vector masking.
- `--min-valid-fraction F`: set means to `NaN` if too few samples are valid.
- `--chunk-size N`: number of time frames loaded at once.
- `--u-inf 4.0`: free-stream velocity used for wake-recovery products.
- `--overwrite`: only passed through to individual file writing after the new
  output folder has been accepted. It does not allow reusing an existing output
  folder.

### Compute A Temporal Average

Create a 3D mean velocity volume from a raw 4D time series. By default,
`NaN` and infinite samples are ignored in the average.

```powershell
python main.py "path\to\raw_file.nc" --temporal-average --output outputs\temporal_mean.nc
python main.py "path\to\raw_file.nc" --temporal-average --min-valid-fraction 0.8 --output outputs\temporal_mean_80pct.nc
```

Options:

- `--output path.nc`: output file for the averaged volume.
- `--overwrite`: allow replacing an existing output file. Without this flag,
  the command refuses to overwrite existing results.
- `--chunk-size N`: number of time steps read at once. Larger values can be
  faster but use more memory.
- `--zero-mask component`: default. When zeros are treated as invalid, apply
  the zero mask independently for `u`, `v`, and `w`.
- `--zero-mask vector`: when zeros are treated as invalid, reject a sample only
  when `u`, `v`, and `w` are all exactly zero.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose which raw samples are
  treated as missing. The default is `nan`. Use `zero` for old exports where
  missing values are exact zeros, or `zero-or-nan` during a transition period
  where both conventions may appear.
- `--min-valid-fraction F`: discard averaged values with fewer than this
  fraction of valid time samples. Use `--min-valid-fraction 0.8` to require
  80 percent valid data at each voxel/component.

The output file contains `x`, `y`, `z`, `u_mean`, `v_mean`, `w_mean`,
`u_count`, `v_count`, `w_count`, and `speed_from_mean`.

When run with `--case`, the output also contains plot-ready wake products:
`wake_deficit = (U_inf - u_mean) / U_inf` and
`wake_mask_u09 = u_mean / U_inf < 0.9`. The file attributes include case
metadata such as `case_id`, `motion_type`, `u_inf`, `rotor_diameter`,
`frequency_hz`, and rotor/blade-passing frequencies.

Processed files store provenance metadata in the file attributes and in a
`provenance` group, including the source file path, source file name, source
file size, creation time, operation, zero-mask mode, chunk size, and minimum
valid-count settings. They also store the `invalid_samples` mode used for the
analysis.

### Compute A Phase Average

Create phase-locked velocity fields over one imposed-motion cycle:

```powershell
python main.py --case surge_st06_x3p5d --phase-average --n-phase-bins 16
python main.py "path\to\raw_file.nc" --phase-average --frequency-hz 2.0 --output outputs\phase_average.nc
```

With `--case`, the phase is inferred from `frequency_hz` in `cases.yaml`, unless
the case provides `files.phase_signal`. For raw-file workflows, pass either
`--frequency-hz` or `--phase-signal`.

Options:

- `--n-phase-bins N`: number of equal phase bins over one cycle.
- `--frequency-hz F`: imposed sinusoidal motion frequency.
- `--phase-signal path`: one phase value per raw time step. Text, `.npy`, and
  HDF5 files with `phase`, `phase_signal`, or `phi` are supported.
- `--phase-offset R`: phase offset in radians added when phase is inferred from
  frequency.
- `--chunk-size`, `--zero-mask`, `--invalid-samples`, `--min-valid-fraction`,
  `--output`, and `--overwrite`: same meaning as temporal averaging.

The output file contains `phase`, `phase_degrees`, `u_phase_mean`,
`v_phase_mean`, `w_phase_mean`, matching phase counts, temporal means,
coherent fields such as `u_coherent = u_phase_mean - u_mean`, and
first-harmonic products `u_harmonic_a`, `u_harmonic_b`,
`u_harmonic_amplitude`, and `u_harmonic_phase` for each velocity component.
When `u_inf` is available, it also stores `wake_deficit_phase` and
`wake_deficit_coherent`.

Incomplete samples are handled independently at each voxel, phase bin, and
velocity component. With the default `--invalid-samples nan`, only finite values
are accumulated into the phase-bin sums and counts; NaNs do not contribute to
the numerator or denominator. A phase-bin mean is written only when its valid
count passes `--min-valid-fraction` for that bin. The coherent field is then the
phase-bin mean minus the valid-sample temporal mean at the same voxel. The
first-harmonic fit is computed as an explicit least-squares fit to
`offset + a cos(phase) + b sin(phase)` using the phase-bin means. Bins that are
NaN are skipped. The stored harmonic amplitude is `sqrt(a^2 + b^2)`, so the
constant offset/mean does not contribute to the amplitude. The harmonic fit is
therefore coverage-aware through the valid/NaN phase means, but it is not
weighted by the number of raw samples inside each valid bin.

To visually inspect the resulting phase-averaged velocity, open an interactive
plane viewer:

```powershell
python main.py outputs\surge_st06_x3p5d\phase_average.nc --phase-average-plane --plane z --plane-value 0 --quantity speed
python main.py outputs\surge_st06_x3p5d\phase_average.nc --phase-average-plane --plane y --plane-value 0 --quantity u
python main.py outputs\surge_st06_x3p5d\phase_average.nc --phase-average-plane --plane z --plane-value 0 --quantity u --phase-field coherent
```

The phase slider and arrow buttons move through the stored phase bins. Use
`--phase-field phase_mean` to see the phase-averaged velocity itself, or
`--phase-field coherent` to see the phase-locked fluctuation relative to the
mean.

To check the phase-locked waveform at one voxel, plot all three components
against phase:

```powershell
python main.py outputs\surge_st06_x3p5d\phase_average.nc --phase-voxel --x 4200 --y 0 --z 0 --phase-field phase_mean --min-valid-fraction 0.5
python main.py outputs\surge_st06_x3p5d\phase_average.nc --phase-voxel --x 4200 --y 0 --z 0 --phase-field coherent --save outputs\phase_voxel.png
```

The upper panel shows `u`, `v`, and `w` versus phase, closed over one cycle.
The lower panel shows the valid-sample count in each phase bin together with
the count required by `--min-valid-fraction`. Rejected bins are omitted from the
lines and marked with crosses when their stored value is finite. This is a
quick diagnostic for whether the selected voxel has enough phase coverage to
support a phase-averaged analysis. The plot reads only the final
`phase_average.nc` file, so it cannot show convergence versus the number of
processed cycles; that would require storing partial phase averages during the
original run or rereading the raw time series.

For true convergence versus the number of motion cycles at one voxel, reread
the source raw/interpolated time series and accumulate the selected phase-bin
means cycle by cycle:

```powershell
python scripts\plot_phase_voxel_convergence.py outputs\surge_st06_x3p5d\phase_average.nc --x 4200 --y 0 --z 0 --phases-deg 0,90,180,270 --min-valid-fraction 0.5 --output outputs\phase_voxel_convergence.png
python scripts\plot_phase_voxel_convergence.py outputs\surge_st06_x3p5d\phase_average.nc --raw-file path\to\interpolated_or_raw.nc --x 4200 --y 0 --z 0 --field coherent
```

By default, the script uses the `source_file`, `frequency_hz`, `phase_offset`,
`n_phase_bins`, `invalid_samples`, and `zero_mask` metadata stored in
`phase_average.nc`. Pass `--raw-file`, `--frequency-hz`, `--phase-signal`,
`--invalid-samples`, or `--zero-mask` to override those values. The PNG shows
four requested phases, using the nearest stored phase-bin center for each one;
the dashed horizontal lines are the final values stored in `phase_average.nc`.
A matching CSV is written next to the PNG with the cumulative means and counts
for each cycle, component, and selected phase bin.

The same `phase_average.nc` file also stores the first-harmonic fit. Plot a
single harmonic map with:

```powershell
python main.py outputs\surge_st06_x3p5d\phase_average.nc --harmonic-plane --plane z --plane-value 0 --harmonic-component u --harmonic-quantity amplitude
python main.py outputs\surge_st06_x3p5d\phase_average.nc --harmonic-plane --plane z --plane-value 0 --harmonic-component u --harmonic-quantity phase
```

Available harmonic quantities are `amplitude`, `phase`, `a`, `b`, and `offset`.
The plotted harmonic amplitude is the coherent first-harmonic response after
the missing-bin handling described above. The phase map is displayed in degrees.

To combine the main quality checks for one voxel in a single figure, run:

```powershell
python scripts\assess_phase_average_quality.py outputs\surge_st06_x3p5d\phase_average.nc --x 4200 --y 0 --z 0 --min-valid-fraction 0.5 --output outputs\phase_quality.png
```

The script rereads the source raw/interpolated time series for the selected
voxel. It plots phase-bin means with standard-error bars, valid-count coverage
per phase bin, cycle-by-cycle convergence for the first requested phase, and a
summary panel with accepted phase-bin counts, coherent-RMS to residual-RMS
ratios, and first-harmonic `R2` values. It also writes a CSV next to the PNG.
Use `--raw-file`, `--frequency-hz`, or `--phase-signal` if the stored metadata
does not point to the desired source file or phase definition.

### Extract A Z Slab

For fast sensitivity studies, extract a thin raw-style slab around a selected
`z` plane:

```powershell
python main.py "path\to\raw_file.nc" --extract-z-slab --z-slab-center 0 --z-slab-width 3 --output outputs\z0_w3_slab.nc
```

With `--z-slab-width 3`, the output keeps the nearest plane to
`--z-slab-center` plus one neighboring `z` plane on each side. The output file
still contains `t`, `z`, `y`, `x`, `u`, `v`, and `w`, but its velocity arrays
have shape `(time, 3, y, x)`. The selected source z indices and nearest center
coordinate are stored in the file metadata.

The slab width must be odd so the slice has a clear center plane. If the
requested slab would run outside the available z range, the command stops
instead of silently clipping the selection.

You can then run interpolation sensitivity tests on the smaller slab:

```powershell
python main.py outputs\z0_w3_slab.nc --interpolate-velocity --zero-mask vector --invalid-samples zero-or-nan --max-temporal-gap 1 --output outputs\z0_w3_interp_gap1.nc
```

### Interpolate Velocity Holes

Fill holes in a raw 4D velocity time series before computing wake products:

```powershell
python main.py "path\to\raw_file.nc" --interpolate-velocity --output outputs\interpolated_velocity.nc
python main.py --case static_x3p5d --interpolate-velocity
python main.py "path\to\raw_file.nc" --interpolate-velocity --invalid-samples zero-or-nan --interpolation-axes t z y x --max-temporal-gap 1 --max-spatial-gap 5 --interpolation-passes 3 --output outputs\interpolated_velocity.nc
```

This is a non-physics-informed interpolation. The code converts selected holes
to `NaN`, then applies one-dimensional linear interpolation sequentially along
the requested axes. Original valid samples are preserved. The default axis order
is `t z y x`, so temporal neighbors are used first, then the spatial grid.
Gap limits can keep the interpolation from crossing large missing regions.

Options:

- `--interpolation-axes t z y x`: choose the interpolation axes and their
  order. Use fewer axes when you want a stricter fill, for example
  `--interpolation-axes t`.
- `--interpolation-passes N`: repeat the selected axis sequence up to `N`
  times. Later passes may use values filled by earlier passes. The command
  stops early if a pass fills nothing.
- `--max-temporal-gap N`: limit temporal interpolation to brackets at most `N`
  frames away on each side of the missing frame. Use `--max-temporal-gap 1`
  for strict `t-1` and `t+1` interpolation. If omitted, temporal brackets can
  be farther away.
- `--max-spatial-gap N`: limit spatial interpolation to brackets at most `N`
  voxels away on each side along `z`, `y`, or `x`. Use this to avoid filling
  wide spatial gaps; for example, `--max-spatial-gap 5` will not fill a point
  whose nearest valid spatial brackets are 30 voxels away.
- `--interpolation-workers N`: interpolate multiple velocity components at the
  same time. Use `--interpolation-workers 3` to process `u`, `v`, and `w` in
  parallel when RAM allows it.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose which samples are
  treated as holes.
- `--zero-mask component`: default. Fill holes independently for `u`, `v`,
  and `w`.
- `--zero-mask vector`: treat a full velocity vector as missing only when all
  selected invalid-value conditions apply to the vector. Use this when holes
  occur at the same locations for `u`, `v`, and `w`; the interpolation reuses
  one shared hole mask for all three components. Interpolation along each
  selected axis is vectorized across all grid lines for speed.
- `--output path.nc`: output file. Without `--output`, raw-file workflows write
  to `outputs\{raw_stem}_interpolated.nc`; case workflows write an
  `interpolated_velocity.nc` product in the case output folder.
- `--overwrite`: allow replacing an existing interpolated file.

The output file keeps the raw-style datasets `t`, `z`, `y`, `x`, `u`, `v`, and
`w`, plus `{component}_filled_mask` datasets showing which cells were filled.
File metadata records the source file, interpolation axes, invalid-sample
policy, and filled/remaining hole counts.

To visualize what was added, open the inspector with the raw file and the
interpolated file together:

```powershell
python main.py "path\to\raw_file.nc" --inspect --compare-interpolated --interpolated-file outputs\interpolated_velocity.nc --z 0 --frame 0
```

The displayed velocity field comes from the interpolated file. Cells where any
velocity component was filled are drawn semi-transparent. Click a cell to see
raw values, interpolated values, per-component filled flags, and the velocity
deltas. Use the `<` and `>` buttons next to the frame slider to move exactly
one frame at a time.

### Plot Mean Wake Z-Plane Composites

After `prepare_mean_wake_products.py` has created reusable `mean.nc` files, plot
the `z=0` mean wake plane for all files in one prepared output folder:

```powershell
python scripts\plot_mean_wake_z0.py "D:\outputs\test_mean_wake_results" --output-folder "D:\outputs\figures\figure1_mean_wake_z0_001" --z 0 --quantity u_over_u_inf
```

The plotting output folder must be new. The script refuses to write into an
existing folder. Each group writes a PNG and a small plot manifest CSV.

All plots created by one command use the same colorbar range, computed from all
loaded `mean.nc` files before any group is plotted. This makes the generated
case figures visually comparable. To force an identical scale across separate
plotting runs, pass explicit limits:

```powershell
python scripts\plot_mean_wake_z0.py "D:\outputs\test_mean_wake_results" --output-folder "D:\outputs\figures\figure1_mean_wake_z0_002" --z 0 --quantity u_over_u_inf --vmin 0 --vmax 1
```

Use `outputs\figures\...` for final/reproducible figure folders, and keep
processed intermediate data in sibling folders such as
`outputs\test_mean_wake_results\...`.

Files are grouped by case name inferred from labels such as
`Static_1D__b64...` and `Static_1.625D__b64...`; those become one `Static`
figure with multiple downstream stations. Volumes are ordered upstream to
downstream by the distance in `D`. If downstream volumes overlap earlier
volumes, overlapping downstream cells are masked so the upstream values are
kept. The overlapped footprint is shaded transparently on the plot.

Useful quantities:

- `u_over_u_inf`: default wake recovery plot.
- `wake_deficit`: normalized streamwise deficit.
- `abs_U` or `speed_from_mean`: mean velocity magnitude.
- `u_mean`, `v_mean`, `w_mean`: component means.

### Plot Radial Wake Deficit

Use the full 3D averaged volume to make a radius-versus-`x` wake-deficit plot.
Here `x` is streamwise, `y` is vertical, and `z` is the lateral direction that
completes the right-handed coordinate system. This averages all voxels in
annular bins around the rotor axis in the vertical-lateral `y-z` plane:

```powershell
python scripts\plot_radial_wake_deficit.py "D:\outputs\test_mean_wake_results" --output-folder "D:\outputs\figures\figure1_radial_wake_deficit_001" --rotor-y 0 --rotor-z 0 --radial-bin-width 25 --invalid-samples zero-or-nan --contour-step 0.05 --contour-label-step 0.1
```

The plotting output folder must be new. The script refuses to write into an
existing folder.

`--rotor-y` and `--rotor-z` are required and must be given in the same
coordinate units as the `mean.nc` file, usually millimetres for the current
DaVis exports. `--rotor-y` is the vertical hub coordinate. `--rotor-z` is the
lateral hub coordinate.
For each voxel, the plotted quantity is:

```text
(U_inf - u_mean) / U_inf
```

By default, `NaN` and infinite values are excluded before radial averaging.
Use `--invalid-samples none` only when you explicitly want to use every value
exported by DaVis. Use `--invalid-samples zero-or-nan` for products where old
exact-zero holes should also be removed. Use `--require-all-components` when a
voxel should be used only if `u_mean`, `v_mean`, and `w_mean` are all valid.

Contour lines are drawn every `0.05` wake-deficit units by default, with
numeric labels every `0.1`. Use `--contour-step 0` to disable contours, or
`--contour-label-step 0` to keep contour lines without numeric labels.

As with the `z=0` composite plot, files are grouped by case and ordered by
downstream distance. If downstream volumes overlap earlier volumes, overlapping
downstream `x` columns are masked so upstream values are kept. The overlapped
columns are shaded transparently on the plot.

### Plot Plane Wake-Deficit Profiles

To make a figure closer to a paper-style wake recovery profile plot, take one
selected plane from each mean volume and extract a 1D deficit profile at the
centre of each volume:

```powershell
python scripts\plot_plane_wake_deficit_profiles.py "D:\outputs\test_mean_wake_results" --output-folder "D:\outputs\figures\figure1_plane_profiles_001" --plane-axis z --plane-value 0 --rotor-y 400 --rotor-diameter 1200 --invalid-samples zero-or-nan
```

For `--plane-axis z`, the script uses the nearest `z` plane and plots the
deficit against `(y - y_rotor) / D`, where `y` is vertical. Pass `--rotor-y`
for this case. For `--plane-axis y`, it uses the nearest `y` plane and plots
against `(z - z_rotor) / D`, where `z` is lateral. Pass `--rotor-z` for this
case. The streamwise profile location is the centre of each volume by default;
pass `--x-value` to force a specific `x`.

The output folder must be new. The script writes the profile PNG, a manifest
CSV, and a `*_data.csv` table with the plotted coordinate, normalized
coordinate, wake deficit, validity flag, selected plane, and selected `x`.

`NaN` and infinite mean values are excluded by default. Use
`--invalid-samples zero-or-nan` for products where old exact-zero holes should
also be removed.

### Compute Turbulent Kinetic Energy

Compute turbulent kinetic energy from a raw 4D time series and an already
computed mean velocity file:

```powershell
python main.py "path\to\raw_file.nc" --tke --mean-file outputs\temporal_mean.nc --output outputs\tke.nc
python main.py --case static_x3p5d --tke
python main.py --case static_x3p5d --processing-id static_x3p5d_02 --tke
```

The command computes:

```text
k = 0.5 * (mean(u_prime^2) + mean(v_prime^2) + mean(w_prime^2))
```

where `u_prime = u - u_mean`, `v_prime = v - v_mean`, and
`w_prime = w - w_mean` at each voxel. Missing raw samples are ignored using the
same `--zero-mask` and `--invalid-samples` modes as temporal averaging.

Options:

- `--mean-file path.nc`: temporal-average file containing `u_mean`, `v_mean`,
  and `w_mean`. It must have the same `x`, `y`, `z` grid as the raw file.
- `--processing-id ID`: with `--case`, choose which case output folder supplies
  `mean.nc`, for example `static_x3p5d_02`. If omitted, exactly one matching
  `outputs/{case_id}*/mean.nc` file must exist.
- `--output path.nc`: output file for the TKE volume.
- `--overwrite`: allow replacing an existing output file.
- `--chunk-size N`: number of time steps read at once.
- `--zero-mask component`: default. When zeros are treated as invalid, apply
  the zero mask independently for `u`, `v`, and `w`.
- `--zero-mask vector`: when zeros are treated as invalid, reject a sample only
  when `u`, `v`, and `w` are all exactly zero.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose whether missing raw
  samples are exact zeros, `NaN`/infinite values, both, or nothing.

The output file contains `x`, `y`, `z`, `u_prime2_mean`, `v_prime2_mean`,
`w_prime2_mean`, component counts, and `tke`. Provenance metadata records both
the raw source file and the mean velocity file used to create the result.
With `--case`, the default output is written beside the mean file, for example
`outputs\static_x3p5d\tke.nc`.

### Compute Reynolds Stresses

Compute selected Reynolds stress components from a raw 4D time series and an
already computed mean velocity file:

```powershell
python main.py "path\to\raw_file.nc" --reynolds-stress --mean-file outputs\temporal_mean.nc --stress-components uv --output outputs\reynolds_uv.nc
python main.py "path\to\raw_file.nc" --reynolds-stress --mean-file outputs\temporal_mean.nc --stress-components uu vv ww --output outputs\reynolds_diagonal.nc
python main.py "path\to\raw_file.nc" --reynolds-stress --mean-file outputs\temporal_mean.nc --stress-components all --output outputs\reynolds_all.nc
python main.py --case static_x3p5d --reynolds-stress --stress-components all
python main.py --case static_x3p5d --processing-id static_x3p5d_02 --reynolds-stress --stress-components uv
```

The available components are `uu`, `uv`, `uw`, `vv`, `vw`, and `ww`, where:

```text
uv = mean(u_prime * v_prime)
```

and similarly for the other components. Missing raw samples are ignored using
the same `--zero-mask` and `--invalid-samples` modes as temporal averaging.

Options:

- `--mean-file path.nc`: temporal-average file containing `u_mean`, `v_mean`,
  and `w_mean`. It must have the same source file provenance and `x`, `y`, `z`
  grid as the raw file.
- `--processing-id ID`: with `--case`, choose which case output folder supplies
  `mean.nc`, for example `static_x3p5d_02`. If omitted, exactly one matching
  `outputs/{case_id}*/mean.nc` file must exist.
- `--stress-components uu uv ...`: one or more components to compute. Use
  `all` to compute all six independent components.
- `--output path.nc`: output file for the Reynolds stress volume.
- `--overwrite`: allow replacing an existing output file.
- `--chunk-size N`: number of time steps read at once.
- `--zero-mask component`: default. When zeros are treated as invalid, apply
  the zero mask independently for each component used in a stress product.
- `--zero-mask vector`: when zeros are treated as invalid, reject a sample only
  when `u`, `v`, and `w` are all exactly zero.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose whether missing raw
  samples are exact zeros, `NaN`/infinite values, both, or nothing.

The output file contains `x`, `y`, `z`, one `{component}_reynolds_stress`
dataset for each requested component, and matching `{component}_count`
datasets. Provenance metadata records both the raw source file and the mean
velocity file used to create the result.
With `--case`, the default output is written beside the mean file, for example
`outputs\static_x3p5d\reynolds_stresses.nc`.

### Apply A Valid-Fraction Cutoff To An Existing Average

If you already have a temporal-average file, apply a valid-count cutoff without
re-reading the raw 4000-frame time series:

```powershell
python main.py outputs\temporal_mean.nc --apply-valid-fraction 0.8 --output outputs\temporal_mean_80pct.nc
```

This reads `u_count`, `v_count`, and `w_count`, sets means to `NaN` where the
valid count is below the requested fraction, recomputes `speed_from_mean`, and
writes a new file. The source average file is not modified.

This command also refuses to overwrite existing outputs unless `--overwrite`
is passed.

### Compare Raw Export Planes

Before choosing which DaVis/export parameters to use, compare the temporal mean
and standard deviation on matching planes from several raw exports:

```powershell
python scripts\compare_export_planes.py Static_3.5D__b128.nc Static_3.5D__b128f.nc --labels b128 b128f --output outputs\export_planes_speed.png
```

By default, the script compares `speed` on three planes:

- two planes perpendicular to the streamwise direction, at one-third and
  two-thirds of the common overlapping `x` range
- one streamwise-vertical plane, centered in the common overlapping `y` range

The output figure has one row per export. For each plane it shows the temporal
mean and temporal standard deviation with shared color scales.
When the command starts, it prints all received arguments and all defaults used,
including the resolved default plane coordinates and nearest plane indices for
each file.

If two exports have different physical overlap, use difference mode to project
one export onto the other export's plane grid over the common domain:

```powershell
python scripts\compare_export_planes.py b96.nc b128.nc --labels b96 b128 --quantity speed --difference --reference-grid second --output outputs\b128_minus_b96_planes.png
```

In this mode the figure shows, for each plane, the reference field, the
comparison field interpolated onto the reference grid, and the difference.
Values outside the common overlap are not extrapolated.

Useful options:

- `--quantity {speed,u,v,w}`: choose what to compare.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose which raw samples are
  excluded from the temporal mean and standard deviation.
- `--x-planes X1 X2`: choose the two streamwise-normal planes manually.
- `--y-plane Y`: choose the streamwise-vertical plane manually.
- `--labels name1 name2 ...`: display names for each export.
- `--difference`: for two files, plot projected differences on a shared grid.
- `--reference-grid {first,second}`: choose which file supplies the plotted
  grid in difference mode.
- `--output path.png`: output figure path.

### Visualize A Temporal-Average Volume

Plot one `x`, `y`, or `z` plane from a postprocessed temporal-average file:

```powershell
python main.py outputs\temporal_mean.nc --average-plane --plane z --plane-value 0
python main.py outputs\temporal_mean.nc --average-plane --plane x --plane-value 0 --quantity u
python main.py outputs\temporal_mean.nc --average-plane --plane y --plane-value 0 --quantity speed
python main.py outputs\temporal_mean.nc --average-plane --plane z --plane-value 0 --min-valid-fraction 0.8
python main.py outputs\temporal_mean.nc --average-plane --plane z --plane-value 0 --save outputs\temporal_mean_z0.png
```

Options:

- `--plane {x,y,z}`: slice direction. The selected coordinate is held
  constant.
- `--plane-value VALUE`: requested coordinate value for the selected plane.
  The nearest available coordinate is used.
- `--quantity {speed,u,v,w}`: scalar field shown as the color background.
  `speed` is the 3D velocity magnitude from the mean vector.
- `--min-valid-fraction F`: mask the displayed slice without modifying the
  file. For `speed`, a pixel is masked if any velocity component is below the
  threshold.
- `--quiver-step N`: arrow spacing. Larger values draw fewer arrows.
- `--save path.png`: save the figure instead of opening an interactive window.

### Compare Single And Double Precision

Compare fluctuation and TKE quantities between the double-precision
`Static_3.5D__b128.nc` file and the single-precision `Static_3.5D__b128f.nc`
file in a wake shear-layer cube:

```powershell
python scripts\compare_precision_fluctuations.py --output outputs\precision_fluctuation_comparison_b128_180mm.txt
```

By default, the cube is centered at `y = 600 mm`, the center of the `x` range,
and `z = 0 mm`, with a half-width of `90 mm` in every direction. This gives a
180 mm x 180 mm x 180 mm analysis volume. Use `--half-width 30` for the
original 60 mm cube. Use `--invalid-samples {zero,nan,zero-or-nan,none}` to
choose which raw samples are excluded from the fluctuation statistics.

### Compare Squared Swirling-Strength Exports

Compare two table-like 3D exports that already contain squared swirling
strength, without recomputing velocity gradients:

```powershell
python scripts\compare_squared_swirling_strength_exports.py fine_export.csv coarse_export.csv --label-a fine --label-b coarse --coordinate-unit mm --output-folder outputs\swirl_compare
```

The script looks for `x`, `y`, `z`, and a squared swirling-strength column with
names such as `lambda_ci_squared`, `lambda_ci^2`, `LambdaCi2`,
`swirling_strength_squared`, `swirl2`, or `lambda2`. If `u` is present it also
plots the centre-plane `u/U_inf`.

Useful options:

- `--rotor-diameter 1.2`: rotor diameter `D` in metres.
- `--u-inf 4.0`: inflow velocity in m/s.
- `--coordinate-unit {m,mm}`: convert coordinates to metres before analysis.
- `--min-component-size N`: remove detected vortex regions with fewer than
  `N` grid cells.
- `--remove-boundary-components`: ignore components touching the domain
  boundary.

Outputs include grid-quality diagnostics, field-comparison metrics, detected
component properties, component matches, plots, and a recommendation report.

### Detect Vortex Cores From Exported Swirling Strength

For vortex-core detection, the code follows the swirling-strength criterion
described in the VortexFitting methodology:
<https://guilindner.github.io/VortexFitting/methodology.html#swirling-strength-criterion>.
In this repository we use the already exported squared swirling-strength field
directly; we do not recompute `lambda_ci` from velocity gradients.

For the DaVis NetCDF exports where `swir` appears to store `-lambda_ci^2`, use
`--use-davis-values` when you want to avoid any zero or velocity-hole masking
and inspect exactly what DaVis exported:

```powershell
python scripts\compare_squared_swirling_strength_exports.py "D:\Static_3.5D__b64v50oallvar_f.nc" --detect-only --label-a b64 --frame 0 --swirl-variable swir --swirl-sign negative --use-davis-values --coordinate-unit mm --core-threshold p99 --min-component-size 3 --output-folder outputs\vortex_cores_b64_frame0
```

This saves `vortex_core_overview.png`, `detected_components.csv`,
`grid_quality.csv`, and `vortex_core_report.json`.

To inspect how vortex-core candidates evolve over time in a `z` plane, open the
interactive viewer:

```powershell
python scripts\compare_squared_swirling_strength_exports.py "D:\Static_3.5D__b64v50oallvar_f.nc" --inspect-z --label-a b64 --frame 0 --z-value 0 --swirl-variable swir --swirl-sign negative --use-davis-values --coordinate-unit mm --inspect-percentile 99 --min-component-size 3
```

The viewer has sliders for frame, `z` index, and percentile threshold. It
displays the non-dimensionalized exported `lambda_ci^2` field, contours the
active threshold, and marks detected component weighted centroids and peaks.
If you instead want to hide missing velocity regions, replace
`--use-davis-values` with `--hole-mask velocity-zero`.

To compare two exports side by side in one interactive window, pass both files
and use `--inspect-z-compare`:

```powershell
python scripts\compare_squared_swirling_strength_exports.py "D:\Static_3.5D__b64v50oallvar_f.nc" "D:\Static_3.5D__b64v75oallvar_f.nc" --inspect-z-compare --label-a b64v50 --label-b b64v75 --frame 0 --z-value 0 --swirl-variable swir --swirl-sign negative --use-davis-values --coordinate-unit mm --inspect-percentile 99 --min-component-size 3
```

The comparison viewer uses one frame slider, one physical `z` slider, and one
threshold-percentile slider. Each file uses its nearest available `z` plane to
the requested physical value, and the two panels share the same color scale at
each slider position.

To save a GIF of the same z-plane detector through time:

```powershell
python scripts\compare_squared_swirling_strength_exports.py "D:\Static_3.5D__b64v50oallvar_f.nc" --animate-z --label-a b64 --z-value 0 --swirl-variable swir --swirl-sign negative --use-davis-values --coordinate-unit mm --inspect-percentile 99 --min-component-size 3 --fps 10 --start 0 --stop 4000 --step 1 --save outputs\vortex_cores_b64_z0.gif
```

The GIF uses one fixed color scale for the whole selected time range. By
default, the color maximum is the largest frame-wise 99.5th percentile among
the selected frames. Use `--color-vmax VALUE` to set it manually. Use a larger
`--step` for a lighter preview GIF, for example `--step 10`.

## Optional Editable Install

For repeated use, install the repository as a local Python package:

```powershell
python -m pip install -e .
```

Then run the command from anywhere:

```powershell
ptv-flow "path\to\file.nc" --animate --fps 40
```

The direct script still works:

```powershell
python main.py "path\to\file.nc" --frame 0
```

## Repository Layout

```text
.
├── main.py                 # Backward-compatible command-line entry point
├── ptv_flow/
│   ├── cli.py              # Command-line arguments
│   ├── inspect.py          # Interactive raw/average cell inspector
│   ├── postprocess.py      # Temporal averaging and averaged-volume reader
│   ├── reader.py           # Lazy NetCDF/HDF5 reader
│   └── visualize.py        # Matplotlib animation code
├── scripts/
│   └── extract_test_fixture.py
├── tests/
│   ├── data/tiny_flow.nc   # Small committed fixture for unit tests
│   ├── test_cli.py
│   ├── test_inspect.py
│   ├── test_postprocess.py
│   └── test_reader.py
├── requirements.txt        # Fast dependency install
├── requirements-dev.txt    # Test and pre-commit dependencies
├── pyproject.toml          # Optional editable/package install
├── .pre-commit-config.yaml # Commit-time checks
├── .gitattributes          # Stable Python line endings
└── .gitignore              # Keeps data, caches, and exports out of git
```

## Data Files

Large `.nc` files are intentionally ignored by git. Keep them locally, on a
shared drive, or in a `data/` directory. Pass the file path explicitly when it
is not named `Static_3.5D__b128f.nc` in the repo root.

## Python API

```python
from ptv_flow import FlowDataset

with FlowDataset("path/to/file.nc") as flow:
    print(flow.describe())
    print(flow.voxel_size())
    plane = flow.read_z_plane(time_index=0, z_index=flow.nearest_z_index(0.0))
    speed = plane.speed
```
