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

The inspector does not compute a full averaged volume. It only reads the time
series for the one clicked voxel, which is why this value appears quickly.
The minimum valid fraction can be changed interactively in the GUI. The
selected-voxel means update immediately and are shown as accepted or rejected
using the corresponding valid-count threshold. In the displayed slice, the
raw frame colors stay unchanged, rejected cells are marked with a translucent
red overlay, and their arrows are suppressed. The side panel still reports the
raw value at the selected frame. Passing
`--min-valid-fraction` sets the initial slider value.

If you already computed a temporal-average output file, pass it too:

```powershell
python main.py "path\to\raw_file.nc" --inspect --z 0 --frame 0 --compare-average --average-file outputs\temporal_mean.nc
```

With `--compare-average --average-file`, the side panel also shows the stored
`u_mean`, `v_mean`, `w_mean`, and counts from the postprocessed file so you can
compare them with the value computed from the raw series.

The comparison file must have the same grid shape and `x`, `y`, `z`
coordinates as the raw file. This prevents accidentally comparing, for example,
a `b128` raw file with a `b96` temporal average.

### Compute A Temporal Average

Create a 3D mean velocity volume from a raw 4D time series. By default,
exact-zero samples are ignored in the average.

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
- `--zero-mask component`: default. Ignore exact zeros independently for
  `u`, `v`, and `w`.
- `--zero-mask vector`: ignore a sample only when `u`, `v`, and `w` are all
  exactly zero.
- `--invalid-samples {zero,nan,zero-or-nan,none}`: choose which raw samples are
  treated as missing. The default is `zero`, matching the current DaVis export.
  Use `nan` when missing values are exported as `NaN`, or `zero-or-nan` during
  a transition period where both conventions may appear.
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
- `--zero-mask component`: default. Ignore exact zeros independently for
  `u`, `v`, and `w`.
- `--zero-mask vector`: ignore a sample only when `u`, `v`, and `w` are all
  exactly zero.
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
- `--zero-mask component`: default. Ignore exact zeros independently for each
  component used in a stress product.
- `--zero-mask vector`: ignore a sample only when `u`, `v`, and `w` are all
  exactly zero.
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
