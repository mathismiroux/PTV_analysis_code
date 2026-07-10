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
```

In the GUI, use the sliders to choose the frame and `z` plane. Click a cell in
the slice to inspect:

- the selected indices and coordinates
- raw `u`, `v`, `w`, and speed at the selected frame
- selected-voxel temporal mean computed on demand from the raw time series

The inspector does not compute a full averaged volume. It only reads the time
series for the one clicked voxel, which is why this value appears quickly.

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

Create a 3D mean velocity volume from a raw 4D time series. Exact-zero values
are ignored in the average.

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
- `--min-valid-fraction F`: discard averaged values with fewer than this
  fraction of valid time samples. Use `--min-valid-fraction 0.8` to require
  80 percent valid data at each voxel/component.

The output file contains `x`, `y`, `z`, `u_mean`, `v_mean`, `w_mean`,
`u_count`, `v_count`, `w_count`, and `speed_from_mean`.

Processed files store provenance metadata in the file attributes and in a
`provenance` group, including the source file path, source file name, source
file size, creation time, operation, zero-mask mode, chunk size, and minimum
valid-count settings.

### Visualize A Temporal-Average Volume

Plot one `x`, `y`, or `z` plane from a postprocessed temporal-average file:

```powershell
python main.py outputs\temporal_mean.nc --average-plane --plane z --plane-value 0
python main.py outputs\temporal_mean.nc --average-plane --plane x --plane-value 0 --quantity u
python main.py outputs\temporal_mean.nc --average-plane --plane y --plane-value 0 --quantity speed
python main.py outputs\temporal_mean.nc --average-plane --plane z --plane-value 0 --save outputs\temporal_mean_z0.png
```

Options:

- `--plane {x,y,z}`: slice direction. The selected coordinate is held
  constant.
- `--plane-value VALUE`: requested coordinate value for the selected plane.
  The nearest available coordinate is used.
- `--quantity {speed,u,v,w}`: scalar field shown as the color background.
  `speed` is the 3D velocity magnitude from the mean vector.
- `--quiver-step N`: arrow spacing. Larger values draw fewer arrows.
- `--save path.png`: save the figure instead of opening an interactive window.

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
