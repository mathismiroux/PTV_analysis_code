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

## Animate A z Plane

Show the nearest available plane to `z=0` at 40 fps:

```powershell
python main.py "U:\FWExp data\DATA_2025_02_OJF__FW_PTV\postprocessed PTV data v2\Static_3.5D__b96.nc" --animate --fps 40
```

The visualization shows:

- velocity magnitude as the color field
- in-plane `(u, v)` velocity vectors as arrows
- the nearest available `z` coordinate in the title

Useful options:

```powershell
python main.py "path\to\file.nc" --animate --z 0 --fps 40
python main.py "path\to\file.nc" --animate --start 0 --stop 400 --step 1
python main.py "path\to\file.nc" --animate --quiver-step 2
python main.py "path\to\file.nc" --animate --save outputs\z0_animation.gif
```

GIF export works with the default dependencies. MP4 export needs an FFmpeg
installation and a small writer configuration change.

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
│   ├── reader.py           # Lazy NetCDF/HDF5 reader
│   └── visualize.py        # Matplotlib animation code
├── requirements.txt        # Fast dependency install
├── pyproject.toml          # Optional editable/package install
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
