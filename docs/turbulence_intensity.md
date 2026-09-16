Run from the repository root:

```powershell
python scripts/plot_turbulence_intensity.py
```

The default input is `D:\binning64voxel50overlap\outputs_flow\Flow_3.5D\interpolated_velocity.nc`.
To select another case/volume, pass its file or directory as the positional argument:

```powershell
python scripts/plot_turbulence_intensity.py "D:\binning64voxel50overlap\outputs_flow\Flow_3.5D" --output-folder "outputs/Flow_3.5D_TI"
```

The script calculates streamwise turbulence intensity at every voxel over all frames:
`TI [%] = 100 * sqrt(mean((u - mean(u))**2)) / abs(mean(u))`.
When adjacent `mean.nc` exists, its `u_mean` is used for both the fluctuations and
normalization. The time series is still needed because this mean file has no variance.
The script checks matching coordinates and sample counts and follows the mean file's
component/vector finite-sample mask. Use `--mean-file PATH` to specify another mean file.
Without an adjacent mean file, the script computes the mean from the time series.
Use `--reference-velocity 4.0` to normalize by a fixed velocity instead (in the same units as `u`).
The population standard deviation uses finite samples, including interpolated values and zeros,
while excluding declared fill values. A voxel requires at least two samples and at least 80%
temporal coverage after interpolation. `--min-valid-fraction` changes that threshold.
Normalization velocities at or below `--min-mean-speed` (default 0.01 in input velocity units)
are masked. This is total temporal fluctuation intensity: coherent oscillations are included,
and interpolation can affect the fluctuation amplitudes.

The output directory must be new. By default it is `turbulence_intensity` beside the input file.
It contains:

- `turbulence_intensity.png`: three central-index planes with a shared color scale; gray means masked.
- `turbulence_intensity.npz`: full `(z,y,x)` fields `ti_percent`, `u_mean`, `u_rms`, `valid_count`,
  `valid_fraction`, and coordinate arrays `x`, `y`, `z`. Load with `numpy.load`.
- `manifest.json`: input path, definition, thresholds, sample count, and plane indices.

Coordinates are plotted without conversion and labeled mm by default; change the label with
`--coordinate-unit`. `--vmax` sets the color scale maximum in percent. Reading uses bounded
time chunks controlled by `--chunk-size` (default 32); the recording is not loaded all at once.

To process every case/volume subfolder in a parent directory:

```powershell
.\scripts\batch_turbulence_intensity.bat "D:\binning64voxel50overlap\outputs_flow"
```

The equivalent Python command is `python scripts/batch_turbulence_intensity.py` with the
same arguments. Each subfolder containing `interpolated_velocity.nc` is processed in
sequence, using its adjacent `mean.nc` when available. By default, `turbulence_intensity.png`,
`turbulence_intensity.npz`, and `turbulence_intensity_manifest.json` are saved directly
beside each case's `mean.nc`. Per-case logs and a batch CSV in the input root have timestamped names.
Existing TI files are protected: that case fails without overwriting them.
For a single case, use `plot_turbulence_intensity.py INPUT --alongside-mean`.
Optionally pass `--output-root PATH` to save into a new `PATH/CASE/` directory instead,
with `batch_manifest.csv` at that root. Failed cases are logged and the remaining cases continue; any failure results
in exit code 1. Missing mean files use the time-series mean calculation.

Use `--dry-run` to preview without writing files, `--case-pattern "Flow_*"` to filter
case folder names, or `--recursive` for nested case directories. The batch script accepts
`--reference-velocity`, `--min-valid-fraction`, `--min-mean-speed`, `--chunk-size`,
`--coordinate-unit`, and `--vmax` and applies them to every case. Use a fixed `--vmax`
when you want identical color scales across the batch.

Combine the saved maps of one case into a single figure:

```powershell
python scripts/plot_case_turbulence_intensity.py "D:\binning64voxel50overlap\outputs_flow" --case Flow
```

This writes `Flow_TI_combined.png` in the parent folder: central-index z slices (xy maps)
of every volume on one shared physical x/y axis with one color scale. Each volume's actual
z coordinate is labeled above the plot. The scale uses the maximum finite TI across the
displayed slices. Gray is masked data or gaps between volumes. Volumes are drawn in downstream
order at their original coordinates, without interpolation or averaging; later volumes
cover earlier values where valid cells overlap.
Use `--vmax 5` to set a common 0–5% range or `--output PATH.pdf` for PDF output.
Existing combined figures are replaced when rerunning. The script reads saved NPZ fields
and does not recompute TI or change the original maps. It prefers files directly beside
`mean.nc`, then checks the older `turbulence_intensity_saved_mean` and `turbulence_intensity`
subfolders. Missing TI volumes are reported and skipped. Use the same normalization and
coverage settings when generating the individual volumes to make them comparable.
