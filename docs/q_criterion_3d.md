# Interactive Q-criterion visualization

From the repository root:

```powershell
python scripts/visualize_q_criterion.py --case Static --distance 1D
python scripts/visualize_q_criterion.py --case SurgeLF --distance 3.5D --frame 100 --threshold 10
python scripts/visualize_q_criterion.py --case Static --all-distances
```

`--all-distances` combines every station for the selected case in a single
HTML plot, sorted by downstream distance. It uses the exported coordinates
without additional offsets or interpolation between stations. Station colors
and legend entries identify the separate surfaces. Overlapping measurement
regions remain separate surfaces. All stations use one shared threshold:
the requested `--threshold`, or the percentile of pooled positive valid Q
samples from all selected frames. Each valid sample has equal weight.
Stations without a surface use points above threshold, if any.
The output is `<case>_all_distances_frame<index>.html` in the same output folder.
The same frame index is selected in each file; this does not imply that the
separate acquisitions were synchronized.

The default folder is `D:\binning64voxel75overlap_z0`. Override it with
`--folder`, or select an exact export with `--file`. The script reads only one
frame of the exported `qcri` variable in `(time,z,y,x)` order. It preserves
the exported sign and does not calculate Q from velocity or swirling strength.

Coordinates default to millimetres and are displayed as x/D, y/D, z/D.
Following the comparison script, D defaults to 1.2 m and U_inf to 4 m/s;
override with `--rotor-diameter` and `--u-inf`. Q is displayed as
Q*=Q D²/U_inf², assuming exported Q has units s⁻².

The surface defaults to the 90th percentile of positive valid Q values.
Use `--percentile` or an explicit positive `--threshold` in Q* units.
Nonfinite velocities and cells with all three velocities zero are masked by
default; `--hole-mask` controls this behavior. Missing values remain missing.
Surfaces require valid neighboring samples; holes can interrupt them.

The self-contained HTML opens in a browser and supports rotation and zoom.
Use the buttons above the plot to switch between the extracted isosurface
and points above the threshold. Surface triangles are computed in Python
only in grid cells with eight valid corners. If no triangles remain, the
points view opens automatically.
Use `--no-open` to save only, and `--output path.html` to set the destination.
The default destination is `outputs/q_criterion/<file>_frame<index>.html`.

## Animated GIF

```powershell
python scripts/visualize_q_criterion.py --case Static --all-distances --gif outputs/q_criterion/Static_all_distances_preview.gif --start-frame 0 --stop-frame 200 --frame-step 5 --fps 15
```

This renders frames 0, 5, ..., 195 into a 40-frame GIF. `--stop-frame` is
exclusive and defaults to the shortest selected recording. `--frame` applies
to HTML only; GIF uses `--start-frame`. GIF replaces HTML export for that run.
The GIF uses Matplotlib to render the same extracted surfaces, with physical
y vertical; `--elevation` and `--azimuth` control the fixed camera.
Use `--no-open` to save without opening a viewer.

One Q threshold remains fixed throughout playback. It is either the explicit
`--threshold`, or the pooled positive-Q percentile from the first selected
animation frame. Empty frames remain in the animation. Each station retains
its color, and the axes retain their original limits and physical aspect ratio.
Playback FPS is independent of acquisition rate; timestamps show elapsed time
from the selected start within each recording. The overlay identifies combined
stations as unsynchronized recordings: evolution within a station is meaningful,
but apparent movement between stations does not track the same structure.

Pillow retains rendered images in memory until saving. Start with short clips
or a larger frame step before rendering thousands of frames.

These z0 exports contain just three z planes, so the view represents a thin
slab at its true aspect ratio. A full 3D wake requires a volumetric export.
