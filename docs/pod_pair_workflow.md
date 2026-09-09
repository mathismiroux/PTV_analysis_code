# Compare a pair of local POD modes

Use `scripts/compare_pod_pair.py` on an existing local `pod.h5`. It reads only
saved POD arrays and never recalculates POD or changes velocity/mean files.

```powershell
python scripts/compare_pod_pair.py "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\SurgeLF_3.5D\pod_energy_valid90\pod.h5" --output "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\SurgeLF_3.5D\pod_energy_valid90\pair_01_02" --modes 1 2 --frequency-hz 2
```

Choose a new output directory for each run. For another pair, change `--modes`.
The required `--frequency-hz` specifies the oscillation to fit, rather than
guessing from a spectral peak. Other options:

- `--z-slices-d -0.15 0 0.15`: x-y sections at the nearest measured z locations.
- `--y-slice-d 0.15`: an x-z section to expose differences away from z=0.
- `--rotor-diameter 1200`: D in source coordinate units; no origin shift.
- `--start-time SECONDS --cycles 2`: reconstruction interval, clipped to the
  available record; defaults to the beginning and two forcing periods.
- `--frames 80 --fps 20`: GIF sampling/playback. Playback speed need not be
  physical time; each frame displays its physical timestamp.
- `--no-animation`: skip the GIF.

Outputs:

1. `spatial_xy_*.png`, `spatial_xz.png`: both modes and all u/v/w components,
   with shared component colour scales across both modes and selected slices.
2. `coefficient_phase_portrait.png`: original normalized coefficients and the
   fitted harmonic ellipse, coloured by nominal phase `2*pi*f*t`. No measured
   platform phase is assumed available. Fits use the full recording and include
   an intercept. Harmonic portraits omit the intercept, normalize by the original
   coefficient RMS and preserve the saved mode signs.
3. `coefficients_and_spectra.png`: original/fit coefficients over the requested
   interval and full-record Hann spectra normalized within each mode.
4. `pair_reconstruction.gif`: left, two-mode reconstruction from saved original
   coefficients; right, reconstruction using only their fitted forcing-frequency
   parts. Both show the slice nearest z=0, omit mean flow and share component
   scales fixed throughout the animation. Coefficients are linearly interpolated
   to animation timestamps; missing spatial regions remain blank.
5. `summary.json`: pair energy, harmonic explained fractions, fitted phase
   difference (second minus first), weighted 3D spatial inner-product matrix,
   per-component inner products, source metadata and actual slice coordinates.

The weighted 3D inner product, rather than visual dissimilarity in a single
slice, is the appropriate orthogonality check. Similar power spectra can coexist
with quadrature coefficients. A phase difference near +/-90 degrees supports
an oscillatory-pair interpretation but does not establish a unique mechanism.
Signs are arbitrary, so a 180-degree change from a sign flip is equivalent.
Near-equal-energy modes may rotate within their pair subspace. Neither the
fitted ellipse nor a two-mode animation demonstrates that the pair captures
all wake dynamics. Treat the original POD's support and gap-filling limitations
as applying to these comparisons too.

Tests: `python -m pytest tests/test_compare_pod_pair.py -q`.
