# Frequency content of saved local POD modes

This standalone batch reads only existing local `pod.h5` files. It never reruns
POD or reads the original velocity fields. The required arrays are `t`,
`temporal_coefficients` (time, mode), and `energy_fraction`. The global
phase-average POD is a different product and is not accepted by this script.

## Review before running

From the project root, preview the matching inputs and proposed outputs:

```powershell
.\scripts\pod_spectra_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/pod_energy_valid90/pod.h5" --output-subfolder modal_spectra --dry-run
```

The preview lists paths only: it does not create files or calculate spectra.
Change `pod_energy_valid90` in the pattern to the folder containing your saved
PODs, for example `pod_valid90`. Prefer an explicit pattern to avoid mixing
different processing runs. After reviewing the list, run:

```powershell
.\scripts\pod_spectra_folder.bat "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10" --pattern "*/pod_energy_valid90/pod.h5" --output-subfolder modal_spectra --welch-seconds 4 --band-half-width-hz 0.2 --harmonics 3 --plot-max-hz 20
```

## Processing workflow

1. Validate finite coefficients and increasing times in seconds. Infer the
   sample interval from the full record. Reject irregular intervals rather than
   silently resampling; the default relative interval tolerance is 0.001 to
   accommodate float32 timestamp rounding.
2. Remove the coefficient mean and calculate a one-sided, full-record Hann PSD
   for each saved mode. Identify the strongest positive-frequency bin across
   the entire available spectrum, not just the plotted frequency range.
3. Calculate an additional Welch PSD using Hann segments, 50% overlap and
   segment-wise demeaning. Default segment length is four seconds, clipped to
   the available record; actual length, segment count, unused tail samples and
   frequency-bin spacing are saved. A single segment provides no averaging
   benefit. No zero padding is used.
4. For quantitative band fractions, use a separate full-record rectangular
   periodogram. Its discrete integral equals the demeaned coefficient variance
   by Parseval's identity. Select bin centres inside the requested bands and
   report the actual selected bin centres. Rectangular spectra can have spectral
   leakage: these fractions depend on record length and bandwidth. The smoother
   Hann/Welch display curves are not the estimator used in this table.
5. For each band, save its fraction of each mode's variance and multiply by the
   saved POD energy fraction to estimate its contribution to total POD energy.
   Summing these contributions covers only the saved modes. Missing modes are
   not renormalized away. This calculation inherits the original POD's spatial
   mask, mean subtraction and gap-filling assumptions.

Default references follow the documented experiment labels in POD source
metadata: LF=2 Hz and HF=5 Hz. Static or unrecognized cases still get spectra
and peak summaries but no forcing-band entries. `--forcing-frequency-hz` supplies
an explicit reference for every selected file. Bands default to +/-0.2 Hz around
harmonics 1, 2 and 3; for SurgeLF these are 1.8–2.2, 3.8–4.2 and 5.8–6.2 Hz.
Bands above Nyquist or containing no Fourier bins are omitted with a note.
Overlapping requested harmonic bands are rejected. For a ten-second record,
full-record bin spacing is about 0.1 Hz; four-second Welch segments give about
0.25 Hz spacing. Window width also limits effective spectral resolution.

## Outputs beside each POD

```text
Case_3.5D/
  pod_energy_valid90/
    pod.h5
    modal_spectra/
      modal_spectra.h5
      mode_summary.csv
      forcing_bands.csv
      mode_frequency_heatmap.png
      mode_spectra_01_06.png
      mode_spectra_07_12.png
      ...
      summary.json
```

`mode_summary.csv` reports POD energy fractions, coefficient variance and
dominant frequencies from both estimators. `forcing_bands.csv` reports all
mode/harmonic pairs. All spectra are stored in HDF5 as (frequency, mode).
Spectral pages show every saved mode, six per page. The heatmap normalizes each
mode over its full frequency range so that weak modes remain visible; it does
not compare absolute modal energies. `--plot-max-hz` affects figures only.

Original POD files are read-only. Existing result directories are never
overwritten. A manifest is saved at the input root; failures include their
reason, processing continues to subsequent files, and the batch returns a
nonzero exit code if any file fails. Choose a new output-subfolder name for
another analysis setting.

## Interpretation

### Integrate an arbitrary frequency band

After generating spectra, collect the 0–1 Hz energy across the 100-mode runs:

```powershell
python scripts/integrate_pod_band.py "D:\binning64voxel50overlap\outputs_flow" --band-hz 0 1 --output "D:\binning64voxel50overlap\outputs_flow\pod100_energy_0_to_1Hz.csv"
```

The default input pattern is
`*/pod_valid90_modes100_all/modal_spectra_100/modal_spectra.h5`.
Use `--pattern` to select another saved run and `--band-hz LOW HIGH` for another
band. The script reads saved spectra only. Existing CSVs are protected; select
a new `--output` for another calculation.

The CSV reports one row per case, including selected bin centres, saved POD
energy fraction, spatial coverage, `band_fraction_of_total_energy` and
`band_energy`. The fraction retains the complete POD fluctuation-energy
denominator, including unsaved modes. Absolute band energy is that fraction
times the original POD's `total_fluctuation_energy`: spatially averaged kinetic
energy per unit mass on retained support, in m²/s² if velocities are m/s.
It excludes the steady mean and unsaved-mode contributions. Integration uses
the full-record rectangular periodogram and includes bin centres on both band
boundaries (with a small tolerance for timestamp rounding). Adjacent bands
therefore share a boundary bin and should not simply be added together.
Bands above Nyquist or without any bin centres are rejected. All matched files
must validate before the comparison CSV is written.

A POD mode can contain multiple frequencies. The strongest peak is not a
unique intrinsic mode frequency. Energy in a forcing-frequency band also
includes broadband energy and leakage; it is not the phase-locked harmonic
energy estimated by `analyze_forcing_response.py`. These results do not measure
platform mechanical work or supply confidence intervals/significance tests.

To compare integrated CSVs at matching downstream positions (Flow background,
Static reference, and all moving cases), run:

```powershell
python scripts/compare_pod_band.py --flow-csv "D:\binning64voxel50overlap\outputs_flow\pod100_energy_0_to_1Hz.csv" --cases-csv "D:\binning64voxel50overlap\outputs\interpolation_x5_t2_t-x-y-z_N10\pod100_energy_0_to_1Hz.csv" --output "outputs/pod_band_comparison_0_to_1Hz"
```

Choose a new output directory for each run. It writes `static_vs_flow.csv`,
`all_comparisons.csv`, `case_energies.csv`, and PNG/PDF plots for static versus
background and all cases. Absolute differences, ratios and percentage changes
are provided for both band energy and total fluctuation energy, with Flow and
Static baselines at the same nominal x/D. Coverage, retained energy fractions,
spectral resolution and numerical settings accompany the data. These are
comparisons on each case's own POD support, not a recomputation on a common
spatial mask. The differences do not isolate turbine-generated turbulence.

Only synthetic fixtures were used to test this new batch workflow before user
review. No experimental POD batch was run automatically.

```powershell
python -m pytest tests/test_pod_spectra.py -q
```
