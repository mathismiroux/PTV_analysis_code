# 7HP velocity PSD command

Run from the repository root without installation:

```powershell
python scripts/psd_7hp.py --help
python scripts/psd_7hp.py 1028 --component 6
python scripts/psd_7hp.py 1028 --compare 1020 1060 --component 5 --heights 200 600 1000
```

The default acquisition sample rate is **245 Hz**, confirmed by the user.
Override it with `--fs HZ` if needed. All recordings in one comparison must
share that rate. The default Nyquist frequency is 122.5 Hz.

After `python -m pip install -e .`, the same command is available as
`ptv-7hp-psd` (for example `ptv-7hp-psd 1028 --component 6`).
`python -m ptv_flow.probe_psd` also works.

Components: `u` or `4` selects column 4 (streamwise), `v` or `5` selects column 5,
and `w` or `6` selects column 6 (default). These are column aliases; the vertical
component is suspected to be column 6 but remains unconfirmed. Velocity signs
are preserved; reversing a sign does not change the PSD.

Data default to `Desktop/7HP_data/Data7HPpost` under the user's university
OneDrive directory. Override with `--data-dir PATH`. Recording 1061 includes
its documented continuation 1063 unless `--no-continuation` is supplied.

The default uses all heights common to the selected recordings. `--heights`
selects raw heights in mm, computed as columns 1 + 2. Only the stationary
100 mm grid is retained. Each height gets its own plot panel.

Welch estimation uses 1024 samples per segment, 50% overlap, a periodic Hann
window, and removal of each segment's mean. Set `--nperseg` and `--overlap`
to change these. Frequency resolution is fs/nperseg. The one-sided PSD has
units (m/s)²/Hz; its integral estimates the window-weighted fluctuation variance.
No segment crosses an invalid row, height change, or file boundary. Short runs
and incomplete tails are excluded and counted in the output. Heights too short
for any compared recording are skipped for all recordings and reported.

The method assumes uniformly sampled rows at the supplied rate within each
contiguous run; these files have no verified timestamps to check missing samples.
PSD curves describe temporal fluctuations at each height, not the variation
of mean velocity along a profile.

Each run creates a new directory under `outputs/7hp_psd` containing `psd.png`,
`psd.pdf`, `psd.csv`, `summary.csv`, and `manifest.json`. Use `--output PATH`
to choose a new directory. `--max-hz` limits the plot while CSVs retain the
complete frequency range, including DC.
