"""Welch PSDs of 7HP recordings, compared separately at matching heights."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np


DEFAULT_DATA = (Path.home() / "OneDrive - Delft University of Technology"
                / "Desktop/7HP_data/Data7HPpost")
COMPONENTS = {"u": 4, "v": 5, "w": 6, "4": 4, "5": 5, "6": 6}


def load_runs(paths, column):
    """Keep contiguous finite, fixed-height runs; never bridge gaps or files."""
    runs = {}
    audit = []
    for path in paths:
        a = np.loadtxt(path, usecols=(0, 1, column-1), ndmin=2)
        height = a[:, 0] + a[:, 1]
        snapped = np.round(height / 100) * 100
        valid = np.isfinite(a).all(axis=1) & np.isclose(height, snapped, rtol=0, atol=1e-6)
        indices = np.flatnonzero(valid)
        breaks = np.flatnonzero((np.diff(indices) != 1)
                               | (np.diff(snapped[indices]) != 0)) + 1
        for chunk in np.split(indices, breaks):
            if chunk.size:
                runs.setdefault(float(snapped[chunk[0]]), []).append(a[chunk, 2])
        audit.append({"file": str(path.resolve()), "rows": len(a),
                      "excluded_rows": int((~valid).sum())})
    return runs, audit


def welch_runs(runs, fs, nperseg, overlap):
    """One-sided density, periodic Hann window, per-segment mean removal."""
    if not np.isfinite(fs) or fs <= 0 or nperseg < 8 or not 0 <= overlap < 1:
        raise ValueError("Require positive finite fs, nperseg >= 8, and 0 <= overlap < 1")
    step = nperseg - int(nperseg * overlap)
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    total = np.zeros(nperseg // 2 + 1)
    count, unused = 0, 0
    for run in runs:
        last = 0
        for start in range(0, len(run)-nperseg+1, step):
            segment = run[start:start+nperseg]
            spectrum = np.abs(np.fft.rfft((segment-segment.mean()) * window)) ** 2
            spectrum /= fs * np.sum(window**2)
            spectrum[1:] *= 2
            if nperseg % 2 == 0:
                spectrum[-1] /= 2
            total += spectrum
            count += 1
            last = start+nperseg
        unused += len(run)-last
    if not count:
        raise ValueError("No contiguous run long enough for --nperseg")
    return np.fft.rfftfreq(nperseg, 1/fs), total/count, count, unused


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=int, help="Primary recording number, e.g. 1028")
    parser.add_argument("--compare", nargs="+", type=int, default=[], metavar="REC")
    parser.add_argument("--component", choices=tuple(COMPONENTS), default="6",
                        help="u/4=column 4, v/5=column 5, w/6=column 6 (default). Vertical mapping unconfirmed.")
    parser.add_argument("--fs", type=float, default=245.0,
                        help="Acquisition sample rate in Hz (default: 245, confirmed by user)")
    parser.add_argument("--heights", nargs="+", type=float, help="Raw heights in mm (column 1 + 2); default common heights")
    parser.add_argument("--nperseg", type=int, default=1024, help="Samples per Welch segment (default 1024)")
    parser.add_argument("--overlap", type=float, default=.5, help="Overlap fraction, default 0.5")
    parser.add_argument("--max-hz", type=float, help="Upper plot frequency; CSV retains all frequencies")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, help="New output directory; default timestamped under outputs/7hp_psd")
    parser.add_argument("--no-continuation", action="store_true", help="Do not include documented continuation 1063 for 1061")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.fs) or args.fs <= 0 or args.nperseg < 8
            or not 0 <= args.overlap < 1):
        parser.error("Require positive finite --fs, --nperseg >= 8, and 0 <= --overlap < 1")
    if args.max_hz is not None and (not np.isfinite(args.max_hz) or args.max_hz <= 0):
        parser.error("--max-hz must be finite and positive")
    if args.heights is not None and not np.isfinite(args.heights).all():
        parser.error("--heights must be finite")
    recordings = list(dict.fromkeys([args.recording, *args.compare]))
    column = COMPONENTS[args.component]
    loaded, audits, labels = {}, {}, {}
    try:
        for rec in recordings:
            ids = [rec, 1063] if rec == 1061 and not args.no_continuation else [rec]
            loaded[rec], audits[rec] = load_runs([args.data_dir/f"post_rec{i}.txt" for i in ids], column)
            labels[rec] = " + ".join(map(str, ids))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    common = set.intersection(*(set(runs) for runs in loaded.values()))
    heights = sorted(set(args.heights) if args.heights is not None else common)
    if not heights or not set(heights) <= common:
        parser.error(f"Select heights present in all recordings. Common heights (mm): {sorted(common)}")
    spectra, summary, skipped = {}, [], []
    for height in heights:
        if any(not any(len(run) >= args.nperseg for run in loaded[r][height]) for r in recordings):
            skipped.append(height)
            continue
        for rec in recordings:
            f, psd, segments, unused = welch_runs(loaded[rec][height], args.fs, args.nperseg, args.overlap)
            spectra[rec, height] = (f, psd)
            summary.append({"recording": rec, "height_mm": height, "column": column,
                            "segments": segments, "unused_samples": unused,
                            "frequency_resolution_hz": args.fs/args.nperseg,
                            "integrated_psd_m2_s2": float(psd.sum() * args.fs/args.nperseg)})
    heights = [h for h in heights if h not in skipped]
    if not heights:
        parser.error("No common height has sufficient contiguous samples; reduce --nperseg")
    output = args.output or Path("outputs/7hp_psd") / (
        "_vs_".join(map(str, recordings)) + f"_col{column}_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists():
        parser.error(f"Output already exists; choose a new --output: {output}")
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncols = min(3, len(heights))
    fig, axes = plt.subplots((len(heights)+ncols-1)//ncols, ncols,
                             figsize=(5*ncols, 3.7*((len(heights)+ncols-1)//ncols)),
                             squeeze=False, layout="constrained")
    with (output/"psd.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recording", "height_mm", "column", "frequency_hz", "psd_m2_s2_per_hz"])
        for ax, height in zip(axes.flat, heights):
            for rec in recordings:
                f, psd = spectra[rec, height]
                ax.loglog(f[1:], np.where(psd[1:] > 0, psd[1:], np.nan), label=labels[rec])
                writer.writerows((rec, height, column, frequency, density) for frequency, density in zip(f, psd))
            ax.set(title=f"Height = {height:g} mm", xlabel="Frequency (Hz)", ylabel="PSD ((m/s)²/Hz)")
            if args.max_hz is not None:
                ax.set_xlim(right=min(args.max_hz, args.fs/2))
            ax.grid(which="both", alpha=.2)
            ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(heights):]:
        ax.set_visible(False)
    fig.suptitle(f"7HP velocity column {column} — Welch PSD; fs = {args.fs:g} Hz")
    for extension in ("png", "pdf"):
        fig.savefig(output/f"psd.{extension}", dpi=180)
    plt.close(fig)
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "inputs": audits, "skipped_short_heights_mm": skipped,
                "method": "One-sided Welch density; periodic Hann; per-segment constant detrend; segment-weighted mean. No joining across height changes, invalid rows, or files.",
                "assumption": "Rows within each contiguous height run are uniformly sampled at the supplied fs. No timestamps available to verify acquisition gaps.",
                "component_mapping": "4=streamwise; 5 and 6=transverse; vertical likely 6, unconfirmed."}
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if skipped:
        print(f"Skipped heights with insufficient contiguous samples: {skipped}")
    print(f"Saved PSD plots and CSVs to {output.resolve()}")


if __name__ == "__main__":
    main()
