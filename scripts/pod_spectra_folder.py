"""Batch frequency analysis of existing local pod.h5 files; never reruns POD."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.pod_spectra import analyze_pod_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--pattern', default='*/pod_valid90/pod.h5', help='Explicit glob identifying the POD run to analyze')
    parser.add_argument('--output-subfolder', default='modal_spectra')
    parser.add_argument('--forcing-frequency-hz', type=float, help='Override for all matched volumes; otherwise LF=2, HF=5 Hz, Static/unknown has no reference')
    parser.add_argument('--band-half-width-hz', type=float, default=.2)
    parser.add_argument('--harmonics', type=int, default=3)
    parser.add_argument('--welch-seconds', type=float, default=4.)
    parser.add_argument('--plot-max-hz', type=float, default=20.)
    parser.add_argument('--sampling-tolerance', type=float, default=.001)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.output_subfolder):
        parser.error('Output subfolder must be a simple name using letters, digits, underscores or hyphens')
    sources = sorted(p.resolve() for p in args.folder.glob(args.pattern) if p.is_file())
    if not sources:
        parser.error('No POD files matched --pattern')
    targets = [p.parent/args.output_subfolder for p in sources]
    if len(set(targets)) != len(targets):
        parser.error('Multiple input files would share the same output folder')
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        parser.error('Output already exists; choose a new --output-subfolder: '+', '.join(existing))
    if args.dry_run:
        for source, target in zip(sources, targets):
            print(f'{source} -> {target}')
        print(f'{len(sources)} POD files planned; no files created or modified.')
        return 0
    manifest = args.folder/f'{args.output_subfolder}_batch_manifest.json'
    if manifest.exists():
        parser.error(f'Manifest exists: {manifest}; choose a new --output-subfolder')
    rows = []
    for source, output in zip(sources, targets):
        row = dict(source=str(source), output=str(output))
        try:
            summary = analyze_pod_file(source, output, **{k: getattr(args, k) for k in
                                       ('forcing_frequency_hz', 'band_half_width_hz', 'harmonics',
                                        'welch_seconds', 'plot_max_hz', 'sampling_tolerance')})
            row.update(status='complete', summary=summary)
        except Exception as exc:
            row.update(status='failed', reason=str(exc))
        rows.append(row)
        manifest.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f"{source}: {row['status']}" + (f" — {row['reason']}" if 'reason' in row else ''), flush=True)
    return int(any(r['status']=='failed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
