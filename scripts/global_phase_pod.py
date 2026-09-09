"""Stitch existing phase averages and compute one global POD per forcing case."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.global_pod import discover_cases, inspect_sources, run_global_pod


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--case', help='Exact case group, e.g. SurgeLF')
    choice.add_argument('--all-cases', action='store_true')
    parser.add_argument('--pattern', default='*/phase_average*.nc')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--min-valid-fraction', type=float, default=.90)
    parser.add_argument('--modes', type=int, default=20)
    parser.add_argument('--spacing', type=float, help='Common isotropic spacing in source coordinate units; default coarsest median input spacing')
    parser.add_argument('--rotor-diameter', type=float, default=1200., help='In source coordinate units; used only for plots')
    parser.add_argument('--max-grid-voxels', type=int, default=2000000)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    cases = discover_cases(args.folder, args.pattern, selected_case=args.case)
    if args.case:
        if args.case not in cases:
            parser.error(f'Case {args.case} not found; available: {", ".join(cases)}')
        cases = {args.case: cases[args.case]}
    if not cases:
        parser.error('No phase-average files matched')
    if args.output_root.exists():
        parser.error('Choose a new output root')
    args.output_root.mkdir(parents=True)
    rows = []
    for case, paths in cases.items():
        row = dict(case=case, sources=[str(p) for p in paths])
        try:
            if args.dry_run:
                infos = inspect_sources(paths)
                row.update(status='planned', frequency_hz=infos[0]['frequency'])
            else:
                summary = run_global_pod(paths, args.output_root / case,
                                         args.min_valid_fraction, args.modes, args.spacing,
                                         args.rotor_diameter, args.max_grid_voxels)
                row.update(status='complete', summary=summary)
        except Exception as exc:
            row.update(status='failed', reason=str(exc))
        rows.append(row)
        (args.output_root / 'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(row, indent=2), flush=True)
    return int(any(row['status'] == 'failed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
