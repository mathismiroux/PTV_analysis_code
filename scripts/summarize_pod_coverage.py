"""Collect spatial support from local POD summary.json files (no velocity reads)."""
import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = ['case', 'pod_folder', 'retained_voxels', 'total_voxels',
          'retained_grid_percent', 'excluded_grid_percent',
          'min_valid_snapshot_percent', 'snapshots', 'source_file', 'summary_file']


def collect_coverage(root, pattern='**/summary.json'):
    root = Path(root).resolve()
    rows, skipped, errors = [], [], []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            # Frequency, energy-profile, mode-pair and global summaries use
            # different schemas/denominators; do not mix them with local POD.
            if not isinstance(data, dict) or not {'retained_voxels', 'total_voxels', 'spatial_coverage'} <= data.keys():
                skipped.append(str(path))
                continue
            retained, total = data['retained_voxels'], data['total_voxels']
            if (not isinstance(retained, int) or not isinstance(total, int)
                    or total <= 0 or not 0 <= retained <= total):
                raise ValueError('Invalid retained/total voxel counts')
            coverage = retained/total
            stored = float(data['spatial_coverage'])
            if not math.isfinite(stored) or not math.isclose(stored, coverage, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError('Stored spatial_coverage disagrees with retained_voxels/total_voxels')
            threshold = data.get('min_valid_fraction')
            if threshold is not None and not 0 < float(threshold) <= 1:
                raise ValueError('Invalid min_valid_fraction')
            source = Path(data.get('source', ''))
            case = source.parent.name if source.stem in ('interpolated_velocity', 'velocity', 'raw') else source.stem
            rows.append(dict(case=case or path.parent.parent.name,
                             pod_folder=str(path.parent.relative_to(root)),
                             retained_voxels=retained, total_voxels=total,
                             retained_grid_percent=100*coverage,
                             excluded_grid_percent=100*(1-coverage),
                             min_valid_snapshot_percent=100*float(threshold) if threshold is not None else '',
                             snapshots=data.get('snapshots', ''), source_file=str(source),
                             summary_file=str(path)))
        except (OSError, ValueError, TypeError) as exc:
            errors.append(dict(summary_file=str(path), reason=str(exc)))
    return rows, skipped, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--pattern', default='**/summary.json', help='Use */pod_energy_valid90/summary.json to select one POD run')
    parser.add_argument('--output', type=Path, help='CSV path; default FOLDER/pod_coverage.csv')
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    output = args.output or args.folder/'pod_coverage.csv'
    if output.exists():
        parser.error(f'Output already exists: {output}; choose a new --output')
    rows, skipped, errors = collect_coverage(args.folder, args.pattern)
    for error in errors:
        print(f"ERROR: {error['summary_file']}: {error['reason']}")
    if not rows:
        print(f'No valid local POD summaries found; {len(skipped)} unrelated summaries skipped.')
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"{'Case':<22} {'Retained':>10} {'Total':>10} {'Grid %':>9}  POD folder")
    for row in rows:
        print(f"{row['case']:<22} {row['retained_voxels']:>10} {row['total_voxels']:>10} {row['retained_grid_percent']:>8.2f}%  {row['pod_folder']}")
    print(f'Wrote {len(rows)} POD runs to {output}. Skipped {len(skipped)} unrelated summaries; {len(errors)} errors.')
    print('Grid % is the retained voxel fraction, not the valid-snapshot threshold. Separate runs remain separate rows.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
