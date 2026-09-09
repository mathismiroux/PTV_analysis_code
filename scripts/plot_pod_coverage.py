"""Plot the POD coverage CSV by forcing case and downstream distance."""
import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--pod-run', default='pod_energy_valid90', help='Select this POD folder name; excludes other runs/trials')
    parser.add_argument('--output', type=Path, help='PNG output; also writes a PDF with the same stem')
    args = parser.parse_args()
    output = args.output or args.csv_file.with_name('pod_coverage_by_distance.png')
    if output.suffix.lower() != '.png':
        parser.error('--output must end in .png')
    pdf = output.with_suffix('.pdf')
    if output.exists() or pdf.exists():
        parser.error('Output exists; choose a new --output')
    groups, thresholds, seen = {}, set(), set()
    with args.csv_file.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if re.split(r'[/\\]', row['pod_folder'])[-1] != args.pod_run:
                continue
            match = re.fullmatch(r'(.+?)_(\d+(?:\.\d+)?)D', row['case'])
            if not match:
                parser.error(f"Cannot parse downstream distance: {row['case']}")
            case, distance = match[1], float(match[2])
            if (case, distance) in seen:
                parser.error(f'Duplicate {case} at {distance}D; select a single POD run')
            seen.add((case, distance))
            groups.setdefault(case, []).append((distance, float(row['retained_grid_percent'])))
            if row['min_valid_snapshot_percent']:
                thresholds.add(float(row['min_valid_snapshot_percent']))
    if not groups:
        parser.error('No rows matched --pod-run')
    styles = {'Static': ('#222222', '-', 'o', 'Static'),
              'SurgeLF': ('#c43c39', '--', 'o', 'Surge LF'),
              'SurgeHF': ('#c43c39', '-', 's', 'Surge HF'),
              'PitchLF': ('#2866ac', '--', '^', 'Pitch LF'),
              'PitchHF': ('#2866ac', '-', 'D', 'Pitch HF')}
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    order = [case for case in styles if case in groups] + sorted(set(groups)-set(styles))
    for case in order:
        points = sorted(groups[case])
        color, line, marker, label = styles.get(case, (None, '-', 'o', case))
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                linestyle=line, marker=marker, label=label, linewidth=1.8, markersize=6)
    locations = sorted({distance for _, distance in seen})
    ax.set(xticks=locations, xticklabels=[f'{x:g}' for x in locations], ylim=(0, 100),
           xlabel='Downstream distance, x/D', ylabel='Retained grid coverage (%)')
    title = 'Local POD spatial coverage'
    if len(thresholds) == 1:
        title += f'\nVoxels retained with ≥{next(iter(thresholds)):g}% valid snapshots'
    ax.set_title(title, pad=12)
    ax.grid(alpha=.25)
    ax.legend(ncol=2, frameon=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f'Plotted {len(seen)} volumes in {len(groups)} cases.\n{output}\n{pdf}')


if __name__ == '__main__':
    main()
