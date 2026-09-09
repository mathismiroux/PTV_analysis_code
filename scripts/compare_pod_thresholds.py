"""Compare existing local POD runs made at different valid-snapshot thresholds."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--run-folders', nargs='+', default=['pod_energy_valid90', 'pod_valid92', 'pod_valid95'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    comparisons, summary_rows, mode_rows = {}, [], []
    invariant_keys = ('source', 'mean_file', 'snapshots', 'total_voxels', 'method',
                      'exclude_filled', 'zero_invalid', 'iterations', 'oversampling', 'seed', 'saved_modes')
    for case in args.cases:
        runs = []
        for folder in args.run_folders:
            path = args.folder/case/folder
            info = json.loads((path/'summary.json').read_text())
            with h5py.File(path/'pod.h5') as f:
                fractions = f['energy_fraction'][:]
                eigenvalues = f['eigenvalue'][:]
                residuals = f['relative_eigen_residual'][:]
                support = f['support'][:].astype(bool)
                retained = int(support.sum())
            if retained != info['retained_voxels'] or not np.isclose(fractions.sum(), info['saved_energy_fraction']):
                raise ValueError(f'Inconsistent summary and POD file: {path}')
            if runs:
                for key in invariant_keys:
                    if info.get(key) != runs[0]['info'].get(key):
                        raise ValueError(f'{case}: setting {key} differs between runs')
            runs.append(dict(path=str(path), info=info, fractions=fractions, support=support))
            row = dict(case=case, valid_snapshot_threshold_percent=100*info['min_valid_fraction'],
                       retained_grid_percent=100*info['spatial_coverage'],
                       retained_voxels=retained, total_voxels=info['total_voxels'],
                       total_pod_fluctuation_energy=info['total_fluctuation_energy'],
                       max_saved_mode_relative_eigen_residual=float(residuals.max()),
                       max_first_two_relative_eigen_residual=float(residuals[:2].max()),
                       pod_folder=str(path))
            for count in (1, 2, 5, 10, 20):
                row[f'first_{count}_modes_energy_percent'] = float(100*fractions[:count].sum()) if len(fractions) >= count else None
            summary_rows.append(row)
            for i in range(len(fractions)):
                mode_rows.append(dict(case=case, valid_snapshot_threshold_percent=row['valid_snapshot_threshold_percent'],
                                      mode=i+1, energy_percent=float(100*fractions[i]),
                                      cumulative_energy_percent=float(100*fractions[:i+1].sum()),
                                      modal_energy=float(eigenvalues[i]/2),
                                      relative_eigen_residual=float(residuals[i])))
        runs.sort(key=lambda r: r['info']['min_valid_fraction'])
        if len(set(r['info']['min_valid_fraction'] for r in runs)) != len(runs):
            raise ValueError(f'Duplicate thresholds for {case}')
        for previous, current in zip(runs, runs[1:]):
            if previous['support'].shape != current['support'].shape or np.any(current['support'] & ~previous['support']):
                raise ValueError(f'{case}: support is not nested as the threshold increases')
        comparisons[case] = runs
    args.output.mkdir(parents=True)
    write_csv(args.output/'coverage_and_energy.csv', summary_rows)
    write_csv(args.output/'modal_energy_comparison.csv', mode_rows)
    fig, axes = plt.subplots(len(comparisons), 3, figsize=(15, 4.2*len(comparisons)), squeeze=False, layout='constrained')
    colors = ['#333333', '#2878b5', '#d34d37', '#7f3c8d']
    for row, (case, runs) in enumerate(comparisons.items()):
        for i, run in enumerate(runs):
            info, fraction = run['info'], run['fractions']
            label = f"{100*info['min_valid_fraction']:g}% valid; {100*info['spatial_coverage']:.1f}% grid"
            index = np.arange(1, len(fraction)+1)
            axes[row, 0].plot(index, 100*fraction, 'o-', color=colors[i%len(colors)], label=label, markersize=3)
            axes[row, 1].plot(index, 100*np.cumsum(fraction), 'o-', color=colors[i%len(colors)], markersize=3)
        threshold = [100*r['info']['min_valid_fraction'] for r in runs]
        coverage = [100*r['info']['spatial_coverage'] for r in runs]
        axes[row, 2].plot(threshold, coverage, 'o-', color='#444444')
        for t, c in zip(threshold, coverage):
            axes[row, 2].annotate(f'{c:.2f}%', (t, c), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
        axes[row, 0].set(title=case, xlabel='Mode', ylabel='Individual POD energy (%)')
        axes[row, 1].set(title='Cumulative modal energy', xlabel='Number of modes', ylabel='Cumulative POD energy (%)', ylim=(0, 100))
        axes[row, 2].set(title='Spatial support', xlabel='Required valid snapshots (%)', ylabel='Retained grid (%)',
                         xticks=threshold, ylim=(0, min(100, max(coverage)*1.3+2)))
        axes[row, 0].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=.25)
    fig.suptitle('Local POD sensitivity to validity threshold\nThreshold changes the retained spatial region; energy fractions have a different domain-specific denominator at each threshold.')
    fig.savefig(args.output/'pod_threshold_comparison.png', dpi=180)
    fig.savefig(args.output/'pod_threshold_comparison.pdf')
    plt.close(fig)
    (args.output/'interpretation.txt').write_text(
        'All snapshots remain in each POD. Increasing the threshold excludes voxels with more missing samples.\n'
        'Remaining gaps are zero fluctuations (local temporal mean), as in the original POD script.\n'
        'Both spatial support and the energy denominator change. This measures sensitivity of the workflow,\n'
        'not a change in the physical experiment. Modes are ranked independently and can rotate or exchange order.\n'
        'Total energy is the spatially averaged, processed fluctuation energy on retained support, not its volume integral.\n'
        'The solver is approximate; eigenpair residuals are included to assess numerical uncertainty.\n', encoding='utf-8')
    for row in summary_rows:
        print(f"{row['case']} | valid {row['valid_snapshot_threshold_percent']:g}% | grid {row['retained_grid_percent']:.2f}% | first 2 modes {row['first_2_modes_energy_percent']:.2f}% | first 20 modes {row['first_20_modes_energy_percent']:.2f}%")
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
