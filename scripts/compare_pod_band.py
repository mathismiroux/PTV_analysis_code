"""Compare integrated POD band energies at matched downstream positions."""
import argparse
import csv
import json
from pathlib import Path
import re

import h5py
import numpy as np


def read_cases(paths):
    cases = {}
    band = None
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            for raw in csv.DictReader(stream):
                match = re.fullmatch(r'(.+)_([0-9]+(?:\.[0-9]+)?)D', raw['case'])
                if not match:
                    raise ValueError(f'Cannot parse downstream position: {raw["case"]}')
                group, position = match[1], float(match[2])
                key = group, position
                if key in cases:
                    raise ValueError(f'Duplicate case: {raw["case"]}')
                row = dict(raw, group=group, x_over_d=position)
                for field in ('band_energy', 'total_fluctuation_energy', 'band_fraction_of_total_energy',
                              'saved_pod_energy_fraction', 'spatial_coverage', 'lower_hz', 'upper_hz',
                              'bin_spacing_hz', 'first_bin_hz', 'last_bin_hz'):
                    row[field] = float(raw[field])
                    if not np.isfinite(row[field]):
                        raise ValueError(f'Non-finite {field}: {raw["case"]}')
                row['modes'] = int(raw['modes'])
                limits = row['lower_hz'], row['upper_hz']
                if not 0 <= limits[0] < limits[1] or (band is not None and limits != band):
                    raise ValueError('All inputs must use the same valid frequency band')
                band = limits
                if (row['total_fluctuation_energy'] <= 0 or row['band_energy'] < 0
                        or not 0 <= row['band_fraction_of_total_energy'] <= row['saved_pod_energy_fraction'] + 1e-6
                        or not 0 <= row['saved_pod_energy_fraction'] <= 1 + 1e-6
                        or not 0 <= row['spatial_coverage'] <= 1):
                    raise ValueError(f'Invalid energy or coverage: {raw["case"]}')
                if not np.isclose(row['band_energy'], row['band_fraction_of_total_energy'] *
                                  row['total_fluctuation_energy'], rtol=1e-6, atol=1e-14):
                    raise ValueError(f'Inconsistent energy normalization: {raw["case"]}')
                # Read only small spectral metadata, never full velocity fields.
                with h5py.File(raw['spectra_file'], 'r') as h:
                    metadata = json.loads(h.attrs['metadata_json'])
                pod = metadata['input_pod_metadata']
                for field in ('min_valid_fraction', 'iterations', 'oversampling', 'seed',
                              'max_relative_eigen_residual', 'spatial_weighting', 'missing_policy'):
                    row[field] = pod.get(field, '')
                row['duration_s'] = metadata['full_record_duration_s']
                cases[key] = row
    if not cases:
        raise ValueError('No cases in input CSVs')
    return cases, band


def compare(cases, background='Flow', reference='Static'):
    rows = []
    for (group, position), case in sorted(cases.items()):
        if group == background:
            continue
        row = dict(case)
        for name, baseline_group in [('background', background), ('static', reference)]:
            if (baseline_group, position) not in cases:
                raise ValueError(f'Missing {baseline_group} baseline at {position:g}D')
            baseline = cases[baseline_group, position]
            for field in ('band_energy', 'total_fluctuation_energy', 'band_fraction_of_total_energy',
                          'spatial_coverage', 'saved_pod_energy_fraction'):
                row[f'{name}_{field}'] = baseline[field]
            for field in ('band_energy', 'total_fluctuation_energy'):
                value, base = case[field], baseline[field]
                row[f'{field}_minus_{name}'] = value-base
                row[f'{field}_ratio_to_{name}'] = value/base if base > 0 else ''
                row[f'{field}_change_percent_vs_{name}'] = 100*(value/base-1) if base > 0 else ''
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_common_support(cases):
    references = {}
    for (_, position), row in cases.items():
        with h5py.File(row['spectra_file'], 'r') as h:
            metadata = json.loads(h.attrs['metadata_json'])
        with h5py.File(metadata['pod_file'], 'r') as h:
            arrays = {key:h[key][:] for key in ('x', 'y', 'z', 'support', 'spatial_weight')}
        if position in references:
            for key, expected in references[position].items():
                if not np.array_equal(arrays[key], expected):
                    raise ValueError(f'{row["case"]}: {key} differs from other cases at {position:g}D')
        else:
            references[position] = arrays


def plot_cases(cases, groups, background, band, output, common_support=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for group in groups:
        data = sorted((r for (g, _), r in cases.items() if g == group), key=lambda r:r['x_over_d'])
        x = [r['x_over_d'] for r in data]
        values = [
            [r['band_energy'] for r in data],
            [r['total_fluctuation_energy'] for r in data],
            [100*r['band_fraction_of_total_energy'] for r in data],
            [r['band_energy']/cases[background, r['x_over_d']]['band_energy']
             if cases[background, r['x_over_d']]['band_energy'] > 0 else np.nan for r in data],
            [100*r['spatial_coverage'] for r in data],
            [100*r['saved_pod_energy_fraction'] for r in data],
        ]
        for ax, y in zip(axes.flat, values):
            ax.plot(x, y, 'o-', label=group)
    titles = ['Captured band energy', 'Total fluctuation energy (all modes)',
              'Captured band / total fluctuation energy', f'Band energy relative to {background}',
              'Retained grid voxels', 'Total energy captured by saved modes']
    labels = ['Energy (input velocity units squared)', 'Energy (input velocity units squared)',
              'Percent (%)', f'Ratio to {background}', 'Coverage (%)', 'Captured energy (%)']
    for ax, title, label in zip(axes.flat, titles, labels):
        ax.set(title=title, ylabel=label, xlabel='Downstream position x/D')
        ax.grid(alpha=.25)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    support_label = ('verified common spatial support at each x/D' if common_support
                     else 'spatial support not verified across cases')
    fig.suptitle(f'{band[0]:g}–{band[1]:g} Hz POD comparison — {support_label}')
    fig.text(.5, .012, 'Volume-averaged fluctuation energy per unit mass; band estimates exclude unsaved modes. '
             'Ratios and differences are descriptive, not isolated turbine-generated turbulence.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-csv', type=Path, required=True)
    parser.add_argument('--cases-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--background', default='Flow')
    parser.add_argument('--reference', default='Static')
    parser.add_argument('--require-common-support', action='store_true',
                        help='Verify identical POD coordinates, masks and weights at each x/D before comparing')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output directory')
    try:
        cases, band = read_cases([args.flow_csv, args.cases_csv])
        rows = compare(cases, args.background, args.reference)
        if args.require_common_support:
            verify_common_support(cases)
        if not rows or not any(r['group'] == args.reference for r in rows):
            raise ValueError('No static reference cases to compare')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True)
    write_csv(args.output/'case_energies.csv', list(cases.values()))
    write_csv(args.output/'all_comparisons.csv', rows)
    write_csv(args.output/'static_vs_flow.csv', [r for r in rows if r['group'] == args.reference])
    plot_cases(cases, [args.background, args.reference], args.background, band, args.output/'static_vs_flow', args.require_common_support)
    groups = [args.background, args.reference] + sorted({g for g, _ in cases} - {args.background, args.reference})
    plot_cases(cases, groups, args.background, band, args.output/'all_cases', args.require_common_support)
    notes = (
        'Comparison of saved POD band-energy estimates at matching nominal x/D.\n'
        'Absolute energy is already volume averaged on each case\'s retained support; do not divide by voxel count.\n'
        'Units are m^2/s^2 only if original velocity units are m/s.\n'
        'Total fluctuation energy includes unsaved modes; band energy includes only saved modes.\n'
        'Ratios are case / baseline; percentage changes are 100 * (ratio - 1).\n'
        'The within-case band fraction has a different denominator: the case\'s own total fluctuation energy.\n'
        'Time-mean subtraction retains coherent motion-induced fluctuations, so moving-case energy is not purely stochastic TKE.\n'
        'Subtracting background energy is a descriptive difference, not a measured turbine production term.\n'
        'Eigenpair residuals are numerical diagnostics, not percentage uncertainty on band energy.\n'
        'No confidence intervals are inferred from these single-record summaries.\n'
    )
    notes += ('Identical coordinates, masks and spatial weights verified across cases at each x/D.\n'
              if args.require_common_support else
              'Spatial support has not been verified across cases; sampled regions may differ.\n')
    (args.output/'README.txt').write_text(notes, encoding='utf-8')
    print(f'Saved {len(rows)} matched comparisons and plots to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
