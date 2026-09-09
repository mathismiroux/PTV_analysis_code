"""Run local POD and forcing-response energy profiles for each velocity volume."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pod_volume import add_settings, settings, run_pod
from scripts.interpolate_velocity_folder import is_raw_velocity_timeseries
from scripts.analyze_forcing_response import add_energy_settings, energy_settings
from ptv_flow.forcing_response import common_bounds, resolve_frequency, run_forcing_response, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output-root', type=Path)
    destination.add_argument('--output-subfolder', help='Save in this subfolder beside each velocity file, e.g. pod_valid90')
    parser.add_argument('--pattern', default='*.nc', help='Use **/interpolated_velocity.nc for nested case folders')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--analysis', choices=('both', 'pod', 'energy'), default='both',
                        help='Default: both; energy can add profiles to an existing POD output folder')
    add_settings(parser)
    add_energy_settings(parser)
    parser.set_defaults(min_valid_fraction=.90)
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error('Input folder does not exist')
    sources = sorted(p for p in args.folder.glob(args.pattern) if p.is_file())
    if args.output_subfolder is not None:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.output_subfolder):
            parser.error('--output-subfolder must be a simple folder name using letters, digits, underscores or hyphens')
        suffix = ('_energy' if args.analysis == 'energy' else '_pod' if args.analysis == 'pod' else '')
        suffix += '_dry_run' if args.dry_run else ''
        manifest = args.folder / f'{args.output_subfolder}{suffix}_manifest.json'
        if manifest.exists():
            parser.error(f'Manifest already exists: {manifest}; choose a new --output-subfolder')
    else:
        if args.output_root.exists() and args.analysis != 'energy':
            parser.error('Choose a new output root')
        suffix = '_dry_run' if args.dry_run else ''
        name = 'energy_manifest' if args.analysis == 'energy' else 'manifest'
        manifest = args.output_root / f'{name}{suffix}.json'
        if manifest.exists():
            parser.error(f'Manifest already exists: {manifest}')
    planned = []
    for source in sources:
        relative = source.relative_to(args.folder).with_suffix('')
        output = (source.parent / args.output_subfolder if args.output_subfolder
                  else args.output_root / relative)
        valid, reason = is_raw_velocity_timeseries(source)
        planned.append((source, output, valid, reason))
    valid_outputs = [output.resolve() for _, output, valid, _ in planned if valid]
    if len(valid_outputs) != len(set(valid_outputs)):
        parser.error('Multiple velocity files would share an output folder; narrow --pattern or use --output-root')
    if not valid_outputs:
        parser.error('No velocity time series matched the input pattern')
    for output in valid_outputs:
        destination = output / 'forcing_response' if args.analysis == 'energy' else output
        if destination.exists():
            parser.error(f'Output already exists: {destination}; choose a new output name')
    if args.analysis != 'pod':
        if args.y_bounds_d is None or args.z_bounds_d is None:
            try:
                bounds = common_bounds([source for source, _, valid, _ in planned if valid],
                                       args.rotor_diameter, args.rotor_y, args.rotor_z,
                                       axes=tuple(k for k in ('y', 'z') if getattr(args, k+'_bounds_d') is None))
            except ValueError as exc:
                parser.error(str(exc))
            if args.y_bounds_d is None:
                args.y_bounds_d = bounds['y']
            if args.z_bounds_d is None:
                args.z_bounds_d = bounds['z']
        print(f'Energy analysis requested window: y/D={args.y_bounds_d}, z/D={args.z_bounds_d}', flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    combined_slabs = []
    for source, output, valid, reason in planned:
        row = dict(source=str(source), output=str(output), analysis=args.analysis)
        if not valid:
            row.update(status='skipped', reason=reason)
        elif args.dry_run:
            try:
                row.update(status='planned', pod_settings=settings(args))
                if args.analysis != 'pod':
                    frequency, provenance = resolve_frequency(source, args.frequency_hz)
                    row.update(energy_settings=energy_settings(args), frequency_hz=frequency,
                               frequency_source=provenance)
            except Exception as exc:
                row.update(status='failed', reason=str(exc))
        else:
            failures = []
            if args.analysis != 'energy':
                try:
                    row['summary'] = run_pod(source, output, **settings(args))
                    row['pod_status'] = 'complete'
                except Exception as exc:
                    row['pod_status'] = 'failed'
                    failures.append(f'POD: {exc}')
            if args.analysis != 'pod':
                try:
                    row['energy_summary'] = run_forcing_response(source, output / 'forcing_response', **energy_settings(args))
                    row['energy_status'] = 'complete'
                    with (output / 'forcing_response' / 'downstream_slabs.csv').open(newline='', encoding='utf-8') as stream:
                        for slab in csv.DictReader(stream):
                            combined_slabs.append(dict(case=row['energy_summary']['case'], source=str(source), **slab))
                except Exception as exc:
                    row['energy_status'] = 'failed'
                    failures.append(f'Energy: {exc}')
            row['status'] = 'failed' if failures else 'complete'
            if failures:
                row['reason'] = '; '.join(failures)
        rows.append(row)
        print(f"{source}: {row['status']}", flush=True)
        manifest.write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
        if combined_slabs:
            write_csv(manifest.with_name(manifest.stem + '_downstream_slabs.csv'), combined_slabs)
    return int(any(row['status'] == 'failed' for row in rows) or not rows)


if __name__ == '__main__':
    raise SystemExit(main())
