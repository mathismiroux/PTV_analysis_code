"""Compute local forcing-response energy profiles from an existing velocity file."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.forcing_response import run_forcing_response


def add_energy_settings(parser):
    parser.add_argument('--frequency-hz', type=float, help='Explicit reference frequency; otherwise velocity metadata or case convention (LF=2, HF=5 Hz)')
    parser.add_argument('--phase-bins', type=int, default=32)
    parser.add_argument('--phase-offset', type=float, default=0., help='Radians added to 2*pi*f*t')
    parser.add_argument('--min-phase-samples', type=int, default=3)
    parser.add_argument('--rotor-diameter', type=float, default=1200., help='In source coordinate units')
    parser.add_argument('--rotor-y', type=float, default=0.)
    parser.add_argument('--rotor-z', type=float, default=0.)
    parser.add_argument('--y-bounds-d', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('--z-bounds-d', nargs=2, type=float, metavar=('MIN', 'MAX'))
    parser.add_argument('--slab-width-d', type=float, default=.1)
    parser.add_argument('--slab-origin-d', type=float, default=0.)


def energy_settings(args):
    return {key: getattr(args, key) for key in ('min_valid_fraction', 'frequency_hz', 'phase_bins',
            'phase_offset', 'min_phase_samples', 'rotor_diameter', 'rotor_y', 'rotor_z',
            'y_bounds_d', 'z_bounds_d', 'slab_width_d', 'slab_origin_d', 'exclude_filled', 'zero_invalid')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-valid-fraction', type=float, default=.90)
    parser.add_argument('--exclude-filled', action='store_true')
    parser.add_argument('--zero-invalid', action='store_true')
    add_energy_settings(parser)
    args = parser.parse_args()
    print(json.dumps(run_forcing_response(args.source, args.output, **energy_settings(args)), indent=2))


if __name__ == '__main__':
    main()
