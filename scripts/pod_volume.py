"""Compute leading 3-component spatial POD modes for one velocity volume."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.pod import run_pod


def add_settings(parser):
    parser.add_argument('--mean-file', type=Path, help='Existing temporal mean; defaults to mean.nc beside each velocity file')
    parser.add_argument('--modes', type=int, default=20)
    parser.add_argument('--min-valid-fraction', type=float, default=1.0)
    parser.add_argument('--exclude-filled', action='store_true')
    parser.add_argument('--zero-invalid', action='store_true', help='Treat all-zero velocity vectors as missing')
    parser.add_argument('--iterations', type=int, default=2, help='Increase for better convergence')
    parser.add_argument('--oversampling', type=int, default=15)
    parser.add_argument('--seed', type=int, default=0)


def settings(args):
    return {key: getattr(args, key) for key in
            ('modes', 'min_valid_fraction', 'exclude_filled', 'zero_invalid',
             'iterations', 'oversampling', 'seed', 'mean_file')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    add_settings(parser)
    args = parser.parse_args()
    result = run_pod(args.source, args.output, **settings(args))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
