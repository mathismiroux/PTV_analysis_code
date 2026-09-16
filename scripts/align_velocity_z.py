"""Align velocity files to a reference z grid without running POD or filling gaps."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np

from ptv_flow.pod_support import axis_map, fingerprint
from ptv_flow.reader import FlowDataset


def align_file(source, reference, output):
    source, reference, output = (Path(p).resolve() for p in (source, reference, output))
    partial = output.with_name(output.name+'.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(f'Output or partial file already exists: {output}')
    inputs = [fingerprint(source), fingerprint(reference)]
    with FlowDataset(source) as data, FlowDataset(reference) as grid:
        for key in 'xy':
            if not np.array_equal(data.coordinate(key), grid.coordinate(key)):
                raise ValueError(f'{key} coordinates differ; this command only aligns z')
        left, right, weight = axis_map(data.coordinate('z'), grid.coordinate('z'))
        shape = (data.shape[0], grid.shape[1], data.shape[2], data.shape[3])
    output.parent.mkdir(parents=True, exist_ok=True)
    valid_total = 0
    with h5py.File(source, 'r') as src, h5py.File(reference, 'r') as ref, h5py.File(partial, 'x') as dst:
        dst.attrs.update(operation='align_velocity_z', method='linear interpolation; all contributors must be valid',
            source_file=str(source), grid_reference_file=str(reference), invalid_samples='nan',
            created_utc=datetime.now(timezone.utc).isoformat(), case_id=src.attrs.get('case_id', source.parent.name),
            label=src.attrs.get('label', source.parent.name), source_fingerprints_json=json.dumps(inputs),
            filled_mask_definition='Upstream gap-filled contributors at valid outputs; grid interpolation tracked separately',
            temporal_policy='Original source timestamps preserved; no temporal interpolation')
        provenance = dst.create_group('provenance/source_attributes')
        for key, value in src.attrs.items():
            provenance.attrs[key] = value
        for key in 'txyz':
            original = src if key == 't' else ref
            original.copy(key, dst)
        dst['grid_interpolated_z'] = left != right
        dst['source_z_left_index'] = left
        dst['source_z_right_index'] = right
        dst['source_z_right_weight'] = weight
        chunks = (min(50, shape[0]), 1, shape[2], shape[3])
        for key in 'uvw':
            variable = dst.create_dataset(key, shape=shape, dtype='float64', chunks=chunks,
                                          compression='gzip', compression_opts=1, shuffle=True, fillvalue=np.nan)
            for attr, value in src[key].attrs.items():
                if attr not in ('_FillValue', 'missing_value', 'DIMENSION_LIST', 'REFERENCE_LIST'):
                    variable.attrs[attr] = value
            variable.attrs['_FillValue'] = np.nan
        filled = dst.create_dataset('filled_mask', shape=shape, dtype='bool', chunks=chunks,
                                    compression='gzip', compression_opts=1)
        count = np.zeros(shape[1:], dtype=np.int64)
        step = src['u'].chunks[0] if src['u'].chunks else min(64, shape[0])
        step = min(step, max(1, 12_000_000//int(np.prod(src['u'].shape[1:]))))
        w = weight[None, :, None, None]
        for start in range(0, shape[0], step):
            stop = min(start+step, shape[0])
            valid = np.ones((stop-start, *shape[1:]), dtype=bool)
            aligned = []
            for key in 'uvw':
                native = src[key][start:stop]
                a, b = native[:, left], native[:, right]
                good = np.isfinite(a) & np.isfinite(b)
                for attr in ('_FillValue', 'missing_value'):
                    for value in np.asarray(src[key].attrs.get(attr, [])).ravel():
                        good &= (a != value) & (b != value)
                values = np.where(w == 0, a, (1-w)*a + w*b)
                valid &= good & np.isfinite(values)
                aligned.append(values)
                del native, a, b, good
            for key, values in zip('uvw', aligned):
                values[~valid] = np.nan
                dst[key][start:stop] = values
            del aligned, values
            if 'filled_mask' in src:
                native_filled = src['filled_mask'][start:stop]
                filled[start:stop] = valid & (native_filled[:, left] | native_filled[:, right])
                del native_filled
            else:
                filled[start:stop] = False
            count += valid.sum(axis=0)
            valid_total += int(valid.sum())
            print(f'{source.parent.name}: aligned {stop}/{shape[0]} snapshots', flush=True)
        dst['valid_fraction'] = count/shape[0]
        dst.attrs['valid_vector_count'] = valid_total
        dst.attrs['remaining_missing_vector_count'] = int(np.prod(shape))-valid_total
        if [fingerprint(source), fingerprint(reference)] != inputs:
            raise ValueError('An input changed during interpolation')
    partial.rename(output)
    return dict(source=str(source), reference=str(reference), output=str(output), shape=list(shape),
                interpolated_z_planes=int(np.count_nonzero(left != right)), valid_vectors=valid_total,
                source_fingerprints=inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True, help='New root; preserves source case directory names')
    args = parser.parse_args(argv)
    if args.output_root.exists():
        parser.error('Choose a new output root')
    names = [p.parent.name for p in args.sources]
    if len(names) != len(set(names)):
        parser.error('Source case names must be unique')
    rows = []
    for source in args.sources:
        rows.append(align_file(source, args.reference, args.output_root/source.parent.name/'interpolated_velocity.nc'))
        (args.output_root/'alignment_manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'Saved {len(rows)} aligned velocity files to {args.output_root}. No POD run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
