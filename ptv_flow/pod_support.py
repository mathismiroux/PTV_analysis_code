"""Shared validity and conservative grid alignment for common-support POD."""
import json
from pathlib import Path

import h5py
import numpy as np


def fingerprint(path):
    path = Path(path).resolve()
    info = path.stat()
    return dict(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns)


def axis_map(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if (not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0)):
        raise ValueError('Common-mask alignment requires finite increasing coordinates')
    if not len(target) or target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError('Common-mask alignment cannot extrapolate')
    right = np.searchsorted(source, target)
    exact = source[right] == target
    left = np.where(exact, right, right-1)
    weight = np.divide(target-source[left], source[right]-source[left],
                       out=np.zeros(len(target)), where=~exact)
    return left, right, weight


def read_native_plane(src, iz, exclude_filled=False, zero_invalid=False):
    data = np.stack([src[k][:, iz].astype(float) for k in 'uvw'])
    finite = np.all(np.isfinite(data), axis=0)
    for ic, key in enumerate('uvw'):
        for attr in ('_FillValue', 'missing_value'):
            for value in np.asarray(src[key].attrs.get(attr, [])).ravel():
                finite &= data[ic] != value
    if zero_invalid:
        finite &= np.any(data != 0, axis=0)
    if exclude_filled:
        if 'filled_mask' not in src:
            raise ValueError('--exclude-filled requires filled_mask')
        finite &= src['filled_mask'][:, iz] == 0
    return data, finite


def interpolate_axis(data, finite, mapping, axis):
    left, right, weight = mapping
    shape = [1]*data.ndim
    shape[axis] = len(weight)
    w = weight.reshape(shape)
    a, b = np.take(data, left, axis), np.take(data, right, axis)
    valid = np.take(finite, left, axis-1) & np.take(finite, right, axis-1)
    values = np.where(w == 0, a, (1-w)*a + w*b)
    return values, valid & np.all(np.isfinite(values), axis=0)


def read_aligned_plane(src, iz, coords, exclude_filled=False, zero_invalid=False):
    """Linear interpolation; all contributing native velocity vectors must be valid."""
    maps = {key: axis_map(src[key][:], coords[key]) for key in 'zyx'}
    lo, hi, w = (v[iz] for v in maps['z'])
    data, finite = read_native_plane(src, lo, exclude_filled, zero_invalid)
    if hi != lo:
        other, other_finite = read_native_plane(src, hi, exclude_filled, zero_invalid)
        data = (1-w)*data + w*other
        finite &= other_finite
    for key, axis in [('y', 2), ('x', 3)]:
        if not np.array_equal(src[key][:], coords[key]):
            data, finite = interpolate_axis(data, finite, maps[key], axis)
    return data, finite & np.all(np.isfinite(data), axis=0)


def scan_valid_counts(src, coords, exclude_filled=False, zero_invalid=False):
    """Scan each time chunk once, without storing or reconstructing velocity fields."""
    maps = {key:axis_map(src[key][:], coords[key]) for key in 'zyx'}
    shape = tuple(len(coords[key]) for key in 'zyx')
    counts = np.zeros(shape, dtype=np.int64)
    nt = src['u'].shape[0]
    step = src['u'].chunks[0] if src['u'].chunks else min(nt, 64)
    # Bound memory even for files chunked across the entire time dimension.
    step = min(step, max(1, 20_000_000//int(np.prod(src['u'].shape[1:]))))
    if exclude_filled and 'filled_mask' not in src:
        raise ValueError('--exclude-filled requires filled_mask')
    for start in range(0, nt, step):
        stop = min(start+step, nt)
        finite = np.ones((stop-start, *src['u'].shape[1:]), dtype=bool)
        nonzero = np.zeros_like(finite) if zero_invalid else None
        for key in 'uvw':
            values = src[key][start:stop]
            finite &= np.isfinite(values)
            for attr in ('_FillValue', 'missing_value'):
                for value in np.asarray(src[key].attrs.get(attr, [])).ravel():
                    finite &= values != value
            if zero_invalid:
                nonzero |= values != 0
            del values
        if zero_invalid:
            finite &= nonzero
        if exclude_filled:
            finite &= src['filled_mask'][start:stop] == 0
        for axis, key in enumerate('zyx', start=1):
            left, right, _ = maps[key]
            finite = np.take(finite, left, axis) & np.take(finite, right, axis)
        counts += finite.sum(axis=0)
        print(f'  scanned snapshots {stop}/{nt}', flush=True)
    return counts


def load_spatial_mask(path, source, min_valid_fraction, exclude_filled, zero_invalid):
    with np.load(path, allow_pickle=False) as saved:
        coords = {key: saved[key].copy() for key in 'zyx'}
        support = saved['support'].copy()
        metadata = json.loads(str(saved['metadata_json']))
    if support.dtype != np.bool_ or support.shape != tuple(len(coords[k]) for k in 'zyx') or not support.any():
        raise ValueError('Invalid or empty common spatial mask')
    for key, value in dict(min_valid_fraction=min_valid_fraction, exclude_filled=exclude_filled,
                           zero_invalid=zero_invalid).items():
        if metadata[key] != value:
            raise ValueError(f'Common-mask {key} differs from POD settings')
    recorded = {item['path']:item for item in metadata['sources']}
    current = fingerprint(source)
    if recorded.get(current['path']) != current:
        raise ValueError('Source is absent from common-mask preview or changed since preparation')
    return coords, support, metadata
