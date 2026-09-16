import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ptv_flow.pod_support import axis_map, read_aligned_plane, scan_valid_counts
from scripts import pod_common_mask as cli


def make_source(root, case, z=(0., 1.)):
    path = root/f'{case}_1D'/'interpolated_velocity.nc'
    path.parent.mkdir(parents=True)
    t = np.arange(8, dtype=float)
    with h5py.File(path, 'w') as h:
        for key, values in dict(t=t, z=z, y=[0., 1.], x=[0., 1.]).items():
            h[key] = values
        shape = (len(t), len(z), 2, 2)
        for ic, key in enumerate('uvw'):
            values = np.broadcast_to(np.asarray(z)[None, :, None, None] +
                       (ic+1)*np.sin(2*np.pi*t/8)[:, None, None, None], shape).copy()
            if case == 'Flow':
                values[0, 0, 0, 0] = np.nan
            if case == 'Static':
                values[:3, 1, 1, 1] = np.nan
            if case == 'Pitch':
                values[:, 0, 0, 1] = np.nan
            h[key] = values
    return path


def prepare_fixture(tmp_path, monkeypatch):
    flow = make_source(tmp_path/'flow', 'Flow')
    static = make_source(tmp_path/'cases', 'Static')
    pitch = make_source(tmp_path/'cases', 'Pitch', (-.5, .5, 1.5))
    monkeypatch.setattr(cli, 'plot_mask', lambda *args: None)
    output = tmp_path/'preview'
    assert cli.main(['prepare', '--flow-folder', str(tmp_path/'flow'), '--cases-folder', str(tmp_path/'cases'),
                     '--output', str(output), '--output-subfolder', 'pod_common', '--modes', '2']) == 0
    return output, [flow, static, pitch]


def test_preview_then_explicit_run_on_identical_support(tmp_path, monkeypatch):
    output, sources = prepare_fixture(tmp_path, monkeypatch)
    plan = json.loads((output/'plan.json').read_text())
    assert plan['status'] == 'ready_for_review'
    group = plan['groups'][0]
    assert group['common_voxels'] == 6
    assert not any((p.parent/'pod_common').exists() for p in sources)
    pitch = next(c for c in group['cases'] if c['case'] == 'Pitch')
    assert pitch['interpolation_axes'] == ['z']
    before = [p.read_bytes() for p in sources]
    monkeypatch.setattr('ptv_flow.pod.plot_pod', lambda *args: None)
    assert cli.main(['run', str(output/'plan.json')]) == 0
    masks, weights, coords = [], [], []
    for source in sources:
        with h5py.File(source.parent/'pod_common'/'pod.h5') as h:
            masks.append(h['support'][:])
            weights.append(h['spatial_weight'][:])
            coords.append(h['z'][:])
            metadata = json.loads(h.attrs['metadata_json'])
            assert metadata['mean_file'] is None
            assert metadata['retained_voxels'] == 6
            assert metadata['common_mask_metadata']['min_valid_fraction'] == .8
    for i in range(1, 3):
        np.testing.assert_array_equal(masks[0], masks[i])
        np.testing.assert_array_equal(weights[0], weights[i])
        np.testing.assert_array_equal(coords[0], coords[i])
    assert before == [p.read_bytes() for p in sources]
    with pytest.raises(SystemExit):
        cli.main(['run', str(output/'plan.json')])


def test_stale_source_rejected_before_any_run(tmp_path, monkeypatch):
    output, sources = prepare_fixture(tmp_path, monkeypatch)
    with h5py.File(sources[-1], 'a') as h:
        h.attrs['changed'] = True
    with pytest.raises(ValueError, match='Source changed'):
        cli.run_plan(output/'plan.json')
    assert not any((p.parent/'pod_common').exists() for p in sources)


def test_missing_case_rejected(tmp_path):
    make_source(tmp_path/'flow', 'Flow')
    source = make_source(tmp_path/'cases', 'Static')
    (source.parent.parent/'Static_2D').mkdir()
    (source.parent.parent/'Static_2D'/'interpolated_velocity.nc').write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match='missing cases'):
        cli.discover(tmp_path/'flow', tmp_path/'cases', '*/interpolated_velocity.nc')


def test_alignment_linear_field_and_invalid_contributors(tmp_path):
    source = make_source(tmp_path, 'Pitch', (-.5, .5, 1.5))
    coords = dict(z=np.array([0., 1.]), y=np.array([0., 1.]), x=np.array([0., 1.]))
    with h5py.File(source) as h:
        data, valid = read_aligned_plane(h, 0, coords)
        counts = scan_valid_counts(h, coords)
        for iz in range(len(coords['z'])):
            _, plane_valid = read_aligned_plane(h, iz, coords)
            np.testing.assert_array_equal(counts[iz], plane_valid.sum(axis=0))
    np.testing.assert_allclose(data[0, :, 1, 0], np.sin(2*np.pi*np.arange(8)/8), atol=1e-15)
    assert not valid[:, 0, 1].any()
    assert valid[:, 1, 0].all()
    with pytest.raises(ValueError, match='extrapolate'):
        axis_map(np.array([0., 1.]), np.array([-1.]))


def test_fill_values_zero_and_filled_flags_agree_between_scan_and_pod(tmp_path):
    source = make_source(tmp_path, 'Static')
    coords = dict(z=np.array([0., 1.]), y=np.array([0., 1.]), x=np.array([0., 1.]))
    with h5py.File(source, 'a') as h:
        h['filled_mask'] = np.zeros(h['u'].shape, dtype=np.uint8)
        h['filled_mask'][1, 0, 1, 0] = 1
        h['u'].attrs['missing_value'] = -999.
        h['u'][2, 0, 1, 0] = -999.
        for key in 'uvw':
            h[key][3, 0, 1, 0] = 0
    with h5py.File(source) as h:
        counts = scan_valid_counts(h, coords, exclude_filled=True, zero_invalid=True)
        for iz in range(2):
            _, valid = read_aligned_plane(h, iz, coords, exclude_filled=True, zero_invalid=True)
            np.testing.assert_array_equal(counts[iz], valid.sum(axis=0))


def test_changed_mask_rejected_before_outputs(tmp_path, monkeypatch):
    output, sources = prepare_fixture(tmp_path, monkeypatch)
    plan = json.loads((output/'plan.json').read_text())
    with Path(plan['groups'][0]['mask_file']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='mask changed'):
        cli.run_plan(output/'plan.json')
    assert not any((p.parent/'pod_common').exists() for p in sources)


def test_run_only_selected_volume(tmp_path, monkeypatch):
    sources = [make_source(tmp_path/'flow', 'Flow'), make_source(tmp_path/'cases', 'Static')]
    for source in sources:
        destination = source.parent.with_name(source.parent.name.replace('_1D', '_2D'))/source.name
        destination.parent.mkdir()
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(cli, 'plot_mask', lambda *args: None)
    out = tmp_path/'preview'
    cli.main(['prepare', '--flow-folder', str(tmp_path/'flow'), '--cases-folder', str(tmp_path/'cases'),
              '--output', str(out), '--output-subfolder', 'pod_common', '--modes', '2'])
    calls = []
    monkeypatch.setattr(cli, 'run_pod', lambda source, output, **kwargs: calls.append(Path(source).parent.name))
    cli.main(['run', str(out/'plan.json'), '--volumes', '1'])
    assert set(calls) == {'Flow_1D', 'Static_1D'}
    assert (out/'run_manifest_1D.json').exists()
    assert not (out/'run_manifest.json').exists()
    with pytest.raises(ValueError, match='Choose volumes'):
        cli.run_plan(out/'plan.json', [3])
