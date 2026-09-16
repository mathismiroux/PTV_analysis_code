"""Prepare reviewable common masks, then explicitly run POD from a saved plan."""
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np

from ptv_flow.pod import run_pod, cell_widths
from ptv_flow.pod_support import fingerprint, scan_valid_counts
from ptv_flow.reader import FlowDataset


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(flow_folder, cases_folder, pattern):
    groups, memberships = {}, [set(), set()]
    for root_index, folder in enumerate((flow_folder, cases_folder)):
        folder = Path(folder)
        if not folder.is_dir():
            raise ValueError(f'Input folder does not exist: {folder}')
        for source in sorted(folder.glob(pattern)):
            if not source.is_file():
                continue
            match = re.fullmatch(r'(.+)_(\d+(?:\.\d+)?)D', source.parent.name)
            if not match:
                raise ValueError(f'Cannot identify case and downstream volume: {source}')
            case, distance = match[1], float(match[2])
            group = groups.setdefault(distance, {})
            if case in group:
                raise ValueError(f'Duplicate {case} at {distance:g}D')
            group[case] = source.resolve()
            memberships[root_index].add(case)
    if not all(memberships):
        raise ValueError('Both input roots must contain matching velocity files')
    if memberships[0] & memberships[1]:
        raise ValueError('Case labels must be distinct between flow and case roots')
    expected = memberships[0] | memberships[1]
    for distance, group in groups.items():
        missing = expected - set(group)
        if missing:
            raise ValueError(f'{distance:g}D is missing cases {sorted(missing)}; no cases silently omitted')
    return groups


def prepare_group(distance, group, output, settings, output_subfolder, reference):
    if reference not in group:
        raise ValueError(f'Missing grid reference {reference} at {distance:g}D')
    sources, coordinates, shapes = {}, {}, {}
    units = {}
    for case, source in group.items():
        sources[case] = fingerprint(source)
        with FlowDataset(source) as flow:
            shapes[case] = flow.shape
            coordinates[case] = {key: flow.coordinate(key).astype(float) for key in 'zyx'}
        if shapes[case][0] < 2:
            raise ValueError(f'{source}: require at least two snapshots')
        with h5py.File(source, 'r') as h:
            for key in 'zyxuvw':
                value = h[key].attrs.get('units', '')
                if isinstance(value, bytes):
                    value = value.decode()
                if value:
                    if key in units and units[key] != str(value):
                        raise ValueError(f'{source}: inconsistent units for {key}')
                    units[key] = str(value)
    target = {}
    for key in 'zyx':
        lower = max(c[key].min() for c in coordinates.values())
        upper = min(c[key].max() for c in coordinates.values())
        axis = coordinates[reference][key]
        target[key] = axis[(axis >= lower) & (axis <= upper)]
        if not len(target[key]):
            raise ValueError(f'{distance:g}D: no overlapping {key} coordinates')
        cell_widths(target[key])
    shape = tuple(len(target[key]) for key in 'zyx')
    minimum = np.ones(shape)
    common = np.ones(shape, dtype=bool)
    rows = []
    for case, source in group.items():
        print(f'{distance:g}D {case}: scanning {shapes[case][0]} snapshots on {shape}', flush=True)
        nt = shapes[case][0]
        with h5py.File(source, 'r') as src:
            count = scan_valid_counts(src, target, settings['exclude_filled'], settings['zero_invalid'])
        fraction = count/nt
        own = (count >= np.ceil(settings['min_valid_fraction']*nt)) & (count >= 2)
        if fingerprint(source) != sources[case]:
            raise ValueError(f'Source changed during preview: {source}')
        common &= own
        minimum = np.minimum(minimum, fraction)
        axes = [key for key in 'zyx' if not np.all(np.isin(target[key], coordinates[case][key]))]
        rows.append(dict(case=case, source=str(source), snapshots=nt,
                         native_shape=list(shapes[case][1:]), interpolation_axes=axes,
                         own_eligible_voxels=int(own.sum()), output=str(source.parent/output_subfolder)))
    weights = (cell_widths(target['z'])[:, None, None] * cell_widths(target['y'])[None, :, None]
               * cell_widths(target['x'])[None, None, :])
    metadata = dict(distance_d=distance, reference_case=reference, sources=list(sources.values()),
                    **{key:settings[key] for key in ('min_valid_fraction', 'exclude_filled', 'zero_invalid')},
                    alignment='linear; every native contributing vector must be valid; no extrapolation',
                    validity='finite uvw excluding declared fill values; interpolated values included unless exclude_filled',
                    units=units)
    name = f'volume_{distance:g}D'
    mask_file = output/f'{name}_mask.npz'
    np.savez_compressed(mask_file, **target, support=common, minimum_valid_fraction=minimum,
                        metadata_json=json.dumps(metadata))
    plot_mask(output/f'{name}_mask.png', target, common, minimum, rows, distance)
    result = dict(distance_d=distance, mask_file=str(mask_file.resolve()), mask_sha256=digest(mask_file),
                  grid_shape=list(shape), common_voxels=int(common.sum()), target_voxels=int(common.size),
                  common_grid_fraction=float(common.mean()),
                  common_volume_fraction=float(weights[common].sum()/weights.sum()),
                  retained_volume_coordinate_units_cubed=float(weights[common].sum()),
                  sources=list(sources.values()), cases=rows,
                  status='ready' if common.any() else 'blocked_empty_mask')
    print(f'{distance:g}D common mask: {common.sum()}/{common.size} voxels ({common.mean():.1%})', flush=True)
    return result


def plot_mask(path, coords, support, minimum, rows, distance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    iz = int(np.argmin(np.abs(coords['z'])))
    for ax, values, title in [(axes[0, 0], minimum[iz], 'Lowest temporal availability across cases'),
                               (axes[0, 1], support[iz], 'Retained common mask')]:
        mesh = ax.pcolormesh(coords['x'], coords['y'], values, shading='nearest', vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax)
        ax.set(title=f'{title}; z={coords["z"][iz]:g}', xlabel='x (source units)', ylabel='y (source units)')
    axes[1, 0].plot(coords['x'], 100*support.mean(axis=(0, 1)))
    axes[1, 0].set(xlabel='x (source units)', ylabel='Retained y-z grid (%)', ylim=(0, 100))
    labels = [r['case'] for r in rows] + ['Common']
    values = [r['own_eligible_voxels'] for r in rows] + [int(support.sum())]
    axes[1, 1].bar(labels, 100*np.array(values)/support.size)
    axes[1, 1].tick_params(axis='x', labelrotation=30)
    axes[1, 1].set(ylabel='Eligible common-grid voxels (%)', ylim=(0, 100))
    fig.suptitle(f'{distance:g}D: common-mask preview — no POD calculated')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_review_html(plan, output):
    """Standalone browser viewer of every saved mask slice; does not run POD."""
    volumes = []
    for group in plan['groups']:
        with np.load(group['mask_file'], allow_pickle=False) as h:
            volumes.append(dict(distance=group['distance_d'], shape=group['grid_shape'],
                coords={k:h[k].tolist() for k in 'zyx'}, coverage=group['common_grid_fraction'],
                cases=[c['case'] for c in group['cases']],
                mask=base64.b64encode(h['support'].astype('uint8').tobytes()).decode(),
                availability=base64.b64encode(np.rint(h['minimum_valid_fraction']*255).astype('uint8').tobytes()).decode()))
    payload = json.dumps(dict(volumes=volumes, settings=plan['settings'])).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Common-mask POD review</title>
<style>body{font:16px system-ui;margin:28px auto;padding:0 22px;max-width:1200px;color:#17232d;background:#f6f8fa}
h1{font-size:26px}p{line-height:1.5}select,input{font:inherit;margin:8px;padding:5px}input{width:300px}
.panels{display:flex;gap:24px;flex-wrap:wrap}.panel{flex:1;min-width:320px;background:white;padding:18px;border:1px solid #d8e0e5;border-radius:8px}
canvas{width:100%;image-rendering:pixelated}#position{font-weight:600}.note{color:#4c5963}label{white-space:nowrap}</style>
<h1>Review the common spatial masks</h1>
<p id="settings"></p><p>No POD has been calculated by this preview. Use the controls to inspect all slices before running the saved plan.</p>
<label>Volume <select id="volume"></select></label><label>Plane <select id="plane"><option>XY</option><option>XZ</option><option>YZ</option></select></label>
<label>Slice <input id="slice" type="range" min="0" step="1"></label><span id="position"></span>
<p id="coverage"></p><div class="panels"><div class="panel"><h2>Minimum availability</h2>
<p class="note">Lowest temporal availability across cases. Purple = 0%; yellow = 100%.</p><canvas id="availability"></canvas></div>
<div class="panel"><h2>Saved common mask</h2><p class="note">Green = retained; pale grey = excluded.</p><canvas id="mask"></canvas></div></div>
<p id="axes"></p><p id="members"></p><p class="note">Coordinates use source units. Availability colours are rounded to 1/255 for display; the saved binary mask is exact.
Coverage refers to the shared target grid. Finite interpolated values count as available by default.
Remaining temporal gaps will be mean-filled during POD. This is not gappy-POD reconstruction.</p>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), byId=id=>document.getElementById(id);
for(const v of data.volumes){v.mask=Uint8Array.from(atob(v.mask),c=>c.charCodeAt(0));v.availability=Uint8Array.from(atob(v.availability),c=>c.charCodeAt(0));
 const o=document.createElement('option');o.textContent=v.distance+'D';byId('volume').append(o);}
byId('settings').textContent=`Retain voxels with at least ${100*data.settings.min_valid_fraction}% temporal availability in every case. Planned modes: ${data.settings.modes}.`;
const dimensions={XY:['z','x','y'],XZ:['y','x','z'],YZ:['x','y','z']};
const palette=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
function draw(reset=false){const v=data.volumes[byId('volume').selectedIndex], [fixed,horizontal,vertical]=dimensions[byId('plane').value];
 const s=byId('slice');s.max=v.coords[fixed].length-1;if(reset)s.value=Math.floor(s.max/2);const index=Number(s.value);
 byId('position').textContent=fixed+' = '+v.coords[fixed][index].toFixed(3);
 byId('coverage').textContent=`${(100*v.coverage).toFixed(1)}% of the shared 3D grid retained (${v.shape.join(' × ')} voxels, z × y × x).`;
 byId('members').textContent='Intersection of: '+v.cases.join(', ');
 const axisText=k=>`${k}: ${v.coords[k][0].toFixed(2)} to ${v.coords[k].at(-1).toFixed(2)}`;
 byId('axes').textContent=`Horizontal ${axisText(horizontal)}; vertical ${axisText(vertical)} (increases upward).`;
 const width=v.coords[horizontal].length,height=v.coords[vertical].length;
 for(const kind of ['mask','availability']){const canvas=byId(kind);canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(width,height);
  for(let j=0;j<height;j++)for(let i=0;i<width;i++){const q={};q[fixed]=index;q[horizontal]=i;q[vertical]=height-1-j;
   const n=(q.z*v.shape[1]+q.y)*v.shape[2]+q.x,value=v[kind][n];let rgb;
   if(kind==='mask')rgb=value?[20,145,110]:[231,237,240];else{const a=value/255*4,k=Math.min(3,Math.floor(a)),f=a-k;rgb=palette[k].map((c,t)=>Math.round(c*(1-f)+palette[k+1][t]*f));}
   pixels.data.set([...rgb,255],4*(j*width+i));}ctx.putImageData(pixels,0,0);}}
byId('volume').onchange=()=>draw(true);byId('plane').onchange=()=>draw(true);byId('slice').oninput=()=>draw();draw(true);
</script></html>'''.replace('PAYLOAD', payload)
    Path(output).write_text(html, encoding='utf-8')


def prepare(args):
    if args.output.exists():
        raise ValueError('Choose a new preview output directory')
    if args.output_subfolder is None:
        threshold_label = f'{100*args.min_valid_fraction:g}'.replace('.', 'p')
        args.output_subfolder = f'pod_common_valid{threshold_label}_modes{args.modes}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.output_subfolder):
        raise ValueError('Output subfolder must be a simple folder name')
    settings = {key:getattr(args, key) for key in ('modes', 'min_valid_fraction', 'exclude_filled',
                'zero_invalid', 'iterations', 'oversampling', 'seed')}
    if (not 0 < args.min_valid_fraction <= 1 or args.modes < 1 or args.iterations < 0
            or args.oversampling < 0 or args.seed < 0):
        raise ValueError('Invalid POD settings')
    groups = discover(args.flow_folder, args.cases_folder, args.pattern)
    for group in groups.values():
        for source in group.values():
            if (source.parent/args.output_subfolder).exists():
                raise ValueError(f'POD output exists: {source.parent/args.output_subfolder}')
    args.output.mkdir(parents=True)
    plan = dict(version=1, status='preparing', settings=settings, groups=[],
                flow_folder=str(args.flow_folder.resolve()), cases_folder=str(args.cases_folder.resolve()),
                pattern=args.pattern, output_subfolder=args.output_subfolder,
                note='Preparation scans velocity and writes masks only. Run is a separate explicit command. '
                     'POD recomputes each case mean; remaining gaps are mean-filled, not gappy-POD reconstructed.')
    path = args.output/'plan.json'
    path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    for distance, group in sorted(groups.items()):
        plan['groups'].append(prepare_group(distance, group, args.output, settings,
                                            args.output_subfolder, args.reference_case))
        path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    plan['status'] = 'ready_for_review' if all(g['status']=='ready' for g in plan['groups']) else 'blocked'
    path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    fields = ['distance_d', 'common_voxels', 'target_voxels', 'common_grid_fraction', 'common_volume_fraction', 'status']
    with (args.output/'coverage.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(plan['groups'])
    lines = ['# Common-mask POD preview', '', plan['note'], '',
             f'Temporal availability threshold: {args.min_valid_fraction:.0%}; modes: {args.modes}.', '',
             '| x/D | Cases | Common voxels | Target grid | Coverage |', '|---:|---|---:|---:|---:|']
    for g in plan['groups']:
        lines.append(f'| {g["distance_d"]:g} | {", ".join(r["case"] for r in g["cases"])} | '
                     f'{g["common_voxels"]} | {g["target_voxels"]} | {g["common_grid_fraction"]:.1%} |')
    lines += ['', 'Review each volume mask PNG and the exact source/output paths in plan.json.',
              'Grid: Flow coordinates cropped to common bounds. Other grids are linearly aligned where needed.',
              'Coverage denominators refer to the common target grid, not the full original grids.',
              'No temporal synchronization between independent recordings is assumed.',
              'After review, execute the saved settings and masks with:', '', '```powershell',
              f'python scripts/pod_common_mask.py run "{path.resolve()}"', '```']
    (args.output/'review.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_review_html(plan, args.output/'review.html')
    print(f'Preview {plan["status"]}: {path.resolve()}. No POD run.', flush=True)
    return plan


def run_plan(path, volumes=None):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan.get('version') != 1 or plan.get('status') != 'ready_for_review' or not plan.get('groups'):
        raise ValueError('Plan must be a complete, nonempty preview with nonempty masks')
    discovered = discover(plan['flow_folder'], plan['cases_folder'], plan['pattern'])
    planned_sources = {r['source'] for g in plan['groups'] for r in g['cases']}
    if planned_sources != {str(p) for g in discovered.values() for p in g.values()}:
        raise ValueError('Discovered cases changed since preview; prepare again')
    groups = plan['groups']
    log_name = 'run_manifest.json'
    if volumes is not None:
        selected = set(volumes)
        available = {g['distance_d'] for g in groups}
        if not selected or not selected <= available:
            raise ValueError(f'Choose volumes present in the plan: {sorted(available)}')
        groups = [g for g in groups if g['distance_d'] in selected]
        label = '_'.join(f'{v:g}D' for v in sorted(selected))
        log_name = f'run_manifest_{label}.json'
    log_path = path.with_name(log_name)
    if log_path.exists():
        raise ValueError('Run manifest already exists; outputs will not be overwritten')
    for group in groups:
        if digest(group['mask_file']) != group['mask_sha256']:
            raise ValueError('Prepared mask changed; prepare again')
        for source in group['sources']:
            if fingerprint(source['path']) != source:
                raise ValueError(f'Source changed since preview: {source["path"]}')
        for case in group['cases']:
            if Path(case['output']).exists():
                raise ValueError(f'Output exists: {case["output"]}')
    logs = []
    for group in groups:
        for case in group['cases']:
            row = dict(source=case['source'], output=case['output'], mask=group['mask_file'], status='running')
            logs.append(row)
            log_path.write_text(json.dumps(logs, indent=2), encoding='utf-8')
            try:
                row['summary'] = run_pod(case['source'], case['output'], spatial_mask=group['mask_file'], **plan['settings'])
                row['status'] = 'complete'
            except Exception as exc:
                row.update(status='failed', reason=str(exc))
                raise
            finally:
                log_path.write_text(json.dumps(logs, indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preview = commands.add_parser('prepare', help='Scan validity and write masks/review; never runs POD')
    preview.add_argument('--flow-folder', type=Path, required=True)
    preview.add_argument('--cases-folder', type=Path, required=True)
    preview.add_argument('--pattern', default='*/interpolated_velocity.nc')
    preview.add_argument('--output', type=Path, required=True, help='New preview directory')
    preview.add_argument('--output-subfolder', help='New folder beside each source; default name follows threshold and mode count')
    preview.add_argument('--reference-case', default='Flow')
    preview.add_argument('--min-valid-fraction', type=float, default=.80)
    preview.add_argument('--modes', type=int, default=100)
    preview.add_argument('--iterations', type=int, default=4)
    preview.add_argument('--oversampling', type=int, default=15)
    preview.add_argument('--seed', type=int, default=0)
    preview.add_argument('--exclude-filled', action='store_true')
    preview.add_argument('--zero-invalid', action='store_true')
    run = commands.add_parser('run', help='Explicitly execute a previously reviewed plan')
    run.add_argument('plan', type=Path)
    run.add_argument('--volumes', type=float, nargs='+', help='Only these downstream x/D locations, e.g. --volumes 1')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(args)
        else:
            run_plan(args.plan, args.volumes)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
