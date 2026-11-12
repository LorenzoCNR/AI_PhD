"""Plot completed descriptive Soft Fig. 3 fits, or watch their training process.

No encoder fitting, inference, dimensionality reduction or decoder estimation.
Use --check for non-mutating checks; --watch requires an exact training PID.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil

from .data import sha256
from .monkey_fig3_descriptive_matched import ARCHITECTURES, CONDITIONS, ROOT, OUT
from cebra_replica.monkey.figure3_style import DIRECTION_COLORS, POSITION_CMAP, POSITION_LIMITS

FIGURES = ROOT / 'results/figura_3_2026-10-28'
ACTIVE = ROOT / 'results/user_soft_monkey_figure3b_direction_4d_v1'
PANEL_LABELS = {'b': 'Direction and Time', 'c': 'Position X,Y', 'd': 'Time',
                'e': 'Direction', 'f': 'Direction + Active/Passive', 'g': 'Position X,Y'}
FILE_LABELS = {'b': 'Direction_Time', 'c': 'Position_XY', 'd': 'Time',
               'e': 'Direction', 'f': 'Direction_Active_Passive', 'g': 'Position_XY'}


def training_process(pid, created=None):
    try:
        p = psutil.Process(pid)
        cmd = p.cmdline()
        if (not p.is_running() or p.status() == psutil.STATUS_ZOMBIE or
            '-m' not in cmd or 'user_soft.monkey_fig3_descriptive_matched' not in cmd or
            '--execute' not in cmd or (created is not None and p.create_time() != created)):
            return None
        return p
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def read_cell(folder, name, architecture, active=False):
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        return None
    record = json.loads(manifest.read_text(encoding='utf8'))
    cfg = record['config']
    if (record['actual_updates'] != 10000 or cfg['seed'] != 42 or
        cfg['output_dimension'] != (4 if active else CONDITIONS[name][2]) or
        cfg['condition'] != ('direction' if active else name) or
        record['architecture'] != architecture):
        raise ValueError(f'Unexpected fit manifest: {manifest}')
    for filename, field in (('embedding.npz', 'embedding_sha256'), ('checkpoint.pt', 'checkpoint_sha256'),
                            ('training_history.csv', 'history_sha256')):
        path = folder / filename
        if not path.is_file() or sha256(path) != record[field]:
            raise ValueError(f'Missing or mismatched artifact: {path}')
    return dict(manifest=manifest, record=record, folder=folder)


def cells():
    found, missing = {}, []
    for name in CONDITIONS:
        for architecture in ARCHITECTURES:
            item = read_cell(OUT / name / architecture / 'seed_42', name, architecture)
            if item is None:
                missing.append(f'{name}/{architecture}')
            else:
                found[(name, architecture)] = item
    for architecture in ARCHITECTURES:
        item = read_cell(ACTIVE / architecture / 'seed_42', None, architecture, active=True)
        if item is None:
            missing.append(f'existing_active_direction/{architecture}')
        else:
            found[('active_direction', architecture)] = item
    return found, missing


def load(item, active=False):
    path = item['folder'] / 'embedding.npz'
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    z = data['embedding']; n = len(z)
    if n % 600 or z.shape != (n, item['record']['config']['output_dimension']) or not np.isfinite(z).all():
        raise ValueError(f'Invalid embedding {path}')
    if (not np.array_equal(data['source_id'], np.arange(n)) or
        not np.array_equal(data['trial_id'], np.repeat(np.arange(n//600), 600)) or
        not np.array_equal(data['time_id'], np.tile(np.arange(600), n//600))):
        raise ValueError(f'Invalid trial/time alignment {path}')
    for key in (('direction',) if active else ('direction', 'passive', 'position_xy')):
        if len(data[key]) != n or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key} {path}')
    return data


def trajectories(data, passive=None, mode=None):
    z = data['embedding'].reshape(-1, 600, data['embedding'].shape[1])
    direction = data['direction'].reshape(-1, 600)
    if not np.all(direction == direction[:, :1]):
        raise ValueError('Direction changes inside trial')
    selector = np.ones(len(z), dtype=bool)
    if passive is not None:
        label = data['passive'].reshape(-1, 600)
        if not np.all(label == label[:, :1]):
            raise ValueError('Active/passive changes inside trial')
        selector = label[:, 0] == passive
    result = []
    for d in range(8):
        selected = z[selector & (direction[:, 0] == d)]
        if not len(selected):
            raise ValueError(f'Missing direction {d} passive={passive}')
        mean = selected.mean(axis=0)
        if mode == 'global':
            mean = mean / max(np.linalg.norm(mean), 1e-12)
        elif mode == 'row':
            mean = mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        result.append(mean)
    return result


def clean(ax):
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_pane_color((1, 1, 1, 0)); dim.set_ticks([])
        dim._axinfo['grid']['color'] = (1, 1, 1, 0)


def panel_figure(panel, arch, data):
    if panel == 'b':
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, key, coords in zip(axes, ('active_direction', 'b_time_active_4d'), ((2, 0), (0, 1))):
            for direction, curve in enumerate(trajectories(data[key])):
                ax.plot(curve[:, coords[0]], curve[:, coords[1]], color=DIRECTION_COLORS[direction], lw=2)
            ax.set_title('Direction' if key == 'active_direction' else 'Time')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
    elif panel in 'cd':
        key = 'c_position_active_3d' if panel == 'c' else 'd_time_active_3d'
        item = data[key]; fig = plt.figure(figsize=(12, 6))
        for i, label in enumerate(('x', 'y')):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            z = item['embedding']
            points = ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=item['position_xy'][:, i],
                cmap=POSITION_CMAP, vmin=POSITION_LIMITS[0], vmax=POSITION_LIMITS[1],
                s=.05, linewidths=0, rasterized=True)
            clean(ax); ax.set_title(f'{label} position')
            if i == 1: fig.colorbar(points, ax=ax, fraction=.03, pad=.05, label='cm')
    else:
        fig = plt.figure(figsize=(14, 7))
        key = {'f':'f_direction_ap_all_4d', 'g':'g_position_all_4d'}.get(panel)
        coords = {'e': (1, 2, 0), 'f': (2, 0, 1), 'g': (0, 1, 2)}[panel]
        mode = 'global' if panel == 'e' else 'row'
        for i, passive in enumerate((False, True)):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            selected = data[('e_direction_passive_4d' if passive else 'active_direction') if panel == 'e' else key]
            # Separate active/passive sources in e; combined sources in f and g.
            selector = None if panel == 'e' else passive
            for direction, curve in enumerate(trajectories(selected, selector, mode)):
                x, y, z = (curve[:, j] for j in coords); color = DIRECTION_COLORS[direction]
                ax.plot(x, y, z, color=color); ax.plot(x[0], y[0], z[0], color=color, marker='o')
                ax.plot(x[-1], y[-1], z[-1], color=color, marker='<')
            clean(ax); ax.set_title('Passive' if passive else 'Active')
    fig.suptitle(f'Figure 3{panel.upper()} Soft — {PANEL_LABELS[panel]} — {arch.upper()} — seed 42')
    return fig


def render(found):
    # Load and validate all parents before writing any panel.
    cache = {arch: {} for arch in ARCHITECTURES}
    for (name, arch), item in found.items():
        cache[arch][name] = load(item, active=(name == 'active_direction'))
    parents = {f'{name}/{arch}': {'manifest': str(item['manifest']),
               'manifest_sha256': sha256(item['manifest']),
               'embedding_sha256': item['record']['embedding_sha256']}
               for (name, arch), item in found.items()}
    destinations = [FIGURES / panel for panel in 'bcdefg']
    targets = []
    for panel, destination in zip('bcdefg', destinations):
        for arch in ARCHITECTURES:
            for extension in ('png', 'pdf'):
                targets.append(destination / f'Figure_3{panel}_Soft_{FILE_LABELS[panel]}_{arch.upper()}_seed42.{extension}')
        targets.append(destination / f'Figure_3{panel}_Soft_descriptive_seed42_provenance.json')
    if any(p.exists() for p in targets):
        raise FileExistsError('A target figure or provenance already exists; refusing overwrite or partial replacement')
    for panel, destination in zip('bcdefg', destinations):
        destination.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for arch in ARCHITECTURES:
            fig = panel_figure(panel, arch, cache[arch])
            for extension in ('png', 'pdf'):
                path = destination / f'Figure_3{panel}_Soft_{FILE_LABELS[panel]}_{arch.upper()}_seed42.{extension}'
                fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                outputs[path.name] = sha256(path)
            plt.close(fig)
        record = dict(panel=panel, seed=42, parents=parents, figure_sha256=outputs,
            renderer_sha256=sha256(Path(__file__)), support='all full-recording rows, no valid-mask filter',
            colors='CEBRA hsv(direction/8); position seismic [-15,15] cm',
            coordinates_zero_based={'b': [[2,0],[0,1]], 'c':[0,1,2], 'd':[0,1,2],
                                    'e':[1,2,0], 'f':[2,0,1], 'g':[0,1,2]}[panel],
            aggregation='per-direction trial means; e global and f/g per-row normalization as CEBRA plotting',
            no_new_fit_or_inference=True, held_out=False,
            note='g Soft and corrected CEBRA target position XY only; the earlier uncorrected CEBRA fit also used active/passive. Legacy input windows cross trial boundaries.')
        (destination / f'Figure_3{panel}_Soft_descriptive_seed42_provenance.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--watch', type=int, metavar='TRAINING_PID')
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    if args.interval <= 0: parser.error('interval must be positive')
    found, missing = cells()
    if args.check:
        print(json.dumps(dict(complete=len(found), missing=missing), indent=2)); return
    proc = training_process(args.watch)
    if proc is None:
        raise RuntimeError(f'Training PID {args.watch} is not a live matching process')
    created = proc.create_time()
    status_path = OUT / f'plot_watcher_{os.getpid()}.json'
    def status(state, **extra):
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(state=state, watcher_pid=os.getpid(), training_pid=args.watch,
            training_created=created, updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2)+'\n', encoding='utf8')
        temp.replace(status_path)
    try:
        while True:
            found, missing = cells()
            if not missing:
                status('rendering', complete=len(found))
                render(found)
                status('complete', complete=len(found)); return
            if training_process(args.watch, created) is None:
                status('training_stopped_before_completion', missing=missing, complete=len(found))
                print('Training stopped before completion:', missing, file=sys.stderr); return
            status('waiting', missing=missing, complete=len(found))
            time.sleep(args.interval)
    except BaseException:
        status('error', diagnostic=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
