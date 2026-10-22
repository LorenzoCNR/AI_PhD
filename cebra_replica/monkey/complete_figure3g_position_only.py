"""Isolated Figure3g protocol correction; dry-run by default, never overwrite old runs."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import joblib
import numpy as np

# Existing scripts import each other by their directory-local names.
DIRECTORY = Path(__file__).resolve().parent
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
import fit_figure3_paper as fit
import export_figure3_paper as export
import plot_figure3_paper as plot

PROJECT = fit.PROJECT
OUTPUT = PROJECT / 'results/monkey_figure3_paper_v1/figure3g_position_only_correction_v1'
SHUFFLE = PROJECT / 'results/user_soft_rat_behavior_only_shuffle_v1'
ITEM = ('figure3g_all_position_only_4d_corrected', 'all', 'position', 4)
OLD = PROJECT / 'results/monkey_figure3_paper_v1/stage01_fit/figure3g_all_position_ap_4d/seed_42'
FIGURE_NAME = 'FIGURE3G_POSITION_ONLY_CORRECTED'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def plan():
    old_manifest = OLD / 'FIT_PROVENANCE.json'
    old = json.loads(old_manifest.read_text(encoding='utf-8'))
    expected = dict(seed=42, output_dimension=4, batch_size=512, learning_rate=0.0003,
                    temperature=1.0, max_iterations=10000, model_architecture='offset10-model',
                    time_offsets=10, distance='cosine', conditional='time_delta', label_mode='position_ap')
    if any(old['config'][k] != value for k, value in expected.items()):
        raise ValueError('Unexpected original Figure3g protocol; do not infer a replacement')
    if fit.sha256(OLD / 'model.pt') != old['checkpoint_sha256']:
        raise ValueError('Original Figure3g checkpoint hash mismatch')
    source = fit.DATA / 'all_all.jl'
    splits = json.loads(fit.SPLITS.read_text(encoding='utf-8'))
    if fit.sha256(source) != old['source_sha256'] or old['source_sha256'] != splits['source_file_hashes'][source.name]['sha256']:
        raise ValueError('Frozen descriptive source hash mismatch')
    values = joblib.load(source)
    if np.asarray(values['spikes']).shape != (218400, 65) or np.asarray(values['pos']).shape != (218400, 2) or int(values['trial_len']) != 600:
        raise ValueError('Figure3g source shape mismatch')
    if not np.isfinite(values['spikes']).all() or not np.isfinite(values['pos']).all():
        raise ValueError('Nonfinite Figure3g source')
    config = dict(old['config'], name=ITEM[0], label_mode='position')
    files = [Path(__file__), Path(fit.__file__), Path(export.__file__), Path(plot.__file__),
             fit.SPLITS, old_manifest, DIRECTORY / 'official_reference/Figure3.py']
    return dict(issue='M03', classification='BUG_FIX', item=list(ITEM), config=config,
                output=str(OUTPUT), source=str(source), source_sha256=old['source_sha256'],
                source_hashes={str(p): fit.sha256(p) for p in files},
                original_checkpoint_sha256=old['checkpoint_sha256'],
                prerequisite=dict(path=str(SHUFFLE), complete_fits=144),
                label_shape=[218400, 2], split='all_descriptive_not_held_out',
                original_artifacts_preserved=True, dry_run_optimizer_updates=0)


def shuffle_ready(state, lock_exists):
    records = state.get('records', {})
    return (not lock_exists and state.get('status') == 'complete'
            and state.get('planned') == 144 and state.get('completed') == 144
            and len(records) == 144 and all(r.get('status') == 'complete' for r in records.values()))


def wait_for_shuffle(wait):
    while True:
        state_path = SHUFFLE / 'CAMPAIGN_STATE.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if shuffle_ready(state, (SHUFFLE / 'RUNNING.lock').exists()):
            return
        if not wait:
            raise RuntimeError('Rat shuffle144-fit completion gate not satisfied')
        if state.get('status') in ('failed', 'interrupted', 'stopped_after_current'):
            raise RuntimeError(f'Rat shuffle stopped: {state.get("status")}')
        time.sleep(30)


@contextmanager
def scoped_paths():
    bindings = [(fit, 'ROOT', OUTPUT / 'stage01_fit'),
                (export, 'FIT_ROOT', OUTPUT / 'stage01_fit'),
                (export, 'ROOT', OUTPUT / 'stage02_embedding'),
                (plot, 'EMBED', OUTPUT / 'stage02_embedding'),
                (plot, 'FIGURES', OUTPUT / 'stage03_figures')]
    old = [(module, key, getattr(module, key)) for module, key, _ in bindings]
    try:
        for module, key, value in bindings:
            setattr(module, key, value)
        yield
    finally:
        for module, key, value in old:
            setattr(module, key, value)


def fit_export_render():
    with scoped_paths():
        print(fit.fit_one(ITEM), flush=True)
        print(export.export_one(ITEM), flush=True)
        plot.plot_3efg(ITEM[0], None, 'Trained with continuous position label',
                      (0, 1, 2), 'row', FIGURE_NAME)


def artifact_paths():
    model = OUTPUT / 'stage01_fit' / ITEM[0] / 'seed_42'
    embedding = OUTPUT / 'stage02_embedding' / ITEM[0] / 'seed_42'
    return [model / 'model.pt', model / 'training_loss.npy', model / 'FIT_PROVENANCE.json',
            embedding / 'all_trials.npz', embedding / 'EMBEDDING_PROVENANCE.json',
            OUTPUT / 'stage03_figures' / (FIGURE_NAME + '.png'),
            OUTPUT / 'stage03_figures' / (FIGURE_NAME + '.pdf')]


def verify_outputs():
    files = artifact_paths()
    record = json.loads(files[2].read_text())
    if record['config']['label_mode'] != 'position' or record['label_shapes'] != [[218400, 2]] or record['optimizer_updates'] != 10000:
        raise ValueError('Corrected fit label/update evidence does not match Figure3g')
    if record['checkpoint_sha256'] != fit.sha256(files[0]):
        raise ValueError('Corrected checkpoint mismatch')
    with np.load(files[3], allow_pickle=False) as arrays:
        if arrays['embedding_raw'].shape != (218400, 4) or arrays['embedding_unit'].shape != (218400, 4):
            raise ValueError('Corrected embedding dimension/cardinality mismatch')
        for key in ('trial_id', 'time_id', 'valid_mask', 'position_xy', 'direction', 'passive'):
            if len(arrays[key]) != 218400:
                raise ValueError(f'Corrected embedding metadata cardinality mismatch:{key}')
        if not np.array_equal(arrays['trial_id'], np.repeat(np.arange(364), 600)) or not np.array_equal(arrays['time_id'], np.tile(np.arange(600), 364)):
            raise ValueError('Corrected global trial/time IDs mismatch')
    return {str(p): fit.sha256(p) for p in files}


def execute(wait=False):
    planned = plan()
    wait_for_shuffle(wait)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = OUTPUT / 'RUNNING.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    state_path = OUTPUT / 'CORRECTION_STATE.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'events': []}
    try:
        seal = OUTPUT / 'CORRECTION_PLAN.json'
        if seal.exists():
            if json.loads(seal.read_text()) != planned:
                raise ValueError('Immutable correction source/config plan mismatch')
        else:
            write_state(seal, planned)
        final = OUTPUT / 'CORRECTION_PROVENANCE.json'
        if final.exists():
            saved = json.loads(final.read_text())
            if saved['plan_sha256'] != fit.sha256(seal) or saved['artifact_hashes'] != verify_outputs():
                raise ValueError('Completed correction provenance mismatch')
            print('REUSE completed Figure3g correction; zero new optimizer updates', flush=True)
        else:
            state.update(status='running', pid=os.getpid(), updated_utc=now())
            write_state(state_path, state)
            fit_export_render()
            write_state(final, dict(created_utc=now(), plan_sha256=fit.sha256(seal),
                        artifact_hashes=verify_outputs(), old_checkpoint_sha256=planned['original_checkpoint_sha256'],
                        scientific_change='remove unintended AP label; XY-only Figure3g',
                        original_artifacts_preserved=True, original_seed=42,
                        raw_unit_preserved=True, full_sample_descriptive=True))
        state.update(status='complete', updated_utc=now())
        state['events'].append(dict(status='complete', utc=now()))
        write_state(state_path, state)
    except BaseException as exc:
        state.update(status='failed', last_error=repr(exc), updated_utc=now())
        state['events'].append(dict(status='failed', error=repr(exc), utc=now()))
        write_state(state_path, state)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--execute', action='store_true')
    action.add_argument('--wait-and-execute', action='store_true')
    args = parser.parse_args()
    if args.execute or args.wait_and_execute:
        execute(wait=args.wait_and_execute)
    else:
        print(json.dumps(plan(), indent=2))


if __name__ == '__main__':
    main()
