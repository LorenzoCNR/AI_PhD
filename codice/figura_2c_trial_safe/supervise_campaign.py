"""Wait for the 72-fit campaign, verify its artifacts, then run 12 diagnostics.

The runner is never started. Diagnostic outputs are grouped under ROOT/plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from prepare_data import CONDITIONS, DEFAULT_OUTPUT, MODELS, RATS

SCOPES = ('split', 'full')
DIAGNOSTICS = Path(__file__).with_name('diagnostics_plots.py')
RUNNER = Path(__file__).with_name('run_campaign.py').resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def report(path, state):
    state['updated'] = now()
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def find_runner(pid=None):
    matches = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            if pid is not None and process.pid != pid:
                continue
            cmd = process.info['cmdline'] or []
            # Python options may precede the script; ignore unrelated Python -c commands.
            for arg in cmd[1:]:
                if not arg.startswith('-') and Path(arg).name.lower() == RUNNER.name.lower():
                    if Path(arg).is_absolute() and Path(arg).resolve() != RUNNER:
                        continue
                    matches.append(process)
                    break
        except (psutil.Error, OSError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f'expected exactly one accessible campaign runner, found {len(matches)}; use --runner-pid')
    return matches[0]


def validate_campaign(root, state):
    expected = list(itertools.product(SCOPES, RATS, MODELS, CONDITIONS))
    if len(expected) != 72 or state.get('total') != 72 or state.get('done') != 72 or not state.get('finished') or state.get('current') is not None or state.get('error'):
        raise RuntimeError('campaign state does not indicate successful completion of 72 fits')
    expected_paths = {root / 'fits' / s / r / m / c / 'manifest.json' for s, r, m, c in expected}
    actual = set((root / 'fits').glob('*/*/*/*/manifest.json'))
    if actual != expected_paths:
        raise RuntimeError(f'manifest cardinality/membership mismatch: {len(actual)} found; missing={len(expected_paths-actual)} extra={len(actual-expected_paths)}')
    prepared_path = root / 'data' / 'PREPARED.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    counts = {}
    for rel, digest in prepared['artifacts'].items():
        file = root / rel
        if not file.is_file() or sha256(file) != digest:
            raise RuntimeError(f'prepared artifact hash mismatch: {file}')
    for scope, rat, model, condition in expected:
        path = root / 'fits' / scope / rat / model / condition / 'manifest.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record.get('scope'), record.get('rat'), record.get('model'), record.get('condition'), record.get('signature')) != (scope, rat, model, condition, state['signature']):
            raise RuntimeError(f'manifest identity/signature mismatch: {path}')
        if record.get('updates') != 10000 or record.get('embedding_dimension') != 3:
            raise RuntimeError(f'invalid fit metadata: {path}')
        splits = ('train', 'valid', 'test') if scope == 'split' else ('full',)
        required = {'checkpoint.pt', 'training_history.csv'} | {f'{s}_embedding.npz' for s in splits}
        if model.startswith('soft'):
            required.add('distance_scales.json')
        artifacts = record.get('artifacts')
        if not isinstance(artifacts, dict) or not required <= artifacts.keys():
            raise RuntimeError(f'missing required artifacts in {path}')
        for name, digest in artifacts.items():
            if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str):
                raise RuntimeError(f'invalid artifact record in {path}')
            file = path.parent / name
            if not file.is_file() or sha256(file) != digest:
                raise RuntimeError(f'artifact hash mismatch: {file}')
        with (path.parent / 'training_history.csv').open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['update', 'train_loss']:
                raise RuntimeError(f'invalid history columns: {path}')
            n = 0
            for n, row in enumerate(reader, 1):
                if int(row['update']) != n or not math.isfinite(float(row['train_loss'])):
                    raise RuntimeError(f'invalid loss history: {path}')
            if n != 10000:
                raise RuntimeError(f'invalid loss count: {path}')
        for split in splits:
            key = (scope, rat, split)
            if key not in counts:
                with np.load(root / 'data' / scope / rat / f'{split}.npz', allow_pickle=False) as data:
                    counts[key] = data['source_id'].copy()
            with np.load(path.parent / f'{split}_embedding.npz', allow_pickle=False) as data:
                ids = counts[key]
                if data['embedding'].shape != (len(ids), 3) or not np.array_equal(data['source_id'], ids):
                    raise RuntimeError(f'embedding cardinality or source IDs mismatch: {path}/{split}')
        counts[(scope, rat, model, condition)] = 1
    return {'fits': len(expected), 'manifest_sha256': {str(p.relative_to(root)): sha256(p) for p in sorted(expected_paths)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--runner-pid', type=int, help='PID of the already-running campaign')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--left-label', default='0')
    parser.add_argument('--right-label', default='1')
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.left_label == args.right_label:
        parser.error('poll-seconds must be positive and direction labels must differ')
    root = args.root.resolve()
    output = root / 'plot'
    output.mkdir(parents=True, exist_ok=True)
    status_path = root / 'PLOT_STATE.json'
    status = {'status': 'waiting', 'started': now(), 'root': str(root), 'output': str(output), 'plots_completed': [], 'left_label': args.left_label, 'right_label': args.right_label}
    report(status_path, status)
    print(f'Status: {status_path}', flush=True)
    try:
        state_path = root / 'CAMPAIGN_STATE.json'
        initial = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
        process = None
        if initial and initial.get('error'):
            raise RuntimeError('runner reported error: ' + str(initial['error']))
        if not initial or not initial.get('finished'):
            process = find_runner(args.runner_pid)
            status['runner_pid'] = process.pid
            status['runner_create_time'] = process.create_time()
        initial_started = initial.get('started') if initial else None
        while True:
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
            if state:
                if initial_started and state.get('started') != initial_started:
                    raise RuntimeError('campaign state was replaced by a different run')
                initial_started = state.get('started')
                status['campaign'] = state
                if state.get('error'):
                    raise RuntimeError('runner reported error: ' + str(state['error']))
                if state.get('finished'):
                    break
            if process and (not process.is_running() or process.status() == psutil.STATUS_ZOMBIE):
                raise RuntimeError('runner exited before recording successful completion')
            report(status_path, status)
            time.sleep(args.poll_seconds)
        status['status'] = 'verifying'
        report(status_path, status)
        status['verification'] = validate_campaign(root, state)
        status['status'] = 'plotting'
        report(status_path, status)
        for scope, rat, model in itertools.product(SCOPES, RATS, MODELS):
            dest = output / scope / rat / model
            command = [sys.executable, str(DIAGNOSTICS), '--root', str(root), '--scope', scope,
                       '--rat', rat, '--model', model, '--output', str(dest),
                       '--left-label', args.left_label, '--right-label', args.right_label]
            status['current_plot'] = {'scope': scope, 'rat': rat, 'model': model, 'output': str(dest)}
            report(status_path, status)
            subprocess.run(command, check=True)
            complete = dest / 'diagnostics_complete.json'
            artifacts = json.loads(complete.read_text(encoding='utf-8'))
            if not artifacts or any(not (dest / name).is_file() or sha256(dest / name) != digest for name, digest in artifacts.items()):
                raise RuntimeError(f'diagnostics output verification failed: {dest}')
            status['plots_completed'].append(status['current_plot'])
            report(status_path, status)
        summary_script = Path(__file__).with_name('summary_comparison.py')
        for scope, rat in itertools.product(SCOPES, RATS):
            dest = output / scope / rat / 'confronto_modelli'
            command = [sys.executable, str(summary_script), '--rat', rat, '--scope', scope,
                       '--diagnostic'] + [str(output / scope / rat / model) for model in MODELS] + ['--output', str(dest)]
            status['current_plot'] = {'scope': scope, 'rat': rat, 'model': 'confronto_modelli', 'output': str(dest)}
            report(status_path, status)
            if not dest.exists():
                subprocess.run(command, check=True)
            status['plots_completed'].append(status['current_plot'])
            report(status_path, status)
        status.update(status='complete', finished=now(), current_plot=None)
        report(status_path, status)
        return 0
    except Exception:
        status.update(status='error', finished=now(), error=traceback.format_exc())
        report(status_path, status)
        print(status['error'], file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
