"""Auditable Transformer seed-47/48 extension across explicit Soft conditions.

The historical five-seed artifacts are read-only. This runner references their
completed cells and creates new outputs only below its separate campaign root.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import unittest
from pathlib import Path

import torch

from . import behavior_only_runner as bo
from . import rat_runner as rr
from . import time_only_runner as tm
from . import transformer_campaign as tc
from . import transformer_runner as tr
from .data import config_hash, load_rows, sha256

DEFAULT_PARENT = rr.ROOT / 'results' / 'user_soft_rat_transformer_5seed_v1'
DEFAULT_BEHAVIOR_PARENT = bo.DEFAULT_OUTPUT
DEFAULT_OUTPUT = rr.ROOT / 'results' / 'user_soft_rat_transformer_seed47_48_extension_v1'
SEEDS = (47, 48)
CONDITIONS = ('time_direction_position', 'time_only', 'direction_position')
SEED_GRID = {
    'time_direction_position': (47, 48),
    'time_only': (47, 48),
    'direction_position': (48,),
}
VERSIONS = {
    'time_direction_position': rr.VERSIONS,
    'time_only': rr.VERSIONS,
    'direction_position': (bo.VERSION,),
}
EXPECTED_PARENT_LOSS_SHA256 = 'e47a9a9ffa14ad2eec71ac4f8f489ad84854820c9a444df047effe568b44b027'
CURRENT_LOSS_SHA256 = 'b7770f8bc082b4b69cc5968a2e0f3ccf20bb914e9116e81e65bbf03c7d7d1bfc'
QUALIFICATION_TESTS = (
    'user_soft.tests.test_transformer',
    'user_soft.tests.test_transformer_campaign',
    'user_soft.tests.test_time_only',
    'user_soft.tests.test_behavior_only',
)


def runtime_versions():
    packages = {}
    for name in ('torch', 'numpy', 'cebra'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages)


def make_config(condition, rat, fold, seed, version):
    if condition == 'time_direction_position':
        config = tr.frozen_config(rat, fold, seed, version)
        config.update(condition=condition, metadata_specs=copy.deepcopy(rr.SPECS),
                      metadata_supervision='time_plus_position_direction')
    elif condition == 'time_only':
        config = tm.frozen_config(rat, fold, seed, version)
        config['condition'] = condition
    elif condition == 'direction_position':
        config = bo.frozen_config('transformer', rat, fold, seed)
        config.update(condition=condition,
                      metadata_supervision='position_direction_without_time')
    else:
        raise ValueError(f'Unknown explicit condition: {condition}')
    return config


def enumerate_configs():
    records = []
    for condition in CONDITIONS:
        for seed, fold, rat, version in itertools.product(
                SEED_GRID[condition], range(3), rr.RATS, VERSIONS[condition]):
            records.append((condition, make_config(condition, rat, fold, seed, version)))
    return records


def state_key(condition, config):
    return (f"{condition}/{config['rat']}/fold_{config['fold']}/"
            f"{config['version']}/seed_{config['seed']}")


def historical_source_delta(parent_plan):
    historical_sources = parent_plan['sources']['joint_training_sources']
    current_sources = tr.sources()
    changed = {
        key: [historical_sources.get(key), current_sources.get(key)]
        for key in set(historical_sources) | set(current_sources)
        if historical_sources.get(key) != current_sources.get(key)
    }
    loss_path = str((Path(rr.__file__).parent / 'loss.py').resolve())
    expected = {loss_path: [EXPECTED_PARENT_LOSS_SHA256, CURRENT_LOSS_SHA256]}
    if changed != expected or sha256(Path(loss_path)) != CURRENT_LOSS_SHA256:
        raise RuntimeError(
            'Historical fit sources differ beyond the reviewed TRAIN-scope loss change; '
            f'refusing extension: {changed}')
    return changed


def _verify_embedding_manifest(path, expected_digest=None):
    path = Path(path)
    if not path.is_file() or (expected_digest and sha256(path) != expected_digest):
        raise RuntimeError(f'Embedding manifest missing or changed: {path}')
    record = json.loads(path.read_text(encoding='utf-8'))
    artifacts = {}
    for split, entry in record.get('embeddings', {}).items():
        artifact = path.parent / f'{split}_embedding.npz'
        digest = sha256(artifact) if artifact.is_file() else None
        if digest is None or digest != entry.get('embedding_sha256'):
            raise RuntimeError(f'Embedding artifact missing or changed: {artifact}')
        artifacts[str(artifact.resolve())] = digest
    if set(record.get('embeddings', {})) != {'train', 'valid', 'test'}:
        raise RuntimeError(f'Unexpected embedding split set: {path}')
    return record, artifacts


def _validate_cell_artifacts(state_record, expected_config=None, expected_sources=None):
    fit_path = Path(state_record['fit_manifest_path'])
    metrics_path = Path(state_record['metrics_path'])
    embedding_path = Path(state_record['embedding_manifest_path'])
    checkpoint_path = fit_path.with_name('checkpoint.pt')
    required = (
        (fit_path, state_record.get('fit_manifest_sha256')),
        (metrics_path, state_record.get('metrics_sha256')),
        (embedding_path, state_record.get('embedding_manifest_sha256')),
        (checkpoint_path, state_record.get('checkpoint_sha256')),
    )
    for path, expected in required:
        if not path.is_file() or not expected or sha256(path) != expected:
            raise RuntimeError(f'Completed parent artifact missing or changed: {path}')
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    if expected_config is not None and fit.get('config') != expected_config:
        raise RuntimeError(f'Fit config disagrees with its sealed plan: {fit_path}')
    if expected_sources is not None and fit.get('source_hashes') != expected_sources:
        raise RuntimeError(f'Fit source hashes disagree with their sealed plan: {fit_path}')
    if fit.get('checkpoint_sha256') != sha256(checkpoint_path):
        raise RuntimeError(f'Checkpoint hash disagrees with its fit manifest: {checkpoint_path}')
    embedding, embedding_artifacts = _verify_embedding_manifest(
        embedding_path, state_record.get('embedding_manifest_sha256'))
    metric = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metric.get('config') is None or not metric.get('signature'):
        raise RuntimeError(f'Metrics artifact has no config/signature: {metrics_path}')
    diagnostics = {}
    diagnostic_path = state_record.get('diagnostics_path')
    if diagnostic_path:
        diagnostic_file = Path(diagnostic_path)
        if not diagnostic_file.is_file():
            raise RuntimeError(f'Diagnostic artifact missing: {diagnostic_file}')
        marker = diagnostic_file.parent / 'STAGE_SIGNATURE.json'
        if marker.is_file():
            marker_record = json.loads(marker.read_text(encoding='utf-8'))
            for name, digest in marker_record.get('artifact_hashes', {}).items():
                artifact = marker.parent / name
                if not artifact.is_file() or sha256(artifact) != digest:
                    raise RuntimeError(f'Diagnostic content mismatch: {artifact}')
            diagnostics = dict(path=str(marker.resolve()), sha256=sha256(marker))
    return dict(
        fit_manifest=dict(path=str(fit_path.resolve()), sha256=sha256(fit_path)),
        checkpoint=dict(path=str(checkpoint_path.resolve()), sha256=sha256(checkpoint_path)),
        embedding_manifest=dict(path=str(embedding_path.resolve()), sha256=sha256(embedding_path)),
        embeddings=embedding_artifacts,
        metrics=dict(path=str(metrics_path.resolve()), sha256=sha256(metrics_path)),
        diagnostics=diagnostics,
        output_dimension=fit.get('config', {}).get('output_dimension'),
        fit_config=fit.get('config'),
        fit_sources=fit.get('source_hashes'),
    )


def validate_five_seed_parent(parent):
    plan_path, state_path = parent / 'CAMPAIGN_PLAN.json', parent / 'CAMPAIGN_STATE.json'
    if not plan_path.is_file() or not state_path.is_file():
        raise RuntimeError('Sealed five-seed Transformer parent plan/state is missing')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    state = json.loads(state_path.read_text(encoding='utf-8'))
    historical_configs = tc.enumerate_configs(5)
    expected_plan_configs = [dict(condition=c, config=x) for c, x in historical_configs]
    if (plan.get('seed_count') != 5 or plan.get('planned') != 240
            or plan.get('configs') != expected_plan_configs):
        raise RuntimeError('Parent is not the sealed 240-cell seeds-42-to-46 campaign')
    expected_keys = {state_key(c, x) for c, x in historical_configs}
    if (state.get('status') != 'complete' or state.get('completed') != 240
            or set(state.get('records', {})) != expected_keys
            or any(r.get('status') != 'complete' for r in state['records'].values())):
        raise RuntimeError('All 240 historical Transformer cells must be complete')
    source_delta = historical_source_delta(plan)
    reuse = []
    for old_condition, config in historical_configs:
        key = state_key(old_condition, config)
        item = state['records'][key]
        expected_sources = tc.training_sources(old_condition, plan['sources'])
        evidence = _validate_cell_artifacts(item, config, expected_sources)
        specs = config.get('metadata_specs', [])
        spec_keys = {spec.get('key') for spec in specs}
        if spec_keys == {'time_id', 'behavior'}:
            condition = 'time_direction_position'
        elif spec_keys == {'time_id'}:
            condition = 'time_only'
        else:
            raise RuntimeError(f'Cannot infer historical modality mapping from {key}: {specs}')
        if condition == 'time_direction_position' and config.get('output_dimension') != 32:
            raise RuntimeError(f'Unexpected historical Transformer dimension: {key}')
        reuse.append(dict(condition=condition, seed=config['seed'], rat=config['rat'],
            fold=config['fold'], version=config['version'], historical_state_key=key,
            artifacts=evidence))
    summary_tables = {}
    for filename in ('RAT_TRANSFORMER_COMPARISON.csv', 'COMPUTE_PERFORMANCE.csv'):
        path = parent / filename
        if not path.is_file():
            raise RuntimeError(f'Historical parent summary missing: {filename}')
        summary_tables[filename] = dict(path=str(path.resolve()), sha256=sha256(path))
    return dict(path=str(parent.resolve()), plan_sha256=sha256(plan_path),
        state_sha256=sha256(state_path), completed_cells=240, source_delta=source_delta,
        summary_tables=summary_tables, reuse=reuse)


def _behavior_state_key(architecture, config):
    return (f"{architecture}/{config['rat']}/fold_{config['fold']}/"
            f"{config['version']}/seed_{config['seed']}")


def validate_behavior_parent(parent):
    plan_path, state_path = parent / 'CAMPAIGN_PLAN.json', parent / 'CAMPAIGN_STATE.json'
    if not plan_path.is_file() or not state_path.is_file():
        raise RuntimeError('Behavior-only parent plan/state is missing')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    state = json.loads(state_path.read_text(encoding='utf-8'))
    planned_configs = plan.get('configs', [])
    if (plan.get('planned') != 168 or len(planned_configs) != 168
            or set(plan.get('seeds', [])) != set(range(42, 49))):
        raise RuntimeError('Behavior-only sealed plan is not the expected seven-seed design')
    if (state.get('status') != 'complete' or state.get('planned') != 144
            or state.get('completed') != 144 or len(state.get('records', {})) != 144
            or any(r.get('status') != 'complete' for r in state['records'].values())):
        raise RuntimeError('Expected complete six-seed behavior-only state (144/144 cells)')
    planned_by_key = {}
    for item in planned_configs:
        architecture, config = item['architecture'], item['config']
        planned_by_key[_behavior_state_key(architecture, config)] = (architecture, config)
    expected_keys = set()
    reuse = []
    for (architecture, config) in planned_by_key.values():
        key = _behavior_state_key(architecture, config)
        if config['seed'] > 47:
            continue
        expected_keys.add(key)
        record = state['records'].get(key)
        if record is None:
            raise RuntimeError(f'Behavior-only completed cell missing from state: {key}')
        if architecture != 'transformer':
            continue
        evidence = _validate_cell_artifacts(record, config, plan['sources'])
        if config.get('output_dimension') != 32:
            raise RuntimeError(f'Unexpected behavior-only Transformer dimension: {key}')
        if config.get('metadata_specs') != bo.SPECS:
            raise RuntimeError(f'Behavior-only fit does not use behavior metadata only: {key}')
        reuse.append(dict(condition='direction_position', seed=config['seed'],
            rat=config['rat'], fold=config['fold'], version=config['version'],
            historical_state_key=key, artifacts=evidence))
    if set(state['records']) != expected_keys:
        raise RuntimeError('Behavior-only state does not match its completed seeds-42-to-47 cells')
    if len(reuse) != 72:
        raise RuntimeError(f'Expected 72 reusable behavior-only Transformer cells, got {len(reuse)}')
    for rat, fold in itertools.product(rr.RATS, range(3)):
        config = make_config('direction_position', rat, fold, 48, bo.VERSION)
        cell = parent / 'transformer' / 'stage01_fit' / rat / f'fold_{fold}' / bo.VERSION / 'seed_48'
        if cell.exists():
            raise RuntimeError(f'Unindexed seed-48 artifact exists in historical output; inspect first: {cell}')
    return dict(path=str(parent.resolve()), plan_sha256=sha256(plan_path),
        state_sha256=sha256(state_path), plan_cells=168, completed_cells=144,
        state_seed_coverage=list(range(42, 48)), plan_state_discrepancy=(
            'sealed plan enumerates seed 48; completed state contains only seeds 42-47; '
            'historical plan/state are preserved unchanged'), reuse=reuse)


def build_source_record():
    return dict(
        extension_runner_sha256=sha256(__file__),
        transformer_sources=tr.sources(),
        cnn_training_sources=rr.sources(),
        time_only_sources=tm.sources(),
        direction_position_sources=bo.sources(),
        explicit_condition_specs={
            'time_direction_position': copy.deepcopy(rr.SPECS),
            'time_only': copy.deepcopy(tm.SPECS),
            'direction_position': copy.deepcopy(bo.SPECS),
        },
        runtime=runtime_versions(),
    )


def validate_parents(parent, behavior_parent):
    return dict(
        time_and_behavior=validate_five_seed_parent(parent),
        behavior_only=validate_behavior_parent(behavior_parent),
    )


def make_plan(parent_record, source_record, reuse_index_digest):
    configs = enumerate_configs()
    return dict(
        schema='transformer_explicit_conditions_seed47_48_extension_v1',
        planned=len(configs),
        seeds_by_condition={name: list(SEED_GRID[name]) for name in CONDITIONS},
        conditions=list(CONDITIONS),
        condition_mapping={
            'time_direction_position': 'time_id temporal + behavior Euclidean (position and direction)',
            'time_only': 'time_id temporal only',
            'direction_position': 'behavior Euclidean only (position and direction), no time',
        },
        output_dimension=32,
        configs=[dict(condition=condition, config=config) for condition, config in configs],
        sources=source_record,
        parent_reuse_index_sha256=reuse_index_digest,
        parents={
            'time_direction_position_and_time_only': {
                'path': parent_record['time_and_behavior']['path'],
                'plan_sha256': parent_record['time_and_behavior']['plan_sha256'],
                'state_sha256': parent_record['time_and_behavior']['state_sha256'],
                'reused_cells': 240,
                'reviewed_source_delta': parent_record['time_and_behavior']['source_delta'],
            },
            'direction_position': {
                'path': parent_record['behavior_only']['path'],
                'plan_sha256': parent_record['behavior_only']['plan_sha256'],
                'state_sha256': parent_record['behavior_only']['state_sha256'],
                'reused_cells': 72,
                'historical_plan_state_discrepancy': parent_record['behavior_only']['plan_state_discrepancy'],
            },
        },
        reused_completed_cells=312,
        new_fit_cells=len(configs),
        optimizer_updates_in_dry_run=0,
        selection='fixed settings and terminal checkpoint; no HPO or test-outcome selection',
    )


def _atomic_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_extension_tables(output, state):
    rows, compute = [], []
    for key, entry in sorted(state.get('records', {}).items()):
        if entry.get('status') != 'complete':
            continue
        metric = json.loads(Path(entry['metrics_path']).read_text(encoding='utf-8'))
        config = metric['config']
        for item in metric['metrics']:
            rows.append(dict(condition=entry['condition'], architecture=config['architecture'],
                rat=config['rat'], fold=config['fold'], seed=config['seed'], method=config['version'],
                **{k: v for k, v in item.items() if k not in ('method', 'ridge_validation_candidates')}))
        fit = json.loads(Path(entry['fit_manifest_path']).read_text(encoding='utf-8'))
        embedding = json.loads(Path(entry['embedding_manifest_path']).read_text(encoding='utf-8'))
        compute.append(dict(condition=entry['condition'], architecture=config['architecture'],
            rat=config['rat'], fold=config['fold'], seed=config['seed'], method=config['version'],
            **{name: fit.get(name) for name in (
                'parameters', 'actual_updates', 'actual_optimizer_updates', 'replayed_updates',
                'fit_seconds', 'seconds_per_100_updates', 'updates_per_second',
                'training_windows_per_second', 'peak_gpu_allocated', 'peak_gpu_reserved',
                'device', 'cuda_available', 'torch_version')},
            inference_seconds=embedding.get('inference_seconds'),
            inference_windows_per_second=embedding.get('inference_windows_per_second')))
    _atomic_csv(output / 'TRANSFORMER_EXTENSION_COMPARISON.csv', rows)
    _atomic_csv(output / 'COMPUTE_PERFORMANCE_EXTENSION.csv', compute)


def _prepare_cell(output, condition, rat, fold):
    parent_folder = rr.DEFAULT_OUTPUT / 'stage00_data' / rat / f'fold_{fold}'
    parent_manifest_path = parent_folder / 'manifest.json'
    if not parent_manifest_path.is_file():
        raise RuntimeError(f'Existing prepared TRAIN/VALID/TEST cache required; refusing regeneration: {parent_manifest_path}')
    data_folder, parent = rr.prepare(rr.DEFAULT_OUTPUT, rat, fold)
    if condition == 'time_direction_position':
        expected = {'time_id', 'behavior'}
        scales = copy.deepcopy(parent['train_scales'])
        specs = copy.deepcopy(rr.SPECS)
    elif condition == 'time_only':
        expected = {'time_id'}
        scales = tm.project_scales(parent['train_scales'])
        specs = copy.deepcopy(tm.SPECS)
    else:
        expected = {'behavior'}
        scales = bo.project_scales(parent['train_scales'])
        specs = copy.deepcopy(bo.SPECS)
    actual = {scale['key'] for scale in scales['scales']}
    if actual != expected:
        raise RuntimeError(f'Scale projection mismatch for {condition}: {actual}')
    record = dict(condition=condition, rat=rat, fold=fold,
        parent_cache_manifest=str((data_folder / 'manifest.json').resolve()),
        parent_cache_sha256=sha256(data_folder / 'manifest.json'),
        parent_cache_signature=parent['signature'], splits=parent['splits'],
        train_scales=scales, metadata_specs=specs, no_scale_refit=True,
        no_test_outcome_selection=True, output_dimension=32)
    record['signature'] = config_hash(record)
    path = output / condition / 'stage00_data' / rat / f'fold_{fold}' / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != record:
            raise RuntimeError(f'Prepared data signature changed; refusing overwrite: {path}')
    else:
        rr.atomic_json(path, record)
    return data_folder, record


def prepare_all(output):
    prepared = {}
    for condition in CONDITIONS:
        for rat, fold in itertools.product(rr.RATS, range(3)):
            prepared[(condition, rat, fold)] = _prepare_cell(output, condition, rat, fold)
    return prepared


def qualification_test_hashes():
    tests_dir = Path(__file__).parent / 'tests'
    return {f'{name}.py': sha256(tests_dir / f'{name}.py')
            for name in ('test_transformer', 'test_transformer_campaign',
                         'test_time_only', 'test_behavior_only')}


def qualify(output, source_record):
    stream_parts, counts, failures, errors = [], 0, [], []
    for name in QUALIFICATION_TESTS:
        suite = unittest.defaultTestLoader.loadTestsFromName(name)
        import io
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        counts += result.testsRun
        stream_parts.append(f'[{name}]\n{stream.getvalue()}')
        failures.extend((name, failure) for _, failure in result.failures)
        errors.extend((name, error) for _, error in result.errors)
    record = dict(created_utc=rr.now(), decision='GO' if counts and not failures and not errors else 'NO-GO',
        training_source_signature=config_hash(source_record), tests_run=counts,
        tests_failed=len(failures), tests_errors=len(errors),
        test_source_hashes=qualification_test_hashes(),
        scope='CPU qualification only; zero rat fits; explicit condition and resume pipeline')
    rr.atomic_json(output / 'qualification' / 'TEST_RESULTS.json', dict(**record, output='\n'.join(stream_parts)))
    rr.atomic_json(output / 'qualification' / 'GATE.json', record)
    print(json.dumps(record, indent=2), flush=True)
    if record['decision'] != 'GO':
        raise RuntimeError('Extension CPU qualification failed')


def _check_gate(path, expected_sources, label, expected_tests=None):
    if not path.is_file():
        raise RuntimeError(f'{label} qualification gate missing: {path}')
    gate = json.loads(path.read_text(encoding='utf-8'))
    if (gate.get('decision') != 'GO'
            or gate.get('training_source_signature') != config_hash(expected_sources)):
        raise RuntimeError(f'{label} qualification gate is stale or NO-GO')
    if expected_tests is not None and gate.get('test_source_hashes') != expected_tests:
        raise RuntimeError(f'{label} qualification test sources changed')


def check_execution_gates(output, source_record):
    _check_gate(output / 'qualification' / 'GATE.json', source_record,
                'Seed extension', qualification_test_hashes())
    _check_gate(tr.DEFAULT_OUTPUT / 'qualification' / 'GATE.json', tr.sources(), 'Initial Transformer')
    _check_gate(rr.DEFAULT_OUTPUT / 'qualification' / 'GATE.json', rr.sources(), 'Parent CNN')
    _check_gate(bo.DEFAULT_OUTPUT / 'qualification' / 'GATE.json', bo.sources(), 'Behavior-only parent')


def _mark_stage(state, state_path, key, stage, details):
    record = state['records'].setdefault(key, dict(condition=key.split('/', 1)[0], stages={}))
    record.setdefault('stages', {})[stage] = dict(status='complete', **details)
    record['status'] = stage + '_complete'
    state['completed'] = sum(r.get('status') == 'complete' for r in state['records'].values())
    state['updated_utc'] = rr.now()
    rr.atomic_json(state_path, state)


def _verify_recorded_stage(stage_record):
    path = Path(stage_record['manifest_path'])
    if not path.is_file() or sha256(path) != stage_record['manifest_sha256']:
        raise RuntimeError(f'Recorded stage manifest changed: {path}')
    for artifact, digest in stage_record.get('artifact_hashes', {}).items():
        file = Path(artifact)
        if not file.is_file() or sha256(file) != digest:
            raise RuntimeError(f'Recorded stage artifact changed: {file}')


def _verify_completed_record(entry, config):
    if entry.get('status') != 'complete' or entry.get('config') != config:
        raise RuntimeError('Completed extension cell has incompatible status/config')
    for stage in ('fit', 'embedding', 'metrics', 'diagnostics'):
        if entry.get('stages', {}).get(stage, {}).get('status') != 'complete':
            raise RuntimeError(f'Completed cell lacks a complete {stage} stage')
        _verify_recorded_stage(entry['stages'][stage])


@contextlib.contextmanager
def condition_bindings(condition):
    with contextlib.ExitStack() as stack:
        stack.enter_context(tr.architecture_bindings())
        if condition == 'time_only':
            stack.enter_context(tm.context_bindings())
        elif condition == 'direction_position':
            stack.enter_context(bo.context_bindings('transformer'))
        elif condition != 'time_direction_position':
            raise ValueError(f'Unknown condition: {condition}')
        yield


def _write_sealed_files(output, plan, reuse_index):
    plan_path, index_path = output / 'CAMPAIGN_PLAN.json', output / 'PARENT_REUSE_INDEX.json'
    if index_path.exists():
        old_index = json.loads(index_path.read_text(encoding='utf-8'))
        if old_index != reuse_index:
            raise RuntimeError('Parent reuse index changed; use a fresh extension root')
    else:
        rr.atomic_json(index_path, reuse_index)
    if sha256(index_path) != plan['parent_reuse_index_sha256']:
        raise RuntimeError('Parent reuse index does not match the sealed plan digest')
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding='utf-8'))
        if old != plan:
            raise RuntimeError('Sealed extension plan/source mismatch; use a fresh output root')
    else:
        unknown = [p.name for p in output.iterdir()
                   if p.name not in ('PARENT_REUSE_INDEX.json', 'RUNNING.lock')]
        if unknown:
            raise RuntimeError(f'Non-empty extension output has no matching plan: {unknown}')
        rr.atomic_json(plan_path, dict(created_utc=rr.now(), **plan))


def _acquire_lock(output):
    lock = output / 'RUNNING.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError('Extension already running or stale lock; inspect it before resuming') from exc
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def execute(output, configs, prepared, source_record, max_new_fits, device):
    check_execution_gates(output, source_record)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    state_path = output / 'CAMPAIGN_STATE.json'
    expected_keys = {state_key(condition, config) for condition, config in configs}
    state = (json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists()
             else dict(planned=len(configs), records={}, created_utc=rr.now(), completed=0))
    if state.get('planned') != len(configs) or not set(state.get('records', {})) <= expected_keys:
        raise RuntimeError('Existing state does not match the sealed explicit-condition extension')
    new_fits = 0
    try:
        for condition, config in configs:
            key = state_key(condition, config)
            existing = state['records'].get(key)
            if existing and existing.get('status') == 'complete':
                _verify_completed_record(existing, config)
                continue
            data_folder, data_manifest = prepared[(condition, config['rat'], config['fold'])]
            root = output / condition
            cell = f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
            fit_folder = root / 'stage01_fit' / cell
            state.update(current=key, status='running', updated_utc=rr.now())
            state['records'].setdefault(key, dict(condition=condition, config=config, stages={}))
            state['records'][key].update(condition=condition, config=config, status='running')
            rr.atomic_json(state_path, state)
            train, valid = load_rows(data_folder / 'train.npz'), load_rows(data_folder / 'valid.npz')
            parent_id = (sha256(data_folder / 'manifest.json') if condition == 'time_direction_position'
                         else config_hash(data_manifest))
            fit_sources = source_record[
                'transformer_sources' if condition == 'time_direction_position' else
                'time_only_sources' if condition == 'time_only' else 'direction_position_sources'
            ]
            if condition == 'time_direction_position':
                fit_sources = dict(fit_sources, extension_runner_sha256=source_record['extension_runner_sha256'])
            elif condition == 'time_only':
                fit_sources = dict(fit_sources, extension_runner_sha256=source_record['extension_runner_sha256'])
            else:
                fit_sources = dict(fit_sources, extension_runner_sha256=source_record['extension_runner_sha256'])
            with condition_bindings(condition):
                fit, created = rr.fit_cell(fit_folder, train, valid, config,
                    data_manifest['train_scales'], parent_id, fit_sources, device)
            new_fits += int(created)
            fit_path = fit_folder / 'manifest.json'
            checkpoint_path = fit_folder / 'checkpoint.pt'
            if not fit_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeError(f'Fit stage returned without complete checkpoint/manifest: {key}')
            _mark_stage(state, state_path, key, 'fit', dict(
                manifest_path=str(fit_path.resolve()), manifest_sha256=sha256(fit_path),
                artifact_hashes={str(checkpoint_path.resolve()): sha256(checkpoint_path)},
                signature=fit.get('signature'), parent_id=parent_id, source_hashes=fit_sources))
            embedding_folder, embedding = tr.export_cell(root, fit_folder, data_folder, config, fit, device)
            embedding_path = embedding_folder / 'manifest.json'
            embedding, embedding_artifacts = _verify_embedding_manifest(embedding_path)
            _mark_stage(state, state_path, key, 'embedding', dict(
                manifest_path=str(embedding_path.resolve()), manifest_sha256=sha256(embedding_path),
                artifact_hashes=embedding_artifacts, signature=embedding.get('signature')))
            metrics = rr.decode_cell(root, embedding_folder, config, embedding)
            metrics_path = root / 'stage03_metrics' / cell / 'metrics.json'
            _mark_stage(state, state_path, key, 'metrics', dict(
                manifest_path=str(metrics_path.resolve()), manifest_sha256=sha256(metrics_path),
                artifact_hashes={}, signature=metrics.get('signature')))
            if condition == 'time_only':
                tm.diagnostic_cell(root, embedding_folder, data_manifest, config)
                diagnostics_path = root / 'stage04_diagnostics' / config['rat'] / f"fold_{config['fold']}" / config['version'] / f"seed_{config['seed']}" / 'STAGE_SIGNATURE.json'
            elif condition == 'direction_position':
                bo.diagnostic_cell(root, embedding_folder, data_manifest, config)
                diagnostics_path = root / 'stage04_diagnostics' / config['rat'] / f"fold_{config['fold']}" / config['version'] / f"seed_{config['seed']}" / 'STAGE_SIGNATURE.json'
            else:
                rr.diagnostic_cell(root, embedding_folder, data_manifest, config)
                diagnostics_path = root / 'stage04_diagnostics' / config['rat'] / f"fold_{config['fold']}" / config['version'] / f"seed_{config['seed']}" / 'STAGE_SIGNATURE.json'
            if not diagnostics_path.is_file():
                raise RuntimeError(f'Diagnostic stage returned without signature: {key}')
            diagnostic_record = json.loads(diagnostics_path.read_text(encoding='utf-8'))
            diagnostic_artifacts = {str((diagnostics_path.parent / name).resolve()): digest
                                    for name, digest in diagnostic_record.get('artifact_hashes', {}).items()}
            _mark_stage(state, state_path, key, 'diagnostics', dict(
                manifest_path=str(diagnostics_path.resolve()), manifest_sha256=sha256(diagnostics_path),
                artifact_hashes=diagnostic_artifacts, signature=diagnostic_record.get('signature')))
            state['records'][key].update(status='complete',
                fit_manifest_path=str(fit_path.resolve()), fit_manifest_sha256=sha256(fit_path),
                checkpoint_sha256=sha256(checkpoint_path),
                embedding_manifest_path=str(embedding_path.resolve()),
                embedding_manifest_sha256=sha256(embedding_path),
                metrics_path=str(metrics_path.resolve()), metrics_sha256=sha256(metrics_path),
                diagnostics_path=str(diagnostics_path.resolve()))
            state['completed'] = sum(r.get('status') == 'complete' for r in state['records'].values())
            state['updated_utc'] = rr.now()
            rr.atomic_json(state_path, state)
            write_extension_tables(output, state)
            print(json.dumps(dict(completed=state['completed'], planned=len(configs),
                last=key, new_fit=created)), flush=True)
            if new_fits >= max_new_fits:
                state['status'] = 'requested_limit'
                break
        else:
            state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     last_error=repr(exc), updated_utc=rr.now())
        if state.get('current'):
            state['records'].setdefault(state['current'], {}).setdefault('last_error', repr(exc))
        raise
    finally:
        state['updated_utc'] = rr.now()
        rr.atomic_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--qualify', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--parent-output', type=Path, default=DEFAULT_PARENT)
    parser.add_argument('--behavior-parent-output', type=Path, default=DEFAULT_BEHAVIOR_PARENT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--max-new-fits', type=int)
    args = parser.parse_args()

    parent, behavior_parent = args.parent_output.resolve(), args.behavior_parent_output.resolve()
    parent_record = validate_parents(parent, behavior_parent)
    source_record = build_source_record()
    configs = enumerate_configs()
    planned = len(configs)
    if planned != 108:
        raise RuntimeError(f'Expected 108 missing fit cells, got {planned}')
    max_new_fits = planned if args.max_new_fits is None else args.max_new_fits
    if not 1 <= max_new_fits <= planned:
        parser.error(f'max-new-fits must be 1..{planned}')
    output = rr.checked_output(args.output)
    protected = (parent, behavior_parent, tr.DEFAULT_OUTPUT.resolve(), rr.DEFAULT_OUTPUT.resolve(),
                 tm.DEFAULT_OUTPUT.resolve(), tc.DEFAULT_OUTPUT.resolve())
    if any(output == root or output in root.parents or root in output.parents for root in protected):
        raise ValueError('Extension output must be a distinct results root; historical roots are read-only')
    reuse_index = dict(schema='explicit_condition_parent_reuse_index_v1',
        parents=parent_record, created_from_live_hash_validation=True)
    index_digest = config_hash(reuse_index)
    plan = make_plan(parent_record, source_record, index_digest)
    if args.dry_run:
        print(json.dumps(dict(planned=planned, new_fit_cells=planned,
            reused_completed_cells=312, conditions={name: dict(seeds=list(SEED_GRID[name]),
                fit_cells=sum(c == name for c, _ in configs)) for name in CONDITIONS},
            output_dimension=32, optimizer_updates=0, parent_plan_state_discrepancy=(
                parent_record['behavior_only']['plan_state_discrepancy']),
            output=str(output)), indent=2), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(output) if args.execute else None
    try:
        _write_sealed_files(output, plan, reuse_index)
        if args.qualify:
            qualify(output, source_record)
            return
        prepared = prepare_all(output)
        if not args.execute:
            print(json.dumps(dict(planned=planned, reused_completed_cells=312,
                new_fit_cells=planned, optimizer_updates=0, output=str(output)), indent=2), flush=True)
            return
        execute(output, configs, prepared, source_record, max_new_fits, torch.device(args.device))
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
