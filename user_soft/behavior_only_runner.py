"""Seven-seed behavior-only stable Soft campaign; existing fits remain immutable."""
from __future__ import annotations
import argparse
import copy
import csv
import io
import itertools
import json
import math
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
import unittest
import numpy as np
import torch
from . import rat_runner as rr, transformer_runner as tr
from .data import config_hash, load_rows, sha256
from .loss import evaluate_soft

DEFAULT_OUTPUT = rr.ROOT / 'results' / 'user_soft_rat_behavior_only_v1'
ARCHITECTURES = ('cnn', 'transformer')
SEEDS = tuple(range(42, 49))
SPECS = [dict(key='behavior', geometry='euclidean', weight=1.)]
VERSION = 'soft_v1_stable'


def frozen_config(architecture, rat, fold, seed, updates=10000):
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown architecture')
    factory = rr.frozen_config if architecture == 'cnn' else tr.frozen_config
    config = factory(rat, fold, seed, VERSION, updates)
    config.update(condition='behavior_only', metadata_specs=copy.deepcopy(SPECS),
                  metadata_supervision='joint_raw_position_right_left_no_time',
                  behavior_coordinate_weights=[1., 1., 1.],
                  behavior_scale_policy='project_existing_full_TRAIN_behavior_bound_no_refit')
    if architecture == 'transformer':
        config['concurrency_context'] = 'sequential behavior-only campaign; no concurrent campaign launched'
    return config


def enumerate_configs():
    return [(architecture, frozen_config(architecture, rat, fold, seed))
            for seed, rat, fold, architecture in itertools.product(SEEDS, rr.RATS, range(3), ARCHITECTURES)]


@contextmanager
def context_bindings(architecture):
    original = rr.SPECS
    rr.SPECS = SPECS
    try:
        with ExitStack() as stack:
            if architecture == 'transformer':
                stack.enter_context(tr.architecture_bindings())
            elif architecture != 'cnn':
                raise ValueError('Unknown architecture')
            yield
    finally:
        rr.SPECS = original


def project_scales(parent_scales):
    if parent_scales.get('split') != 'train':
        raise ValueError('Existing TRAIN scales required')
    selected = [s for s in parent_scales['scales'] if s['key'] == 'behavior' and s['geometry'] == 'euclidean']
    if len(selected) != 1:
        raise ValueError('Exactly one existing behavioral TRAIN scale required')
    result = copy.deepcopy(parent_scales)
    result.update(scales=copy.deepcopy(selected), specs=copy.deepcopy(SPECS),
                  parent_scale_sha256=config_hash(parent_scales),
                  projection='retain existing TRAIN behavior bound; no refit; no temporal contribution')
    return result


def sources():
    if rr.SPECS == SPECS:
        raise RuntimeError('Compute signatures outside behavior-only bindings')
    return dict(wrapper_sha256=sha256(__file__), metadata_specs=copy.deepcopy(SPECS),
                parent_cnn=rr.sources(), parent_transformer=tr.sources())


def prepare(output, rat, fold):
    if rr.SPECS == SPECS:
        raise RuntimeError('Validate parent data outside behavior-only binding')
    parent_manifest = rr.DEFAULT_OUTPUT / 'stage00_data' / rat / f'fold_{fold}' / 'manifest.json'
    if not parent_manifest.is_file():
        raise RuntimeError(f'Expected reusable prepared cache missing: {parent_manifest}')
    data_folder, parent = rr.prepare(rr.DEFAULT_OUTPUT, rat, fold)
    record = dict(condition='behavior_only', rat=rat, fold=fold,
                  parent_cache_manifest=str(parent_manifest), parent_cache_sha256=sha256(parent_manifest),
                  parent_cache_signature=parent['signature'], splits=parent['splits'],
                  train_scales=project_scales(parent['train_scales']), metadata_specs=copy.deepcopy(SPECS),
                  no_scale_refit=True, no_test_outcome_selection=True)
    record['signature'] = config_hash(record)
    for architecture in ARCHITECTURES:
        path = output / architecture / 'stage00_data' / rat / f'fold_{fold}' / 'manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise RuntimeError('Prepared behavior-only parent changed; refuse overwrite')
        else:
            rr.atomic_json(path, record)
    return data_folder, record


def diagnostic_cell(output, embedding_folder, data_manifest, config):
    """Interpret the existing objective on fixed TRAIN/validation evaluation batches."""
    from .downstream import _LazyArchive
    path = output / 'stage04_diagnostics' / config['rat'] / f"fold_{config['fold']}" / VERSION / f"seed_{config['seed']}"
    complete = path / 'STAGE_SIGNATURE.json'
    parent = json.loads((embedding_folder / 'manifest.json').read_text())
    signature = config_hash(dict(source=sha256(__file__), config=config,
        train=sha256(embedding_folder / 'train_embedding.npz'),
        valid=sha256(embedding_folder / 'valid_embedding.npz'), scales=data_manifest['train_scales']))
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['signature'] != signature:
            raise RuntimeError('Behavior-only diagnostic signature changed')
        for name, digest in saved['artifact_hashes'].items():
            if sha256(path / name) != digest:
                raise RuntimeError('Behavior-only diagnostic hash mismatch')
        return
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'Partial diagnostic artifacts require inspection: {path}')
    generator = torch.Generator().manual_seed(config['seed'] + 20000)
    records = []
    for split in ('train', 'valid'):
        data = _LazyArchive(embedding_folder / f'{split}_embedding.npz')
        ids = torch.randint(len(data['embedding']), (config['batch_size'],), generator=generator).numpy()
        batch = {key: torch.as_tensor(data[key][ids]) for key in ('behavior', 'trial_id', 'source_id', 'time_id', 'session_id')}
        with torch.no_grad():
            result = evaluate_soft(torch.as_tensor(data['embedding'][ids]), batch, SPECS, version=VERSION,
                scales=data_manifest['train_scales'], temperature=config['embedding_temperature'],
                metadata_temperature=config['metadata_temperature'])
        entropy = result.diagnostics()['Q_entropy_mean']
        records.append(dict(split=split, cross_entropy=float(result.loss), target_entropy=entropy,
            kl_gap=float(result.loss) - entropy, uniform_prediction_baseline=math.log(config['batch_size'] - 1),
            evaluation_seed=config['seed'] + 20000, **result.diagnostics()))
        if split == 'train':
            reference_ids = torch.randperm(len(data['embedding']), generator=torch.Generator().manual_seed(config['seed'] + 30000))[:64].numpy()
            reference_batch = {key: torch.as_tensor(data[key][reference_ids]) for key in batch}
            with torch.no_grad():
                reference = evaluate_soft(torch.as_tensor(data['embedding'][reference_ids]), reference_batch, SPECS,
                    scales=data_manifest['train_scales'], temperature=config['embedding_temperature'],
                    metadata_temperature=config['metadata_temperature'])
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / 'TRAIN_REFERENCE_W_Q_P.npz', W=reference.W.numpy(), Q=reference.Q.numpy(),
        P=reference.P.numpy(), reference_train_row_indices=reference_ids,
        **{key: value.numpy() for key, value in reference_batch.items()})
    rr.atomic_json(path / 'LOSS_INTERPRETATION.json', dict(config=config, records=records,
        interpretation='Cross-entropy = target entropy + KL. Diagnostic only, no selection.',
        evaluation_batch_policy='same RNG seed and TRAIN then VALID draws as parent fit diagnostics'))
    rr.atomic_json(path / 'PROVENANCE.json', dict(created_utc=rr.now(), config=config,
        source_sha256=sha256(__file__), parent_embedding_manifest_sha256=sha256(embedding_folder / 'manifest.json'),
        parent_checkpoint_sha256=parent['parent_checkpoint_sha256'], train_scales=data_manifest['train_scales'],
        input_splits=['train', 'valid'], no_test_data_read_by_diagnostics=True))
    rr.atomic_json(complete, dict(signature=signature,
        artifact_hashes={p.name: sha256(p) for p in path.iterdir() if p.is_file()}))


def write_csv(path, records):
    if not records:
        return
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def write_tables(output, state):
    metrics, compute, diagnostics = [], [], []
    for key, entry in sorted(state['records'].items()):
        if entry.get('status') != 'complete':
            continue
        fit = json.loads(Path(entry['fit_manifest_path']).read_text())
        config = fit['config']
        common = dict(architecture=entry['architecture'], condition='behavior_only', rat=config['rat'],
                      fold=config['fold'], seed=config['seed'], version=VERSION)
        scored = json.loads(Path(entry['metrics_path']).read_text())
        metrics.extend(dict(**common, **{k: v for k, v in row.items() if k not in ('method', 'ridge_validation_candidates')})
                       for row in scored['metrics'])
        embedding = json.loads(Path(entry['embedding_manifest_path']).read_text())
        compute.append(dict(**common, **{name: fit.get(name) for name in
            ('parameters', 'actual_updates', 'actual_optimizer_updates', 'replayed_updates', 'fit_seconds',
             'seconds_per_100_updates', 'updates_per_second', 'training_windows_per_second',
             'peak_gpu_allocated', 'peak_gpu_reserved', 'device', 'cuda_available', 'torch_version')},
            inference_seconds=embedding['inference_seconds'], inference_windows_per_second=embedding['inference_windows_per_second']))
        interpreted = json.loads(Path(entry['diagnostics_path']).read_text())
        diagnostics.extend(dict(**common, **row) for row in interpreted['records'])
    write_csv(output / 'DECODING_LONG.csv', metrics)
    write_csv(output / 'COMPUTE_PERFORMANCE.csv', compute)
    write_csv(output / 'LOSS_DIAGNOSTICS.csv', diagnostics)


def seal_plan(output, source_record):
    configs = enumerate_configs()
    content = dict(planned=len(configs), seeds=list(SEEDS), version=VERSION, metadata_specs=copy.deepcopy(SPECS),
        configs=[dict(architecture=a, config=c) for a, c in configs], sources=source_record,
        parent_prepared_data_root=str(rr.DEFAULT_OUTPUT / 'stage00_data'), no_existing_campaign_resume=True,
        selection='fixed terminal checkpoint; no HPO; no test-based selection')
    path = output / 'CAMPAIGN_PLAN.json'
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise RuntimeError('Behavior-only sealed plan changed; do not overwrite existing study')
    else:
        rr.atomic_json(path, content)
    return configs


def qualify(output, source_record):
    suite = unittest.defaultTestLoader.loadTestsFromName('user_soft.tests.test_behavior_only')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = dict(created_utc=rr.now(), decision='GO' if result.wasSuccessful() and result.testsRun else 'NO-GO',
        training_source_signature=config_hash(source_record), tests_run=result.testsRun,
        tests_failed=len(result.failures), tests_errors=len(result.errors),
        test_source_sha256=sha256(Path(__file__).parent / 'tests' / 'test_behavior_only.py'),
        scope='CPU toy qualification only; zero rat fits; existing frozen pipeline reused')
    qualification = output / 'qualification'
    rr.atomic_json(qualification / 'TEST_RESULTS.json', dict(**record, output=stream.getvalue()))
    rr.atomic_json(qualification / 'GATE.json', record)
    print(stream.getvalue(), flush=True)
    print(json.dumps(record), flush=True)
    if not result.wasSuccessful() or not result.testsRun:
        raise RuntimeError('Behavior-only CPU qualification failed')


def execute(output, configs, prepared, source_record, device, max_new_fits, report_every):
    gate_path = output / 'qualification' / 'GATE.json'
    gate = json.loads(gate_path.read_text())
    if gate.get('decision') != 'GO' or gate.get('training_source_signature') != config_hash(source_record):
        raise RuntimeError('Behavior-only source qualification stale/NO-GO')
    if gate['test_source_sha256'] != sha256(Path(__file__).parent / 'tests' / 'test_behavior_only.py'):
        raise RuntimeError('Qualification test source changed')
    for root, expected in ((rr.DEFAULT_OUTPUT, rr.sources()), (tr.DEFAULT_OUTPUT, tr.sources())):
        parent = json.loads((root / 'qualification' / 'GATE.json').read_text())
        if parent.get('decision') != 'GO' or parent.get('training_source_signature') != config_hash(expected):
            raise RuntimeError('Existing parent qualification stale/NO-GO')
    lock = output / 'RUNNING.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    state_path = output / 'CAMPAIGN_STATE.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(records={}, completed=0, planned=len(configs), created_utc=rr.now())
    new_fits = 0
    try:
        for architecture, config in configs:
            if (output / 'STOP_AFTER_CURRENT').exists():
                state['status'] = 'stopped_after_current'
                break
            key = f"{architecture}/{config['rat']}/fold_{config['fold']}/{VERSION}/seed_{config['seed']}"
            root = output / architecture
            cell = f"{config['rat']}/fold_{config['fold']}/{VERSION}/seed_{config['seed']}"
            folder = root / 'stage01_fit' / cell
            data_folder, data_manifest = prepared[(config['rat'], config['fold'])]
            state.update(status='running', current=key, pid=os.getpid(), updated_utc=rr.now())
            rr.atomic_json(state_path, state)
            train = load_rows(data_folder / 'train.npz'); valid = load_rows(data_folder / 'valid.npz')
            with context_bindings(architecture):
                fit, created = rr.fit_cell(folder, train, valid, config, data_manifest['train_scales'],
                    config_hash(data_manifest), source_record, device)
                new_fits += int(created)
                state['records'][key] = dict(status='fit_complete', architecture=architecture,
                    checkpoint_sha256=fit['checkpoint_sha256'], fit_manifest_path=str(folder / 'manifest.json'))
                rr.atomic_json(state_path, state)
                exporter = rr.export_cell if architecture == 'cnn' else tr.export_cell
                embedding_folder, embedding_record = exporter(root, folder, data_folder, config, fit, device)
            rr.decode_cell(root, embedding_folder, config, embedding_record)
            diagnostic_cell(root, embedding_folder, data_manifest, config)
            diagnostic_path = root / 'stage04_diagnostics' / cell / 'LOSS_INTERPRETATION.json'
            state['records'][key].update(status='complete', fit_manifest_sha256=sha256(folder / 'manifest.json'),
                embedding_manifest_path=str(embedding_folder / 'manifest.json'),
                embedding_manifest_sha256=sha256(embedding_folder / 'manifest.json'),
                metrics_path=str(root / 'stage03_metrics' / cell / 'metrics.json'), diagnostics_path=str(diagnostic_path))
            state['completed'] = sum(r.get('status') == 'complete' for r in state['records'].values())
            state['updated_utc'] = rr.now(); rr.atomic_json(state_path, state)
            write_tables(output, state)
            if state['completed'] == 1 or state['completed'] % report_every == 0 or state['completed'] == len(configs):
                print(json.dumps(dict(completed=state['completed'], planned=len(configs), last=key,
                    last_fit_seconds=fit['fit_seconds'], updated_utc=rr.now())), flush=True)
            if max_new_fits is not None and new_fits >= max_new_fits:
                state['status'] = 'requested_limit'
                break
        else:
            state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', last_error=repr(exc))
        raise
    finally:
        state['updated_utc'] = rr.now(); rr.atomic_json(state_path, state)
        lock.unlink(missing_ok=True)
    print(json.dumps(dict(status=state['status'], completed=state['completed'], planned=len(configs))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--qualify', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--max-new-fits', type=int)
    parser.add_argument('--report-every', type=int, default=20)
    args = parser.parse_args()
    if args.max_new_fits is not None and not 1 <= args.max_new_fits <= 168:
        parser.error('max-new-fits must be1..168')
    if args.report_every < 1:
        parser.error('report-every must be positive')
    output = rr.checked_output(args.output)
    if not output.name.startswith('user_soft_rat_behavior_only'):
        raise ValueError('Output must be a new behavior-only study root')
    output.mkdir(parents=True, exist_ok=True)
    source_record = sources()
    configs = seal_plan(output, source_record)
    if args.qualify:
        qualify(output, source_record)
        return
    prepared = {(rat, fold): prepare(output, rat, fold) for rat, fold in itertools.product(rr.RATS, range(3))}
    if not args.execute:
        record = dict(planned=len(configs), seeds=list(SEEDS), architectures=list(ARCHITECTURES),
            version=VERSION, metadata_specs=SPECS, temporal_weight=0., optimizer_updates=0,
            reused_parent_caches=len(prepared), parent_manifest_hashes={f'{rat}/fold_{fold}': m['parent_cache_sha256']
                for (rat, fold), (_, m) in prepared.items()}, source_signature=config_hash(source_record),
            output=str(output), existing_campaigns_unchanged=True)
        rr.atomic_json(output / 'DRY_RUN.json', record)
        print(json.dumps(record), flush=True)
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = False
    execute(output, configs, prepared, source_record, torch.device(args.device), args.max_new_fits, args.report_every)


if __name__ == '__main__':
    main()
