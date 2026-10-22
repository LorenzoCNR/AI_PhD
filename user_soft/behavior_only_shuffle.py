"""Matched behavior-only TRAIN shuffle; wait for the six-seed observed campaign."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import unittest
import io
import torch
from . import behavior_only_runner as b, shuffle_runner as sh, rat_runner as rr, transformer_runner as tr
from .behavior_only_six_seed import limited_configs
from .data import config_hash, sha256, load_rows

OUTPUT = rr.ROOT / 'results' / 'user_soft_rat_behavior_only_shuffle_v1'


def configs():
    result = []
    for architecture, original in limited_configs(b.enumerate_configs()):
        config = copy.deepcopy(original)
        config.update(condition='behavior_only_shuffled', shuffle_seed=config['seed'] + 10000,
            shuffle_unit='complete_joint_behavior_row_withinTRAINrat_session',
            shuffle_target='TRAINbehavior_only;validation_and_exports_original',
            time_metadata_shuffled=False)
        result.append((architecture, config))
    return result


def ready(state, locked):
    return (not locked and state.get('status') == 'complete' and state.get('completed') == 144
            and sum(r.get('status') == 'complete' for r in state.get('records', {}).values()) == 144)


def source_record():
    return dict(launcher_sha256=sha256(__file__), behavior_sources=b.sources(),
                shuffle_sources=sh.sources())


def audit_existing():
    path = rr.ROOT/'results/user_soft_rat_overnight_v1/CAMPAIGN_STATE.json'
    state = json.loads(path.read_text())
    checked = []
    for key, entry in state['records'].items():
        if not key.startswith(('shuffle_cnn/', 'shuffle_transformer/')) or entry.get('seed') not in range(42,48):
            continue
        if entry.get('status') != 'complete':
            raise RuntimeError('Incomplete existing matched shuffle: '+key)
        for component in ('fit_manifest','embedding_manifest','metrics'):
            if sha256(Path(entry[component+'_path'])) != entry[component+'_sha256']:
                raise RuntimeError('Existing shuffle artifact hash mismatch: '+key)
        fit_path = Path(entry['fit_manifest_path'])
        fit = json.loads(fit_path.read_text())
        if sha256(fit_path.parent/'checkpoint.pt') != entry['checkpoint_sha256']:
            raise RuntimeError('Existing shuffle checkpoint hash mismatch: '+key)
        checked.append(dict(id=key, checkpoint_sha256=entry['checkpoint_sha256'],
            fit_manifest_path=str(fit_path), fit_manifest_sha256=entry['fit_manifest_sha256']))
    if len(checked) != 288:
        raise RuntimeError(f'Expected288 existing matched joint shuffles, found{len(checked)}')
    record = dict(existing_joint_shuffles=checked, verified_count=288, matched_seeds=list(range(42,48)),
        missing_behavior_only_fits=144, time_only_behavior_shuffle='NOT_APPLICABLE: behavior absent from objective',
        existing_parent_manifest_sha256=sha256(path), source_sha256=sha256(__file__),
        embedding_payloads='not rehashed; embedding manifests checked against indexed hashes', created_utc=rr.now())
    rr.atomic_json(OUTPUT/f'EXISTING_SHUFFLE_AUDIT_{time.time_ns()}.json',record)


def diagnostic(branch, embedding_folder, original, shuffled, permutation, parent, config, record):
    from .diagnostics import write_diagnostics
    from .downstream import _LazyArchive
    import numpy as np
    path = branch/'stage04_diagnostics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    signature = config_hash(dict(embedding=sha256(embedding_folder/'train_embedding.npz'),
        shuffle=record['signature'], source=sha256(__file__)))
    marker = path/'STAGE_SIGNATURE.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['signature'] != signature or any(sha256(path/n)!=h for n,h in saved['artifact_hashes'].items()):
            raise RuntimeError('Shuffle diagnostic provenance mismatch')
        return
    data = _LazyArchive(embedding_folder/'train_embedding.npz')
    if not np.array_equal(data['source_id'],original.source_id):
        raise RuntimeError('Shuffle diagnostic source rows mismatch')
    ids = torch.randperm(len(data['embedding']),generator=torch.Generator().manual_seed(config['seed']+30000))[:64].numpy()
    batch = {k:torch.as_tensor(getattr(shuffled,k)[ids]) for k in ('behavior','time_id','trial_id','source_id','session_id')}
    batch.update(split='train',behavior_donor_source_id=torch.as_tensor(original.source_id[permutation[ids]]))
    write_diagnostics(path,torch.as_tensor(data['embedding'][ids]),batch,b.SPECS,
        parent['train_scales'],config['version'],config['seed'])
    rr.atomic_json(marker,dict(signature=signature,config=config,shuffle_record=record,
        target_semantics='shuffled TRAIN behavior ONLY; time has zero contribution',
        artifact_hashes={p.name:sha256(p) for p in path.iterdir() if p.is_file()}))


def qualify():
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'user_soft.tests.test_behavior_only_shuffle', 'user_soft.tests.test_shuffle',
        'user_soft.tests.test_behavior_only'])
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    gate = dict(decision='GO' if result.wasSuccessful() else 'NO-GO',
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        sources=source_record(), test_hash=sha256(Path(__file__).parent/'tests/test_behavior_only_shuffle.py'),
        created_utc=rr.now(), output=stream.getvalue())
    rr.atomic_json(OUTPUT/'qualification/GATE.json', gate)
    print(stream.getvalue(), flush=True)
    if not result.wasSuccessful():
        raise RuntimeError('Behavior-only shuffle qualification failed')


def run():
    gate = json.loads((OUTPUT/'qualification/GATE.json').read_text())
    sources = source_record()
    if gate['decision'] != 'GO' or gate['sources'] != sources:
        raise RuntimeError('Shuffle qualification missing or stale')
    if gate['test_hash'] != sha256(Path(__file__).parent/'tests/test_behavior_only_shuffle.py'):
        raise RuntimeError('Shuffle tests changed')
    parent = b.DEFAULT_OUTPUT
    if not ready(json.loads((parent/'CAMPAIGN_STATE.json').read_text()), (parent/'RUNNING.lock').exists()):
        raise RuntimeError('Observed campaign not complete')
    if json.loads((parent/'CAMPAIGN_PLAN.json').read_text())['sources'] != b.sources():
        raise RuntimeError('Observed sealed sources changed')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    lock = OUTPUT/'RUNNING.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    state_path = OUTPUT/'CAMPAIGN_STATE.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(planned=144, records={})
    try:
        for architecture, config in configs():
            if (OUTPUT/'STOP_AFTER_CURRENT').exists():
                state['status'] = 'stopped_after_current'
                break
            cell = f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
            key = architecture+'/'+cell
            branch = OUTPUT/architecture
            state.update(status='running', current=key, pid=os.getpid(), updated_utc=rr.now())
            rr.atomic_json(state_path, state)
            folder, original_parent, shuffled, permutation, shuffle_record = sh.prepare_shuffle(
                OUTPUT, config['rat'], config['fold'], config['seed'])
            scales = b.project_scales(original_parent['train_scales'])
            dependency = config_hash(dict(shuffle=shuffle_record['signature'], scales=scales))
            fit_folder = branch/'stage01_fit'/cell
            with b.context_bindings(architecture):
                fit, _ = rr.fit_cell(fit_folder, shuffled, load_rows(folder/'valid.npz'),
                    config, scales, dependency, sources, device)
                state['records'][key] = dict(status='fit_complete', architecture=architecture,
                    fit_manifest_path=str(fit_folder/'manifest.json'), checkpoint_sha256=fit['checkpoint_sha256'])
                rr.atomic_json(state_path, state)
                exporter = rr.export_cell if architecture == 'cnn' else tr.export_cell
                embedding_folder, embedding_record = exporter(branch, fit_folder, folder, config, fit, device)
                rr.decode_cell(branch, embedding_folder, config, embedding_record)
                diagnostic(branch, embedding_folder, load_rows(folder/'train.npz'), shuffled,
                    permutation, dict(original_parent, train_scales=scales), config, shuffle_record)
            state['records'][key].update(status='complete',
                fit_manifest_sha256=sha256(fit_folder/'manifest.json'),
                embedding_manifest_path=str(embedding_folder/'manifest.json'),
                embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'),
                metrics_path=str(branch/'stage03_metrics'/cell/'metrics.json'),
                shuffle_manifest_path=str(OUTPUT/'stage00_shuffle'/config['rat']/f"fold_{config['fold']}"/f"seed_{config['seed']}"/'manifest.json'))
            state['completed'] = sum(r.get('status') == 'complete' for r in state['records'].values())
            rr.atomic_json(state_path, state)
            rows = []
            compute = []
            for k, entry in sorted(state['records'].items()):
                if entry['status'] != 'complete':
                    continue
                metrics = json.loads(Path(entry['metrics_path']).read_text())
                c = metrics['config']
                common = dict(architecture=k.split('/')[0], condition='behavior_only_shuffled',
                    rat=c['rat'], fold=c['fold'], seed=c['seed'], version=c['version'])
                rows.extend(dict(**common, **{n:v for n,v in row.items() if n not in ('method','ridge_validation_candidates')}) for row in metrics['metrics'])
                f = json.loads(Path(entry['fit_manifest_path']).read_text())
                e = json.loads(Path(entry['embedding_manifest_path']).read_text())
                compute.append(dict(**common, **{n:f.get(n) for n in ('parameters','actual_updates','actual_optimizer_updates','fit_seconds','seconds_per_100_updates','updates_per_second','training_windows_per_second','peak_gpu_allocated','peak_gpu_reserved','device','cuda_available','torch_version')}, inference_seconds=e['inference_seconds'], inference_windows_per_second=e['inference_windows_per_second']))
            b.write_csv(OUTPUT/'DECODING_LONG.csv', rows)
            b.write_csv(OUTPUT/'COMPUTE_PERFORMANCE.csv', compute)
            if state['completed'] % 20 == 0 or state['completed'] == 144:
                print(json.dumps(dict(completed=state['completed'], planned=144, last=key)), flush=True)
        else:
            state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', last_error=repr(exc))
        raise
    finally:
        state['updated_utc'] = rr.now()
        rr.atomic_json(state_path, state)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--wait-and-execute', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan = dict(configs=[dict(architecture=a,config=c) for a,c in configs()], sources=source_record(),
        planned=144, observed_parent=str(b.DEFAULT_OUTPUT), no_time_target=True,
        shuffle_definition='existing TRAIN joint-row permutation; original validation/test/export labels; no scale refit')
    path = OUTPUT/'CAMPAIGN_PLAN.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError('Sealed shuffle plan changed')
    if not path.exists():
        rr.atomic_json(path, plan)
    if args.qualify:
        qualify()
        audit_existing()
    if not args.wait_and_execute:
        return
    queue_lock = OUTPUT/'QUEUE.lock'
    fd = os.open(queue_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        while True:
            parent = b.DEFAULT_OUTPUT
            state = json.loads((parent/'CAMPAIGN_STATE.json').read_text())
            rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json', dict(status='waiting_observed',
                parent_completed=state.get('completed'), required=144, pid=os.getpid(), updated_utc=rr.now()))
            if ready(state, (parent/'RUNNING.lock').exists()):
                break
            if state.get('status') in ('failed','interrupted','stopped_after_current'):
                raise RuntimeError('Observed campaign stopped; no shuffle launched')
            time.sleep(30)
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json', dict(status='running_shuffle', pid=os.getpid(), updated_utc=rr.now()))
        run()
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json', dict(status='finished', updated_utc=rr.now()))
    finally:
        queue_lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
