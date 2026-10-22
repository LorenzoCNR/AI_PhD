"""Matched TRAIN behavior-row shuffle for joint CNN/Transformer soft fits."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import torch
import numpy as np
from . import rat_runner as rr,transformer_runner as tr
from .data import TrialRows,load_rows,sha256,config_hash

DEFAULT_OUTPUT=rr.ROOT/'results'/'user_soft_rat_shuffle_v1'
ARCHITECTURES=('cnn','transformer')


def shuffle_train(rows,seed):
    """Exactly the replica's joint Nx3 row permutation; only TRAIN is allowed."""
    if rows.split!='train':raise ValueError('Only TRAIN behavior may be shuffled')
    permutation=np.random.default_rng(int(seed)+10000).permutation(len(rows.behavior))
    shuffled=TrialRows(neural=rows.neural,behavior=rows.behavior[permutation].copy(),
        source_id=rows.source_id,trial_id=rows.trial_id,time_id=rows.time_id,
        session_id=rows.session_id,split=rows.split)
    return shuffled,permutation


def frozen_config(architecture,rat,fold,seed,version,updates=10000):
    if architecture=='cnn':config=rr.frozen_config(rat,fold,seed,version,updates)
    elif architecture=='transformer':config=tr.frozen_config(rat,fold,seed,version,updates)
    else:raise ValueError('Unknown architecture')
    config.update(condition='position_direction_shuffled',shuffle_seed=seed+10000,
        shuffle_unit='complete_joint_behavior_row_withinTRAINrat_session',
        shuffle_target='TRAINbehavior_only;validation_and_exports_original',
        time_metadata_shuffled=False,training_metadata='joint_behavior_row_permutation_plus_original_local_time')
    return config


def build_configs(seed_count=7):
    if seed_count not in (5,7):raise ValueError('seed_count must be5 or7')
    return [(architecture,frozen_config(architecture,rat,fold,seed,version))
        for architecture,seed,fold,rat,version in itertools.product(ARCHITECTURES,range(42,42+seed_count),range(3),rr.RATS,rr.VERSIONS)]


def sources():
    return dict(wrapper_source_sha256=sha256(__file__),CNN_training_sources=rr.sources(),
        Transformer_training_sources=tr.sources(),original_shuffle_reference_sha256=sha256(rr.ROOT/'cebra_replica'/'rat_benchmark.py'),
        shuffle_function_sha256=rr.code_hash(shuffle_train),configuration_source_sha256=rr.code_hash(frozen_config))


def training_sources(architecture,source_record):
    base=dict(source_record['CNN_training_sources'] if architecture=='cnn' else source_record['Transformer_training_sources'])
    base.update(shuffle_wrapper_source_sha256=source_record['wrapper_source_sha256'],
        shuffle_function_sha256=source_record['shuffle_function_sha256'],
        shuffle_reference_sha256=source_record['original_shuffle_reference_sha256'])
    return base


def prepare_shuffle(output,rat,fold,seed):
    parent_folder,parent=rr.prepare(rr.DEFAULT_OUTPUT,rat,fold)
    original=load_rows(parent_folder/'train.npz')
    shuffled,permutation=shuffle_train(original,seed)
    path=output/'stage00_shuffle'/rat/f'fold_{fold}'/f'seed_{seed}'
    expected=dict(rat=rat,fold=fold,seed=seed,shuffle_seed=seed+10000,
        parent_cache_manifest=str(parent_folder/'manifest.json'),parent_cache_sha256=sha256(parent_folder/'manifest.json'),
        parent_train_npz_sha256=sha256(parent_folder/'train.npz'),
        permutation_sha256=config_hash(permutation.tolist()),source_function_sha256=rr.code_hash(shuffle_train),
        train_scales=parent['train_scales'],parent_scales_sha256=config_hash(parent['train_scales']),
        unit='joint_behavior_rows;TRAINonly;allcomponentsmovedtogether',
        neural_time_trial_source_ids_unchanged=True,validation_labels_original=True,export_labels_original=True,
        architecture_version_independent=True,no_scale_refit=True)
    manifest=path/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['signature']!=config_hash(expected):raise RuntimeError('Shuffle dependencies changed')
        if sha256(path/'TRAIN_BEHAVIOR_PERMUTATION.npz')!=record['permutation_file_sha256']:
            raise RuntimeError('Stored shuffle permutation hash mismatch')
        with np.load(path/'TRAIN_BEHAVIOR_PERMUTATION.npz',allow_pickle=False) as stored:
            if not np.array_equal(stored['permutation'],permutation):raise RuntimeError('Shuffle reproducibility mismatch')
        return parent_folder,parent,shuffled,permutation,record
    if path.exists() and any(path.iterdir()):raise RuntimeError('Partial shuffle preparation requires inspection')
    path.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(path/'TRAIN_BEHAVIOR_PERMUTATION.npz',permutation=permutation,
        recipient_source_id=original.source_id,behavior_donor_source_id=original.source_id[permutation],
        recipient_trial_id=original.trial_id,behavior_donor_trial_id=original.trial_id[permutation],
        session_id=original.session_id,seed=seed,shuffle_seed=seed+10000,split='train')
    record=dict(**expected,signature=config_hash(expected),created_utc=rr.now(),
        permutation_file_sha256=sha256(path/'TRAIN_BEHAVIOR_PERMUTATION.npz'))
    rr.atomic_json(manifest,record)
    return parent_folder,parent,shuffled,permutation,record


def diagnostic_cell(output,embedding_folder,original_train,shuffled,permutation,parent,config,shuffle_record):
    from .diagnostics import write_diagnostics
    from .downstream import _LazyArchive
    path=output/'stage04_diagnostics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    signature=config_hash(dict(parent_train_embedding=sha256(embedding_folder/'train_embedding.npz'),
        shuffle_signature=shuffle_record['signature'],source=rr.code_hash(diagnostic_cell),
        diagnostics_source=sha256(Path(__file__).with_name('diagnostics.py'))))
    complete=path/'STAGE_SIGNATURE.json'
    if complete.exists():
        record=json.loads(complete.read_text())
        if record['signature']!=signature:raise RuntimeError('Shuffle diagnostic dependencies changed')
        for name,digest in record['artifact_hashes'].items():
            if sha256(path/name)!=digest:raise RuntimeError('Shuffle diagnostic content mismatch')
        return
    data=_LazyArchive(embedding_folder/'train_embedding.npz')
    generator=torch.Generator().manual_seed(config['seed']+30000)
    indices=torch.randperm(len(data['embedding']),generator=generator)[:64].numpy()
    if not np.array_equal(data['source_id'],original_train.source_id):
        raise RuntimeError('Original embedding/TRAIN IDs mismatch')
    batch={key:torch.as_tensor(getattr(shuffled,key)[indices]) for key in ('behavior','time_id','trial_id','source_id','session_id')}
    batch['behavior_donor_source_id']=torch.as_tensor(original_train.source_id[permutation[indices]])
    batch['split']='train'
    write_diagnostics(path,torch.as_tensor(data['embedding'][indices]),batch,rr.SPECS,
        parent['train_scales'],config['version'],config['seed'])
    artifacts={p.name:sha256(p) for p in path.iterdir() if p.is_file()}
    rr.atomic_json(complete,dict(signature=signature,config=config,artifact_hashes=artifacts,
        original_train_row_indices=indices.tolist(),shuffle_record=shuffle_record,
        target_semantics='shuffledTRAINbehavior+originaltime;notoriginalexportbehavior'))


def write_tables(output,state):
    rows=[];compute=[]
    for _,entry in sorted(state['records'].items()):
        if entry.get('status')!='complete':continue
        metric=json.loads(Path(entry['metrics_path']).read_text());config=metric['config']
        for item in metric['metrics']:
            rows.append(dict(condition='position_direction_shuffled',architecture=config['architecture'],
                rat=config['rat'],fold=config['fold'],seed=config['seed'],method=config['version'],
                **{key:value for key,value in item.items() if key not in ('method','ridge_validation_candidates')}))
        fit=json.loads(Path(entry['fit_manifest_path']).read_text())
        embedding=json.loads(Path(entry['embedding_manifest_path']).read_text())
        compute.append(dict(condition='position_direction_shuffled',architecture=config['architecture'],
            rat=config['rat'],fold=config['fold'],seed=config['seed'],method=config['version'],
            **{name:fit.get(name) for name in ('parameters','actual_updates','actual_optimizer_updates','replayed_updates',
                'fit_seconds','seconds_per_100_updates','updates_per_second','training_windows_per_second',
                'peak_gpu_allocated','peak_gpu_reserved','device','cuda_available','torch_version')},
            inference_seconds=embedding.get('inference_seconds'),inference_windows_per_second=embedding.get('inference_windows_per_second')))
    for name,data in (('RAT_SHUFFLE_COMPARISON.csv',rows),('COMPUTE_PERFORMANCE.csv',compute)):
        if not data:continue
        path=output/name;tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        tmp.replace(path)


def prerequisites(transformer_output=None,transformer_fit_count=336):
    if transformer_fit_count not in (240,336):raise ValueError('Transformer prerequisite count must be240 or336')
    transformer_output=Path(transformer_output) if transformer_output is not None else rr.ROOT/'results'/'user_soft_rat_transformer_campaign_v1'
    for root,count in ((rr.ROOT/'results'/'user_soft_rat_time_only_v1',168),(transformer_output,transformer_fit_count)):
        name=root.name
        path=root/'CAMPAIGN_STATE.json'
        if not path.exists():raise RuntimeError(f'Previous campaign incomplete: {name}')
        state=json.loads(path.read_text())
        complete_records=sum(record.get('status')=='complete' for record in state.get('records',{}).values())
        if (state.get('completed')!=count or state.get('status') not in ('complete','requested_limit')
                or complete_records!=count or (path.parent/'RUNNING.lock').exists()):
            raise RuntimeError(f'Previous campaign must finish before shuffle: {name}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--seed-count',type=int,choices=(5,7),default=7)
    parser.add_argument('--transformer-output',type=Path)
    parser.add_argument('--transformer-fit-count',type=int,choices=(240,336),default=336)
    parser.add_argument('--max-new-fits',type=int)
    args=parser.parse_args()
    planned=48*args.seed_count
    max_new_fits=planned if args.max_new_fits is None else args.max_new_fits
    if not 1<=max_new_fits<=planned:parser.error(f'max-new-fits must be1..{planned}')
    output=rr.checked_output(args.output)
    for protected in (rr.DEFAULT_OUTPUT,tr.DEFAULT_OUTPUT,rr.ROOT/'results'/'user_soft_rat_time_only_v1',rr.ROOT/'results'/'user_soft_rat_transformer_campaign_v1'):
        if output==protected.resolve() or protected.resolve() in output.parents:
            raise ValueError('Shuffle study must use separate output root')
    output.mkdir(parents=True,exist_ok=True)
    source_record=sources();configs=build_configs(args.seed_count)
    predecessor=args.transformer_output or rr.ROOT/'results'/'user_soft_rat_transformer_campaign_v1'
    plan=dict(created_utc=rr.now(),planned=planned,seed_count=args.seed_count,configs=[dict(architecture=a,config=c) for a,c in configs],
        transformer_predecessor=dict(path=str(predecessor.resolve()),completed_fits_required=args.transformer_fit_count),
        sources=source_record,no_time_only_redundant_shuffle=True,permutation_shared_across_architectures_versions=True,
        no_test_outcome_selection=True,optimizer_updates_in_dry_run=0)
    path=output/'CAMPAIGN_PLAN.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['configs']!=plan['configs'] or old['sources']!=source_record or old.get('transformer_predecessor')!=plan['transformer_predecessor']:raise RuntimeError('Sealed shuffle plan changed')
    else:rr.atomic_json(path,plan)
    prepared={(rat,fold,seed):prepare_shuffle(output,rat,fold,seed)
        for seed,fold,rat in itertools.product(range(42,42+args.seed_count),range(3),rr.RATS)}
    if not args.execute:
        print(json.dumps(dict(planned=planned,CNN=planned//2,Transformer=planned//2,stored_TRAIN_permutations=12*args.seed_count,optimizer_updates=0,output=str(output)),indent=2),flush=True)
        return
    gate_path=output/'qualification'/'GATE.json'
    if not gate_path.exists():raise RuntimeError('Shuffle qualification gate missing')
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_record):
        raise RuntimeError('Shuffle qualification stale/NO-GO')
    prerequisites(args.transformer_output,args.transformer_fit_count)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock=output/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=planned,records={},created_utc=rr.now())
    new_fits=0
    try:
        for architecture,config in configs:
            key=f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
            state_key=architecture+'/'+key;branch=output/architecture
            data_folder,parent,shuffled,permutation,shuffle_record=prepared[(config['rat'],config['fold'],config['seed'])]
            folder=branch/'stage01_fit'/key
            state.update(current=state_key,status='running',updated_utc=rr.now());rr.atomic_json(state_path,state)
            valid=load_rows(data_folder/'valid.npz')
            if architecture=='transformer':
                with tr.architecture_bindings():
                    fit,created=rr.fit_cell(folder,shuffled,valid,config,parent['train_scales'],
                        config_hash(shuffle_record),training_sources(architecture,source_record),device)
            else:
                fit,created=rr.fit_cell(folder,shuffled,valid,config,parent['train_scales'],
                    config_hash(shuffle_record),training_sources(architecture,source_record),device)
            new_fits+=int(created)
            state['records'][state_key]=dict(status='fit_complete',architecture=config['architecture'],
                fit_manifest_path=str(folder/'manifest.json'),checkpoint_sha256=fit['checkpoint_sha256'],
                shuffle_signature=shuffle_record['signature'])
            rr.atomic_json(state_path,state)
            exporter=tr.export_cell if architecture=='transformer' else rr.export_cell
            embedding_folder,embedding_record=exporter(branch,folder,data_folder,config,fit,device)
            rr.decode_cell(branch,embedding_folder,config,embedding_record)
            diagnostic_cell(branch,embedding_folder,load_rows(data_folder/'train.npz'),shuffled,permutation,parent,config,shuffle_record)
            metric_path=branch/'stage03_metrics'/key/'metrics.json'
            state['records'][state_key].update(status='complete',fit_manifest_sha256=sha256(folder/'manifest.json'),
                metrics_path=str(metric_path),metrics_sha256=sha256(metric_path),embedding_manifest_path=str(embedding_folder/'manifest.json'),
                embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'))
            state['completed']=sum(r.get('status')=='complete' for r in state['records'].values())
            rr.atomic_json(state_path,state);write_tables(output,state)
            print(json.dumps(dict(completed=state['completed'],planned=planned,last=state_key,fit_seconds=fit['fit_seconds'])),flush=True)
            if new_fits>=max_new_fits:
                state['status']='requested_limit';break
        else:state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',last_error=repr(exc))
        if state.get('current'):state['records'].setdefault(state['current'],{}).update(error=repr(exc))
        raise
    finally:
        state['updated_utc']=rr.now();rr.atomic_json(state_path,state);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
