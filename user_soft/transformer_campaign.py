"""Seven-seed joint/time Transformer extension with exact initial-fit reuse."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import torch
from . import rat_runner as rr,transformer_runner as tr,time_only_runner as tm
from .data import sha256,config_hash,load_rows

DEFAULT_OUTPUT=rr.ROOT/'results'/'user_soft_rat_transformer_campaign_v1'
CONDITIONS=('joint','time_only')


def config_for(rat,fold,seed,version,condition):
    config=tr.frozen_config(rat,fold,seed,version)
    if condition=='time_only':
        config.update(condition='time_only',metadata_specs=tm.SPECS,
                      temporal_coordinate='official trial-relative25msbin',
                      metadata_supervision='local_time_only_no_behavior')
    elif condition!='joint':raise ValueError('unknown Transformer condition')
    return config


def enumerate_configs(seed_count=7):
    if seed_count not in (5,7):raise ValueError('seed_count must be5 or7')
    return [(condition,config_for(rat,fold,seed,version,condition))
        for condition,seed,fold,rat,version in itertools.product(CONDITIONS,range(42,42+seed_count),range(3),rr.RATS,rr.VERSIONS)]


def sources():
    return dict(wrapper_source_sha256=sha256(__file__),joint_training_sources=tr.sources(),
                time_projection_source=rr.code_hash(tm.project_scales),
                time_binding_source=rr.code_hash(tm.context_bindings),
                time_metadata_specs=tm.SPECS,configuration_source=rr.code_hash(config_for))


def training_sources(condition,record):
    if condition=='joint':return record['joint_training_sources']
    return dict(**record['joint_training_sources'],
                time_only_metadata_specification=config_hash(tm.SPECS),
                time_only_configuration_source=record['configuration_source'],
                time_only_projection_source=record['time_projection_source'],
                time_only_binding_source=record['time_binding_source'])


def artifact_root(output,condition,config):
    """Original first-block artifacts retain their exact immutable identity."""
    if condition=='joint' and config['seed']==42 and config['fold']==0:
        key=f"{config['rat']}/fold_0/{config['version']}/seed_42"
        folder=tr.DEFAULT_OUTPUT/'stage01_fit'/key
        if (folder/'checkpoint.pt').exists() or (folder/'manifest.json').exists():
            return tr.DEFAULT_OUTPUT
    return output/condition


def prepare(output):
    prepared={}
    for condition,rat,fold in itertools.product(CONDITIONS,rr.RATS,range(3)):
        if condition=='joint':
            folder,record=tr.prepare(output/'joint',rat,fold)
        else:
            folder,record=tm.prepare(output/'time_only',rat,fold)
        prepared[(condition,rat,fold)]=(folder,record)
    return prepared


def write_tables(output,state):
    rows=[];compute=[]
    for key,entry in sorted(state['records'].items()):
        if entry.get('status')!='complete':continue
        metric=json.loads(Path(entry['metrics_path']).read_text())
        config=metric['config']
        for item in metric['metrics']:
            rows.append(dict(condition=entry['condition'],architecture='TemporalTransformerEncoder',
                rat=config['rat'],fold=config['fold'],seed=config['seed'],method=config['version'],
                **{k:v for k,v in item.items() if k not in ('method','ridge_validation_candidates')}))
        fit=json.loads(Path(entry['fit_manifest_path']).read_text())
        embedding=json.loads(Path(entry['embedding_manifest_path']).read_text())
        compute.append(dict(condition=entry['condition'],architecture='TemporalTransformerEncoder',
            rat=config['rat'],fold=config['fold'],seed=config['seed'],method=config['version'],
            **{name:fit.get(name) for name in ('parameters','actual_updates','actual_optimizer_updates','replayed_updates',
            'fit_seconds','seconds_per_100_updates','updates_per_second','training_windows_per_second',
            'peak_gpu_allocated','peak_gpu_reserved','device','cuda_available','torch_version')},
            inference_seconds=embedding.get('inference_seconds'),
            inference_windows_per_second=embedding.get('inference_windows_per_second'),
            concurrency_context='May share GPU with CNN; not an isolated efficiency comparison',
            original_initial_artifact_reused=entry['artifact_root']==str(tr.DEFAULT_OUTPUT)))
    for name,data in (('RAT_TRANSFORMER_COMPARISON.csv',rows),('COMPUTE_PERFORMANCE.csv',compute)):
        if not data:continue
        path=output/name;tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--seed-count',type=int,choices=(5,7),default=7)
    parser.add_argument('--max-new-fits',type=int)
    args=parser.parse_args()
    planned=48*args.seed_count
    max_new_fits=planned if args.max_new_fits is None else args.max_new_fits
    if not 1<=max_new_fits<=planned:parser.error(f'max-new-fits must be1..{planned}')
    output=rr.checked_output(args.output)
    for protected in (rr.DEFAULT_OUTPUT,tr.DEFAULT_OUTPUT,tm.DEFAULT_OUTPUT):
        if output==protected.resolve() or protected.resolve() in output.parents:
            raise ValueError('Expanded study root must be separate from parent experiments')
    output.mkdir(parents=True,exist_ok=True)
    source_record=sources()
    configs=enumerate_configs(args.seed_count)
    plan=dict(created_utc=rr.now(),planned=planned,seed_count=args.seed_count,conditions=CONDITIONS,
        configs=[dict(condition=c,config=config) for c,config in configs],sources=source_record,
        parent_initial_root=str(tr.DEFAULT_OUTPUT),parent_joint_CNN_root=str(rr.DEFAULT_OUTPUT),
        reuse_policy='seed42fold0joint original completed/partial fits retain exact sources/configs/signatures',
        shuffle='not activated by this runner; separate upcoming controlled branch',
        selection='none; fixed settings/terminal checkpoint',optimizer_updates_in_dry_run=0)
    path=output/'CAMPAIGN_PLAN.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['configs']!=plan['configs'] or old['sources']!=source_record:
            raise RuntimeError('Expanded sealed plan mismatch; refuse overwrite')
    else:rr.atomic_json(path,plan)
    prepared=prepare(output)
    if not args.execute:
        existing=[]
        for condition,config in configs:
            root=artifact_root(output,condition,config)
            if root==tr.DEFAULT_OUTPUT:
                key=f"{config['rat']}/fold_0/{config['version']}/seed_42"
                folder=root/'stage01_fit'/key
                existing.append(dict(cell=key,root=str(root),completed=(folder/'manifest.json').exists(),
                    checkpoint_sha256=sha256(folder/'checkpoint.pt')))
        rr.atomic_json(output/'INITIAL_REUSE_INDEX.json',dict(existing=existing,created_utc=rr.now()))
        print(json.dumps(dict(planned=planned,joint=planned//2,time_only=planned//2,prepared_data_conditions=24,
            optimizer_updates=0,existing_original_cells=len(existing),output=str(output)),indent=2),flush=True)
        return
    gate_path=output/'qualification'/'GATE.json'
    if not gate_path.exists():raise RuntimeError('Expanded Transformer gate missing')
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_record):
        raise RuntimeError('Expanded Transformer gate stale/NO-GO')
    initial_gate_path=tr.DEFAULT_OUTPUT/'qualification'/'GATE.json'
    if not initial_gate_path.exists():raise RuntimeError('Initial Transformer CPU qualification missing')
    initial_gate=json.loads(initial_gate_path.read_text())
    if initial_gate.get('decision')!='GO' or initial_gate.get('training_source_signature')!=config_hash(tr.sources()):
        raise RuntimeError('Initial Transformer CPU qualification stale/NO-GO')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    # Own both locks: initial runner cannot race the exact reused checkpoint.
    locks=[]
    try:
        for root in (output,tr.DEFAULT_OUTPUT):
            lock=root/'RUNNING.lock'
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.write(fd,str(os.getpid()).encode());os.close(fd);locks.append(lock)
    except BaseException:
        for lock in locks:lock.unlink(missing_ok=True)
        raise
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=planned,records={},created_utc=rr.now())
    new_fits=0
    try:
        for condition,config in configs:
            key=f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
            state_key=condition+'/'+key
            data_folder,data_manifest=prepared[(condition,config['rat'],config['fold'])]
            root=artifact_root(output,condition,config)
            folder=root/'stage01_fit'/key
            state.update(current=state_key,status='running',updated_utc=rr.now());rr.atomic_json(state_path,state)
            train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
            parent_id=sha256(data_folder/'manifest.json') if condition=='joint' else config_hash(data_manifest)
            with tr.architecture_bindings():
                if condition=='time_only':
                    with tm.context_bindings():
                        fit,created=rr.fit_cell(folder,train,valid,config,data_manifest['train_scales'],
                            parent_id,training_sources(condition,source_record),device)
                else:
                    fit,created=rr.fit_cell(folder,train,valid,config,data_manifest['train_scales'],
                        parent_id,training_sources(condition,source_record),device)
            new_fits+=int(created)
            state['records'][state_key]=dict(status='fit_complete',condition=condition,artifact_root=str(root),
                fit_manifest_path=str(folder/'manifest.json'),checkpoint_sha256=fit['checkpoint_sha256'])
            rr.atomic_json(state_path,state)
            embedding_folder,embedding_record=tr.export_cell(root,folder,data_folder,config,fit,device)
            rr.decode_cell(root,embedding_folder,config,embedding_record)
            if condition=='time_only':tm.diagnostic_cell(root,embedding_folder,data_manifest,config)
            else:rr.diagnostic_cell(root,embedding_folder,data_manifest,config)
            metric_path=root/'stage03_metrics'/key/'metrics.json'
            state['records'][state_key].update(status='complete',
                fit_manifest_sha256=sha256(folder/'manifest.json'),metrics_path=str(metric_path),metrics_sha256=sha256(metric_path),
                embedding_manifest_path=str(embedding_folder/'manifest.json'),embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'))
            state['completed']=sum(r.get('status')=='complete' for r in state['records'].values())
            rr.atomic_json(state_path,state);write_tables(output,state)
            print(json.dumps(dict(completed=state['completed'],planned=planned,last=state_key,new_fit=created,fit_seconds=fit['fit_seconds'])),flush=True)
            if new_fits>=max_new_fits:
                state['status']='requested_limit';break
        else:state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',last_error=repr(exc))
        if state.get('current'):state['records'].setdefault(state['current'],{}).update(error=repr(exc))
        raise
    finally:
        state['updated_utc']=rr.now();rr.atomic_json(state_path,state)
        for lock in locks:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
