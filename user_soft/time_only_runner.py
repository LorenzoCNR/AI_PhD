"""Pure local-time soft CNN control, preserving joint-reference artifacts."""
from __future__ import annotations
import argparse
import copy
import csv
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from . import rat_runner as rr
from .data import sha256,config_hash,load_rows
from .loss import evaluate_soft

SPECS=[dict(key='time_id',geometry='temporal',weight=1.)]
DEFAULT_OUTPUT=rr.ROOT/'results'/'user_soft_rat_time_only_v1'
PARENT_OUTPUT=rr.DEFAULT_OUTPUT


def frozen_config(rat,fold,seed,version,updates=10000):
    config=rr.frozen_config(rat,fold,seed,version,updates)
    config.update(condition='time_only',metadata_specs=copy.deepcopy(SPECS),
                  temporal_coordinate='official trial-relative25msbin',
                  cross_trial_candidates=True,metadata_supervision='local_time_only_no_behavior')
    return config


@contextmanager
def context_bindings():
    """Inject only metadata specs into the separate process; restore finally."""
    original=rr.SPECS
    rr.SPECS=SPECS
    try:
        yield
    finally:
        rr.SPECS=original


def sources():
    if rr.SPECS==SPECS:
        raise RuntimeError('Compute source signatures outside time-only binding')
    hashes=rr.sources()
    hashes[str(Path(__file__).resolve())]=sha256(__file__)
    for function in (frozen_config,context_bindings,project_scales,diagnostic_cell):
        hashes['time_only_function:'+function.__name__]=rr.code_hash(function)
    return hashes


def project_scales(parent_scales):
    """Reuse the exact full-TRAIN temporal bound; never refit the subset."""
    if parent_scales.get('split')!='train':
        raise ValueError('TRAIN parent scales required')
    temporal=[scale for scale in parent_scales['scales'] if scale['key']=='time_id' and scale['geometry']=='temporal']
    if len(temporal)!=1:
        raise ValueError('Exactly one existing TRAIN temporal scale required')
    result=copy.deepcopy(parent_scales)
    result.update(scales=copy.deepcopy(temporal),specs=copy.deepcopy(SPECS),
                  parent_scale_sha256=config_hash(parent_scales),
                  projection='retain existingTRAINtime component; no scale refit')
    return result


def prepare(output,rat,fold):
    if rr.SPECS==SPECS:
        raise RuntimeError('Validate parent preparation outside time-only binding')
    parent_folder,parent=rr.prepare(PARENT_OUTPUT,rat,fold)
    scales=project_scales(parent['train_scales'])
    record=dict(parent_cache_manifest=str(parent_folder/'manifest.json'),
        parent_cache_sha256=sha256(parent_folder/'manifest.json'),parent_cache_signature=parent['signature'],
        rat=rat,fold=fold,splits=parent['splits'],train_scales=scales,
        metadata_specs=SPECS,condition='time_only',no_scale_refit=True)
    record['signature']=config_hash(record)
    path=output/'stage00_data'/rat/f'fold_{fold}'/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=record:
            raise RuntimeError('Time-only parent cache/scales changed; refuse overwrite')
    else:rr.atomic_json(path,record)
    return parent_folder,record


def diagnostic_cell(output,embedding_folder,data_manifest,config):
    """One-component TRAIN diagnostics without modifying joint diagnostics."""
    from .downstream import _LazyArchive
    path=output/'stage04_diagnostics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    complete=path/'STAGE_SIGNATURE.json'
    signature=config_hash(dict(parent_train=sha256(embedding_folder/'train_embedding.npz'),
        source=rr.code_hash(diagnostic_cell),scales=data_manifest['train_scales'],config=config))
    if complete.exists():
        record=json.loads(complete.read_text())
        if record['signature']!=signature:raise RuntimeError('Time-only diagnostic dependencies changed')
        for name,digest in record['artifact_hashes'].items():
            if sha256(path/name)!=digest:raise RuntimeError('Time-only diagnostic content mismatch')
        return
    if path.exists() and any(path.iterdir()):raise RuntimeError('Partial time-only diagnostic output')
    data=_LazyArchive(embedding_folder/'train_embedding.npz')
    generator=torch.Generator().manual_seed(config['seed']+30000)
    ids=torch.randperm(len(data['embedding']),generator=generator)[:64].numpy()
    z=torch.as_tensor(data['embedding'][ids])
    batch={key:torch.as_tensor(data[key][ids]) for key in ('time_id','trial_id','session_id','source_id')}
    rows=[]
    def probe(kind,meta=.5,embedding=.1,count=len(ids)):
        selected={k:v[:count] for k,v in batch.items()}
        with torch.no_grad():
            result=evaluate_soft(z[:count],selected,SPECS,version=config['version'],
                scales=data_manifest['train_scales'],temperature=embedding,metadata_temperature=meta)
        row=dict(probe=kind,seed=config['seed'],split='train',condition='time_only',rows=count,
            metadata_temperature=meta,embedding_temperature=embedding,loss=float(result.loss),
            pair01_log_W=float(result.log_W[0,1]),pair01_Q=float(result.Q[0,1]))
        row.update(result.diagnostics());rows.append(row)
        return result
    reference=probe('reference')
    for meta in (.25,.5,1.):probe('metadata_temperature',meta=meta)
    for embedding in (.05,.1,.2):probe('embedding_temperature',embedding=embedding)
    if len(ids)>32:
        probe('batch_composition_first32',count=32)
        probe('batch_composition_first64_or_available')
    path.mkdir(parents=True,exist_ok=False)
    with (path/'TRAIN_SENSITIVITY.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(path/'TRAIN_REFERENCE_W_Q_P.npz',W=reference.W.numpy(),Q=reference.Q.numpy(),
        P=reference.P.numpy(),log_W=reference.log_W.numpy(),reference_train_row_indices=ids,
        **{key:value.numpy() for key,value in batch.items()})
    rr.atomic_json(path/'PROVENANCE.json',dict(created_utc=rr.now(),config=config,split='train',
        parent_train_embedding_sha256=sha256(embedding_folder/'train_embedding.npz'),
        parent_checkpoint_sha256=json.loads((embedding_folder/'manifest.json').read_text())['parent_checkpoint_sha256'],
        train_scales=data_manifest['train_scales'],source_sha256=sha256(__file__),
        probe_policy='prespecified temperatures/composition only; one weight1; no selection',
        reference_train_row_indices=ids.tolist()))
    artifacts={p.name:sha256(p) for p in path.iterdir() if p.is_file()}
    rr.atomic_json(complete,dict(signature=signature,artifact_hashes=artifacts))


def write_comparison(output):
    rr.write_comparison(output)
    for name in ('RAT_SOFT_COMPARISON.csv','COMPUTE_PERFORMANCE.csv'):
        path=output/name
        if not path.exists():continue
        with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        for row in rows:row.update(condition='time_only',architecture='offset10-model')
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--max-new-fits',type=int,default=168)
    args=parser.parse_args()
    if not 1<=args.max_new_fits<=168:parser.error('max-new-fits must be1..168')
    output=rr.checked_output(args.output)
    if output==PARENT_OUTPUT.resolve() or PARENT_OUTPUT.resolve() in output.parents:
        raise ValueError('Time-only branch cannot write parent joint experiment')
    output.mkdir(parents=True,exist_ok=True)
    source_hashes=sources()
    configs=[frozen_config(rat,fold,seed,version) for seed,fold,rat,version in
             itertools.product(range(42,49),range(3),rr.RATS,rr.VERSIONS)]
    plan=dict(created_utc=rr.now(),planned=168,configs=configs,sources=source_hashes,
        condition='time_only',parent_joint_root=str(PARENT_OUTPUT),optimizer_updates_in_dry_run=0,
        no_behavior_metadata_in_loss=True,no_test_outcome_selection=True)
    path=output/'CAMPAIGN_PLAN.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['configs']!=configs or old['sources']!=source_hashes:
            raise RuntimeError('Time-only sealed plan changed; new study root required')
    else:rr.atomic_json(path,plan)
    prepared={(rat,fold):prepare(output,rat,fold) for rat,fold in itertools.product(rr.RATS,range(3))}
    if not args.execute:
        print(json.dumps(dict(planned=168,prepared_caches=12,optimizer_updates=0,output=str(output)),indent=2),flush=True)
        return
    gate_path=output/'qualification'/'GATE.json'
    if not gate_path.exists():raise RuntimeError('Time-only gate missing')
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_hashes):
        raise RuntimeError('Time-only gate stale/NO-GO')
    parent_gate=PARENT_OUTPUT/'qualification'/'GATE.json'
    if not parent_gate.exists() or json.loads(parent_gate.read_text()).get('decision')!='GO':
        raise RuntimeError('Parent joint gate missing/NO-GO')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock=output/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=168,records={},created_utc=rr.now())
    new_fits=0
    try:
        with context_bindings():
            for config in configs:
                key=f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
                data_folder,data_manifest=prepared[(config['rat'],config['fold'])]
                folder=output/'stage01_fit'/key
                state.update(current=key,status='running',updated_utc=rr.now());rr.atomic_json(state_path,state)
                train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
                fit,created=rr.fit_cell(folder,train,valid,config,data_manifest['train_scales'],
                    config_hash(data_manifest),source_hashes,device)
                new_fits+=int(created)
                state['records'][key]=dict(status='fit_complete',checkpoint_sha256=fit['checkpoint_sha256'],actual_updates=fit['actual_updates'])
                rr.atomic_json(state_path,state)
                embedding_folder,embedding_record=rr.export_cell(output,folder,data_folder,config,fit,device)
                rr.decode_cell(output,embedding_folder,config,embedding_record)
                diagnostic_cell(output,embedding_folder,data_manifest,config)
                state['records'][key].update(status='complete',embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'))
                state['completed']=sum(r['status']=='complete' for r in state['records'].values())
                rr.atomic_json(state_path,state);write_comparison(output)
                print(json.dumps(dict(completed=state['completed'],planned=168,last=key,fit_seconds=fit['fit_seconds'])),flush=True)
                if new_fits>=args.max_new_fits:
                    state['status']='requested_limit';break
            else:state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',last_error=repr(exc))
        if state.get('current'):state['records'].setdefault(state['current'],{}).update(error=repr(exc))
        raise
    finally:
        state['updated_utc']=rr.now();rr.atomic_json(state_path,state);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
