"""Controlled first eight recovered-Transformer rat fits, separately staged."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from . import rat_runner as rr
from .recovered_encoders import TemporalTransformerEncoder
from .data import load_rows, assert_disjoint, eligible_centers, sha256, config_hash

DEFAULT_OUTPUT=rr.ROOT/'results'/'user_soft_rat_transformer_v1'
CNN_OUTPUT=rr.DEFAULT_OUTPUT
ARCHITECTURE=dict(model_dim=64,n_heads=4,n_layers=2,dropout=.1,max_length=4096,normalize=True)


class TransformerInputAdapter(nn.Module):
    """Transpose the unchanged official CNN input contexts, nothing else."""
    def __init__(self,n_features,config):
        super().__init__()
        self.encoder=TemporalTransformerEncoder(n_features=n_features,
            embedding_dim=config['output_dimension'],**config['transformer_settings'])

    def forward(self,x):
        if x.ndim!=3 or x.shape[-1]!=10:
            raise ValueError('Expected official-context input [B,N,10]')
        return self.encoder(x.transpose(1,2))


def make_encoder(n_features,config,device):
    return TransformerInputAdapter(n_features,config).to(device)


def frozen_config(rat,fold,seed,version,updates=10000):
    config=rr.frozen_config(rat,fold,seed,version,updates)
    config['architecture']='TemporalTransformerEncoder'
    config['transformer_settings']=dict(ARCHITECTURE)
    config.pop('hidden_units')
    config['input_adapter']='transpose_B_N_10_to_B_10_N'
    config['output_normalization']='native_L2_normalized'
    config['concurrency_context']='may_share_GPU_with_CNN; not an isolated efficiency comparison'
    return config


@contextmanager
def architecture_bindings():
    """Dependency injection in this separate process; restore even on failure."""
    original=rr.make_encoder
    rr.make_encoder=make_encoder
    try:
        yield
    finally:
        rr.make_encoder=original


def sources():
    base=rr.sources()
    base[str(Path(__file__).resolve())]=sha256(__file__)
    recovered=Path(__file__).with_name('recovered_encoders.py')
    base[str(recovered)]=sha256(recovered)
    for function in (frozen_config,make_encoder,architecture_bindings,export_cell):
        base['transformer_function:'+function.__name__]=rr.code_hash(function)
    return base


def initial_configs():
    return [frozen_config(rat,0,42,version) for rat,version in itertools.product(rr.RATS,rr.VERSIONS)]


def prepare(output,rat,fold):
    existing=CNN_OUTPUT/'stage00_data'/rat/f'fold_{fold}'/'manifest.json'
    if existing.exists():
        # Exact source/split/cache content validation, no re-estimation.
        return rr.prepare(CNN_OUTPUT,rat,fold)
    return rr.prepare(output,rat,fold)


def export_cell(output,folder,data_folder,config,fit_record,device):
    path=output/'stage02_embedding'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    manifest=path/'manifest.json'
    signature=config_hash(dict(parent_checkpoint=fit_record['checkpoint_sha256'],data_parent=fit_record['parent_id'],
        exporter_source=rr.code_hash(export_cell),context_source=rr.code_hash(rr.encode_batch),
        encoder_factory_source=rr.code_hash(make_encoder),architecture=config['transformer_settings']))
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['signature']!=signature:
            raise RuntimeError('Transformer embedding dependency mismatch')
        for split,entry in record['embeddings'].items():
            if sha256(path/f'{split}_embedding.npz')!=entry['embedding_sha256']:
                raise RuntimeError('Transformer embedding content mismatch')
        return path,record
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Partial Transformer embedding output requires inspection')
    data={s:load_rows(data_folder/f'{s}.npz') for s in ('train','valid','test')}
    assert_disjoint(data)
    model=make_encoder(data['train'].neural.shape[1],config,device)
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=device,weights_only=False)['model'])
    model.eval()
    path.mkdir(parents=True,exist_ok=False)
    record=dict(signature=signature,created_utc=rr.now(),config=config,seed=config['seed'],
        rat=config['rat'],fold=config['fold'],version=config['version'],architecture=config['architecture'],
        parent_checkpoint_sha256=fit_record['checkpoint_sha256'],embeddings={},
        raw_semantics='native normalized recovered Transformer output; not pre-normalization activations')
    started=time.perf_counter()
    if device.type=='cuda':torch.cuda.synchronize()
    for split,rows in data.items():
        values=rr.tensors(rows,device);parts=[]
        with torch.no_grad():
            for start in range(0,len(rows.neural),config['batch_size']):
                ids=torch.arange(start,min(start+config['batch_size'],len(rows.neural)))
                z,_=rr.encode_batch(model,values,ids)
                parts.append(z.cpu().numpy())
        raw=np.concatenate(parts);unit=raw/np.linalg.norm(raw,axis=1,keepdims=True)
        if raw.shape!=(len(rows.neural),config['output_dimension']) or not np.isfinite(unit).all():
            raise RuntimeError('Transformer export invariants failed')
        mask=np.zeros(len(raw),dtype=bool);mask[eligible_centers(rows)]=True
        file=path/f'{split}_embedding.npz'
        np.savez_compressed(file,embedding=raw,embedding_raw=raw,embedding_unit=unit,
            behavior=rows.behavior,source_id=rows.source_id,trial_id=rows.trial_id,time_id=rows.time_id,
            session_id=rows.session_id,valid_mask=mask,split=split,rat=config['rat'],fold=config['fold'],
            seed=config['seed'],model=config['architecture'],objective=config['version'])
        record['embeddings'][split]=dict(embedding_sha256=sha256(file),shape=list(raw.shape),valid_rows=int(mask.sum()))
    if device.type=='cuda':torch.cuda.synchronize()
    record['inference_seconds']=time.perf_counter()-started
    record['inference_windows_per_second']=sum(len(v.neural) for v in data.values())/record['inference_seconds']
    record['concurrency_context']=config['concurrency_context']
    rr.atomic_json(manifest,record)
    return path,record


def write_comparison(output):
    """Separate Transformer tables; preserve architecture in every row."""
    import csv
    rr.write_comparison(output)
    for name in ('RAT_SOFT_COMPARISON.csv','COMPUTE_PERFORMANCE.csv'):
        path=output/name
        if not path.exists():continue
        with path.open(newline='',encoding='utf-8') as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            row['architecture']='TemporalTransformerEncoder'
            row['concurrency_context']='GPU may be concurrent with CNN; efficiency not directly comparable'
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--max-new-fits',type=int,default=8)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    if args.max_new_fits<1 or args.max_new_fits>8:
        parser.error('Initial Transformer block permits only1..8 new fits')
    output=rr.checked_output(args.output)
    if output==CNN_OUTPUT.resolve() or CNN_OUTPUT.resolve() in output.parents:
        raise ValueError('Transformer output cannot write CNN branch')
    output.mkdir(parents=True,exist_ok=True)
    source_hashes=sources()
    configs=initial_configs()
    plan=dict(created_utc=rr.now(),planned=8,configs=configs,sources=source_hashes,
        potential_full_design=240,full_design_authorized=False,parent_CNN_root=str(CNN_OUTPUT),
        no_test_outcome_selection=True,optimizer_updates_in_dry_run=0,
        concurrency_context='CNN GPU campaign may be running concurrently')
    plan_path=output/'CAMPAIGN_PLAN.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if old['configs']!=configs or old['sources']!=source_hashes:
            raise RuntimeError('Transformer sealed plan changed; choose new root')
    else:rr.atomic_json(plan_path,plan)
    prepared={(rat,0):prepare(output,rat,0) for rat in rr.RATS}
    references={rat:dict(path=str(folder/'manifest.json'),sha256=sha256(folder/'manifest.json'))
                for (rat,_),(folder,_) in prepared.items()}
    if not args.execute:
        print(json.dumps(dict(planned=8,prepared_caches=4,optimizer_updates=0,parent_data=references,
            configs=configs,output=str(output)),indent=2),flush=True)
        return
    gate_path=output/'qualification'/'GATE.json'
    if not gate_path.exists():raise RuntimeError('Transformer qualification gate missing')
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_hashes):
        raise RuntimeError('Transformer qualification gate stale/NO-GO')
    parent_gate=CNN_OUTPUT/'qualification'/'GATE.json'
    if not parent_gate.exists() or json.loads(parent_gate.read_text()).get('decision')!='GO':
        raise RuntimeError('Parent CNN gate missing/NO-GO')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device(args.device);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock=output/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=8,records={},created_utc=rr.now())
    new_fits=0
    try:
        with architecture_bindings():
            for config in configs:
                key=f"{config['rat']}/fold_0/{config['version']}/seed_42"
                data_folder,data_manifest=prepared[(config['rat'],0)]
                folder=output/'stage01_fit'/key
                state.update(current=key,status='running',updated_utc=rr.now());rr.atomic_json(state_path,state)
                train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
                fit,created=rr.fit_cell(folder,train,valid,config,data_manifest['train_scales'],
                    sha256(data_folder/'manifest.json'),source_hashes,device)
                new_fits+=int(created)
                state['records'][key]=dict(status='fit_complete',checkpoint_sha256=fit['checkpoint_sha256'],actual_updates=fit['actual_updates'])
                rr.atomic_json(state_path,state)
                embedding_folder,embedding_record=export_cell(output,folder,data_folder,config,fit,device)
                rr.decode_cell(output,embedding_folder,config,embedding_record)
                rr.diagnostic_cell(output,embedding_folder,data_manifest,config)
                state['records'][key].update(status='complete',embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'))
                state['completed']=sum(r['status']=='complete' for r in state['records'].values())
                rr.atomic_json(state_path,state);write_comparison(output)
                print(json.dumps(dict(completed=state['completed'],planned=8,last=key,fit_seconds=fit['fit_seconds'])),flush=True)
                if new_fits>=args.max_new_fits:
                    state['status']='requested_limit';break
            else:state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',last_error=repr(exc))
        if state.get('current'):
            previous=state['records'].get(state['current'],{})
            state['records'][state['current']]=dict(**previous,error=repr(exc))
        raise
    finally:
        state['updated_utc']=rr.now();rr.atomic_json(state_path,state);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
