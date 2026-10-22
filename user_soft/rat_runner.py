"""Staged/resumable joint soft rat campaign; dry-run never updates parameters."""
from __future__ import annotations
import argparse
import csv
import inspect
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone
import numpy as np
import torch
from .data import (sha256,config_hash,load_official_splits,save_rows,load_rows,
                   eligible_centers,assert_disjoint)
from .loss import evaluate_soft,fit_distance_scales

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'results'/'user_soft_rat_v1'
RATS=('achilles','buddy','cicero','gatsby')
VERSIONS=('soft_v1_frozen','soft_v1_stable')
SPECS=[dict(key='time_id',geometry='temporal',weight=.5),
       dict(key='behavior',geometry='euclidean',weight=.5)]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def sources():
    import cebra.models,cebra.models.layers
    local=Path(__file__).parent
    paths=[local/name for name in ('data.py','loss.py','legacy_infonce.py','legacy_batch_similarity.py')]
    paths += [Path(inspect.getfile(cebra.models.init('offset10-model',num_neurons=1,num_units=32,num_output=32).__class__)),
              Path(inspect.getfile(cebra.models.layers))]
    result={str(p):sha256(p) for p in paths}
    result.update({f'training_function:{f.__name__}':code_hash(f) for f in
                  (frozen_config,make_encoder,tensors,encode_batch,objective,fit_cell)})
    return result


def code_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def frozen_config(rat,fold,seed,version,updates=10000):
    return dict(rat=rat,fold=fold,seed=seed,version=version,condition='position_direction',
                architecture='offset10-model',hidden_units=32,output_dimension=32,
                batch_size=512,learning_rate=3e-4,optimizer='Adam',betas=[.9,.999],eps=1e-8,
                weight_decay=0.,max_updates=updates,early_stopping=False,
                embedding_temperature=.1,metadata_temperature=.5,metadata_specs=SPECS,
                input_policy='legacy_concat',offset_left=5,offset_right=5,time_bin_ms=25,
                cross_trial_candidates=True,sampler='uniform_reference_with_replacement',
                diagonal_excluded=True,checkpoint_interval=500,diagnostic_interval=500,
                empty_candidate_policy='error',qualification_only=updates!=10000)


def checked_output(output):
    output=Path(output).resolve()
    permitted=(ROOT/'results').resolve()
    if permitted not in output.parents or output.name in ('rat_trial_benchmark_v1','rat_campaign_v1'):
        raise ValueError('Output must be a separate new study under AI_PhD/results')
    for p in (ROOT/'results'/'rat_trial_benchmark_v1',ROOT/'results'/'PAPER_FIGURE_DELIVERY'):
        if output==p.resolve() or p.resolve() in output.parents:
            raise ValueError('Existing reference output is protected')
    return output


def prepare(output,rat,fold):
    from cebra.datasets.hippocampus import SingleRatTrialSplitDataset
    folder=output/'stage00_data'/rat/f'fold_{fold}'
    parent=ROOT/'data'/'rat_hippocampus'/f'{rat}.jl'
    expected=dict(rat=rat,fold=fold,parent_source_sha256=sha256(parent),
                  source_data_adapter_sha256=sha256(Path(__file__).parent/'data.py'),
                  official_dataset_sha256=sha256(Path(inspect.getfile(SingleRatTrialSplitDataset))),
                  input_policy='legacy_concat',metadata_specs=SPECS,scale_policy='train_bounds')
    manifest=folder/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['signature']!=config_hash(expected):
            raise RuntimeError('Prepared cache mismatch; choose a new study root instead of overwriting')
        for split,entry in record['splits'].items():
            if sha256(folder/f'{split}.npz')!=entry['sha256']:
                raise RuntimeError('Prepared cache hash mismatch')
        return folder,record
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f'Incomplete preparation requires inspection: {folder}')
    splits=load_official_splits(ROOT,rat,fold)
    scales=fit_distance_scales(splits['train'].metadata(),SPECS,split='train',source_ids=splits['train'].source_id.tolist())
    folder.mkdir(parents=True,exist_ok=False)
    record=dict(**expected,signature=config_hash(expected),created_utc=now(),splits={},train_scales=scales,
                no_preprocessing_refit=True,no_test_statistics_for_scaling=True)
    for split,rows in splits.items():
        path=folder/f'{split}.npz'
        save_rows(path,rows)
        record['splits'][split]=dict(sha256=sha256(path),rows=len(rows.neural),
                                    trials=np.unique(rows.trial_id).tolist(),
                                    valid_centers=len(eligible_centers(rows)),
                                    source_id_hash=config_hash(rows.source_id.tolist()))
    atomic_json(manifest,record)
    return folder,record


def make_encoder(n_features,config,device):
    import cebra.models
    return cebra.models.init('offset10-model',num_neurons=n_features,
                             num_units=config['hidden_units'],num_output=config['output_dimension']).to(device)


def parameter_hash(model):
    import hashlib
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def tensors(rows,device):
    return dict(neural=torch.as_tensor(rows.neural,device=device,dtype=torch.float32),
                **{k:torch.as_tensor(getattr(rows,k),device=device) for k in
                   ('behavior','time_id','trial_id','session_id','source_id')})


def encode_batch(model,values,indices):
    indices=indices.to(values['neural'].device)
    centers=indices.clamp(5,len(values['neural'])-5)
    support=centers[:,None]+torch.arange(-5,5,device=centers.device)
    z=model(values['neural'][support].transpose(1,2))
    if z.ndim!=2 or not torch.isfinite(z).all():
        raise FloatingPointError('Invalid official CNN output')
    batch={k:v[indices] for k,v in values.items() if k!='neural'}
    return z,batch


def objective(model,values,indices,config,scales):
    z,batch=encode_batch(model,values,indices)
    return evaluate_soft(z,batch,SPECS,version=config['version'],scales=scales,
                         temperature=config['embedding_temperature'],
                         metadata_temperature=config['metadata_temperature'])


def fit_cell(folder,train,valid,config,scales,parent_id,source_hashes,device):
    """Never load test observations. Resume optimizer/RNG exactly at checkpoint."""
    signature=config_hash(dict(config=config,parent_id=parent_id,sources=source_hashes))
    complete=folder/'manifest.json'
    checkpoint=folder/'checkpoint.pt'
    if complete.exists():
        record=json.loads(complete.read_text())
        if record['signature']!=signature or sha256(checkpoint)!=record['checkpoint_sha256']:
            raise RuntimeError('Completed fit provenance mismatch')
        return record,False
    folder.mkdir(parents=True,exist_ok=True)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    if device.type=='cuda':
        torch.cuda.manual_seed_all(config['seed']);torch.cuda.reset_peak_memory_stats(device)
    model=make_encoder(train.neural.shape[1],config,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=tuple(config['betas']),
                               eps=config['eps'],weight_decay=config['weight_decay'])
    generator=torch.Generator().manual_seed(config['seed'])
    step=0;history=[];diagnostics=[];elapsed=0.;initial_hash=parameter_hash(model)
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved['signature']!=signature:
            raise RuntimeError('Partial checkpoint dependencies changed; refusing automatic retrain/overwrite')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'].cpu());torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        step=saved['step'];history=saved['history'];diagnostics=saved['diagnostics'];elapsed=saved['seconds'];initial_hash=saved['initial_hash']
    train_t=tensors(train,device);valid_t=tensors(valid,device)
    evaluation_generator=torch.Generator().manual_seed(config['seed']+20000)
    train_eval=torch.randint(len(train.neural),(config['batch_size'],),generator=evaluation_generator)
    valid_eval=torch.randint(len(valid.neural),(config['batch_size'],),generator=evaluation_generator)
    started=time.perf_counter()
    def save_checkpoint():
        if device.type=='cuda': torch.cuda.synchronize()
        value=dict(signature=signature,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   generator=generator.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                   step=step,history=history,diagnostics=diagnostics,seconds=elapsed+time.perf_counter()-started,
                   config=config,parent_id=parent_id,source_hashes=source_hashes,initial_hash=initial_hash)
        tmp=checkpoint.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(checkpoint)
    model.train()
    start_step=step
    attempt_path=folder/'attempts'/f'{time.time_ns()}.json'
    if not checkpoint.exists(): save_checkpoint()
    try:
        while step<config['max_updates']:
            indices=torch.randint(len(train.neural),(config['batch_size'],),generator=generator)
            optimizer.zero_grad(set_to_none=True)
            result=objective(model,train_t,indices,config,scales)
            loss=result.loss
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite encoder gradients')
            optimizer.step();step+=1
            history.append(dict(update=step,train_loss=float(loss.detach())))
            if step%config['diagnostic_interval']==0 or step==config['max_updates']:
                model.eval()
                with torch.no_grad():
                    a=objective(model,train_t,train_eval,config,scales)
                    b=objective(model,valid_t,valid_eval,config,scales)
                diagnostics.append(dict(update=step,train_evaluation_loss=float(a.loss),validation_loss=float(b.loss),
                                        duplicate_fraction=1-len(torch.unique(indices))/len(indices),**a.diagnostics()))
                model.train()
            if step%config['checkpoint_interval']==0: save_checkpoint()
    except BaseException as exc:
        # Only the last committed checkpoint contains a consistent optimizer/RNG
        # transaction. Never publish a potentially partial update as resumable.
        atomic_json(attempt_path,dict(start_step=start_step,last_successful_step=step,
                    successful_updates=step-start_step,seconds=time.perf_counter()-started,
                    status='interrupted',error=repr(exc),last_checkpoint_sha256=sha256(checkpoint)))
        raise
    save_checkpoint()
    atomic_json(attempt_path,dict(start_step=start_step,last_successful_step=step,
                successful_updates=step-start_step,seconds=time.perf_counter()-started,status='complete'))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    attempts=[json.loads(p.read_text()) for p in (folder/'attempts').glob('*.json')]
    total_seconds=sum(a['seconds'] for a in attempts)
    actual_optimizer_updates=sum(a['successful_updates'] for a in attempts)
    record=dict(signature=signature,created_utc=now(),config=config,seed=config['seed'],
                rat=config['rat'],fold=config['fold'],version=config['version'],parent_id=parent_id,
                source_hashes=source_hashes,checkpoint_sha256=sha256(checkpoint),actual_updates=step,
                initial_parameter_hash=initial_hash,final_parameter_hash=parameter_hash(model),
                actual_optimizer_updates=actual_optimizer_updates,replayed_updates=actual_optimizer_updates-step,
                fit_seconds=total_seconds,seconds_per_100_updates=total_seconds/actual_optimizer_updates*100,
                updates_per_second=actual_optimizer_updates/total_seconds,training_windows_per_second=config['batch_size']*actual_optimizer_updates/total_seconds,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                peak_gpu_allocated=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                peak_gpu_reserved=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                cuda_available=torch.cuda.is_available(),device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                torch_version=torch.__version__,history=history,diagnostics=diagnostics,
                selection='fixed terminal checkpoint; validation diagnostics only; no early stopping/model selection')
    atomic_json(complete,record)
    return record,True


def export_cell(output,folder,data_folder,config,fit_record,device):
    path=output/'stage02_embedding'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    manifest=path/'manifest.json'
    signature=config_hash(dict(parent_checkpoint=fit_record['checkpoint_sha256'],data_parent=fit_record['parent_id'],
                               exporter_source=code_hash(export_cell),context_source=code_hash(encode_batch)))
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['signature']!=signature: raise RuntimeError('Embedding dependency mismatch')
        for split,entry in record['embeddings'].items():
            if sha256(path/f'{split}_embedding.npz')!=entry['embedding_sha256']: raise RuntimeError('Embedding hash mismatch')
        return path,record
    if path.exists() and any(path.iterdir()): raise RuntimeError('Incomplete embedding output requires inspection')
    data={s:load_rows(data_folder/f'{s}.npz') for s in ('train','valid','test')}
    assert_disjoint(data)
    model=make_encoder(data['train'].neural.shape[1],config,device)
    model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location=device,weights_only=False)['model']);model.eval()
    path.mkdir(parents=True,exist_ok=False)
    record=dict(signature=signature,created_utc=now(),seed=config['seed'],rat=config['rat'],fold=config['fold'],version=config['version'],
                config=config,parent_checkpoint_sha256=fit_record['checkpoint_sha256'],embeddings={},
                raw_semantics='native normalized official encoder output; not pre-normalization activations')
    started=time.perf_counter()
    for split,rows in data.items():
        values=tensors(rows,device);parts=[]
        with torch.no_grad():
            for start in range(0,len(rows.neural),config['batch_size']):
                z,_=encode_batch(model,values,torch.arange(start,min(start+config['batch_size'],len(rows.neural))))
                parts.append(z.cpu().numpy())
        raw=np.concatenate(parts);unit=raw/np.linalg.norm(raw,axis=1,keepdims=True)
        if raw.shape!=(len(rows.neural),config['output_dimension']) or not np.isfinite(unit).all():
            raise RuntimeError('Embedding cardinality/finiteness invariant failed')
        mask=np.zeros(len(raw),dtype=bool);mask[eligible_centers(rows)]=True
        file=path/f'{split}_embedding.npz'
        np.savez_compressed(file,embedding=raw,embedding_raw=raw,embedding_unit=unit,
                            behavior=rows.behavior,source_id=rows.source_id,trial_id=rows.trial_id,time_id=rows.time_id,
                            session_id=rows.session_id,valid_mask=mask,split=split,rat=config['rat'],fold=config['fold'],
                            seed=config['seed'],model='offset10-model',objective=config['version'])
        record['embeddings'][split]=dict(embedding_sha256=sha256(file),shape=list(raw.shape),valid_rows=int(mask.sum()))
    record['inference_seconds']=time.perf_counter()-started
    record['inference_windows_per_second']=sum(len(v.neural) for v in data.values())/record['inference_seconds']
    atomic_json(manifest,record)
    return path,record


def decode_cell(output,embedding_folder,config,embedding_record):
    from .downstream import score_arrays
    destination=output/'stage03_metrics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    path=destination/'metrics.json'
    signature=config_hash(dict(parent=sha256(embedding_folder/'manifest.json'),scorer=sha256(Path(__file__).parent/'downstream.py')))
    if path.exists():
        record=json.loads(path.read_text())
        if record['signature']!=signature: raise RuntimeError('Metric dependencies changed; separate downstream version required')
        return record
    from .downstream import _LazyArchive
    arrays={s:_LazyArchive(embedding_folder/f'{s}_embedding.npz') for s in ('train','valid','test')}
    records=score_arrays(arrays)
    record=dict(signature=signature,config=config,created_utc=now(),parent_embedding_manifest_sha256=sha256(embedding_folder/'manifest.json'),metrics=records)
    atomic_json(path,record)
    return record


def write_comparison(output):
    rows=[];compute=[]
    paths=list((output/'stage03_metrics').rglob('metrics.json'))+list((output/'stage03_reference').rglob('metrics.json'))
    for path in sorted(paths):
        record=json.loads(path.read_text());c=record['config']
        for metric in record['metrics']:
            rows.append(dict(rat=c['rat'],fold=c['fold'],seed=c['seed'],method=c['version'],**{k:v for k,v in metric.items() if k not in ('ridge_validation_candidates','method')}))
    for path in sorted((output/'stage01_fit').rglob('manifest.json')):
        r=json.loads(path.read_text());c=r['config']
        compute.append(dict(rat=c['rat'],fold=c['fold'],seed=c['seed'],method=c['version'],**{k:r[k] for k in
            ('parameters','actual_updates','actual_optimizer_updates','replayed_updates','fit_seconds','seconds_per_100_updates','updates_per_second','training_windows_per_second',
             'peak_gpu_allocated','peak_gpu_reserved','device','cuda_available','torch_version')}))
        emb=output/'stage02_embedding'/c['rat']/f"fold_{c['fold']}"/c['version']/f"seed_{c['seed']}"/'manifest.json'
        e=json.loads(emb.read_text()) if emb.exists() else {}
        compute[-1].update(inference_seconds=e.get('inference_seconds'),inference_windows_per_second=e.get('inference_windows_per_second'))
    for name,records in (('RAT_SOFT_COMPARISON.csv',rows),('COMPUTE_PERFORMANCE.csv',compute)):
        if not records: continue
        tmp=output/(name+'.tmp')
        with tmp.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        tmp.replace(output/name)


def diagnostic_cell(output,embedding_folder,data_manifest,config):
    from .diagnostics import write_diagnostics
    from .downstream import _LazyArchive
    path=output/'stage04_diagnostics'/config['rat']/f"fold_{config['fold']}"/config['version']/f"seed_{config['seed']}"
    signature=config_hash(dict(parent_train=sha256(embedding_folder/'train_embedding.npz'),
                               source=sha256(Path(__file__).parent/'diagnostics.py')))
    complete=path/'STAGE_SIGNATURE.json'
    if complete.exists():
        record=json.loads(complete.read_text())
        if record['signature']!=signature: raise RuntimeError('Diagnostic source changed; separate downstream version required')
        for name,digest in record['artifact_hashes'].items():
            if sha256(path/name)!=digest: raise RuntimeError('Diagnostic artifact hash mismatch')
        return
    data=_LazyArchive(embedding_folder/'train_embedding.npz')
    generator=torch.Generator().manual_seed(config['seed']+30000)
    indices=torch.randperm(len(data['embedding']),generator=generator)[:64].numpy()
    z=torch.as_tensor(data['embedding'][indices])
    batch={k:torch.as_tensor(data[k][indices]) for k in ('behavior','time_id','trial_id','session_id','source_id')}
    batch['split']='train'
    write_diagnostics(path,z,batch,SPECS,data_manifest['train_scales'],config['version'],config['seed'])
    files={p.name:sha256(p) for p in path.iterdir() if p.is_file()}
    atomic_json(complete,dict(signature=signature,parent_train_embedding_sha256=sha256(embedding_folder/'train_embedding.npz'),
                             parent_checkpoint_sha256=json.loads((embedding_folder/'manifest.json').read_text())['parent_checkpoint_sha256'],
                             config=config,reference_train_row_indices=indices.tolist(),artifact_hashes=files))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--max-new-fits',type=int)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    if args.dry_run and args.execute: parser.error('dry-run and execute are mutually exclusive')
    output=checked_output(args.output);output.mkdir(parents=True,exist_ok=True)
    source_hashes=sources()
    if args.execute:
        gate_path=output/'qualification'/'GATE.json'
        if not gate_path.exists(): raise RuntimeError('Qualification gate missing; execute tests/smoke first')
        gate=json.loads(gate_path.read_text())
        if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_hashes):
            raise RuntimeError('Qualification gate is stale or NO-GO')
        if len(list((output/'stage00_reference_verification').rglob('proof.json')))!=240:
            raise RuntimeError('All240 reused Behavior/Time references must pass the hash gate')
    configs=[frozen_config(rat,fold,seed,version) for seed,fold,rat,version in itertools.product(range(42,52),range(3),RATS,VERSIONS)]
    plan=dict(created_utc=now(),planned=len(configs),configs=configs,sources=source_hashes,
              existing_baseline_root=str(ROOT/'results'/'rat_trial_benchmark_v1'),
              prepared_only=not args.execute,no_test_outcome_selection=True)
    plan_path=output/'CAMPAIGN_PLAN.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text())
        if old['configs']!=configs or old['sources']!=source_hashes:
            raise RuntimeError('Sealed campaign plan changed; new study root required')
    else: atomic_json(plan_path,plan)
    prepared={(rat,fold):prepare(output,rat,fold) for rat,fold in itertools.product(RATS,range(3))}
    if not args.execute:
        print(json.dumps(dict(planned=len(configs),prepared_caches=len(prepared),optimizer_updates=0,output=str(output)),indent=2),flush=True)
        return
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    device=torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    lock=output/'RUNNING.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError('Campaign already running or stale lock; inspect PID before resuming')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=240,records={},created_utc=now())
    new_fits=0
    try:
        for config in configs:
            key=f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"
            data_folder,data_manifest=prepared[(config['rat'],config['fold'])]
            folder=output/'stage01_fit'/key
            state['current']=key;state['status']='running';state['updated_utc']=now();atomic_json(state_path,state)
            train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
            try:
                fit,created=fit_cell(folder,train,valid,config,data_manifest['train_scales'],sha256(data_folder/'manifest.json'),source_hashes,device)
                new_fits+=int(created)
                state['records'][key]=dict(status='fit_complete',checkpoint_sha256=fit['checkpoint_sha256'],actual_updates=fit['actual_updates'])
                atomic_json(state_path,state)
                embedding_folder,embedding_record=export_cell(output,folder,data_folder,config,fit,device)
                decode_cell(output,embedding_folder,config,embedding_record)
                diagnostic_cell(output,embedding_folder,data_manifest,config)
                state['records'][key]['status']='complete'
                state['records'][key]['embedding_manifest_sha256']=sha256(embedding_folder/'manifest.json')
            except Exception as exc:
                state['records'][key]=dict(status='failed',error=repr(exc),updated_utc=now())
                atomic_json(state_path,state)
                raise
            state['completed']=sum(r['status']=='complete' for r in state['records'].values())
            atomic_json(state_path,state);write_comparison(output)
            print(json.dumps(dict(completed=state['completed'],planned=240,last=key,fit_seconds=fit['fit_seconds'],new_fit=created)),flush=True)
            if args.max_new_fits is not None and new_fits>=args.max_new_fits:
                state['status']='requested_limit';break
        else: state['status']='complete'
    except BaseException as exc:
        state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        state['last_error']=repr(exc)
        raise
    finally:
        state['updated_utc']=now();atomic_json(state_path,state);lock.unlink(missing_ok=True)


if __name__=='__main__': main()
