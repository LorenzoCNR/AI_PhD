"""Staged active-monkey XY Soft comparison; no HPO and no rat/reference writes."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path
import time
import io
import unittest
import numpy as np
import torch
from . import rat_runner as rr, transformer_runner as tr, monkey_data as md
from .data import TrialRows,sha256,config_hash,load_rows,eligible_centers
from .loss import fit_distance_scales,evaluate_soft

OUTPUT=rr.ROOT/'results/user_soft_monkey_v1'
CONDITIONS=('time_only','joint','behavior_only')
DIMENSIONS=(3,4,8,16,32)
SEEDS=(42,43,44,45,46)
ARCHITECTURES=('cnn','transformer')


def specs_for(condition):
    if condition not in CONDITIONS:raise ValueError('Unknown monkey condition')
    if condition=='time_only':return [dict(key='time_id',geometry='temporal',weight=1.)]
    if condition=='behavior_only':return [dict(key='behavior',geometry='euclidean',weight=1.)]
    return [dict(key='time_id',geometry='temporal',weight=.5),dict(key='behavior',geometry='euclidean',weight=.5)]


def frozen_config(architecture,condition,dimension,seed,shuffle=False,updates=10000):
    if architecture not in ARCHITECTURES or condition not in CONDITIONS or dimension not in DIMENSIONS:
        raise ValueError('Unknown monkey configuration')
    if shuffle and condition=='time_only':raise ValueError('Behavior shuffle cannot change time-only objective')
    factory=rr.frozen_config if architecture=='cnn' else tr.frozen_config
    config=factory('monkey_active',0,seed,'soft_v1_stable',updates)
    role='behavior_shuffle' if shuffle else 'observed'
    config.update(condition=condition,output_dimension=dimension,time_bin_ms=1,
        metadata_specs=specs_for(condition),population='active',behavior_target='position_xy_joint',
        behavior_coordinate_encoding='native_XY_together_no_standardization',
        split_universe='official_active_train_valid_test',shuffle=shuffle,
        shuffle_seed=seed+10000 if shuffle else None,
        shuffle_unit='joint_XY_row_within_TRAIN_only' if shuffle else None,
        trial_id=f'{role}/{architecture}/{condition}/dim_{dimension}/seed_{seed}',
        evaluation_support='strict_trial_interior_10bins; shared_support_required_for_baseline_comparison',
        concurrency_context='sequential after rat shuffles and Figure3g correction')
    return config


def enumerate_configs():
    return [(architecture,frozen_config(architecture,condition,dimension,seed,shuffle))
        for seed,dimension,condition,architecture in itertools.product(SEEDS,DIMENSIONS,CONDITIONS,ARCHITECTURES)
        for shuffle in (False,True) if not(shuffle and condition=='time_only')]


@contextmanager
def context_bindings(config):
    original=rr.SPECS
    rr.SPECS=copy.deepcopy(config['metadata_specs'])
    try:
        if config['architecture']=='TemporalTransformerEncoder':
            with tr.architecture_bindings():yield
        else:yield
    finally:rr.SPECS=original


def project_scales(parent,condition):
    if parent.get('split')!='train':raise ValueError('TRAIN scales only')
    specs=specs_for(condition)
    result=copy.deepcopy(parent)
    result.update(specs=specs,scales=[copy.deepcopy(s) for s in parent['scales'] if any(s['key']==p['key'] for p in specs)],
        parent_scale_sha256=config_hash(parent),projection='retain full TRAIN bounds; no refit')
    if len(result['scales'])!=len(specs):raise ValueError('Missing required TRAIN bound')
    return result


def shuffled_train(rows,seed):
    if rows.split!='train':raise ValueError('Shuffle TRAIN only')
    permutation=np.random.default_rng(seed+10000).permutation(len(rows.neural))
    return TrialRows(rows.neural,rows.behavior[permutation].copy(),rows.source_id,rows.trial_id,
        rows.time_id,rows.session_id,rows.split),permutation


def training_sources():
    return dict(parent_cnn=rr.sources(),parent_transformer=tr.sources(),data_adapter=sha256(md.__file__),
        functions={f.__name__:rr.code_hash(f) for f in (specs_for,frozen_config,context_bindings,project_scales,shuffled_train)},
        target='active_joint_XY',input_context='unchanged legacy_concat offset10')


def prepare(output=OUTPUT):
    folder=output/'stage00_data'
    marker=folder/'manifest.json'
    parent=rr.ROOT/'results/monkey_figure3_v1/stage00_data/DATA_AND_SPLIT_PROVENANCE.json'
    source=json.loads(parent.read_text())
    dependencies=dict(adapter=sha256(md.__file__),split_metadata=sha256(parent),
        source_hashes={s:source['source_file_hashes'][f'active_{s}.jl']['sha256'] for s in ('train','valid')})
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['dependencies']!=dependencies:raise RuntimeError('Monkey prepared source changed')
        for split,h in saved['cache_hashes'].items():
            if sha256(folder/f'{split}.npz')!=h:raise RuntimeError('Monkey cache changed')
        return folder,saved
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('Partial monkey input cache retained')
    rows=md.load_train_validation(rr.ROOT,'active')
    train=rows['train'].as_trial_rows(rows['train'].position_xy)
    scales=fit_distance_scales(train.metadata(),specs_for('joint'),split='train',source_ids=train.source_id)
    folder.mkdir(parents=True,exist_ok=False)
    for split,values in rows.items():save_cache(folder/f'{split}.npz',values)
    saved=dict(dependencies=dependencies,cache_hashes={s:sha256(folder/f'{s}.npz') for s in rows},
        train_scales=scales,splits={s:r.provenance for s,r in rows.items()},
        prepared_utc=rr.now(),preprocessing_refit=False,test_container_opened=False)
    saved['signature']=config_hash(dict(dependencies=dependencies,cache_hashes=saved['cache_hashes'],train_scales=scales))
    rr.atomic_json(marker,saved)
    return folder,saved


def save_cache(path,rows):
    np.savez_compressed(path,neural=rows.neural,behavior=rows.position_xy,position_xy=rows.position_xy,
        velocity_xy=rows.velocity_xy,direction=rows.direction,direction_active_passive=rows.direction_active_passive,
        passive=rows.passive,source_id=rows.source_id,trial_id=rows.trial_id,time_id=rows.time_id,
        session_id=rows.session_id,split=rows.split,valid_mask=rows.valid_mask)


def prepare_test(folder):
    marker=folder/'test_manifest.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['cache_sha256']!=sha256(folder/'test.npz'):raise RuntimeError('TEST export cache mismatch')
        return saved
    if (folder/'test.npz').exists():raise RuntimeError('Partial TEST cache retained')
    rows=md.load_test_for_export(rr.ROOT,'active')
    save_cache(folder/'test.npz',rows)
    saved=dict(parent=rows.provenance,cache_sha256=sha256(folder/'test.npz'),only_for_downstream_export=True)
    rr.atomic_json(marker,saved)
    return saved


def exported_arrays(rows,extra_labels,config,model,device):
    for key,value in extra_labels.items():
        array=np.asarray(value)
        if array.ndim<1 or len(array)!=len(rows.neural):raise ValueError(f'Metadata cardinality mismatch:{key}')
        if key in ('position_xy','velocity_xy') and array.shape!=(len(rows.neural),2):raise ValueError(f'Metadata XY shape mismatch:{key}')
        if key in ('direction','direction_active_passive','passive') and array.shape!=(len(rows.neural),):raise ValueError(f'Metadata categorical shape mismatch:{key}')
        if not np.isfinite(array).all():raise ValueError(f'Nonfinite metadata:{key}')
    values=rr.tensors(rows,device)
    parts=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows.neural),config['batch_size']):
            z,_=rr.encode_batch(model,values,torch.arange(start,min(start+config['batch_size'],len(rows.neural))))
            parts.append(z.cpu().numpy())
    raw=np.concatenate(parts)
    norms=np.linalg.norm(raw,axis=1,keepdims=True)
    if raw.shape!=(len(rows.neural),config['output_dimension']) or not np.isfinite(raw).all() or np.any(norms<=0):
        raise RuntimeError('Monkey embedding shape/finiteness invariant')
    mask=np.zeros(len(raw),dtype=bool);mask[eligible_centers(rows)]=True
    return dict(embedding_raw=raw,embedding_unit=raw/norms,behavior=rows.behavior,
        **{k:getattr(rows,k) for k in ('trial_id','time_id','source_id','session_id')},
        **extra_labels,valid_mask=mask,split=np.asarray(rows.split),seed=config['seed'],
        model=config['architecture'],objective=config['version'],condition=config['condition'])


def export_cell(config,fit_folder,fit,data_folder,data_record):
    folder=OUTPUT/'stage02_embedding'/config['trial_id']
    marker=folder/'manifest.json'
    test_record=prepare_test(data_folder)
    signature=config_hash(dict(checkpoint=fit['checkpoint_sha256'],data=data_record['signature'],
        test=test_record['cache_sha256'],exporter=rr.code_hash(exported_arrays),
        context=rr.code_hash(rr.encode_batch),config=config))
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['signature']!=signature or any(sha256(folder/n)!=h for n,h in saved['artifact_hashes'].items()):
            raise RuntimeError('Completed monkey embedding dependencies changed')
        return folder,saved
    if folder.exists():raise RuntimeError('Partial monkey embedding output retained')
    device=torch.device('cuda')
    with context_bindings(config):
        model=rr.make_encoder(65,config,device)
    model.load_state_dict(torch.load(fit_folder/'checkpoint.pt',map_location=device,weights_only=False)['model'])
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    n=0
    for split in ('train','valid','test'):
        rows=load_rows(data_folder/f'{split}.npz')
        with np.load(data_folder/f'{split}.npz',allow_pickle=False) as raw:
            labels={k:raw[k] for k in ('position_xy','velocity_xy','direction','direction_active_passive','passive')}
        payload=exported_arrays(rows,labels,config,model,device)
        np.savez_compressed(folder/f'{split}_embedding.npz',**payload)
        n+=len(rows.neural)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-started
    saved=dict(signature=signature,config=config,parent_checkpoint_sha256=fit['checkpoint_sha256'],
        created_utc=rr.now(),inference_seconds=seconds,inference_samples_per_second=n/seconds,
        metadata_policy='original unshuffled behavior for exports and downstream decoding',
        raw_semantics='native encoder output, already L2 normalized; raw/unit kept separate',
        neural_cache=str(data_folder),parent_data_signature=data_record['signature'],
        artifact_hashes={f'{s}_embedding.npz':sha256(folder/f'{s}_embedding.npz') for s in ('train','valid','test')})
    rr.atomic_json(marker,saved)
    return folder,saved


def metrics_cell(config,embedding_folder):
    from .downstream import _LazyArchive
    from .monkey_downstream import score_arrays
    folder=OUTPUT/'stage03_metrics'/config['trial_id']
    marker=folder/'manifest.json'
    signature=config_hash(dict(embedding=sha256(embedding_folder/'manifest.json'),
        scorer=sha256(Path(__file__).with_name('monkey_downstream.py'))))
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['signature']!=signature or any(sha256(folder/n)!=h for n,h in saved['artifact_hashes'].items()):
            raise RuntimeError('Monkey metrics dependencies changed')
        return folder
    if folder.exists():raise RuntimeError('Partial monkey metric output retained')
    result=score_arrays({s:_LazyArchive(embedding_folder/f'{s}_embedding.npz') for s in ('train','valid','test')})
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'predictions.npz',**result.pop('predictions'))
    rr.atomic_json(folder/'RESULTS.json',dict(config=config,**result))
    rr.atomic_json(marker,dict(signature=signature,artifact_hashes={p.name:sha256(p) for p in folder.iterdir()}))
    return folder


def diagnostic_cell(config,embedding_folder,train,scales,permutation):
    folder=OUTPUT/'stage04_diagnostics'/config['trial_id']
    marker=folder/'manifest.json'
    signature=config_hash(dict(checkpoint_embedding=sha256(embedding_folder/'manifest.json'),
        function=rr.code_hash(diagnostic_cell),scales=scales,permutation=config_hash(permutation.tolist()) if permutation is not None else None))
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['signature']!=signature or any(sha256(folder/n)!=h for n,h in saved['artifact_hashes'].items()):
            raise RuntimeError('Monkey diagnostic dependencies changed')
        return
    if folder.exists():raise RuntimeError('Partial monkey diagnostic output retained')
    ids=torch.randperm(len(train.neural),generator=torch.Generator().manual_seed(config['seed']+30000))[:64].numpy()
    with np.load(embedding_folder/'train_embedding.npz',allow_pickle=False) as saved:
        if not np.array_equal(saved['source_id'],train.source_id):raise RuntimeError('Diagnostic source mismatch')
        z=torch.as_tensor(saved['embedding_raw'][ids])
    batch=train.metadata(ids)
    with torch.no_grad():
        result=evaluate_soft(z,batch,config['metadata_specs'],version=config['version'],scales=scales,
            temperature=config['embedding_temperature'],metadata_temperature=config['metadata_temperature'])
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'TRAIN_REFERENCE_W_Q_P.npz',W=result.W.numpy(),Q=result.Q.numpy(),P=result.P.numpy(),
        reference_train_row_indices=ids,**{k:v.numpy() for k,v in batch.items()},
        behavior_donor_source_id=train.source_id[permutation[ids]] if permutation is not None else train.source_id[ids])
    entropy=result.diagnostics()['Q_entropy_mean']
    rr.atomic_json(folder/'LOSS_DIAGNOSTICS.json',dict(config=config,split='train',loss=float(result.loss),
        target_entropy=entropy,kl_gap=float(result.loss)-entropy,**result.diagnostics(),
        scales=scales,selection=False,source_sha256=sha256(__file__)))
    rr.atomic_json(marker,dict(signature=signature,artifact_hashes={p.name:sha256(p) for p in folder.iterdir()}))


def write_tables(state):
    from .behavior_only_runner import write_csv
    records=[];compute=[]
    for key,entry in state['records'].items():
        if entry.get('status')!='complete':continue
        fit=json.loads(Path(entry['fit_manifest']).read_text())
        embedding=json.loads(Path(entry['embedding_manifest']).read_text())
        result=json.loads(Path(entry['metrics']).read_text())
        c=fit['config']
        common=dict(trial_id=key,architecture=c['architecture'],condition=c['condition'],
            shuffle=c['shuffle'],dimension=c['output_dimension'],seed=c['seed'])
        for row in result['records']:
            records.append(dict(**common,**{k:v for k,v in row.items() if k!='position_coordinate_mae'}))
        compute.append(dict(**common,**{k:fit.get(k) for k in ('parameters','actual_updates','actual_optimizer_updates',
            'fit_seconds','seconds_per_100_updates','updates_per_second','training_windows_per_second',
            'peak_gpu_allocated','peak_gpu_reserved','device','cuda_available','torch_version')},
            inference_seconds=embedding['inference_seconds'],inference_samples_per_second=embedding['inference_samples_per_second']))
    write_csv(OUTPUT/'DECODING_LONG.csv',records)
    write_csv(OUTPUT/'COMPUTE_PERFORMANCE.csv',compute)


def qualify():
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromNames(['user_soft.tests.test_monkey_runner',
        'user_soft.tests.test_monkey_data','user_soft.tests.test_monkey_downstream'])
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record=dict(decision='GO' if result.wasSuccessful() else 'NO-GO',tests_run=result.testsRun,
        sources=training_sources(),test_sources={name:sha256(Path(__file__).parent/'tests'/name) for name in
            ('test_monkey_runner.py','test_monkey_data.py','test_monkey_downstream.py')},
        result_output=stream.getvalue(),created_utc=rr.now(),production_optimizer_updates=0)
    rr.atomic_json(OUTPUT/'qualification/GATE.json',record)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise RuntimeError('Monkey qualification failed')


def execute(data_folder,parent,sources):
    gate=json.loads((OUTPUT/'qualification/GATE.json').read_text())
    if gate['decision']!='GO' or gate['sources']!=sources:raise RuntimeError('Monkey source gate stale')
    if any(sha256(Path(__file__).parent/'tests'/n)!=h for n,h in gate['test_sources'].items()):raise RuntimeError('Monkey test gate stale')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda');torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    lock=OUTPUT/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=OUTPUT/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(planned=250,records={},completed=0)
    if state.get('last_error'):state.setdefault('previous_errors',[]).append(state.pop('last_error'))
    try:
        for architecture,config in enumerate_configs():
            if (OUTPUT/'STOP_AFTER_CURRENT').exists():
                state['status']='stopped_after_current';break
            key=config['trial_id'];fit_folder=OUTPUT/'stage01_fit'/key
            state.update(status='running',current=key,pid=os.getpid(),updated_utc=rr.now());rr.atomic_json(state_path,state)
            train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
            scales=project_scales(parent['train_scales'],config['condition']);permutation=None
            if config['shuffle']:
                train,permutation=shuffled_train(train,config['seed'])
                perm_folder=OUTPUT/'stage00_shuffle'/f"seed_{config['seed']}"
                perm_path=perm_folder/'permutation.npz'
                if not perm_path.exists():
                    perm_folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(perm_path,permutation=permutation,source_id=train.source_id,
                        behavior_donor_source_id=train.source_id[permutation],shuffle_seed=config['shuffle_seed'])
                    rr.atomic_json(perm_folder/'PROVENANCE.json',dict(seed=config['seed'],split='train',
                        parent_train_sha256=parent['cache_hashes']['train'],permutation_sha256=sha256(perm_path),
                        shuffle_unit='joint XY row, original neural/time/trial unchanged'))
                else:
                    saved=json.loads((perm_folder/'PROVENANCE.json').read_text())
                    if sha256(perm_path)!=saved['permutation_sha256']:raise RuntimeError('Shuffle permutation changed')
                    with np.load(perm_path) as p:
                        if not np.array_equal(p['permutation'],permutation):raise RuntimeError('Shuffle reproducibility failed')
            dependency=config_hash(dict(data=parent['signature'],scales=scales,
                permutation=config_hash(permutation.tolist()) if permutation is not None else None))
            with context_bindings(config):
                fit,created=rr.fit_cell(fit_folder,train,valid,config,scales,dependency,sources,device)
            state['records'][key]=dict(status='fit_complete',fit_manifest=str(fit_folder/'manifest.json'),
                checkpoint_sha256=fit['checkpoint_sha256'],new_fit=created)
            rr.atomic_json(state_path,state)
            emb,embedding=export_cell(config,fit_folder,fit,data_folder,parent)
            metrics=metrics_cell(config,emb)
            diagnostic_cell(config,emb,train,scales,permutation)
            state['records'][key].update(status='complete',embedding_manifest=str(emb/'manifest.json'),
                metrics=str(metrics/'RESULTS.json'))
            state['completed']=sum(r['status']=='complete' for r in state['records'].values())
            state['updated_utc']=rr.now();rr.atomic_json(state_path,state);write_tables(state)
            if state['completed']%20==0 or state['completed']==250:
                print(json.dumps(dict(completed=state['completed'],planned=250,current=key)),flush=True)
        else:state['status']='complete'
    except BaseException as exc:
        state.update(status='failed',last_error=repr(exc));raise
    finally:
        state['updated_utc']=rr.now();rr.atomic_json(state_path,state);lock.unlink(missing_ok=True)


def wait_for_predecessors():
    rat=rr.ROOT/'results/user_soft_rat_behavior_only_shuffle_v1'
    correction=rr.ROOT/'results/monkey_figure3_paper_v1/figure3g_position_only_correction_v1'
    while True:
        r=json.loads((rat/'CAMPAIGN_STATE.json').read_text())
        c=json.loads((correction/'CORRECTION_STATE.json').read_text()) if (correction/'CORRECTION_STATE.json').exists() else {}
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json',dict(status='waiting_predecessors',rat_completed=r.get('completed'),
            correction_status=c.get('status','queued'),pid=os.getpid(),updated_utc=rr.now()))
        if r.get('status') in ('failed','interrupted','stopped_after_current') or c.get('status')=='failed':
            raise RuntimeError('Predecessor stopped; no monkey fit launched')
        if r.get('status')=='complete' and r.get('completed')==144 and not(rat/'RUNNING.lock').exists() and c.get('status')=='complete' and not(correction/'RUNNING.lock').exists():return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true');modes.add_argument('--qualify',action='store_true')
    modes.add_argument('--wait-and-execute',action='store_true');modes.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.qualify:qualify();return
    folder,parent=prepare();sources=training_sources()
    plan=dict(configs=[dict(architecture=a,config=c) for a,c in enumerate_configs()],sources=sources,
        data_signature=parent['signature'],population='active',behavior='native jointXY',
        observed=150,shuffle=100,maximum_seeds=6,actual_seeds=list(SEEDS),
        hyperparameter_selection=False,ED8_unknown_recipe_panels='deferred_by_user',
        source_commit_before=os.environ.get('GIT_COMMIT','not_recorded_by_environment'))
    path=OUTPUT/'CAMPAIGN_PLAN.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise RuntimeError('Monkey sealed plan changed')
    if not path.exists():rr.atomic_json(path,plan)
    if not(args.execute or args.wait_and_execute):
        print(json.dumps(dict(planned=250,observed=150,shuffle=100,optimizer_updates=0,output=str(OUTPUT))),flush=True);return
    queue_lock=OUTPUT/'QUEUE.lock'
    fd=os.open(queue_lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if args.wait_and_execute:wait_for_predecessors()
        else:
            correction=rr.ROOT/'results/monkey_figure3_paper_v1/figure3g_position_only_correction_v1/CORRECTION_STATE.json'
            if not correction.exists() or json.loads(correction.read_text()).get('status')!='complete':raise RuntimeError('Complete CEBRA correction first')
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json',dict(status='running_monkey',pid=os.getpid(),updated_utc=rr.now()))
        execute(folder,parent,sources)
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json',dict(status='finished',updated_utc=rr.now()))
    except BaseException as exc:
        rr.atomic_json(OUTPUT/'SEQUENCE_STATE.json',dict(status='failed',error=repr(exc),updated_utc=rr.now()));raise
    finally:queue_lock.unlink(missing_ok=True)


if __name__=='__main__':main()
