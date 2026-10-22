"""Immutable ten-seed potential plan, staged execution ceiling and UTC deadline."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import time
import torch
from . import rat_runner as rr,time_only_runner as tm,transformer_runner as tr
from . import transformer_campaign as tc,shuffle_runner as sh
from .data import sha256,config_hash,load_rows

DEFAULT_OUTPUT=rr.ROOT/'results'/'user_soft_rat_overnight_v1'
TRANSFORMER5=rr.ROOT/'results'/'user_soft_rat_transformer_5seed_v1'
SHUFFLE5=rr.ROOT/'results'/'user_soft_rat_shuffle_5seed_v1'
FAMILIES=('cnn_joint','cnn_time','transformer_joint','transformer_time','shuffle_cnn','shuffle_transformer')


def config_for(family,rat,fold,seed,version):
    if family=='cnn_joint':return rr.frozen_config(rat,fold,seed,version)
    if family=='cnn_time':return tm.frozen_config(rat,fold,seed,version)
    if family=='transformer_joint':return tc.config_for(rat,fold,seed,version,'joint')
    if family=='transformer_time':return tc.config_for(rat,fold,seed,version,'time_only')
    if family=='shuffle_cnn':return sh.frozen_config('cnn',rat,fold,seed,version)
    if family=='shuffle_transformer':return sh.frozen_config('transformer',rat,fold,seed,version)
    raise ValueError('unknown family')


def job_for(family,rat,fold,seed,version):
    return dict(id=f'{family}/{rat}/fold_{fold}/{version}/seed_{seed}',family=family,
                config=config_for(family,rat,fold,seed,version))


def build_jobs():
    return [job_for(family,rat,fold,seed,version) for seed,family,fold,rat,version in
        itertools.product(range(42,52),FAMILIES,range(3),rr.RATS,rr.VERSIONS)]


def execution_jobs():
    jobs=[]
    for seeds,families in (((47,48),('transformer_joint','transformer_time')),
                          (range(42,49),('shuffle_cnn','shuffle_transformer')),
                          ((49,),FAMILIES),((50,),FAMILIES),((51,),FAMILIES)):
        jobs.extend(job_for(family,rat,fold,seed,version) for seed,fold,rat,version,family in
            itertools.product(seeds,range(3),rr.RATS,rr.VERSIONS,families))
    return jobs


def sources():
    return dict(overnight_source_sha256=sha256(__file__),cnn_joint=rr.sources(),cnn_time=tm.sources(),
                transformer=tc.sources(),shuffle=sh.sources())


def training_sources(family,source_record):
    if family=='cnn_joint':return source_record['cnn_joint']
    if family=='cnn_time':return source_record['cnn_time']
    if family.startswith('transformer_'):
        return tc.training_sources('joint' if family=='transformer_joint' else 'time_only',source_record['transformer'])
    return sh.training_sources('cnn' if family=='shuffle_cnn' else 'transformer',source_record['shuffle'])


def artifact_root(output,job):
    family=job['family'];config=job['config'];seed=config['seed']
    if family=='cnn_joint' and seed<=48:return rr.DEFAULT_OUTPUT
    if family=='cnn_time' and seed<=48:return tm.DEFAULT_OUTPUT
    if family.startswith('transformer_') and seed<=46:
        return tc.artifact_root(TRANSFORMER5,'joint' if family=='transformer_joint' else 'time_only',config)
    if family.startswith('shuffle_'):
        architecture='cnn' if family=='shuffle_cnn' else 'transformer'
        old=SHUFFLE5/architecture/'stage01_fit'/cell_key(config)
        if (old/'checkpoint.pt').exists() or (old/'manifest.json').exists():return SHUFFLE5/architecture
    return Path(output)/'families'/family


def cell_key(config):
    return f"{config['rat']}/fold_{config['fold']}/{config['version']}/seed_{config['seed']}"


def can_start(deadline_utc,now=None):
    if deadline_utc is None:return True
    deadline=datetime.fromisoformat(deadline_utc.replace('Z','+00:00')) if isinstance(deadline_utc,str) else deadline_utc
    if deadline.tzinfo is None:raise ValueError('deadline must have explicit UTC offset')
    current=now or datetime.now(timezone.utc)
    if current.tzinfo is None:raise ValueError('current time must be timezone aware')
    return current<deadline


def prerequisite_status():
    states=[]
    for root,count in ((rr.DEFAULT_OUTPUT,168),(tm.DEFAULT_OUTPUT,168),(TRANSFORMER5,240)):
        path=root/'CAMPAIGN_STATE.json'
        if not path.exists():return False,str(root)
        state=json.loads(path.read_text())
        complete=sum(record.get('status')=='complete' for record in state.get('records',{}).values())
        if (state.get('completed')!=count or complete!=count or
                state.get('status') not in ('complete','requested_limit') or (root/'RUNNING.lock').exists()):
            return False,str(root)
        states.append(state)
    for root in (tr.DEFAULT_OUTPUT,SHUFFLE5):
        if (root/'RUNNING.lock').exists():return False,str(root)
    return True,None


def prepare_inputs(output,job):
    family=job['family'];config=job['config'];rat=config['rat'];fold=config['fold']
    if family.startswith('shuffle_'):
        old=SHUFFLE5/'stage00_shuffle'/rat/f'fold_{fold}'/f"seed_{config['seed']}"/'manifest.json'
        shared=SHUFFLE5 if old.exists() else Path(output)/'shared_shuffle'
        folder,parent,_,_,permutation_record=sh.prepare_shuffle(shared,rat,fold,config['seed'])
        return dict(data_folder=str(folder),parent=parent,parent_id=config_hash(permutation_record),
                    shuffle_root=str(shared),shuffle_record=permutation_record)
    if family.endswith('_time'):
        folder,parent=tm.prepare(Path(output)/'shared_time',rat,fold)
        return dict(data_folder=str(folder),parent=parent,parent_id=config_hash(parent))
    folder,parent=rr.prepare(rr.DEFAULT_OUTPUT,rat,fold)
    return dict(data_folder=str(folder),parent=parent,parent_id=sha256(folder/'manifest.json'))


def seal_plan(output,source_record):
    """All potential jobs/configs/parents sealed before any extension fit."""
    inputs={};jobs=[]
    for job in build_jobs():
        config=job['config'];family=job['family']
        kind='shuffle' if family.startswith('shuffle_') else 'time' if family.endswith('_time') else 'joint'
        key=f"{kind}/{config['rat']}/{config['fold']}"+(f"/{config['seed']}" if kind=='shuffle' else '')
        if key not in inputs:inputs[key]=prepare_inputs(output,job)
        entry=dict(**job,input_key=key,parent_id=inputs[key]['parent_id'],
                   training_sources_signature=config_hash(training_sources(family,source_record)))
        jobs.append(entry)
    plan=dict(created_utc=rr.now(),potential_seeds=list(range(42,52)),potential_models=1440,
        sources=source_record,inputs=inputs,jobs=jobs,
        execution_order=[job['id'] for job in execution_jobs()],
        policy='Transformerto7;shuffleto7;then balanced seed49,50,51; external ceiling/deadline')
    path=Path(output)/'POTENTIAL_PLAN.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['jobs']!=jobs or old['sources']!=source_record or old['inputs']!=inputs:
            raise RuntimeError('Immutable overnight potential plan changed')
        return old
    rr.atomic_json(path,plan)
    return plan


def verify_gate(output,source_record):
    gate_path=Path(output)/'qualification'/'GATE.json'
    if not gate_path.exists():raise RuntimeError('Overnight qualification gate missing')
    gate=json.loads(gate_path.read_text())
    if gate.get('decision')!='GO' or gate.get('training_source_signature')!=config_hash(source_record):
        raise RuntimeError('Overnight qualification stale/NO-GO')
    if gate.get('plan_sha256')!=sha256(Path(output)/'POTENTIAL_PLAN.json'):
        raise RuntimeError('Overnight potential plan gate mismatch')
    dependencies=((rr.DEFAULT_OUTPUT,source_record['cnn_joint']),
        (tm.DEFAULT_OUTPUT,source_record['cnn_time']),(TRANSFORMER5,source_record['transformer']),
        (SHUFFLE5,source_record['shuffle']),(tr.DEFAULT_OUTPUT,tr.sources()))
    for root,expected in dependencies:
        record=json.loads((root/'qualification'/'GATE.json').read_text())
        if record.get('decision')!='GO' or record.get('training_source_signature')!=config_hash(expected):
            raise RuntimeError(f'Upstream gate stale/NO-GO: {root}')


def verify_fit(job,inputs,source_record,output):
    folder=artifact_root(output,job)/'stage01_fit'/cell_key(job['config'])
    manifest=folder/'manifest.json'
    if not manifest.exists():return None
    record=json.loads(manifest.read_text())
    expected=config_hash(dict(config=job['config'],parent_id=inputs['parent_id'],
                              sources=training_sources(job['family'],source_record)))
    if record['signature']!=expected or sha256(folder/'checkpoint.pt')!=record['checkpoint_sha256']:
        raise RuntimeError(f'Existing fit dependency/content mismatch: {job["id"]}')
    return record


def record_complete(output,state,job,root,fit,embedding_folder):
    config=job['config'];key=cell_key(config)
    paths=dict(fit_manifest=root/'stage01_fit'/key/'manifest.json',
        embedding_manifest=embedding_folder/'manifest.json',metrics=root/'stage03_metrics'/key/'metrics.json')
    embedding_record=json.loads(paths['embedding_manifest'].read_text())
    if embedding_record.get('parent_checkpoint_sha256')!=fit['checkpoint_sha256'] or embedding_record.get('config')!=config:
        raise RuntimeError('Indexed embedding parent/config mismatch')
    for split,entry in embedding_record['embeddings'].items():
        if sha256(embedding_folder/f'{split}_embedding.npz')!=entry['embedding_sha256']:
            raise RuntimeError('Indexed embedding content hash mismatch')
    metric_record=json.loads(paths['metrics'].read_text())
    metric_signature=config_hash(dict(parent=sha256(paths['embedding_manifest']),
                                     scorer=sha256(Path(__file__).with_name('downstream.py'))))
    if metric_record.get('signature')!=metric_signature or metric_record.get('config')!=config:
        raise RuntimeError('Indexed metric parent/source/config mismatch')
    diagnostics=root/'stage04_diagnostics'/key
    diagnostic_record=json.loads((diagnostics/'STAGE_SIGNATURE.json').read_text())
    for name,digest in diagnostic_record['artifact_hashes'].items():
        if sha256(diagnostics/name)!=digest:raise RuntimeError('Indexed diagnostic content mismatch')
    entry=dict(status='complete',family=job['family'],seed=config['seed'],rat=config['rat'],fold=config['fold'],
        version=config['version'],artifact_root=str(root),checkpoint_sha256=fit['checkpoint_sha256'])
    for name,path in paths.items():entry[name+'_path']=str(path);entry[name+'_sha256']=sha256(path)
    state['records'][job['id']]=entry
    write_summary(output,state)


def write_summary(output,state):
    rows=[]
    for seed,family in itertools.product(range(42,52),FAMILIES):
        entries=[r for r in state['records'].values() if r.get('status')=='complete' and r.get('seed')==seed and r.get('family')==family]
        rows.append(dict(family=family,seed=seed,completed=len(entries),expected=24,status='complete' if len(entries)==24 else 'partial' if entries else 'pending'))
    state['groups']={family:dict(completed=sum(row['completed'] for row in rows if row['family']==family),
        complete_seeds=[row['seed'] for row in rows if row['family']==family and row['status']=='complete']) for family in FAMILIES}
    state['completed']=sum(r.get('status')=='complete' for r in state['records'].values())
    state['updated_utc']=rr.now()
    rr.atomic_json(Path(output)/'CAMPAIGN_STATE.json',state)
    tmp=Path(output)/'SIX_GROUP_PROGRESS.tmp'
    with tmp.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tmp.replace(Path(output)/'SIX_GROUP_PROGRESS.csv')


def finish_downstream(output,job,inputs,fit,root,device):
    config=job['config'];family=job['family'];folder=root/'stage01_fit'/cell_key(config)
    data_folder=Path(inputs['data_folder'])
    exporter=tr.export_cell if 'transformer' in family else rr.export_cell
    embedding_folder,embedding_record=exporter(root,folder,data_folder,config,fit,device)
    rr.decode_cell(root,embedding_folder,config,embedding_record)
    if family.startswith('shuffle_'):
        _,parent,shuffled,permutation,record=sh.prepare_shuffle(Path(inputs['shuffle_root']),config['rat'],config['fold'],config['seed'])
        sh.diagnostic_cell(root,embedding_folder,load_rows(data_folder/'train.npz'),shuffled,permutation,parent,config,record)
    elif family.endswith('_time'):tm.diagnostic_cell(root,embedding_folder,inputs['parent'],config)
    else:rr.diagnostic_cell(root,embedding_folder,inputs['parent'],config)
    return embedding_folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--dry-run',action='store_true');group.add_argument('--execute',action='store_true')
    parser.add_argument('--max-seeds',type=int,choices=(7,8,9,10),default=7)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    can_start(args.deadline_utc)
    output=rr.checked_output(args.output)
    for protected in (rr.DEFAULT_OUTPUT,tm.DEFAULT_OUTPUT,tr.DEFAULT_OUTPUT,TRANSFORMER5,SHUFFLE5):
        if output==protected.resolve() or protected.resolve() in output.parents:raise ValueError('Overnight output must be separate')
    output.mkdir(parents=True,exist_ok=True)
    source_record=sources();plan=seal_plan(output,source_record)
    if not args.execute:
        print(json.dumps(dict(potential_jobs=1440,baseline_references=576,execution_jobs=864,
            seven_seed_extension_jobs=432,optimizer_updates=0,output=str(output)),indent=2),flush=True)
        return
    verify_gate(output,source_record)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    lock=output/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state_path=output/'CAMPAIGN_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(records={},created_utc=rr.now())
    directive=dict(created_utc=rr.now(),max_seeds=args.max_seeds,deadline_utc=args.deadline_utc,
        plan_sha256=sha256(output/'POTENTIAL_PLAN.json'),policy='deadline checked before every new/partial fit; in-flight completes')
    rr.atomic_json(output/'execution_directives'/f'{time.time_ns()}.json',directive)
    state['execution_directive']=directive
    try:
        while True:
            ready,pending=prerequisite_status()
            if ready:break
            state.update(status='waiting_for_parent',pending_parent=pending);write_summary(output,state)
            if not can_start(args.deadline_utc):
                state['status']='deadline';return
            if not args.wait:return
            time.sleep(15)
        device=torch.device(args.device);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
        by_id={entry['id']:entry for entry in plan['jobs']}
        future={job['id'] for job in execution_jobs()}
        # Existing baseline artifacts are verified; no optimizer or embedding
        # regeneration is necessary simply to make them visible in the index.
        for job in build_jobs():
            if job['id'] in future:continue
            entry=by_id[job['id']];inputs=plan['inputs'][entry['input_key']]
            fit=verify_fit(job,inputs,source_record,output)
            if fit is None:raise RuntimeError(f'Missing expected baseline fit: {job["id"]}')
            root=artifact_root(output,job);embedding=root/'stage02_embedding'/cell_key(job['config'])
            for stage in (embedding/'manifest.json',root/'stage03_metrics'/cell_key(job['config'])/'metrics.json'):
                if not stage.exists():raise RuntimeError(f'Missing baseline downstream artifact: {stage}')
            record_complete(output,state,job,root,fit,embedding)
        for job in execution_jobs():
            if job['config']['seed']>=42+args.max_seeds:
                state['status']='execution_ceiling';break
            entry=by_id[job['id']];inputs=plan['inputs'][entry['input_key']]
            root=artifact_root(output,job);config=job['config'];family=job['family']
            state.update(current=job['id'],status='running');write_summary(output,state)
            fit=verify_fit(job,inputs,source_record,output)
            if fit is None:
                # Includes a saved but incomplete checkpoint: resumption starts
                # optimizer work and therefore also requires deadline clearance.
                if not can_start(args.deadline_utc):
                    state['status']='deadline';break
                data_folder=Path(inputs['data_folder']);train=load_rows(data_folder/'train.npz');valid=load_rows(data_folder/'valid.npz')
                if family.startswith('shuffle_'):
                    train,_=sh.shuffle_train(train,config['seed'])
                with ExitStack() as stack:
                    if 'transformer' in family:stack.enter_context(tr.architecture_bindings())
                    if family.endswith('_time'):stack.enter_context(tm.context_bindings())
                    # Recheck after input/context construction, immediately
                    # before entering a fresh fit transaction.
                    if not can_start(args.deadline_utc):
                        state['status']='deadline';break
                    fit,_=rr.fit_cell(root/'stage01_fit'/cell_key(config),train,valid,config,
                        inputs['parent']['train_scales'],inputs['parent_id'],training_sources(family,source_record),device)
            embedding=finish_downstream(output,job,inputs,fit,root,device)
            record_complete(output,state,job,root,fit,embedding)
            print(json.dumps(dict(completed=state['completed'],last=job['id'],groups=state['groups'])),flush=True)
        else:state['status']='complete'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',last_error=repr(exc))
        if state.get('current'):state['records'].setdefault(state['current'],{}).update(error=repr(exc))
        raise
    finally:
        write_summary(output,state);lock.unlink(missing_ok=True)


if __name__=='__main__':main()
