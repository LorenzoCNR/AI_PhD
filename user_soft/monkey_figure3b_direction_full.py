"""Soft Direction Figure 3b: separate full active-data fits, resumable."""
import csv, json, random, time
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import torch
from . import rat_runner as rr, transformer_runner as tr
from .full_recording_runner import save_checkpoint, make_encoder, write_history
from .data import config_hash, sha256, eligible_centers
from .loss import fit_distance_scales, evaluate_soft

ROOT=rr.ROOT
SRC=ROOT/'data/monkey_reaching_preload_smth_40/active_all.jl'
OUT=ROOT/'results/user_soft_monkey_figure3b_direction_4d_v1'
SPECS=[dict(key='behavior',geometry='categorical',weight=1.)]

def main():
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=joblib.load(SRC); n=int(data['num_trials']); length=int(data['trial_len'])
    x=np.asarray(data['spikes'],dtype=np.float32)
    direction=np.repeat(np.asarray(data['movement_dir'],dtype=np.int64),length)
    if (n,length,x.shape)!=(193,600,(115800,65)) or set(np.unique(direction))!=set(range(8)):
        raise ValueError('Unexpected source or labels')
    rows=SimpleNamespace(neural=x,behavior=direction[:,None],source_id=np.arange(len(x)),
        trial_id=np.repeat(np.arange(n),length),time_id=np.tile(np.arange(length),n),
        session_id=np.zeros(len(x),dtype=np.int64))
    values=rr.tensors(rows,device)
    scales=fit_distance_scales({'behavior':values['behavior']},SPECS,split='full_recording',source_ids=rows.source_id)
    for architecture in ('cnn','transformer'):
        factory=rr.frozen_config if architecture=='cnn' else tr.frozen_config
        cfg=factory('monkey_active',0,42,'soft_v1_stable',10000)
        cfg.update(condition='direction',fit_scope='active_all_full_sample',output_dimension=4,
            metadata_specs=SPECS,embedding_temperature=1.,metadata_temperature=.5,
            batch_size=512,learning_rate=3e-4,max_updates=10000,time_bin_ms=1,
            input_policy='legacy_concat',offset_left=5,offset_right=5,early_stopping=False)
        signature=config_hash(dict(config=cfg,source_sha256=sha256(SRC),script_sha256=sha256(Path(__file__)),scales=scales))
        folder=OUT/architecture/'seed_42'; checkpoint=folder/'checkpoint.pt'; manifest=folder/'manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text(encoding='utf8'))
            if m['signature']!=signature or sha256(checkpoint)!=m['checkpoint_sha256'] or sha256(folder/'embedding.npz')!=m['embedding_sha256']: raise RuntimeError('Completed output mismatch')
            print('REUSE',architecture,flush=True); continue
        if folder.exists() and any(folder.iterdir()) and not checkpoint.exists(): raise RuntimeError('Partial output; inspect')
        folder.mkdir(parents=True,exist_ok=True)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if device.type=='cuda': torch.cuda.manual_seed_all(42)
        model=make_encoder(architecture,65,cfg,device)
        opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
        gen=torch.Generator().manual_seed(42); step=0; history=[]
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location=device,weights_only=False)
            if state['signature']!=signature: raise RuntimeError('Resume signature mismatch')
            model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);gen.set_state(state['generator'].cpu())
            torch.set_rng_state(state['torch_rng'].cpu())
            if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
            step=state['step'];history=state['history']
        print('START',architecture,'step',step,'device',device,flush=True)
        started=time.perf_counter();model.train()
        while step<10000:
            idx=torch.randint(len(x),(512,),generator=gen)
            opt.zero_grad(set_to_none=True);z,batch=rr.encode_batch(model,values,idx)
            result=evaluate_soft(z,batch,SPECS,version='soft_v1_stable',scales=scales,
                temperature=1.,metadata_temperature=.5,allow_full_recording_scales=True)
            if not torch.isfinite(result.loss):raise FloatingPointError('Nonfinite loss')
            result.loss.backward();opt.step();step+=1
            history.append(dict(update=step,train_loss=float(result.loss.detach())))
            if step%500==0:
                save_checkpoint(checkpoint,dict(signature=signature,model=model.state_dict(),optimizer=opt.state_dict(),
                    generator=gen.get_state(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,step=step,history=history,config=cfg))
                write_history(folder/'training_history.csv',history)
                print('UPDATE',architecture,step,flush=True)
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,len(x),512):
                z,_=rr.encode_batch(model,values,torch.arange(start,min(start+512,len(x))));parts.append(z.cpu().numpy())
        emb=np.concatenate(parts); valid=np.zeros(len(x),dtype=bool);valid[eligible_centers(rows)]=True
        if emb.shape!=(len(x),4) or not np.isfinite(emb).all():raise ValueError('Invalid embedding')
        np.savez_compressed(folder/'embedding.npz',embedding=emb,direction=direction,trial_id=rows.trial_id,time_id=rows.time_id,source_id=rows.source_id,valid_mask=valid)
        m=dict(signature=signature,config=cfg,source=str(SRC),source_sha256=sha256(SRC),script_sha256=sha256(Path(__file__)),
            architecture=architecture,actual_updates=step,fit_seconds=time.perf_counter()-started,training_rows=len(x),
            checkpoint_sha256=sha256(checkpoint),embedding_sha256=sha256(folder/'embedding.npz'),
            history_sha256=sha256(folder/'training_history.csv'),scales=scales,device=str(device),
            labels='8 discrete target directions; categorical equality, not ordinal distance',
            note='descriptive full active dataset; not held-out; legacy concatenated input boundaries')
        (folder/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf8')
        print('COMPLETE',architecture,flush=True)

if __name__=='__main__':main()
