"""Resumable Figure 2b/c CEBRA experiments on the complete Achilles recording."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import time
import traceback
from pathlib import Path

import cebra
import cebra.datasets
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reproduction' / 'figure2_bc'
CONDITIONS = ('behavior_position_input', 'behavior_position_direction_input', 'time',
              'position', 'position_direction', 'time_position_5d', 'position_shuffled',
              'direction_paper_demo', 'direction_shuffled_paper_demo',
              'direction_controlled', 'direction_shuffled_controlled', 'position_direction_shuffled')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def data():
    dataset = cebra.datasets.init('rat-hippocampus-single-achilles')
    x, b = dataset.neural.numpy(), dataset.continuous_index.numpy()
    if x.ndim != 2 or b.shape != (len(x), 3) or not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError('Invalid input data')
    if not np.allclose(b[:, 1]+b[:, 2], 1): raise ValueError('Direction not complementary')
    return x, b


def setup(name, steps):
    behavioral = name.startswith('behavior_')
    p = dict(model_architecture='offset10-model-mse' if behavioral else 'offset10-model',
             num_hidden_units=32, output_dimension=2 if behavioral else (5 if name=='time_position_5d' else 3),
             batch_size=512, learning_rate=3e-4, optimizer='adam', criterion='infonce',
             distance='euclidean' if behavioral else 'cosine', temperature_mode='constant',
             temperature=1.12 if name=='time' else 1., max_iterations=steps,
             device='cuda_if_available', verbose=False)
    if name=='time': p.update(conditional='time', time_offsets=10)
    elif not name.endswith('_paper_demo'): p.update(conditional='time_delta', time_offsets=10)
    if name=='time_position_5d': p['hybrid']=True
    return p


def fit_one(name, steps, seed, x, b, mode):
    folder = OUT / mode / name / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder/'manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text(encoding='utf-8'))
        if record['status']!='complete' or record['steps']!=steps: raise RuntimeError(f'Invalid manifest: {manifest}')
        for file, expected in record['sha256'].items():
            if digest(folder/file)!=expected: raise RuntimeError(f'Hash mismatch: {file}')
        print(f'REUSED {name}',flush=True); return
    if any(folder.iterdir()): raise RuntimeError(f'Partial run requires manual review: {folder}')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    params=setup(name,steps)
    rng=np.random.default_rng(seed)
    permutation=rng.permutation(len(b))
    shuffled='shuffled' in name
    if name.startswith('behavior_'):
        inputs=b[:, :1] if name=='behavior_position_input' else b
        labels=b[:, 0]
        description='behavioral inputs; position label; approximate reconstruction'
    else:
        inputs=x
        if name=='time': labels=None
        elif name.startswith('position_direction'): labels=b
        elif name.startswith('direction'): labels=b[:,1]
        else: labels=b[:,0]
        description='neural inputs; full recording'
    if shuffled: labels=labels[permutation]
    status={'status':'running','name':name,'seed':seed,'steps':steps,'pid':os.getpid(),'parameters':params,
            'input_shape':list(inputs.shape),'label_shape':None if labels is None else list(labels.shape),
            'description':description,'shuffle':shuffled,'source_sha256':digest(ROOT/'data/rat_hippocampus/achilles.jl')}
    (folder/'RUNNING.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(f'START {name} {steps} updates',flush=True)
    start=time.time()
    try:
        model=cebra.CEBRA(**params)
        if labels is None: model.fit(inputs)
        else: model.fit(inputs,labels)
        z=model.transform(inputs)
        recorded_loss = model.state_dict_['log']['total']
        if not len(recorded_loss): recorded_loss = model.state_dict_['loss']
        if isinstance(recorded_loss, torch.Tensor): recorded_loss = recorded_loss.detach().cpu().numpy()
        loss=np.asarray(recorded_loss,dtype=float)
        if z.shape != (len(x),params['output_dimension']) or len(loss)!=steps or not np.isfinite(z).all() or not np.isfinite(loss).all(): raise ValueError('Invalid embedding/loss')
        model.save(str(folder/'checkpoint.pt'))
        np.savez_compressed(folder/'embedding.npz',embedding=z,behavior=b,source_id=np.arange(len(x)),shuffle_permutation=permutation if shuffled else np.empty(0,dtype=int))
        with (folder/'loss.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['update','loss']); w.writerows(enumerate(loss,1))
        status.update(status='complete',elapsed_seconds=time.time()-start,cebra_version=cebra.__version__,torch_version=torch.__version__,
                      n_parameters=sum(p.numel() for p in model.model_.parameters()),
                      loss_final=float(loss[-1]),loss_last100_mean=float(np.mean(loss[-100:])),loss_min=float(np.min(loss)),
                      effective_model_parameters=model.get_params(),
                      sha256={f:digest(folder/f) for f in ('checkpoint.pt','embedding.npz','loss.csv')})
        manifest.write_text(json.dumps(status,indent=2,default=str),encoding='utf-8')
        (folder/'RUNNING.json').unlink()
        print(f'COMPLETE {name} {status["elapsed_seconds"]:.1f}s',flush=True)
    except Exception:
        (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8')
        print(f'FAILED {name}',flush=True); raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=('smoke','smoke2','full'),required=True)
    parser.add_argument('--seed',type=int,default=42); parser.add_argument('--condition',choices=CONDITIONS,action='append')
    args=parser.parse_args(); steps=10000 if args.mode=='full' else 10
    x,b=data(); OUT.mkdir(parents=True,exist_ok=True)
    stats={'neural_shape':list(x.shape),'behavior_shape':list(b.shape),'neural_dtype':str(x.dtype),'behavior_dtype':str(b.dtype),
           'neural_range':[float(x.min()),float(x.max())], 'behavior_min':b.min(axis=0).tolist(),'behavior_max':b.max(axis=0).tolist(),
           'neural_nan':int(np.isnan(x).sum()),'behavior_nan':int(np.isnan(b).sum()),'bin_ms':25,'sha256':digest(ROOT/'data/rat_hippocampus/achilles.jl')}
    (OUT/'dataset_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    for name in args.condition or CONDITIONS: fit_one(name,steps,args.seed,x,b,args.mode)


if __name__=='__main__': main()
