"""Reproduce Achilles 8D 2c loss fits for Buddy, Cicero, and Gatsby."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import run_cebra as base

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reproduction/figure2_bc/figure2c_8d_other_rats'
FIG=ROOT/'results/figura_2_2026-10-28/c'
CONDITIONS=('position','direction_paper_demo','position_direction','position_shuffled',
            'direction_shuffled_paper_demo','position_direction_shuffled','direction_controlled','direction_shuffled_controlled')
LABELS=('Position','Direction','Position + Direction','Position shuffled','Direction shuffled','Position + Direction shuffled')
RATS=('buddy','cicero','gatsby')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def train(rat,name,x,y):
    folder=OUT/rat/name/'seed_42';folder.mkdir(parents=True,exist_ok=True)
    params=base.setup(name,10000);params['output_dimension']=8
    manifest=folder/'manifest.json';source=ROOT/'data/rat_hippocampus'/f'{rat}.jl'
    if manifest.exists():
        m=json.loads(manifest.read_text(encoding='utf-8'))
        if m['parameters']!=params or m['source_sha256']!=sha(source) or m['cebra_version']!=cebra.__version__:raise ValueError(f'Incompatible {manifest}')
        if any(sha(folder/key)!=value for key,value in m['sha256'].items()):raise ValueError('Artifact changed')
        print('REUSE',rat,name,flush=True);return
    if any(folder.iterdir()):raise RuntimeError(f'Partial run requires inspection: {folder}')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    targets=y if name.startswith('position_direction') else (y[:,1] if name.startswith('direction') else y[:,0])
    perm=np.random.default_rng(42).permutation(len(y)) if 'shuffled' in name else np.empty(0,dtype=np.int64)
    if len(perm):targets=targets[perm]
    (folder/'RUNNING.json').write_text(json.dumps({'pid':os.getpid(),'rat':rat,'condition':name,'params':params},indent=2),encoding='utf-8')
    print('START',rat,name,flush=True);start=time.time()
    try:
        model=cebra.CEBRA(**params);model.fit(x,targets)
        z=model.transform(x);loss=np.asarray(model.state_dict_['log']['total'],dtype=float)
        if z.shape!=(len(x),8) or len(loss)!=10000 or not np.isfinite(z).all() or not np.isfinite(loss).all():raise ValueError('Invalid fit outputs')
        model.save(str(folder/'checkpoint.pt'))
        np.savez_compressed(folder/'embedding.npz',embedding=z,behavior=y,source_id=np.arange(len(x)),shuffle_permutation=perm)
        with (folder/'loss.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(('update','loss'));w.writerows(enumerate(loss,1))
        rec=dict(status='complete',rat=rat,condition=name,seed=42,parameters=params,input_shape=list(x.shape),
                 label_shape=list(targets.shape),label_dtype=str(targets.dtype),fit_scope='full_recording',
                 source_sha256=sha(source),script_sha256=sha(Path(__file__)),cebra_version=cebra.__version__,
                 torch_version=torch.__version__,elapsed_seconds=time.time()-start,loss_final=float(loss[-1]),
                 sha256={k:sha(folder/k) for k in ('checkpoint.pt','embedding.npz','loss.csv')})
        manifest.write_text(json.dumps(rec,indent=2),encoding='utf-8');(folder/'RUNNING.json').unlink()
        print('COMPLETE',rat,name,round(rec['elapsed_seconds'],1),flush=True)
    except Exception:
        (folder/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


def plot(rat,mode):
    names=[('direction_'+mode if c=='direction' else 'direction_shuffled_'+mode if c=='direction_shuffled' else c)
           for c in ('position','direction','position_direction','position_shuffled','direction_shuffled','position_direction_shuffled')]
    fig,ax=plt.subplots(figsize=(7,4.5));refs=[]
    for i,(name,label) in enumerate(zip(names,LABELS)):
        folder=OUT/rat/name/'seed_42';m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        path=folder/'loss.csv'
        if sha(path)!=m['sha256']['loss.csv']:raise ValueError('Loss hash mismatch')
        a=np.loadtxt(path,delimiter=',',skiprows=1)
        if a.shape!=(10000,2) or not np.isfinite(a).all():raise ValueError('Invalid loss')
        ax.plot(a[:,0],a[:,1],color='gray' if i>=3 else 'deepskyblue',alpha=(.35,.65,1)[i%3],lw=.8,label=label)
        refs.append({'condition':name,'loss':str(path.relative_to(ROOT)),'sha256':sha(path),'manifest':str((folder/'manifest.json').relative_to(ROOT))})
    ax.set_xlabel('Update');ax.set_ylabel('InfoNCE loss');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();dest=FIG/rat.capitalize();dest.mkdir(parents=True,exist_ok=True)
    stem=dest/f'Figura_2c_{rat.capitalize()}_CEBRA_{mode}_loss'
    for ext in ('.png','.pdf','.svg'):
        path=stem.with_suffix(ext)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=220,bbox_inches='tight')
    plt.close(fig)
    (dest/f'RIFERIMENTI_CEBRA_{mode}.json').write_text(json.dumps({'rat':rat,'seed':42,'source':refs,'fit_scope':'full recording','fit_dimension':8,'updates':10000},indent=2),encoding='utf-8')


def main():
    torch.set_num_threads(2)
    for rat in RATS:
        ds=cebra.datasets.init(f'rat-hippocampus-single-{rat}');x=ds.neural.numpy();y=ds.continuous_index.numpy()
        if x.ndim!=2 or y.shape!=(len(x),3) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError(f'Invalid data {rat}')
        for name in CONDITIONS:train(rat,name,x,y)
        for mode in ('paper_demo','controlled'):plot(rat,mode)
        print('FINISHED RAT',rat,flush=True)


if __name__=='__main__':main()
