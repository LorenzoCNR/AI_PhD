"""Decode frozen matching Soft 3h fits; TRAIN-fitted kNN, TEST-only score."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
TASKS={'position':'monkey_fig3h_soft_position_v1','direction':'monkey_fig3h_soft_direction_v1','ap':'monkey_fig3h_soft_ap_v1'}
DIMS=(3,4,8,16,32); SEEDS=(42,43,44,45,46); ARCHS=('cnn','transformer')
OUT=ROOT/'results/monkey_fig3h_soft_matched_decoding_v1'
PLOT=ROOT/'results/figura_3_2026-10-28/h'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path,dim):
    with np.load(path,allow_pickle=False) as f:
        d={k:f[k] for k in ('embedding_raw','valid_mask','position_xy','direction','passive','trial_id','source_id')}
    n=len(d['source_id']); mask=d['valid_mask'].astype(bool)
    if d['embedding_raw'].shape!=(n,dim) or not mask.any() or len(np.unique(d['source_id']))!=n:raise ValueError(f'Invalid {path}')
    return {k:v[mask] for k,v in d.items() if k!='valid_mask'}
def knn(train,test,labels,device,k=36):
    x=torch.nn.functional.normalize(torch.as_tensor(train,dtype=torch.float32,device=device),dim=1)
    y=torch.as_tensor(labels,device=device);out=[]
    with torch.no_grad():
        for start in range(0,len(test),128):
            q=torch.nn.functional.normalize(torch.as_tensor(test[start:start+128],dtype=torch.float32,device=device),dim=1)
            inds=(q@x.T).topk(k,dim=1,largest=True,sorted=True).indices
            if y.ndim==2: pred=y[inds].mean(dim=1)
            else:
                classes=torch.unique(y,sorted=True); votes=(y[inds][...,None]==classes).sum(dim=1);pred=classes[votes.argmax(dim=1)]
            out.append(pred.cpu().numpy())
    return np.concatenate(out)
def run_one(task,arch,dim,seed,device):
    folder=ROOT/'results'/TASKS[task]/'stage02_embedding'/arch/f'dim_{dim}'/f'seed_{seed}'
    src={split:folder/f'{split}_embedding.npz' for split in ('train','test')}
    if any(not p.is_file() for p in src.values()):raise FileNotFoundError(folder)
    hashes={s:sha(p) for s,p in src.items()}
    dest=OUT/task/arch/f'dim_{dim}'/f'seed_{seed}';result=dest/'RESULT.json'
    if result.exists():
        old=json.loads(result.read_text(encoding='utf8'))
        if old['source_sha256']!=hashes:raise RuntimeError(f'Parent changed {result}')
        return old
    train=load(src['train'],dim);test=load(src['test'],dim)
    if np.intersect1d(train['source_id'],test['source_id']).size or np.intersect1d(train['trial_id'],test['trial_id']).size:raise ValueError('TRAIN/TEST overlap')
    key={'position':'position_xy','direction':'direction','ap':'passive'}[task]
    pred=knn(train['embedding_raw'],test['embedding_raw'],train[key],device)
    metric='r2_position_xy' if task=='position' else 'accuracy'
    score=float(r2_score(test[key],pred,multioutput='uniform_average') if task=='position' else accuracy_score(test[key],pred))
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'test_predictions.npz',true=test[key],predicted=pred,source_id=test['source_id'],trial_id=test['trial_id'])
    record=dict(task=task,architecture=arch,dimension=dim,seed=seed,metric=metric,score=score,
        encoder_label=('direction categorical' if task=='direction' else 'position XY'),decoder='uniform kNN',
        k=36,distance='cosine',decoder_train='TRAIN only',evaluation='TEST only',
        train_rows=len(train['source_id']),test_rows=len(test['source_id']),test_trials=len(np.unique(test['trial_id'])),
        source_sha256=hashes,predictions_sha256=sha(dest/'test_predictions.npz'))
    result.write_text(json.dumps(record,indent=2),encoding='utf8');return record
def plot(records):
    fig,axes=plt.subplots(1,3,figsize=(18,5.5),constrained_layout=True)
    titles={'position':'Fit Position X,Y\nDecoding Position X,Y',
            'direction':'Fit Direction\nDecoding Direction',
            'ap':'Fit Position X,Y\nDecoding Active / Passive'}
    colors={'cnn':'#222222','transformer':'#2166ac'}
    for ax,task in zip(axes,TASKS):
        for arch in ARCHS:
            for i,dim in enumerate(DIMS):
                vals=[r['score']*100 for r in records if r['task']==task and r['architecture']==arch and r['dimension']==dim]
                if len(vals)!=5:raise ValueError('Incomplete grid')
                offset=-.08 if arch=='cnn' else .08
                ax.scatter(i+offset+np.linspace(-.025,.025,5),vals,s=32,facecolors='none',edgecolors=colors[arch])
                ax.plot(i+offset,np.median(vals),marker='_',markersize=17,markeredgewidth=2.4,color=colors[arch])
        ax.set_title(titles[task],fontsize=12);ax.set_xticks(range(5),DIMS);ax.set_xlabel('Embedding dimension')
        ax.set_ylabel('R² Position × 100' if task=='position' else 'Accuracy (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(handles=[Line2D([0],[0],color=colors[a],label=a.upper()) for a in ARCHS],frameon=False)
    PLOT.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(PLOT/f'figura_3h_Soft_test_posizione_direzione_condizione.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
def main():
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records=[]
    for task in TASKS:
        for arch in ARCHS:
            for dim in DIMS:
                for seed in SEEDS:
                    records.append(run_one(task,arch,dim,seed,device))
                    print('SCORED',task,arch,dim,seed,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'SUMMARY.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    plot(records)
    print('FINISHED 3h Soft',len(records),flush=True)
if __name__=='__main__':main()
