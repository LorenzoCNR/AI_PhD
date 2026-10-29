"""Figure 2d version with per-task k selected on validation only; no encoder training."""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rat_rebuild_fig2d as src

ROOT=src.ROOT
OUT=ROOT/'results/rat_rebuild_20261027/Figure_2/2D_validation_joint_r2'
BASE=ROOT/'results/rat_manifold_baselines_v1'
# Match the shared Figure 1e/ED4 reconstruction, without claiming paper exactness.
K_GRID=(1,4,9,16,25)
ALPHAS=(.1,1.,10.)

def baseline(rat,fold,seed,method,data):
    key={'PCA':'pca','UMAP':'umap','tSNE':'opentsne'}[method]
    folder=BASE/key/rat/f'fold_{fold}'/f'seed_{seed}';marker=folder/'result.json'
    if not marker.is_file():return None
    m=json.loads(marker.read_text()); expected_seed=None if seed=='deterministic' else int(seed)
    if (m['rat'],m['fold'],m['seed'])!=(rat,fold,expected_seed) or m['status'] not in ('complete','complete_embeddings_only'):raise ValueError(marker)
    if m.get('representation_fit')!='training valid windows only; behavior labels are not used':raise ValueError(f'unverified fit scope {marker}')
    arrays={};parents=[{'path':str(marker.relative_to(ROOT)),'sha256':src.sha(marker)}]
    for s in ('train','valid','test'):
        p=folder/f'{s}_embedding.npz';h=src.sha(p)
        if h!=m['embeddings'][s]['sha256']:raise ValueError(p)
        a=src.load(p);base=data[s]
        for k in ('source_id','trial_id','time_id','behavior','valid_mask'):
            if not np.array_equal(a[k],base[k]):raise ValueError(f'baseline support mismatch {k}: {p}')
        arrays[s]=a['embedding_raw'];parents.append({'path':str(p.relative_to(ROOT)),'sha256':h})
    return arrays,parents

def score(arrays,data):
    x={};y={}
    for s in ('train','valid','test'):
        mask=data[s]['valid_mask'];z=arrays[s]
        if z.ndim!=2 or len(z)!=len(mask) or not np.isfinite(z[mask]).all():raise ValueError('invalid embedding')
        x[s]=z[mask].astype('float64');y[s]=data[s]['behavior'][mask].astype('float64')
        if not np.isin(y[s][:,1],[0,1]).all():raise ValueError('non-binary direction')
    normalize=lambda z:z/np.linalg.norm(z,axis=1,keepdims=True)
    if any(np.any(np.linalg.norm(x[s],axis=1)==0) for s in ('train','valid','test')):raise ValueError('zero cosine norm')
    tree=cKDTree(normalize(x['train']));_,iv=tree.query(normalize(x['valid']),k=max(K_GRID),workers=1)
    if any(not np.allclose(y[s][:,1:].sum(axis=1),1) for s in ('train','valid','test')):raise ValueError('direction must be one-hot')
    val_pos=[];val_dir=[];val_joint=[]
    for k in K_GRID:
        pos=y['train'][iv[:,:k],0].mean(axis=1)
        label=np.argmax(y['train'][:,1:],axis=1)
        direction=(label[iv[:,:k]].mean(axis=1)>.5).astype(int)
        pred=np.column_stack((pos,np.eye(2)[direction]))
        val_pos.append(float(np.median(np.abs(pos-y['valid'][:,0]))*100))
        val_dir.append(float(np.mean(direction==np.argmax(y['valid'][:,1:],axis=1))))
        val_joint.append(float(r2_score(y['valid'],pred)))
    if not np.isfinite(val_joint).all():raise ValueError('invalid joint validation R2')
    kp=kd=K_GRID[int(np.argmax(val_joint))]
    alpha_scores=[];models=[]
    for alpha in ALPHAS:
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x['train'],y['train'][:,0]);models.append(model)
        alpha_scores.append(float(r2_score(y['valid'][:,0],model.predict(x['valid']))))
    if not np.isfinite(alpha_scores).all():raise ValueError('invalid validation R2')
    best=int(np.argmax(alpha_scores));
    # Test coordinates and outcomes are used only after both k and alpha are fixed.
    _,it=tree.query(normalize(x['test']),k=max(kp,kd),workers=1)
    if it.ndim==1:it=it[:,None]
    pred_pos=y['train'][it[:,:kp],0].mean(axis=1)
    pred_dir=(np.argmax(y['train'][:,1:],axis=1)[it[:,:kd]].mean(axis=1)>.5).astype(int)
    pred_ridge=models[best].predict(x['test'])
    return {'position_knn_cm':float(np.median(np.abs(pred_pos-y['test'][:,0]))*100),
        'direction_knn_accuracy':float(np.mean(pred_dir==np.argmax(y['test'][:,1:],axis=1))),
        'position_ridge_cm':float(np.median(np.abs(pred_ridge-y['test'][:,0]))*100),
        'position_k':kp,'direction_k':kd,'k_candidates':list(K_GRID),'validation_position_median_error_cm':val_pos,
        'validation_direction_accuracy':val_dir,'validation_joint_r2':val_joint,'ridge_alpha':ALPHAS[best],'ridge_validation_r2':alpha_scores,
        'n_train':len(x['train']),'n_valid':len(x['valid']),'n_test':len(x['test'])}

def cell(rat,fold,seed,method,data,loader):
    path=OUT/'cells'/rat/method/f'fold_{fold}'/f'seed_{seed}.json'
    if path.exists():
        v=json.loads(path.read_text())
        if v.get('k_candidates')!=list(K_GRID):raise ValueError(f'cache policy mismatch {path}')
        for p in v['parents']:
            if src.sha(ROOT/p['path'])!=p['sha256']:raise ValueError(f'parent changed {p}')
        return v
    item=loader()
    if item is None:return None
    arrays,parents=item
    obj=dict(rat=rat,fold=fold,seed=seed,method=method,parents=parents,**score(arrays,data))
    src.exclusive_json(path,obj);return obj

def output(rat,records,missing):
    methods=['CEBRA direction + position','CEBRA time','conv-piVAE','PCA','UMAP','tSNE']+[f'Soft {a.upper()} {v}' for a in ('cnn','transformer') for v in src.SC.values()]
    groups={}
    for r in records:
        for k in ('position_knn_cm','position_ridge_cm','direction_knn_accuracy'):
            groups.setdefault((r['method'],k,str(r['seed'])),[]).append((r['fold'],r[k]))
    fig,axes=plt.subplots(1,2,figsize=(15,8),sharey=True); ypos=np.arange(len(methods))
    for j,ax in enumerate(axes):
        for i,name in enumerate(methods):
            for metric,offset,marker,color in (((('position_knn_cm' if j==0 else 'direction_knn_accuracy')),-.13,'o','#315c99'),('position_ridge_cm',.13,'s','#c45d32')):
                if j==1 and metric=='position_ridge_cm':continue
                vals=[np.mean([v for _,v in g]) for (m,k,s),g in groups.items() if m==name and k==metric and {f for f,_ in g}=={0,1,2}]
                if vals:ax.errorbar(np.mean(vals),i+offset,xerr=np.std(vals) if len(vals)>1 else None,fmt=marker,color=color,capsize=2)
        ax.set_title('Position: median absolute error (cm)' if j==0 else 'Direction: accuracy');ax.grid(axis='x',alpha=.25)
    axes[0].set_yticks(ypos,methods,fontsize=8);axes[0].invert_yaxis();axes[1].set_xlim(0,1.01)
    axes[0].plot([],[],color='#315c99',marker='o',ls='',label='kNN: k selected on validation');axes[0].plot([],[],color='#c45d32',marker='s',ls='',label='Ridge');axes[0].legend(loc='lower right')
    fig.suptitle(f'{rat.title()} | held-out test · fold means / seed mean ± SD\nValidation-selected k; {len(missing)} missing cells',fontsize=11);fig.tight_layout()
    stem=OUT/f'{rat}_decoding'
    for ext in ('.png','.pdf'):
        p=stem.with_suffix(ext)
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=180,bbox_inches='tight')
    plt.close(fig)
    src.exclusive_json(stem.with_suffix('.json'),{'rat':rat,'missing_cells':missing,'records':len(records),
        'split_policy':'train decoder, select k and Ridge alpha on validation, score held-out test; trial-disjoint and valid context',
        'k_grid':list(K_GRID),'k_grid_provenance':'shared Figure 1e/ED4 reconstruction; not verified as official paper grid',
        'k_selection':'one k maximizes validation mean R2 across position and both one-hot direction outputs; ties favor smallest k',
        'direction_selection':'same validation-selected k as position; binary classification tie predicts one-hot index 0',
        'ridge_selection':'max validation R2, train-standardized alpha in [0.1,1,10]',
        'baseline_policy':'PCA, UMAP and tSNE re-scored on verified stored train/valid/test transforms; tSNE perplexity 30, distinct from previously documented paper configuration',
        'support_difference_from_1e_ed4':'all methods here use common valid context centers on every split; the separate 1e/ED4 run uses all train rows for CEBRA/Soft/piVAE',
        'aggregation':'three folds equally weighted per seed; six seed mean ± population SD; PCA deterministic',
        'records_path':str((OUT/'cells'/rat).relative_to(ROOT))})
    fields=['rat','method','seed','fold','position_knn_cm','position_ridge_cm','direction_knn_accuracy','position_k','direction_k','ridge_alpha','n_train','n_valid','n_test']
    with (OUT/f'{rat}_fold_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for rat in src.RATS:
        records=[];missing=[]
        for fold in range(3):
            data=src.cache(rat,fold)
            for seed in src.SEEDS:
                for cond,name in (('behavior','CEBRA direction + position'),('time','CEBRA time')):
                    v=cell(rat,fold,seed,name,data,lambda:src.load_cebra(rat,fold,seed,cond,data));records.append(v)
                records.append(cell(rat,fold,seed,'conv-piVAE',data,lambda:src.load_pivae(rat,fold,seed,data)))
                for cond,title in src.SC.items():
                    for arch in ('cnn','transformer'):
                        name=f'Soft {arch.upper()} {title}';v=cell(rat,fold,seed,name,data,lambda:src.load_soft(rat,fold,seed,cond,arch,data))
                        if v is None:missing.append([name,fold,seed])
                        else:records.append(v)
                for name in ('UMAP','tSNE'):
                    v=cell(rat,fold,seed,name,data,lambda:baseline(rat,fold,seed,name,data))
                    if v is None:missing.append([name,fold,seed])
                    else:records.append(v)
            v=cell(rat,fold,'deterministic','PCA',data,lambda:baseline(rat,fold,'deterministic','PCA',data))
            if v is None:missing.append(['PCA',fold,'deterministic'])
            else:records.append(v)
            print(rat,'fold',fold,'complete',flush=True)
        output(rat,records,missing);print(rat,'complete; missing',len(missing),flush=True)

if __name__=='__main__':main()
