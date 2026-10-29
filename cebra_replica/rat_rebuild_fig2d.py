"""Score existing split embeddings and stage Figure 2d without encoder fitting."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rat_rebuild_20261027/Figure_2/2D'
CAM=ROOT/'results/rat_paper_fig1e_ed4_6seed_v1'
SHARED=ROOT/'results/rat_rebuild_20261027/Figure_1E_ED4/embeddings/soft'
OLD=ROOT/'results/RAT_FIGURES/Figure_2/2D/FIGURE2D_METRICS_FOLD_RECORDS.csv'
RATS=('achilles','buddy','cicero','gatsby')
SC={'time_only':'Time','direction_position':'Direction + position','time_direction_position':'Time + direction + position'}
SEEDS=range(42,48)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as a: return {k:a[k].copy() for k in a.files}

def cache(rat,fold):
    folder=CAM/'soft/stage00_data'/rat/f'fold_{fold}'; m=json.loads((folder/'manifest.json').read_text()); data={}
    for split in ('train','valid','test'):
        p=folder/f'{split}.npz'
        if sha(p)!=m['splits'][split]['sha256']: raise ValueError(f'data cache hash: {p}')
        data[split]=load(p)
    for s,t in (('train','valid'),('train','test'),('valid','test')):
        for k in ('trial_id','source_id'):
            if np.intersect1d(data[s][k],data[t][k]).size: raise ValueError(f'{k} leakage {rat}/{fold}')
    return data

def load_cebra(rat,fold,seed,cond,data):
    d=CAM/'cebra'/rat/f'fold_{fold}'/cond/f'seed_{seed}'; m=json.loads((d/'manifest.json').read_text())
    if m['status']!='complete' or m['identity']!={'rat':rat,'fold':fold,'seed':seed,'condition':cond}: raise ValueError(d)
    result={}; parents=[]
    for s in ('train','valid','test'):
        p=d/'embeddings'/f'{s}.npz'; h=sha(p)
        if h!=m['embeddings'][s]['sha256']: raise ValueError(p)
        a=load(p); base=data[s]
        for k in ('source_id','trial_id','behavior'):
            if not np.array_equal(a[k],base[k]): raise ValueError(f'CEBRA support mismatch {k} {p}')
        if np.any(base['valid_mask'] & ~a['context_valid']): raise ValueError(f'CEBRA invalid context {p}')
        result[s]=a['embedding'];parents.append({'path':str(p.relative_to(ROOT)),'sha256':h})
    return result,parents

def load_pivae(rat,fold,seed,data):
    parent=CAM/'pivae/runs'/rat/f'fold_{fold}'/f'seed_{seed}'; candidates=[p for p in parent.glob('*/run.json') if json.loads(p.read_text()).get('status')=='complete']
    if len(candidates)!=1: raise ValueError(f'piVAE run count {parent}: {len(candidates)}')
    marker=candidates[0];m=json.loads(marker.read_text()); c=m['config']
    if m['status']!='complete' or (c['rat'],c['fold'],c['seed'])!=(rat,fold,seed): raise ValueError(marker)
    result={};parents=[]
    for s in ('train','valid','test'):
        p=marker.parent/f'embedding_{s}.npz';h=sha(p)
        if h!=m['artifact_hashes'][f'embedding_{s}_sha256']:raise ValueError(p)
        a=load(p);base=data[s]
        for k in ('source_id','trial_id'):
            if not np.array_equal(a[k],base[k]):raise ValueError(f'piVAE support mismatch {k} {p}')
        if np.any(a['crosses_trial_boundary'].astype(bool) & base['valid_mask']):raise ValueError(f'piVAE boundary {p}')
        result[s]=a['z_mean_raw'];parents.append({'path':str(p.relative_to(ROOT)),'sha256':h})
    return result,parents

def load_soft(rat,fold,seed,cond,arch,data):
    d=SHARED/cond/arch/rat/f'fold_{fold}/soft_v1_stable'/f'seed_{seed}'
    files=[d/f'{s}.npz' for s in ('train','valid','test')]
    if not all(p.is_file() for p in files): return None
    fit=CAM/'soft/stage01_fit'/cond/arch/rat/f'fold_{fold}/soft_v1_stable'/f'seed_{seed}'
    cell=json.loads((fit/'CELL_MANIFEST.json').read_text()); checkpoint=sha(fit/'checkpoint.pt')
    if checkpoint!=cell['checkpoint_sha256'] or cell['actual_updates']!=10000:raise ValueError(fit)
    result={};parents=[]
    for s,p in zip(('train','valid','test'),files):
        a=load(p);base=data[s]
        for k in ('source_id','trial_id','behavior','valid_mask'):
            if not np.array_equal(a[k],base[k]):raise ValueError(f'Soft support mismatch {k} {p}')
        expected_source=sha(CAM/'soft/stage00_data'/rat/f'fold_{fold}'/f'{s}.npz')
        if str(a['checkpoint_sha256'])!=checkpoint or str(a['source_npz_sha256'])!=expected_source:raise ValueError(f'Soft parent mismatch {p}')
        if (str(a['rat']),int(a['fold']),int(a['seed']),str(a['condition']),str(a['architecture']),str(a['split']))!=(rat,fold,seed,cond,arch,s):raise ValueError(p)
        result[s]=a['embedding']; parents.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
    return result,parents

def score(arrays,data):
    x={};y={}
    for s in ('train','valid','test'):
        mask=data[s]['valid_mask']; z=arrays[s]
        if z.ndim!=2 or len(z)!=len(mask) or not np.isfinite(z[mask]).all():raise ValueError('invalid embedding')
        x[s]=z[mask].astype('float64');y[s]=data[s]['behavior'][mask].astype('float64')
    alphas=(.1,1.,10.); candidates=[]; models=[]
    for alpha in alphas:
        model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x['train'],y['train'][:,0]);models.append(model)
        candidates.append(float(r2_score(y['valid'][:,0],model.predict(x['valid']))))
    if not np.isfinite(candidates).all():raise ValueError('invalid validation R2')
    best=int(np.argmax(candidates)); pred_r=models[best].predict(x['test'])
    # cosine neighbour ordering equals Euclidean ordering on unit-normalized rows.
    norm=lambda a:np.linalg.norm(a,axis=1,keepdims=True)
    if np.any(norm(x['train'])==0) or np.any(norm(x['test'])==0):raise ValueError('zero norm in cosine kNN')
    tree=cKDTree(x['train']/norm(x['train']));_,ii=tree.query(x['test']/norm(x['test']),k=36,workers=1)
    pred_k=y['train'][ii,0].mean(axis=1)
    labels=y['train'][ii,1]
    if not np.isin(labels,[0,1]).all():raise ValueError('non-binary direction')
    pred_d=(labels.mean(axis=1)>.5).astype(int)
    return dict(position_knn_cm=float(np.median(np.abs(pred_k-y['test'][:,0]))*100),
                position_ridge_cm=float(np.median(np.abs(pred_r-y['test'][:,0]))*100),
                direction_knn_accuracy=float(np.mean(pred_d==y['test'][:,1])),alpha=alphas[best],
                validation_r2=candidates,n_train=len(x['train']),n_valid=len(x['valid']),n_test=len(x['test']))

def exclusive_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)

def saved_cell(rat,fold,seed,method,data,loader):
    path=OUT/'cells'/rat/method/f'fold_{fold}'/f'seed_{seed}.json'
    if path.exists():
        obj=json.loads(path.read_text());
        for p in obj['parents']:
            if sha(ROOT/p['path'])!=p['sha256']:raise ValueError(f'changed parent {p["path"]}')
        return obj
    loaded=loader()
    if loaded is None:return None
    arr,parents=loaded;obj=dict(rat=rat,fold=fold,seed=seed,method=method,parents=parents,**score(arr,data))
    exclusive_json(path,obj);return obj

def old_baselines(rat):
    rows=[]
    with OLD.open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['rat']!=rat or r['method'] not in ('PCA','UMAP','tSNE') or r['split']!='test':continue
            p=ROOT/r['parent_metrics_path']
            if sha(p)!=r['parent_metrics_sha256']:raise ValueError(f'baseline provenance {p}')
            rows.append(r)
    return rows

def render(rat,records,baseline,missing):
    # Seed values are equal-weight means of the available fold metrics.
    methods=['CEBRA direction + position','CEBRA time','conv-piVAE','PCA','UMAP','tSNE']+[f'Soft {a.upper()} {v}' for a in ('cnn','transformer') for v in SC.values()]
    values={};counts={}
    for rec in records:
        for key in ('position_knn_cm','position_ridge_cm','direction_knn_accuracy'):
            values.setdefault((rec['method'],key,str(rec['seed'])),[]).append(rec[key])
    for rec in baseline:
        key='position_'+rec['decoder']+'_cm'
        if rec['position_median_absolute_error_cm']:values.setdefault((rec['method'],key,rec['seed']),[]).append(float(rec['position_median_absolute_error_cm']))
        if rec['decoder']=='knn' and rec['direction_accuracy']:values.setdefault((rec['method'],'direction_knn_accuracy',rec['seed']),[]).append(float(rec['direction_accuracy']))
    fig,axs=plt.subplots(1,2,figsize=(15,8),sharey=True); ypos=np.arange(len(methods))
    for j,(metric,title) in enumerate((('position','Position: median absolute error (cm)'),('direction','Direction: accuracy'))):
        ax=axs[j]
        for i,name in enumerate(methods):
            for key,offset,marker,color in ((('position_knn_cm' if j==0 else 'direction_knn_accuracy'),-.13,'o','#315c99'),('position_ridge_cm',.13,'s','#c45d32')):
                if j==1 and key=='position_ridge_cm':continue
                groups=[v for (m,k,s),v in values.items() if m==name and k==key]
                seeds=np.asarray([np.mean(v) for v in groups]);
                if seeds.size:ax.errorbar(float(seeds.mean()),i+offset,xerr=float(seeds.std()) if len(seeds)>1 else None,fmt=marker,color=color,capsize=2)
        ax.set_title(title);ax.grid(axis='x',alpha=.25)
    axs[0].set_yticks(ypos,methods,fontsize=8);axs[0].invert_yaxis();axs[1].set_xlim(0,1.01)
    axs[0].plot([],[],color='#315c99',marker='o',ls='',label='kNN (k=36)');axs[0].plot([],[],color='#c45d32',marker='s',ls='',label='Ridge')
    coverage='all 108 Soft fit cells' if missing==0 else f'partial Soft coverage: {missing} of 108 fit cells missing'
    axs[0].legend(loc='lower right');fig.suptitle(f'{rat.title()} | held-out test · fold means / seed mean ± SD\n{coverage}',fontsize=11)
    fig.tight_layout();stem=OUT/f'{rat}_decoding_{"complete" if missing==0 else "partial_v2"}'
    for ext in ('.png','.pdf'):
        p=stem.with_suffix(ext)
        if p.exists():raise FileExistsError(p)
        fig.savefig(p,dpi=180,bbox_inches='tight')
    plt.close(fig)
    exclusive_json(stem.with_suffix('.json'),{'rat':rat,'status':'partial' if missing else 'complete','missing_soft_cells':missing,
        'sample_policy':'trial-disjoint train/validation/test; trial-interior centers',
        'decoder':'kNN cosine k=36 fixed; Ridge standardized on train, alpha (0.1,1,10) selected by validation R2; test scored once',
        'aggregation':'mean fold scores within seed, then seed mean ± population SD; missing Soft cells omitted, not imputed',
        'reference_baselines_source':str(OLD.relative_to(ROOT)),'reference_baselines_source_sha256':sha(OLD),'records':records,'baseline_records':baseline})
    table=OUT/f'{rat}_fold_metrics.csv'
    with table.open('x',encoding='utf-8',newline='') as f:
        fields=['rat','method','seed','fold','decoder','position_median_error_cm','direction_accuracy','selected_alpha','source']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in records:
            for decoder in ('knn','ridge'):
                w.writerow(dict(rat=rat,method=r['method'],seed=r['seed'],fold=r['fold'],decoder=decoder,
                    position_median_error_cm=r[f'position_{decoder}_cm'],direction_accuracy=r['direction_knn_accuracy'] if decoder=='knn' else '',
                    selected_alpha=r['alpha'] if decoder=='ridge' else '',source='new_score_on_frozen_split_embeddings'))
        for r in baseline:
            w.writerow(dict(rat=rat,method=r['method'],seed=r['seed'],fold=r['fold'],decoder=r['decoder'],
                position_median_error_cm=r['position_median_absolute_error_cm'],direction_accuracy=r['direction_accuracy'] if r['decoder']=='knn' else '',
                selected_alpha='',source='previous_verified_reference_metric'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for rat in RATS:
        records=[]; missing=0
        for fold in range(3):
            data=cache(rat,fold)
            for seed in SEEDS:
                for cond,name in (('behavior','CEBRA direction + position'),('time','CEBRA time')):
                    records.append(saved_cell(rat,fold,seed,name,data,lambda:load_cebra(rat,fold,seed,cond,data)))
                records.append(saved_cell(rat,fold,seed,'conv-piVAE',data,lambda:load_pivae(rat,fold,seed,data)))
                for condition,title in SC.items():
                    for arch in ('cnn','transformer'):
                        name=f'Soft {arch.upper()} {title}'
                        result=saved_cell(rat,fold,seed,name,data,lambda:load_soft(rat,fold,seed,condition,arch,data))
                        if result is None:missing+=1
                        else:records.append(result)
            print(rat,'fold',fold,'done',flush=True)
        render(rat,records,old_baselines(rat),missing)
        print(rat,'done; missing Soft',missing,flush=True)

if __name__=='__main__':main()
