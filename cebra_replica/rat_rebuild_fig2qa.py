"""Read-only checks for Figure 2 staged products and validation-selected kNN."""
import json
from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
import rat_rebuild_fig2d as src
import rat_rebuild_fig2d_validation as v

def main():
    rat='achilles';fold=0;seed=42;data=src.cache(rat,fold)
    tests=[('CEBRA time',lambda:src.load_cebra(rat,fold,seed,'time',data)),
           ('conv-piVAE',lambda:src.load_pivae(rat,fold,seed,data)),
           ('Soft CNN Direction + position',lambda:src.load_soft(rat,fold,seed,'direction_position','cnn',data)),
           ('UMAP',lambda:v.baseline(rat,fold,seed,'UMAP',data))]
    comparison=[]
    for name,loader in tests:
        arrays,_=loader();r=json.loads((v.OUT/'cells'/rat/name/'fold_0'/'seed_42.json').read_text())
        mask0=data['train']['valid_mask'];mask2=data['test']['valid_mask']
        x=arrays['train'][mask0].astype('float64');xt=arrays['test'][mask2].astype('float64')
        y=data['train']['behavior'][mask0];yt=data['test']['behavior'][mask2];k=r['position_k']
        p=KNeighborsRegressor(n_neighbors=k,metric='cosine').fit(x,y[:,0]).predict(xt)
        d=KNeighborsClassifier(n_neighbors=k,metric='cosine').fit(x,np.argmax(y[:,1:],axis=1)).predict(xt)
        error=float(np.median(np.abs(p-yt[:,0]))*100);accuracy=float(np.mean(d==np.argmax(yt[:,1:],axis=1)))
        if abs(error-r['position_knn_cm'])>1e-5 or abs(accuracy-r['direction_knn_accuracy'])>1e-8:raise ValueError(f'kNN discrepancy: {name}')
        comparison.append(dict(method=name,k=k,position_abs_difference=abs(error-r['position_knn_cm']),accuracy_abs_difference=abs(accuracy-r['direction_knn_accuracy'])))
    counts={}
    for rat in src.RATS:
        files=list((v.OUT/'cells'/rat).rglob('*.json'));counts[rat]=len(files)
        if len(files)!=201:raise ValueError(f'Unexpected cell count {rat}: {len(files)}')
        for f in files:
            r=json.loads(f.read_text());scores=r['validation_joint_r2'];grid=r['k_candidates']
            if grid!=list(v.K_GRID) or r['position_k']!=grid[int(np.argmax(scores))] or r['direction_k']!=r['position_k']:
                raise ValueError(f'Validation selection mismatch {f}')
        manifest=json.loads((v.OUT/f'{rat}_decoding.json').read_text())
        if manifest['missing_cells']:raise ValueError(f'Incomplete {rat}')
    src.exclusive_json(v.OUT/'QA.json',{'cells_per_rat':counts,'selection_rechecked_for_every_cell':True,
        'sklearn_cosine_knn_independent_comparisons':comparison,'encoder_fit_count':0,
        'warning':'Checks verify decoding, support and staged hashes; they do not establish exact equivalence with paper fit parameters or grid.'})
    print(json.dumps({'counts':counts,'sklearn_checks':comparison},indent=2))

if __name__=='__main__':main()
