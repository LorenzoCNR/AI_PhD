"""One rat and one decoder per gallery; never overlay different model trajectories."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rat_revision_common import ROOT,STAGE,RATS,DISPLAY,embedding,soft_root,sha,save,write_csv


def load_reference(rat,label,split):
    folder=ROOT/'results/user_soft_rat_v1/stage02_reference'/rat/'fold_0'/label/'seed_42'
    marker=folder/'manifest.json'
    if not marker.exists():
        marker=folder/'EMBEDDING_PROVENANCE.json'
    record=json.loads(marker.read_text())
    path=folder/f'{split}_embedding.npz'
    if sha(path)!=record['embeddings'][split]['embedding_sha256']:raise RuntimeError('Reference embedding hash mismatch')
    with np.load(path,allow_pickle=False) as saved:
        data={k:saved[k].copy() for k in ('embedding_raw','behavior','source_id','trial_id','time_id','valid_mask')}
    return data,dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),manifest_sha256=sha(marker))


def choose_trial(data):
    candidates=[]
    for trial in np.unique(data['trial_id']):
        rows=np.flatnonzero((data['trial_id']==trial)&data['valid_mask'])
        rows=rows[np.argsort(data['time_id'][rows],kind='stable')]
        if len(rows) and np.all(np.diff(data['time_id'][rows])==1):candidates.append((len(rows),int(trial),rows))
    if not candidates:raise RuntimeError('No complete contiguous valid test trial')
    candidates.sort(key=lambda x:(-x[0],x[1]))
    return candidates[0][2][:700]


def alpha_from_metrics(path):
    record=json.loads(path.read_text())
    selected=[r for r in record['metrics'] if r['representation']=='embedding_raw' and r['decoder']=='ridge' and r['split']=='valid']
    scored=[r for r in record['metrics'] if r['representation']=='embedding_raw' and r['decoder']=='ridge' and r['split']=='test']
    if len(selected)!=1 or len(scored)!=1 or selected[0]['selected_alpha']!=scored[0]['selected_alpha']:
        raise RuntimeError('Stored validation-selected Ridge alpha missing/mismatched')
    return float(selected[0]['selected_alpha'])


def predict(train,test,rows,alpha):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if np.intersect1d(train['trial_id'],test['trial_id']).size:raise RuntimeError('Trial leakage in trajectory probe')
    mask=train['valid_mask']
    x=train['embedding_raw'][mask];y=train['behavior'][mask,0]
    if not np.all(test['valid_mask'][rows]):raise RuntimeError('Invalid displayed test rows')
    query=test['embedding_raw'][rows]
    knn=KNeighborsRegressor(n_neighbors=36,metric='cosine').fit(x,y)
    ridge=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(x,y)
    return dict(kNN=knn.predict(query),Ridge=ridge.predict(query))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=STAGE/'Figure_2/2D';out.mkdir(parents=True,exist_ok=True)
    issue=out/'TRAJECTORY_ISSUES_BEFORE_CHANGE.md'
    if not issue.exists():
        issue.write_text('# Issues recorded before revision\n\nOld trajectory gallery stacked four rats and overlaid multiple models on the same axes, obscuring correspondence. The old implementation also truncated every rat to the shortest selected trial across rats. Latest behavior-only CNN/Transformer were absent. Reporting/layout issue only; no model or scientific metric is invalidated.\n\nCorrection: one rat and decoder per gallery, one model per axis, exactly two curves (truth/prediction); shared trial/time/source support within each rat, no trial joins, deterministic longest valid test trial capped at700points (not700curves). Retain seed42/fold0, existing kNN36 and stored validation-selected Ridge alpha. New downstream probe fits only; no encoder fitting or hyperparameter search.\n',encoding='utf-8')
    clipping_issue=out/'TRAJECTORY_Y_RANGE_ISSUE.md'
    if not clipping_issue.exists():
        clipping_issue.write_text('Before final promotion: first staging used physical track limits0..160cm, but saved predictions include values outside that range (Achilles -66.84..179.11cm). Fixed limits would hide decoder errors. Reporting-only issue, no training/metric impact. Correction: common per-rat/per-decoder limits cover all displayed truth and predictions with5percent margin; preserve the values, do not clip or smooth. No scientific result recomputation required.\n',encoding='utf-8')
    index=[]
    for rat in RATS:
        reference,parent=embedding('CNN','behavior_only',rat)
        rows=choose_trial(reference)
        times=(reference['time_id'][rows]-reference['time_id'][rows][0])*.025
        truth=reference['behavior'][rows,0]*100
        parents=[parent];methods=[];long=[]
        identities=[('CEBRA',c) for c in ('CEBRA-Behavior','CEBRA-Time')]+[(a,c) for c in ('joint','time_only','behavior_only') for a in ('CNN','Transformer')]
        for architecture,condition in identities:
            if architecture=='CEBRA':
                train,ptrain=load_reference(rat,condition,'train');test,ptest=load_reference(rat,condition,'test')
                metric=ROOT/'results/user_soft_rat_v1/stage03_reference'/rat/'fold_0'/condition/'seed_42/metrics.json'
                title=condition;own=False
            else:
                train,ptrain=embedding(architecture,condition,rat,split='train')
                test,ptest=embedding(architecture,condition,rat,split='test')
                metric=soft_root(architecture,condition,rat)/'stage03_metrics'/rat/'fold_0/soft_v1_stable/seed_42/metrics.json'
                title=DISPLAY[(architecture,condition)].replace('\n',' - ');own=True
            for key in ('source_id','trial_id','time_id','valid_mask','behavior'):
                if not np.array_equal(test[key],reference[key]):raise RuntimeError(f'Common trajectory support mismatch:{rat}/{title}/{key}')
            alpha=alpha_from_metrics(metric)
            predictions=predict(train,test,rows,alpha)
            parents.extend((ptrain,ptest,dict(path=metric.relative_to(ROOT).as_posix(),sha256=sha(metric),selected_alpha=alpha)))
            methods.append((title,own,predictions))
            for decoder,predicted in predictions.items():
                if not np.isfinite(predicted).all():raise RuntimeError('Nonfinite decoded trajectory')
                for i,j in enumerate(rows):
                    long.append(dict(rat=rat,method=title,architecture=architecture,condition=condition,decoder=decoder,
                        seed=42,fold=0,split='held_out_test',source_id=int(test['source_id'][j]),
                        trial_id=int(test['trial_id'][j]),time_id=int(test['time_id'][j]),time_s=float(times[i]),
                        true_position_cm=float(truth[i]),predicted_position_cm=float(predicted[i]*100),
                        ridge_alpha=alpha if decoder=='Ridge' else ''))
        write_csv(out/f'FIGURE2D_TRAJECTORIES_{rat.upper()}_LONG.csv',long)
        for decoder in ('kNN','Ridge'):
            displayed=np.concatenate([truth]+[m[2][decoder]*100 for m in methods])
            low,high=float(displayed.min()),float(displayed.max())
            margin=max(5.,.05*(high-low))
            fig,axes=plt.subplots(4,2,figsize=(14,12),sharex=True,sharey=True)
            for ax,(title,own,predictions) in zip(axes.flat,methods):
                ax.plot(times,truth,color='black',lw=1.6)
                ax.plot(times,predictions[decoder]*100,color='deepskyblue',lw=1.3)
                ax.set_title(title,color='red' if own else 'black',fontsize=11)
                ax.set_facecolor('white');ax.set_ylim(low-margin,high+margin)
                ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
                ax.set_ylabel('Position (cm)')
            for ax in axes[-1]:ax.set_xlabel('Time within displayed trial (s)')
            fig.suptitle(f'{rat.capitalize()} - {decoder} held-out trajectories',fontsize=16,y=.99)
            fig.legend(handles=[Line2D([],[],color='black',lw=1.6,label='Ground truth'),Line2D([],[],color='deepskyblue',lw=1.3,label='Prediction')],loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,frameon=False)
            trial=int(reference['trial_id'][rows[0]])
            fig.text(.5,.012,f'Seed42, fold0, trial{trial}, {len(rows)} samples; each panel has one truth and one prediction. Models frozen; Ridge alpha from validation.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.035,1,.93))
            base=out/f'FIGURE2D_TRAJECTORIES_{rat.upper()}_{decoder.upper()}'
            save(fig,base,'2D trajectories',parents,dict(rat=rat,decoder=decoder,split='held_out_test',
                trial_id=trial,samples=len(rows),seed=42,fold=0,curves_per_axis=2,
                selection='longest valid contiguous test trial; tie smallest ID; first700samplesmaximum; no performance selection',
                decoder_refitted=True,encoder_retrained=False))
            index.append(dict(rat=rat,decoder=decoder,png=base.with_suffix('.png').name,trial_id=trial,
                n_samples=len(rows),n_axes=8,curves_per_axis=2))
    write_csv(out/'TRAJECTORY_PANEL_INDEX.csv',index)


if __name__=='__main__':main()
