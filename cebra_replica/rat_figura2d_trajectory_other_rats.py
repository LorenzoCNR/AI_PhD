"""Plot held-out position trajectories from existing 2d decoder settings."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rat_rebuild_1e_ed4_metrics as source
import rat_figura2_20261028 as fig2

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/figura_2_2026-10-28/d'
RATS=('buddy','cicero','gatsby')


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (DEST/'calcoli').glob('*.json')]
    methods=[(label,kind,variant) for label,kind,variant in source.METHODS if not (kind=='pivae' and variant=='post_mean_behavior_informed_raw')]
    for rat in RATS:
        cells=[];common=None
        for label,kind,variant in methods:
            matches=[r for r in records if r['rat']==rat and r['fold']==0 and r['seed']==42 and r['method']==label]
            if len(matches)!=1:raise ValueError(f'Expected one decoder {rat}/{label}: {len(matches)}')
            record=matches[0]
            data=[source.resolve(kind,variant,rat,0,42,s) for s in ('train','valid','test')]
            for s,d in enumerate(data):
                if d[3]!=record['input_paths'][s] or d[4]!=record['input_sha256'][s]:raise ValueError('Input provenance mismatch')
            common=data[2][2] if common is None else np.intersect1d(common,data[2][2])
            cells.append((label,record,data))
        common=np.sort(common)
        if not len(common):raise ValueError(f'No common held-out test rows for {rat}')
        segments=np.split(common,np.flatnonzero(np.diff(common)!=1)+1)
        chosen=max(segments,key=len)[:700]
        if len(chosen)<2:raise ValueError('Insufficient contiguous shared test rows')
        t=np.arange(len(chosen))*.025
        fig,axes=plt.subplots(3,4,figsize=(18,9),sharex=True,sharey=True)
        references=[];first_y=None
        for ax,(label,rec,(tr,va,te)) in zip(axes.flat,cells):
            lookup={int(v):i for i,v in enumerate(te[2])};idx=np.asarray([lookup[int(v)] for v in chosen])
            y=te[1][idx,0]
            if first_y is None:first_y=y
            elif not np.array_equal(y,first_y):raise ValueError('True behavior mismatch')
            x=torch.nn.functional.normalize(torch.as_tensor(tr[0],device=device),dim=1)
            q=torch.nn.functional.normalize(torch.as_tensor(te[0][idx],device=device),dim=1)
            with torch.inference_mode():near=torch.topk(q @ x.T,25,dim=1).indices.cpu().numpy()[:,:rec['k']]
            pred=tr[1][near,0].mean(axis=1)
            ridge=make_pipeline(StandardScaler(),Ridge(alpha=float(rec['ridge_alpha'])))
            ridge.fit(tr[0],tr[1][:,0]);pred_ridge=ridge.predict(te[0][idx])
            ax.plot(t,y*100,color='black',lw=1,label='Position reale')
            ax.plot(t,pred*100,color='#2961a7',lw=1,label='kNN')
            ax.plot(t,pred_ridge*100,color='#d1782b',lw=1,label='Ridge')
            ax.set_title(fig2.display(label),fontsize=8);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7)
            references.append({'method':label,'k':rec['k'],'ridge_alpha':rec['ridge_alpha'],
                               'input_paths':rec['input_paths'],'input_sha256':rec['input_sha256']})
        for ax in axes[-1,:]:ax.set_xlabel('Time (s)')
        for ax in axes[:,0]:ax.set_ylabel('Position (cm)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        fig.subplots_adjust(left=.06,right=.98,bottom=.08,top=.90,wspace=.2,hspace=.45)
        folder=DEST/rat.capitalize();stem=folder/f'Figura_2d_{rat.capitalize()}_traiettorie_test'
        fig2.save(fig,stem)
        meta={'rat':rat,'seed':42,'fold':0,'split':'test held-out','actual_source_rows':chosen.tolist(),
              'samples':len(chosen),'sampling_ms':25,'interval':'longest contiguous support common to all methods, first up to 700 bins',
              'decoder':'kNN cosine k selected on validation for position+direction R2; Ridge alpha selected on validation for position R2',
              'references':references,'training_performed':False}
        (folder/'RIFERIMENTI_TRAIETTORIE.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(f'{rat}: {len(chosen)} consecutive test bins, source {chosen[0]}..{chosen[-1]}',flush=True)


if __name__=='__main__':main()
