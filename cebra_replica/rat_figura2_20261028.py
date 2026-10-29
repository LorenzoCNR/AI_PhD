"""Figura 2: rendering and decoding from verified saved fits; no training."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import rat_rebuild_1e_ed4_metrics as source

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/figura_2_2026-10-28'
REPRO=ROOT/'reproduction/figure2_bc'
RATS=('achilles','buddy','cicero','gatsby')
NORM=Normalize(0,1.6)
CONDITIONS=[('position','Position'),('direction','Direction'),('position_direction','Position + Direction'),
            ('position_shuffled','Position shuffled'),('direction_shuffled','Direction shuffled'),
            ('position_direction_shuffled','Position + Direction shuffled')]


def source_cebra(folder):
    m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    p=folder/'embedding.npz'
    if m['status']!='complete' or source.sha256(p)!=m['sha256']['embedding.npz']: raise ValueError(f'Invalid {p}')
    with np.load(p,allow_pickle=False) as a: z=a['embedding'].copy(); y=a['behavior'].copy()
    if not np.isfinite(z).all() or y.shape!=(len(z),3): raise ValueError(p)
    return z,y,{'path':str(p.relative_to(ROOT)),'sha256':source.sha256(p),'manifest':str((folder/'manifest.json').relative_to(ROOT))}


def source_soft(folder):
    m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    p=folder/'embedding_full.npz'
    if source.sha256(p)!=m['embedding_sha256']: raise ValueError(f'Invalid {p}')
    with np.load(p,allow_pickle=False) as a: z=a['embedding'].copy(); y=a['behavior'].copy()
    if not np.isfinite(z).all() or y.shape!=(len(z),3): raise ValueError(p)
    return z,y,{'path':str(p.relative_to(ROOT)),'sha256':source.sha256(p),'manifest':str((folder/'manifest.json').relative_to(ROOT))}


def save(fig,stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.png','.pdf','.svg'):
        p=stem.with_suffix(ext)
        if p.exists(): raise FileExistsError(p)
        fig.savefig(p,dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def colored(ax,z,y,dims):
    for col,cmap in ((2,'cool'),(1,'viridis')):
        mask=y[:,col]==1
        if len(dims)==2: ax.scatter(z[mask,dims[0]],z[mask,dims[1]],c=y[mask,0],cmap=cmap,norm=NORM,s=.6,linewidths=0,rasterized=True)
        else: ax.scatter(*z[mask][:,dims].T,c=y[mask,0],cmap=cmap,norm=NORM,s=.6,linewidths=0,rasterized=True)
    if len(dims)==3: ax.view_init(elev=20,azim=-60)
    ax.tick_params(labelsize=6,pad=0)


def render_b(family,entries):
    fig=plt.figure(figsize=(18,9))
    for i,(title,z,y,dims,_) in enumerate(entries):
        ax=fig.add_subplot(2,4,i+1,projection='3d' if len(dims)==3 else None)
        colored(ax,z,y,dims);ax.set_title(title,fontsize=10)
    for x,cmap,label in ((.915,'cool','Left'),(.955,'viridis','Right')):
        bar=fig.colorbar(ScalarMappable(norm=NORM,cmap=cmap),cax=fig.add_axes([x,.28,.012,.40]),ticks=[0,.5,1,1.5])
        bar.ax.tick_params(labelsize=7);fig.text(x+.006,.245,label,ha='center',fontsize=9)
    fig.subplots_adjust(left=.035,right=.89,top=.93,bottom=.08,wspace=.25,hspace=.38)
    stem=DEST/'b/Achilles'/f'Figura_2b_Achilles_{family}'
    save(fig,stem)
    (stem.parent/f'RIFERIMENTI_{family}.json').write_text(json.dumps({'rat':'achilles','seed':42,'sources':[e[4] for e in entries],
       'fit_and_plot':'full recording; all 10178 rows','labels':'colors are actual position/direction even for shuffled fits',
       'behavior_inputs':'two exploratory fits, output 2D; exact original configuration unverified','training_performed':False},indent=2),encoding='utf-8')


def render_c(family,entries,histories):
    fig=plt.figure(figsize=(14,9))
    for i,(title,z,y,dims,_) in enumerate(entries):
        ax=fig.add_subplot(2,3,i+1,projection='3d');ax.scatter(*z[:,dims].T,c='gray',s=.45,linewidths=0,rasterized=True)
        ax.set_title(title,fontsize=10);ax.view_init(elev=20,azim=-60);ax.tick_params(labelsize=6,pad=0)
    fig.subplots_adjust(wspace=.12,hspace=.24)
    stem=DEST/'c/Achilles'/f'Figura_2c_Achilles_{family}'
    save(fig,stem)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for i,(title,loss) in enumerate(histories):
        shuffled='shuffled' in title.lower();alpha=(.35,.65,1)[i%3]
        ax.plot(np.arange(1,len(loss)+1),loss,color='gray' if shuffled else 'deepskyblue',alpha=alpha,linewidth=.8,label=title)
    ax.set_xlabel('Update');ax.set_ylabel('Training loss');ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,stem.with_name(stem.name+'_loss'))
    (stem.parent/f'RIFERIMENTI_{family}.json').write_text(json.dumps({'rat':'achilles','seed':42,'sources':[e[4] for e in entries],
      'fit_and_plot':'full recording; first three coordinates of native 8D embedding','updates':10000,'original_file':'official arrays 8D; official loss arrays contain 5000 values','training_performed':False},indent=2),encoding='utf-8')


def loss_cebra(folder):
    p=folder/'loss.csv';m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if source.sha256(p)!=m['sha256']['loss.csv']:raise ValueError(p)
    return np.loadtxt(p,delimiter=',',skiprows=1)[:,1]


def loss_soft(folder):
    p=folder/'training_history.csv'
    a=np.loadtxt(p,delimiter=',',skiprows=1)
    if len(a)!=10000 or not np.isfinite(a).all():raise ValueError(p)
    return a[:,1]


def figures_bc():
    ce=REPRO/'full'; missing=REPRO/'figure2b_missing/full'
    ce_b=[('CEBRA — Position input',ce/'behavior_position_input/seed_42',[0,1]),
          ('CEBRA — Position + Direction input',missing/'behavior_position_direction_input/seed_42',[0,1]),
          ('CEBRA — Time',ce/'time/seed_42',[0,1,2]),('CEBRA — Position',ce/'position/seed_42',[0,1,2]),
          ('CEBRA — Position + Direction',missing/'position_direction/seed_42',[0,1,2]),
          ('CEBRA — Time + Position + Direction',REPRO/'time_position_direction_5d/achilles/seed_42',[1,2,0]),
          ('CEBRA — Position shuffled',missing/'position_shuffled/seed_42',[0,1,2]),
          ('CEBRA — Position + Direction shuffled',missing/'position_direction_shuffled/seed_42',[0,1,2])]
    entries=[]
    for title,path,dims in ce_b:
        z,y,ref=source_cebra(path);entries.append((title,z,y,dims,ref))
    render_b('CEBRA',entries)
    for mode in ('paper_demo','controlled'):
        entries=[];histories=[]
        for cond,title in CONDITIONS:
            key=(cond+'_'+mode) if cond in ('direction','direction_shuffled') else cond
            folder=REPRO/'figure2c_8d/full'/key/'seed_42'
            z,y,ref=source_cebra(folder);entries.append((f'CEBRA — {title}',z,y,[0,1,2],ref));histories.append((title,loss_cebra(folder)))
        render_c('CEBRA_'+mode,entries,histories)
    original=ROOT/'results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1/achilles'
    for arch in ('cnn','transformer'):
        label='Soft '+arch.upper();extra=REPRO/'soft_2b_shuffle_3d/full/achilles'
        behaviors=REPRO/'soft_behavior_2d_v2/full/achilles'
        specs=[('Position input',behaviors/'position_input'/arch/'seed_42',[0,1]),
               ('Position + Direction input',behaviors/'position_direction_input'/arch/'seed_42',[0,1]),
               ('Time',original/'time_only'/arch/'seed_42',[0,1,2]),('Position',extra/'position'/arch/'seed_42',[0,1,2]),
               ('Position + Direction',original/'direction_position'/arch/'seed_42',[0,1,2]),
               ('Time + Position + Direction',original/'time_direction_position'/arch/'seed_42',[0,1,2]),
               ('Position shuffled',extra/'position_shuffled'/arch/'seed_42',[0,1,2]),
               ('Time + Position + Direction shuffled',extra/'position_direction_shuffled'/arch/'seed_42',[0,1,2])]
        entries=[]
        for title,path,dims in specs:
            z,y,ref=source_soft(path);entries.append((label+' — '+title,z,y,dims,ref))
        render_b('SOFT_'+arch.upper(),entries)
        entries=[];histories=[]
        for cond,title in CONDITIONS:
            path=REPRO/'soft_2c_8d/full/achilles'/cond/arch/'seed_42'
            z,y,ref=source_soft(path);entries.append((label+' — '+title,z,y,[0,1,2],ref));histories.append((title,loss_soft(path)))
        render_c('SOFT_'+arch.upper(),entries,histories)


def calc_2d():
    metrics=ROOT/'results/figura_ext_4_2026-10-28/calcoli/heldout_decoding_due_variabili.csv'
    with metrics.open(newline='',encoding='utf-8') as f: cells=list(csv.DictReader(f))
    methods={label:(kind,variant) for label,kind,variant in source.METHODS}
    cache=DEST/'d/calcoli';cache.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    all_rows=[]
    for i,cell in enumerate(cells):
        path=cache/f'{i:04d}.json'
        if path.exists():
            row=json.loads(path.read_text(encoding='utf-8'))
            if any(source.sha256(ROOT/p)!=h for p,h in zip(row['input_paths'],row['input_sha256'])):raise ValueError(f'Changed input {path}')
        else:
            label=cell['method'];rat=cell['rat'];fold=int(cell['fold']);seed=int(cell['seed']);k=int(cell['selected_k'])
            kind,variant=methods[label]
            tr,va,te=[source.resolve(kind,variant,rat,fold,seed,s) for s in ('train','valid','test')]
            for a,b in ((tr,va),(tr,te),(va,te)):
                if np.intersect1d(a[2],b[2]).size:raise ValueError('overlapping split')
            x=torch.nn.functional.normalize(torch.as_tensor(tr[0],device=device),dim=1)
            q=torch.nn.functional.normalize(torch.as_tensor(te[0],device=device),dim=1)
            neighbors=[]
            with torch.inference_mode():
                for start in range(0,len(q),512): neighbors.append(torch.topk(q[start:start+512] @ x.T,25,dim=1).indices.cpu().numpy())
            idx=np.concatenate(neighbors)[:,:k]
            pred_pos=tr[1][idx,0].mean(axis=1)
            pred_dir=(tr[1][idx,1].sum(axis=1)>k/2).astype(int)
            knn=float(np.median(abs(pred_pos-te[1][:,0]))*100)
            if abs(knn-float(cell['test_position_error_m'])*100)>1e-4:raise ValueError(f'kNN inconsistent with ED4: {path}')
            candidates=[];models=[]
            for alpha in (.1,1.,10.):
                m=make_pipeline(StandardScaler(),Ridge(alpha=alpha));m.fit(tr[0],tr[1][:,0]);models.append(m)
                candidates.append(float(r2_score(va[1][:,0],m.predict(va[0]))))
            best=int(np.argmax(candidates));ridge=float(np.median(abs(models[best].predict(te[0])-te[1][:,0]))*100)
            row=dict(method=label,rat=rat,fold=fold,seed=seed,k=k,position_knn_cm=knn,
                     position_ridge_cm=ridge,direction_knn_accuracy=float(np.mean(pred_dir==te[1][:,1])),
                     ridge_alpha=(.1,1.,10.)[best],input_paths=[v[3] for v in (tr,va,te)],input_sha256=[v[4] for v in (tr,va,te)])
            path.write_text(json.dumps(row,indent=2),encoding='utf-8')
        all_rows.append(row)
        if i%40==0 or i+1==len(cells):print(f'DECODING {i+1}/{len(cells)}',flush=True)
    with (DEST/'d'/'metriche_fold.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    return all_rows


def display(label):
    return label.replace('behavior (position + direction)','Position + Direction').replace('time_direction_position','Time + Position + Direction').replace('direction_position','Position + Direction').replace('time_only','Time').replace('conv-piVAE q(z|x)','conv-piVAE — Position + Direction (q(z|x))').replace('Soft cnn','Soft CNN').replace('Soft transformer','Soft Transformer').replace('CEBRA time','CEBRA Time')


def figures_d(rows):
    order=[label for label,kind,variant in source.METHODS if not (kind=='pivae' and variant=='post_mean_behavior_informed_raw')]
    for rat in RATS:
        fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
        for i,method in enumerate(order):
            subset=[r for r in rows if r['rat']==rat and r['method']==method]
            for ax,metric,color,marker,offset in ((axes[0],'position_knn_cm','#3268ae','o',-.12),
                    (axes[0],'position_ridge_cm','#cf793b','s',.12),(axes[1],'direction_knn_accuracy','#3268ae','o',0)):
                grouped=defaultdict(list)
                for r in subset:grouped[r['seed']].append(r)
                values=[np.mean([v[metric] for v in a]) for a in grouped.values() if {v['fold'] for v in a}=={0,1,2}]
                if values:
                    ax.scatter(values,np.full(len(values),i+offset),marker=marker,s=26,facecolors='none',edgecolors=color)
                    ax.scatter([np.median(values)],[i+offset],marker='|',s=170,color=color)
                else:ax.text(.02,i+offset,'n/a',transform=ax.get_yaxis_transform(),fontsize=7)
        axes[0].set_yticks(range(len(order)),[display(m) for m in order],fontsize=8);axes[0].invert_yaxis()
        axes[0].set_xlabel('Median position error (cm)');axes[1].set_xlabel('Direction accuracy');axes[1].set_xlim(0,1.02)
        for ax in axes:ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
        axes[0].plot([],[],marker='o',color='#3268ae',ls='',label='kNN');axes[0].plot([],[],marker='s',color='#cf793b',ls='',label='Ridge');axes[0].legend(fontsize=8)
        fig.suptitle(rat.capitalize(),fontsize=13);fig.tight_layout()
        stem=DEST/'d'/rat.capitalize()/f'Figura_2d_{rat.capitalize()}'
        save(fig,stem)
        (stem.parent/'RIFERIMENTI.md').write_text('''# Figura 2d\n\nMetriche per fold: `../metriche_fold.csv` (i singoli record con percorsi e hash sono nella cartella `../calcoli`). Fit encoder 1e/ED4, sei seed 42–47, tre fold per trial, train/validation/test separati. kNN cosine, k scelto sulla validation in base alla media degli R² Position e Direction; errore mediano e accuracy sul test. Ridge usa standardizzazione e alpha 0.1, 1, 10 selezionato sulla validation per R² posizione. Punti: media tre fold per seed; barre: mediana. PCA è deterministica. Non si usa q(z|x,u) per decoding.\n''',encoding='utf-8')


def main():
    for panel in ('b','c','d'):
        for rat in RATS:(DEST/panel/rat.capitalize()).mkdir(parents=True,exist_ok=True)
    figures_bc()
    rows=calc_2d();figures_d(rows)
    for panel in ('b','c'):
        for rat in RATS[1:]:
            (DEST/panel/rat.capitalize()/'RIFERIMENTI.md').write_text('Nessun grafico: le nuove stime richieste 2b–2c sono state eseguite solo per Achilles.\n',encoding='utf-8')
    (DEST/'RIFERIMENTI.md').write_text('''# Figura 2 — riferimenti\n\nCodice: `cebra_replica/rat_figura2_20261028.py`. I pannelli b e c usano solo le stime Achilles in `reproduction/figure2_bc` e le stime Soft full-recording T=1 indicate nei JSON per grafico. CEBRA e Soft sono separati. Due prove di input comportamentale 2D sono esplorative, non configurazioni originali confermate. In 2c le stime 8D mostrano coordinate 1–3; i dati ufficiali contengono 5000 valori loss, questi fit 10000.\n\nIl pannello d usa i quattro ratti e le stime 1e/ED4; i calcoli sono in `d/calcoli` e `d/metriche_fold.csv`. Non sono stati riaddestrati encoder. Le righe incomplete e le esclusioni sono esplicitate, nessun q(z|x,u) per decoding.\n\nNotebook ufficiale: https://cebra.ai/docs/cebra-figures/figures/Figure2.html\n''',encoding='utf-8')
    print('COMPLETE',DEST,flush=True)


if __name__=='__main__':main()
