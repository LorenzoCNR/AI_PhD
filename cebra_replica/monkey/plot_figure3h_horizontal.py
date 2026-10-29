"""Figure 3h: cached held-out scores in a horizontal layout."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/figura_3_2026-10-28/h'
CEBRA=ROOT/'results/PAPER_FIGURE_DELIVERY/Figure_3/FIGURE3H_PROVENANCE.json'
SOFT=ROOT/'results/PAPER_FIGURE_DELIVERY/Figure_3/FIGURE3H_SOFT_HELDOUT_SEED_VALUES.csv'
DIMS=(3,4,8,16,32)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(OUT/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    records=json.loads(CEBRA.read_text(encoding='utf8'))['score_records']
    fig,axes=plt.subplots(1,3,figsize=(18,5.4),constrained_layout=True)
    tasks=('position','direction','ap')
    titles=('Fit Position X,Y\nDecoding Position X,Y',
            'Fit Direction\nDecoding Direction',
            'Fit Position X,Y\nDecoding Active / Passive')
    ylabels=('R² Position × 100','Accuracy Direction (%)','Accuracy Active / Passive (%)')
    for ax,task,title,ylabel in zip(axes,tasks,titles,ylabels):
        for i,dim in enumerate(DIMS):
            vals=[float(r['score']) for r in records if r['task']==task and int(r['dimension'])==dim]
            if len(vals)!=5:raise ValueError(f'Expected 5 scores {task}/{dim}')
            ax.scatter(np.full(5,i),vals,s=38,facecolors='none',edgecolors='black')
            ax.hlines(np.median(vals),i-.23,i+.23,color='#e07b00',lw=2.4)
        ax.set_title(title,fontsize=12,pad=10);ax.set_xticks(range(5),DIMS)
        ax.set_xlabel('Embedding dimension');ax.set_ylabel(ylabel)
        ax.spines[['top','right']].set_visible(False)
    save(fig,'figura_3h_CEBRA_test_posizione_direzione_condizione')
    with SOFT.open(encoding='utf8',newline='') as f: data=list(csv.DictReader(f))
    objectives=[('time only (no behavior target)','Time'),
        ('position X,Y only (no direction target)','Position X,Y'),
        ('time + position X,Y (no direction target)','Time + Position X,Y')]
    # Older exports and regenerated exports may use the cleaned labels.
    fallback={'Time':'Time','Position X,Y':'Position X,Y','Time + Position X,Y':'Time + Position X,Y'}
    colors={'cnn':'#222222','transformer':'#2166ac'}
    fig,axes=plt.subplots(1,6,figsize=(33,5.5),constrained_layout=True)
    for j,(raw,label) in enumerate(objectives):
        for k,(metric,decoded) in enumerate((('position_r2','Position X,Y'),('direction_accuracy','Direction'))):
            ax=axes[j*2+k]
            for arch in ('cnn','transformer'):
                for i,dim in enumerate(DIMS):
                    vals=[float(r[metric])*100 for r in data if r['architecture']==arch and int(r['dimension'])==dim and r['objective'] in (raw,fallback[label])]
                    if len(vals)!=5:raise ValueError(f'Expected 5 scores {arch}/{label}/{dim}: {len(vals)}')
                    offset=-.08 if arch=='cnn' else .08
                    ax.scatter(i+offset+np.linspace(-.025,.025,5),vals,s=30,facecolors='none',edgecolors=colors[arch])
                    ax.plot(i+offset,np.median(vals),marker='_',markersize=16,markeredgewidth=2.2,color=colors[arch])
            ax.set_title(f'Fit {label}\nDecoding {decoded}',fontsize=12,pad=10)
            ax.set_xticks(range(5),DIMS);ax.set_xlabel('Embedding dimension')
            ax.set_ylabel('R² Position × 100' if k==0 else 'Accuracy Direction (%)')
            ax.spines[['top','right']].set_visible(False)
    axes[0].legend(handles=[Line2D([0],[0],color=colors[a],label=a.upper()) for a in ('cnn','transformer')],frameon=False,loc='best')
    save(fig,'figura_3h_Soft_test_posizione_direzione_CNN_Transformer')
    (OUT/'RIFERIMENTI_layout_orizzontale.json').write_text(json.dumps({
        'source_cebra':str(CEBRA.relative_to(ROOT)),'sha256_cebra':sha(CEBRA),
        'source_soft':str(SOFT.relative_to(ROOT)),'sha256_soft':sha(SOFT),
        'script':str(Path(__file__).relative_to(ROOT)),
        'note':'Solo ripresentazione dei risultati esistenti; nessun fit o decoder rieseguito.',
    },indent=2),encoding='utf8')

if __name__=='__main__':main()
