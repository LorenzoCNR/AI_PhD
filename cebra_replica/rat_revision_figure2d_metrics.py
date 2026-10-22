"""Per-rat Figure 2d decoding panels from existing held-out metric records."""
import json
from collections import defaultdict
import numpy as np
from rat_revision_common import ROOT, STAGE, RATS, SEEDS, SOFT_KEYS, DISPLAY, soft_root, sha, csv_rows, write_csv, save

OUT = STAGE / 'Figure_2/2D'
METRICS = ('position_median_absolute_error_cm', 'direction_accuracy', 'direction_balanced_accuracy')
REF_NAMES = {'CEBRA-Behavior-32D':'CEBRA-Behavior','CEBRA-Time-32D':'CEBRA-Time',
             'Conv-piVAE-z_mean':'conv-piVAE','PCA':'PCA','UMAP':'UMAP','OPENTSNE':'tSNE'}
ORDER = ['CEBRA-Behavior','CEBRA-Time','conv-piVAE','PCA','UMAP','tSNE'] + [DISPLAY[k].replace('\n',' - ') for k in SOFT_KEYS]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    OUT.mkdir(parents=True,exist_ok=True)
    issue = OUT/'FIGURE2D_METRICS_ISSUES_BEFORE_CHANGE.json'
    if not issue.exists():
        issue.write_text(json.dumps({'issue':'Crowded multi-rat decoding figures omit new behavior-only Soft and mix decoding quantities',
            'severity':'REPORTING ONLY','evidence':'Existing presentation table lacks behavior-only campaign completed after its snapshot',
            'fix':'Four separate rat figures; full six Soft conditions; kNN/Ridge legend; no R2 plot; mark unavailable reference metrics NA',
            'training_required':False},indent=2),encoding='utf-8')
    parents=[]; long=[]; seed_rows=[]; configs=0
    for rat in RATS:
        for arch,condition in SOFT_KEYS:
            name=DISPLAY[arch,condition].replace('\n',' - ')
            for seed in SEEDS:
                for fold in range(3):
                    root=soft_root(arch,condition,rat,seed,fold)
                    path=root/'stage03_metrics'/rat/f'fold_{fold}'/'soft_v1_stable'/f'seed_{seed}'/'metrics.json'
                    record=json.loads(path.read_text())
                    # The metric record is linked to its frozen embedding manifest.
                    marker=root/'stage02_embedding'/rat/f'fold_{fold}'/'soft_v1_stable'/f'seed_{seed}'/'manifest.json'
                    if record.get('parent_embedding_manifest_sha256') != sha(marker):
                        raise ValueError(f'Metric parent manifest mismatch:{path}')
                    parent=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
                    parents.append(parent); configs+=1
                    found={r['decoder']:r for r in record['metrics'] if r['split']=='test' and r['representation']=='embedding_raw'}
                    if set(found)!={'knn','ridge'}:raise ValueError(f'Missing frozen decoder metric:{path}')
                    for decoder,item in found.items():
                        long.append(dict(rat=rat,method=name,seed=seed,fold=fold,decoder=decoder,split='test',representation='embedding_raw',
                            **{m:item.get(m) for m in METRICS},position_r2=item.get('position_r2'),parent_metrics_path=parent['path'],parent_metrics_sha256=parent['sha256']))
    ref_path=ROOT/'results/RAT_FIGURES/Tables/CEBRA_REFERENCE_DECODING_BY_SEED.csv'
    parents.append(dict(path=ref_path.relative_to(ROOT).as_posix(),sha256=sha(ref_path)))
    for r in csv_rows(ref_path):
        if r['method'] not in ('CEBRA-Behavior','CEBRA-Time') or r['split']!='test' or r['representation']!='embedding_raw' or int(r['seed']) not in SEEDS:continue
        if int(r['n_folds'])!=3:raise ValueError('Incomplete CEBRA reference fold aggregation')
        for m in METRICS+('position_r2',):
            if r.get(m) not in ('',None):seed_rows.append(dict(rat=r['rat'],method=r['method'],seed=r['seed'],decoder=r['decoder'],metric=m,value=float(r[m]),n_folds=3,source='frozen_reference_seed_table'))
    base_path=ROOT/'results/RAT_FIGURES/Tables/PAPER_REFERENCE_METHOD_DECODING_ALL_METHODS_LONG.csv'
    parents.append(dict(path=base_path.relative_to(ROOT).as_posix(),sha256=sha(base_path)))
    checked={}
    for r in csv_rows(base_path):
        if r['method'] not in REF_NAMES or r['method'].startswith('CEBRA') or r['split']!='test':continue
        if r['seed']!='deterministic' and int(r['seed']) not in SEEDS:continue
        parent=ROOT/r['parent_result_path']
        if parent not in checked:
            if sha(parent)!=r['parent_result_sha256']:raise ValueError(f'Reference parent mismatch:{parent}')
            checked[parent]=True
        long.append(dict(rat=r['rat'],method=REF_NAMES[r['method']],seed=r['seed'],fold=int(r['fold']),decoder='knn',split='test',representation='embedding_raw',
            **{m:float(r[m]) if r.get(m) not in ('',None) else None for m in METRICS},position_r2=float(r['position_r2']),parent_metrics_path=r['parent_result_path'],parent_metrics_sha256=r['parent_result_sha256']))
    grouped=defaultdict(list)
    for r in long:
        for metric in METRICS+('position_r2',):
            if r.get(metric) is not None:grouped[r['rat'],r['method'],str(r['seed']),r['decoder'],metric].append((r['fold'],float(r[metric])))
    for (rat,name,seed,decoder,metric),vals in grouped.items():
        folds=[f for f,v in vals]
        if len(set(folds))!=len(folds):raise ValueError('Duplicate fold scores')
        if name.startswith('Soft') and set(folds)!={0,1,2}:raise ValueError('Incomplete Soft fold aggregation')
        seed_rows.append(dict(rat=rat,method=name,seed=seed,decoder=decoder,metric=metric,value=float(np.mean([v for f,v in vals])),n_folds=len(vals),source='equal_weight_available_fold_mean'))
    if configs!=432:raise ValueError(f'Expected432 Soft cells,got{configs}')
    write_csv(OUT/'FIGURE2D_METRICS_FOLD_RECORDS.csv',long)
    write_csv(OUT/'FIGURE2D_METRICS_SEED_RECORDS.csv',seed_rows)
    coverage=[]
    plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','text.color':'black'})
    colors={'knn':'#202020','ridge':'#3276a7'}; markers={'knn':'o','ridge':'^'}
    for rat in RATS:
        fig,axes=plt.subplots(1,3,figsize=(15,8.8),sharey=True)
        for ax,metric in zip(axes,METRICS):
            for i,name in enumerate(ORDER):
                for decoder in ('knn','ridge') if metric==METRICS[0] else ('knn',):
                    rs=[r for r in seed_rows if r['rat']==rat and r['method']==name and r['decoder']==decoder and r['metric']==metric]
                    coverage.append(dict(rat=rat,method=name,metric=metric,decoder=decoder,n_seeds=len(rs),seeds=';'.join(str(r['seed']) for r in rs),n_folds=';'.join(str(r['n_folds']) for r in rs),status='AVAILABLE' if rs else 'NOT_AVAILABLE'))
                    if not rs:
                        if decoder=='knn':ax.text(.97,i,'NA',ha='right',va='center',transform=ax.get_yaxis_transform(),color='#777777',fontsize=9)
                        continue
                    vals=np.array([r['value'] for r in rs]);offset=-.11 if decoder=='knn' else .11
                    ys=np.full(len(vals),i+offset)+np.linspace(-.045,.045,len(vals))
                    ax.scatter(vals,ys,s=22,color=colors[decoder],marker=markers[decoder],alpha=.75)
                    med=float(np.median(vals));ax.plot([med,med],[i+offset-.09,i+offset+.09],color=colors[decoder],lw=2)
            ax.set_xlabel({'position_median_absolute_error_cm':'Median absolute position error (cm)', 'direction_accuracy':'Direction accuracy','direction_balanced_accuracy':'Direction balanced accuracy'}[metric])
            ax.set_yticks(range(len(ORDER)),ORDER)
            ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
            if metric!=METRICS[0]:ax.set_xlim(0,1)
        axes[0].invert_yaxis()
        for tick in axes[0].get_yticklabels():
            if tick.get_text().startswith('Soft'):tick.set_color('red')
        fig.suptitle(f'{rat.capitalize()} - held-out decoding',fontsize=15)
        fig.legend(handles=[Line2D([],[],marker=markers[d],color=colors[d],linestyle='None',label='kNN' if d=='knn' else 'Ridge (position only)') for d in ('knn','ridge')],loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False)
        fig.text(.5,.015,'Dots: training seeds; short lines: median. Each seed averages available folds. NA: metric not saved.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.22,right=.98,bottom=.16,top=.91,wspace=.18)
        save(fig,OUT/f'FIGURE2D_{rat.upper()}_DECODING','Figure2D',parents,dict(rat=rat,metrics=list(METRICS),representation='raw',split='held-out test',soft_seeds=list(SEEDS),soft_folds=[0,1,2],r2_retained_in_csv_only=True,reference_missing_accuracy_or_ridge='NA, not inferred from BAcc',aggregation='equal-weight fold means per seed; no pooling overlapping windows'))
    write_csv(OUT/'FIGURE2D_METRICS_COVERAGE.csv',coverage)
    print(json.dumps(dict(soft_cells=configs,fold_records=len(long),seed_records=len(seed_rows),rat_figures=4,missing_reference_metric_cells=sum(r['status']=='NOT_AVAILABLE' for r in coverage))))


if __name__=='__main__':main()
