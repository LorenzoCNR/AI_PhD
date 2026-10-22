"""Stage requested rat1D/1E revision using frozen embeddings; zero encoder fitting."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from threadpoolctl import threadpool_limits
import rat_revision_common as common
from rat_figure_gap_closure_analysis import index_manifests, load_fit, method_specs, support_for

ROOT, STAGE, RATS, SEEDS = common.ROOT, common.STAGE, common.RATS, common.SEEDS
TABLES = STAGE / 'Tables'
KEYS = common.SOFT_KEYS
REPS = ('embedding_raw', 'embedding_unit')
OLD_TABLE = common.DELIVERY / 'Tables/SOFT_CROSS_RAT_CONSISTENCY_LONG.csv'
BASE_TABLE = common.DELIVERY / 'Tables/REFERENCE_CROSS_RAT_CONSISTENCY_BY_SEED.csv'
NEW_TABLE = TABLES / 'FIGURE1E_NEW_CROSS_RAT_ROWS_20261021.csv'


def architecture_label(a):
    return 'CNN1D' if a == 'CNN' else 'Transformer'


def validate_support(arrays, rat, fold):
    baseline = support_for(rat, fold)['test']
    for field in ('source_id', 'behavior', 'valid_mask'):
        if not np.array_equal(arrays[field], baseline[field]):
            raise RuntimeError(f'Frozen common support mismatch:{rat}/fold{fold}/{field}')
    n = len(arrays['source_id'])
    for rep in REPS:
        if arrays[rep].shape != (n, 32) or not np.isfinite(arrays[rep]).all():
            raise RuntimeError(f'Invalid32D Soft export:{rat}/{rep}')


def figure1d(index):
    specs = [s for s in method_specs(index) if s['available'] and s['display'] != 'Conv-piVAE (test labels)']
    if len(specs) != 6 or sum(s['kind'] == 'pivae' for s in specs) != 1:
        raise RuntimeError('Expected6 reference methods with one q(z|x) piVAE output')
    rows = []
    for rat in RATS:
        panels, parents = [], []
        for spec in specs:
            d = load_fit(index, spec['kind'], spec['method'], rat, 0, spec['seed'], 'test', spec.get('variant', 'raw'))
            mask = d['valid_mask']
            label = 'Conv-piVAE' if spec['kind'] == 'pivae' else spec['display']
            panels.append((label, d['embedding'][mask], d['behavior'][mask], d['source_id'][mask], False))
            parents.append(dict(path=d['embedding_path'].relative_to(ROOT).as_posix(), sha256=d['embedding_sha256'],
                                manifest=d['manifest_path'].relative_to(ROOT).as_posix(), manifest_sha256=common.sha(d['manifest_path'])))
        for a, c in KEYS:
            arrays, parent = common.embedding(a, c, rat, seed=42, fold=0)
            validate_support(arrays, rat, 0)
            mask = arrays['valid_mask']
            panels.append((common.DISPLAY[(a,c)], arrays['embedding_raw'][mask], arrays['behavior'][mask], arrays['source_id'][mask], True))
            parents.append(parent)
        ids, behavior = panels[0][3], panels[0][2]
        for label, emb, beh, source, _ in panels:
            if not np.array_equal(ids, source) or not np.array_equal(behavior, beh):
                raise RuntimeError(f'Gallery held-out support mismatch:{rat}/{label}')
            rows.append(dict(rat=rat, method=label.replace('\n',' '), seed=42, fold=0,
                             split='held_out_test', n_samples=len(ids), fit_dimension=emb.shape[1],
                             plotted_dimensions=2, representation='embedding_raw'))
        norm = Normalize(float(behavior[:,0].min()), float(behavior[:,0].max()))
        fig, axes = plt.subplots(2, 6, figsize=(21, 8.0), facecolor='white')
        for ax, (label, emb, beh, _, soft) in zip(axes.flat, panels):
            for col, cmap in ((2,'cool'), (1,'viridis')):
                selected = beh[:,col] == 1
                ax.scatter(emb[selected,0], emb[selected,1], c=beh[selected,0], cmap=cmap, norm=norm,
                           s=2.2, linewidths=0, rasterized=True)
            ax.set_title(label, color='red' if soft else 'black', fontsize=12, pad=10)
            ax.set_facecolor('white'); ax.set_axis_off()
        fig.subplots_adjust(left=.025, right=.99, top=.94, bottom=.12, hspace=.23, wspace=.16)
        for bounds, cmap, title in (([.36,.055,.12,.015],'cool','Left'),([.53,.055,.12,.015],'viridis','Right')):
            colorbar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap), cax=fig.add_axes(bounds), orientation='horizontal')
            colorbar.set_ticks([]); colorbar.set_label(title, fontsize=11)
        common.save(fig, STAGE/'Figure_1/1D'/f'FIGURE1D_{rat.upper()}_BEHAVIOR_INCLUDED', '1D', parents,
                    dict(rat=rat, seed=42, fold=0, split='held_out_test', n_samples=len(ids),
                         projection='first2nativecoords; neural32D vs baseline2D; descriptive not dimension-matched',
                         piVAE='one q(z|x) output; no test-time labels', marker_size=2.2,
                         legends='Left and Right only; cool/viridis', encoder_fit=False))
    common.write_csv(TABLES/'FIGURE1D_PANEL_INDEX_20261021.csv', rows)


def row_key(r):
    return (r['architecture'], r['condition'], r['representation'], int(r['fold']), int(r['seed']), r['source_rat'], r['target_rat'])


def prepare_consistency():
    reused = [r for r in common.csv_rows(OLD_TABLE) if int(r['seed']) in SEEDS]
    for r in reused:
        r['value_origin'] = 'existing_frozen_metric_unchanged'
    existing = {row_key(r): r for r in reused}
    if len(existing) != len(reused):
        raise RuntimeError('Duplicate existing consistency cells')
    checked = {}
    for r in reused:
        for side in ('source','target'):
            p = ROOT/r[f'{side}_embedding_path']
            if str(p) not in checked:
                checked[str(p)] = common.sha(p)
            if checked[str(p)] != r[f'{side}_embedding_sha256']:
                raise RuntimeError(f'Existing consistency parent hash changed:{p}')
    if NEW_TABLE.exists():
        fresh = common.csv_rows(NEW_TABLE)
        marker = NEW_TABLE.with_suffix('.json')
        provenance = json.loads(marker.read_text())
        if provenance['new_table_sha256'] != common.sha(NEW_TABLE) or provenance['existing_table_sha256'] != common.sha(OLD_TABLE):
            raise RuntimeError('Staged consistency cache parent/hash mismatch')
        for p in provenance['embedding_parents']:
            if common.sha(ROOT/p['path']) != p['sha256']:
                raise RuntimeError('Staged consistency embedding parent changed')
    else:
        from cebra.integrations.sklearn.metrics import consistency_score
        fresh, parents = [], {}
        with threadpool_limits(limits=2):
            for a,c in KEYS:
                for fold in (0,1,2):
                    for seed in SEEDS:
                        missing = [rep for rep in REPS if not all((architecture_label(a),c,rep,fold,seed,ra,rb) in existing for ra in RATS for rb in RATS if ra != rb)]
                        if not missing:
                            continue
                        data = [common.embedding(a,c,rat,seed=seed,fold=fold) for rat in RATS]
                        for rat, (arrays,parent) in zip(RATS,data):
                            validate_support(arrays,rat,fold)
                            parents[parent['path']] = parent
                        for rep in missing:
                            selected = [arrays['valid_mask'] for arrays,_ in data]
                            values,pairs,_ = consistency_score([arrays[rep][mask] for (arrays,_),mask in zip(data,selected)],
                                between='datasets', labels=[arrays['behavior'][mask,0] for (arrays,_),mask in zip(data,selected)],
                                dataset_ids=list(RATS), num_discretization_bins=100)
                            for value,pair in zip(values,pairs):
                                ra,rb = map(str,pair)
                                k=(architecture_label(a),c,rep,fold,seed,ra,rb)
                                if k in existing:
                                    continue
                                pa,pb = data[RATS.index(ra)][1],data[RATS.index(rb)][1]
                                fresh.append(dict(method=f'Soft {a}',condition=c,architecture=architecture_label(a),representation=rep,
                                    fold=fold,seed=seed,source_rat=ra,target_rat=rb,held_out_behavior_aligned_r2=float(value),
                                    n_source_rows=int(selected[RATS.index(ra)].sum()),n_target_rows=int(selected[RATS.index(rb)].sum()),
                                    source_embedding_path=pa['path'],source_embedding_sha256=pa['sha256'],
                                    target_embedding_path=pb['path'],target_embedding_sha256=pb['sha256'],value_origin='new_missing_downstream_only'))
                        print(f'1E evaluated missing:{a}/{c}/fold{fold}/seed{seed}',flush=True)
        if len(fresh) != 1008:
            raise RuntimeError(f'Expected864behavior+144Transformer47 newrows, found{len(fresh)}')
        common.write_csv(NEW_TABLE,fresh)
        NEW_TABLE.with_suffix('.json').write_text(json.dumps(dict(new_table_sha256=common.sha(NEW_TABLE),
            existing_table_sha256=common.sha(OLD_TABLE),script_sha256=common.sha(__file__),
            embedding_parents=list(parents.values()),metric='CEBRA consistency_score between=datasets,position labels,100bins,directedR2',
            optimizer_updates=0,selection='prespecified6seeds42-47,3folds,all4rats'),indent=2),encoding='utf-8')
    combined = reused + fresh
    if len(combined) != 2592 or len({row_key(r) for r in combined}) != 2592:
        raise RuntimeError('Incomplete/duplicate6methods x6seeds x3fold x2representation x12directedpairs')
    common.write_csv(TABLES/'FIGURE1E_SOFT_CROSS_RAT_COMPLETE_20261021.csv',combined)
    return combined


def figure1e():
    soft = prepare_consistency()
    refs = [r for r in common.csv_rows(BASE_TABLE) if r['method'] != 'Conv-piVAE with test labels' and (r['seed']=='deterministic' or int(r['seed']) in SEEDS)]
    baseline_methods = ('CEBRA-Behavior','CEBRA-Time','Conv-piVAE no test labels','PCA','UMAP','t-SNE')
    seed_rows = []
    grouped = defaultdict(list)
    for r in soft:
        grouped[(r['architecture'],r['condition'],r['representation'],int(r['seed']),r['source_rat'],r['target_rat'])].append(r)
    for key, rows in grouped.items():
        if sorted(int(r['fold']) for r in rows) != [0,1,2]:
            raise RuntimeError('Every seed must have exactly3folds')
        a,c,rep,seed,ra,rb=key
        seed_rows.append(dict(method='Soft '+('CNN' if a=='CNN1D' else a),condition=c,architecture=a,representation=rep,
                             seed=seed,source_rat=ra,target_rat=rb,n_folds=3,
                             mean_across_folds=mean(float(r['held_out_behavior_aligned_r2']) for r in rows)))
    common.write_csv(TABLES/'FIGURE1E_SOFT_BY_SEED_20261021.csv',seed_rows)
    panels=[]
    for method in baseline_methods:
        selected=[r for r in refs if r['method']==method]
        panels.append((method.replace('Conv-piVAE no test labels','Conv-piVAE'),selected,False))
    for a,c in KEYS:
        selected=[r for r in seed_rows if r['architecture']==architecture_label(a) and r['condition']==c and r['representation']=='embedding_raw']
        panels.append((common.DISPLAY[(a,c)],selected,True))
    matrices,summary=[],[]
    for label,rows,soft_flag in panels:
        matrix=np.full((4,4),np.nan)
        expected_seeds=1 if label=='PCA' else 6
        for i,ra in enumerate(RATS):
            for j,rb in enumerate(RATS):
                if i==j:continue
                values=[float(r['mean_across_folds']) for r in rows if r['source_rat']==ra and r['target_rat']==rb]
                if len(values)!=expected_seeds:raise RuntimeError(f'Wrong6seed coverage:{label}/{ra}->{rb}/{len(values)}')
                matrix[i,j]=mean(values)*100
                summary.append(dict(method=label.replace('\n',' '),source_rat=ra,target_rat=rb,n_seeds=len(values),
                    mean_r2=mean(values),sd_across_seed_means=stdev(values) if len(values)>1 else None,
                    representation='embedding_raw',metric='directed_behavior_aligned_CEBRA_consistency_R2',
                    averaging='3fold_mean_per_seed_then6seed_mean; PCA deterministic'))
        matrices.append(matrix)
    low=min(float(np.nanmin(m)) for m in matrices)
    high=max(100.,max(float(np.nanmax(m)) for m in matrices))
    norm=Normalize(vmin=min(0.,low),vmax=high,clip=False)
    fig,axes=plt.subplots(2,6,figsize=(20,6.9),facecolor='white')
    cmap=plt.colormaps['gray_r'].copy();cmap.set_bad('white')
    for ax,(label,rows,soft_flag),matrix in zip(axes.flat,panels,matrices):
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,norm=norm,interpolation='nearest')
        for i in range(4):
            for j in range(4):
                if i!=j:
                    color='white' if norm(matrix[i,j])>.55 else 'black'
                    ax.text(j,i,f'{matrix[i,j]:.0f}',ha='center',va='center',fontsize=12,color=color)
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(label,fontsize=12,color='red' if soft_flag else 'black',pad=10)
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.02,right=.94,top=.93,bottom=.065,hspace=.35,wspace=.24)
    cb=fig.colorbar(im,cax=fig.add_axes([.958,.22,.012,.55]));cb.set_label('$R^2$ (%)')
    common.save(fig,STAGE/'Figure_1/1E/FIGURE1E_SHARED6SEED_BEHAVIOR_INCLUDED','1E',
        [dict(path=p.relative_to(ROOT).as_posix(),sha256=common.sha(p)) for p in (OLD_TABLE,BASE_TABLE,NEW_TABLE)],
        dict(metric='exact existing CEBRA consistency_score directedR2 between datasets,positionaligned,100bins',
             seeds=list(SEEDS),folds=[0,1,2],matrix_order=list(RATS),rows='source rat',columns='target rat',
             representations='galleryraw;rawandunitseparateinCSV',clipping='none;exactnumericvaluesretained',
             newly_computed_rows=1008,reused_soft_rows=1584,encoder_updates=0,piVAE='oneq(z|x),notlabelconditionedinference'))
    common.write_csv(TABLES/'FIGURE1E_MATRIX_SUMMARY_20261021.csv',summary)
    common.write_csv(TABLES/'FIGURE1E_REFERENCE_6SEED_SUBSET_20261021.csv',refs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',choices=('1d','1e','all'),default='all')
    args=parser.parse_args()
    if args.panel in ('1d','all'):figure1d(index_manifests())
    if args.panel in ('1e','all'):figure1e()


if __name__=='__main__':main()
