"""Verify the requested rat replacement, then remove obsolete active plot files only."""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from rat_revision_common import ROOT,DELIVERY,STAGE,sha

PANELS={'Figure_1/1D':4,'Figure_1/1E':1,'Figure_2/2B':8,'Figure_2/2C':4,
        'Figure_2/2D':12,'Extended_Data/ED3':16}


def contained(path,parent):
    if parent.resolve() not in path.resolve().parents:raise RuntimeError(f'Path outside approved directory:{path}')


def validate():
    images=[]
    for name,count in PANELS.items():
        folder=STAGE/name
        found=sorted(folder.glob('*.png'))
        if len(found)!=count:raise RuntimeError(f'Expected{count} PNG in{name}, found{len(found)}')
        for path in found:
            contained(path,STAGE)
            pdf=path.with_suffix('.pdf')
            if not pdf.is_file() or pdf.stat().st_size<1000:raise RuntimeError(f'Missing/empty PDF:{pdf}')
            with Image.open(path) as im:
                if min(im.size)<200:raise RuntimeError('Figure resolution too small')
                if 'A' in im.getbands() and im.getchannel('A').getextrema()!=(255,255):raise RuntimeError('Transparent figure')
                small=np.asarray(im.convert('RGB').resize((128,128)))
                if np.std(small)<3 or not np.all(small[0,0]>245):raise RuntimeError(f'Blank/nonwhite figure:{path}')
            images.append(dict(path=path.relative_to(STAGE).as_posix(),png_sha256=sha(path),pdf_sha256=sha(pdf)))
    if len(images)!=45:raise RuntimeError('Unexpected total figure count')
    return images


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    images=validate()
    obsolete=[]
    for name in PANELS:
        folder=DELIVERY/name
        if not folder.is_dir():raise RuntimeError(f'Existing panel directory missing:{folder}')
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix.lower() in ('.png','.pdf','.svg','.html'):
                contained(path,folder);obsolete.append(path)
                sidecar=path.with_name(path.stem+'_PROVENANCE.json')
                if sidecar.is_file():contained(sidecar,folder);obsolete.append(sidecar)
    obsolete=sorted(set(obsolete))
    removed=[dict(path=p.relative_to(DELIVERY).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in obsolete]
    staged=sorted(p for p in STAGE.rglob('*') if p.is_file())
    for src in staged:
        contained(src,STAGE)
        target=DELIVERY/src.relative_to(STAGE)
        contained(target,DELIVERY)
        if target.is_file() and target.suffix=='.csv' and target.name!='ED3_PANEL_INDEX.csv':
            raise RuntimeError(f'Refuse overwrite of existing scientific CSV:{target}')
    if not args.execute:
        print(json.dumps(dict(status='PASS_DRY_RUN',new_figure_pairs=len(images),obsolete_plot_and_sidecar_files=len(removed),new_files=len(staged)),indent=2));return
    # Exact target list is persisted before deletion; checkpoints/embeddings and
    # unrequested panels are outside this whitelist and are never enumerated.
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='VERIFIED_BEFORE_PROMOTION',
        new_figures=images,removed_obsolete_files=removed,encoder_training=False,
        removal_scope=list(PANELS),source_sha256=sha(__file__),
        note='User explicitly requested deleting superseded plots; no archive copies of those plots retained.')
    manifest=DELIVERY/'RAT_FIGURE_REVISION_20261021.json'
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    for path in obsolete:path.unlink()
    replaced=[]
    for src in staged:
        target=DELIVERY/src.relative_to(STAGE)
        if target.exists():replaced.append(dict(path=target.relative_to(DELIVERY).as_posix(),old_sha256=sha(target)))
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(src),str(target))
    relocated=[]
    for name in ['Figure_1/1E/FIGURE1E_SHARED6SEED_BEHAVIOR_INCLUDED_PROVENANCE.json',
                 'Extended_Data/ED3/PROVENANCE.json','Tables/ISSUES_FIGURE1_20261021.md']:
        path=DELIVERY/name
        text=path.read_text(encoding='utf-8')
        new=text.replace('.revision_20261021/','')
        if new!=text:path.write_text(new,encoding='utf-8');relocated.append(name)
    for entry in images:
        path=DELIVERY/entry['path']
        if sha(path)!=entry['png_sha256'] or sha(path.with_suffix('.pdf'))!=entry['pdf_sha256']:
            raise RuntimeError('Image hash changed during promotion')
    # Only remove empty, resolved staging directories; never recursive deletion.
    for directory in sorted((p for p in STAGE.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        contained(directory,STAGE);directory.rmdir()
    STAGE.rmdir()
    record.update(status='COMPLETE',finished_utc=datetime.now(timezone.utc).isoformat(),
        replaced_nonimage_metadata=replaced,relocated_provenance_paths=relocated,
        active_plot_counts={n:len(list((DELIVERY/n).glob('*.png'))) for n in PANELS},
        new_files=[str((DELIVERY/p.relative_to(STAGE)).relative_to(DELIVERY)) for p in staged])
    manifest.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',figure_pairs=45,removed_obsolete_files=len(removed),manifest=str(manifest)),indent=2))


if __name__=='__main__':main()
