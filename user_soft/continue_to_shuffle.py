"""Launch the qualified five-seed shuffle only after all Transformer cells."""
import json
import subprocess
import sys
import time
from .rat_runner import ROOT, atomic_json, now


def main():
    parent=ROOT/'results/user_soft_rat_transformer_5seed_v1'
    output=ROOT/'results/user_soft_rat_shuffle_5seed_v1'
    output.mkdir(parents=True,exist_ok=True)
    path=output/'SEQUENCE_STATE.json'
    command=[sys.executable,'-m','user_soft.shuffle_runner','--execute',
             '--seed-count','5','--output',str(output),
             '--transformer-output',str(parent),'--transformer-fit-count','240']
    state=dict(created_utc=now(),parent=str(parent),required_completed=240,command=command)
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        manifest=parent/'CAMPAIGN_STATE.json'
        previous=json.loads(manifest.read_text()) if manifest.exists() else {}
        state.update(status='waiting_Transformer',parent_completed=previous.get('completed',0),updated_utc=now())
        atomic_json(path,state)
        if previous.get('status') in ('failed','interrupted'):
            raise RuntimeError('Transformer interrupted/failed; shuffle not launched')
        if previous.get('completed')==240 and previous.get('status') in ('complete','requested_limit') and not (parent/'RUNNING.lock').exists():
            if json.loads((output/'qualification/GATE.json').read_text()).get('decision')!='GO':
                raise RuntimeError('Shuffle gate not GO')
            state.update(status='running_shuffle',updated_utc=now())
            atomic_json(path,state)
            # Executor independently checks source hashes and both completed predecessors.
            result=subprocess.run(command,cwd=ROOT,check=False)
            state.update(status='shuffle_finished' if result.returncode==0 else 'failed',
                         returncode=result.returncode,updated_utc=now())
            atomic_json(path,state)
            if result.returncode:raise RuntimeError('Shuffle worker failed; artifacts retained')
            return
        time.sleep(30)
    state.update(status='timeout_no_launch',updated_utc=now())
    atomic_json(path,state)
    raise RuntimeError('Timed out awaiting Transformer; no shuffle launched')


if __name__=='__main__':main()
