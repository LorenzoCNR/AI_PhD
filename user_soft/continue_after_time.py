"""Resume the approved Transformer campaign only after CNN time-only finishes."""
import json
import subprocess
import sys
import time
import psutil
from .rat_runner import ROOT, atomic_json, now


def main():
    cnn = ROOT/'results/user_soft_rat_time_only_v1'
    output = ROOT/'results/user_soft_rat_transformer_5seed_v1'
    output.mkdir(parents=True,exist_ok=True)
    path = output/'SEQUENCE_STATE.json'
    status = dict(created_utc=now(),cnn_target=168,transformer_target=240,
                  training_seeds=list(range(42,47)),folds=[0,1,2],
                  command=[sys.executable,'-m','user_soft.transformer_campaign','--execute',
                           '--seed-count','5','--output',str(output)])
    deadline = time.monotonic()+12*3600
    while time.monotonic()<deadline:
        state = json.loads((cnn/'CAMPAIGN_STATE.json').read_text())
        status.update(status='waiting_CNN_time_and_qualification',cnn_completed=state.get('completed',0),updated_utc=now())
        atomic_json(path,status)
        if state.get('status') in ('failed','interrupted'):
            raise RuntimeError('CNN time stopped unexpectedly; next phase not launched')
        if state.get('completed')==168 and state.get('status') in ('complete','requested_limit') and not (cnn/'RUNNING.lock').exists():
            gate = output/'qualification/GATE.json'
            if gate.exists():
                if json.loads(gate.read_text()).get('decision')!='GO':
                    raise RuntimeError('Expanded Transformer qualification not GO')
                lock = ROOT/'results/user_soft_rat_transformer_v1/RUNNING.lock'
                if lock.exists():
                    worker_id=int(lock.read_text())
                    if worker_id!=72064 or psutil.pid_exists(worker_id):
                        raise RuntimeError('Unexpected/alive original Transformer worker; refusing lock removal')
                    lock.unlink()
                    status['removed_stale_original_lock_pid']=worker_id
                status.update(status='running_Transformer',updated_utc=now())
                atomic_json(path,status)
                # This runner verifies source-qualified gates and immutable resume dependencies.
                result=subprocess.run(status['command'],cwd=ROOT,check=False)
                status.update(status='Transformer_finished' if result.returncode==0 else 'failed',
                              returncode=result.returncode,updated_utc=now())
                atomic_json(path,status)
                if result.returncode:raise RuntimeError('Transformer worker failed; preserve state for inspection')
                return
        time.sleep(30)
    status.update(status='timeout_no_launch',updated_utc=now())
    atomic_json(path,status)
    raise RuntimeError('Timed out; no new fit launched')


if __name__=='__main__':main()
