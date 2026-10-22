"""Start the approved overnight sequence only after its qualification GO gate."""
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
from .rat_runner import ROOT, atomic_json, now


def main():
    output=ROOT/'results/user_soft_rat_overnight_v1'
    output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat('2026-10-15T07:00:00+00:00')
    command=[sys.executable,'-m','user_soft.overnight_runner','--execute','--wait',
             '--max-seeds','10','--deadline-utc',deadline.isoformat()]
    gate=output/'qualification/GATE.json'
    status=dict(created_utc=now(),deadline_utc=deadline.isoformat(),
                deadline_local='2026-10-15 09:00 Europe/Rome',command=command)
    path=output/'LAUNCH_STATE.json'
    while datetime.now(timezone.utc)<deadline:
        status.update(status='waiting_qualification',updated_utc=now())
        atomic_json(path,status)
        if gate.exists():
            if json.loads(gate.read_text()).get('decision')!='GO':
                raise RuntimeError('Overnight gate not GO; no launch')
            status.update(status='running_overnight_supervisor',updated_utc=now())
            atomic_json(path,status)
            result=subprocess.run(command,cwd=ROOT,check=False)
            status.update(status='overnight_finished' if result.returncode==0 else 'failed',
                          returncode=result.returncode,updated_utc=now())
            atomic_json(path,status)
            if result.returncode:raise RuntimeError('Overnight worker failed; preserve all state')
            return
        time.sleep(15)
    status.update(status='deadline_no_launch',updated_utc=now())
    atomic_json(path,status)


if __name__=='__main__':main()
