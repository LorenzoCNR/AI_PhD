"""Resume the sealed overnight campaign with retries for Windows file sharing.

This launcher changes no training configuration or artifact validation.  It
retries only transient PermissionError from atomic os.replace operations.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


_replace = os.replace


def replace_with_retry(source, destination):
    target = Path(destination)
    if target.name not in ('CAMPAIGN_STATE.json', 'SIX_GROUP_PROGRESS.csv'):
        return _replace(source, destination)
    for attempt in range(12):
        try:
            return _replace(source, destination)
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(0.1 * (attempt + 1), 0.5))


def main():
    os.replace = replace_with_retry
    from . import rat_runner as rr
    from .data import sha256
    output = rr.ROOT / 'results' / 'user_soft_rat_overnight_v1'
    state = output / 'CAMPAIGN_STATE.json'
    if '--execute' in sys.argv:
        rr.atomic_json(output / 'execution_directives' / f'{time.time_ns()}_resume_override.json', {
            'created_utc': rr.now(),
            'launcher': 'user_soft.resume_overnight',
            'launcher_sha256': sha256(__file__),
            'command': [sys.executable, '-m', 'user_soft.resume_overnight', *sys.argv[1:]],
            'previous_state_sha256': sha256(state) if state.exists() else None,
            'reason': 'Resume after transient Windows state-file replace failure; user requested continuation.',
            'deadline_override': 'The earlier 2026-10-15 09:00 Europe/Rome cutoff is superseded by the subsequent request to continue.',
            'scientific_settings_changed': False,
        })
    from .overnight_runner import main as campaign_main
    campaign_main()


if __name__ == '__main__':
    main()
