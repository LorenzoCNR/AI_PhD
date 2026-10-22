"""Apply the user-authorized six-seed execution cap to the unchanged sealed study."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from . import behavior_only_runner as campaign
from . import rat_runner as rr
from .data import config_hash, sha256

EXECUTION_SEEDS = tuple(range(42, 48))


def limited_configs(configs):
    selected = [(architecture, config) for architecture, config in configs if config['seed'] in EXECUTION_SEEDS]
    if len(selected) != 144 or {c['seed'] for _, c in selected} != set(EXECUTION_SEEDS):
        raise ValueError('Expected exactly144 unchanged configurations for seeds42-47')
    identities = {(a,c['rat'],c['fold'],c['seed'],c['version']) for a,c in selected}
    if len(identities) != 144:
        raise ValueError('Duplicate configuration in execution subset')
    return selected


def budget_state(state, directive):
    return dict(state, original_planned=168, planned=144, execution_seeds=list(EXECUTION_SEEDS),
                execution_budget_directive=str(directive),
                execution_budget_directive_sha256=sha256(directive))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-for-stop', action='store_true')
    args = parser.parse_args()
    output = campaign.DEFAULT_OUTPUT
    plan_path = output / 'CAMPAIGN_PLAN.json'
    gate_path = output / 'qualification' / 'GATE.json'
    plan = json.loads(plan_path.read_text())
    source_record = campaign.sources()
    if plan['sources'] != source_record:
        raise RuntimeError('Original sealed training sources changed')
    selected = limited_configs(campaign.enumerate_configs())
    specification = dict(original_planned=168, execution_planned=144, execution_seeds=list(EXECUTION_SEEDS),
        original_plan_sha256=sha256(plan_path), original_gate_sha256=sha256(gate_path),
        launcher_sha256=sha256(__file__), training_source_signature=config_hash(source_record),
        fit_configs_changed=False, training_source_changed=False, completed_fits_retrained=False,
        update='User explicitly reduced preliminary replication budget to six seeds on2026-10-19')
    specification_path = output / 'SIX_SEED_EXECUTION_PLAN.json'
    if specification_path.exists():
        if json.loads(specification_path.read_text()) != specification:
            raise RuntimeError('Six-seed execution specification changed; preserve prior directive')
    else:
        rr.atomic_json(specification_path, specification)
    if not args.execute:
        print(json.dumps(dict(**specification, optimizer_updates=0)), flush=True)
        return
    lock = output / 'RUNNING.lock'
    if lock.exists() and not args.wait_for_stop:
        raise RuntimeError('Previous worker still owns lock; use --wait-for-stop after requesting safe stop')
    deadline = time.monotonic() + 1800
    if lock.exists():
        print(json.dumps(dict(status='waiting_for_previous_worker_to_finish_current_fit')), flush=True)
    while lock.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Previous worker did not finish safe stop within30minutes')
        time.sleep(5)
    state_path = output / 'CAMPAIGN_STATE.json'
    state = json.loads(state_path.read_text())
    if any(int(key.rsplit('seed_',1)[1]) not in EXECUTION_SEEDS for key in state['records']):
        raise RuntimeError('Existing out-of-budget records require explicit accounting; no artifact deleted')
    directives = output / 'execution_directives'
    directives.mkdir(exist_ok=True)
    directive = directives / f'{time.time_ns()}_six_seed_budget.json'
    rr.atomic_json(directive, dict(**specification, created_utc=rr.now(),
        previous_state_sha256=sha256(state_path), previous_completed=state.get('completed',0),
        test_source_sha256=sha256(Path(__file__).parent/'tests'/'test_behavior_only_six_seed.py')))
    stop = output / 'STOP_AFTER_CURRENT'
    if stop.exists():
        if state.get('status') != 'stopped_after_current':
            raise RuntimeError('Stop request not yet acknowledged by previous worker')
        stop.rename(directives / f'{time.time_ns()}_consumed_stop_request.txt')
    original_execute, original_atomic = campaign.execute, rr.atomic_json
    def atomic_with_budget(path, value):
        if Path(path).resolve() == state_path.resolve():
            value = budget_state(value, directive)
        return original_atomic(path, value)
    def execute_capped(output, configs, prepared, sources, device, max_new_fits, report_every):
        return original_execute(output, limited_configs(configs), prepared, sources, device, max_new_fits, report_every)
    rr.atomic_json = atomic_with_budget
    campaign.execute = execute_capped
    original_argv = sys.argv
    try:
        sys.argv = [__file__, '--execute', '--report-every', '20']
        campaign.main()
    finally:
        sys.argv = original_argv
        campaign.execute, rr.atomic_json = original_execute, original_atomic


if __name__ == '__main__':
    main()
