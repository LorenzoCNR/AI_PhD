"""Independent CPU-only frozen-reference downstream worker; no encoder fits."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import itertools
import json
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from . import baseline_reference as reference
from .data import sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def publish_immutable_json(path, value):
    """Publish complete JSON exclusively, never replace an immutable artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable target already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == "nt":
            # Windows rename fails if target exists: atomic publication without overwrite.
            temporary.rename(path)
        else:
            # link is exclusive on POSIX, where rename would otherwise replace a target.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def save_state(path, value):
    """The explicitly mutable campaign state is atomically replaced."""
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def publication_smoke():
    with tempfile.TemporaryDirectory(prefix="reference_atomic_json_") as directory:
        path = Path(directory) / "metrics.json"
        payload = {"rows": list(range(1000)), "complete": True}
        publish_immutable_json(path, payload)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
        before = sha256(path)
        try:
            publish_immutable_json(path, {"overwrite": True})
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing artifact overwritten")
        assert sha256(path) == before
        assert not list(Path(directory).glob("*.tmp"))
        try:
            publish_immutable_json(Path(directory) / "bad.json", {"bad": float("nan")})
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite JSON unexpectedly published")
        assert not (Path(directory) / "bad.json").exists()
        assert not list(Path(directory).glob("*.tmp"))
    return {"complete_json": "PASS", "overwrite_rejected": "PASS", "cleanup": "PASS", "failed_serialization_not_published": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=reference.ROOT / "results" / "user_soft_rat_v1")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    checks = publication_smoke()
    if args.smoke_only:
        print(json.dumps(checks), flush=True)
        return
    output = reference._output(args.output)
    gate = output / "qualification" / "GATE.json"
    # Qualification gate pathname is supplied by the parent campaign.
    if not gate.exists():
        alternate = reference.ROOT / "user_soft" / "qualification" / "GATE.json"
        if alternate.exists():
            gate = alternate
        else:
            raise RuntimeError("Technical qualification GATE.json is absent")
    gate_value = json.loads(gate.read_text(encoding="utf-8"))
    if gate_value.get("decision") != "GO" or gate_value.get("tests_failed") != 0 or not gate_value.get("plan_sources_current"):
        raise RuntimeError("Technical qualification gate does not report GO")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "BASELINE_DOWNSTREAM_RUNNING.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state_path = output / "BASELINE_DOWNSTREAM_STATE.json"
    provenance_path = output / "BASELINE_DOWNSTREAM_PROVENANCE.json"
    try:
        source_hashes = {name: sha256(Path(__file__).parent / name) for name in
                         ("baseline_campaign.py", "baseline_reference.py", "downstream.py", "data.py")}
        provenance = dict(created_utc=now(), sources=source_hashes,
                          gate_path=str(gate), gate_sha256=sha256(gate),
                          planned=240, rats=reference.RATS, folds=[0, 1, 2], seeds=list(range(42, 52)),
                          conditions=reference.CONDITIONS,
                          io_override="baseline_reference._write_json -> exclusive temporary write+fsync+atomic rename; numerical semantics unchanged",
                          cpu_threads=1, encoder_fits=0, root_comparison_csv_writer=False,
                          publication_checks=checks)
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            if previous["sources"] != source_hashes or previous["gate_sha256"] != provenance["gate_sha256"]:
                raise RuntimeError("Reference worker source/gate provenance changed; use separate worker version")
        else:
            publish_immutable_json(provenance_path, provenance)
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else dict(
            created_utc=now(), planned=240, records={}, encoder_fits=0)
        state.update(status="running", pid=os.getpid(), updated_utc=now())
        save_state(state_path, state)
        reference._write_json = publish_immutable_json
        for rat, fold, seed, condition in itertools.product(reference.RATS, range(3), range(42, 52), reference.CONDITIONS):
            key = f"{rat}/fold_{fold}/{reference.CONDITIONS[condition]}/seed_{seed}"
            state.update(current=key, updated_utc=now())
            save_state(state_path, state)
            record = reference.evaluate_reference(output, rat, fold, seed, condition)
            path = output / "stage03_reference" / key / "metrics.json"
            state["records"][key] = dict(status="complete", metric_signature=record["signature"],
                                         metrics_sha256=sha256(path), n_records=len(record["metrics"]))
            state.update(completed=len(state["records"]), updated_utc=now())
            save_state(state_path, state)
            if state["completed"] % 16 == 0 or state["completed"] == 240:
                print(json.dumps(dict(completed=state["completed"], planned=240, last=key, encoder_fits=0)), flush=True)
        state.update(status="complete", current=None, updated_utc=now())
        save_state(state_path, state)
    except BaseException as exc:
        if "state" in locals():
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=repr(exc), updated_utc=now())
            save_state(state_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
