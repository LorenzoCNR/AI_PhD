"""Rolling descriptive fold/seed aggregation of already computed rat metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

METRICS=("position_median_absolute_error_cm","position_r2","direction_balanced_accuracy","direction_accuracy")
GROUP_KEYS=("rat","method","representation","split","decoder")
SEED_KEYS=("rat","method","seed","representation","split","decoder")
EXPECTED_FOLDS=(0,1,2)
METHODS=("CEBRA-Behavior","CEBRA-Time","soft_v1_frozen","soft_v1_stable")
RATS=("achilles","buddy","cicero","gatsby")


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if value is None or value=="":
        return None
    value=float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite scientific metric in input")
    return value


def aggregate(records):
    """Mean folds first; seed SD is computed only from complete three-fold seeds."""
    cells=defaultdict(list)
    seen=set()
    for source in records:
        row=dict(source)
        if row.get("method") not in METHODS or row.get("representation") not in ("embedding_raw","embedding_unit"):
            raise ValueError("Unknown method or representation")
        if row.get("split") not in ("valid","test") or row.get("decoder") not in ("knn","ridge"):
            raise ValueError("Explicit valid/test and knn/ridge required")
        row["seed"]=int(row["seed"])
        row["fold"]=int(row["fold"])
        if row.get("rat") not in RATS or row["seed"] not in range(42,52):
            raise ValueError("Unexpected rat or training seed")
        if row["fold"] not in EXPECTED_FOLDS:
            raise ValueError("Unexpected fold")
        key=tuple(row[k] for k in SEED_KEYS)
        unique=key+(row["fold"],)
        if unique in seen:
            raise ValueError(f"Duplicate scientific cell: {unique}")
        seen.add(unique)
        for metric in METRICS:
            row[metric]=number(row.get(metric))
        cells[key].append(row)
    seeds=[]
    for key,rows in sorted(cells.items()):
        folds=sorted(r["fold"] for r in rows)
        complete=folds==list(EXPECTED_FOLDS)
        seed=dict(zip(SEED_KEYS,key),n_folds=len(folds),folds_present=";".join(map(str,folds)),
                  fold_status="COMPLETE" if complete else "PARTIAL",aggregation="equal_weight_mean_of_three_folds")
        for metric in METRICS:
            values=[r[metric] for r in rows]
            if any(v is None for v in values) and not all(v is None for v in values):
                raise ValueError(f"Mixed defined/undefined metric across folds: {key}/{metric}")
            seed[metric]=statistics.mean(values) if complete and values[0] is not None else None
        seeds.append(seed)
    groups=defaultdict(list)
    for row in seeds:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    summaries=[]
    for key,rows in sorted(groups.items()):
        complete=[r for r in rows if r["fold_status"]=="COMPLETE"]
        group=dict(zip(GROUP_KEYS,key),n_seeds_complete=len(complete),n_seeds_partial=len(rows)-len(complete),
                   seeds_complete=";".join(str(r["seed"]) for r in complete),
                   seeds_observed=";".join(str(r["seed"]) for r in rows),
                   n_folds_observed=sum(r["n_folds"] for r in rows),expected_folds_per_seed=3,
                   scope="training_seed_variability_of_equal_weight_fold_means; no_CI_no_pvalue")
        for metric in METRICS:
            values=[r[metric] for r in complete if r[metric] is not None]
            group[metric+"_n_seeds"]=len(values)
            group[metric+"_mean"]=statistics.mean(values) if values else None
            group[metric+"_sd"]=statistics.stdev(values) if len(values)>1 else None
            group[metric+"_min"]=min(values) if values else None
            group[metric+"_max"]=max(values) if values else None
        summaries.append(group)
    return seeds,summaries


def atomic_text(path,text):
    path=Path(path)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    try:
        with tmp.open("w",encoding="utf-8",newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path,rows,empty_fields):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else empty_fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path,stream.getvalue())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def scientific_key(row):
    return tuple(int(row[k]) if k=="seed" else row[k] for k in SEED_KEYS)+(int(row["fold"]),)


def combine_recorded_sources(output,csv_rows):
    """Supplement rolling CSV from exact immutable metric records, without recomputation."""
    rows={}
    for row in csv_rows:
        key=scientific_key(row)
        if key in rows:
            raise ValueError(f"Duplicate CSV scientific cell: {key}")
        rows[key]=dict(row)
    parents={}
    seen_direct=set()
    for stage in ("stage03_metrics","stage03_reference"):
        for path in sorted((output/stage).glob("*/fold_*/*/seed_*/metrics.json")):
            payload=path.read_bytes()
            record=json.loads(payload)
            config=record["config"]
            parents[str(path.relative_to(output))]=hashlib.sha256(payload).hexdigest()
            for metric in record["metrics"]:
                # Metric-level method is a decoder alias; config.version identifies the encoder/loss.
                row=dict(metric,rat=config["rat"],fold=config["fold"],seed=config["seed"],method=config["version"])
                key=scientific_key(row)
                if key in seen_direct:
                    raise ValueError(f"Duplicate immutable scientific cell: {key}")
                seen_direct.add(key)
                if key in rows:
                    if any(number(rows[key].get(m))!=number(row.get(m)) for m in METRICS):
                        raise ValueError(f"CSV/source metric discrepancy: {key}")
                else:
                    rows[key]=row
    return list(rows.values()),parents


def summarize(output):
    output=Path(output).resolve()
    input_path=output/"RAT_SOFT_COMPARISON.csv"
    # Read one atomic CSV snapshot; scientific source rows are never rewritten.
    payload=input_path.read_bytes() if input_path.exists() else b""
    csv_rows=list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))) if payload else []
    rows,metric_parents=combine_recorded_sources(output,csv_rows)
    seeds,groups=aggregate(rows)
    gpu=read_json(output/"CAMPAIGN_STATE.json")
    baseline=read_json(output/"BASELINE_DOWNSTREAM_STATE.json")
    failed=gpu.get("status") in ("failed","interrupted") or baseline.get("status") in ("failed","interrupted")
    finished=gpu.get("status")=="complete" and baseline.get("status")=="complete"
    complete_rows=sum(r["fold_status"]=="COMPLETE" for r in seeds)
    expected_seed_rows=4*4*10*2*2*2
    coverage_complete=len(rows)==3840 and len(seeds)==expected_seed_rows and complete_rows==expected_seed_rows
    status="FAILED_PARTIAL" if failed else "COMPLETE" if finished and coverage_complete else "PARTIAL"
    report=dict(updated_utc=now(),status=status,input_rows=len(rows),rolling_csv_rows=len(csv_rows),immutable_metric_files=len(metric_parents),seed_summary_rows=len(seeds),
                complete_seed_cells=complete_rows,partial_seed_cells=len(seeds)-complete_rows,
                expected_seed_cells=expected_seed_rows,group_summary_rows=len(groups),
                gpu_status=gpu.get("status","NOT_STARTED"),gpu_completed=gpu.get("completed",0),gpu_planned=gpu.get("planned",240),
                baseline_status=baseline.get("status","NOT_STARTED"),baseline_completed=baseline.get("completed",0),baseline_planned=baseline.get("planned",240),
                complete_requires_both_workers_and_all_four_methods=True,
                inference="descriptive_only; fold_means_then_training_seed_SD; no_CI_no_pvalues",
                resume_instruction="Resume the existing GPU/reference worker from its existing checkpoint/state if status is FAILED_PARTIAL; do not rerun completed cells")
    write_csv(output/"RAT_SOFT_SEED_SUMMARY.csv",seeds,list(SEED_KEYS)+["n_folds","fold_status"]+list(METRICS))
    write_csv(output/"RAT_SOFT_GROUP_SUMMARY.csv",groups,list(GROUP_KEYS)+["n_seeds_complete","n_seeds_partial"])
    parents={input_path.name:hashlib.sha256(payload).hexdigest()}
    parents.update(metric_parents)
    for name in ("CAMPAIGN_PLAN.json","CAMPAIGN_STATE.json","BASELINE_DOWNSTREAM_STATE.json","BASELINE_DOWNSTREAM_PROVENANCE.json"):
        path=output/name
        if path.exists():
            parents[name]=file_hash(path)
    provenance=dict(updated_utc=report["updated_utc"],script_sha256=file_hash(__file__),
                    input_parent_hashes=parents,config=dict(expected_folds=EXPECTED_FOLDS,
                    methods=METHODS,metrics=METRICS,fold_aggregation="equal_weight_mean",seed_sd="sample_sd_ddof1",
                    partial_seed_values="blank_until_three_folds_complete",n1_sd="undefined_blank"),
                    training_seeds=list(range(42,52)),splits=["valid","test"],representation_policy="raw_and_unit_separate",
                    scientific_metric_recomputation=False,model_fit=False)
    atomic_text(output/"SUMMARY_PROVENANCE.json",json.dumps(provenance,indent=2,allow_nan=False)+"\n")
    atomic_text(output/"SUMMARY_STATUS.json",json.dumps(report,indent=2,allow_nan=False)+"\n")
    text=("# Rolling rat soft comparison summary\n\n"
          f"Updated {report['updated_utc']}. Status **{status}**. GPU {report['gpu_completed']}/{report['gpu_planned']}; "
          f"reference downstream {report['baseline_completed']}/{report['baseline_planned']}.\n\n"
          f"{len(rows)} source rows, {complete_rows} complete three-fold seed cells, {len(seeds)-complete_rows} partial seed cells.\n\n"
          "RAT_SOFT_SEED_SUMMARY.csv averages the three folds within each rat, method, training seed, representation, split and decoder. "
          "Incomplete fold sets retain explicit coverage and blank estimates. RAT_SOFT_GROUP_SUMMARY.csv reports mean, sample SD, minimum and maximum "
          "over complete training-seed values. SD is blank with one seed; undefined Ridge direction metrics remain blank. "
          "Raw/unit and validation/test are separate. Fold averages and training-seed SD are descriptive; overlapping fold samples are not treated as IID trials. "
          "No confidence intervals, p-values, fitting, model selection or recomputed scientific metrics are produced.\n")
    atomic_text(output/"RAT_SOFT_DESCRIPTIVE_SUMMARY.md",text)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"results"/"user_soft_rat_v1")
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--poll-seconds",type=float,default=30)
    args=parser.parse_args()
    if args.poll_seconds<1:
        parser.error("poll-seconds must be at least one")
    output=args.output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    lock=output/"SUMMARY_RUNNING.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    worker=dict(pid=os.getpid(),started_utc=now(),script_sha256=file_hash(__file__),watch=args.watch,poll_seconds=args.poll_seconds,
                encoder_fits=0,scientific_metric_recomputation=False)
    atomic_text(output/"SUMMARY_WORKER.json",json.dumps(worker,indent=2)+"\n")
    last_signature=None
    try:
        while True:
            report=summarize(output)
            signature=(report["status"],report["input_rows"],report["gpu_completed"],report["baseline_completed"])
            if signature!=last_signature:
                print(json.dumps(report),flush=True)
                last_signature=signature
            if not args.watch or report["status"] in ("COMPLETE","FAILED_PARTIAL"):
                break
            time.sleep(args.poll_seconds)
    except BaseException as exc:
        atomic_text(output/"SUMMARY_FAILURE.json",json.dumps(dict(updated_utc=now(),error=repr(exc),pid=os.getpid()),indent=2)+"\n")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    main()
