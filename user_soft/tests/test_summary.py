"""Aggregation tests: complete folds, independent seeds, representation separation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from user_soft.summary import aggregate,summarize,combine_recorded_sources,METRICS


def row(fold,seed=42,value=1.,representation="embedding_raw",decoder="knn"):
    return dict(rat="achilles",fold=fold,seed=seed,method="soft_v1_stable",representation=representation,
                split="test",decoder=decoder,position_median_absolute_error_cm=value,position_r2=value/10,
                direction_balanced_accuracy=None if decoder=="ridge" else value/100,
                direction_accuracy=None if decoder=="ridge" else value/100)


class SummaryQualification(unittest.TestCase):
    def test_fold_means_then_seed_sd(self):
        rows=[row(f,42,v) for f,v in enumerate((1.,2.,3.))]+[row(f,43,v) for f,v in enumerate((7.,8.,9.))]
        seeds,groups=aggregate(rows)
        self.assertEqual([s[METRICS[0]] for s in seeds],[2.,8.])
        self.assertEqual(groups[0][METRICS[0]+"_mean"],5.)
        self.assertAlmostEqual(groups[0][METRICS[0]+"_sd"],(18.)**.5)
        self.assertEqual(groups[0]["n_seeds_complete"],2)
    def test_partial_not_in_seed_mean(self):
        seeds,groups=aggregate([row(0),row(1)])
        self.assertEqual(seeds[0]["fold_status"],"PARTIAL")
        self.assertIsNone(seeds[0][METRICS[0]])
        self.assertEqual(groups[0]["n_seeds_complete"],0)
        self.assertEqual(groups[0]["n_seeds_partial"],1)
    def test_one_seed_sd_and_ridge_direction_undefined(self):
        seeds,groups=aggregate([row(f,decoder="ridge") for f in range(3)])
        self.assertIsNone(groups[0][METRICS[0]+"_sd"])
        self.assertIsNone(seeds[0]["direction_accuracy"])
        self.assertEqual(groups[0]["direction_accuracy_n_seeds"],0)
    def test_duplicate_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): aggregate([row(0),row(0)])
        with self.assertRaises(ValueError): aggregate([row(0,value=float("nan"))])
    def test_raw_unit_and_valid_test_separate(self):
        rows=[row(f) for f in range(3)]+[row(f,representation="embedding_unit") for f in range(3)]
        extra=[dict(row(f),split="valid") for f in range(3)]
        seeds,groups=aggregate(rows+extra)
        self.assertEqual(len(seeds),3)
        self.assertEqual(len(groups),3)
    def test_running_old_error_does_not_mean_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"CAMPAIGN_STATE.json").write_text(json.dumps(dict(status="running",completed=1,last_error="previous resolved failure")))
            (root/"BASELINE_DOWNSTREAM_STATE.json").write_text(json.dumps(dict(status="running",completed=1)))
            report=summarize(root)
            self.assertEqual(report["status"],"PARTIAL")
            self.assertFalse((root/"RAT_SOFT_COMPARISON.csv").exists())
    def test_complete_workers_require_metric_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ("CAMPAIGN_STATE.json","BASELINE_DOWNSTREAM_STATE.json"):
                (root/name).write_text(json.dumps(dict(status="complete",completed=240)))
            report=summarize(root)
            self.assertEqual(report["status"],"PARTIAL")
    def test_failed_gpu_retains_partial_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"CAMPAIGN_STATE.json").write_text(json.dumps(dict(status="failed",completed=10)))
            report=summarize(root)
            self.assertEqual(report["status"],"FAILED_PARTIAL")
    def test_immutable_sources_supplement_stale_csv_and_keep_model_method(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/"stage03_reference"/"achilles"/"fold_0"/"CEBRA-Behavior"/"seed_42"/"metrics.json"
            path.parent.mkdir(parents=True)
            metric=dict(row(0),method="knn")
            path.write_text(json.dumps(dict(config=dict(rat="achilles",fold=0,seed=42,version="CEBRA-Behavior"),metrics=[metric])))
            rows,parents=combine_recorded_sources(root,[])
            self.assertEqual(rows[0]["method"],"CEBRA-Behavior")
            self.assertEqual(rows[0]["decoder"],"knn")
            self.assertEqual(len(parents),1)
            same=dict(row(0),method="CEBRA-Behavior")
            rows,_=combine_recorded_sources(root,[same])
            self.assertEqual(len(rows),1)
            same[METRICS[0]]=999
            with self.assertRaises(ValueError): combine_recorded_sources(root,[same])


if __name__=="__main__":
    unittest.main()
