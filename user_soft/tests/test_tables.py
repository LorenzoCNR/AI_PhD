"""Regression: decoder identity must not collide with encoder/objective identity."""
import csv
import tempfile
from pathlib import Path
from unittest import TestCase
from user_soft.rat_runner import atomic_json,write_comparison

class TableQualification(TestCase):
    def test_decoder_model_identifiers_distinct_in_rolling_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic_json(root/'stage03_metrics'/'toy'/'metrics.json',dict(
                config=dict(rat='toy',fold=0,seed=42,version='soft_v1_frozen'),
                metrics=[dict(method='knn',decoder='knn',representation='embedding_raw',split='test',position_r2=.5)]))
            write_comparison(root)
            with (root/'RAT_SOFT_COMPARISON.csv').open() as f:
                row=next(csv.DictReader(f))
            self.assertEqual(row['method'],'soft_v1_frozen')
            self.assertEqual(row['decoder'],'knn')
            self.assertEqual(row['position_r2'],'0.5')
