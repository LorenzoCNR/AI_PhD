"""Fast CPU toy tests of fit/checkpoint/export stage invariants."""
import copy
import json
import tempfile
from pathlib import Path
from unittest import TestCase,mock
import numpy as np
import torch
from user_soft import rat_runner as runner
from user_soft.data import TrialRows,save_rows,eligible_centers
from user_soft.loss import fit_distance_scales


class RunnerQualification(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix="user_soft_runner_cpu_")
        self.root=Path(self.temporary.name)
        rng=np.random.default_rng(42)
        self.splits={}
        for offset,split in enumerate(("train","valid","test")):
            times=np.tile(np.arange(24),2)
            pos=np.tile(np.linspace(0,1,24),2).astype(np.float32)
            signal=np.column_stack((pos,np.sin(pos*3),np.cos(pos*3),pos**2))
            neural=(signal+.03*rng.normal(size=signal.shape)).astype(np.float32)
            behavior=np.column_stack((pos,np.ones(48),np.zeros(48))).astype(np.float32)
            self.splits[split]=TrialRows(neural=neural,behavior=behavior,
                source_id=np.arange(offset*100,offset*100+48),trial_id=np.repeat([offset*10,offset*10+1],24),
                time_id=times,session_id=np.zeros(48,dtype=np.int64),split=split)
        self.config=runner.frozen_config("toy",0,42,"soft_v1_stable",updates=4)
        self.config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)
        self.scales=fit_distance_scales(self.splits["train"].metadata(),runner.SPECS,
                                      split="train",source_ids=self.splits["train"].source_id.tolist())
        self.sources={"qualification":"fixed_toy_source"}
        self.device=torch.device("cpu")
    def tearDown(self):
        self.temporary.cleanup()
    def fit(self,folder=None,config=None):
        return runner.fit_cell(self.root/"fit" if folder is None else folder,
                               self.splits["train"],self.splits["valid"],
                               self.config if config is None else config,self.scales,
                               "immutable_toy_parent",self.sources,self.device)
    def test_completed_checkpoint_reused_without_fit(self):
        record,created=self.fit()
        self.assertTrue(created)
        self.assertEqual(record["actual_updates"],4)
        self.assertNotEqual(record["initial_parameter_hash"],record["final_parameter_hash"])
        before=(self.root/"fit"/"checkpoint.pt").read_bytes()
        with mock.patch.object(runner,"objective",side_effect=AssertionError("Unexpected model refit")):
            reused,created=self.fit()
        self.assertFalse(created)
        self.assertEqual(reused["final_parameter_hash"],record["final_parameter_hash"])
        self.assertEqual((self.root/"fit"/"checkpoint.pt").read_bytes(),before)
    def test_config_invalidation_refuses_overwrite(self):
        self.fit()
        changed=copy.deepcopy(self.config)
        changed["embedding_temperature"]*=2
        before=(self.root/"fit"/"checkpoint.pt").read_bytes()
        with self.assertRaises(RuntimeError):
            self.fit(config=changed)
        self.assertEqual((self.root/"fit"/"checkpoint.pt").read_bytes(),before)
    def test_embedding_metadata_and_resume(self):
        record,_=self.fit()
        data_folder=self.root/"data"
        data_folder.mkdir()
        for split,rows in self.splits.items():
            save_rows(data_folder/f"{split}.npz",rows)
        output=self.root/"output"
        folder,manifest=runner.export_cell(output,self.root/"fit",data_folder,self.config,record,self.device)
        original_manifest=(folder/"manifest.json").read_bytes()
        for split,rows in self.splits.items():
            with np.load(folder/f"{split}_embedding.npz",allow_pickle=False) as archive:
                self.assertEqual(archive["embedding_raw"].shape,(48,32))
                for key in ("embedding_unit","behavior","source_id","trial_id","time_id","session_id","valid_mask"):
                    self.assertEqual(len(archive[key]),48)
                np.testing.assert_array_equal(archive["trial_id"],rows.trial_id)
                np.testing.assert_array_equal(archive["time_id"],rows.time_id)
                self.assertEqual(archive["valid_mask"].sum(),len(eligible_centers(rows)))
                np.testing.assert_allclose(np.linalg.norm(archive["embedding_unit"],axis=1),1,rtol=1e-5)
        with mock.patch.object(runner,"make_encoder",side_effect=AssertionError("Unexpected encoder loading")):
            reused,second=runner.export_cell(output,self.root/"fit",data_folder,self.config,record,self.device)
        self.assertEqual(folder,reused)
        self.assertEqual(manifest,second)
        self.assertEqual((folder/"manifest.json").read_bytes(),original_manifest)
    def test_interrupted_fit_exact_rng_resume(self):
        uninterrupted,_=self.fit(folder=self.root/"reference")
        original=runner.objective
        calls=0
        def interrupt(model,values,indices,config,scales):
            nonlocal calls
            # Diagnostics use model.eval(); interrupt only training update3.
            if model.training:
                calls+=1
                if calls==3:
                    raise KeyboardInterrupt("Artificial CPU qualification interruption")
            return original(model,values,indices,config,scales)
        with mock.patch.object(runner,"objective",side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.fit(folder=self.root/"resumed")
        saved=torch.load(self.root/"resumed"/"checkpoint.pt",weights_only=False)
        self.assertEqual(saved["step"],2)
        resumed,created=self.fit(folder=self.root/"resumed")
        self.assertTrue(created)
        self.assertEqual(resumed["actual_updates"],4)
        self.assertEqual(uninterrupted["final_parameter_hash"],resumed["final_parameter_hash"])
    def test_context_and_center_metadata_match_official_expand_index(self):
        import cebra.data
        from user_soft.data import legacy_windows
        rows=self.splits["train"]
        dataset=cebra.data.TensorDataset(rows.neural,continuous=rows.behavior,
                                       offset=cebra.data.Offset(5,5),device="cpu")
        centers=torch.tensor([0,4,5,23,24,42,47])
        support=dataset.expand_index(centers)
        expected=torch.as_tensor(rows.neural)[support].transpose(1,2)
        torch.testing.assert_close(legacy_windows(rows,centers),expected,rtol=0,atol=0)
        values=runner.tensors(rows,self.device)
        # A simple shape-preserving fake encoder inspects the exact context.
        class InspectEncoder(torch.nn.Module):
            def forward(self,x):
                self.context=x.detach().clone()
                return x.mean(2)
        encoder=InspectEncoder()
        z,batch=runner.encode_batch(encoder,values,centers)
        torch.testing.assert_close(encoder.context,expected,rtol=0,atol=0)
        np.testing.assert_array_equal(batch["source_id"].numpy(),rows.source_id[centers])
        np.testing.assert_array_equal(batch["time_id"].numpy(),rows.time_id[centers])
