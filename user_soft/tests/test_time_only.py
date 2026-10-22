"""Qualification of an explicitly time-only dense soft target, using CPU toys."""
import copy
import unittest
import torch
from user_soft import time_only_runner as temporal
from user_soft import rat_runner as runner
from user_soft.loss import evaluate_soft,fit_distance_scales
from user_soft.tests import test_runner as runner_tests


class TimeOnlyQualification(runner_tests.RunnerQualification):
    def setUp(self):
        super().setUp()
        self.config=temporal.frozen_config("toy",0,42,"soft_v1_stable",updates=4)
        self.config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)
        self.scales=fit_distance_scales(self.splits["train"].metadata(),temporal.SPECS,
            split="train",source_ids=self.splits["train"].source_id.tolist())
    def fit(self,folder=None,config=None):
        with temporal.context_bindings():
            return super().fit(folder,config)
    def test_time_only_specs_and_behavior_permutation_invariance(self):
        self.assertEqual(len(temporal.SPECS),1)
        self.assertEqual(temporal.SPECS[0]["key"],"time_id")
        self.assertEqual(temporal.SPECS[0]["geometry"],"temporal")
        batch=self.splits["train"].metadata()
        z=torch.randn(48,3)
        permuted={k:v.clone() for k,v in batch.items()}
        permuted["behavior"]=batch["behavior"][torch.arange(47,-1,-1)]*1000
        for version in ("soft_v1_frozen","soft_v1_stable"):
            a=evaluate_soft(z,batch,temporal.SPECS,version=version,scales=self.scales)
            b=evaluate_soft(z,permuted,temporal.SPECS,version=version,scales=self.scales)
            torch.testing.assert_close(a.Q,b.Q,rtol=0,atol=0)
            torch.testing.assert_close(a.loss,b.loss,rtol=0,atol=0)
    def test_relative_time_cross_trial_relationship(self):
        batch=self.splits["train"].metadata()
        result=evaluate_soft(torch.randn(48,3),batch,temporal.SPECS,scales=self.scales)
        self.assertNotEqual(batch["trial_id"][0],batch["trial_id"][24])
        self.assertEqual(batch["time_id"][0],batch["time_id"][24])
        self.assertGreater(result.Q[0,24],result.Q[0,23])
        self.assertGreater(result.Q[0,24],0)
    def test_frozen_time_formula_independent(self):
        batch=self.splits["train"].metadata()
        z=torch.randn(48,3,dtype=torch.float64)
        result=evaluate_soft(z,batch,temporal.SPECS,version="soft_v1_frozen",scales=self.scales)
        times=batch["time_id"].double()
        d=abs(times[:,None]-times[None,:]);d=d/d.max()
        w=torch.exp(-d/.5);w.fill_diagonal_(0)
        q=w/w.sum(1,keepdim=True)
        unit=z/z.norm(dim=1,keepdim=True)
        logits=unit@unit.T/.1;logits.fill_diagonal_(-torch.inf)
        logp=torch.log_softmax(logits,1)
        mask=~torch.eye(48,dtype=torch.bool)
        reference=-(q[mask]*logp[mask]).sum()/48
        torch.testing.assert_close(result.Q,q,rtol=2e-6,atol=2e-7)
        torch.testing.assert_close(result.loss,reference,rtol=2e-6,atol=2e-7)
    def test_training_scales_not_reestimated_and_no_test_scale_fit(self):
        before=copy.deepcopy(self.scales)
        batch=self.splits["valid"].metadata()
        batch["time_id"]=batch["time_id"]*100
        evaluate_soft(torch.randn(48,3),batch,temporal.SPECS,scales=self.scales)
        self.assertEqual(before,self.scales)
        with self.assertRaises(ValueError):
            fit_distance_scales(batch,temporal.SPECS,split="test",source_ids=batch["source_id"])
    def test_bindings_preserve_original_cnn_factory_and_restore_specs(self):
        factory=runner.make_encoder
        original_specs=runner.SPECS
        with temporal.context_bindings():
            self.assertIs(runner.make_encoder,factory)
            self.assertEqual(runner.SPECS,temporal.SPECS)
            model=runner.make_encoder(4,self.config,self.device)
            self.assertEqual(type(model).__name__,"Offset10Model")
            self.assertEqual(tuple(model(torch.randn(8,4,10)).shape),(8,32))
        self.assertIs(runner.make_encoder,factory)
        self.assertIs(runner.SPECS,original_specs)
    def test_both_loss_versions_have_finite_cpu_encoder_gradients(self):
        values=runner.tensors(self.splits["train"],self.device)
        for version in ("soft_v1_frozen","soft_v1_stable"):
            model=runner.make_encoder(4,self.config,self.device)
            with temporal.context_bindings():
                result=runner.objective(model,values,torch.arange(16),dict(self.config,version=version),self.scales)
            result.loss.backward()
            self.assertTrue(torch.isfinite(result.loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None),0)
    def test_both_versions_complete_small_cpu_fits(self):
        for version in ("soft_v1_frozen","soft_v1_stable"):
            config=dict(self.config,version=version)
            record,created=self.fit(folder=self.root/version,config=config)
            self.assertTrue(created)
            self.assertEqual(record["actual_updates"],4)
            self.assertTrue(all(torch.isfinite(torch.tensor(row["train_loss"])) for row in record["history"]))
            self.assertNotEqual(record["initial_parameter_hash"],record["final_parameter_hash"])
    def test_source_signatures_outside_binding_and_exact_scale_projection(self):
        outside=temporal.sources()
        with temporal.context_bindings():
            with self.assertRaises(RuntimeError):temporal.sources()
        self.assertEqual(outside,temporal.sources())
        parent=fit_distance_scales(self.splits["train"].metadata(),runner.SPECS,
            split="train",source_ids=self.splits["train"].source_id.tolist())
        projected=temporal.project_scales(parent)
        original=[s for s in parent["scales"] if s["key"]=="time_id"]
        self.assertEqual(projected["scales"],original)
        self.assertEqual(projected["train_id_sha256"],parent["train_id_sha256"])
        self.assertEqual(projected["specs"],temporal.SPECS)
        parent["split"]="test"
        with self.assertRaises(ValueError):temporal.project_scales(parent)


if __name__=="__main__":
    unittest.main()
