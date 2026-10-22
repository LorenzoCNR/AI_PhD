"""CPU-only qualification of the frozen full recovered Transformer architecture."""
import unittest
import numpy as np
import torch
from user_soft import transformer_runner as transformer
from user_soft import rat_runner as runner
from user_soft.tests import test_runner as runner_tests


class TransformerQualification(runner_tests.RunnerQualification):
    def setUp(self):
        super().setUp()
        self.config=transformer.frozen_config("toy",0,42,"soft_v1_stable",updates=4)
        self.config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)

    def fit(self,folder=None,config=None):
        with transformer.architecture_bindings():
            return super().fit(folder,config)

    def test_full_frozen_architecture_and_dropout(self):
        config=self.config
        self.assertEqual(config["transformer_settings"],dict(model_dim=64,n_heads=4,n_layers=2,
                         dropout=.1,max_length=4096,normalize=True))
        self.assertEqual(config["output_dimension"],32)
        self.assertEqual(config["offset_left"],5)
        self.assertEqual(config["offset_right"],5)
        self.assertNotIn("hidden_units",config)
        model=transformer.make_encoder(4,config,self.device)
        x=torch.randn(8,4,10)
        model.train()
        a,b=model(x),model(x)
        self.assertEqual(tuple(a.shape),(8,32))
        self.assertFalse(torch.equal(a,b))
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a.norm(dim=1),torch.ones(8),rtol=2e-5,atol=2e-6)
        model.eval()
        torch.testing.assert_close(model(x),model(x),rtol=0,atol=0)
        with self.assertRaises(ValueError):model(torch.randn(8,4,21))

    def test_encoder_independent_q_and_finite_full_transformer_gradients(self):
        model=transformer.make_encoder(4,self.config,self.device)
        values=runner.tensors(self.splits["train"],self.device)
        ids=torch.arange(8)
        model.train()
        stable=runner.objective(model,values,ids,self.config,self.scales)
        frozen=runner.objective(model,values,ids,dict(self.config,version="soft_v1_frozen"),self.scales)
        for result in (stable,frozen):
            model.zero_grad(set_to_none=True)
            result.loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0)

    def test_architecture_binding_restores_cnn_on_exception(self):
        original=runner.make_encoder
        with self.assertRaises(RuntimeError):
            with transformer.architecture_bindings():
                self.assertIs(runner.make_encoder,transformer.make_encoder)
                raise RuntimeError("qualification context exception")
        self.assertIs(runner.make_encoder,original)

    def test_embedding_metadata_and_resume(self):
        record,_=self.fit()
        from user_soft.data import save_rows,eligible_centers
        data_folder=self.root/"data_transformer"
        data_folder.mkdir()
        for split,rows in self.splits.items():save_rows(data_folder/f"{split}.npz",rows)
        output=self.root/"transformer_output"
        folder,manifest=transformer.export_cell(output,self.root/"fit",data_folder,self.config,record,self.device)
        for split,rows in self.splits.items():
            with np.load(folder/f"{split}_embedding.npz",allow_pickle=False) as archive:
                self.assertEqual(archive["embedding_raw"].shape,(48,32))
                self.assertEqual(str(archive["model"]),"TemporalTransformerEncoder")
                np.testing.assert_array_equal(archive["trial_id"],rows.trial_id)
                np.testing.assert_array_equal(archive["time_id"],rows.time_id)
                self.assertEqual(archive["valid_mask"].sum(),len(eligible_centers(rows)))
        from unittest.mock import patch
        with patch.object(transformer,"TemporalTransformerEncoder",side_effect=AssertionError("Unexpected reload")):
            reused,second=transformer.export_cell(output,self.root/"fit",data_folder,self.config,record,self.device)
        self.assertEqual(reused,folder)
        self.assertEqual(second,manifest)


if __name__=="__main__":
    unittest.main()
