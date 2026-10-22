import json
from unittest.mock import patch
import numpy as np
from user_soft.tests.test_runner import RunnerQualification
from user_soft import behavior_only_runner as b, shuffle_runner as sh, rat_runner as rr
from user_soft import resume_behavior_shuffle as recovery
from user_soft.data import save_rows,sha256
from user_soft.loss import fit_distance_scales,evaluate_soft


class RecoveryQualification(RunnerQualification):
    def test_behavior_shuffle_fit_export_diagnostic_resume(self):
        original=self.splits['train']
        shuffled,permutation=sh.shuffle_train(original,42)
        config=b.frozen_config('cnn','toy',0,42,updates=4)
        config.update(batch_size=32,checkpoint_interval=2,diagnostic_interval=2)
        scales=fit_distance_scales(original.metadata(),b.SPECS,split='train',source_ids=original.source_id)
        with b.context_bindings('cnn'):
            fit,_=rr.fit_cell(self.root/'fit',shuffled,self.splits['valid'],config,scales,'parent',self.sources,self.device)
            data=self.root/'data';data.mkdir()
            for split,rows in self.splits.items():save_rows(data/f'{split}.npz',rows)
            emb,manifest=rr.export_cell(self.root/'out',self.root/'fit',data,config,fit,self.device)
        checkpoint_hash=sha256(self.root/'fit/checkpoint.pt')
        embedding_hash=sha256(emb/'manifest.json')
        recovery.diagnostic(self.root/'out',emb,original,shuffled,permutation,dict(train_scales=scales),config,dict(signature='toy'))
        path=self.root/'out/stage04_diagnostics/toy/fold_0/soft_v1_stable/seed_42'
        import torch
        with np.load(path/'TRAIN_REFERENCE_W_Q_P.npz') as saved:
            np.testing.assert_array_equal(saved['behavior_donor_source_id'],original.source_id[permutation[saved['reference_train_row_indices']]])
            batch={k:torch.as_tensor(saved[k]) for k in ('behavior','source_id','time_id','trial_id','session_id')}
            with np.load(emb/'train_embedding.npz') as e:
                z=torch.as_tensor(e['embedding'][saved['reference_train_row_indices']])
            expected=evaluate_soft(z,batch,b.SPECS,scales=scales)
            np.testing.assert_allclose(saved['Q'],expected.Q.numpy())
        with patch.object(recovery,'evaluate_soft',side_effect=AssertionError('Unexpected diagnostic recomputation')):
            recovery.diagnostic(self.root/'out',emb,original,shuffled,permutation,dict(train_scales=scales),config,dict(signature='toy'))
        with patch.object(rr,'make_encoder',side_effect=AssertionError('Unexpected fit')):
            reused,created=rr.fit_cell(self.root/'fit',None,None,config,None,'parent',self.sources,self.device)
        self.assertFalse(created)
        self.assertEqual(sha256(self.root/'fit/checkpoint.pt'),checkpoint_hash)
        self.assertEqual(sha256(emb/'manifest.json'),embedding_hash)
        self.assertTrue(json.loads((path/'PROVENANCE.json').read_text())['no_time_contribution'])
