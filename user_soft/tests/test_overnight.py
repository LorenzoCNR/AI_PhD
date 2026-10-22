"""Overnight orchestration qualification; no GPU or scientific-outcome selection."""
from collections import Counter
from datetime import datetime,timedelta,timezone
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from user_soft import overnight_runner as overnight
from user_soft import rat_runner as cnn,time_only_runner as temporal
from user_soft import transformer_runner as transformer,transformer_campaign as expanded
from user_soft import shuffle_runner as shuffle
from user_soft.data import config_hash,sha256

FAMILIES=('cnn_joint','cnn_time','transformer_joint','transformer_time','shuffle_cnn','shuffle_transformer')


class OvernightQualification(unittest.TestCase):
    def test_full_potential_1440_unique_and_fixed_training_settings(self):
        jobs=overnight.build_jobs()
        self.assertEqual(len(jobs),1440)
        self.assertEqual(Counter(job['family'] for job in jobs),{family:240 for family in FAMILIES})
        self.assertEqual({job['config']['seed'] for job in jobs},set(range(42,52)))
        keys={(job['family'],job['config']['rat'],job['config']['fold'],job['config']['seed'],job['config']['version']) for job in jobs}
        self.assertEqual(len(keys),1440)
        self.assertTrue(all(job['config']['max_updates']==10000 for job in jobs))
    def test_future_phase_order_align_seven_then_eight_nine_ten(self):
        jobs=overnight.execution_jobs()
        self.assertEqual(len(jobs),864)
        first,second=jobs[:96],jobs[96:432]
        self.assertEqual({job['family'] for job in first},{'transformer_joint','transformer_time'})
        self.assertEqual({job['config']['seed'] for job in first},{47,48})
        self.assertEqual({job['family'] for job in second},{'shuffle_cnn','shuffle_transformer'})
        self.assertEqual({job['config']['seed'] for job in second},set(range(42,49)))
        for index,seed in enumerate((49,50,51)):
            chunk=jobs[432+index*144:432+(index+1)*144]
            self.assertEqual(len(chunk),144)
            self.assertEqual({job['config']['seed'] for job in chunk},{seed})
            self.assertEqual(Counter(job['family'] for job in chunk),{family:24 for family in FAMILIES})
    def test_new_seed_chunks_are_family_balanced_at_every_prefix(self):
        jobs=overnight.execution_jobs()[432:]
        for seed in (49,50,51):
            count=Counter({family:0 for family in FAMILIES})
            for job in (job for job in jobs if job['config']['seed']==seed):
                count[job['family']]+=1
                self.assertLessEqual(max(count.values())-min(count.values()),1)
    def test_each_family_exact_original_config(self):
        for family in FAMILIES:
            for seed in (42,48,51):
                args=('achilles',1,seed,'soft_v1_stable')
                if family=='cnn_joint':expected=cnn.frozen_config(*args)
                elif family=='cnn_time':expected=temporal.frozen_config(*args)
                elif family=='transformer_joint':expected=expanded.config_for(*args,'joint')
                elif family=='transformer_time':expected=expanded.config_for(*args,'time_only')
                else:expected=shuffle.frozen_config('cnn' if family=='shuffle_cnn' else 'transformer',*args)
                actual=overnight.config_for(family,*args)
                self.assertEqual(actual,expected)
    def test_each_family_exact_original_training_sources(self):
        record=overnight.sources()
        expected={
            'cnn_joint':cnn.sources(),
            'cnn_time':temporal.sources(),
            'transformer_joint':transformer.sources(),
            'transformer_time':expanded.training_sources('time_only',expanded.sources()),
            'shuffle_cnn':shuffle.training_sources('cnn',shuffle.sources()),
            'shuffle_transformer':shuffle.training_sources('transformer',shuffle.sources())}
        for family,value in expected.items():
            self.assertEqual(overnight.training_sources(family,record),value)
    def test_deadline_is_strict_between_fit_gate_not_fit_configuration(self):
        deadline=datetime(2026,10,15,7,tzinfo=timezone.utc)
        self.assertTrue(overnight.can_start(deadline,now=deadline-timedelta(seconds=1)))
        self.assertFalse(overnight.can_start(deadline,now=deadline))
        self.assertFalse(overnight.can_start(deadline,now=deadline+timedelta(seconds=1)))
        before=[config_hash(job['config']) for job in overnight.build_jobs()]
        for seed_ceiling in (7,8,9,10):
            selected=[job for job in overnight.build_jobs() if job['config']['seed']<42+seed_ceiling]
            self.assertTrue(all(job['config']['max_updates']==10000 for job in selected))
        self.assertEqual(before,[config_hash(job['config']) for job in overnight.build_jobs()])
    def test_verified_reuse_refuses_corrupt_checkpoint_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            job=overnight.job_for('cnn_joint','achilles',0,42,'soft_v1_stable')
            sources=overnight.sources();inputs=dict(parent_id='frozen_parent')
            folder=root/'stage01_fit'/overnight.cell_key(job['config']);folder.mkdir(parents=True)
            checkpoint=folder/'checkpoint.pt';checkpoint.write_bytes(b'qualification frozen checkpoint')
            signature=config_hash(dict(config=job['config'],parent_id=inputs['parent_id'],
                sources=overnight.training_sources(job['family'],sources)))
            manifest=folder/'manifest.json'
            manifest.write_text(json.dumps(dict(signature=signature,checkpoint_sha256=sha256(checkpoint))))
            before=manifest.read_bytes()
            with patch.object(overnight,'artifact_root',return_value=root):
                self.assertIsNotNone(overnight.verify_fit(job,inputs,sources,root))
                checkpoint.write_bytes(b'artificially corrupted fixture')
                with self.assertRaises(RuntimeError):overnight.verify_fit(job,inputs,sources,root)
            self.assertEqual(manifest.read_bytes(),before)

    def run_main_fixture(self,cached=False):
        """Exercise real transaction ordering with stubbed I/O, never an optimizer."""
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'POTENTIAL_PLAN.json').write_text('{}')
            jobs=[overnight.job_for('cnn_joint','achilles',0,seed,'soft_v1_stable') for seed in (49,50)]
            entries=[dict(job,input_key='toy') for job in jobs]
            plan=dict(jobs=entries,inputs=dict(toy=dict(parent_id='toy',data_folder=str(root),parent=dict(train_scales={}))))
            expired=[cached];events=[]
            def fit(*args):
                events.append('fit');expired[0]=True
                self.assertEqual(args[3]['max_updates'],10000)
                return dict(checkpoint_sha256='toy'),True
            def downstream(*args):
                self.assertTrue(expired[0]);events.append('downstream')
                return root/'embedding'
            def complete(output,state,job,*args):
                state['records'][job['id']]=dict(status='complete',family=job['family'],seed=job['config']['seed'])
                state['completed']=len(state['records'])
            with patch.object(overnight,'sources',return_value={'cnn_joint':{}}),patch.object(overnight,'seal_plan',return_value=plan),patch.object(overnight,'verify_gate'),patch.object(overnight,'prerequisite_status',return_value=(True,None)),patch.object(overnight,'build_jobs',return_value=[]),patch.object(overnight,'execution_jobs',return_value=jobs),patch.object(overnight,'artifact_root',return_value=root),patch.object(overnight,'verify_fit',return_value=dict(checkpoint_sha256='toy') if cached else None),patch.object(overnight,'can_start',side_effect=lambda *a,**k:not expired[0]),patch.object(overnight,'load_rows',return_value=None),patch.object(overnight,'finish_downstream',side_effect=downstream),patch.object(overnight,'record_complete',side_effect=complete),patch.object(cnn,'checked_output',return_value=root),patch.object(cnn,'fit_cell',side_effect=fit) as fitting,patch('sys.argv',['qualification','--execute','--device','cpu','--max-seeds','10','--deadline-utc','2026-10-15T07:00:00+00:00']):
                overnight.main()
            state=json.loads((root/'CAMPAIGN_STATE.json').read_text())
            return events,fitting.call_count,state
    def test_inflight_fit_flushes_downstream_then_deadline_blocks_next(self):
        events,calls,state=self.run_main_fixture()
        self.assertEqual(events,['fit','downstream'])
        self.assertEqual(calls,1)
        self.assertEqual(state['status'],'deadline')
        self.assertEqual(state['completed'],1)
    def test_completed_fit_reuse_remains_allowed_past_deadline(self):
        events,calls,state=self.run_main_fixture(cached=True)
        self.assertEqual(events,['downstream','downstream'])
        self.assertEqual(calls,0)
        self.assertEqual(state['status'],'complete')


if __name__=='__main__':unittest.main()
