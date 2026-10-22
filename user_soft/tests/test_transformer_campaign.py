"""Expanded Transformer plan/reuse qualification; no GPU or scientific evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from user_soft import transformer_campaign as campaign
from user_soft import transformer_runner as initial
from user_soft import time_only_runner as temporal
from user_soft import rat_runner as runner
from user_soft.data import config_hash,sha256


class ExpandedTransformerQualification(unittest.TestCase):
    def test_exact_336_design_and_full_frozen_architecture(self):
        cells=campaign.enumerate_configs()
        self.assertEqual(len(cells),336)
        keys={(condition,c['rat'],c['fold'],c['seed'],c['version']) for condition,c in cells}
        self.assertEqual(len(keys),336)
        self.assertEqual({c['seed'] for _,c in cells},set(range(42,49)))
        self.assertEqual({c['fold'] for _,c in cells},{0,1,2})
        for condition,config in cells:
            self.assertEqual(config['transformer_settings'],initial.ARCHITECTURE)
            self.assertEqual(config['output_dimension'],32)
            self.assertEqual((config['offset_left'],config['offset_right']),(5,5))
            self.assertEqual(config['metadata_specs'],temporal.SPECS if condition=='time_only' else runner.SPECS)
    def test_first_eight_joint_configs_and_sources_exactly_preserved(self):
        sources=campaign.sources()
        for original in initial.initial_configs():
            extended=campaign.config_for(original['rat'],0,42,original['version'],'joint')
            self.assertEqual(extended,original)
            self.assertEqual(campaign.training_sources('joint',sources),initial.sources())
    def test_completed_and_partial_initial_signatures_reusable(self):
        sources=campaign.sources()
        count=0
        for original in initial.initial_configs():
            root=campaign.artifact_root(campaign.DEFAULT_OUTPUT,'joint',original)
            if root!=initial.DEFAULT_OUTPUT:continue
            folder=root/'stage01_fit'/original['rat']/'fold_0'/original['version']/'seed_42'
            manifest=folder/'manifest.json'
            record=json.loads(manifest.read_text()) if manifest.exists() else torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            parent=initial.CNN_OUTPUT/'stage00_data'/original['rat']/'fold_0'/'manifest.json'
            expected=config_hash(dict(config=original,parent_id=sha256(parent),sources=campaign.training_sources('joint',sources)))
            self.assertEqual(record['signature'],expected)
            self.assertEqual(record['config'],original)
            count+=1
        self.assertGreater(count,0)
    def test_time_only_training_signature_distinct_and_bindings_restore(self):
        sources=campaign.sources()
        self.assertNotEqual(campaign.training_sources('joint',sources),campaign.training_sources('time_only',sources))
        before=runner.SPECS
        with initial.architecture_bindings():
            with temporal.context_bindings():
                self.assertEqual(runner.SPECS,temporal.SPECS)
                self.assertIs(runner.make_encoder,initial.make_encoder)
        self.assertIs(runner.SPECS,before)
    def test_original_artifacts_route_exactly_and_new_cells_stay_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'initial'
            output=Path(tmp)/'expanded'
            with patch.object(initial,'DEFAULT_OUTPUT',parent):
                config=campaign.config_for('achilles',0,42,'soft_v1_stable','joint')
                folder=parent/'stage01_fit'/'achilles'/'fold_0'/'soft_v1_stable'/'seed_42'
                folder.mkdir(parents=True)
                (folder/'checkpoint.pt').write_bytes(b'qualification existence only')
                self.assertEqual(campaign.artifact_root(output,'joint',config),parent)
                self.assertEqual(campaign.artifact_root(output,'time_only',config),output/'time_only')
                newer=dict(config,seed=43)
                self.assertEqual(campaign.artifact_root(output,'joint',newer),output/'joint')
    def test_own_and_initial_mutex_prevents_racing_reused_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'expanded';output.mkdir()
            parent=Path(tmp)/'initial';parent.mkdir()
            (parent/'RUNNING.lock').write_text('existing-owner')
            sources=campaign.sources()
            gate=output/'qualification'/'GATE.json';gate.parent.mkdir()
            gate.write_text(json.dumps(dict(decision='GO',training_source_signature=config_hash(sources))))
            initial_gate=parent/'qualification'/'GATE.json';initial_gate.parent.mkdir()
            initial_gate.write_text(json.dumps(dict(decision='GO',training_source_signature=config_hash(initial.sources()))))
            with patch.object(initial,'DEFAULT_OUTPUT',parent),patch.object(runner,'checked_output',return_value=output),patch.object(campaign,'prepare',return_value={}),patch.object(campaign,'sources',return_value=sources),patch.object(initial,'sources',return_value=sources['joint_training_sources']),patch('sys.argv',['qualification','--execute','--device','cpu']),patch.object(runner,'fit_cell',side_effect=AssertionError('A fit must not execute')):
                with self.assertRaises(FileExistsError):campaign.main()
            self.assertFalse((output/'RUNNING.lock').exists())
            self.assertEqual((parent/'RUNNING.lock').read_text(),'existing-owner')


if __name__=='__main__':unittest.main()
