"""Export existing six-seed Soft checkpoints to isolated split embeddings; no training."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from user_soft import fig1e_ed4_6seed_runner as campaign
from user_soft import rat_runner, transformer_runner
from user_soft.data import assert_disjoint, eligible_centers, load_rows, sha256

ROOT = campaign.ROOT
OUT = ROOT / 'results' / 'rat_rebuild_20261027' / 'Figure_1E_ED4' / 'embeddings' / 'soft'


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_record = campaign.collect_source_record()
    done = skipped = 0
    cache = {}
    for config in campaign.iter_configs():
        if config['version'] != 'soft_v1_stable':
            continue
        cell = campaign.cell_id(config)
        original = campaign.cell_folder(config)
        data_path = campaign.condition_manifest_path(config)
        fit = campaign.verify_cell(original, config, data_path, source_record)
        checkpoint = original / 'checkpoint.pt'
        parent = campaign.OUTPUT_ROOT / 'stage00_data' / config['rat'] / f"fold_{config['fold']}"
        parent_manifest = json.loads((parent / 'manifest.json').read_text())
        key = (config['rat'], config['fold'])
        if key not in cache:
            rows = {}
            for split in ('train', 'valid', 'test'):
                source = parent / f'{split}.npz'
                if sha256(source) != parent_manifest['splits'][split]['sha256']:
                    raise RuntimeError(f'Input cache hash mismatch: {source}')
                rows[split] = load_rows(source)
            assert_disjoint(rows)
            cache = {key: rows}
        rows = cache[key]
        dest = OUT / cell
        dest.mkdir(parents=True, exist_ok=True)
        pending = []
        for split in ('train', 'valid', 'test'):
            path = dest / f'{split}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if (str(saved['checkpoint_sha256'].item()) != fit['checkpoint_sha256']
                            or str(saved['source_npz_sha256'].item()) != parent_manifest['splits'][split]['sha256']
                            or not np.array_equal(saved['source_id'], rows[split].source_id)
                            or saved['embedding'].shape != (len(rows[split].source_id), 3)
                            or not np.isfinite(saved['embedding']).all()):
                        raise RuntimeError(f'Existing staged embedding incompatible: {path}')
                skipped += 1
            else:
                pending.append(split)
        if pending:
            factory = rat_runner.make_encoder if config['campaign_architecture_key'] == 'cnn' else transformer_runner.make_encoder
            model = factory(rows['train'].neural.shape[1], config, device)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['signature'] != fit['signature'] or saved['step'] != 10000:
                raise RuntimeError(f'Checkpoint signature/update mismatch: {cell}')
            model.load_state_dict(saved['model']); model.eval()
            for split in pending:
                row = rows[split]
                values = rat_runner.tensors(row, device)
                parts = []
                with torch.inference_mode():
                    for start in range(0, len(row.neural), config['batch_size']):
                        indices = torch.arange(start, min(start + config['batch_size'], len(row.neural)))
                        z, _ = rat_runner.encode_batch(model, values, indices)
                        parts.append(z.cpu().numpy())
                embedding = np.concatenate(parts).astype(np.float32)
                if embedding.shape != (len(row.neural), 3) or not np.isfinite(embedding).all():
                    raise RuntimeError(f'Invalid embedding: {cell}/{split}')
                mask = np.zeros(len(row.neural), dtype=bool)
                mask[eligible_centers(row)] = True
                path = dest / f'{split}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, embedding=embedding, behavior=row.behavior,
                        source_id=row.source_id, trial_id=row.trial_id, time_id=row.time_id,
                        session_id=row.session_id, valid_mask=mask, split=np.asarray(split),
                        rat=np.asarray(config['rat']), fold=np.asarray(config['fold']),
                        seed=np.asarray(config['seed']), condition=np.asarray(config['condition']),
                        architecture=np.asarray(config['campaign_architecture_key']),
                        loss_version=np.asarray(config['version']),
                        checkpoint_sha256=np.asarray(fit['checkpoint_sha256']),
                        source_npz_sha256=np.asarray(parent_manifest['splits'][split]['sha256']))
                done += 1
            del model, saved
        print(json.dumps({'cell': cell, 'created_splits': done, 'reused_splits': skipped, 'device': str(device)}), flush=True)
    print(json.dumps({'status': 'complete', 'created_splits': done, 'reused_splits': skipped, 'cells': 432, 'device': str(device)}), flush=True)


if __name__ == '__main__':
    main()
