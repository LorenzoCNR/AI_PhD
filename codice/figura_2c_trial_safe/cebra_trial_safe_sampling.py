"""CEBRA adapter with a trial-safe pool for time-delta donor pairs.

The reference prior, final positive search, neural windows, model, and loss are
inherited unchanged. Only the pool of position displacements is rebuilt from
10-bin pairs whose endpoints belong to the same trial.
"""
from __future__ import annotations

import numpy as np
import torch
import cebra
from sklearn.utils.validation import check_is_fitted
from cebra.integrations.sklearn import utils as sklearn_utils

from cebra_trial_safe import TrialSafeCEBRA, TrialSafeDataset


class TrialSafeDeltaCEBRA(TrialSafeCEBRA):
    """Use only within-trial donor pairs for CEBRA's ``time_delta`` sampler."""

    @classmethod
    def load(cls, filename, backend="auto", weights_only=False, **kwargs):
        # CEBRA 0.5's solver exposes model; newer releases also expose get_model.
        # This checkpoint was created by this local campaign.
        model = cebra.CEBRA.load(filename, backend=backend,
                                 weights_only=weights_only, **kwargs)
        if isinstance(model.n_features_, (list, tuple)):
            raise ValueError("only single-session checkpoints are supported")
        solver_model = getattr(model.solver_, "model", None)
        if solver_model is None and hasattr(model.solver_, "_get_model"):
            solver_model = model.solver_._get_model(session_id=None)
        if solver_model is None:
            raise ValueError("unsupported CEBRA solver: no encoder model is exposed")
        offset = solver_model.get_offset()
        if (offset.left, offset.right) != (5, 5):
            raise ValueError("checkpoint model must have offset (5, 5)")
        model.__class__ = cls
        return model

    def transform(self, X, *, trial_ids, batch_size=None, session_id=None):
        check_is_fitted(self, "n_features_")
        if session_id is not None:
            raise ValueError("only single-session input is supported")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer")
        X = sklearn_utils.check_input_array(X, min_samples=2)
        if X.shape[1] != self.n_features_:
            raise ValueError("input feature count differs from training")
        dataset = TrialSafeDataset(X, (), device=self.device_)
        dataset.set_trials(trial_ids)
        solver = self.solver_
        model = getattr(solver, "model", None)
        if model is None and hasattr(solver, "_get_model"):
            model = solver._get_model(session_id=None)
        if model is None:
            raise ValueError("unsupported CEBRA solver: no encoder model is exposed")
        model.eval()
        chunks = []
        with torch.no_grad():
            step = batch_size or len(X)
            for start in range(0, len(X), step):
                indices = torch.arange(start, min(start + step, len(X)), device=self.device_)
                windows = dataset[indices]
                chunks.append(model(windows).squeeze(-1).cpu())
        return torch.cat(chunks, dim=0).numpy()

    def _prepare_loader(self, dataset, max_iterations, is_multisession):
        loader = super()._prepare_loader(dataset, max_iterations, is_multisession)
        distribution = getattr(loader, "distribution", None)

        # The direction-only DiscreteDataLoader has no time_difference pool.
        # Time-only CEBRA uses TimeContrastive and also has no donor pairs.
        if not hasattr(distribution, "time_difference"):
            return loader

        delta = int(distribution.time_delta)
        if delta < 1:
            raise ValueError(f"time_delta must be positive, got {delta}")
        trial_ids = np.asarray(self._trial_ids_in_fit)
        if len(trial_ids) != len(dataset):
            raise ValueError("trial IDs and CEBRA training dataset have different lengths")

        # Endpoints are indexed in the same concatenated array CEBRA receives.
        # A donor pair (i-delta, i) is valid only when both endpoints are in the
        # same contiguous trial. This also excludes the stock sampler's first
        # `delta` zero placeholders, which do not correspond to donor pairs.
        endpoints = np.arange(delta, len(trial_ids), dtype=np.int64)
        valid = trial_ids[endpoints] == trial_ids[endpoints - delta]
        endpoints = endpoints[valid]
        if not len(endpoints):
            raise ValueError("no within-trial donor pairs are available")

        continuous = dataset.continuous_index
        if continuous is None or continuous.ndim != 2:
            raise ValueError("time_delta sampling requires a 2D continuous index")
        end_idx = torch.as_tensor(endpoints, dtype=torch.long, device=continuous.device)
        start_idx = end_idx - delta
        distribution.time_difference = (
            continuous.index_select(0, end_idx)
            - continuous.index_select(0, start_idx)
        ).contiguous()
        distribution.trial_safe_donor_endpoints = end_idx.detach().cpu()
        distribution.trial_safe_donor_count = int(len(endpoints))
        distribution.trial_safe_delta = delta
        return loader
