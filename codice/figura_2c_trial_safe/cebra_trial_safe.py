"""CEBRA 0.6.1 single-session adapter with trial-clamped neural windows.

Trial IDs identify contiguous runs, one ID per original bin. Sampling and loss
are the upstream CEBRA implementation; only neural window extraction changes.
In particular, CEBRA's temporal positive-pair sampler is not trial constrained.
"""
import numpy as np
import torch
import cebra
from cebra.integrations.sklearn.dataset import SklearnDataset
from cebra.integrations.sklearn import utils as sklearn_utils
from sklearn.utils.validation import check_is_fitted


def trial_bounds(trial_ids, n):
    """Return inclusive per-bin run bounds; reject repeated non-contiguous IDs."""
    ids = np.asarray(trial_ids)
    if ids.ndim != 1 or len(ids) != n or n == 0:
        raise ValueError('trial_ids must be a nonempty 1D array with one ID per bin')
    if ids.dtype.kind not in 'iuUS':
        raise ValueError('trial_ids must contain integer or string IDs')
    changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes - 1, n - 1]
    if len(np.unique(ids[starts])) != len(starts):
        raise ValueError('each trial ID must identify exactly one contiguous run')
    lengths = ends - starts + 1
    if np.any(lengths < 10):
        raise ValueError('each trial must contain at least 10 bins')
    return np.repeat(starts, lengths), np.repeat(ends, lengths)


class TrialSafeDataset(SklearnDataset):
    """CEBRA sklearn dataset preserving each sampled center and its labels."""
    def set_trials(self, trial_ids):
        starts, ends = trial_bounds(trial_ids, len(self))
        self.trial_starts = torch.as_tensor(starts, device=self.device)
        self.trial_ends = torch.as_tensor(ends, device=self.device)

    def expand_index(self, index):
        index = torch.as_tensor(index, dtype=torch.long, device=self.device).reshape(-1)
        if torch.any(index < 0) or torch.any(index >= len(self)):
            raise IndexError('center index outside data')
        center = torch.maximum(torch.minimum(index, self.trial_ends[index] - 4),
                               self.trial_starts[index] + 5)
        return center[:, None] + torch.arange(-5, 5, device=self.device)


class TrialSafeCEBRA(cebra.CEBRA):
    """CEBRA estimator: fit(X, *labels, trial_ids=...), transform(X, trial_ids=...).

    Single session, 10-bin receptive field only. IDs can differ between fit and
    transform. Fit retains upstream sampler, architecture and criterion.
    """
    @classmethod
    def load(cls, filename, backend='auto', weights_only=None, **kwargs):
        """Load a CEBRA checkpoint with trial-safe inference, without fitting.

        Loading does not establish how the encoder was originally trained.
        Provide the appropriate trial IDs to every subsequent transform call.
        """
        model = cebra.CEBRA.load(filename, backend=backend,
                                 weights_only=weights_only, **kwargs)
        if isinstance(model.n_features_, (list, tuple)):
            raise ValueError('only single-session checkpoints are supported')
        offset = model.solver_._get_model(session_id=None).get_offset()
        if (offset.left, offset.right) != (5, 5):
            raise ValueError('checkpoint model must have offset (5, 5)')
        model.__class__ = cls
        return model

    def fit(self, X, *y, trial_ids, adapt=False, callback=None, callback_frequency=None):
        if adapt:
            raise ValueError('adapt is not supported by this trial-safe adapter')
        trial_bounds(trial_ids, len(X))
        self._trial_ids_in_fit = np.asarray(trial_ids).copy()
        try:
            return super().fit(X, *y, callback=callback, callback_frequency=callback_frequency)
        finally:
            del self._trial_ids_in_fit

    def partial_fit(self, X, *y, callback=None, callback_frequency=None):
        if not hasattr(self, '_trial_ids_in_fit'):
            raise NotImplementedError('use fit with trial_ids; partial_fit is not supported')
        return super().partial_fit(X, *y, callback=callback, callback_frequency=callback_frequency)

    def _prepare_data(self, X, y):
        if (self.offset_.left, self.offset_.right) != (5, 5):
            raise ValueError('model architecture must have offset (5, 5), i.e. a 10-bin window')
        dataset, multi = super()._prepare_data(X, y)
        if multi:
            raise ValueError('only single-session input is supported')
        dataset.__class__ = TrialSafeDataset
        dataset.set_trials(self._trial_ids_in_fit)
        return dataset, False

    def transform(self, X, *, trial_ids, batch_size=None, session_id=None):
        check_is_fitted(self, 'n_features_')
        if session_id is not None:
            raise ValueError('only single-session input is supported')
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError('batch_size must be a positive integer')
        X = sklearn_utils.check_input_array(X, min_samples=2)
        if X.shape[1] != self.n_features_:
            raise ValueError('input feature count differs from training')
        dataset = TrialSafeDataset(X, (), device=self.device_)
        dataset.set_trials(trial_ids)
        model = self.solver_._get_model(session_id=None)
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size or len(X)):
                windows = dataset[torch.arange(start, min(start + (batch_size or len(X)), len(X)), device=self.device_)]
                output = model(windows)
                chunks.append(output.squeeze(-1).cpu())
        return torch.cat(chunks, dim=0).numpy()


def load_trial_safe(path, **kwargs):
    """Load a checkpoint as TrialSafeCEBRA; this does not retrain the encoder."""
    return TrialSafeCEBRA.load(path, **kwargs)
