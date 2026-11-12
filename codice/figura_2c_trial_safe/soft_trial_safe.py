"""In-memory trial-safe Soft training; inputs are independent, ordered trials.

Each trial is a mapping with neural (T, neurons) and the labels required by
the chosen condition. Trials shorter than ten bins are rejected.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from user_soft.loss import evaluate_soft, fit_distance_scales
from user_soft import rat_runner, transformer_runner


CONDITIONS = {"Position": ("position",), "Direction": ("direction",),
              "Position+Direction": ("position", "direction")}
OFFSETS = torch.arange(-5, 5)


def _specs(condition):
    if condition not in CONDITIONS:
        raise ValueError("condition must be Position, Direction or Position+Direction")
    keys = CONDITIONS[condition]
    return [dict(key=key, geometry="euclidean" if key == "position" else "categorical",
                 weight=.5 if len(keys) == 2 else 1.) for key in keys]


def prepare_trials(trials, condition="Position+Direction"):
    """Validate trials and return windows and center metadata on CPU.

    Windows are (total_bins, 10, neurons); row order is trial then bin.
    ``trial_index`` and ``bin_index`` map every row back to its source.
    """
    keys = CONDITIONS.get(condition)
    if keys is None:
        raise ValueError("condition must be Position, Direction or Position+Direction")
    if not isinstance(trials, (list, tuple)) or not trials:
        raise ValueError("trials must be a nonempty list or tuple")
    windows, positions, directions, trial_indices, bin_indices = [], [], [], [], []
    n_features = position_shape = None
    for trial_index, trial in enumerate(trials):
        if not isinstance(trial, dict) or not ({"neural"} | set(keys)) <= trial.keys():
            raise ValueError("each trial needs neural and the labels required by condition")
        try:
            neural = torch.as_tensor(trial["neural"], dtype=torch.float32).detach().cpu()
            position = (torch.as_tensor(trial["position"], dtype=torch.float32).detach().cpu()
                        if "position" in keys else None)
            direction = (torch.as_tensor(trial["direction"]).detach().cpu()
                         if "direction" in keys else None)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("trial arrays must be numeric") from exc
        if neural.ndim != 2 or neural.shape[0] == 0 or neural.shape[1] == 0:
            raise ValueError("neural must have nonempty shape (bins, neurons)")
        n = len(neural)
        if n < 10:
            raise ValueError(f"trial {trial_index} has {n} bins; at least 10 are required")
        if position is not None and (position.ndim not in (1, 2) or len(position) != n or (position.ndim == 2 and position.shape[1] == 0)):
            raise ValueError("position must have shape (bins,) or (bins, features)")
        if direction is not None and (direction.ndim != 1 or len(direction) != n or direction.is_complex()):
            raise ValueError("direction must have shape (bins,) with categorical integer labels")
        if not all(torch.isfinite(value).all() for value in (neural, position, direction) if value is not None):
            raise ValueError("trial arrays must be finite")
        if direction is not None and direction.is_floating_point() and not torch.equal(direction, direction.round()):
            raise ValueError("direction labels must be integers")
        current_shape = position.shape[1:] if position is not None else None
        if n_features is not None and (neural.shape[1] != n_features or current_shape != position_shape):
            raise ValueError("feature dimensions must agree across trials")
        n_features, position_shape = neural.shape[1], current_shape
        support = torch.arange(n).clamp(5, n - 5)[:, None] + OFFSETS
        windows.append(neural[support])
        if position is not None:
            positions.append(position)
        if direction is not None:
            directions.append(direction.to(torch.int64))
        trial_indices.append(torch.full((n,), trial_index, dtype=torch.long))
        bin_indices.append(torch.arange(n))
    result = dict(windows=torch.cat(windows), trial_index=torch.cat(trial_indices),
                  bin_index=torch.cat(bin_indices))
    if positions:
        result["position"] = torch.cat(positions)
    if directions:
        result["direction"] = torch.cat(directions)
    return result


def make_model(architecture, n_features, embedding_dim=3, device="cpu"):
    """Create the existing user_soft CNN or Transformer architecture."""
    if architecture not in ("cnn", "transformer"):
        raise ValueError("architecture must be 'cnn' or 'transformer'")
    factory = rat_runner if architecture == "cnn" else transformer_runner
    config = factory.frozen_config("trial_safe", 0, 42, "soft_v1_stable")
    config["output_dimension"] = embedding_dim
    return factory.make_encoder(n_features, config, torch.device(device))


def _encode(model, windows):
    z = model(windows.transpose(1, 2))
    if z.ndim != 2 or len(z) != len(windows) or not torch.isfinite(z).all():
        raise ValueError("encoder must return one finite embedding per window")
    return z


def _data_hash(data, metadata):
    """Hash exact prepared inputs, trial mapping, true labels and training targets."""
    digest = hashlib.sha256()
    for group, values in (("data", data), ("targets", metadata)):
        for key in sorted(values):
            value = values[key].detach().cpu().contiguous()
            digest.update(repr((group, key, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _read_checkpoint(path):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("format") != "soft_trial_safe_v1":
        raise ValueError("unsupported Soft checkpoint")
    return record


def _save_checkpoint(path, record):
    """Replace only after a complete checkpoint has been flushed to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            torch.save(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def load_model(path, n_features, architecture, device="cpu"):
    """Load the checkpoint encoder with its saved embedding dimension."""
    record = _read_checkpoint(path)
    config = record["config"]
    if config["architecture"] != architecture or config["n_features"] != n_features:
        raise ValueError("checkpoint architecture or n_features mismatch")
    model = make_model(architecture, n_features, config["embedding_dim"], device)
    model.load_state_dict(record["model"])
    return model


def fit(trials, architecture="cnn", *, condition="Position+Direction", targets=None,
        embedding_dim=3, steps=1000, batch_size=512, learning_rate=3e-4, seed=42, device="cpu",
        checkpoint_path=None, checkpoint_signature=None, checkpoint_every=500,
        progress_callback=None):
    """Fit on the supplied training trials and return model, scales and losses.

    Sampling is uniform over original bins, with replacement. Scale fitting
    uses every supplied training bin; no validation or test data is accessed.
    Optional targets is a list of per-trial mappings of training labels;
    it can contain shuffled labels and does not modify the true trial labels.
    Existing checkpoints resume automatically; steps is the total target step.
    Saves occur after each checkpoint_every completed updates and on successful
    completion, before callback(step, loss). Interrupted updates are not saved.
    """
    for name, value, minimum in (("steps", steps, 1), ("batch_size", batch_size, 2),
                                 ("embedding_dim", embedding_dim, 1),
                                 ("checkpoint_every", checkpoint_every, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if checkpoint_signature is not None and not isinstance(checkpoint_signature, str):
        raise ValueError("checkpoint_signature must be a string or None")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable")
    specs = _specs(condition)
    data = prepare_trials(trials, condition)
    if len(data["windows"]) < 2:
        raise ValueError("training needs at least two bins")
    metadata = {key: data[key] for key in CONDITIONS[condition]}
    if targets is not None:
        if not isinstance(targets, (list, tuple)) or len(targets) != len(trials):
            raise ValueError("targets must have one label mapping per training trial")
        target_trials = []
        for trial, target in zip(trials, targets):
            if not isinstance(target, dict) or not set(CONDITIONS[condition]) <= target.keys():
                raise ValueError("targets must contain labels required by condition")
            target_trials.append({"neural": trial["neural"], **{key: target[key] for key in CONDITIONS[condition]}})
        target_data = prepare_trials(target_trials, condition)
        metadata = {key: target_data[key] for key in CONDITIONS[condition]}
    scales = fit_distance_scales(metadata, specs, split="train", source_ids=range(len(data["windows"])))
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if architecture not in ("cnn", "transformer"):
        raise ValueError("architecture must be 'cnn' or 'transformer'")
    factory = rat_runner if architecture == "cnn" else transformer_runner
    model_config = factory.frozen_config("trial_safe", 0, 42, "soft_v1_stable")
    model_settings = (dict(hidden_units=model_config["hidden_units"]) if architecture == "cnn"
                      else dict(model_config["transformer_settings"]))
    config = dict(architecture=architecture, model_settings=model_settings,
                  loss_version="soft_v1_stable", temperature=1., metadata_temperature=.5,
                  n_features=data["windows"].shape[-1],
                  condition=condition, embedding_dim=embedding_dim, batch_size=batch_size,
                  learning_rate=float(learning_rate), seed=seed, device=str(device),
                  cuda_device_count=torch.cuda.device_count() if device.type == "cuda" else 0,
                  torch_version=str(torch.__version__), specs=specs,
                  data_sha256=_data_hash(data, metadata),
                  targets_supplied=targets is not None, checkpoint_signature=checkpoint_signature)
    record = _read_checkpoint(checkpoint_path) if checkpoint_path is not None and Path(checkpoint_path).exists() else None
    if record is not None:
        if record["config"] != config or record["scales"] != scales:
            raise ValueError("checkpoint configuration, signature or training data mismatch")
        if record["step"] > steps:
            raise ValueError("checkpoint step exceeds requested total steps")
        if len(record["losses"]) != record["step"]:
            raise ValueError("invalid checkpoint loss history")
    torch.manual_seed(seed)
    model = make_model(architecture, data["windows"].shape[-1], embedding_dim, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    start = 0
    if record is not None:
        model.load_state_dict(record["model"])
        optimizer.load_state_dict(record["optimizer"])
        generator.set_state(record["batch_generator_rng"])
        torch.set_rng_state(record["torch_cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(record["torch_cuda_rng"])
        losses = list(record["losses"])
        start = record["step"]
    model.train()
    for step in range(start + 1, steps + 1):
        indices = torch.randint(len(data["windows"]), (batch_size,), generator=generator)
        z = _encode(model, data["windows"][indices].to(device))
        batch = {key: value[indices].to(device) for key, value in metadata.items()}
        result = evaluate_soft(z, batch, specs, version="soft_v1_stable", scales=scales,
                               temperature=1., metadata_temperature=.5)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        optimizer.step()
        losses.append(float(result.loss.detach().cpu()))
        if checkpoint_path is not None and (step % checkpoint_every == 0 or step == steps):
            _save_checkpoint(checkpoint_path, dict(
                format="soft_trial_safe_v1", config=config, scales=scales, step=step,
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                batch_generator_rng=generator.get_state(), torch_cpu_rng=torch.get_rng_state(),
                torch_cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                losses=losses))
        if progress_callback is not None:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            try:
                progress_callback(step, losses[-1])
            finally:
                torch.set_rng_state(cpu_rng)
                if device.type == "cuda":
                    torch.cuda.set_rng_state_all(cuda_rng)
    return dict(model=model, scales=scales, losses=losses, architecture=architecture,
                condition=condition, embedding_dim=embedding_dim, shuffled_targets_supplied=targets is not None)


def transform(model, trials, *, condition="Position+Direction", batch_size=512):
    """Return one embedding per original bin and its center metadata."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data = prepare_trials(trials, condition)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            z = torch.cat([_encode(model, chunk.to(device)).cpu()
                           for chunk in data["windows"].split(batch_size)])
    finally:
        model.train(was_training)
    return {"embedding": z, **{k: data[k] for k in (*CONDITIONS[condition], "trial_index", "bin_index")}}
