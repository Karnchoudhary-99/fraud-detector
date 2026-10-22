"""Save/load for trained supervised models.

A trained model is only useful if it survives the process that made
it.  Persistence is a plain pickle of the ``TrainedModel`` (engineer +
estimator, both sklearn/numpy objects), with an explicit version
marker so a file written by an older layout fails loudly instead of
loading silently wrong.  Pickle files are trusted input by design —
only ever load models you trained yourself or received from a source
you trust, never an arbitrary file from the internet.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from fraud_detector.training import TrainedModel

FORMAT_VERSION = 1


def save_model(model: TrainedModel, path: str | Path) -> Path:
    """Serialise ``model`` to ``path`` (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": FORMAT_VERSION, "model": model}
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: str | Path) -> TrainedModel:
    """Load a model written by :func:`save_model`."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: not a fraud-detector model file (version mismatch)")
    model = payload["model"]
    if not isinstance(model, TrainedModel):
        raise ValueError(f"{path}: payload is not a TrainedModel")
    return model
