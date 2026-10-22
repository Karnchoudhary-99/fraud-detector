"""Plain-language explanations for flagged transactions.

A score alone does not help an analyst: they need to know *why* a row
was flagged.  The matrices this module works on are already
standardised (see :class:`fraud_detector.features.FeatureEngineer`),
so a feature value is a deviation from the typical transaction in
standard deviations — large absolute values are exactly what makes a
row anomalous.  ``top_contributors`` therefore ranks a row's features
by absolute standardised value, and ``explain_flags`` applies that to
the highest-scored rows.  The explanations are model-agnostic: they
describe the row, not the detector's internals, so they stay valid for
both the z-score and the isolation-forest detector.

A contribution's ``direction`` is deliberately boring — "above typical"
or "below typical" — because a 3-SD deviation is the fact; whether it
is suspicious is the analyst's call.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Contribution:
    """One feature's share of a row's anomalousness."""

    feature: str
    value: float  # standardised value (SDs from typical)
    direction: str  # "above typical" or "below typical"

    @property
    def magnitude(self) -> float:
        return abs(self.value)


@dataclass(frozen=True)
class FlagExplanation:
    """Why one flagged row stands out, in ranked feature order."""

    row_index: int
    score: float
    contributors: tuple[Contribution, ...]


def top_contributors(
    row: np.ndarray, names: list[str], top_n: int = 3
) -> tuple[Contribution, ...]:
    """Return the ``top_n`` most deviant features of one standardised row."""
    values = np.asarray(row, dtype=float)
    if values.ndim != 1:
        raise ValueError("row must be a 1D array")
    if len(values) != len(names):
        raise ValueError(
            f"row has {len(values)} values but {len(names)} names were given"
        )
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    order = np.argsort(np.abs(values))[::-1][:top_n]
    return tuple(
        Contribution(
            feature=names[int(i)],
            value=float(values[int(i)]),
            direction="above typical" if values[int(i)] >= 0 else "below typical",
        )
        for i in order
    )


def explain_flags(
    X: np.ndarray,
    scores: np.ndarray,
    names: list[str],
    top_n_rows: int = 10,
    top_n_features: int = 3,
) -> list[FlagExplanation]:
    """Explain the ``top_n_rows`` highest-scored rows of ``X``."""
    matrix = np.asarray(X, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or len(matrix) != len(scores):
        raise ValueError("X rows and scores must have equal length")
    order = np.argsort(scores)[::-1][:top_n_rows]
    return [
        FlagExplanation(
            row_index=int(i),
            score=float(scores[int(i)]),
            contributors=top_contributors(matrix[int(i)], names, top_n_features),
        )
        for i in order
    ]


def format_explanation(explanation: FlagExplanation) -> str:
    """Render one :class:`FlagExplanation` as a single readable line."""
    parts = ", ".join(
        f"{c.feature} {c.value:+.2f} SD {c.direction}" for c in explanation.contributors
    )
    return f"row {explanation.row_index} (score {explanation.score:.3f}): {parts}"
