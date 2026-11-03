"""Preference model and pairwise preference indices.

A :class:`Model` holds everything PROMETHEE needs besides the performances: criterion
polarities, weights, thresholds and preference-function types. Keeping the model separate
from the decision matrix makes it possible to compare *any* two sets of alternatives,
which the insertion experiments require.

Notation follows the manuscript: ``P_j(a, b)`` unicriterion preference degree, ``pi(a, b)``
global preference index, ``P(a, b) = pi(a, b) - pi(b, a)`` antisymmetric part.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .preference import CODES, preference_degree


@dataclass(frozen=True)
class Model:
    maximize: np.ndarray          # (n,) bool
    w: np.ndarray                 # (n,) weights, normalised to sum 1
    q: np.ndarray                 # (n,) indifference thresholds
    p: np.ndarray                 # (n,) preference thresholds
    types: tuple = ()             # (n,) preference-function codes; default 'linear'
    s: np.ndarray | None = None   # (n,) Gaussian inflexion points (optional)

    def __post_init__(self):
        n = len(self.w)
        object.__setattr__(self, "maximize", np.asarray(self.maximize, bool))
        w = np.asarray(self.w, float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        object.__setattr__(self, "w", w / w.sum())
        object.__setattr__(self, "q", np.asarray(self.q, float))
        object.__setattr__(self, "p", np.asarray(self.p, float))
        types = tuple(self.types) if len(self.types) else ("linear",) * n
        bad = [t for t in types if t not in CODES]
        if bad:
            raise ValueError(f"unknown preference function(s) {bad}")
        object.__setattr__(self, "types", types)

    @property
    def n(self) -> int:
        return len(self.w)

    def with_(self, **kw) -> "Model":
        return replace(self, **kw)


@dataclass
class Dataset:
    name: str
    alternatives: list
    criteria: list
    X: np.ndarray                 # (m, n) raw performances
    model: Model
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.X.shape[0]

    @property
    def n(self) -> int:
        return self.X.shape[1]


def unicriterion(model: Model, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
    """Array (n, |A|, |B|) of unicriterion preference degrees P_j(a, b), a in A, b in B."""
    XA = np.atleast_2d(np.asarray(XA, float))
    XB = np.atleast_2d(np.asarray(XB, float))
    out = np.empty((model.n, XA.shape[0], XB.shape[0]))
    for j in range(model.n):
        sign = 1.0 if model.maximize[j] else -1.0
        d = sign * (XA[:, j][:, None] - XB[:, j][None, :])
        s = None if model.s is None else float(model.s[j])
        out[j] = preference_degree(d, model.types[j], float(model.q[j]), float(model.p[j]), s)
    return out


def preference_index(model: Model, XA: np.ndarray, XB: np.ndarray | None = None,
                     Pj: np.ndarray | None = None) -> np.ndarray:
    """Global preference index pi(a, b) = sum_j w_j P_j(a, b) for a in A, b in B (default B = A)."""
    if Pj is None:
        Pj = unicriterion(model, XA, XA if XB is None else XB)
    return np.tensordot(model.w, Pj, axes=1)
