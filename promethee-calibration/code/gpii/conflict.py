"""Exact conflict measures of a PROMETHEE II ranking, accelerated by the universal certificate.

For the antisymmetric matrix P of a problem A (m alternatives), D(a,b) = sum_s [P(a,s) - P(b,s)]
has the sign of the PROMETHEE II order of a and b, and deleting x changes it by
-delta_x(a,b) = -(P(a,x) + P(x,b)), with |delta_x| <= 2 (Theorem 4 of the theory note).
Hence a strictly ordered pair with |D(a,b)| > 2 can never be broken by one deletion, and
only the *candidate* pairs |D| <= 2 need to be examined: the cost is O(|C| m) instead of
O(m^3), with |C| typically a small fraction of the pairs.

Measures (all exact, all in [0, 1]):

``fragility``   share of strictly ordered pairs whose order is broken (reversed or tied) by
                deleting at least one other alternative -- the share of pairwise
                conclusions of the ranking that a single deletion can overturn;
``event_rate``  share of (pair, deleted alternative) events that break the order -- the
                leave-one-out rank-reversal rate (equals ``reversal.loo_reversal_rate``);
``impact``      share of alternatives whose deletion breaks at least one pair;
``cyclicity``   K = ||R_c||^2 / ||P||^2, the share of P not explained by any ranking
                potential (least-squares / Hodge decomposition).
"""
from __future__ import annotations

import numpy as np

from .cycles import cyclicity_index
from .reversal import TIE_TOL, tsign

MEASURES = ("fragility", "event_rate", "impact", "cyclicity")


def deletion_events(P: np.ndarray):
    """Candidate pairs and their flip indicators.

    Returns (a, b, flips, n_strict) where (a[i], b[i]) are the candidate strictly ordered
    pairs (|D| <= 2), flips[i, x] is True when deleting x breaks the order of that pair
    (x not in {a, b}), and n_strict is the number of strictly ordered pairs.
    """
    m = P.shape[0]
    r = P.sum(1)
    D = r[:, None] - r[None, :]
    iu = np.triu_indices(m, 1)
    d = D[iu]
    strict = np.abs(d) > TIE_TOL
    n_strict = int(strict.sum())
    cand = strict & (np.abs(d) <= 2.0 + TIE_TOL)
    a, b = iu[0][cand], iu[1][cand]
    if a.size == 0:
        return a, b, np.zeros((0, m), bool), n_strict
    Dn = d[cand][:, None] - (P[a, :] + P[:, b].T)          # D(a,b) - delta_x(a,b), all x
    flips = tsign(Dn) != np.sign(d[cand])[:, None]
    flips[np.arange(a.size), a] = False
    flips[np.arange(a.size), b] = False
    return a, b, flips, n_strict


def conflict_measures(P: np.ndarray, which=MEASURES) -> dict:
    m = P.shape[0]
    out = {}
    if {"fragility", "event_rate", "impact"} & set(which):
        a, b, flips, n_strict = deletion_events(P)
        if "fragility" in which:
            out["fragility"] = float(flips.any(1).sum() / n_strict) if n_strict else 0.0
        if "event_rate" in which:
            out["event_rate"] = float(flips.sum() / (n_strict * (m - 2))) if n_strict and m > 2 else 0.0
        if "impact" in which:
            out["impact"] = float(flips.any(0).sum() / m)
    if "cyclicity" in which:
        out["cyclicity"] = cyclicity_index(P)
    return out


def conflict(P: np.ndarray, measure: str = "fragility") -> float:
    return conflict_measures(P, (measure,))[measure]
