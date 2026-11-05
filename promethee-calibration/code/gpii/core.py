"""PROMETHEE I, II and gamma.

Two incomparability sources are implemented, each returning boolean matrices
``(P, I, R)`` with ``P[a, b]`` = "a strictly preferred to b":

``"PI"``       PROMETHEE I: a R b iff the positive and negative flows cross.
``"gamma"``    PROMETHEE gamma (Dejaegere & De Smet, 2023): valued coalitions of
               unicriterion net flows, indicator rule with (T_I, T_J, P_f).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .model import Model, preference_index, unicriterion

Source = Literal["PI", "gamma"]
SOURCES: tuple = ("PI", "gamma")
GAMMA_DEFAULT = dict(T_I=0.10, T_J=0.10, P_f=1.0)
TOL = 1e-12


# ---------------------------------------------------------------------------
# PROMETHEE II quantities for one problem (decision matrix + model)
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    X: np.ndarray
    model: Model
    Pj: np.ndarray        # (n, m, m)
    pi: np.ndarray        # (m, m)
    phi_plus: np.ndarray
    phi_minus: np.ndarray
    phi: np.ndarray
    phi_uni: np.ndarray   # (m, n) unicriterion net flows

    @property
    def m(self) -> int:
        return self.pi.shape[0]

    @property
    def P(self) -> np.ndarray:
        """Antisymmetric part P(a, b) = pi(a, b) - pi(b, a)."""
        return self.pi - self.pi.T

    @property
    def ci(self) -> np.ndarray:
        """Conflict intensity CI(a, b) = min(pi(a, b), pi(b, a)) in [0, 1/2]."""
        return np.minimum(self.pi, self.pi.T)

    def subproblem(self, idx) -> "Problem":
        """PROMETHEE II quantities recomputed on the subset ``idx`` (no re-evaluation of P_j)."""
        idx = np.asarray(idx)
        return _from_Pj(self.X[idx], self.model, self.Pj[:, idx][:, :, idx])


def _from_Pj(X, model, Pj) -> Problem:
    m = Pj.shape[1]
    if m < 2:
        raise ValueError("PROMETHEE needs at least two alternatives")
    pi = np.tensordot(model.w, Pj, axes=1)
    phi_plus = pi.sum(1) / (m - 1)
    phi_minus = pi.sum(0) / (m - 1)
    phi_uni = ((Pj.sum(2) - Pj.sum(1)) / (m - 1)).T
    return Problem(np.asarray(X), model, Pj, pi, phi_plus, phi_minus, phi_plus - phi_minus, phi_uni)


def promethee(X: np.ndarray, model: Model) -> Problem:
    """Compute P_j, pi, the PROMETHEE flows and the unicriterion net flows of X."""
    return _from_Pj(X, model, unicriterion(model, X, X))


# ---------------------------------------------------------------------------
# Incomparability sources
# ---------------------------------------------------------------------------
def _diag_false(*Ms):
    for M in Ms:
        np.fill_diagonal(M, False)
    return Ms


def relations_from_flows(phi_plus_a, phi_minus_a, phi_plus_b, phi_minus_b):
    """PROMETHEE I relations between the alternatives of two (broadcastable) flow arrays."""
    pp = phi_plus_a - phi_plus_b
    pm = phi_minus_a - phi_minus_b
    P = (pp >= -TOL) & (pm <= TOL) & ((pp > TOL) | (pm < -TOL))
    R = ((pp > TOL) & (pm > TOL)) | ((pp < -TOL) & (pm < -TOL))
    I = (np.abs(pp) <= TOL) & (np.abs(pm) <= TOL)
    return P, I, R


def relations_promethee_i(pr: Problem):
    P, I, R = relations_from_flows(pr.phi_plus[:, None], pr.phi_minus[:, None],
                                   pr.phi_plus[None, :], pr.phi_minus[None, :])
    return _diag_false(P, I, R)


def gamma_rule(g: np.ndarray, gt: np.ndarray, T_I=0.10, T_J=0.10, P_f=1.0):
    """Indicator rule of PROMETHEE gamma given gamma_ab (g) and gamma_ba (gt).

    The relation is the one with the largest indicator among indifference T_I - max,
    incomparability min - T_J and the two preference indicators (g - gt)/P_f; as in the
    reference implementation, indifference is also returned when no indicator is positive.
    """
    Ii = T_I - np.maximum(g, gt)
    Jj = np.minimum(g, gt) - T_J
    Pab = (g - gt) / P_f
    Pba = -Pab
    isI = (Ii >= np.maximum.reduce([Jj, Pab, Pba])) | ((Pab <= 0) & (Pba <= 0) & (Jj <= 0))
    isR = (~isI) & (Jj >= np.maximum(Pab, Pba))
    isP = (~isI) & (~isR) & (Pab > Pba)
    return isP, isI, isR


def gamma_indices(phi_uni: np.ndarray, w: np.ndarray) -> np.ndarray:
    """gamma_ab = sum_c w_c (phi_c(a) - phi_c(b))^+ (PROMETHEE gamma)."""
    d = phi_uni[:, None, :] - phi_uni[None, :, :]
    g = np.tensordot(np.where(d > 0, d, 0.0), w, axes=([2], [0]))
    np.fill_diagonal(g, 0.0)
    return g


def relations_gamma(pr: Problem, T_I=0.10, T_J=0.10, P_f=1.0):
    g = gamma_indices(pr.phi_uni, pr.model.w)
    return _diag_false(*gamma_rule(g, g.T, T_I, T_J, P_f))


def relations(pr: Problem, source: Source, **gamma_params):
    params = {**GAMMA_DEFAULT, **gamma_params}
    if source == "PI":
        return relations_promethee_i(pr)
    if source == "gamma":
        return relations_gamma(pr, **params)
    raise ValueError(f"unknown incomparability source {source!r}; expected one of {SOURCES}")


def rank_desc(x: np.ndarray) -> np.ndarray:
    """Ranks, 1 = best; exact ties broken by input order (stable)."""
    order = np.argsort(-np.asarray(x), kind="stable")
    r = np.empty(len(x), int)
    r[order] = np.arange(1, len(x) + 1)
    return r


def incomparability_count(pr: Problem, source: Source, **gamma_params) -> int:
    """Number of unordered pairs declared incomparable by the source."""
    _, _, R = relations(pr, source, **gamma_params)
    return int(np.triu(R, 1).sum())
