"""Rank-reversal-aware calibration of PROMETHEE II.

Given a decision matrix X, elicited parameters theta0 = (w0, p0) and the decision maker's
admissible information, find the parameters that trade off the conflict of the PROMETHEE II
ranking against the deviation from theta0:

    min_{theta in Theta}  ( C(theta) / C(theta0) ,  Delta(theta, theta0) ).

Admissible set Theta
--------------------
* weights, ``mode="order"``: the weak order of w0 is kept -- criteria with equal elicited
  weights keep equal weights, strictly ordered criteria keep their order. With levels
  L_1 > ... > L_k (decreasing elicited weight) this set is the image of the standard simplex
  under the affine bijection  lambda -> V lambda,  V[:, l] = 1_{L_1 u ... u L_l} / |L_1 u ... u L_l|
  (Proposition "feasible encoding"), so evolving lambda on the simplex keeps every individual
  feasible and Dirichlet(1) sampling of lambda is uniform on the admissible weights;
* weights, ``mode="free"``: the whole simplex (used when equal weights are a convention and
  carry no preference information, e.g. BLI and WHR);
* preference thresholds (scenario "WT"): p_j = p0_j R^{u_j}, u_j in [-1, 1], i.e. within a
  factor R of the elicited value (shapes and q fixed);
* full preference functions (scenario "F"): each criterion uses either the piecewise-linear
  family (Type V, which contains Type III for q = 0) with p_j = p0_j R^{u_j} and
  q_j = rho_j * QMAX * p_j, rho_j in [0, 1], or the Gaussian (Type VI) with inflexion point
  s_j = s0_j R^{u_j}; the family is a binary gene (t_j < 1/2: piecewise linear, >= 1/2: Gaussian).

Deviation: Delta_w = 1/2 ||w - w0||_1 (total variation, in [0, 1]); per criterion,
delta_j = min(1, |u_j| + |rho_j - rho0_j| [Type V] + 1[family changed]) and Delta_f = mean_j delta_j
(in WT, delta_j = |u_j|); Delta = Delta_w (W) or (Delta_w + Delta_f) / 2 (WT, F).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conflict import conflict
from .core import _from_Pj, relations_promethee_i
from .model import Model, unicriterion

CONFLICTS = ("fragility", "event_rate", "cyclicity", "incomparability")
SCENARIOS = ("W", "WT", "F")
QMAX = 0.5          # indifference threshold at most half the preference threshold
LINEAR_FAMILY = ("linear", "vshape")


# ---------------------------------------------------------------------------
# Admissible weights
# ---------------------------------------------------------------------------
def weak_order_levels(w0: np.ndarray, tol: float = 1e-9):
    """Levels of the weak order of w0, by decreasing weight (list of index arrays)."""
    order = np.argsort(-w0, kind="stable")
    levels, cur = [], [order[0]]
    for i in order[1:]:
        if abs(w0[i] - w0[cur[0]]) <= tol:
            cur.append(i)
        else:
            levels.append(np.array(cur))
            cur = [i]
    levels.append(np.array(cur))
    return levels


@dataclass
class WeightSpace:
    V: np.ndarray            # (n, k): w = V @ lam, lam in the standard simplex
    lam0: np.ndarray         # coordinates of w0
    mode: str
    levels: list

    @classmethod
    def build(cls, w0: np.ndarray, mode: str = "order") -> "WeightSpace":
        w0 = np.asarray(w0, float) / np.sum(w0)
        n = len(w0)
        if mode == "free":
            return cls(np.eye(n), w0.copy(), mode, [np.array([j]) for j in range(n)])
        levels = weak_order_levels(w0)
        k = len(levels)
        V = np.zeros((n, k))
        top = np.zeros(n, bool)
        for l, L in enumerate(levels):
            top[L] = True
            V[:, l] = top / top.sum()
        lam0, *_ = np.linalg.lstsq(V, w0, rcond=None)
        lam0 = np.clip(lam0, 0, None)
        lam0 /= lam0.sum()
        assert np.allclose(V @ lam0, w0, atol=1e-9)
        return cls(V, lam0, mode, levels)

    @property
    def k(self) -> int:
        return self.V.shape[1]

    def weights(self, lam: np.ndarray) -> np.ndarray:
        return self.V @ lam

    def centroid(self) -> np.ndarray:
        """Centroid of the admissible weights (the image of the simplex barycentre)."""
        return self.V @ np.full(self.k, 1.0 / self.k)

    def contains(self, w: np.ndarray, tol: float = 1e-9) -> bool:
        if abs(w.sum() - 1) > tol or np.any(w < -tol):
            return False
        if self.mode == "free":
            return True
        for L in self.levels:
            if np.ptp(w[L]) > tol:
                return False
        for L, M in zip(self.levels[:-1], self.levels[1:]):
            if w[L].min() < w[M].max() - tol:
                return False
        return True


# ---------------------------------------------------------------------------
# Calibration problem
# ---------------------------------------------------------------------------
class Calibration:
    """Evaluation of (relative conflict, deviation) for an encoded parameter vector.

    Genes: x = (lam_1..lam_k, u_1..u_n); lam is projected on the simplex (repair), u is used
    only in scenario "WT".
    """

    def __init__(self, X: np.ndarray, model0: Model, measure: str = "fragility",
                 scenario: str = "W", weight_mode: str = "order", ratio_range: float = 2.0,
                 max_dev: float | None = None, dev_alpha: float = 0.5, dev_agg: str = "mean",
                 nu_bound: float | None = None, dev_weights_only: bool = False):
        # deviation variants (revision): Delta = alpha Delta_w + (1 - alpha) Delta_f, Delta_f as the
        # mean (default) or max over criteria; nu_bound = separate box budget on the threshold
        # exponents; dev_weights_only = objective Delta_w with the function budget as a constraint
        self.dev_alpha, self.dev_agg = dev_alpha, dev_agg
        self.nu_bound, self.dev_weights_only = nu_bound, dev_weights_only
        if measure not in CONFLICTS:
            raise ValueError(measure)
        self.X = np.asarray(X, float)
        self.model0 = model0
        self.measure = measure
        self.scenario = scenario
        self.ws = WeightSpace.build(model0.w, weight_mode)
        self.n = model0.n
        self.R = ratio_range
        self.max_dev = max_dev
        self.sigma = np.std(self.X, axis=0)
        self.sigma[self.sigma == 0] = 1.0
        if scenario not in SCENARIOS:
            raise ValueError(scenario)
        self.r0 = model0.p / self.sigma
        self._Pj0 = unicriterion(model0, self.X, self.X)          # cache for scenario W
        n, k = self.n, self.ws.k
        # elicited shape information (scenario F)
        self.gauss0 = np.array([t == "gauss" for t in model0.types])
        bad = [t for t in model0.types if t not in LINEAR_FAMILY + ("gauss",)]
        if scenario == "F" and bad:
            raise ValueError(f"scenario F supports Types III, V and VI only, got {bad}")
        q0 = np.asarray(model0.q, float)
        self.rho0 = np.where(self.gauss0, 0.0, np.clip(q0 / (QMAX * model0.p), 0, 1))
        self.s0 = (np.asarray(model0.s, float) if model0.s is not None else (q0 + model0.p) / 2.0)
        nf = {"W": 0, "WT": n, "F": 3 * n}[scenario]
        self.n_var = k + nf
        if scenario == "F":
            self.xl = np.concatenate([np.zeros(k), -np.ones(n), np.zeros(2 * n)])
            self.x0 = np.concatenate([self.ws.lam0, np.zeros(n), self.rho0, np.where(self.gauss0, 0.75, 0.25)])
        else:
            self.xl = np.concatenate([np.zeros(k), -np.ones(nf)])
            self.x0 = np.concatenate([self.ws.lam0, np.zeros(nf)])
        self.xu = np.ones(self.n_var)
        if nu_bound is not None and scenario in ("WT", "F"):
            self.xl[k:k + n] = -nu_bound
            self.xu[k:k + n] = nu_bound
        self.C0 = self.raw_conflict(self.decode(self.x0))
        self.n_eval = 0

    # -------------------------------------------------------------- encoding
    @staticmethod
    def repair_lam(lam: np.ndarray) -> np.ndarray:
        lam = np.clip(lam, 0, None)
        s = lam.sum(-1, keepdims=True)
        return np.where(s > 0, lam / np.where(s > 0, s, 1), 1.0 / lam.shape[-1])

    def repair(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, float)
        k = self.ws.k
        x[..., :k] = self.repair_lam(x[..., :k])
        x[..., k:] = np.clip(x[..., k:], self.xl[k:], self.xu[k:])
        return x

    def split(self, x: np.ndarray):
        """(lam, u, rho, gauss) of a repaired gene vector."""
        k, n = self.ws.k, self.n
        lam = x[:k]
        u = x[k:k + n] if self.scenario in ("WT", "F") else np.zeros(n)
        if self.scenario == "F":
            rho, gauss = x[k + n:k + 2 * n], x[k + 2 * n:] >= 0.5
        else:
            rho, gauss = self.rho0, self.gauss0
        return lam, u, rho, gauss

    def decode(self, x: np.ndarray) -> Model:
        x = self.repair(x)
        lam, u, rho, gauss = self.split(x)
        w = self.ws.weights(lam)
        if self.scenario == "W":
            return self.model0.with_(w=w)
        p0 = self.r0 * self.sigma
        if self.scenario == "WT":
            # the main threshold is scaled (p for the linear family, s for the Gaussian); the
            # elicited types and indifference ratios q/p are kept, as in scenario F
            g = self.gauss0
            p = np.where(g, p0, p0 * self.R ** u)
            q = np.where(g, self.model0.q, self.rho0 * QMAX * p)
            s = np.where(g, self.s0 * self.R ** u, self.s0)
            return self.model0.with_(w=w, p=p, q=q, s=s)
        p = np.where(gauss, p0, p0 * self.R ** u)
        q = np.where(gauss, 0.0, rho * QMAX * p)
        s = np.where(gauss, self.s0 * self.R ** u, self.s0)
        types = tuple("gauss" if g else "linear" for g in gauss)
        return self.model0.with_(w=w, p=p, q=q, s=s, types=types)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample of the admissible set; row 0 = theta0.

        Without a deviation cap: uniform (Dirichlet(1) on lam, uniform u). With a cap D:
        x = x0 + t (y - x0), y uniform and t ~ U[0, D]; by the contraction property
        Delta(x) <= t <= D, so every sampled point satisfies the cap.
        """
        lam = rng.dirichlet(np.ones(self.ws.k), size=size)
        u = rng.uniform(self.xl[self.ws.k:], self.xu[self.ws.k:], size=(size, self.n_var - self.ws.k))
        x = np.hstack([lam, u])
        if self.max_dev is not None:
            t = rng.uniform(0, self.max_dev, size=(size, 1))
            x = self.x0 + t * (x - self.x0)
        x[0] = self.x0
        return x

    # -------------------------------------------------------------- objectives
    def problem_of(self, model: Model):
        Pj = self._Pj0 if self.scenario == "W" else unicriterion(model, self.X, self.X)
        return _from_Pj(self.X, model, Pj)

    def raw_conflict(self, model: Model) -> float:
        pr = self.problem_of(model)
        if self.measure == "incomparability":
            _, _, R = relations_promethee_i(pr)
            return float(np.triu(R, 1).sum() / (pr.m * (pr.m - 1) / 2))
        return conflict(pr.P, self.measure)

    def function_deviation(self, x: np.ndarray) -> np.ndarray:
        """Per-criterion deviation delta_j of the preference function, in [0, 1]."""
        lam, u, rho, gauss = self.split(self.repair(x))
        if self.scenario == "W":
            return np.zeros(self.n)
        if self.scenario == "WT":
            return np.abs(u)
        d = np.abs(u) + np.where(gauss, 0.0, np.abs(rho - self.rho0)) + (gauss != self.gauss0)
        return np.minimum(d, 1.0)

    def deviation_parts(self, x: np.ndarray):
        """(Delta_w, Delta_f): weight and preference-function deviations."""
        x = self.repair(x)
        w = self.ws.weights(x[:self.ws.k])
        dw = 0.5 * float(np.abs(w - self.model0.w).sum())
        fd = self.function_deviation(x)
        return dw, float(np.max(fd) if self.dev_agg == "max" else np.mean(fd))

    def deviation(self, x: np.ndarray) -> float:
        dw, dp = self.deviation_parts(x)
        if self.scenario == "W" or self.dev_weights_only:
            return dw
        return self.dev_alpha * dw + (1 - self.dev_alpha) * dp

    def shape_changes(self, x: np.ndarray) -> int:
        """Number of criteria whose preference-function family differs from the elicited one."""
        return int(np.sum(self.split(self.repair(x))[3] != self.gauss0))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        self.n_eval += 1
        c = self.raw_conflict(self.decode(x))
        return np.array([c / self.C0 if self.C0 > 0 else c, self.deviation(x)])


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------
def nondominated(F: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of F (minimisation), duplicates kept once."""
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        if dom.any():
            keep[i] = False
            continue
        same = np.all(F == F[i], 1)
        same[:i + 1] = False
        keep[same] = False
    return keep


def knee_point(F: np.ndarray) -> int:
    """Index of the knee of a bi-objective front: largest distance below the chord joining
    the two extreme points (Das, 1999), objectives normalised to [0, 1] on the front."""
    F = np.asarray(F, float)
    if len(F) <= 2:
        return int(np.argmin(F.sum(1)))
    lo, hi = F.min(0), F.max(0)
    G = (F - lo) / np.where(hi > lo, hi - lo, 1)
    a, b = G[np.argmin(G[:, 0])], G[np.argmin(G[:, 1])]
    d = b - a
    nrm = np.hypot(*d) or 1.0
    dist = (d[0] * (G[:, 1] - a[1]) - d[1] * (G[:, 0] - a[0])) / nrm
    return int(np.argmax(np.abs(dist)))


def best_within(F: np.ndarray, budget: float) -> float:
    """Smallest relative conflict reachable with deviation <= budget (inf if none)."""
    ok = F[:, 1] <= budget + 1e-12
    return float(F[ok, 0].min()) if ok.any() else np.inf
