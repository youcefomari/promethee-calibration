"""Rank-reversal certificates and empirical rank-reversal protocols.

Certificates (manuscript, Section 4)
------------------------------------
D_B(a,b) = sum_{s in B} [P(a,s) - P(b,s)] = |B| (phi_B(a) - phi_B(b)) has the sign of the
PROMETHEE II order of a and b in the problem B. Deleting x (inserting y) changes it by
-delta_x (+delta_y), delta_x(a,b) = P(a,x) + P(x,b) = P(a,b) - C(a,b,x) in [-2, 2].

* universal certificate : |D| > 2k  => order kept under any k insertions/deletions;
* exact deletion radius : minimal number of deletions that breaks the strict order;
* exact insertion bound : worst delta_y over a product domain Y, separable by criterion.

Empirical protocols
-------------------
A *method* is a callable ``f(X, ids) -> scores`` where ``ids`` identifies the original
alternatives (index in the initial problem, -1 for inserted alternatives).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .model import Model
from .preference import breakpoints, preference_degree

Method = Callable[[np.ndarray, np.ndarray], np.ndarray]
TIE_TOL = 1e-10  # differences below this are treated as exact ties (floating-point noise)


def tsign(x, tol: float = TIE_TOL):
    """Sign with a tolerance: |x| <= tol counts as an exact tie (0)."""
    x = np.asarray(x, float)
    return np.where(np.abs(x) <= tol, 0.0, np.sign(x))


# ---------------------------------------------------------------------------
# Certificates
# ---------------------------------------------------------------------------
def D_matrix(P: np.ndarray) -> np.ndarray:
    r = P.sum(1)
    return r[:, None] - r[None, :]


def delta(P: np.ndarray, x: int) -> np.ndarray:
    """delta_x(a, b) = P(a, x) + P(x, b) for all pairs (a, b)."""
    return P[:, x][:, None] + P[x, :][None, :]


def deletion_radius(P: np.ndarray, a: int, b: int) -> float:
    """Minimal number of deletions (x not in {a,b}) after which a is no longer strictly above b.

    Assumes D(a,b) > 0. Deleting a set X gives D - sum_{x in X} delta_x; the largest
    reduction with |X| = k is the sum of the k largest delta_x, so the radius is the first
    k at which that sum reaches D (np.inf if never).
    """
    r = P.sum(1)
    D = r[a] - r[b]
    if D <= 0:
        raise ValueError("deletion_radius expects a pair with D(a, b) > 0")
    others = np.setdiff1d(np.arange(P.shape[0]), [a, b])
    d = np.sort(P[a, others] + P[others, b])[::-1]
    c = np.cumsum(d)
    hit = np.nonzero(c >= D - 1e-12)[0]
    return float(hit[0] + 1) if hit.size else np.inf


def loo_reversal_rate(P: np.ndarray) -> float:
    """Exact leave-one-out reversal rate of PROMETHEE II from Theorem 4 (no re-ranking).

    Share, over all deletions x and all pairs {a, b} not containing x that are strictly
    ordered in A, of the pairs whose order changes (reversal or tie) when x is deleted.
    """
    m = P.shape[0]
    D = D_matrix(P)
    sD = tsign(D)
    iu = np.triu(np.ones((m, m), bool), 1)
    flips = total = 0
    for x in range(m):
        Dn = D - (P[:, x][:, None] + P[x, :][None, :])
        keep = iu.copy()
        keep[x, :] = False
        keep[:, x] = False
        strict = keep & (sD != 0)
        flips += int((strict & (tsign(Dn) != sD)).sum())
        total += int(strict.sum())
    return flips / total if total else 0.0


def adjacent_pairs(phi: np.ndarray):
    order = np.argsort(-phi, kind="stable")
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)
            if phi[order[i]] > phi[order[i + 1]]]


def _h_criterion(t, ga, gb, code, q, p, s, sign):
    """Contribution of one criterion to delta_y(a,b) = P(a,y) + P(y,b) when y_j = t."""
    F = lambda d: preference_degree(d, code, q, p, s)
    da, db = sign * (ga - t), sign * (t - gb)
    return F(da) - F(-da) + F(db) - F(-db)


def insertion_worst_case(model: Model, xa: np.ndarray, xb: np.ndarray,
                         lo: np.ndarray, hi: np.ndarray, grid: int = 2001) -> float:
    """min_{y in prod_j [lo_j, hi_j]} delta_y(a, b) (exact for piecewise-linear shapes).

    delta_y is a weighted sum of one-dimensional functions of y_j, so the minimum over a
    product domain is the weighted sum of one-dimensional minima. For piecewise-linear F
    each minimum is attained at a breakpoint of the reflected functions or at a bound.
    """
    total = 0.0
    for j in range(model.n):
        code, q, p = model.types[j], float(model.q[j]), float(model.p[j])
        s = None if model.s is None else float(model.s[j])
        sign = 1.0 if model.maximize[j] else -1.0
        ga, gb = float(xa[j]), float(xb[j])
        bps = breakpoints(code, q, p)
        if bps:
            cand = [lo[j], hi[j]]
            for g in (ga, gb):
                for bp in bps:
                    cand += [g - bp, g + bp]
            eps = 1e-9 * max(1.0, abs(hi[j] - lo[j]))
            cand = np.array(cand)
            cand = np.concatenate([cand, cand - eps, cand + eps])   # one-sided limits at jumps
            cand = cand[(cand >= lo[j]) & (cand <= hi[j])]
        else:
            cand = np.linspace(lo[j], hi[j], grid)
        total += model.w[j] * float(np.min(_h_criterion(cand, ga, gb, code, q, p, s, sign)))
    return total


# ---------------------------------------------------------------------------
# Empirical protocols
# ---------------------------------------------------------------------------
def _pair_signs(s: np.ndarray) -> np.ndarray:
    return tsign(s[:, None] - s[None, :])


def _reversal_rate(base_sign: np.ndarray, new_scores: np.ndarray, idx: np.ndarray):
    o = _pair_signs(new_scores)
    b = base_sign[np.ix_(idx, idx)]
    strict = np.triu(b != 0, 1)
    flips = strict & (o != b)
    return int(flips.sum()), int(strict.sum())


def rr_deletion(method: Method, X: np.ndarray, top1: bool = True) -> dict:
    """Leave-one-out: delete each alternative in turn; share of strictly ordered pairs that change."""
    m = X.shape[0]
    ids = np.arange(m)
    base = method(X, ids)
    bs = _pair_signs(base)
    best = int(np.argmax(base))
    f = t = top_change = 0
    for x in range(m):
        idx = np.delete(ids, x)
        sc = method(X[idx], idx)
        ff, tt = _reversal_rate(bs, sc, idx)
        f, t = f + ff, t + tt
        if x != best and idx[int(np.argmax(sc))] != best:
            top_change += 1
    return {"pair_rr": f / t if t else 0.0, "top1_change": top_change / (m - 1)}


def rr_subsample(method: Method, X: np.ndarray, frac: float = 0.7, reps: int = 30,
                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    ids = np.arange(m)
    bs = _pair_signs(method(X, ids))
    rates = []
    for _ in range(reps):
        idx = np.sort(rng.choice(m, max(2, int(round(frac * m))), replace=False))
        ff, tt = _reversal_rate(bs, method(X[idx], idx), idx)
        rates.append(ff / tt if tt else 0.0)
    return {"pair_rr": float(np.mean(rates)), "pair_rr_sd": float(np.std(rates))}


def synthetic_alternatives(X: np.ndarray, k: int, rng) -> np.ndarray:
    """New alternatives drawn independently from the empirical marginal of each criterion."""
    return np.column_stack([rng.choice(X[:, j], size=k, replace=True) for j in range(X.shape[1])])


def rr_insertion(method: Method, X: np.ndarray, frac: float = 0.1, reps: int = 30,
                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    ids = np.arange(m)
    bs = _pair_signs(method(X, ids))
    k = max(1, int(np.ceil(frac * m)))
    rates = []
    for _ in range(reps):
        Y = synthetic_alternatives(X, k, rng)
        sc = method(np.vstack([X, Y]), np.concatenate([ids, -np.ones(k, int)]))[:m]
        ff, tt = _reversal_rate(bs, sc, ids)
        rates.append(ff / tt if tt else 0.0)
    return {"pair_rr": float(np.mean(rates)), "pair_rr_sd": float(np.std(rates))}


# ---------------------------------------------------------------------------
# Screening: exact single-change reversibility versus threshold rules
# ---------------------------------------------------------------------------
def single_change_reversible(P: np.ndarray, model=None, X=None, domain=None):
    """Exact sets of pairs whose strict PROMETHEE II order can be broken by ONE change.

    Returns boolean upper-triangular matrices (pairs with D != 0):
      deletion[a, b]  -- some x in A \\ {a, b} breaks the order when deleted (Theorem 4);
      insertion[a, b] -- some y in the box ``domain`` breaks it when inserted (Proposition 3);
    together with the per-pair worst-case insertion effect.
    """
    m = P.shape[0]
    D = D_matrix(P)
    sD = tsign(D)
    iu = np.triu(np.ones((m, m), bool), 1) & (sD != 0)
    dele = np.zeros((m, m), bool)
    for x in range(m):
        Dn = D - (P[:, x][:, None] + P[x, :][None, :])
        hit = iu & (tsign(Dn) != sD)
        hit[x, :] = False
        hit[:, x] = False
        dele |= hit
    ins = np.zeros((m, m), bool)
    worst = np.full((m, m), np.nan)
    if model is not None:
        lo, hi = domain if domain is not None else (X.min(0), X.max(0))
        for a, b in zip(*np.nonzero(iu)):
            hi_, lo_ = (a, b) if D[a, b] > 0 else (b, a)      # orient: hi_ strictly above lo_
            wc = insertion_worst_case(model, X[hi_], X[lo_], lo, hi)
            worst[a, b] = wc
            ins[a, b] = abs(D[a, b]) + wc <= TIE_TOL
    return {"pairs": iu, "deletion": dele, "insertion": ins, "D": D, "worst_insertion": worst}


def screening_report(P: np.ndarray, model, X) -> dict:
    """Precision/recall of threshold rules against the exact single-change sets."""
    g = single_change_reversible(P, model, X)
    iu, D = g["pairs"], g["D"]
    m = P.shape[0]
    universal = iu & (np.abs(D) <= 2 + TIE_TOL)                 # |phi(a)-phi(b)| <= 2/(m-1)
    maxdel = np.zeros((m, m))
    for x in range(m):
        dx = np.abs(P[:, x][:, None] + P[x, :][None, :])
        dx[x, :] = 0
        dx[:, x] = 0
        maxdel = np.maximum(maxdel, dx)
    sample_dep = iu & (np.abs(D) <= maxdel + TIE_TOL)          # necessary condition for deletion

    def pr(flag, truth):
        tp = (flag & truth).sum()
        return (tp / flag.sum() if flag.sum() else 1.0, tp / truth.sum() if truth.sum() else 1.0)

    out = {"pairs": int(iu.sum()), "true_del": int(g["deletion"].sum()), "true_ins": int(g["insertion"].sum()),
           "flag_universal": int(universal.sum()), "flag_sampledep": int(sample_dep.sum())}
    out["prec_univ_del"], out["rec_univ_del"] = pr(universal, g["deletion"])
    out["prec_univ_ins"], out["rec_univ_ins"] = pr(universal, g["insertion"])
    out["prec_sd_del"], out["rec_sd_del"] = pr(sample_dep, g["deletion"])
    return out
