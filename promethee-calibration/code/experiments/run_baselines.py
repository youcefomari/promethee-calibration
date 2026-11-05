"""Comparison with robustness-oriented weight choices from the literature (scenario W).

For each data set and deviation budget eps (share of weight moved, TV(w, w0) <= eps), every
method returns one admissible weight vector:

  calibration   best point of the final reversal-rate front of scenario W within eps
                (evolutionary + exact points, Sec. 3.4)
  max-margin    maximises the smallest normalised margin w'd_ab / span(d_ab) over the pairs
                ordered by the elicited model (largest stability radius, Lemma 1); a linear
                program, in the spirit of stability analysis (Mareschal 1988; Doan & De Smet 2018)
  most-stable   the most stable ranking (Asudeh et al. 2018): the PROMETHEE II ranking that
                occupies the largest share of the admissible weights within eps, estimated by
                uniform sampling; its weights are the mean of the samples producing it
  smaa-mean     SMAA-type expected net flows (Corrente et al. 2014): ranking by the expected net
                flow over the admissible weights within eps, i.e. (by linearity) the model with
                the mean admissible weight vector

All are evaluated with the same measures: reversal rate and fragility (relative to the
elicited model), Kendall tau, first-ranked alternative, and out-of-sample reversal rate under
insertion (10%) and sub-sampling (80%), 50 repetitions each.

Usage:  python -m experiments.run_baselines [--workers 2]
"""
from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

from gpii.calibration import best_within
from gpii.conflict import conflict_measures
from gpii.core import promethee
from gpii.model import unicriterion
from experiments.run_calibration import DATASETS, MODE, RES as CAL_RES, _cal

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "baselines"
RES.mkdir(parents=True, exist_ok=True)
EPS = [0.01, 0.02, 0.05]
N_SAMPLES = 4000
REPS = 50


# ---------------------------------------------------------------------------
def structures(d):
    """Per-criterion preference differences Pa (n, m, m) and net-flow sums phij (n, m)."""
    Pj = unicriterion(d.model, d.X, d.X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    return Pa, Pa.sum(2)


def tv(w, w0):
    return 0.5 * np.abs(w - w0).sum()


def max_margin(V, w0, rj, eps):
    """LP: max t  s.t.  (V lam)' d_ab >= t span(d_ab) for the elicited strict pairs."""
    n, k = V.shape
    D0 = w0 @ rj
    a, b = np.nonzero(D0[:, None] - D0[None, :] > 1e-10)
    Dj = (rj[:, a] - rj[:, b]).T                       # (pairs, n)
    span = np.ptp(Dj, axis=1)
    keep = span > 1e-12
    Dj, span = Dj[keep], span[keep]
    # variables: lam (k), h (n), t (1); minimise -t
    c = np.zeros(k + n + 1); c[-1] = -1.0
    A1 = np.hstack([-(Dj @ V), np.zeros((len(Dj), n)), span[:, None]])
    b1 = np.zeros(len(Dj))
    A2 = np.hstack([V, -np.eye(n), np.zeros((n, 1))]); b2 = w0
    A3 = np.hstack([-V, -np.eye(n), np.zeros((n, 1))]); b3 = -w0
    A4 = np.concatenate([np.zeros(k), np.ones(n), [0.0]])[None, :]; b4 = [2 * eps]
    Aeq = np.concatenate([np.ones(k), np.zeros(n + 1)])[None, :]
    bounds = [(0, None)] * k + [(0, None)] * n + [(None, None)]
    r = linprog(c, A_ub=np.vstack([A1, A2, A3, A4]), b_ub=np.concatenate([b1, b2, b3, b4]),
                A_eq=Aeq, b_eq=[1.0], bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return V @ r.x[:k]


def hit_and_run(V, lam0, w0, eps, n_samples, rng, burn=300, thin=5):
    """Uniform samples of lam in {simplex, TV(V lam, w0) <= eps} (hit-and-run)."""
    k = V.shape[1]
    if k == 1:
        return np.tile(lam0, (n_samples, 1))
    lam = (1 - eps / 2) * lam0 + (eps / 2) / k
    out = []
    feas = lambda l: tv(V @ l, w0) <= eps
    for it in range(burn + thin * n_samples):
        u = rng.normal(size=k); u -= u.mean(); u /= np.linalg.norm(u)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = -lam / u
        thi = np.min(np.where(u < 0, r, np.inf)); tlo = np.max(np.where(u > 0, r, -np.inf))

        def edge(t_out):                                  # bisection on the TV constraint
            lo, hi = 0.0, t_out
            if feas(lam + t_out * u):
                return t_out
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if feas(lam + mid * u) else (lo, mid)
            return lo
        a, b = edge(tlo), edge(thi)
        lam = np.clip(lam + rng.uniform(a, b) * u, 0, None); lam /= lam.sum()
        if it >= burn and (it - burn) % thin == 0:
            out.append(lam.copy())
    return np.array(out)


def most_stable_and_mean(V, lam0, w0, eps, phij, rng):
    L = hit_and_run(V, lam0, w0, eps, N_SAMPLES, rng)
    W = L @ V.T                                           # (samples, n)
    phi = W @ phij                                        # net flows up to a positive factor
    keys = [tuple(np.argsort(-p, kind="stable")) for p in np.round(phi, 12)]
    cnt = Counter(keys)
    key, freq = cnt.most_common(1)[0]
    sel = np.array([kk == key for kk in keys])
    return W[sel].mean(0), W.mean(0), freq / len(keys), len(cnt)


# ---------------------------------------------------------------------------
def evaluate(d, w, C0, phi0, top0):
    mdl = d.model.with_(w=w)
    pr = promethee(d.X, mdl)
    c = conflict_measures(pr.P, which=("fragility", "event_rate"))
    from gpii.reversal import rr_insertion, rr_subsample
    meth = lambda X, ids: promethee(X, mdl).phi
    return dict(RR=c["event_rate"] / C0["event_rate"] if C0["event_rate"] else 0.0,
                Fr=c["fragility"] / C0["fragility"] if C0["fragility"] else 0.0,
                tau=kendalltau(phi0, pr.phi)[0], top1=int(np.argmax(pr.phi) == top0),
                ins=rr_insertion(meth, d.X, 0.1, REPS, seed=7)["pair_rr"],
                sub=rr_subsample(meth, d.X, 0.8, REPS, seed=7)["pair_rr"])


def _task(name):
    t0 = time.time()
    d, cal = _cal(name, "event_rate", "W")
    V, lam0, w0 = cal.ws.V, cal.ws.lam0, d.model.w
    Pa, rj = structures(d)
    pr0 = promethee(d.X, d.model)
    C0 = conflict_measures(pr0.P, which=("fragility", "event_rate"))
    phi0, top0 = pr0.phi, int(np.argmax(pr0.phi))
    base = evaluate(d, w0, C0, phi0, top0)
    rows = [dict(dataset=name, eps=0.0, method="elicited", dev=0.0, **base)]
    Z = np.load(CAL_RES / "final_fronts.npz")
    X, F = Z[f"{name}|event_rate|W|X"], Z[f"{name}|event_rate|W|F"]
    rng = np.random.default_rng(11)
    for eps in EPS:
        ok = np.nonzero(F[:, 1] <= eps + 1e-12)[0]
        i = ok[np.argmin(F[ok, 0] + 1e-9 * F[ok, 1])]
        w_cal = cal.decode(X[i]).w
        w_mm = max_margin(V, w0, rj, eps)
        w_ms, w_mean, share, n_rank = most_stable_and_mean(V, lam0, w0, eps, rj, rng)
        for lab, w in (("calibration", w_cal), ("max-margin", w_mm), ("most-stable", w_ms), ("smaa-mean", w_mean)):
            r = dict(dataset=name, eps=eps, method=lab, dev=tv(w, w0), **evaluate(d, w, C0, phi0, top0))
            if lab == "most-stable":
                r.update(mode_share=share, n_rankings=n_rank)
            rows.append(r)
    for r in rows:
        r["time"] = time.time() - t0
    return rows


def main(workers):
    with Pool(workers) as pool:
        rows = [r for rr in pool.imap_unordered(_task, DATASETS) for r in rr]
    df = pd.DataFrame(rows).sort_values(["dataset", "eps", "method"])
    df.to_csv(RES / "baselines.csv", index=False)
    print(df.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args().workers)
