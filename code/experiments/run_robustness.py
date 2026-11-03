"""Monte Carlo robustness study: sensitivity of the calibration to the elicited parameters.

The benchmarks publish weights but no preference functions, so the main experiments anchor
every criterion on one convention (Type V, q = 0, p = standard deviation). Here the elicited
model theta0 is drawn at random, and the calibration is repeated for each draw:

  type mix   all Type V, all Gaussian, or mixed (each criterion Gaussian w.p. 1/2), w.p. 1/3 each
  scale      c ~ log-uniform[0.5, 2]; reference thresholds p~_j = c * sd_j, q~_j = r_j p~_j,
             r_j ~ U[0, 0.5]  (so rho0_j = r_j / qbar is uniform on [0, 1])
             Type V: (q0, p0) = (q~, p~); Gaussian: s0 = (q~ + p~) / 2, with p0 = 2 s0, q0 = 0
             for a change of type (conventions of Sec. 4.1)
  weights    w0 = V((1 - t) lam_pub + t lam), lam ~ Dirichlet(1), t ~ U[0, 0.3]: a perturbation
             of the published weights with the same weak order (no statement: the simplex)

For each draw, measure in {RR, K} and scenario in {W, WT, F}: one NSGA-II run (3,000
evaluations); the F front is nested with the embedded WT front (as in Sec. 7). The knees of the
RR fronts (W and F) are validated out of sample.

Usage:  python -m experiments.run_robustness [--draws 30] [--workers 2]
        python -m experiments.run_robustness --tables      (figure, table and macros only)
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from gpii.calibration import QMAX, Calibration, best_within, knee_point, nondominated
from gpii.core import promethee
from gpii.data import load
from gpii.moea import run
from gpii.reversal import rr_insertion, rr_subsample

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "robustness"
RES.mkdir(parents=True, exist_ok=True)

DATASETS = ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]
MODE = {"HDI": "order", "WHR24": "free", "BLI": "free", "ASEM": "order", "ELECTRIC": "order"}
MEASURES = ["event_rate", "cyclicity"]
SCENARIOS = ["W", "WT", "F"]
BUDGET, POP = 3000, 50
BUDGETS = [0.01, 0.02, 0.05, 0.10]
REPS = 20


def draw_model(name: str, draw: int):
    """Random elicited model for data set ``name`` (reproducible from (name, draw))."""
    d = load(name)
    rng = np.random.default_rng([DATASETS.index(name), draw, 2026])
    n = d.n
    sd = np.std(d.X, axis=0)
    sd[sd == 0] = 1.0
    mix = ["V", "VI", "mixed"][draw % 3]                     # balanced design over the type mix
    gauss = np.zeros(n, bool) if mix == "V" else np.ones(n, bool) if mix == "VI" else rng.random(n) < 0.5
    c = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    r = rng.uniform(0.0, QMAX, n)
    pt = c * sd
    qt = r * pt
    s0 = (qt + pt) / 2
    p0 = np.where(gauss, 2 * s0, pt)
    q0 = np.where(gauss, 0.0, qt)
    # weights: perturbation of the published ones within the same admissible set
    from gpii.calibration import WeightSpace
    ws = WeightSpace.build(d.model.w, MODE[name])
    t = rng.uniform(0.0, 0.3)
    lam = (1 - t) * ws.lam0 + t * rng.dirichlet(np.ones(ws.k))
    w0 = ws.weights(lam)
    types = tuple("gauss" if g else "linear" for g in gauss)
    model = d.model.with_(w=w0, q=q0, p=p0, s=s0, types=types)
    factors = dict(mix=mix, scale=c, q_ratio=float(np.mean(r)), n_gauss=int(gauss.sum()),
                   w_shift=float(0.5 * np.abs(w0 - d.model.w).sum()))
    return d, model, factors


def _phi(model):
    return lambda X, ids: promethee(X, model).phi


def _task(args):
    name, draw = args
    from scipy.stats import kendalltau
    t0 = time.time()
    d, model, factors = draw_model(name, draw)
    rows, phi0 = [], promethee(d.X, model).phi
    for me in MEASURES:
        fronts = {}
        for sc in SCENARIOS:
            cal = Calibration(d.X, model, me, sc, MODE[name])
            if cal.C0 <= 0:
                fronts[sc] = None
                continue
            o = run(cal, "NSGA-II", budget=BUDGET, pop=POP, seed=1000 + draw)
            X, F = o["X"], o["F"]
            if sc == "F" and fronts.get("WT") is not None:        # nest WT into F
                XT = fronts["WT"][1]
                k, n = cal.ws.k, cal.n
                emb = np.tile(cal.x0, (len(XT), 1))
                emb[:, :k + n] = XT[:, :k + n]
                X = np.vstack([X, emb])
                F = np.vstack([F, np.array([cal.evaluate(x) for x in emb])])
            keep = nondominated(F)
            X, F = X[keep], F[keep]
            o = np.argsort(F[:, 1])
            fronts[sc] = (cal, X[o], F[o])
        for sc in SCENARIOS:
            base = dict(dataset=name, draw=draw, measure=me, scenario=sc, **factors)
            if fronts[sc] is None:
                rows.append(dict(base, C0=0.0))
                continue
            cal, X, F = fronts[sc]
            k = knee_point(F)
            mk = cal.decode(X[k])
            phk = promethee(d.X, mk).phi
            row = dict(base, C0=cal.C0, n_front=len(F), knee_C=F[k, 0], knee_D=F[k, 1],
                       knee_tau=kendalltau(phi0, phk)[0], knee_top1=int(np.argmax(phk) == np.argmax(phi0)),
                       knee_shapes=cal.shape_changes(X[k]) if sc == "F" else 0,
                       share_shapes=float(np.mean([cal.shape_changes(x) > 0 for x in X])) if sc == "F" else 0.0,
                       min_C=float(F[:, 0].min()))
            for b in BUDGETS:
                row[f"best@{b}"] = best_within(F, b)
            if me == "event_rate" and sc in ("W", "F"):
                for lab, x in (("elicited", cal.x0), ("knee", X[k])):
                    mdl = cal.decode(x)
                    row[f"ins_{lab}"] = rr_insertion(_phi(mdl), d.X, 0.1, REPS, seed=7)["pair_rr"]
                    row[f"sub_{lab}"] = rr_subsample(_phi(mdl), d.X, 0.8, REPS, seed=7)["pair_rr"]
            rows.append(row)
    for r in rows:
        r["time"] = time.time() - t0
    return rows


def experiment(draws: int, workers: int):
    out = RES / "robustness.csv"
    done = set()
    old = []
    if out.exists():
        dfo = pd.read_csv(out)
        old = dfo.to_dict("records")
        done = set(zip(dfo.dataset, dfo.draw))
    tasks = [(n, i) for i in range(draws) for n in DATASETS if (n, i) not in done]
    print(f"{len(tasks)} draws to run", flush=True)
    rows = old
    with Pool(workers) as pool:
        for j, rr in enumerate(pool.imap_unordered(_task, tasks)):
            rows.extend(rr)
            if j % 5 == 4 or j == len(tasks) - 1:
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{j + 1}/{len(tasks)} ({rr[0]['dataset']} #{rr[0]['draw']}, {rr[0]['time']:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    if a.tables:
        from experiments.robustness_tables import make_all
        make_all()
    else:
        experiment(a.draws, a.workers)
