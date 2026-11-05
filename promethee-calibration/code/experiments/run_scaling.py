"""Empirical check of the complexity analysis (Table 2): time of each component as a function of
m and n on random instances, and per-evaluation cost and search overhead on the benchmarks.
Usage: python -m experiments.run_scaling"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from experiments.run_calibration import DATASETS, ROOT, _cal
from gpii.conflict import conflict_measures
from gpii.core import promethee
from gpii.model import Model, unicriterion
from gpii.moea import run

OUT = ROOT / "results" / "revision"


def tmed(f, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def rerank(P):
    m = len(P)
    r0 = P.sum(1)
    base = np.sign(r0[:, None] - r0[None, :])
    for x in range(m):
        idx = np.r_[0:x, x + 1:m]
        rs = P[np.ix_(idx, idx)].sum(1)
        (np.sign(rs[:, None] - rs[None, :]) != base[np.ix_(idx, idx)]).sum()


def synthetic():
    rows = []
    for n in (4, 16):
        for m in (50, 100, 200, 400, 800, 1600):
            rng = np.random.default_rng(m + n)
            X = rng.normal(size=(m, n))
            w = rng.dirichlet(np.ones(n))
            mdl = Model(np.ones(n, bool), w, np.zeros(n), X.std(0), types=("linear",) * n)
            reps = 7 if m <= 400 else 3
            Pj = unicriterion(mdl, X, X)
            P = np.tensordot(w, Pj, 1)
            r = P.sum(1)
            d = np.abs(r[:, None] - r[None, :])[np.triu_indices(m, 1)]
            row = dict(m=m, n=n, Q=int((d <= 2).sum()),
                       t_Pj=tmed(lambda: unicriterion(mdl, X, X), reps),
                       t_agg=tmed(lambda: np.tensordot(w, Pj, 1), reps),
                       t_eval=tmed(lambda: conflict_measures(P, which=("fragility", "event_rate")), reps),
                       t_rerank=tmed(lambda: rerank(P), reps) if m <= 800 else np.nan)
            rows.append(row); print(row, flush=True)
    return pd.DataFrame(rows)


def benchmarks():
    rows = []
    for name in DATASETS:
        row = dict(dataset=name)
        for sc in ("W", "WT"):
            d, cal = _cal(name, "event_rate", sc)
            rng = np.random.default_rng(0)
            xs = [cal.repair(cal.x0 + 0.05 * rng.normal(size=cal.x0.shape)) for _ in range(200)]
            t = time.perf_counter()
            for x in xs:
                cal.evaluate(x)
            row[f"t_eval_{sc}"] = (time.perf_counter() - t) / len(xs)
            t = time.perf_counter()
            run(cal, "NSGA-II", budget=3000, pop=50, seed=0)
            row[f"t_run_{sc}"] = time.perf_counter() - t
            row[f"overhead_{sc}"] = row[f"t_run_{sc}"] - 3000 * row[f"t_eval_{sc}"]
        row.update(m=d.m, n=d.n)
        rows.append(row); print(row, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from experiments.hardware import record
    OUT.mkdir(parents=True, exist_ok=True)
    record(OUT / "hardware_scaling.txt")
    s = synthetic(); s.to_csv(OUT / "scaling_synthetic.csv", index=False)
    b = benchmarks(); b.to_csv(OUT / "scaling_benchmarks.csv", index=False)
    for c in ("t_Pj", "t_agg", "t_eval", "t_rerank"):
        for n, g in s.groupby("n"):
            g = g[g.m >= 200].dropna(subset=[c])
            print(c, n, "log-log slope in m:", round(np.polyfit(np.log(g.m), np.log(g[c]), 1)[0], 2))
    print(b.round(5).to_string())
