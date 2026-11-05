"""Controlled factorial benchmark of synthetic instances (review comments 2 and 4).

Factors (3 x 3 x 3 x 2 x 2 = 108 families, 2 independent instances each):
  m            number of alternatives      {20, 50, 100}
  n            number of criteria          {4, 8, 12}
  structure    dependence among criteria   concordant (all correlations +0.5), independent,
                                            conflicting (two blocks, +0.5 within, -0.5 across)
  ties         prevalence of near ties     continuous values, or values rounded to 5 levels
  weights      weight statement            strict order (Dirichlet weights, ordinal statement),
                                            equal weights (no statement)

Elicited preference functions: linear, q = 0, p = standard deviation (the convention of the
benchmarks). For each instance: conflict of the elicited model, cost of one exact evaluation
against re-ranking, one NSGA-II run (3,000 evaluations) on the reversal rate in scenarios W
and WT, and the exact model of scenario W for eps = 0.02 when it has at most 5,000 binary
variables (time limit 120 s).

Usage:  python -m experiments.run_synthetic [--workers 2]
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

warnings.filterwarnings("ignore")

from gpii.calibration import Calibration, best_within, knee_point
from gpii.conflict import conflict_measures
from gpii.core import promethee, relations_promethee_i
from gpii.model import Model
from gpii.moea import run

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "synthetic"
RES.mkdir(parents=True, exist_ok=True)

MS, NS = [20, 50, 100], [4, 8, 12]
STRUCT = ["concordant", "independent", "conflicting"]
TIES = ["continuous", "rounded"]
WEIGHTS = ["order", "equal"]
REPS = 2
MAX_BIN, TLIM = 5000, 120


def instance(m, n, structure, ties, weights, rep):
    rng = np.random.default_rng([m, n, STRUCT.index(structure), TIES.index(ties), WEIGHTS.index(weights), rep])
    S = np.eye(n)
    if structure == "concordant":
        S = np.full((n, n), 0.5) + 0.5 * np.eye(n)
    elif structure == "conflicting":
        half = n // 2
        s = np.where(np.arange(n) < half, 1, -1)
        S = 0.5 * np.outer(s, s) + 0.5 * np.eye(n)
    X = rng.multivariate_normal(np.zeros(n), S, size=m)
    if ties == "rounded":
        for j in range(n):
            q = np.quantile(X[:, j], [0.2, 0.4, 0.6, 0.8])
            X[:, j] = np.digitize(X[:, j], q).astype(float)
    if weights == "order":
        w = rng.dirichlet(np.ones(n))
        mode = "order"
    else:
        w = np.full(n, 1.0 / n)
        mode = "free"
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return X, Model(np.ones(n, bool), w, np.zeros(n), sd, types=("linear",) * n), mode


def _task(args):
    m, n, structure, ties, weights, rep = args
    from gpii.exact import model_size, solve_eps
    X, model, mode = instance(*args)
    pr = promethee(X, model)
    c = conflict_measures(pr.P)
    _, _, R = relations_promethee_i(pr)
    inc = float(np.triu(R, 1).sum() / (m * (m - 1) / 2))
    # cost of one exact evaluation against re-ranking every subset A minus {x}
    t = time.perf_counter()
    for _ in range(5):
        conflict_measures(promethee(X, model).P, which=("fragility", "event_rate"))
    t_exact = (time.perf_counter() - t) / 5
    # re-ranking: net flows of every subset A minus {x} from P, and comparison of all pair orders
    P = pr.P
    t = time.perf_counter()
    r0 = P.sum(1)
    base = np.sign(r0[:, None] - r0[None, :])
    flips = 0
    for x in range(m):
        idx = np.r_[0:x, x + 1:m]
        rs = P[np.ix_(idx, idx)].sum(1)
        flips += int((np.sign(rs[:, None] - rs[None, :]) != base[np.ix_(idx, idx)]).sum())
    t_rerank = time.perf_counter() - t
    r = P.sum(1)
    D = np.abs(r[:, None] - r[None, :])[np.triu_indices(m, 1)]
    row = dict(m=m, n=n, structure=structure, ties=ties, weights=weights, rep=rep,
               RR0=c["event_rate"], Fr0=c["fragility"], K0=c["cyclicity"], Inc0=inc,
               cand_share=float((D <= 2).mean()), t_eval_ms=1e3 * t_exact, t_rerank_ms=1e3 * t_rerank)
    if c["event_rate"] <= 0:
        return row
    for sc in ("W", "WT"):
        cal = Calibration(X, model, "event_rate", sc, mode)
        o = run(cal, "NSGA-II", budget=3000, pop=50, seed=rep)
        F, Xs = o["F"], o["X"]
        k = knee_point(F)
        ph = promethee(X, cal.decode(Xs[k])).phi
        row.update({f"{sc}_time": o["time"], f"{sc}_knee_red": 1 - F[k, 0], f"{sc}_knee_dev": F[k, 1],
                    f"{sc}_knee_tau": kendalltau(pr.phi, ph)[0],
                    f"{sc}_knee_top1": int(np.argmax(ph) == np.argmax(pr.phi))})
        for b in (0.01, 0.02, 0.05):
            row[f"{sc}_red@{b}"] = 1 - min(1.0, best_within(F, b))
        if sc == "W":
            cal_w, F_w = cal, F
    size = model_size(cal_w, 0.02)
    row.update(bin_002=size["binaries"], cand_pairs_002=size["candidate_pairs"], pairs=size["pairs"])
    if size["binaries"] <= MAX_BIN:
        t = time.time()
        ex = solve_eps(cal_w, 0.02, time_limit=TLIM)
        row.update(ex_status=ex["status"], ex_time=time.time() - t, ex_rows=ex["n_rows"], ex_bin=ex["n_binaries"])
        if "F" in ex:
            row.update(ex_red=1 - ex["F"][0], evo_red=1 - min(1.0, best_within(F_w, 0.02)))
    return row


def main(workers, resume=False):
    tasks = list(itertools.product(MS, NS, STRUCT, TIES, WEIGHTS, range(REPS)))
    rows = []
    if resume and (RES / "synthetic.csv").exists():
        rows = pd.read_csv(RES / "synthetic.csv").to_dict("records")
        done = {(r["m"], r["n"], r["structure"], r["ties"], r["weights"], r["rep"]) for r in rows}
        tasks = [t for t in tasks if t not in done]
    with Pool(workers) as pool:
        for j, r in enumerate(pool.imap_unordered(_task, tasks)):
            rows.append(r)
            if j % 20 == 19:
                pd.DataFrame(rows).to_csv(RES / "synthetic.csv", index=False)
                print(f"{len(rows)}/216", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "synthetic.csv", index=False)
    print(df.describe().T.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="skip instances already in synthetic.csv")
    a = ap.parse_args()
    main(a.workers, a.resume)
