"""Wall-clock time of each stage of the procedure on the five benchmarks, in the two modes of use
(Section 4.4). Every stage runs in one process on one core.

Decision-maker mode: diagnosis, one NSGA-II run of 3,000 evaluations for the reversal rate in
scenario W, the exact model for the stated budget (0.02) when the data set is tractable, and the
selection of the least-conflict model within the budget.

Analyst mode (configuration of the paper): diagnosis, five NSGA-II runs of 6,000 evaluations in
scenarios W and WT, the exact model for the four budgets (W) and the refinement of up to eight
points (WT) on tractable data sets, selection of the knee, and validation of the knee with 50
insertions and 50 sub-samples.

Usage: python -m experiments.run_timing [--workers 2]
"""
from __future__ import annotations

import argparse
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from experiments.run_calibration import (BUDGET_FINAL, DATASETS, EXACT_EPS, FINAL_SEEDS, MODE, POP, ROOT,
                                         TRACTABLE, _cal)
from gpii.calibration import best_within, knee_point, nondominated
from gpii.conflict import conflict_measures
from gpii.core import promethee, relations_promethee_i
from gpii.cycles import cyclic_residual
from gpii.data import load
from gpii.model import unicriterion
from gpii.moea import run
from gpii.reversal import rr_insertion, rr_subsample

OUT = ROOT / "results" / "revision" / "timing.csv"


def _diagnose(d, cal):
    pr = promethee(d.X, d.model)
    conflict_measures(pr.P)
    relations_promethee_i(pr)
    Pj = unicriterion(d.model, d.X, d.X)
    Pa = Pj - Pj.transpose(0, 2, 1)
    Rc = np.array([cyclic_residual(p) for p in Pa])
    A = np.einsum("iab,jab->ij", Rc, Rc)
    B = np.einsum("iab,jab->ij", Pa, Pa)
    from scipy.linalg import eigh
    eigh(cal.ws.V.T @ A @ cal.ws.V, cal.ws.V.T @ B @ cal.ws.V, eigvals_only=True)


def diagnosis(name):
    """Median time of the diagnosis over five repetitions, after one warm-up call; reading the
    data and the first-call overhead of the libraries are excluded."""
    d = load(name)
    _, cal = _cal(name, "cyclicity", "W")
    _diagnose(d, cal)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        _diagnose(d, cal)
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def evo(cal, runs, budget):
    t = time.perf_counter()
    Xs, Fs = [], []
    for s in range(runs):
        o = run(cal, "NSGA-II", budget=budget, pop=POP, seed=100 + s)
        Xs.append(o["X"]); Fs.append(o["F"])
    X, F = np.vstack(Xs), np.vstack(Fs)
    keep = nondominated(F)
    return X[keep], F[keep], time.perf_counter() - t


def _task(name):
    from gpii.exact import refine, solve_eps
    row = dict(dataset=name)
    d = load(name)
    row.update(m=d.m, n=d.n)
    row["diagnosis"] = diagnosis(name)
    tract = name in TRACTABLE
    # ---- decision-maker mode ---------------------------------------------
    _, cal = _cal(name, "event_rate", "W")
    X, F, row["dm_search"] = evo(cal, 1, 3000)
    row["dm_exact"] = 0.0
    if tract:
        t = time.perf_counter()
        o = solve_eps(cal, 0.02, time_limit=900)
        row["dm_exact"] = time.perf_counter() - t
        if "F" in o:
            F = np.vstack([F, o["F"]])
    t = time.perf_counter()
    best_within(F, 0.02)
    row["dm_select"] = time.perf_counter() - t
    row["dm_total"] = row["diagnosis"] + row["dm_search"] + row["dm_exact"] + row["dm_select"]
    # ---- analyst mode ----------------------------------------------------
    row["an_search"], row["an_exact"] = 0.0, 0.0
    knee = None
    for sc in ("W", "WT"):
        _, cal = _cal(name, "event_rate", sc)
        X, F, ts = evo(cal, FINAL_SEEDS, BUDGET_FINAL)
        row["an_search"] += ts
        if tract:
            t = time.perf_counter()
            if sc == "W":
                for eps in EXACT_EPS:
                    solve_eps(cal, eps, time_limit=900)
            else:
                sel = np.nonzero(F[:, 1] <= 0.1)[0]
                sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 8)).round().astype(int)] if len(sel) else sel
                for i in np.unique(sel):
                    refine(cal, X[i], time_limit=300)
            row["an_exact"] += time.perf_counter() - t
        if sc == "WT":
            t = time.perf_counter()
            k = knee_point(F)
            row["an_select"] = time.perf_counter() - t
            knee = cal.decode(X[k])
    t = time.perf_counter()
    phi = lambda Xa, ids: promethee(Xa, knee).phi
    rr_insertion(phi, d.X, 0.1, 50, seed=7)
    rr_subsample(phi, d.X, 0.8, 50, seed=7)
    row["an_validation"] = time.perf_counter() - t
    row["an_total"] = row["diagnosis"] + row["an_search"] + row["an_exact"] + row["an_select"] + row["an_validation"]
    print(row, flush=True)
    return row


def main(workers):
    with Pool(workers) as pool:
        rows = pool.map(_task, DATASETS)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    from experiments.hardware import record
    OUT.parent.mkdir(parents=True, exist_ok=True)
    record(OUT.parent / "hardware_timing.txt")
    main(ap.parse_args().workers)
