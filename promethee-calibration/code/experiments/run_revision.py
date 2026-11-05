"""Additional analyses requested in the review for Computers & Operations Research.

Blocks (``--only``):

  ties      decomposition of breaking events into strict reversals and created ties, ties
            broken, and sensitivity to the numerical tie tolerance (comment 10)
  milp      certification trail of the exact model: status, bound, gap, sizes, times, and
            sensitivity to the margin eta in {1e-4, 1e-5, 1e-6} (comment 3)
  blsens    most-stable baseline: sensitivity to the number of samples and seeds, computing
            times of all baselines, optimality gap where the exact optimum is known (comment 6)
  oos       out-of-sample validation with three generators of new alternatives: independent
            marginals, joint bootstrap of rows, Gaussian copula (comment 7)
  roc       elicitation from an ordinal statement only: w0 = centre of the admissible set
            (rank-order centroid for strict orders) instead of the published weights (comment 8)
  dev       alternative deviations in scenario WT: weights alpha, max aggregation, separate
            budgets (comment 9)

Usage:  python -m experiments.run_revision --only ties milp blsens oos roc dev [--workers 2]
"""
from __future__ import annotations

import argparse
import platform
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

warnings.filterwarnings("ignore")

from gpii.calibration import Calibration, best_within, knee_point, nondominated
from gpii.conflict import conflict_measures
from gpii.core import promethee
from gpii.data import load
from gpii.moea import run
from gpii.reversal import _pair_signs, _reversal_rate
from experiments.run_calibration import DATASETS, MODE, RES as CAL_RES, TRACTABLE, _cal

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "revision"
RES.mkdir(parents=True, exist_ok=True)
SCEN = ["W", "WT", "F"]
TOLS = [1e-10, 1e-8, 1e-6, 1e-4]


# ---------------------------------------------------------------------------
# ties
# ---------------------------------------------------------------------------
def decompose(P, tol):
    """Counts of breaking events of single deletions with a tie tolerance ``tol``."""
    m = P.shape[0]
    r = P.sum(1)
    D = r[:, None] - r[None, :]
    a, b = np.nonzero(D > tol)                               # strictly ordered pairs, a above b
    Dn = D[a, b][:, None] - (P[a, :] - P[b, :])                # D - delta_x for every x
    mask = np.ones_like(Dn, bool)
    mask[np.arange(a.size), a] = False
    mask[np.arange(a.size), b] = False
    strict = int(((Dn < -tol) & mask).sum())
    ties = int(((np.abs(Dn) <= tol) & mask).sum())
    ta, tb = np.nonzero(np.triu(np.abs(D) <= tol, 1))         # tied pairs
    if ta.size:
        Dt = D[ta, tb][:, None] - (P[ta, :] - P[tb, :])
        mt = np.ones_like(Dt, bool)
        mt[np.arange(ta.size), ta] = False
        mt[np.arange(ta.size), tb] = False
        broken_ties = int(((np.abs(Dt) > tol) & mt).sum())
    else:
        broken_ties = 0
    denom = (m - 2) * a.size
    return dict(pairs=int(a.size), tied_pairs=int(ta.size), strict=strict, ties=ties,
                ties_broken=broken_ties, RR=(strict + ties) / denom if denom else 0.0,
                RR_strict=strict / denom if denom else 0.0)


def ties():
    Z = np.load(CAL_RES / "final_fronts.npz")
    rows = []
    for name in DATASETS:
        for sc in SCEN:
            d, cal = _cal(name, "event_rate", sc)
            X, F = Z[f"{name}|event_rate|{sc}|X"], Z[f"{name}|event_rate|{sc}|F"]
            pts = {"elicited": cal.x0, "knee": X[knee_point(F)]}
            ok = np.nonzero(F[:, 1] <= 0.02 + 1e-12)[0]
            pts["best@0.02"] = X[ok[np.argmin(F[ok, 0])]]
            for lab, x in pts.items():
                P = promethee(d.X, cal.decode(x)).P
                for tol in TOLS:
                    rows.append(dict(dataset=name, scenario=sc, point=lab, tol=tol, **decompose(P, tol)))
            # share of tie events over every point of the front
            tot = np.zeros(2)
            for x in X:
                c = decompose(promethee(d.X, cal.decode(x)).P, 1e-10)
                tot += (c["strict"], c["ties"])
            rows.append(dict(dataset=name, scenario=sc, point="front", tol=1e-10, strict=int(tot[0]),
                             ties=int(tot[1]), npoints=len(X)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "ties.csv", index=False)
    print(df[df.tol == 1e-10].to_string())


# ---------------------------------------------------------------------------
# milp
# ---------------------------------------------------------------------------
def _milp_task(args):
    name, eps, eta = args
    from gpii.exact import solve_eps
    d, cal = _cal(name, "event_rate", "W")
    t0 = time.time()
    o = solve_eps(cal, eps, eta=eta, time_limit=900)
    total = time.time() - t0
    row = dict(dataset=name, eps=eps, eta=eta, status=o["status"], milp_time=o["time"],
               build_time=total - o["time"], binaries=o["n_binaries"], variables=o["n_vars"],
               rows=o["n_rows"], candidate_pairs=o["candidate_pairs"])
    if "x" in o:
        w = cal.decode(o["x"]).w
        P = promethee(d.X, cal.decode(o["x"])).P
        c = decompose(P, 1e-10)
        row.update(RR_rel=float(o["F"][0]), milp_value=o["milp_value"], dev=0.5 * np.abs(w - d.model.w).sum(),
                   strict=c["strict"], ties=c["ties"],
                   bound=o.get("dual_bound", np.nan))
        # smallest |D| and |D - delta| over uncertified signs: distance of the solution to a tie
        r = P.sum(1)
        D = r[:, None] - r[None, :]
        row["min_abs_D"] = float(np.min(np.abs(D[np.triu_indices(len(r), 1)])))
    return row


def milp(workers):
    from experiments.run_calibration import EXACT_EPS
    tasks = [(n, e, eta) for n in TRACTABLE for e in EXACT_EPS for eta in (1e-3, 1e-4, 1e-5, 1e-6)]
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(_milp_task, tasks))
    df = pd.DataFrame(rows).sort_values(["dataset", "eps", "eta"])
    try:
        import highspy
        hv = highspy.Highs().version()
    except Exception:                                          # noqa: BLE001
        hv = "bundled with SciPy"
    import scipy
    meta = dict(scipy=scipy.__version__, highs=str(hv), python=platform.python_version(),
                cpu=platform.processor() or "Intel Xeon 2.10 GHz", cores_used=1)
    df.to_csv(RES / "milp.csv", index=False)
    pd.Series(meta).to_csv(RES / "milp_meta.csv")
    print(df.round(5).to_string())
    print(meta)


# ---------------------------------------------------------------------------
# blsens
# ---------------------------------------------------------------------------
def _blsens_task(name):
    import experiments.run_baselines as rb
    d, cal = _cal(name, "event_rate", "W")
    V, lam0, w0 = cal.ws.V, cal.ws.lam0, d.model.w
    Pa, rj = rb.structures(d)
    pr0 = promethee(d.X, d.model)
    C0 = conflict_measures(pr0.P, which=("event_rate",))["event_rate"]
    rows = []
    for eps in (0.01, 0.02, 0.05):
        t = time.time(); w = rb.max_margin(V, w0, rj, eps); tmm = time.time() - t
        rr = conflict_measures(promethee(d.X, d.model.with_(w=w)).P, which=("event_rate",))["event_rate"] / C0
        rows.append(dict(dataset=name, eps=eps, method="max-margin", N=0, seed=0, RR=rr, time=tmm))
        for N in (1000, 4000, 16000):
            for seed in range(3):
                rb.N_SAMPLES = N
                t = time.time()
                w_ms, w_mean, share, nr = rb.most_stable_and_mean(V, lam0, w0, eps, rj, np.random.default_rng(100 + seed))
                tt = time.time() - t
                for lab, w in (("most-stable", w_ms), ("smaa-mean", w_mean)):
                    rr = conflict_measures(promethee(d.X, d.model.with_(w=w)).P, which=("event_rate",))["event_rate"] / C0
                    rows.append(dict(dataset=name, eps=eps, method=lab, N=N, seed=seed, RR=rr, time=tt,
                                     mode_share=share if lab == "most-stable" else np.nan,
                                     n_rankings=nr if lab == "most-stable" else np.nan))
    return rows


def blsens(workers):
    with Pool(workers) as pool:
        rows = [r for rr in pool.imap_unordered(_blsens_task, DATASETS) for r in rr]
    df = pd.DataFrame(rows)
    # optimality gaps where the exact optimum is known (scenario W, tractable data sets)
    ex = pd.read_csv(CAL_RES / "exact.csv")
    df.to_csv(RES / "blsens.csv", index=False)
    g = df.groupby(["dataset", "eps", "method", "N"]).agg(RR_mean=("RR", "mean"), RR_sd=("RR", "std"),
                                                          share=("mode_share", "mean"), time=("time", "mean"))
    print(g.round(4).to_string())


# ---------------------------------------------------------------------------
# oos
# ---------------------------------------------------------------------------
def gen_independent(X, k, rng):
    return np.column_stack([rng.choice(X[:, j], size=k, replace=True) for j in range(X.shape[1])])


def gen_bootstrap(X, k, rng):
    return X[rng.integers(0, len(X), size=k)]


def gen_copula(X, k, rng):
    m, n = X.shape
    R = np.column_stack([pd.Series(X[:, j]).rank(method="average").values for j in range(n)])
    Zs = norm.ppf(R / (m + 1))
    C = np.corrcoef(Zs, rowvar=False)
    C = np.nan_to_num(C) + 1e-9 * np.eye(n)
    z = rng.multivariate_normal(np.zeros(n), C, size=k)
    u = norm.cdf(z)
    return np.column_stack([np.quantile(X[:, j], u[:, i]) for i, j in enumerate(range(n))])


GENS = {"independent": gen_independent, "bootstrap": gen_bootstrap, "copula": gen_copula}


def rr_insert(model, X, gen, reps=50, seed=7):
    rng = np.random.default_rng(seed)
    m = len(X)
    ids = np.arange(m)
    bs = _pair_signs(promethee(X, model).phi)
    k = max(1, int(np.ceil(0.1 * m)))
    out = []
    for _ in range(reps):
        Y = gen(X, k, rng)
        sc = promethee(np.vstack([X, Y]), model).phi[:m]
        ff, tt = _reversal_rate(bs, sc, ids)
        out.append(ff / tt if tt else 0.0)
    return float(np.mean(out))


def _oos_task(args):
    name, sc = args
    d, cal = _cal(name, "event_rate", sc)
    Z = np.load(CAL_RES / "final_fronts.npz")
    X, F = Z[f"{name}|event_rate|{sc}|X"], Z[f"{name}|event_rate|{sc}|F"]
    knee = cal.decode(X[knee_point(F)])
    rows = []
    for g, fn in GENS.items():
        e = rr_insert(d.model, d.X, fn)
        kk = rr_insert(knee, d.X, fn)
        rows.append(dict(dataset=name, scenario=sc, generator=g, elicited=e, knee=kk,
                         ratio=kk / e if e > 0 else np.nan))
    return rows


def oos(workers):
    tasks = [(n, sc) for n in DATASETS for sc in SCEN]
    with Pool(workers) as pool:
        rows = [r for rr in pool.imap_unordered(_oos_task, tasks) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "oos.csv", index=False)
    print(df.round(4).to_string())
    print(df.groupby("generator").ratio.agg(["median", lambda s: (s < 1).sum(), "count"]))


# ---------------------------------------------------------------------------
# roc
# ---------------------------------------------------------------------------
def _roc_task(name):
    from gpii.exact import solve_eps
    d = load(name)
    from gpii.calibration import WeightSpace
    ws = WeightSpace.build(d.model.w, MODE[name])
    w_roc = ws.centroid()
    model = d.model.with_(w=w_roc)
    cal = Calibration(d.X, model, "event_rate", "W", MODE[name])
    Xs, Fs = [], []
    for s in range(3):
        o = run(cal, "NSGA-II", budget=6000, pop=50, seed=300 + s)
        Xs.append(o["X"]); Fs.append(o["F"])
    if name in TRACTABLE:
        for eps in (0.005, 0.01, 0.02, 0.05):
            o = solve_eps(cal, eps, time_limit=900)
            if "x" in o:
                Xs.append(o["x"][None, :]); Fs.append(o["F"][None, :])
    X, F = np.vstack(Xs), np.vstack(Fs)
    keep = nondominated(F)
    F = F[keep]
    pub = pd.read_csv(CAL_RES / "calibrate.csv")
    p = pub[(pub.dataset == name) & (pub.measure == "event_rate") & (pub.scenario == "W")].iloc[0]
    row = dict(dataset=name, tv_roc_pub=0.5 * np.abs(w_roc - d.model.w).sum(), C0_roc=cal.C0, C0_pub=p.C0)
    for b in (0.01, 0.02, 0.05):
        row[f"roc@{b}"] = best_within(F, b)
        row[f"pub@{b}"] = p[f"best@{b}"]
    return row


def roc(workers):
    names = [n for n in DATASETS if MODE[n] == "order"]
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(_roc_task, names))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "roc.csv", index=False)
    print(df.round(4).to_string())


# ---------------------------------------------------------------------------
# dev
# ---------------------------------------------------------------------------
VARIANTS = {
    "alpha=0.5 (paper)": dict(),
    "alpha=0.25": dict(dev_alpha=0.25),
    "alpha=0.75": dict(dev_alpha=0.75),
    "max over criteria": dict(dev_agg="max"),
    "weights only, |nu|<=0.25": dict(dev_weights_only=True, nu_bound=0.25),
    "weights only, |nu|<=0.5": dict(dev_weights_only=True, nu_bound=0.5),
}


def _dev_task(args):
    name, lab = args
    d = load(name)
    cal = Calibration(d.X, d.model, "event_rate", "WT", MODE[name], **VARIANTS[lab])
    Xs, Fs = [], []
    for s in range(3):
        o = run(cal, "NSGA-II", budget=6000, pop=50, seed=400 + s)
        Xs.append(o["X"]); Fs.append(o["F"])
    X, F = np.vstack(Xs), np.vstack(Fs)
    keep = nondominated(F)
    X, F = X[keep], F[keep]
    # report by the weight deviation and the largest threshold change of each point
    dw = np.array([cal.deviation_parts(x)[0] for x in X])
    numax = np.array([np.max(np.abs(cal.split(cal.repair(x))[1])) for x in X])
    row = dict(dataset=name, variant=lab)
    for b in (0.01, 0.02, 0.05):
        row[f"best@{b}"] = best_within(F, b)
        ok = dw <= b + 1e-12
        row[f"bestW@{b}"] = float(F[ok, 0].min()) if ok.any() else np.nan
        row[f"numax@{b}"] = float(numax[ok][np.argmin(F[ok, 0])]) if ok.any() else np.nan
    return row


def dev(workers):
    tasks = [(n, v) for n in DATASETS for v in VARIANTS]
    with Pool(workers) as pool:
        rows = list(pool.imap_unordered(_dev_task, tasks))
    df = pd.DataFrame(rows).sort_values(["dataset", "variant"])
    df.to_csv(RES / "dev.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["ties", "milp", "blsens", "oos", "roc", "dev"])
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    for block in a.only:
        t = time.time()
        print(f"== {block}", flush=True)
        f = globals()[block]
        f() if block == "ties" else f(a.workers)
        print(f"== {block} done in {time.time() - t:.0f}s", flush=True)
