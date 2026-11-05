"""Experiments of the calibration manuscript (Sections 7.1-7.7).

Blocks (``--only`` selects a subset; results are cached as CSV/NPZ in results/calibration):

  diagnose   conflict of the elicited parameters (Table 2)
  compare    NSGA-II vs SMS-EMOA, MOEA/D, SPEA2 and random sampling: HV, IGD+ (Table 3)
  exact      epsilon-constraint MILP vs NSGA-II on small instances (Table 4)
  calibrate  final NSGA-II fronts (5 seeds pooled) for the four conflict measures (Table 5, Fig. 1)
  validate   out-of-sample rank reversal (insertion, sub-sampling) of the calibrated models (Table 6)
  tables     LaTeX tables, figures and numbers macros from the cached results

Usage:  python -m experiments.run_calibration [--only compare exact] [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from gpii.calibration import Calibration, best_within, knee_point, nondominated
from gpii.conflict import conflict_measures
from gpii.core import promethee, relations_promethee_i
from gpii.cycles import cyclic_residual, two_cycle_index
from gpii.data import load
from gpii.model import Model, unicriterion
from gpii.moea import ALGORITHMS, hypervolume, igd_plus, run
from gpii.reversal import rr_insertion, rr_subsample

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "calibration"
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
for d in (RES, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

DATASETS = ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]
# Admissible weight information: the weak order of the published weights, except where equal
# weights are a convention without preference content (WHR, BLI): then the whole simplex.
MODE = {"HDI": "order", "WHR24": "free", "BLI": "free", "ASEM": "order", "ELECTRIC": "order"}
SCENARIOS = ["W", "WT", "F"]
RUN_SCENARIOS = SCENARIOS          # restricted with --scenarios; results are merged into the caches
MEASURES = ["fragility", "event_rate", "cyclicity", "incomparability"]
CMP_MEASURES = ["event_rate", "incomparability"]
CMP_SEEDS = 10
BUDGET_CMP, POP = 3000, 50
BUDGET_FINAL, FINAL_SEEDS = 6000, 5
FINAL_ALGO = os.environ.get("CAL_ALGO", "NSGA-II")   # set from the comparison of Sec. 7.3
BUDGETS = [0.01, 0.02, 0.05, 0.10]


def _cal(name, measure, scenario):
    d = load(name)
    return d, Calibration(d.X, d.model, measure, scenario, MODE[name])


# ---------------------------------------------------------------------------
def diagnose():
    rows = []
    for name in DATASETS:
        d = load(name)
        pr = promethee(d.X, d.model)
        c = conflict_measures(pr.P)
        _, _, R = relations_promethee_i(pr)
        cal = Calibration(d.X, d.model, "cyclicity", "W", MODE[name])
        # Rayleigh bound on the cyclicity reachable by re-weighting (Proposition 4)
        Pj = unicriterion(d.model, d.X, d.X)
        Pa = Pj - Pj.transpose(0, 2, 1)
        Rc = np.array([cyclic_residual(p) for p in Pa])
        A = np.einsum("iab,jab->ij", Rc, Rc)
        B = np.einsum("iab,jab->ij", Pa, Pa)
        V = cal.ws.V
        from scipy.linalg import eigh
        lmin = float(eigh(V.T @ A @ V, V.T @ B @ V, eigvals_only=True)[0])
        rows.append(dict(dataset=name, m=d.m, n=d.n, info=MODE[name], k=cal.ws.k,
                         fragility=c["fragility"], event_rate=c["event_rate"], impact=c["impact"],
                         cyclicity=c["cyclicity"], K_bound=lmin, I2=two_cycle_index(pr.pi),
                         incomparability=float(np.triu(R, 1).sum() / (d.m * (d.m - 1) / 2)),
                         strict_pairs=int(np.triu(np.abs(pr.phi[:, None] - pr.phi[None, :]) > 1e-10, 1).sum())))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "diagnose.csv", index=False)
    print(df.round(4).to_string())
    return df


# ---------------------------------------------------------------------------
def _cmp_task(args):
    name, measure, scenario, algo, seed = args
    _, cal = _cal(name, measure, scenario)
    out = run(cal, algo, budget=BUDGET_CMP, pop=POP, seed=seed)
    return dict(dataset=name, measure=measure, scenario=scenario, algorithm=algo, seed=seed,
                time=out["time"], n_eval=out["n_eval"], F=out["F"].tolist())


def compare(workers):
    tasks = [(n, me, sc, a, s) for n in DATASETS for me in CMP_MEASURES for sc in RUN_SCENARIOS
             for a in ALGORITHMS for s in range(CMP_SEEDS)]
    t0 = time.time()
    with Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_cmp_task, tasks, chunksize=2)):
            rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  compare {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    path = RES / "compare_fronts.json"
    if path.exists():
        rows = [r for r in json.load(open(path)) if r["scenario"] not in RUN_SCENARIOS] + rows
    with open(path, "w") as f:
        json.dump(rows, f)
    _compare_metrics()


def _compare_metrics():
    rows = json.load(open(RES / "compare_fronts.json"))
    df = pd.DataFrame(rows)
    out = []
    for (n, me, sc), g in df.groupby(["dataset", "measure", "scenario"]):
        allF = np.vstack([np.array(F) for F in g.F])
        Z = allF[nondominated(allF)]
        Z = Z[np.all(Z <= 1.0, 1)]
        for _, r in g.iterrows():
            F = np.array(r.F)
            out.append(dict(dataset=n, measure=me, scenario=sc, algorithm=r.algorithm, seed=r.seed,
                            hv=hypervolume(F), igd=igd_plus(F[np.all(F <= 1.0, 1)] if np.any(np.all(F <= 1.0, 1)) else F, Z),
                            n_front=len(F), time=r.time))
    m = pd.DataFrame(out)
    m.to_csv(RES / "compare_metrics.csv", index=False)
    print(m.groupby(["measure", "scenario", "dataset", "algorithm"]).hv.mean().unstack().round(4).to_string())
    return m


# ---------------------------------------------------------------------------
def _sim_instance(m, n, seed):
    rng = np.random.default_rng(np.random.SeedSequence([20261020, m, n, seed]))
    X = rng.normal(size=(m, n))
    w = rng.dirichlet(np.ones(n))
    return X, Model(np.ones(n, bool), w, np.zeros(n), 0.5 * X.std(0))


EXACT_ALGOS = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]
EXACT_EPS = [0.005, 0.01, 0.02, 0.05]
TRACTABLE = ["ELECTRIC", "BLI", "ASEM"]      # pruned MILP solved within the time limit (Sec. 7.4)


def _exact_task(args):
    from gpii.exact import solve_eps
    kind, m, n, seed = args
    if kind == "sim":
        X, model = _sim_instance(m, n, seed)
        cal = Calibration(X, model, "event_rate", "W", "free")
        label, tl = f"sim m={m} n={n} #{seed}", 300
    else:
        d, cal = _cal(kind, "event_rate", "W")
        label, tl = kind, 900
    fronts = {a: [run(cal, a, budget=BUDGET_CMP, pop=POP, seed=s)["F"] for s in range(3)]
              for a in EXACT_ALGOS}
    res = []
    for eps in EXACT_EPS:
        o = solve_eps(cal, eps, time_limit=tl)
        row = dict(instance=label, kind="sim" if kind == "sim" else "bench", m=cal.X.shape[0], n=cal.n,
                   eps=eps, status=o["status"], milp_time=o["time"], binaries=o["n_binaries"],
                   exact=o["F"][0] if "F" in o else np.nan, C0=cal.C0)
        for a, Fs in fronts.items():
            v = [best_within(F, eps) for F in Fs]
            row[f"best_{a}"] = float(np.min(v))
            row[f"mean_{a}"] = float(np.mean(v))
        res.append(row)
    return res


def exact(workers):
    tasks = [("sim", m, n, s) for m, n in [(10, 3), (10, 5), (12, 4)] for s in range(5)]
    tasks += [(d, 0, 0, 0) for d in TRACTABLE]
    with Pool(workers) as pool:
        rows = [r for rr in pool.imap_unordered(_exact_task, tasks) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exact.csv", index=False)
    print(df.round(4).to_string())


# ---------------------------------------------------------------------------
def _final_task(args):
    name, measure, scenario = args
    _, cal = _cal(name, measure, scenario)
    Xs, Fs = [], []
    for s in range(FINAL_SEEDS):
        o = run(cal, FINAL_ALGO, budget=BUDGET_FINAL, pop=POP, seed=100 + s)
        Xs.append(o["X"]); Fs.append(o["F"])
    X, F = np.vstack(Xs), np.vstack(Fs)
    keep = nondominated(F)
    X, F = X[keep], F[keep]
    n_evo = len(F)
    n_exact = 0
    if measure in ("event_rate", "fragility") and name in TRACTABLE:
        from gpii.exact import refine, solve_eps
        add_X, add_F = [], []
        if scenario == "W":
            for eps in EXACT_EPS:
                o = solve_eps(cal, eps, time_limit=900)
                if "x" in o:
                    add_X.append(o["x"]); add_F.append(o["F"])
        else:
            sel = np.nonzero(F[:, 1] <= 0.1)[0]
            sel = sel[np.linspace(0, len(sel) - 1, min(len(sel), 8)).round().astype(int)] if len(sel) else sel
            for i in np.unique(sel):
                xn, fn, _ = refine(cal, X[i], time_limit=300)
                add_X.append(xn); add_F.append(fn)
        if add_X:
            Fe = np.vstack(add_F)
            X, F = np.vstack([X, np.vstack(add_X)]), np.vstack([F, Fe])
            keep = nondominated(F)
            n_exact = int(keep[n_evo:].sum())
            X, F = X[keep], F[keep]
    o = np.argsort(F[:, 1])
    return name, measure, scenario, X[o], F[o], n_exact


def _summary_row(name, me, sc, X, F, n_exact):
    from scipy.stats import kendalltau
    d, cal = _cal(name, me, sc)
    k = knee_point(F)
    phi0 = promethee(d.X, d.model).phi

    def info(x):
        mdl = cal.decode(x)
        ph = promethee(d.X, mdl).phi
        dw, dp = cal.deviation_parts(x)
        return dict(tau=kendalltau(phi0, ph)[0], top1=int(np.argmax(ph) == np.argmax(phi0)),
                    hhi=float((mdl.w ** 2).sum()), dw=dw, dp=dp, shapes=cal.shape_changes(x))
    imin = int(np.argmin(F[:, 0] + 1e-9 * F[:, 1]))
    row = dict(dataset=name, measure=me, scenario=sc, C0=cal.C0, n_front=len(F), n_exact=n_exact,
               knee_C=F[k, 0], knee_D=F[k, 1], **{f"knee_{a}": b for a, b in info(X[k]).items()},
               min_C=F[imin, 0], min_D=F[imin, 1], **{f"min_{a}": b for a, b in info(X[imin]).items()},
               hhi0=float((d.model.w ** 2).sum()))
    for b in BUDGETS:
        row[f"best@{b}"] = best_within(F, b)
    return row


def nest(workers=None):
    """Scenario WT is the restriction of F to the elicited shapes and indifference thresholds,
    with the same deviation (Sec. 5.1). Every WT point is therefore admissible in F; the F front
    reported is the non-dominated union of the F search and of the embedded WT front."""
    Z = np.load(RES / "final_fronts.npz")
    store = {k: Z[k] for k in Z.files}
    old = pd.read_csv(RES / "calibrate.csv")
    rows = old[old.scenario != "F"].to_dict("records")
    for name in DATASETS:
        for me in MEASURES:
            _, calF = _cal(name, me, "F")
            _, calT = _cal(name, me, "WT")
            XF, FF = Z[f"{name}|{me}|F|X"], Z[f"{name}|{me}|F|F"]
            XT = Z[f"{name}|{me}|WT|X"]
            k, n = calF.ws.k, calF.n
            emb = np.tile(calF.x0, (len(XT), 1))
            emb[:, :k + n] = XT[:, :k + n]
            FT = np.array([calF.evaluate(x) for x in emb])
            X, F = np.vstack([XF, emb]), np.vstack([FF, FT])
            keep = nondominated(F)
            X, F = X[keep], F[keep]
            o = np.argsort(F[:, 1])
            X, F = X[o], F[o]
            store[f"{name}|{me}|F|X"], store[f"{name}|{me}|F|F"] = X, F
            n_ex = int(old[(old.dataset == name) & (old.measure == me) & (old.scenario == "F")].n_exact.iloc[0])
            row = _summary_row(name, me, "F", X, F, n_ex)
            row["from_WT"] = int(keep[len(XF):].sum())
            rows.append(row)
    np.savez(RES / "final_fronts.npz", **store)
    df = pd.DataFrame(rows).sort_values(["measure", "dataset", "scenario"])
    df.to_csv(RES / "calibrate.csv", index=False)
    print(df[df.scenario == "F"][["dataset", "measure", "from_WT", "n_front", "best@0.02", "best@0.1"]].round(3).to_string())


def calibrate(workers):
    tasks = [(n, me, sc) for n in DATASETS for me in MEASURES for sc in RUN_SCENARIOS]
    with Pool(workers) as pool:
        res = list(pool.imap_unordered(_final_task, tasks))
    store, rows = {}, []
    if (RES / "final_fronts.npz").exists():
        old = np.load(RES / "final_fronts.npz")
        store = {k: old[k] for k in old.files if k.split("|")[2] not in RUN_SCENARIOS}
        rows = pd.read_csv(RES / "calibrate.csv")
        rows = rows[~rows.scenario.isin(RUN_SCENARIOS)].to_dict("records")
    for name, me, sc, X, F, n_exact in res:
        store[f"{name}|{me}|{sc}|X"] = X
        store[f"{name}|{me}|{sc}|F"] = F
        rows.append(_summary_row(name, me, sc, X, F, n_exact))
    np.savez(RES / "final_fronts.npz", **store)
    df = pd.DataFrame(rows).sort_values(["measure", "dataset", "scenario"])
    df.to_csv(RES / "calibrate.csv", index=False)
    print(df.round(3).to_string())


# ---------------------------------------------------------------------------
def _phi_method(model):
    return lambda X, ids: promethee(X, model).phi


def _validate_task(args):
    name, scenario = args
    d, cal = _cal(name, "event_rate", scenario)
    Z = np.load(RES / "final_fronts.npz")
    X, F = Z[f"{name}|event_rate|{scenario}|X"], Z[f"{name}|event_rate|{scenario}|F"]
    k = knee_point(F)
    cand = {"elicited": cal.x0, "knee": X[k]}
    ok = F[:, 1] <= 0.02 + 1e-12
    cand["best@0.02"] = X[ok][np.argmin(F[ok, 0])]
    cen = cal.x0.copy(); cen[:cal.ws.k] = 1.0 / cal.ws.k
    cand["centroid"] = cen
    rows = []
    for lab, x in cand.items():
        mdl = cal.decode(x)
        ins = rr_insertion(_phi_method(mdl), d.X, 0.1, 50, seed=7)
        sub = rr_subsample(_phi_method(mdl), d.X, 0.8, 50, seed=7)
        rows.append(dict(dataset=name, scenario=scenario, model=lab, D=cal.deviation(x),
                         C=cal.evaluate(x)[0], ins=ins["pair_rr"], sub=sub["pair_rr"]))
    # local random perturbations with the deviation of the knee
    rng = np.random.default_rng(13)
    Dk = cal.deviation(X[k])
    loc = []
    while len(loc) < 20:
        u = cal.sample(2, rng)[1]
        t = rng.uniform(0, 0.5)
        x = cal.repair((1 - t) * cal.x0 + t * u)
        if Dk * 0.5 <= cal.deviation(x) <= Dk:
            loc.append(x)
    vals = []
    for x in loc:
        mdl = cal.decode(x)
        vals.append((cal.evaluate(x)[0], rr_insertion(_phi_method(mdl), d.X, 0.1, 50, seed=7)["pair_rr"],
                     rr_subsample(_phi_method(mdl), d.X, 0.8, 50, seed=7)["pair_rr"], cal.deviation(x)))
    v = np.mean(vals, 0)
    rows.append(dict(dataset=name, scenario=scenario, model="local-random", D=v[3], C=v[0], ins=v[1], sub=v[2]))
    return rows


def validate(workers):
    tasks = [(n, sc) for n in DATASETS for sc in RUN_SCENARIOS]
    with Pool(workers) as pool:
        rows = [r for rr in pool.imap_unordered(_validate_task, tasks) for r in rr]
    if (RES / "validate.csv").exists():
        old = pd.read_csv(RES / "validate.csv")
        rows = old[~old.scenario.isin(RUN_SCENARIOS)].to_dict("records") + rows
    df = pd.DataFrame(rows).sort_values(["dataset", "scenario"])
    df.to_csv(RES / "validate.csv", index=False)
    print(df.round(5).to_string())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["diagnose", "compare", "exact", "calibrate", "validate", "tables"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS)
    a = ap.parse_args()
    RUN_SCENARIOS = list(a.scenarios)
    for block in a.only:
        t = time.time()
        print(f"== {block}", flush=True)
        if block == "tables":
            from experiments.calibration_tables import make_all
            make_all()
        elif block in ("compare", "exact", "calibrate", "validate", "nest"):
            globals()[block](a.workers)
        else:
            globals()[block]()
        print(f"== {block} done in {time.time() - t:.0f}s", flush=True)
