"""Statistical comparison of the search methods with multiplicity control (review comment 5).

Indicators (unchanged): hypervolume with reference point (1, 1) of the fronts in the objective
space (kappa / kappa(theta0), Delta), both objectives already on [0, 1] up to points with more
conflict than the elicited model, which lie outside the reference box; IGD+ with respect to
the non-dominated union of all fronts of all methods and runs for the case.

Inference:
  * per case (dataset x measure x scenario, 10 independent runs per method): two-sided
    Mann-Whitney U test of NSGA-II against each method (runs are independent, not paired),
    Holm correction over all case x method tests, Vargha-Delaney A12 effect size;
  * across cases (the inferential unit is the case, 30 cases): Friedman test on mean
    hypervolumes, and Wilcoxon signed-rank tests of NSGA-II against each method on the mean
    hypervolumes per case, Holm-corrected over the four comparisons, with the median paired
    difference and its bootstrap 95% interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

from experiments.run_calibration import RES

ALGS = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, k in enumerate(o):
        run = max(run, (len(p) - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return (gt + 0.5 * eq) / (len(x) * len(y))


def main():
    df = pd.read_csv(RES / "compare_metrics.csv")
    cases = df.groupby(["dataset", "measure", "scenario"])
    rows = []
    for key, g in cases:
        base = g[g.algorithm == "NSGA-II"].hv.values
        for alg in ALGS[1:]:
            other = g[g.algorithm == alg].hv.values
            if np.allclose(base, base[0]) and np.allclose(other, base[0]):
                p = 1.0
            else:
                p = mannwhitneyu(base, other, alternative="two-sided").pvalue
            rows.append(dict(dataset=key[0], measure=key[1], scenario=key[2], other=alg, p=p,
                             A12=a12(base, other), med_diff=np.median(base) - np.median(other)))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    t["verdict"] = np.where(t.p_holm >= 0.05, "equivalent", np.where(t.med_diff > 0, "NSGA-II better", "NSGA-II worse"))
    t.to_csv(RES.parent / "revision" / "stats_cases.csv", index=False)
    summ = t.groupby("other").agg(better=("verdict", lambda s: int((s == "NSGA-II better").sum())),
                                  equivalent=("verdict", lambda s: int((s == "equivalent").sum())),
                                  worse=("verdict", lambda s: int((s == "NSGA-II worse").sum())),
                                  A12_median=("A12", "median"))
    # across cases
    means = df.groupby(["dataset", "measure", "scenario", "algorithm"]).hv.mean().unstack()
    fr = friedmanchisquare(*[means[a].values for a in ALGS])
    rng = np.random.default_rng(0)
    across = []
    for alg in ALGS[1:]:
        dlt = means["NSGA-II"].values - means[alg].values
        p = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
        boots = [np.median(rng.choice(dlt, len(dlt))) for _ in range(5000)]
        across.append(dict(other=alg, p=p, median_diff=np.median(dlt),
                           ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                           wins=int((dlt > 0).sum()), ties=int((dlt == 0).sum()), losses=int((dlt < 0).sum())))
    a = pd.DataFrame(across)
    a["p_holm"] = holm(a.p.values)
    out = summ.join(a.set_index("other"))
    times = df.groupby("algorithm").time.agg(["mean", "std"])
    out.to_csv(RES.parent / "revision" / "stats_summary.csv")
    times.to_csv(RES.parent / "revision" / "stats_times.csv")
    print(f"Friedman chi2={fr.statistic:.1f} p={fr.pvalue:.2e}")
    print(out.round(4).to_string())
    print(times.round(2))
    return out, fr, times


if __name__ == "__main__":
    main()
