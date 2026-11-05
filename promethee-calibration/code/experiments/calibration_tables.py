"""LaTeX tables, figures and number macros for the calibration manuscript."""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from experiments.run_calibration import (BUDGETS, DATASETS, FIG, MODE, RES, SCENARIOS, TAB)

DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
       "7": "seven", "8": "eight", "9": "nine"}
NAMES = {"HDI": "HDI", "WHR24": "WHR", "BLI": "BLI", "ASEM": "ASEM", "ELECTRIC": "EV"}
ALG_ORDER = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]


def macro(name):
    name = re.sub(r"[^A-Za-z0-9]", "", name)
    return "".join(DIG.get(c, c) for c in name)


class Numbers:
    def __init__(self):
        self.d = {}

    def __setitem__(self, k, v):
        self.d[macro(k)] = v

    def write(self):
        with open(TAB / "numbers_cal.tex", "w") as f:
            for k, v in sorted(self.d.items()):
                f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def tab(df_lines, fname):
    with open(TAB / fname, "w") as f:
        f.write("\n".join(df_lines) + "\n")


# ---------------------------------------------------------------------------
def t_diagnose(N):
    df = pd.read_csv(RES / "diagnose.csv")
    L = []
    for _, r in df.iterrows():
        info = "order" if r["info"] == "order" else "none"
        L.append(f"{NAMES[r.dataset]} & {r.m} & {r.n} & {info} & {r.k} & {pct(r.fragility, 2)} & "
                 f"{pct(r.event_rate, 3)} & {pct(r.impact, 1)} & {r.cyclicity:.3f} & {max(r.K_bound, 0.0):.3f} & "
                 f"{r.I2:.3f} & {pct(r.incomparability, 1)} \\\\")
        N[f"frag{r.dataset}"] = pct(r.fragility, 2)
        N[f"event{r.dataset}"] = pct(r.event_rate, 3)
        N[f"impact{r.dataset}"] = pct(r.impact, 0)
        N[f"K{r.dataset}"] = f"{r.cyclicity:.3f}"
        N[f"Kbound{r.dataset}"] = f"{max(r.K_bound, 0.0):.3f}"
        N[f"inc{r.dataset}"] = pct(r.incomparability, 1)
        N[f"fragpairs{r.dataset}"] = str(int(round(r.fragility * r.strict_pairs)))
    tab(L, "cal_diagnose.tex")


def holm_cases():
    """Within-case tests of revision_stats.py: Mann-Whitney on the ten runs, Holm over all 120 tests."""
    st = pd.read_csv(RES.parent / "revision" / "stats_cases.csv")
    return {(r.dataset, r.measure, r.scenario, r.other): (r.p_holm, r.med_diff) for r in st.itertuples()}


def t_compare(N):
    global HOLM
    HOLM = holm_cases()
    m = pd.read_csv(RES / "compare_metrics.csv")
    L = []
    wins = {a: 0 for a in ALG_ORDER}
    ranks_all = []
    for me in ["event_rate", "incomparability"]:
        for sc in SCENARIOS:
            for ds in DATASETS:
                g = m[(m.measure == me) & (m.scenario == sc) & (m.dataset == ds)]
                if g.empty:
                    continue
                piv = g.pivot(index="seed", columns="algorithm", values="hv")[ALG_ORDER]
                ranks_all.append(piv.rank(axis=1, ascending=False).mean())
                mean, sd = piv.mean(), piv.std()
                best = mean.idxmax()
                wins[best] += 1
                cells = []
                for a in ALG_ORDER:
                    s = f"{mean[a]:.3f}\\,({sd[a]:.3f})"
                    if a == best:
                        s = f"\\textbf{{{mean[a]:.3f}}}\\,({sd[a]:.3f})"
                    elif a != "NSGA-II":
                        # Mann-Whitney test on the ten runs, Holm-corrected over all 120 within-case tests
                        p, d = HOLM.get((ds, me, sc, a), (1.0, 0.0))
                        mark = "$^{+}$" if (p < 0.05 and d > 0) else (
                            "$^{-}$" if (p < 0.05 and d < 0) else "$^{\\approx}$")
                        s += mark
                    cells.append(s)
                mlabel = "RR" if me == "event_rate" else "Inc"
                L.append(f"{mlabel} & {sc} & {NAMES[ds]} & " + " & ".join(cells) + " \\\\")
            L.append("\\midrule")
    L = L[:-1]
    tab(L, "cal_compare.tex")
    R = pd.DataFrame(ranks_all)
    avg = R.mean()
    piv_all = []
    for (me, sc, ds), g in m.groupby(["measure", "scenario", "dataset"]):
        piv_all.append(g.groupby("algorithm").hv.mean()[ALG_ORDER])
    PA = pd.DataFrame(piv_all)
    stat, p = friedmanchisquare(*[PA[a] for a in ALG_ORDER])
    L2 = [" & ".join(["Average rank"] + [f"{avg[a]:.2f}" for a in ALG_ORDER]) + " \\\\",
          " & ".join(["Best mean HV (cases)"] + [str(wins[a]) for a in ALG_ORDER]) + " \\\\",
          " & ".join(["Mean time (s)"] + [f"{m[m.algorithm == a].time.mean():.1f}" for a in ALG_ORDER]) + " \\\\"]
    tab(L2, "cal_compare_summary.tex")
    for a in ALG_ORDER:
        N[f"rank{a}"] = f"{avg[a]:.2f}"
        N[f"wins{a}"] = str(wins[a])
        N[f"time{a}"] = f"{m[m.algorithm == a].time.mean():.1f}"
    N["friedmanstat"] = f"{stat:.1f}"
    N["friedmanp"] = f"{p:.1e}".replace("e-0", "e-").replace("e", r"\times 10^{") + "}" if p < 1e-3 else f"{p:.3f}"
    N["ncases"] = str(len(PA))
    # NSGA-II vs each: number of cases significantly better / worse
    for a in ALG_ORDER[1:]:
        better = worse = 0
        for (me, sc, ds), g in m.groupby(["measure", "scenario", "dataset"]):
            piv = g.pivot(index="seed", columns="algorithm", values="hv")
            try:
                p = wilcoxon(piv["NSGA-II"], piv[a]).pvalue
            except ValueError:
                p = 1.0
            if p < 0.05:
                if piv["NSGA-II"].mean() > piv[a].mean():
                    better += 1
                else:
                    worse += 1
        N[f"better{a}"] = str(better)
        N[f"worse{a}"] = str(worse)
    # IGD+ ranks
    ig = m.groupby(["measure", "scenario", "dataset", "algorithm"]).igd.mean().unstack()[ALG_ORDER]
    N["igdbestNSGAII"] = str(int((ig.idxmin(axis=1) == "NSGA-II").sum()))
    igr = ig.rank(axis=1).mean()
    for a in ALG_ORDER:
        N[f"igdrank{a}"] = f"{igr[a]:.2f}"


def t_exact(N):
    df = pd.read_csv(RES / "exact.csv")
    algos = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]
    df["opt"] = df.status == 0
    L = []
    sim = df[df.kind == "sim"].copy()
    sim["size"] = sim.apply(lambda r: f"$m={r.m}$, $n={r.n}$", axis=1)
    for (size, eps), g in sim.groupby(["size", "eps"], sort=False):
        g2 = g[g.opt]
        cells = [f"{g2[f'best_{a}'].mean():.3f}" for a in algos]
        hit = [int((g2[f'best_{a}'] <= g2.exact + 1e-9).sum()) for a in algos]
        L.append(f"{size} & {eps:.3f} & {g.opt.sum()}/{len(g)} & {g2.binaries.mean():.0f} & {g2.milp_time.mean():.1f} & "
                 f"{g2.exact.mean():.3f} & " + " & ".join(f"{c} ({h})" for c, h in zip(cells, hit)) + " \\\\")
    L.append("\\midrule")
    for _, r in df[df.kind == "bench"].iterrows():
        ex = f"{r.exact:.3f}" if np.isfinite(r.exact) else "--"
        cells = [f"{r[f'best_{a}']:.3f}" for a in algos]
        L.append(f"{NAMES[r.instance]} & {r.eps:.3f} & {'opt.' if r.status == 0 else 'limit'} & {int(r.binaries)} & {r.milp_time:.1f} & {ex} & "
                 + " & ".join(cells) + " \\\\")
    tab(L, "cal_exact.tex")
    s = sim[sim.opt]
    N["exactcases"] = str(len(s))
    for a in algos:
        N[f"exacthit{a}"] = str(int((s[f'best_{a}'] <= s.exact + 1e-9).sum()))
        N[f"exactgap{a}"] = f"{(s[f'best_{a}'] - s.exact).clip(lower=0).mean():.3f}"
    N["exactmaxtime"] = f"{s.milp_time.max():.0f}"
    b_ = df[df.kind == "bench"]
    N["benchexacttimemax"] = f"{b_.milp_time.max():.0f}"
    N["benchbinmax"] = str(int(b_.binaries.max()))
    N["benchbinmin"] = str(int(b_.binaries.min()))
    N["benchopt"] = str(int(b_.opt.sum()))
    N["benchn"] = str(len(b_))
    for _, r in b_.iterrows():
        k = f"{r.instance}at{int(round(r.eps * 1000))}"
        N[f"exact{k}"] = f"{r.exact:.3f}"
        best = min(r[f'best_{a}'] for a in algos[:-1])
        N[f"evo{k}"] = f"{best:.3f}"


def t_calibrate(N):
    df = pd.read_csv(RES / "calibrate.csv")
    L = []
    for me, ml in [("fragility", "Fragility"), ("event_rate", "RR rate"), ("cyclicity", "Cyclicity $K$"),
                   ("incomparability", "PROMETHEE I inc.")]:
        g = df[df.measure == me]
        first = True
        for ds in DATASETS:
            for sc in SCENARIOS:
                r = g[(g.dataset == ds) & (g.scenario == sc)]
                if r.empty:
                    continue
                r = r.iloc[0]
                b = [r[f"best@{x}"] for x in BUDGETS]
                bs = [f"{pct(1 - v, 0)}" if np.isfinite(v) else "--" for v in b]
                lab = ml if first else ""
                first = False
                L.append(f"{lab} & {NAMES[ds]} & {sc} & " + " & ".join(bs) +
                         f" & {pct(1 - r.knee_C, 0)} & {r.knee_D:.3f} & {r.knee_tau:.3f} & "
                         f"{pct(1 - r.min_C, 0)} & {r.min_D:.3f} & {r.min_tau:.3f} \\\\")
                key = f"{me}{ds}{sc}"
                N[f"kneered{key}"] = pct(1 - r.knee_C, 0)
                N[f"kneeD{key}"] = f"{r.knee_D:.3f}"
                N[f"kneetau{key}"] = f"{r.knee_tau:.3f}"
                N[f"minred{key}"] = pct(1 - r.min_C, 0)
                N[f"minD{key}"] = f"{r.min_D:.3f}"
                N[f"mintau{key}"] = f"{r.min_tau:.3f}"
                if "knee_shapes" in r and pd.notna(r.knee_shapes):
                    N[f"kneeshapes{key}"] = str(int(r.knee_shapes))
                    N[f"minshapes{key}"] = str(int(r.min_shapes))
                N[f"minhhi{key}"] = f"{r.min_hhi:.3f}"
                for x, v in zip(BUDGETS, b):
                    N[f"red{key}at{int(round(x * 100))}"] = pct(1 - v, 0) if np.isfinite(v) else "--"
        L.append("\\midrule")
    tab(L[:-1], "cal_results.tex")
    # aggregates
    for me in ["fragility", "event_rate", "cyclicity", "incomparability"]:
        for sc in SCENARIOS:
            g = df[(df.measure == me) & (df.scenario == sc)]
            for x in BUDGETS:
                v = 1 - g[f"best@{x}"]
                N[f"medred{me}{sc}at{int(round(x * 100))}"] = pct(np.median(v), 0)
                N[f"minred{me}{sc}at{int(round(x * 100))}"] = pct(v.min(), 0)
                N[f"maxred{me}{sc}at{int(round(x * 100))}"] = pct(v.max(), 0)
            N[f"kneetaumin{me}{sc}"] = f"{g.knee_tau.min():.3f}"
            N[f"kneeDmax{me}{sc}"] = f"{g.knee_D.max():.3f}"
            N[f"top1kept{me}{sc}"] = str(int(g.knee_top1.sum()))


def t_validate(N):
    df = pd.read_csv(RES / "validate.csv")
    L = []
    order = ["elicited", "centroid", "local-random", "knee", "best@0.02"]
    labels = {"elicited": "Elicited $\\theta^0$", "centroid": "Centroid", "local-random": "Local random",
              "knee": "Knee", "best@0.02": "Best, $\\Delta\\le 0.02$"}
    for ds in DATASETS:
        for sc in SCENARIOS:
            g = df[(df.dataset == ds) & (df.scenario == sc)].set_index("model")
            base = g.loc["elicited"]
            cells = []
            for mo in ["centroid", "local-random", "knee", "best@0.02"]:
                r = g.loc[mo]
                cells.append(f"{r['D']:.3f} & {pct(r['ins'], 3)} & {pct(r['sub'], 3)}")
            L.append(f"{NAMES[ds]} & {sc} & {pct(base['ins'], 3)} & {pct(base['sub'], 3)} & " + " & ".join(cells) + " \\\\")
    tab(L, "cal_validate.tex")
    piv = df.pivot_table(index=["dataset", "scenario"], columns="model", values=["ins", "sub"])
    for crit in ["ins", "sub"]:
        for mo in ["knee", "best@0.02", "local-random", "centroid"]:
            rel = piv[(crit, mo)] / piv[(crit, "elicited")]
            N[f"{crit}{mo}better"] = str(int((rel < 1).sum()))
            N[f"{crit}{mo}medchange"] = pct(np.median(1 - rel), 0)
    N["validcases"] = str(len(piv))


def f_fronts():
    Z = np.load(RES / "final_fronts.npz")
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.0), sharey="row")
    cols = {"W": "#1f5fa8", "WT": "#c2571a"}
    for j, ds in enumerate(DATASETS):
        for i, me in enumerate(["event_rate", "incomparability"]):
            ax = axes[i, j]
            for sc in SCENARIOS:
                F = Z[f"{ds}|{me}|{sc}|F"]
                F = F[np.argsort(F[:, 1])]
                ax.step(F[:, 1], F[:, 0], where="post", color=cols[sc], lw=1.4, label=sc)
                ax.plot(F[:, 1], F[:, 0], ".", color=cols[sc], ms=3)
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.03, 1.05)
            ax.grid(alpha=0.3, lw=0.5)
            if i == 0:
                ax.set_title(NAMES[ds], fontsize=10)
            if i == 1:
                ax.set_xlabel(r"deviation $\Delta$")
            if j == 0:
                ax.set_ylabel(("rank-reversal rate" if me == "event_rate" else "PROMETHEE I incomp.") + "\n(relative to $\\theta^0$)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fronts.pdf")
    plt.close(fig)


def f_hv():
    m = pd.read_csv(RES / "compare_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, me in zip(axes, ["event_rate", "incomparability"]):
        g = m[m.measure == me]
        g = g.assign(case=g.dataset.map(NAMES) + "-" + g.scenario)
        # HV normalised by the best mean HV of the case
        best = g.groupby("case").hv.transform("max")
        g = g.assign(rel=g.hv / best)
        data = [g[g.algorithm == a].rel.values for a in ALG_ORDER]
        ax.boxplot(data, tick_labels=ALG_ORDER, showfliers=False)
        ax.set_ylabel("HV / best HV of the case")
        ax.set_title("rank-reversal rate" if me == "event_rate" else "PROMETHEE I incomparability", fontsize=10)
        ax.grid(alpha=0.3, lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_hv.pdf")
    plt.close(fig)


def t_full(N):
    """Macros on the full preference-function scenario F."""
    from experiments.run_calibration import _cal
    df = pd.read_csv(RES / "calibrate.csv")
    raw = pd.read_csv(RES / "calibrate_before_nesting.csv") if (RES / "calibrate_before_nesting.csv").exists() else None
    Z = np.load(RES / "final_fronts.npz")
    for me in ["fragility", "event_rate", "cyclicity", "incomparability"]:
        g = df[df.measure == me].pivot(index="dataset", columns="scenario", values=["best@0.02", "best@0.1"])
        for b in ["0.02", "0.1"]:
            gain = 100 * (g[(f"best@{b}", "WT")] - g[(f"best@{b}", "F")])
            N[f"gainF{me}at{b.replace('0.', '')}"] = f"{np.median(gain):.0f}"
            N[f"gainFmax{me}at{b.replace('0.', '')}"] = f"{gain.max():.0f}"
        f = df[(df.measure == me) & (df.scenario == "F")]
        N[f"kneeshapessum{me}"] = str(int(f.knee_shapes.sum()))
        N[f"minshapessum{me}"] = str(int(f.min_shapes.sum()))
        N[f"minshapesmax{me}"] = str(int(f.min_shapes.max()))
        # share of the F front made of points with at least one Gaussian criterion
        tot = gau = 0
        for ds in DATASETS:
            _, cal = _cal(ds, me, "F")
            X = Z[f"{ds}|{me}|F|X"]
            ch = np.array([cal.shape_changes(x) for x in X])
            tot += len(ch); gau += int((ch > 0).sum())
        N[f"gaussshare{me}"] = pct(gau / tot, 0)
        if raw is not None:
            r = raw[(raw.measure == me) & (raw.scenario == "F")]
            N[f"fromWT{me}"] = pct((f.from_WT / f.n_front).mean(), 0) if "from_WT" in f else "--"


def make_all():
    N = Numbers()
    for f in (t_diagnose, t_compare, t_exact, t_calibrate, t_validate, t_full):
        try:
            f(N)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    for f in ():          # figures are produced by experiments.figures_cal
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    N.write()
    print(f"{len(N.d)} macros")


if __name__ == "__main__":
    make_all()
