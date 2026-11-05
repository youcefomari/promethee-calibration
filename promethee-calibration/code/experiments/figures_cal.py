"""Publication figures of the calibration manuscript (vector PDF, journal style).

  fig_diagnosis.pdf   conflict profile of the elicited models
  fig_algorithms.pdf  comparison of the search methods
  fig_exact.pdf       exact certification of the evolutionary fronts
  fig_fronts.pdf      final fronts (reversal rate, cyclicity, incomparability)
  fig_tradeoff.pdf    conflict reduction versus ranking change; anatomy of a calibration
  fig_validation.pdf  out-of-sample rank reversal
  fig_summary.pdf     attainable reduction by measure and deviation budget

Usage:  python -m experiments.figures_cal          (after run_calibration)
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import kendalltau

from experiments.run_calibration import BUDGETS, DATASETS, FIG, MODE, RES, SCENARIOS, _cal
from gpii.calibration import knee_point, nondominated
from gpii.core import promethee
from gpii.data import load
from gpii.exact import model_size
from gpii.model import unicriterion

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e4e3df"
NEUTRAL = "#9a9993"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # validated, fixed order
DS_COL = dict(zip(DATASETS, CAT))
DS_MARK = dict(zip(DATASETS, ["o", "s", "^", "D", "v"]))
NAMES = {"HDI": "HDI", "WHR24": "WHR", "BLI": "BLI", "ASEM": "ASEM", "ELECTRIC": "EV"}
ALGS = ["NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random"]
ALG_COL = {"NSGA-II": CAT[0], "SMS-EMOA": CAT[1], "MOEA/D": CAT[2], "SPEA2": CAT[3], "Random": NEUTRAL}
ALG_MARK = {"NSGA-II": "o", "SMS-EMOA": "s", "MOEA/D": "^", "SPEA2": "D", "Random": "x"}
SC_COL = {"W": CAT[0], "WT": CAT[1], "F": CAT[2]}
SC_LAB = {"W": "weights (W)", "WT": "weights and thresholds (WT)", "F": "weights and preference functions (F)"}
MM = 1 / 25.4
FULL = 174 * MM           # text width of the manuscript

plt.rcParams.update({
    "font.family": "TeX Gyre Heros", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "mathtext.fontset": "custom", "mathtext.rm": "TeX Gyre Heros", "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def panel(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Cached auxiliary data
# ---------------------------------------------------------------------------
def front_points():
    """Front points with deviation, relative conflict, Kendall tau and top-1 retention."""
    path = RES / "front_points.csv"
    if path.exists():
        return pd.read_csv(path)
    Z = np.load(RES / "final_fronts.npz")
    rows = []
    for ds in DATASETS:
        d = load(ds)
        phi0 = promethee(d.X, d.model).phi
        for me in ["fragility", "event_rate", "cyclicity", "incomparability"]:
            for sc in SCENARIOS:
                _, cal = _cal(ds, me, sc)
                X, F = Z[f"{ds}|{me}|{sc}|X"], Z[f"{ds}|{me}|{sc}|F"]
                k = knee_point(F)
                for i, (x, f) in enumerate(zip(X, F)):
                    ph = promethee(d.X, cal.decode(x)).phi
                    rows.append(dict(dataset=ds, measure=me, scenario=sc, D=f[1], C=f[0],
                                     tau=kendalltau(phi0, ph)[0], top1=int(np.argmax(ph) == np.argmax(phi0)),
                                     knee=int(i == k)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def gaps():
    path = RES / "netflow_gaps.npz"
    if path.exists():
        return dict(np.load(path))
    out = {}
    for ds in DATASETS:
        d = load(ds)
        P = promethee(d.X, d.model).P
        r = P.sum(1)
        iu = np.triu_indices(d.m, 1)
        out[ds] = np.abs(r[iu[0]] - r[iu[1]])
    np.savez(path, **out)
    return out


def sizes():
    path = RES / "milp_sizes.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for ds in DATASETS:
        _, cal = _cal(ds, "event_rate", "W")
        for eps in [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]:
            s = model_size(cal, eps)
            rows.append(dict(dataset=ds, eps=eps, **s))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_diagnosis():
    diag = pd.read_csv(RES / "diagnose.csv").set_index("dataset").loc[DATASETS]
    G = gaps()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 46 * MM), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    # a: ECDF of scaled net-flow gaps |D|/2 (pairs left of 1 are candidates for a reversal)
    ax = axes[0]
    for ds in DATASETS:
        g = np.sort(G[ds] / 2.0)
        y = np.arange(1, len(g) + 1) / len(g)
        ax.step(g, y, where="post", color=DS_COL[ds], lw=1.2, label=NAMES[ds])
    ax.axvline(1, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"scaled net-flow gap  $|D(a,b)|/2$")
    ax.set_ylabel("cumulative share of pairs")
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1, fontsize=7, labelspacing=0.25)
    panel(ax, "a", x=-0.2, y=1.13)
    # b: fragility and reversal rate (dot plot, log)
    ax = axes[1]
    y = np.arange(len(DATASETS))[::-1]
    ax.hlines(y, diag.event_rate * 100, diag.fragility * 100, color=GRID, lw=2.2, zorder=1)
    ax.scatter(diag.fragility * 100, y, s=22, color=CAT[0], zorder=3, label="fragility Fr")
    ax.scatter(diag.event_rate * 100, y, s=22, facecolor="white", edgecolor=CAT[0], lw=1.1, zorder=3,
               label="reversal rate RR")
    ax.set_yticks(y, [NAMES[d] for d in DATASETS])
    ax.set_xscale("log")
    ax.set_xlabel("% (log scale)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.1, fontsize=7)
    panel(ax, "b", x=-0.3, y=1.13)
    # c: cyclicity K and reweighting bound
    ax = axes[2]
    kb = diag.K_bound.clip(lower=0)
    ax.hlines(y, kb, diag.cyclicity, color=CAT[1], lw=2.2, alpha=0.35, zorder=1)
    ax.scatter(diag.cyclicity, y, s=22, color=CAT[1], zorder=3, label=r"$K(\theta^0)$")
    ax.scatter(kb, y, s=22, marker="|", color=INK, lw=1.2, zorder=3, label=r"bound $K_{\min}$")
    ax.set_yticks(y, [NAMES[d] for d in DATASETS])
    ax.set_xlim(-0.005, 0.13)
    ax.set_xlabel("cyclicity")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.6,
              borderaxespad=0.1, fontsize=7)
    panel(ax, "c", x=-0.3, y=1.13)
    # d: PROMETHEE I incomparability and impact
    ax = axes[3]
    ax.barh(y, diag.incomparability * 100, height=0.55, color=CAT[2], label="incomparable pairs")
    for yi, v in zip(y, diag.incomparability * 100):
        ax.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [NAMES[d] for d in DATASETS])
    ax.set_xlabel("PROMETHEE I incomparability (%)")
    ax.set_xlim(0, 33)
    ax.grid(axis="y", visible=False)
    panel(ax, "d", x=-0.3, y=1.13)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_diagnosis.pdf")


def fig_algorithms():
    m = pd.read_csv(RES / "compare_metrics.csv")
    m["case"] = m.measure + "|" + m.scenario + "|" + m.dataset
    best = m.groupby("case").hv.transform("max")
    m["rel"] = m.hv / best
    fig = plt.figure(figsize=(FULL, 52 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.25, 1.0], wspace=0.62)
    # a: average Friedman rank with 95% bootstrap interval over cases
    ax = fig.add_subplot(gs[0])
    ranks = pd.DataFrame({c: g.pivot(index="seed", columns="algorithm", values="hv")[ALGS]
                          .rank(axis=1, ascending=False).mean() for c, g in m.groupby("case")}).T
    rng = np.random.default_rng(0)
    boot = np.array([ranks.iloc[rng.integers(0, len(ranks), len(ranks))].mean().values for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    mean = ranks.mean().values
    order = np.argsort(mean)
    yy = np.arange(len(ALGS))[::-1]
    for pos, i in zip(yy, order):
        a = ALGS[i]
        ax.hlines(pos, lo[i], hi[i], color=ALG_COL[a], lw=1.6, alpha=0.55)
        ax.scatter(mean[i], pos, s=28, color=ALG_COL[a], marker=ALG_MARK[a], zorder=3,
                   linewidths=1.2 if a == "Random" else 0)
        ax.text(lo[i] - 0.1, pos, f"{mean[i]:.2f}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_yticks(yy, [ALGS[i] for i in order])
    ax.set_xlim(1, 5.4)
    ax.invert_xaxis()
    ax.set_xlabel(f"average rank (hypervolume), {len(ranks)} cases")
    ax.grid(axis="y", visible=False)
    panel(ax, "a", x=-0.36)
    # b: distribution of HV relative to the best mean of the case
    ax = fig.add_subplot(gs[1])
    for i, a in enumerate(ALGS):
        v = m[m.algorithm == a].groupby("case").rel.mean().values
        jit = (rng.random(len(v)) - 0.5) * 0.35
        ax.scatter(np.full(len(v), i) + jit, v, s=7, color=ALG_COL[a], alpha=0.75, linewidths=0,
                   marker=ALG_MARK[a] if a != "Random" else "o")
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.hlines(med, i - 0.3, i + 0.3, color=INK, lw=1.3, zorder=4)
        ax.vlines(i, q1, q3, color=INK, lw=0.7, zorder=4)
    ax.set_xticks(range(len(ALGS)), ALGS, rotation=30, ha="right")
    ax.set_ylabel("hypervolume / best mean of the case")
    ax.set_ylim(0.6, 1.02)
    ax.grid(axis="x", visible=False)
    panel(ax, "b", x=-0.22)
    # c: within-case Mann-Whitney tests of NSGA-II against each competitor, Holm-corrected over all
    #    120 tests (results/revision/stats_cases.csv, written by experiments.revision_stats)
    st = pd.read_csv(RES.parent / "revision" / "stats_cases.csv")
    ax = fig.add_subplot(gs[2])
    res = {}
    for a in ALGS[1:]:
        g = st[st.other == a]
        w = int(((g.p_holm < 0.05) & (g.med_diff > 0)).sum())
        l = int(((g.p_holm < 0.05) & (g.med_diff < 0)).sum())
        res[a] = (w, len(g) - w - l, l)
    yy = np.arange(len(res))[::-1]
    for pos, (a, (w, t, l)) in zip(yy, res.items()):
        ax.barh(pos, w, color=CAT[0], height=0.6, edgecolor="white", lw=0.8)
        ax.barh(pos, t, left=w, color=GRID, height=0.6, edgecolor="white", lw=0.8)
        ax.barh(pos, l, left=w + t, color=CAT[1], height=0.6, edgecolor="white", lw=0.8)
        ax.text(w / 2, pos, str(w), ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        if t:
            ax.text(w + t / 2, pos, str(t), ha="center", va="center", fontsize=7, color=INK2)
    ax.set_yticks(yy, list(res))
    ax.set_xlim(0, m.case.nunique())
    ax.set_xlabel("cases (Mann-Whitney, Holm, p < 0.05)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CAT[0]), plt.Rectangle((0, 0), 1, 1, color=GRID),
                       plt.Rectangle((0, 0), 1, 1, color=CAT[1])],
              labels=["NSGA-II better", "no difference", "NSGA-II worse"], loc="lower center",
              bbox_to_anchor=(0.4, 1.0), ncol=3, handlelength=0.9, columnspacing=0.6, handletextpad=0.3,
              fontsize=7)
    panel(ax, "c", x=-0.4, y=1.1)
    save(fig, "fig_algorithms.pdf")


def fig_exact():
    ex = pd.read_csv(RES / "exact.csv")
    S = sizes()
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 46 * MM))
    eps = np.array(sorted(ex.eps.unique()))
    for ax, ds, letter in zip(axes[:3], ["ELECTRIC", "BLI", "ASEM"], "abc"):
        g = ex[ex.instance == ds].set_index("eps").loc[eps]
        ax.plot(eps, g.exact, color=INK, lw=1.3, marker="o", ms=4, mfc="white", mew=1.1, zorder=4,
                label="exact (MILP)")
        off = np.linspace(-0.06, 0.06, 4)
        for o, a in zip(off, ALGS[:4]):
            ax.scatter(eps * (1 + o), g[f"best_{a}"], s=14, color=ALG_COL[a], marker=ALG_MARK[a], zorder=3,
                       linewidths=0, label=a)
        ax.set_xscale("log")
        ax.set_xticks(eps, ["0.005", "0.01", "0.02", "0.05"])
        ax.minorticks_off()
        ax.set_ylim(-0.04, 1.0)
        ax.set_xlabel(r"weight budget $\varepsilon$")
        k = int(load(ds).X.shape[1]) if MODE[ds] == "free" else _cal(ds, "event_rate", "W")[1].ws.k
        ax.set_title(f"{NAMES[ds]}  (k = {k})", loc="left", fontsize=8.5)
        panel(ax, letter, x=-0.25 if letter == "a" else -0.12)
        if letter == "a":
            ax.set_ylabel("best relative reversal rate")
    axes[1].legend(loc="upper right", fontsize=7, handletextpad=0.2, borderaxespad=0.1)
    # d: MILP size against the budget; filled = solved to optimality within the time limit
    ax = axes[3]
    solved = {("HDI", 0.005)} | {(d_, e) for d_ in ["ELECTRIC", "BLI", "ASEM"] for e in [0.005, 0.01, 0.02, 0.05]}
    for ds in DATASETS:
        g = S[(S.dataset == ds) & (S.eps.isin([0.005, 0.01, 0.02, 0.05]))].sort_values("eps")
        ax.plot(g.eps, g.binaries, color=DS_COL[ds], lw=1.0, label=NAMES[ds], zorder=2)
        for e, b in zip(g.eps, g.binaries):
            ok = (ds, round(e, 4)) in solved
            ax.scatter(e, b, s=16, marker=DS_MARK[ds], facecolor=DS_COL[ds] if ok else "white",
                       edgecolor=DS_COL[ds], lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.005, 0.01, 0.02, 0.05], ["0.005", "0.01", "0.02", "0.05"])
    ax.minorticks_off()
    ax.set_ylim(80, 4e6)
    ax.set_xlabel(r"weight budget $\varepsilon$")
    ax.set_ylabel("binaries after pruning")
    ax.legend(loc="upper left", fontsize=7, handlelength=0.9, ncol=3, borderaxespad=0.1, columnspacing=0.5,
              handletextpad=0.3)
    panel(ax, "d", x=-0.32)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_exact.pdf")


def fig_fronts():
    Z = np.load(RES / "final_fronts.npz")
    fp = front_points()
    rows = [("event_rate", "reversal rate"), ("cyclicity", "cyclicity"), ("incomparability", "incomparability")]
    fig, axes = plt.subplots(3, 5, figsize=(FULL, 115 * MM), sharey=True, sharex=True)
    for i, (me, lab) in enumerate(rows):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            for sc in SCENARIOS:
                F = Z[f"{ds}|{me}|{sc}|F"]
                F = F[np.argsort(F[:, 1])]
                x = np.concatenate([[0.0], F[:, 1], [1.0]])
                y = np.concatenate([[1.0], F[:, 0], [F[-1, 0]]])
                ax.step(np.maximum(x, 1e-4), y, where="post", color=SC_COL[sc], lw=1.2)
                kn = fp[(fp.dataset == ds) & (fp.measure == me) & (fp.scenario == sc) & (fp.knee == 1)]
                if len(kn):
                    ax.scatter(max(kn.D.iloc[0], 1e-4), kn.C.iloc[0], s=22, facecolor="white",
                               edgecolor=SC_COL[sc], lw=1.2, zorder=5)
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 1)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 1], ["0.001", "0.01", "0.1", ""])
            ax.minorticks_off()
            if i == 0:
                ax.set_title(NAMES[ds], fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"relative\n{lab}")
            if i == 2:
                ax.set_xlabel(r"deviation $\Delta$")
    handles = [Line2D([], [], color=SC_COL[sc], lw=1.2) for sc in SCENARIOS] + \
              [Line2D([], [], color=INK2, marker="o", mfc="white", lw=0, ms=4)]
    fig.legend(handles, [SC_LAB[sc] for sc in SCENARIOS] + ["knee"], loc="upper center",
               ncol=4, bbox_to_anchor=(0.5, 1.02), handlelength=1.5)
    for i, letter in enumerate("abc"):
        panel(axes[i, 0], letter, x=-0.62, y=0.98)
    fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0, 1, 0.97))
    save(fig, "fig_fronts.pdf")


def fig_tradeoff():
    fp = front_points()
    fig = plt.figure(figsize=(FULL, 52 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.38)
    for col, (me, lab, letter) in enumerate([("event_rate", "reversal rate", "a"),
                                             ("incomparability", "incomparability", "b")]):
        ax = fig.add_subplot(gs[col])
        g = fp[(fp.measure == me) & (fp.scenario == "F")]
        for ds in DATASETS:
            h = g[g.dataset == ds].sort_values("C")
            ax.plot(1 - h.C, h.tau, color=DS_COL[ds], lw=1.0, marker=DS_MARK[ds], ms=2.5, mew=0,
                    label=NAMES[ds], alpha=0.95)
        ax.axhline(0, color=INK3, lw=0.6)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.4, 1.02)
        ax.set_xlabel(f"reduction of the {lab}")
        if col == 0:
            ax.set_ylabel(r"Kendall $\tau$ with the elicited ranking")
            ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.2)
        ax.set_title(f"{lab}, scenario F", loc="left", fontsize=8.5)
        panel(ax, letter, x=-0.2 if col == 0 else -0.12)
    # c: anatomy of the EV calibration (reversal rate, W): weight shifts at the best point within 0.02
    ax = fig.add_subplot(gs[2])
    Z = np.load(RES / "final_fronts.npz")
    X, F = Z["ELECTRIC|event_rate|W|X"], Z["ELECTRIC|event_rate|W|F"]
    ok = F[:, 1] <= 0.02 + 1e-12
    i = np.nonzero(ok)[0][np.argmin(F[ok, 0])]
    d, cal = _cal("ELECTRIC", "event_rate", "W")
    w0, w1 = d.model.w, cal.decode(X[i]).w
    order = np.argsort(-w0)
    y = np.arange(len(w0))[::-1]
    ax.hlines(y, 0, w0[order], color=GRID, lw=2.5)
    ax.scatter(w0[order], y, s=14, color=INK3, zorder=3, label=r"elicited $w^0$")
    ax.scatter(w1[order], y, s=14, color=CAT[0], zorder=4, marker="D", label=r"calibrated $w^\star$")
    ax.set_yticks(y, [f"C{j + 1}" for j in order], fontsize=7)
    ax.set_xlabel("weight")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    ax.set_title(f"EV: RR {100 * (1 - F[i, 0]):.0f}% lower, "
                 rf"$\Delta_w$ = {F[i, 1]:.3f}", loc="left", fontsize=8.5)
    panel(ax, "c", x=-0.22)
    save(fig, "fig_tradeoff.pdf")


def fig_validation():
    v = pd.read_csv(RES / "validate.csv")
    models = [("centroid", "centroid of the\nadmissible set", NEUTRAL, "s"),
              ("local-random", "random move,\nsame deviation", INK3, "x"),
              ("knee", "knee", CAT[0], "o"),
              ("best@0.02", r"best within $\Delta\leq0.02$", CAT[1], "D")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.78, 76 * MM), sharey=True)
    cases = [(ds, sc) for ds in DATASETS for sc in SCENARIOS]
    y = np.arange(len(cases))[::-1]
    for ax, crit, title, letter in zip(axes, ["ins", "sub"], ["insertion of 10% new",
                                                              "sub-samples of 80%"], "ab"):
        for k, (mo, lab, col, mk) in enumerate(models):
            vals = []
            for ds, sc in cases:
                g = v[(v.dataset == ds) & (v.scenario == sc)].set_index("model")
                vals.append(g.loc[mo, crit] / g.loc["elicited", crit])
            off = (k - 1.5) * 0.16
            ax.scatter(vals, y + off, s=13, color=col, marker=mk, zorder=3,
                       linewidths=1.0 if mk == "x" else 0, label=lab)
        ax.axvline(1, color=INK2, lw=0.8)
        ax.set_yticks(y, [f"{NAMES[ds]} {sc}" for ds, sc in cases])
        ns = len(SCENARIOS)
        for b in range(0, len(cases), 2 * ns):
            ax.axhspan(len(cases) - b - ns - 0.5, len(cases) - b - 0.5, color="#f4f3f0", lw=0, zorder=0)
        ax.set_ylim(-0.6, len(cases) - 0.4)
        ax.set_xlabel("reversal rate / elicited")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="y", visible=False)
        panel(ax, letter, x=-0.25 if letter == "a" else -0.06)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.3, labelspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig_validation.pdf")


def fig_summary():
    c = pd.read_csv(RES / "calibrate.csv")
    meas = [("fragility", "fragility"), ("event_rate", "reversal rate"), ("cyclicity", "cyclicity"),
            ("incomparability", "incomparability")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 40 * MM), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3f0", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for ax, sc, letter in zip(axes, SCENARIOS, "abc"):
        M = np.array([[np.median(1 - c[(c.measure == me) & (c.scenario == sc)][f"best@{b}"]) for b in BUDGETS]
                      for me, _ in meas]) * 100
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 55 else INK)
        ax.set_xticks(range(len(BUDGETS)), [f"{b:g}" for b in BUDGETS])
        ax.set_yticks(range(len(meas)), [l for _, l in meas])
        ax.set_xlabel(r"deviation budget $\Delta$")
        ax.set_title(SC_LAB[sc], loc="center", fontsize=8.5)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        panel(ax, letter, x=-0.42 if letter == "a" else -0.05)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_summary.pdf")


def fig_procedure():
    """Schematic of the five-stage procedure."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(FULL, 56 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8,
                                     color=INK2, shrinkA=0, shrinkB=0))

    light = {"blue": "#e8f1fc", "orange": "#fdeee6", "gray": "#f4f3f0"}
    W_, H_ = 17.6, 16.0
    xs = [0.5, 20.6, 40.7, 60.8, 80.9]
    y = 16.5
    box(xs[0], y, W_, H_, "1  Diagnose",
        "Fr, RR, K, Inc at $\\theta^0$\nreweighting bound $K_{\\min}$\nstability radius\nwhich measure, which\nscenario", light["gray"], INK3)
    box(xs[1], y, W_, H_, "2  Explore",
        "NSGA-II on $(\\bar\\kappa,\\,\\Delta)$\nencoding $\\lambda\\mapsto V\\lambda$\nevery candidate\nadmissible\nscenarios W, WT, F", light["blue"], CAT[0])
    box(xs[2], y, W_, H_, "3  Certify, refine",
        "pruned exact MILP\nexact points on a\nbudget grid in W\nweights re-optimised\nin WT and F", light["orange"], CAT[1])
    box(xs[3], y, W_, H_, "4  Select",
        "merged Pareto front\nknee of the front, or\nleast conflict within\nthe tolerance $\\bar\\Delta$", light["gray"], INK3)
    box(xs[4], y, W_, H_, "5  Validate",
        "insertions and\nsub-samples not used\nin the calibration\nKendall $\\tau$, top-1", light["gray"], INK3)
    for i in range(4):
        arrow(xs[i] + W_ + 0.45, y + H_ / 2, xs[i + 1] - 0.45, y + H_ / 2)
    notes = ["exact, cheap\nconflict measures\n$\\Rightarrow$ search affordable",
             "step-function\nobjectives\n$\\Rightarrow$ no gradients",
             "linear in $w$, sign\nradius\n$\\Rightarrow$ small exact models",
             "degenerate\nminimisers\n$\\Rightarrow$ trade-off, front",
             "leave-one-out\nobjective\n$\\Rightarrow$ out-of-sample check"]
    for x, t in zip(xs, notes):
        ax.text(x + W_ / 2, 13.6, t, ha="center", va="top", fontsize=7, color=INK3, linespacing=1.35)
    ax.plot([0.5, 98.5], [14.6, 14.6], color=GRID, lw=0.6)
    ax.text(0.5, 1.2, "Inputs are the evaluations $g_j$, elicited parameters $\\theta^0=(w^0,\\mathbf{0},\\rho^0,\\gamma^0)$, ordinal weight "
            "statement, threshold ratio $\\beta$, conflict measure $\\kappa$.", fontsize=5.9, color=INK2)
    save(fig, "fig_procedure.pdf")


def main():
    for f in (fig_procedure, fig_diagnosis, fig_algorithms, fig_exact, fig_fronts, fig_tradeoff, fig_validation,
              fig_summary):
        f()


if __name__ == "__main__":
    main()
