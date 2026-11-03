"""Additional figures of the C&OR manuscript, written to paper/figures.

  fig_scaling.pdf    cost of the exact evaluation and of the exact model, and gap of the evolutionary
                     search to the exact optimum (synthetic benchmark and random instances)
  fig_mechanism.pdf  stability radii of reversal events and of pairwise orders (why small budgets work)
  fig_budget.pdf     effect of growing budgets on reversals, ranking and cyclicity

Usage: python -m experiments.figures_cor
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.figures_cal import CAT, DS_COL, DS_MARK, FULL, INK, INK2, INK3, MM, NAMES, NEUTRAL, panel
from experiments.run_calibration import DATASETS, ROOT, _cal
from gpii.exact import _structures

OUT = ROOT / "paper" / "figures"
REV = ROOT / "results" / "revision"
SYN = ROOT / "results" / "synthetic"


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


def fig_scaling():
    s = pd.read_csv(REV / "scaling_synthetic.csv")
    d = pd.read_csv(SYN / "synthetic.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 54 * MM))
    # a: time per evaluation, exact versus re-ranking
    ax = axes[0]
    for n, ls in ((4, "-"), (16, "--")):
        g = s[s.n == n]
        ax.plot(g.m, 1e3 * g.t_eval, ls, color=CAT[0], marker="o", ms=3, label=f"exact, n = {n}")
        h = g.dropna(subset=["t_rerank"])
        ax.plot(h.m, 1e3 * h.t_rerank, ls, color=CAT[1], marker="s", ms=3, label=f"re-ranking, n = {n}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("alternatives m"); ax.set_ylabel("time per evaluation (ms)")
    ax.legend(loc="lower right", fontsize=6.6, handlelength=1.8)
    panel(ax, "a", x=-0.24)
    # b: size of the exact model
    ax = axes[1]
    pos = d[d.RR0 > 0]
    rng = np.random.default_rng(1)
    for m, g in pos.groupby("m"):
        x = m * np.exp(rng.uniform(-0.08, 0.08, len(g)))
        st = g.ex_status
        ax.scatter(x[st == 0], g.bin_002[st == 0], s=9, color=CAT[2], lw=0, label="optimal" if m == 20 else None)
        ax.scatter(x[st == 1], g.bin_002[st == 1], s=11, facecolor="none", edgecolor=CAT[1], lw=0.8,
                   label="time limit" if m == 20 else None)
        ax.scatter(x[st.isna()], g.bin_002[st.isna()], s=9, color=NEUTRAL, marker="x", lw=0.7,
                   label="not attempted" if m == 100 else None)
    ax.axhline(5000, color=INK3, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([20, 50, 100], ["20", "50", "100"]); ax.minorticks_off()
    ax.set_xlabel("alternatives m"); ax.set_ylabel("binary variables, $\\varepsilon$ = 0.02")
    ax.legend(loc="upper left", fontsize=6.8, handletextpad=0.2)
    panel(ax, "b", x=-0.26)
    # c: gap of the evolutionary search to the exact optimum
    ax = axes[2]
    ok = pos[(pos.ex_status == 0)].dropna(subset=["ex_red"])
    gap = 100 * (ok.ex_red - ok.evo_red)
    for i, (lab, g) in enumerate(ok.assign(gap=gap).groupby("ties")):
        v = np.sort(g.gap.values)
        ax.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where="post",
                color=CAT[i], label=f"{lab} values ({len(v)})")
    ax.set_xlabel("gap to the exact optimum (points)"); ax.set_ylabel("share of instances")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=6.8)
    panel(ax, "c", x=-0.22)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_scaling.pdf")


def radii(name):
    _, cal = _cal(name, "event_rate", "W")
    Pa, rj = _structures(cal)
    w0 = cal.model0.w
    n, m, _ = Pa.shape
    iu = np.triu_indices(m, 1)
    Dj = rj[:, iu[0]] - rj[:, iu[1]]
    D0 = w0 @ Dj
    span = np.ptp(Dj, 0)
    strict = np.abs(D0) > 1e-10
    pair_r = np.abs(D0[strict]) / np.where(span[strict] > 0, span[strict], np.inf)
    ev_r = []
    for i in np.nonzero(strict & (np.abs(D0) <= 2))[0]:
        a, b = iu[0][i], iu[1][i]
        if D0[i] < 0:
            a, b = b, a
        xs = np.setdiff1d(np.arange(m), [a, b])
        d = rj[:, a] - rj[:, b]
        Ej = d[:, None] - Pa[:, a, xs] - Pa[:, xs, b]
        E0 = w0 @ Ej
        br = E0 <= 1e-12
        sp = np.ptp(Ej, 0)
        ev_r += list(np.abs(E0[br]) / np.where(sp[br] > 0, sp[br], np.inf))
    return pair_r, np.array(ev_r)


def fig_mechanism():
    N = {}
    ev02, pr02, evmax = [], [], []
    for name in DATASETS:
        pr, er = radii(name)
        ev02.append((er <= 0.02).mean()); pr02.append((pr <= 0.02).mean()); evmax.append(np.quantile(er, 0.9))
    N["mechevshare"] = f"{100 * min(ev02):.0f}"
    N["mechprlo"] = f"{100 * min(pr02):.1f}"; N["mechprhi"] = f"{100 * max(pr02):.0f}"
    N["mechevninety"] = f"{max(evmax):.3f}"
    with open(ROOT / "paper" / "tables" / "numbers_fig.tex", "w") as fh:
        for k, v in N.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(N)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 60 * MM))
    grid = np.logspace(-4, 0, 200)
    for name in DATASETS:
        pr, er = radii(name)
        c = DS_COL[name]
        axes[0].plot(grid, [(er <= g).mean() for g in grid], color=c, label=NAMES[name])
        axes[0].plot(grid, [(pr <= g).mean() for g in grid], color=c, ls="--", lw=1.0)
    ax = axes[0]
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.axvspan(1e-4, 0.02, color=CAT[2], alpha=0.07, lw=0)
    ax.text(0.0014, 1.05, "budgets up to 2%", fontsize=6.8, color=INK2, ha="center", va="bottom")
    ax.set_xlabel("stability radius (share of weight to move)")
    ax.set_ylabel("cumulative share")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=INK2), Line2D([], [], color=INK2, ls="--", lw=1.0)],
              ["reversal events", "pairwise orders"], loc="lower right", fontsize=6.8, bbox_to_anchor=(1.0, 0.25))
    panel(ax, "a", x=-0.16, y=1.04)
    # b: what a budget buys
    d = pd.read_csv(REV / "sideeffects.csv")
    order = ["best@0.02", "best@0.05", "best@0.1", "best@0.2", "best@0.5"]
    xs = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax = axes[1]
    for name in DATASETS:
        g = d[(d.dataset == name) & (d.scenario == "W")].set_index("point")
        rr = [g.loc[p, "RR"] / g.loc[p, "RR0"] for p in order]
        tau = [g.loc[p, "tau"] for p in order]
        ax.plot(xs, rr, color=DS_COL[name], marker=DS_MARK[name], ms=3.5, label=NAMES[name])
        ax.plot(xs, tau, color=DS_COL[name], marker=DS_MARK[name], ms=3.5, ls=":", lw=1.0, mfc="white")
    ax.set_xscale("log"); ax.set_xticks(xs, ["0.02", "0.05", "0.1", "0.2", "0.5"]); ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("budget on the moved weight (scenario W)")
    ax.set_ylabel("relative RR (solid) and $\\tau$ (dotted)")
    ax.legend(loc="center right", fontsize=6.8, bbox_to_anchor=(1.0, 0.55))
    panel(ax, "b", x=-0.2, y=1.04)
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig_mechanism.pdf")


if __name__ == "__main__":
    fig_scaling()
    fig_mechanism()
