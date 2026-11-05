"""Figure, table and number macros of the Monte Carlo robustness study (run_robustness.py)."""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.calibration_tables import NAMES, macro
from experiments.figures_cal import CAT, FULL, INK, INK2, MM, NEUTRAL, SC_COL, panel, save
from experiments.run_calibration import DATASETS, TAB
from experiments.run_robustness import RES

MIX = ["V", "mixed", "VI"]
MIX_LAB = {"V": "all Type V", "mixed": "mixed", "VI": "all Gaussian"}


def load():
    df = pd.read_csv(RES / "robustness.csv")
    for b in (0.01, 0.02, 0.05, 0.1):
        df[f"red@{b}"] = 1 - df[f"best@{b}"].clip(upper=1)        # reduction relative to theta0
    return df


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


def make_all():
    df = load()
    N = {}
    ok = df[df.C0 > 0]
    rr = ok[ok.measure == "event_rate"]
    kk = ok[ok.measure == "cyclicity"]
    N["robdraws"] = str(df.draw.nunique())
    N["robmodels"] = str(len(df[["dataset", "draw"]].drop_duplicates()))
    nz = df[(df.measure == "event_rate") & (df.scenario == "W")]
    N["robzeroRR"] = str(int((nz.C0 <= 0).sum()))
    # ---------------------------------------------------------------- reversal rate
    for sc in ("W", "WT", "F"):
        g = rr[rr.scenario == sc]
        for b in (0.01, 0.02, 0.05):
            N[f"robRR{sc}med{b}"] = pct(g[f"red@{b}"].median())
            N[f"robRR{sc}qlo{b}"] = pct(g[f"red@{b}"].quantile(0.25))
            N[f"robRR{sc}qhi{b}"] = pct(g[f"red@{b}"].quantile(0.75))
        N[f"robRR{sc}halfshare"] = pct((g["red@0.02"] >= 0.5).mean())
        N[f"robRR{sc}kneetaumed"] = f"{g.knee_tau.median():.3f}"
        N[f"robRR{sc}kneetauqlo"] = f"{g.knee_tau.quantile(0.05):.3f}"
        N[f"robRR{sc}kneeDmed"] = f"{g.knee_D.median():.3f}"
        N[f"robRR{sc}topone"] = pct(g.knee_top1.mean())
        N[f"robRR{sc}tauninetyfive"] = pct((g.knee_tau >= 0.95).mean())
    # by data set (W, within 0.02)
    for ds in DATASETS:
        g = rr[(rr.scenario == "W") & (rr.dataset == ds)]
        N[f"robRRWmedtwo{ds}"] = pct(g["red@0.02"].median())
    # ---------------------------------------------------------------- out of sample
    for sc in ("W", "F"):
        g = rr[rr.scenario == sc].dropna(subset=["ins_knee"])
        ri = g.ins_knee / g.ins_elicited
        rs = g.sub_knee / g.sub_elicited
        N[f"robins{sc}better"] = pct((ri < 1).mean())
        N[f"robsub{sc}better"] = pct((rs < 1).mean())
        N[f"robins{sc}med"] = pct(1 - ri.median())
        N[f"robsub{sc}med"] = pct(1 - rs.median())
    # ---------------------------------------------------------------- cyclicity and type changes
    for sc in ("W", "WT", "F"):
        g = kk[kk.scenario == sc]
        N[f"robK{sc}med"] = pct(g["red@0.1"].median())
        for mx in MIX:
            N[f"robK{sc}med{mx}"] = pct(g[g.mix == mx]["red@0.1"].median())
    gainK, gainR = [], []
    for (ds, dr), g in ok.groupby(["dataset", "draw"]):
        for me, store in (("cyclicity", gainK), ("event_rate", gainR)):
            h = g[g.measure == me].set_index("scenario")
            if {"WT", "F"} <= set(h.index):
                store.append(dict(mix=h.loc["F", "mix"], g01=h.loc["WT", "best@0.1"] - h.loc["F", "best@0.1"],
                                  g002=h.loc["WT", "best@0.02"] - h.loc["F", "best@0.02"],
                                  shapes=h.loc["F", "share_shapes"], kshapes=h.loc["F", "knee_shapes"]))
    gK, gR = pd.DataFrame(gainK), pd.DataFrame(gainR)
    for mx in MIX:
        N[f"robgainK{mx}"] = pct(gK[gK.mix == mx].g01.median(), 1)
        N[f"robgainKmax{mx}"] = pct(gK[gK.mix == mx].g01.max(), 0)
        N[f"robgainR{mx}"] = pct(gR[gR.mix == mx].g002.median(), 1)
        N[f"robshapesK{mx}"] = pct(gK[gK.mix == mx].shapes.mean())
    N["robgainRmax"] = pct(gR.g002.max(), 0)
    N["robgainKmax"] = pct(gK.g01.max(), 0)
    N["robkneeshapesR"] = pct((gR.kshapes > 0).mean())
    # ---------------------------------------------------------------- factors (Spearman)
    g = rr[rr.scenario == "W"]
    for f in ("scale", "q_ratio", "w_shift"):
        rho, p = spearmanr(g[f], g["red@0.02"])
        N[f"robrho{f}"] = f"{rho:.2f}"
        rho0, _ = spearmanr(g[f], g.C0)
        N[f"robrhoC{f}"] = f"{rho0:.2f}"
    gk = kk[kk.scenario == "W"]
    rho, _ = spearmanr(gk.scale, gk.C0)
    N["robrhoKscale"] = f"{rho:.2f}"
    write_macros(N)
    table(df, rr, kk)
    figure(rr, kk, gK)
    print(pd.Series(N).to_string())


def write_macros(N):
    with open(TAB / "numbers_rob.tex", "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\{macro(k)}}}{{{v}}}\n")


def table(df, rr, kk):
    L = []
    for ds in DATASETS:
        a = rr[(rr.dataset == ds)]
        w, t, fF = (a[a.scenario == s] for s in ("W", "WT", "F"))
        k = kk[kk.dataset == ds]
        kw, kt, kf = (k[k.scenario == s] for s in ("W", "WT", "F"))
        nW = len(df[(df.dataset == ds) & (df.measure == "event_rate") & (df.scenario == "W")])
        ins = (w.ins_knee < w.ins_elicited).mean()
        sub = (w.sub_knee < w.sub_elicited).mean()

        def q(s, b):
            return f"{pct(s[f'red@{b}'].median())} [{pct(s[f'red@{b}'].quantile(.25))}, {pct(s[f'red@{b}'].quantile(.75))}]"
        L.append(f"{NAMES[ds]} & {len(w)}/{nW} & {q(w, 0.01)} & {q(w, 0.02)} & {q(fF, 0.02)} & "
                 f"{w.knee_tau.min():.3f} & {pct(ins)} & {pct(sub)} & "
                 f"{pct(kw['red@0.1'].median())} & {pct(kt['red@0.1'].median())} & {pct(kf['red@0.1'].median())} \\\\")
    with open(TAB / "rob_summary.tex", "w") as f:
        f.write("\n".join(L) + "\n")


def figure(rr, kk, gK):
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 64 * MM), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    rng = np.random.default_rng(0)
    # a: reversal-rate reduction within Delta <= 0.02, per data set and scenario
    ax = axes[0]
    for i, ds in enumerate(DATASETS):
        for j, sc in enumerate(("W", "WT", "F")):
            v = 100 * rr[(rr.dataset == ds) & (rr.scenario == sc)]["red@0.02"].values
            x = i + (j - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=SC_COL[sc], alpha=0.45, lw=0)
            ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS])
    ax.set_ylabel(r"reversal-rate reduction within $\Delta\leq0.02$ (%)")
    ax.set_ylim(-3, 103)
    ax.grid(axis="x", visible=False)
    for sc in ("W", "WT", "F"):
        ax.scatter([], [], s=12, color=SC_COL[sc], label=sc)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.8,
              title="scenario", title_fontsize=7.5)
    panel(ax, "a", y=1.12)
    # b: cyclicity reduction within Delta <= 0.1 by elicited type mix
    ax = axes[1]
    for i, mx in enumerate(MIX):
        for j, sc in enumerate(("W", "WT", "F")):
            v = 100 * kk[(kk.mix == mx) & (kk.scenario == sc)]["red@0.1"].values
            x = i + (j - 1) * 0.26
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=SC_COL[sc], alpha=0.45, lw=0,
                       label=sc if i == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.1, columnspacing=0.8,
              markerscale=2, title="scenario", title_fontsize=7.5)
    ax.set_xticks(range(len(MIX)), [MIX_LAB[m] for m in MIX])
    ax.set_xlabel("elicited preference functions")
    ax.set_ylabel(r"cyclicity reduction within $\Delta\leq0.1$ (%)")
    ax.grid(axis="x", visible=False)
    panel(ax, "b", x=-0.2, y=1.12)
    # c: out of sample, knee of the reversal-rate front (scenario W)
    ax = axes[2]
    g = rr[rr.scenario == "W"].dropna(subset=["ins_knee"])
    for i, ds in enumerate(DATASETS):
        h = g[g.dataset == ds]
        for j, (crit, col, lab) in enumerate((("ins", CAT[0], "insertion"), ("sub", CAT[1], "sub-sampling"))):
            v = (h[f"{crit}_knee"] / h[f"{crit}_elicited"]).values
            x = i + (j - 0.5) * 0.3
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=4, color=col, alpha=0.5, lw=0,
                       label=lab if i == 0 else None)
            ax.plot([x - 0.11, x + 0.11], [np.median(v)] * 2, color=INK, lw=1.2)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(range(len(DATASETS)), [NAMES[d] for d in DATASETS])
    ax.set_ylabel("out-of-sample reversal rate,\nknee / elicited")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.1, markerscale=2,
              columnspacing=0.8)
    panel(ax, "c", x=-0.2, y=1.12)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            for h in leg.legend_handles:
                h.set_alpha(1)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_robustness.pdf")


if __name__ == "__main__":
    make_all()
