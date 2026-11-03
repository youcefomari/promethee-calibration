"""Benchmark datasets.

The manuscript uses five public benchmarks (``BENCHMARKS``), all previously used in Q1/Q2
MCDA or composite-indicator research:

========  =====================================================  ======  ========================
label     source                                                  m x n   weights
========  =====================================================  ======  ========================
HDI       UNDP Human Development Report 2025 (2023 data)          193x4   official (1/3,1/6,1/6,1/3)
WHR24     World Happiness Report 2024, explained components      144x6   equal
BLI       OECD Better Life Index 2017 (via R package Compind)     35x11   equal (OECD default)
ASEM      ASEM Sustainable Connectivity (JRC; R package COINr)    51x8    official hierarchy
HPI       Happy Planet Index 2019 (Compind) -- optional, unused   147x3   equal
ELECTRIC  Wieckowski et al. (2024), Artif. Intell. Rev.           19x15   elicited by the source
========  =====================================================  ======  ========================

All use linear (Type V) preference functions with q = 0 and p = population standard deviation,
a common convention for composite indices.

Files are read from ``<repository>/data`` (override with ``GPII_DATA_ROOT``); see
``data/public/README.md`` for provenance.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Dataset, Model

DATA_ROOT = Path(os.environ.get("GPII_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
BENCHMARKS = ["HDI", "WHR24", "BLI", "ASEM", "ELECTRIC"]
OPTIONAL = ["HPI"]   # Happy Planet Index: loader available, not used in the manuscript
ALL_NAMES = BENCHMARKS


def _std(X):
    return np.std(X, axis=0)  # population standard deviation (p_j = sigma_j)


def load_electric() -> Dataset:
    df = pd.read_excel(DATA_ROOT / "electric_v_selection.xlsx")
    crit = list(df.columns[1:])
    X = df[crit].to_numpy(float)
    mx = np.ones(15, bool)
    mx[[0, 13, 14]] = False
    w = np.array([0.129, 0.071, 0.009, 0.031, 0.084, 0.071, 0.040, 0.009, 0.049, 0.111,
                  0.022, 0.058, 0.093, 0.102, 0.120])
    return Dataset("ELECTRIC", df["Ai"].astype(str).tolist(), crit, X,
                   Model(mx, w, np.zeros(15), _std(X)), {"source": "Wieckowski et al. (2024)"})


def load_hdi() -> Dataset:
    df = pd.read_excel(DATA_ROOT / "public" / "HDI_2023.xlsx")
    crit = list(df.columns[1:5])
    X = df[crit].to_numpy(float)
    return Dataset("HDI", df["Country"].astype(str).tolist(), crit, X,
                   Model(np.ones(4, bool), np.array([1 / 3, 1 / 6, 1 / 6, 1 / 3]), np.zeros(4), _std(X)),
                   {"source": "UNDP HDR 2025 (2023 data)"})


def load_whr24() -> Dataset:
    df = pd.read_excel(DATA_ROOT / "public" / "WHR24.xlsx")
    crit = ["Explained by: Log GDP per capita", "Explained by: Social support",
            "Explained by: Healthy life expectancy", "Explained by: Freedom to make life choices",
            "Explained by: Generosity", "Explained by: Perceptions of corruption"]
    df = df.dropna(subset=crit).reset_index(drop=True)
    X = df[crit].to_numpy(float)
    return Dataset("WHR24", df["Country name"].astype(str).tolist(), crit, X,
                   Model(np.ones(6, bool), np.full(6, 1 / 6), np.zeros(6), _std(X)),
                   {"source": "World Happiness Report 2024"})


def load_bli() -> Dataset:
    df = pd.read_csv(DATA_ROOT / "public" / "BLI_2017.csv")
    crit = [c for c in df.columns if c != "country"]
    X = df[crit].to_numpy(float)
    n = len(crit)
    return Dataset("BLI", df["country"].tolist(), crit, X,
                   Model(np.ones(n, bool), np.full(n, 1 / n), np.zeros(n), _std(X)),
                   {"source": "OECD Better Life Index 2017 (Compind)"})


def load_asem() -> Dataset:
    """ASEM Sustainable Connectivity Index: 8 pillars built with the COINr default recipe.

    Indicators are denominated as declared in the metadata, min-max normalised to [0, 100]
    with their direction, and averaged within each pillar over the available values. Pillar
    weights reproduce the official hierarchy: Connectivity and Sustainability sub-indices
    weigh 1/2 each and are split equally among their pillars.
    """
    d = pd.read_csv(DATA_ROOT / "public" / "ASEM_iData.csv")
    meta = pd.read_csv(DATA_ROOT / "public" / "ASEM_iMeta.csv")
    ind = meta[meta.Level == 1]
    pillars = meta[meta.Level == 2]
    cols = {}
    for _, r in ind.iterrows():
        v = d[r.iCode].astype(float)
        if isinstance(r.Denominator, str) and r.Denominator in d:
            v = v / d[r.Denominator].astype(float)
        lo, hi = v.min(), v.max()
        z = (v - lo) / (hi - lo) * 100 if hi > lo else v * 0 + 50
        cols[r.iCode] = z if r.Direction > 0 else 100 - z
    norm = pd.DataFrame(cols)
    P, names, w = [], [], []
    for _, p in pillars.iterrows():
        members = ind[ind.Parent == p.iCode].iCode.tolist()
        P.append(norm[members].mean(axis=1, skipna=True).to_numpy())
        names.append(p.iCode)
        w.append(0.5 / (pillars.Parent == p.Parent).sum())
    X = np.column_stack(P)
    n = X.shape[1]
    return Dataset("ASEM", d["uName"].tolist(), names, X,
                   Model(np.ones(n, bool), np.array(w), np.zeros(n), _std(X)),
                   {"source": "ASEM Sustainable Connectivity Portal (JRC), COINr"})


def load_hpi(year: int = 2019) -> Dataset:
    df = pd.read_csv(DATA_ROOT / "public" / "HPI_2017_2019.csv")
    df = df[df.year == year].reset_index(drop=True)
    crit = ["Life_Expectancy", "Ladder_of_life", "Ecological_Footprint"]
    X = df[crit].to_numpy(float)
    return Dataset("HPI", df["Country"].tolist(), crit, X,
                   Model(np.array([True, True, False]), np.full(3, 1 / 3), np.zeros(3), _std(X)),
                   {"source": f"Happy Planet Index {year} (Compind)"})


LOADERS = {"ELECTRIC": load_electric,
           "HDI": load_hdi, "WHR24": load_whr24, "BLI": load_bli, "ASEM": load_asem,
           "HPI": load_hpi}


def load(name: str) -> Dataset:
    return LOADERS[name]()


def load_all() -> dict:
    return {k: LOADERS[k]() for k in ALL_NAMES}


