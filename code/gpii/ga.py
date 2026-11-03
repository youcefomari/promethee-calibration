"""Genetic-PROMETHEE II: a genetic algorithm over the PROMETHEE parameters.

This is a corrected and reproducible rewrite of the original notebook. Two uses are
supported, selected by ``objective``:

``"incomparability"``  the original Genetic-PROMETHEE II objective: the share of pairs
                       declared incomparable by a source (PI or gamma).
``"cyclicity"``        the cyclicity index K (manuscript, Section 3);
``"reversal"``         the exact leave-one-out rank-reversal rate of PROMETHEE II, computed
                       from Theorem 4 without re-ranking (O(m^3)).

and both can be regularised by the deviation from the elicited parameters,

    fitness(theta) = objective(theta) + lam * Delta(theta, theta0) + mu * violations(theta),

(with ``relative=True`` the objective is divided by its value at theta0, so that lam is
comparable across datasets),

where ``violations`` measures the breach of the declared weight order (strict pairs and,
new here, ties). the degeneracy results of the previous draft show that both objectives have
trivial minimisers when lam = 0 (a single dominant criterion for the incomparability count;
unsaturated linear preference functions for K), so lam > 0 is required for a meaningful
calibration; the experiments sweep lam to trace the trade-off ("minimal adjustment").

Corrections with respect to the original notebook
-------------------------------------------------
1. preference functions were never applied (string/int comparison) -- fixed in
   ``preference.py``; thresholds and types now affect the objective;
2. the U-shape used a data value as threshold -- fixed;
3. q > p was penalised with an arbitrary constant; it is now repaired by sorting (q, p);
4. the incomparability count was doubled (each pair counted for both alternatives);
5. an AHP matrix was applied to data with different criteria -- removed;
6. weight ties in the elicited order were ignored; they can now be enforced;
7. global ``np.random`` state and per-evaluation ``print`` replaced by a seeded Generator
   and an optional history; everything is vectorised (feasible on m = 650).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import Source, incomparability_count, promethee
from .cycles import cyclicity_index
from .model import Model, unicriterion
from .preference import CODE_TO_INDEX, INDEX_TO_CODE


@dataclass
class GAConfig:
    pop_size: int = 40
    generations: int = 40
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1      # Gaussian sd as a fraction of the gene range
    eta_c: float = 15.0              # SBX distribution index
    elitism: int = 2
    seed: int = 0


@dataclass
class SearchSpace:
    optimize_weights: bool = True
    optimize_thresholds: bool = False
    optimize_types: bool = False
    threshold_upper: np.ndarray | None = None   # per-criterion upper bound for q and p
    enforce_order: bool = True
    enforce_ties: bool = True
    tie_tol: float = 1e-3                        # |w_i - w_j| allowed for elicited ties
    penalty_factor: float = 100.0


@dataclass
class GAResult:
    theta: np.ndarray
    model: Model
    fitness: float
    objective: float
    deviation: float
    violation: float
    history: list = field(default_factory=list)
    population: np.ndarray | None = None


class GeneticPromethee:
    def __init__(self, X: np.ndarray, model0: Model, objective: str = "incomparability",
                 source: Source = "PI", lam: float = 0.0, space: SearchSpace | None = None,
                 config: GAConfig | None = None, gamma_params: dict | None = None,
                 relative: bool = False):
        self.X = np.asarray(X, float)
        self.model0 = model0
        self.n = model0.n
        self.objective_name = objective
        self.source = source
        self.lam = lam
        self.space = space or SearchSpace()
        self.cfg = config or GAConfig()
        self.gamma_params = gamma_params or {}
        self.relative = relative
        self._phi0 = None
        self.rng = np.random.default_rng(self.cfg.seed)
        self.range = np.ptp(self.X, axis=0)
        self.range[self.range == 0] = 1.0
        up = self.space.threshold_upper
        self.th_up = self.range.copy() if up is None else np.asarray(up, float)
        self.theta0 = self.encode(model0)
        self._order_pairs, self._tie_pairs = self._weight_order(model0.w)
        # cache of unicriterion preferences when only the weights change
        self._Pj = None if (self.space.optimize_thresholds or self.space.optimize_types) \
            else unicriterion(model0, self.X, self.X)
        n = self.n
        lo = np.concatenate([np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)])
        hi = np.concatenate([np.ones(n), self.th_up, self.th_up, np.full(n, 5.0)])
        self.lo, self.hi = lo, hi
        self.free = np.concatenate([
            np.full(n, self.space.optimize_weights), np.full(2 * n, self.space.optimize_thresholds),
            np.full(n, self.space.optimize_types)])

    # ------------------------------------------------------------------ encoding
    def encode(self, model: Model) -> np.ndarray:
        t = np.array([CODE_TO_INDEX[c] for c in model.types], float)
        return np.concatenate([model.w, model.q, model.p, t])

    def decode(self, theta: np.ndarray) -> Model:
        n = self.n
        w = np.clip(theta[:n], 1e-9, None)
        q, p = theta[n:2 * n], theta[2 * n:3 * n]
        q, p = np.minimum(q, p), np.maximum(q, p)                   # repair q <= p
        t = np.clip(np.rint(theta[3 * n:]), 0, 5).astype(int)
        return self.model0.with_(w=w / w.sum(), q=q, p=p, types=tuple(INDEX_TO_CODE[i] for i in t))

    @staticmethod
    def _weight_order(w0, tol=1e-6):
        n = len(w0)
        strict = [(i, j) for i in range(n) for j in range(n) if w0[i] - w0[j] > tol]
        ties = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(w0[i] - w0[j]) <= tol]
        return strict, ties

    # ------------------------------------------------------------------ fitness
    def deviation(self, model: Model) -> float:
        """Delta = total variation of weights + mean relative threshold change + share of type changes."""
        m0 = self.model0
        dw = 0.5 * np.abs(model.w - m0.w).sum()
        dth = np.mean((np.abs(model.q - m0.q) + np.abs(model.p - m0.p)) / self.range)
        dty = np.mean([a != b for a, b in zip(model.types, m0.types)])
        return float(dw + dth + dty)

    def violation(self, w: np.ndarray) -> float:
        v = 0.0
        if self.space.enforce_order:
            v += sum(max(0.0, w[j] - w[i]) for i, j in self._order_pairs)
        if self.space.enforce_ties:
            v += sum(max(0.0, abs(w[i] - w[j]) - self.space.tie_tol) for i, j in self._tie_pairs)
        return float(v)

    def objective(self, model: Model) -> float:
        if self._Pj is not None:
            from .core import _from_Pj
            pr = _from_Pj(self.X, model, self._Pj)
        else:
            pr = promethee(self.X, model)
        if self.objective_name == "incomparability":
            m = pr.m
            return incomparability_count(pr, self.source, **self.gamma_params) / (m * (m - 1) / 2)
        if self.objective_name == "cyclicity":
            return cyclicity_index(pr.P)
        if self.objective_name == "reversal":
            from .reversal import loo_reversal_rate
            return loo_reversal_rate(pr.P)
        raise ValueError(self.objective_name)

    def scaled_objective(self, model: Model) -> float:
        """Objective, divided by its value at the elicited parameters when ``relative`` is set,
        so that lam has the same meaning across datasets of different sizes."""
        obj = self.objective(model)
        if not self.relative:
            return obj
        if self._phi0 is None:
            self._phi0 = self.objective(self.model0)
        return obj / self._phi0 if self._phi0 > 0 else obj

    def evaluate(self, theta: np.ndarray):
        model = self.decode(theta)
        obj = self.scaled_objective(model)
        dev = self.deviation(model)
        vio = self.violation(model.w)
        return obj + self.lam * dev + self.space.penalty_factor * vio, obj, dev, vio

    # ------------------------------------------------------------------ operators
    def _init_population(self):
        P = np.tile(self.theta0, (self.cfg.pop_size, 1))
        r = self.rng.uniform(self.lo, self.hi, size=P.shape)
        P[1:, self.free] = r[1:, self.free]          # individual 0 = elicited parameters
        return P

    def _tournament(self, fit):
        i, j = self.rng.choice(len(fit), 2, replace=False)
        return i if fit[i] < fit[j] else j

    def _crossover(self, a, b):
        if self.rng.random() >= self.cfg.crossover_rate:
            return a.copy()
        n, c, eta = self.n, a.copy(), self.cfg.eta_c
        # weights: simulated binary crossover (Deb & Agrawal, 1995)
        for i in range(n):
            if abs(a[i] - b[i]) > 1e-14:
                u = self.rng.random()
                beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                c1 = 0.5 * ((a[i] + b[i]) - beta * abs(b[i] - a[i]))
                c2 = 0.5 * ((a[i] + b[i]) + beta * abs(b[i] - a[i]))
                c[i] = c1 if self.rng.random() < 0.5 else c2
        # thresholds: uniform crossover
        mask = self.rng.random(2 * n) < 0.5
        c[n:3 * n] = np.where(mask, a[n:3 * n], b[n:3 * n])
        # preference-function types: single-point crossover
        if n > 1:
            k = self.rng.integers(1, n)
            c[3 * n:3 * n + k], c[3 * n + k:] = a[3 * n:3 * n + k], b[3 * n + k:]
        return c

    def _mutate(self, x):
        hit = (self.rng.random(len(x)) < self.cfg.mutation_rate) & self.free
        x = x.copy()
        x[hit] += self.rng.normal(0, self.cfg.mutation_scale * (self.hi - self.lo)[hit])
        return np.clip(x, self.lo, self.hi)

    def _restore_fixed(self, x):
        x[~self.free] = self.theta0[~self.free]
        return x

    # ------------------------------------------------------------------ main loop
    def run(self, verbose: bool = False) -> GAResult:
        pop = self._init_population()
        evals = [self.evaluate(t) for t in pop]
        fit = np.array([e[0] for e in evals])
        history = []
        for g in range(self.cfg.generations):
            order = np.argsort(fit)
            new = [pop[i].copy() for i in order[:self.cfg.elitism]]
            while len(new) < self.cfg.pop_size:
                child = self._crossover(pop[self._tournament(fit)], pop[self._tournament(fit)])
                new.append(self._restore_fixed(self._mutate(child)))
            pop = np.array(new)
            evals = [self.evaluate(t) for t in pop]
            fit = np.array([e[0] for e in evals])
            b = int(np.argmin(fit))
            history.append({"generation": g + 1, "fitness": float(fit[b]), "objective": evals[b][1],
                            "deviation": evals[b][2], "violation": evals[b][3]})
            if verbose:
                print(f"gen {g + 1:3d}  fitness {fit[b]:.5f}  objective {evals[b][1]:.5f}  "
                      f"deviation {evals[b][2]:.4f}")
        b = int(np.argmin(fit))
        return GAResult(pop[b], self.decode(pop[b]), float(fit[b]), evals[b][1], evals[b][2],
                        evals[b][3], history, pop)


def herfindahl(w: np.ndarray) -> float:
    """Concentration of a weight vector (1/n for equal weights, 1 for a single criterion)."""
    w = np.asarray(w) / np.sum(w)
    return float((w ** 2).sum())
