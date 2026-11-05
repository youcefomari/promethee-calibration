"""Multi-objective evolutionary search for the calibration problem (pymoo back end).

All algorithms share the same encoding, the same feasibility-preserving repair (projection of
lam on the simplex, clipping of u), the same initial-population sampler (uniform on the
admissible set, theta0 included) and the same evaluation budget, so that differences are
due to the selection/variation scheme only.

Algorithms: NSGA-II (Deb et al., 2002), SMS-EMOA (Beume et al., 2007), MOEA/D (Zhang & Li,
2007), SPEA2 (Zitzler et al., 2001), and pure random sampling of the admissible set.
"""
from __future__ import annotations

import time

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from .calibration import Calibration, nondominated

ALGORITHMS = ("NSGA-II", "SMS-EMOA", "MOEA/D", "SPEA2", "Random")


class _Problem(Problem):
    def __init__(self, cal: Calibration):
        self.cal = cal
        self.archive_X, self.archive_F = [], []
        super().__init__(n_var=cal.n_var, n_obj=2, n_ieq_constr=1 if cal.max_dev is not None else 0,
                         xl=cal.xl, xu=cal.xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([self.cal.evaluate(x) for x in X])
        self.archive_X.append(np.array(X, copy=True))
        self.archive_F.append(F)
        out["F"] = F
        if self.cal.max_dev is not None:
            out["G"] = F[:, 1:2] - self.cal.max_dev


class _Repair(Repair):
    def __init__(self, cal):
        super().__init__()
        self.cal = cal

    def _do(self, problem, X, **kwargs):
        return self.cal.repair(X)


class _Sampling(Sampling):
    def __init__(self, cal, seed):
        super().__init__()
        self.cal = cal
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        return self.cal.sample(n_samples, self.rng)


def _algorithm(name, cal, pop, seed):
    common = dict(sampling=_Sampling(cal, seed), repair=_Repair(cal),
                  crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                  eliminate_duplicates=True)
    if name == "NSGA-II":
        from pymoo.algorithms.moo.nsga2 import NSGA2
        return NSGA2(pop_size=pop, **common)
    if name == "SMS-EMOA":
        from pymoo.algorithms.moo.sms import SMSEMOA
        return SMSEMOA(pop_size=pop, **common)
    if name == "SPEA2":
        from pymoo.algorithms.moo.spea2 import SPEA2
        return SPEA2(pop_size=pop, **common)
    if name == "MOEA/D":
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions
        common.pop("eliminate_duplicates")
        ref = get_reference_directions("das-dennis", 2, n_partitions=pop - 1)
        return MOEAD(ref, n_neighbors=max(2, pop // 5), prob_neighbor_mating=0.9, **common)
    raise ValueError(name)


def run(cal: Calibration, algorithm: str = "NSGA-II", budget: int = 3000, pop: int = 50,
        seed: int = 0, archive: bool = True) -> dict:
    """Run one algorithm; returns the non-dominated archive of *all* evaluated points
    (``archive=True``, fair across algorithms with different survival rules) and run data."""
    t0 = time.time()
    cal.n_eval = 0
    if algorithm == "Random":
        rng = np.random.default_rng(seed)
        X = cal.sample(budget, rng)
        F = np.array([cal.evaluate(x) for x in X])
    else:
        algo = _algorithm(algorithm, cal, pop, seed)
        prob = _Problem(cal)
        res = minimize(prob, algo, ("n_eval", budget), seed=seed, verbose=False)
        if archive:     # every evaluated point (fair across survival schemes)
            X, F = np.vstack(prob.archive_X), np.vstack(prob.archive_F)
        else:
            X, F = res.pop.get("X"), res.pop.get("F")
    if cal.max_dev is not None:
        ok = F[:, 1] <= cal.max_dev + 1e-12
        X, F = X[ok], F[ok]
    keep = nondominated(F)
    order = np.argsort(F[keep, 1])
    return {"X": X[keep][order], "F": F[keep][order], "time": time.time() - t0,
            "n_eval": cal.n_eval}


def hypervolume(F: np.ndarray, ref=(1.0, 1.0)) -> float:
    """Exact 2-D hypervolume of a minimisation front with respect to ``ref``."""
    F = np.asarray(F, float)
    F = F[np.all(F < np.asarray(ref), 1)]
    if len(F) == 0:
        return 0.0
    F = F[nondominated(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in F:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def igd_plus(F: np.ndarray, Z: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al., 2015) of front F with respect to reference front Z."""
    F, Z = np.asarray(F, float), np.asarray(Z, float)
    d = np.maximum(F[None, :, :] - Z[:, None, :], 0)
    return float(np.sqrt((d ** 2).sum(2)).min(1).mean())
