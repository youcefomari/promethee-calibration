"""The exact model with ties in the elicited model (duplicate alternatives, rounded data)."""
import numpy as np

from gpii.calibration import Calibration
from gpii.exact import solve_eps
from gpii.model import Model


def _instance(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(9, 3)).astype(float)
    X[8] = X[0]                                   # a duplicate: tied for every weight vector
    model = Model(np.ones(3, bool), np.full(3, 1 / 3), np.zeros(3), X.std(0) + 1e-9, types=("linear",) * 3)
    return Calibration(X, model, "event_rate", "W", "free")


def test_exact_model_feasible_and_not_worse_than_elicited_with_ties():
    for seed in range(4):
        cal = _instance(seed)
        if cal.C0 <= 0:
            continue
        o = solve_eps(cal, 0.05, time_limit=60)
        assert o["status"] == 0
        assert o["F"][0] <= 1.0 + 1e-9               # the elicited weights are feasible
        assert abs(o["F"][0] - o["milp_value"]) < 1e-6  # the model counts what the measure counts
