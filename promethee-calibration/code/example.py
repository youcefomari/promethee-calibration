import numpy as np
from gpii.data import load
from gpii.core import promethee
from gpii.conflict import conflict_measures
from gpii.calibration import Calibration
from gpii.moea import run
from gpii.exact import solve_eps

d = load("ELECTRIC")                                    # decision matrix and elicited model
print(conflict_measures(promethee(d.X, d.model).P))     # 1. diagnosis of the elicited ranking

cal = Calibration(d.X, d.model, measure="event_rate", scenario="W")   # keep the order of the weights
evo = run(cal, "NSGA-II", budget=3000, pop=50, seed=0)  # 2. evolutionary front
ex = solve_eps(cal, 0.02)                               # 3. certified optimum for a budget of 0.02
X, F = evo["X"], evo["F"]
if ex["status"] == 0:
    X, F = np.vstack([X, ex["x"]]), np.vstack([F, ex["F"]])

ok = np.flatnonzero(F[:, 1] <= 0.02 + 1e-9)                  # 4. least reversal rate within the budget
best = ok[np.argmin(F[ok, 0])]
model = cal.decode(X[best])
print("relative reversal rate", F[best, 0], "deviation", F[best, 1])
print("calibrated weights", np.round(model.w, 3))
