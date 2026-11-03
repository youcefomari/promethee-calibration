import numpy as np
import pytest

from gpii import data
from gpii.core import incomparability_count, promethee
from gpii.ga import GAConfig, GeneticPromethee, SearchSpace

SIZES = {"ELECTRIC": (19, 15), "HDI": (193, 4), "WHR24": (144, 6), "BLI": (35, 11), "ASEM": (51, 8)}


@pytest.mark.parametrize("name", list(SIZES))
def test_dataset_shapes(name):
    ds = data.load(name)
    assert ds.X.shape == SIZES[name]
    assert np.isclose(ds.model.w.sum(), 1.0)
    assert np.all(ds.model.q <= ds.model.p)


def test_ga_is_reproducible_and_never_worse_than_elicited():
    ds = data.load("ELECTRIC")
    cfg = GAConfig(pop_size=12, generations=6, seed=7)
    r1 = GeneticPromethee(ds.X, ds.model, "cyclicity", lam=1.0, config=cfg).run()
    r2 = GeneticPromethee(ds.X, ds.model, "cyclicity", lam=1.0, config=cfg).run()
    assert np.array_equal(r1.theta, r2.theta)
    ga = GeneticPromethee(ds.X, ds.model, "cyclicity", lam=1.0, config=cfg)
    f0 = ga.evaluate(ga.theta0)[0]
    assert r1.fitness <= f0 + 1e-12          # elicited parameters are individual 0 + elitism


def test_ga_objective_depends_on_thresholds_and_types():
    ds = data.load("ELECTRIC")
    space = SearchSpace(optimize_weights=False, optimize_thresholds=True, optimize_types=True)
    ga = GeneticPromethee(ds.X, ds.model, "incomparability", "PI", space=space)
    rng = np.random.default_rng(0)
    thetas = [np.r_[ga.theta0[:15], rng.uniform(0, ga.th_up), rng.uniform(0, ga.th_up),
                    rng.integers(0, 6, 15)] for _ in range(30)]
    vals = {round(ga.evaluate(t)[1], 6) for t in thetas}
    assert len(vals) > 1


def test_incomparability_objective_counts_pairs_once():
    ds = data.load("ELECTRIC")
    pr = promethee(ds.X, ds.model)
    ga = GeneticPromethee(ds.X, ds.model, "incomparability", "PI")
    assert np.isclose(ga.objective(ds.model), incomparability_count(pr, "PI") / 171)
