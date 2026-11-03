"""gpii -- rank-reversal-aware calibration of PROMETHEE II.

Modules
-------
preference  six PROMETHEE preference functions (corrected)
model       preference model and pairwise preference indices
core        PROMETHEE I/II and PROMETHEE gamma
conflict    conflict measures (reversal rate, fragility, impact, cyclicity)
calibration admissible encoding, deviation and calibration problem
moea        evolutionary search (pymoo)
exact       exact mixed-integer model (HiGHS via SciPy)
cycles      two-cycle / gradient / cyclic decomposition and indices
reversal    rank-reversal certificates and empirical protocols
ga          Genetic-PROMETHEE II (corrected) and minimal-adjustment calibration
data        the five public benchmarks
"""
from .model import Dataset, Model, preference_index, unicriterion
from .core import SOURCES, Problem, incomparability_count, promethee, rank_desc, relations
from .cycles import (cyclic_residual, cyclicity_index, decomposition_summary, inversion_share,
                     mean_triangle, two_cycle_index)

__version__ = "0.1.0"
