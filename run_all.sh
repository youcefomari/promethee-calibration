#!/usr/bin/env bash
# Reproduce every result, table and figure of the paper from the raw data.
# Usage: bash run_all.sh   (about 6 h on two cores; seeds are fixed in the scripts)
set -euo pipefail
cd "$(dirname "$0")/code"
python -m pip install -r requirements-lock.txt
python -m pytest tests -q                                    # numerical and symbolic checks of the theory
python -m experiments.run_calibration --workers 2            # diagnosis, search methods, exact optimum, fronts, validation
python -m experiments.figures_cal                            # figures of the calibration study
python -m experiments.run_robustness --draws 45 --workers 2  # 225 random elicited models
python -m experiments.run_robustness --tables
python -m experiments.run_baselines --workers 2              # robustness-oriented weight choices from the literature
python -m experiments.run_revision --workers 2               # ties, MILP certification, sensitivity analyses
python -m experiments.revision_stats                         # statistics with Holm correction
python -m experiments.run_synthetic --workers 2              # factorial synthetic benchmark (216 instances)
python -m experiments.run_timing --workers 1                 # wall-clock time of each stage
python -m experiments.run_scaling                            # empirical check of the complexity analysis
python -m experiments.run_sideeffects                        # incomparability and cyclicity after calibration
python -m experiments.run_litmeasures                        # impact and two-cycle index
python -m experiments.figures_cor                            # scaling and mechanism figures
python -m experiments.cor_tables ../paper/tables
python -m experiments.revision_tables ../paper/tables
python -m experiments.baselines_tables ../paper/tables
# Figures 1 and 8 (TikZ): cd paper/figures_src && pdflatex fig_method && pdflatex fig_decision
