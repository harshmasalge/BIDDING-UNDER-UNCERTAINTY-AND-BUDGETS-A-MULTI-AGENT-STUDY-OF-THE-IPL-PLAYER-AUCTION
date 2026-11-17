#!/usr/bin/env bash
# Reproduces every number, table, figure and screenshot in the report
# (about 1.5-2 hours on 2 CPU cores; the slowest steps are RQ3 and RQ2/RQ4).
set -e
cd "$(dirname "$0")"
run() { local log=$1; shift; echo ">> $*"; "$@" 2>&1 | tee "results/$log"; }

[ -d data/raw/cricsheet_ipl_json ] || unzip -q data/raw/cricsheet_ipl_json.zip -d data/raw/cricsheet_ipl_json
python src/parse_cricsheet.py data/raw/cricsheet_ipl_json data/processed/deliveries.csv.gz
python src/build_dataset.py                                   # lots + pre-auction features (no look-ahead)
run money_scale.log python experiments/fit_money_scale.py     # feedback 1: fit on 2022, test on 2025
run retained.log    python src/build_retained.py              # values of the 46 retained players
run tests.log       python -m pytest -q tests                 # invariants, exact best XI, curse sanity
run rq1.log         python experiments/rq1_winners_curse.py   # RQ1 + feedback 3 (tight budgets)
run rq3.log         python experiments/rq3_pacing_equilibrium.py  # RQ3: best-response dynamics
run rq3b.log        python experiments/rq3b_smoothed_br.py    # RQ3: smoothed BR, epsilon check -> lambda*
run validation.log  python experiments/validate_2025.py       # out-of-sample validation vs real 2025
run rq2.log         python experiments/rq2_rq4_budget_aware.py    # RQ2 + RQ4 (DP, Prop. 1, dual pacing, RL)
run rl_curve.log    python experiments/rq4_rl_curve.py        # Q-learning learning curve
run fb2.log         python experiments/fb2_team_specific.py   # feedback 2: team-specific best-XI values
run order.log       python experiments/order_effect.py        # sale order / declining prices
python experiments/make_figures.py
python experiments/make_tables.py
python experiments/make_screenshots.py
(cd report && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null)
echo "done: report/main.pdf"
