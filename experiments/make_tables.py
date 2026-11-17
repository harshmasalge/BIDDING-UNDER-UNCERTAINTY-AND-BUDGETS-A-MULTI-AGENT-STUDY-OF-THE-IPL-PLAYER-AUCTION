"""Write every numeric table of the report from results/*.json (so the paper's numbers come from the code)."""
import json, os
import numpy as np
R = lambda f: json.load(open(f"results/{f}"))
os.makedirs("report/tables", exist_ok=True)
W = lambda name, s: open(f"report/tables/{name}.tex", "w").write(s)

ms = R("money_scale.json")
rows = []
for k, lab in [("train_2022", "Fit on 2022 (primary)"), ("train_2018_2024", "Fit on 2018--2024 (pooled)"),
               ("insample_2025", "Fit on 2025 (in-sample, old check)")]:
    tr, te = ms[k]["train"], ms[k]["test_2025"]
    rows.append(f"{lab} & {tr['n']} & {tr['r2_log']:.2f} & {te['r2_log']:.2f} & {te['spearman']:.2f} & {te['mae_log']:.2f} & {te['bias_log']:+.2f}\\\\")
W("money_scale", "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n& & train & \\multicolumn{4}{c}{test: 2025 sold players ($n$=" +
  str(ms['train_2022']['test_2025']['n']) + ")}\\\\\\cmidrule(l){4-7}\nMoney scale & $n$ & $R^2$ & $R^2$ & Spearman & MAE & bias\\\\\\midrule\n" +
  "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

c = R("rq1_winners_curse.json")
rows = []
for sc, lab in [(20.0, "slack ($20\\times$)"), (3.0, "loose ($3\\times$)"), (1.0, "real 2025"), (0.5, "tight ($0.5\\times$)")]:
    g = lambda pop, key, s=0.3: c["grid"][f"{pop}|{sc}|{s}"][key][0]
    rows.append(f"{lab} & {g('naive','paid_over_value',0.0):.2f} & {g('naive','paid_over_value'):.2f} & {g('naive','overpay'):+.2f} & "
                f"{g('naive','belief_bias'):.2f} & {g('naive','capped_share'):.2f} & {g('equilibrium','paid_over_value'):.2f} & {g('naive','n_sold'):.0f}\\\\")
W("curse", "\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n& \\multicolumn{5}{c}{naive bidders} & Eq.~(5) & \\\\\\cmidrule(lr){2-6}\\cmidrule(lr){7-7}\n"
  "Purses & $p/v$, $\\sigma$=0 & $p/v$ & $\\E[p-v]$ & $\\E[\\hat v_w-v]$ & capped & $p/v$ & sold\\\\\\midrule\n" + "\n".join(rows) +
  "\n\\bottomrule\\end{tabular}\n")

v = R("validation_2025.json")
rows = []
pp = v["prior_fit_2022"]
labs = {"naive": "Naive", "equilibrium_eq5": "Eq.~(5) equilibrium", "pacing_equilibrium": "Pacing equilibrium $\\lambda^*$",
        "need_aware_pacing": "Need-aware pacing $\\lambda^*$"}
for k, lab in labs.items():
    d = pp[k]
    rows.append(f"{lab} & {d['n_sold']:.0f} & {d['median']:.2f} & {d['mean']:.2f} & {d['max_mean']:.1f} & {d['total_spend']:.0f} & {d['ks']:.2f} & "
                f"{d['spearman_sold_only']:.2f} & {d['sold_prob_auc']:.2f} & {d['team_role_mad']:.2f}\\\\")
d = pp["naive"]
rows.append(f"\\midrule Real 2025 & {d['n_sold_real']} & 1.50 & 3.51 & 27.0 & {d['total_spend_real']:.0f} & -- & -- & -- & --\\\\")
for k, lab in [("prior_train_2018_2024", "Pacing eq., prior fit 2018--24"), ("prior_insample_2025", "Pacing eq., prior fit 2025 (in-sample)")]:
    d = v[k]["pacing_equilibrium"]
    rows.append(f"{lab} & {d['n_sold']:.0f} & {d['median']:.2f} & {d['mean']:.2f} & {d['max_mean']:.1f} & {d['total_spend']:.0f} & {d['ks']:.2f} & "
                f"{d['spearman_sold_only']:.2f} & {d['sold_prob_auc']:.2f} & {d['team_role_mad']:.2f}\\\\")
W("validation", "\\begin{tabular}{@{}lrrrrrrrrr@{}}\\toprule\nOpponent model & sold & median & mean & max & spend & KS & $\\rho$ & AUC & role MAD\\\\\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

b = R("rq2_rq4_budget_aware.json")
agents = [("naive", "Naive $\\hat v$"), ("shade0.85", "Shade 0.85"), ("equilibrium", "Eq.~(5)"), ("pace_eq", "Pacing $\\lambda^*_i$"),
          ("dual_pace", "Dual pacing (9)"), ("rl_qlearning", "Q-learning"), ("dp", "DP (6)--(7)"), ("dp_beta", "DP + Eq.~(5)")]
cfgs = [k for k in b if "|" in k]
head = {"RR|1.0": "RR", "DC|1.0": "DC", "PBKS|1.0": "PBKS", "DC|0.5": "DC $\\times0.5$", "DC|2.0": "DC $\\times2$"}
rows = []
for a, lab in agents:
    cells = []
    for k in cfgs:
        best = max(b[k][x]["utility"] for x, _ in agents)
        u = b[k][a]["utility"]; s = f"{u:.1f}"
        cells.append(f"\\textbf{{{s}}}" if u == best else s)
    cells.append(f"{np.mean([b[k][a]['opt_ratio'] for k in cfgs]):.2f}")
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
W("rq2", "\\begin{tabular}{@{}l" + "r" * (len(cfgs) + 1) + "@{}}\\toprule\nFocal agent & " + " & ".join(head[k] for k in cfgs) +
  " & $U/\\mathrm{OPT}$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print("tables written")

# share of the DP's gain over naive bidding recovered by each cheaper agent
rows = []
for a, lab in [("pace_eq", "Pacing $\\lambda^*_i$"), ("dual_pace", "Dual pacing (9)"), ("rl_qlearning", "Q-learning"), ("dp_beta", "DP + Eq.~(5)")]:
    cells = []
    for k in cfgs:
        g = (b[k][a]["utility"] - b[k]["naive"]["utility"]) / (b[k]["dp"]["utility"] - b[k]["naive"]["utility"])
        cells.append(f"{100 * g:.0f}\\%")
    rows.append(f"{lab} & " + " & ".join(cells) + "\\\\")
W("recovery", "\\begin{tabular}{@{}l" + "r" * len(cfgs) + "@{}}\\toprule\nShare of DP gain over naive & " +
  " & ".join(head[k] for k in cfgs) + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(open("report/tables/recovery.tex").read())

import pandas as pd, sys
sys.path.insert(0, "src")
from teams import RETAINED, OVERSEAS_RETAINED
rows = []
for t, (p, names) in RETAINED.items():
    rows.append(f"{t} & {p:.1f} & {len(names)} & {sum(n in OVERSEAS_RETAINED for n in names)} & \\footnotesize " + ", ".join(names) + "\\\\")
W("teams", "\\begin{tabularx}{\\linewidth}{@{}lrrrX@{}}\\toprule\nTeam & purse & $m^0$ & $q^0$ & retained players\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabularx}\n")
