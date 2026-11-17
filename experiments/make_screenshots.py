"""Render the console output of the experiment runs as terminal-style screenshots for the README."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, re

def shot(title, cmd, lines, out, width=118):
    lines = [l[:width] for l in lines]
    h = 0.75 + 0.142 * (len(lines) + 2)
    fig = plt.figure(figsize=(10.5, h)); fig.patch.set_facecolor("#1a1a19")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_facecolor("#1a1a19")
    ax.text(0.012, 1 - 0.18 / h, title, color="#c3c2b7", family="monospace", fontsize=8.5, va="top", weight="bold")
    ax.text(0.012, 1 - 0.42 / h, "$ " + cmd, color="#1baf7a", family="monospace", fontsize=8, va="top")
    ax.text(0.012, 1 - 0.62 / h, "\n".join(lines), color="#f0efec", family="monospace", fontsize=7.4, va="top", linespacing=1.32)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor()); plt.close(fig)

R = lambda f: open(f"results/{f}").read().splitlines()
shot("Unit tests", "python -m pytest -q tests", R("tests.log"), "screenshots/01_tests.png")
shot("Feedback 1: money scale fitted on 2022, tested on 2025", "python experiments/fit_money_scale.py",
     R("money_scale.log")[:3] + ["..."] + [l for l in R("money_scale.log") if "slope" in l or "mean_surprise" in l], "screenshots/02_money_scale.png")
rq1 = R("rq1.log")
shot("RQ1 + feedback 3: winner's curse vs noise and purse tightness", "python experiments/rq1_winners_curse.py",
     [rq1[0][:118]] + [l for l in rq1 if "|" not in l and ("0.3 {" in l)] + rq1[-8:], "screenshots/03_rq1.png")
shot("RQ3: best-response and smoothed best-response dynamics over pacing multipliers",
     "python experiments/rq3_pacing_equilibrium.py && python experiments/rq3b_smoothed_br.py", R("rq3.log") + R("rq3b.log"), "screenshots/04_rq3.png")
shot("Out-of-sample validation against the real 2025 auction", "python experiments/validate_2025.py",
     [re.sub(r"'(\w+)': ", r"\1=", l).replace("{", "").replace("}", "") for l in R("validation.log")], "screenshots/05_validation.png", width=160)
shot("RQ2 + RQ4: focal team vs nine pacing-equilibrium opponents", "python experiments/rq2_rq4_budget_aware.py",
     R("rq2.log"), "screenshots/06_rq2_rq4.png")
shot("Feedback 2: team-specific (best-XI) values", "python experiments/fb2_team_specific.py",
     R("fb2.log") + R("order.log") + ["", "Q-learning learning curve (episodes, utility, s.e.):"] + R("rl_curve.log"), "screenshots/07_team_specific.png")
print("ok")
