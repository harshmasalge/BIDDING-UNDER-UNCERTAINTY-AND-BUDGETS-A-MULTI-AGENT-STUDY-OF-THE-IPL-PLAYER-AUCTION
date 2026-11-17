"""All report figures from results/*.json (run after the experiments)."""
import sys, json, os
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
                     "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight"})
R = lambda f: json.load(open(f"results/{f}")) if os.path.exists(f"results/{f}") else None
os.makedirs("figures", exist_ok=True)

def money_scale():
    ms = R("money_scale.json"); L = pd.read_csv("data/processed/lots_2025_valued.csv")
    s = L[L.sold == 1]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for o, lab, c in [(0, "Indian", C[0]), (1, "Overseas", C[1])]:
        d = s[s.overseas == o]
        ax[0].scatter(np.exp(d.mu_log_train_2022), d.price, s=10, color=c, label=lab, alpha=.8, edgecolor="white", linewidth=.4)
    ax[0].plot([0.1, 40], [0.1, 40], color=INK2, lw=1, ls="--")
    ax[0].set(xscale="log", yscale="log", xlabel="predicted price (model fitted on 2022), cr", ylabel="real 2025 price, cr",
              title="Out-of-sample: 2022 fit vs 2025 prices")
    ax[0].legend(loc="upper left")
    names = ["train_2022", "train_2018_2024", "insample_2025"]
    labs = ["fit 2022\n(primary)", "fit 2018-24", "fit 2025\n(in-sample)"]
    r2 = [ms[n]["test_2025"]["r2_log"] for n in names]; rho = [ms[n]["test_2025"]["spearman"] for n in names]
    xx = np.arange(3)
    ax[1].bar(xx - .18, r2, .34, color=C[0], label="$R^2$ (log price)")
    ax[1].bar(xx + .18, rho, .34, color=C[2], label="Spearman")
    for i in range(3):
        ax[1].text(i - .18, r2[i] + .01, f"{r2[i]:.2f}", ha="center", fontsize=7, color=INK)
        ax[1].text(i + .18, rho[i] + .01, f"{rho[i]:.2f}", ha="center", fontsize=7, color=INK)
    ax[1].set_xticks(xx, labs); ax[1].set_ylim(0, 1.15); ax[1].set_title("Accuracy on 2025 sold players"); ax[1].legend(loc="upper left", ncol=2, fontsize=7)
    fig.savefig("figures/fig_money_scale.png"); plt.close(fig)

def curse():
    d = R("rq1_winners_curse.json")
    if d is None: return
    sig = [0.0, 0.15, 0.3, 0.5, 0.75]
    regimes = [(20.0, "slack (20x purse)"), (3.0, "loose (3x)"), (1.0, "real 2025 purses"), (0.5, "tight (0.5x)")]
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for k, (sc, lab) in enumerate(regimes):
        for pop, ls in [("naive", "-"), ("equilibrium", ":")]:
            y = [d["grid"][f"{pop}|{sc}|{s}"]["paid_over_value"][0] for s in sig]
            ax[0].plot(sig, y, ls, color=C[k], marker="o", ms=3, label=lab if pop == "naive" else None)
            yb = [d["grid"][f"{pop}|{sc}|{s}"]["belief_bias"][0] for s in sig]
            if pop == "naive": ax[1].plot(sig, yb, "-", color=C[k], marker="o", ms=3)
    ax[0].axhline(1, color=INK2, lw=1)
    ax[0].set(xlabel="signal noise $\\sigma$", ylabel="price paid / true value", title="Overpayment (solid naive,\ndotted Eq. 5 equilibrium)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.36, -0.06))
    ax[1].set(xlabel="signal noise $\\sigma$", ylabel="E[$\\hat v_w - v$] (cr)", title="Winner's belief bias\n(naive)")
    n = [4, 6, 8, 10]
    for pop, c, lab in [("naive", C[0], "naive"), ("equilibrium", C[1], "equilibrium (Eq. 5)")]:
        ax[2].plot(n, [d["n_teams_slack"][f"{pop}|{k}"]["paid_over_value"][0] for k in n], color=c, marker="o", ms=3, label=lab)
    ax[2].axhline(1, color=INK2, lw=1)
    ax[2].set(xlabel="number of teams $n$", ylabel="price paid / true value", title="Curse vs. bidders\n(slack, $\\sigma$=0.3)")
    ax[2].legend(fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("figures/fig_curse.png"); plt.close(fig)
    dv = d["single_lot"]["deviation_payoff_bidder0"]
    k = sorted(map(float, dv)); y = [dv[str(x) if str(x) in dv else x][0] for x in k]
    e = [dv[str(x) if str(x) in dv else x][1] for x in k]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.errorbar(k, y, yerr=1.96 * np.array(e), color=C[0], marker="o", ms=3, capsize=2)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set(xlabel="bidder 1 bids $k\\cdot\\beta(x)$", ylabel="expected payoff", title="Single lot: $\\beta$ is a best response")
    fig.savefig("figures/fig_beta_check.png"); plt.close(fig)

def rq3():
    d = R("rq3_pacing_equilibrium.json")
    if d is None: return
    T = d["teams"]
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for a, lab in [(ax[0], "from_0"), (ax[1], "from_4")]:
        H = np.array(d[lab]["history"])
        im = a.imshow(H.T, aspect="auto", cmap="Blues", vmin=0, vmax=6)
        a.set_yticks(range(10), T, fontsize=6.5); a.set_xlabel("best-response round"); a.grid(False)
        a.set_title(f"BR dynamics, start $\\lambda$={lab[-1]}")
    fig.colorbar(im, ax=ax[1], fraction=.05, label="$\\lambda_i$")
    sm = d.get("smoothed")
    if sm:
        H = np.array(sm["smoothed_history"])
        for i in range(10):
            ax[2].plot(H[:, i], color=INK2, lw=1, alpha=.7)
        ax[2].plot(H.mean(1), color=C[0], lw=2.2, label="mean over teams")
        ends = sorted([(H[-1, i], T[i]) for i in range(10)])
        for k, (y, t) in enumerate(ends):
            ax[2].text(len(H) - 0.85, 2.0 + k * 0.25, t, fontsize=5.5, color=INK2, va="center")
            ax[2].plot([len(H) - 1, len(H) - 0.9], [y, 2.0 + k * 0.25], color=GRID, lw=.6)
        ax[2].set_xlim(0, len(H) + 0.6)
        ax[2].set(xlabel="smoothed-BR iteration", ylabel="$\\lambda_i$", title="Smoothed best response")
        ax[2].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig("figures/fig_rq3.png"); plt.close(fig)

def validation():
    d = R("validation_2025.json")
    if d is None: return
    L = pd.read_csv("data/processed/lots_2025_valued.csv")
    real = np.sort(np.log(L.price[L.sold == 1]))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax[0].plot(real, np.linspace(0, 1, len(real)), color=INK, lw=2.5, label="real 2025")
    names = [("naive", "naive"), ("equilibrium_eq5", "Eq. 5 equilibrium"), ("pacing_equilibrium", "pacing equilibrium"),
             ("need_aware_pacing", "need-aware pacing")]
    for k, (n, lab) in enumerate(names):
        P = np.load(f"results/sim_prices_fit2022_{n}.npy"); p = np.sort(np.log(P[P > 0]))
        ax[0].plot(p, np.linspace(0, 1, len(p)), color=C[k], lw=1.5, label=f"{lab} (KS={d['prior_fit_2022'][n]['ks']:.2f})")
    ax[0].set(xlabel="log price (cr)", ylabel="CDF", title="Sold-price distribution, 2025")
    ax[0].legend(fontsize=6.5, loc="lower right")
    P = np.load("results/sim_prices_fit2022_pacing_equilibrium.npy").mean(0)
    sold = L.sold.values == 1
    ax[1].scatter(P[sold] + 0.05, L.price[sold], s=9, color=C[2], edgecolor="white", linewidth=.4)
    ax[1].plot([0.05, 30], [0.05, 30], color=INK2, lw=1, ls="--")
    ax[1].set(xscale="log", yscale="log", xlabel="simulated mean price (pacing eq.), cr", ylabel="real price, cr",
              title=f"Player level ($\\rho$={d['prior_fit_2022']['pacing_equilibrium']['spearman_sold_only']:.2f}, sold players)")
    fig.tight_layout(); fig.savefig("figures/fig_validation.png"); plt.close(fig)

def rq2():
    d = R("rq2_rq4_budget_aware.json")
    if d is None: return
    agents = ["naive", "shade0.85", "equilibrium", "pace_eq", "dual_pace", "rl_qlearning", "dp", "dp_beta"]
    labs = ["naive", "shade .85", "Eq. 5", "pacing $\\lambda^*$", "dual pacing", "Q-learning", "DP", "DP + Eq. 5"]
    cfgs = [k for k in d if "|" in k]
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    w = 0.8 / len(agents)
    for a, (ag, lab) in enumerate(zip(agents, labs)):
        y = [d[c][ag]["utility"] for c in cfgs]; e = [1.96 * d[c][ag]["utility_se"] for c in cfgs]
        ax.bar(np.arange(len(cfgs)) + (a - len(agents) / 2 + .5) * w, y, w * .9, yerr=e, color=C[a], label=lab,
               error_kw=dict(lw=.7, ecolor=INK2))
    cl = {"RR|1.0": "RR (41 cr)", "DC|1.0": "DC (73 cr)", "PBKS|1.0": "PBKS (110.5 cr)", "DC|0.5": "DC, purse x0.5", "DC|2.0": "DC, purse x2"}
    ax.set_xticks(range(len(cfgs)), [cl.get(c, c) for c in cfgs])
    ax.axhline(0, color=INK2, lw=.8)
    ax.set(ylabel="focal utility  $f(S)-\\alpha\\cdot$spend (cr)", title="Focal team vs. 9 pacing-equilibrium opponents (100 seeds, 95% CI)")
    ax.legend(ncol=4, fontsize=6.5, loc="upper left")
    fig.savefig("figures/fig_rq2_rq4.png"); plt.close(fig)
    c = d.get("DC|1.0", {})
    if "dual_trace" in c:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ax.plot(c["dual_trace"], color=C[0], lw=1.2)
        ax.axhline(d["lambda_star"][1], color=C[1], lw=1, ls="--", label="equilibrium $\\lambda^*_{DC}$")
        ax.set(xlabel="lot", ylabel="$\\lambda$", title="Dual pacing multiplier (DC)"); ax.legend(fontsize=7)
        fig.savefig("figures/fig_dual_trace.png"); plt.close(fig)

def fb2():
    d = R("fb2_team_specific.json")
    if d is None: return
    h = d["heterogeneity"]; Ev = np.array(h["common_value"])
    T = list(d["focal"])
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
    k = 15; rel = np.array(h["rel_mean"])[:k]
    for j in range(k):
        ax[0].plot([rel[j].min(), rel[j].max()], [j, j], color=GRID, lw=3, solid_capstyle="round")
        ax[0].scatter(rel[j], [j] * 10, s=9, color=C[0], zorder=3, edgecolor="white", linewidth=.3)
    ax[0].set_yticks(range(k), [n[:18] for n in h["players"][:k]], fontsize=6); ax[0].invert_yaxis()
    ax[0].axvline(1, color=INK2, lw=1, ls="--")
    ax[0].set(xlabel="marginal best-XI value / common value", title="Same player, 10 teams (mid-auction)")
    g = [d["focal"][t]["gain"] for t in T]; e = [1.96 * d["focal"][t]["gain_se"] for t in T]
    ax[1].bar(range(10), g, .7, yerr=e, color=[C[2] if x > 0 else C[7] for x in g], error_kw=dict(lw=.7, ecolor=INK2))
    ax[1].axhline(0, color=INK2, lw=.8)
    ax[1].set_xticks(range(10), T, rotation=45, fontsize=7)
    ax[1].set(ylabel="gain in own-XI utility (cr)", title="Need-aware minus common-value pacing")
    fig.tight_layout(); fig.savefig("figures/fig_team_specific.png"); plt.close(fig)

def order_expost():
    d = R("order_effect.json")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    if d:
        for k, o in enumerate(["setwise", "random"]):
            y = np.array(d[o]["price_over_value"]); e = np.array(d[o]["se"])
            ax[0].errorbar(np.arange(10) * 10 + 5, y, yerr=1.96 * e, color=C[k], marker="o", ms=3, capsize=2, label=f"{o} order")
        ax[0].set(xlabel="position in auction (percentile)", ylabel="price / true value", title="Prices decline along the auction")
        ax[0].legend(fontsize=7)
    s = pd.read_csv("results/expost_2025.csv"); ms = R("money_scale.json")["expost_2025"]
    ax[1].scatter(s.premium_log, s.surprise, s=10, color=C[0], edgecolor="white", linewidth=.4)
    xx = np.linspace(s.premium_log.min(), s.premium_log.max(), 10)
    ax[1].plot(xx, ms["slope"] * xx + (s.surprise.mean() - ms["slope"] * s.premium_log.mean()), color=C[1], lw=1.5,
               label=f"fit slope {ms['slope']:.2f}")
    ax[1].plot(xx, ms["slope_if_premium_justified"] * xx + (s.surprise.mean() - ms["slope_if_premium_justified"] * s.premium_log.mean()),
               color=INK2, lw=1, ls=":", label=f"slope if premium justified ({ms['slope_if_premium_justified']:.0f})")
    ax[1].set_ylim(-16, 14)
    ax[1].axhline(0, color=INK2, lw=.8)
    ax[1].set(xlabel="price premium over public model (log)", ylabel="realised - predicted\nimpact (runs/match)",
              title="Real data: premiums not repaid on field")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("figures/fig_order_expost.png"); plt.close(fig)

def rl_curve():
    d = R("rq4_rl_curve.json"); b = R("rq2_rq4_budget_aware.json")
    if d is None or b is None: return
    c = np.array(d["curve"])
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.errorbar(c[:, 0], c[:, 1], yerr=1.96 * c[:, 2], color=C[0], marker="o", ms=3, capsize=2, label="Q-learning (greedy eval)")
    ax.axhline(b["DC|1.0"]["dp"]["utility"], color=C[6], ls="--", lw=1.2, label="DP (full information)")
    ax.axhline(b["DC|1.0"]["pace_eq"]["utility"], color=C[3], ls="--", lw=1.2, label="pacing $\\lambda^*_{DC}$")
    ax.axhline(b["DC|1.0"]["naive"]["utility"], color=INK2, ls=":", lw=1.2, label="naive")
    ax.set(xlabel="training auctions", ylabel="utility (cr)", title="RQ4: Q-learning bidder (DC)")
    ax.set_ylim(30, 175)
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    fig.savefig("figures/fig_rl_curve.png"); plt.close(fig)

if __name__ == "__main__":
    for f in [money_scale, curse, rq3, validation, rq2, fb2, order_expost, rl_curve]:
        try: f(); print("ok", f.__name__)
        except Exception as ex: print("FAIL", f.__name__, ex)
