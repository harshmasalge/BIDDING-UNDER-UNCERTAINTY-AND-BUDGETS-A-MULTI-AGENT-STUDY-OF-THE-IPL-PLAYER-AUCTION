"""Feedback point 1: out-of-sample validation against the real 2025 mega auction.
The value prior is fitted on the 2022 auction only; opponent behaviour is fixed a priori (naive, Eq.-5 equilibrium,
or the pacing equilibrium computed inside the simulator in RQ3). Nothing here is tuned on 2025 outcomes."""
import sys, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy import stats
from sim import Market, Agent, Equilibrium, Pace, NeedAware, run_auction, TEAMS

SEEDS = range(100)
ROLES = ["BAT", "WK", "AR", "BOWL"]

def summarize(M, pops, prior="mu_log", label=""):
    L = M.L
    real_sold = L.sold.values == 1
    real_p = np.where(real_sold, L.price.values, 0.0)
    real_team = L.team.values
    real_alloc = np.array([[np.sum((real_team == t) & (M.role == r)) for r in ROLES] for t in TEAMS])
    real_os = np.array([np.sum((real_team == t) & (M.os == 1)) for t in TEAMS])
    out = {}
    for name, mk in pops.items():
        P = np.zeros((len(SEEDS), M.K)); nsold, spend, alloc_d, os_d = [], [], [], []
        for k, s in enumerate(SEEDS):
            res = run_auction(M, mk(), sig=0.3, seed=s, order="setwise")
            sold = res["owner"] >= 0
            P[k] = np.where(sold, res["price"], 0.0)
            nsold.append(sold.sum()); spend.append(res["price"][sold].sum())
            A = np.array([[np.sum((res["owner"] == i) & (M.role == r)) for r in ROLES] for i in range(10)])
            alloc_d.append(np.abs(A - real_alloc).mean())
            os_d.append(np.abs(np.array([np.sum((res["owner"] == i) & (M.os == 1)) for i in range(10)]) - real_os).mean())
        sim_prices = P[P > 0]
        lp_sim, lp_real = np.log(sim_prices), np.log(real_p[real_sold])
        mean_p = P.mean(0)
        top_real = set(np.argsort(-real_p)[:20]); top_sim = set(np.argsort(-mean_p)[:20])
        out[name] = dict(n_sold=float(np.mean(nsold)), n_sold_real=int(real_sold.sum()),
                         median=float(np.median(sim_prices)), mean=float(sim_prices.mean()),
                         p99=float(np.percentile(sim_prices, 99)), max_mean=float(P.max(1).mean()),
                         total_spend=float(np.mean(spend)), total_spend_real=float(real_p.sum()),
                         ks=float(stats.ks_2samp(lp_sim, lp_real).statistic),
                         w1_log=float(stats.wasserstein_distance(lp_sim, lp_real)),
                         spearman_player=float(stats.spearmanr(mean_p, real_p).correlation),
                         spearman_sold_only=float(stats.spearmanr(mean_p[real_sold], real_p[real_sold]).correlation),
                         sold_prob_auc=float(roc_auc((P > 0).mean(0), real_sold)),
                         top20_overlap=len(top_real & top_sim) / 20,
                         team_role_mad=float(np.mean(alloc_d)), team_overseas_mad=float(np.mean(os_d)))
        print(label, name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
        np.save(f"results/sim_prices_{label}_{name}.npy", P[:20])
    return out

def roc_auc(score, y):
    return stats.mannwhitneyu(score[y], score[~y]).statistic / (y.sum() * (~y).sum())

def main():
    lam = json.load(open("results/rq3_pacing_equilibrium.json"))["lambda_star"]
    pops = {"naive": lambda: [Agent() for _ in range(10)],
            "equilibrium_eq5": lambda: [Equilibrium() for _ in range(10)],
            "pacing_equilibrium": lambda: [Pace(l) for l in lam],
            "need_aware_pacing": lambda: [NeedAware(l) for l in lam]}
    out = {"lambda_star": lam}
    M = Market(prior="mu_log_train_2022")
    out["prior_fit_2022"] = summarize(M, pops, label="fit2022")
    for pr in ["mu_log_train_2018_2024", "mu_log_insample_2025"]:
        M2 = Market(prior=pr)
        out[f"prior_{pr[7:]}"] = summarize(M2, {"pacing_equilibrium": pops["pacing_equilibrium"]}, label=pr[7:])
    json.dump(out, open("results/validation_2025.json", "w"), indent=1)

if __name__ == "__main__":
    main()
