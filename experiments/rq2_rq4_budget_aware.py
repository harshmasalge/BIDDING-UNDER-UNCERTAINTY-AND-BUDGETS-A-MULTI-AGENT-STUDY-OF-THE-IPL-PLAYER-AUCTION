"""RQ2 (budget-aware vs myopic bidding) and RQ4 (learning without an opponent model).
A focal team plays against the nine other teams using their pacing-equilibrium multipliers (from RQ3)."""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from sim import (Market, Agent, Shade, Equilibrium, Pace, DualPace, run_auction, team_utility, curse_stats,
                 hindsight_opt, TEAMS, post_mean, MIN_SQ, R_MIN)
from dp import collect_pilot, DPSolution, DPAgent
from rl import QAgent, KAPPAS

SIG, ALPHA = 0.3, 1.0
EVAL = range(100)
CONFIGS = [("RR", 1.0), ("DC", 1.0), ("PBKS", 1.0), ("DC", 0.5), ("DC", 2.0)]

def train_rl(M, opp, focal, scale, episodes=1500):
    Q = None
    for e in range(episodes):
        eps = max(0.3 * (1 - e / (0.8 * episodes)), 0.02)
        ag = opp(); qa = QAgent(Q=Q, eps=eps, train=True, seed=e); ag[focal] = qa
        res = run_auction(M, ag, sig=SIG, seed=50_000 + e, alpha=ALPHA, order="fixed", purse_scale=scale)
        f = res["fill"] & (res["owner"] == focal)
        qa.finish(float((res["v"][f] - ALPHA * res["price"][f]).sum()))
        Q = qa.Q
    return Q

def prop1_check(M, sol, pilot, focal, n_states=300, seed=0):
    """Brute-force the best drop-out price against the pilot's rival-bid samples and compare with Eq. 7."""
    rng = np.random.default_rng(seed)
    gaps, agree = [], 0
    mono = bool(np.all(np.diff(sol.V[:, MIN_SQ:, :], axis=2) >= -1e-4))
    for _ in range(n_states):
        t = rng.integers(0, M.K - 1); j = sol.perm[t]
        m = int(rng.integers(M.m0[focal], 24)); b = float(rng.uniform(0.3 * max(MIN_SQ - m, 0) + 0.5, sol.B))
        cap = b - R_MIN * max(MIN_SQ - m - 1, 0)
        x = np.log(pilot["V"][rng.integers(len(pilot["V"])), j]) + SIG * rng.standard_normal()
        val = float(post_mean(x, M.mu[j], M.s[j], SIG))
        D = pilot["D"][:, j]
        p = np.where(D >= M.r[j], D + 0.05, M.r[j])
        def payoff(a):
            win = (a >= p) & (a >= M.r[j]) & (p <= cap)
            cont_w = np.array([sol.interp(t + 1, m + 1, b - pp) if w else 0 for pp, w in zip(p, win)])
            return np.mean(np.where(win, val - ALPHA * p + cont_w, sol.interp(t + 1, m, b)))
        grid = np.r_[0.0, np.arange(M.r[j], max(cap, M.r[j]) + 0.05, 0.05)]
        vals = np.array([payoff(a) for a in grid])
        a_star = sol.dropout(t, j, val, b, m, cap)
        best = vals.max(); got = payoff(a_star)
        gaps.append(best - got); agree += got >= best - 1e-6
    return dict(states=n_states, share_optimal=agree / n_states, mean_gap=float(np.mean(gaps)),
                max_gap=float(np.max(gaps)), V_monotone_in_budget=mono)

def main():
    M = Market()
    lam = json.load(open("results/rq3_pacing_equilibrium.json"))["lambda_star"]
    out = {"lambda_star": lam}
    for team, scale in CONFIGS:
        t0 = time.time()
        focal = TEAMS.index(team)
        opp = lambda: [Pace(l) for l in lam]
        pilot = collect_pilot(M, opp, focal, n_runs=40, sig=SIG, alpha=ALPHA, purse_scale=scale)
        perm = M.setwise_order(np.random.default_rng(12345))
        B = M.purse[focal] * scale
        sol = DPSolution(M, pilot, B, M.m0[focal], perm, SIG, ALPHA)
        sol_b = DPSolution(M, pilot, B, M.m0[focal], perm, SIG, ALPHA, value="beta")
        Q = train_rl(M, opp, focal, scale)
        makers = {"naive": Agent, "shade0.85": lambda: Shade(0.85), "equilibrium": Equilibrium,
                  "pace_eq": lambda: Pace(lam[focal]), "dual_pace": DualPace,
                  "rl_qlearning": lambda: QAgent(Q=Q.copy(), train=False),
                  "dp": lambda: DPAgent(sol), "dp_beta": lambda: DPAgent(sol_b, "dp_beta")}
        cfg = {}
        for name, mk in makers.items():
            U, F, S, N, OPT, WC = [], [], [], [], [], []
            for s in EVAL:
                ag = opp(); ag[focal] = mk()
                res = run_auction(M, ag, sig=SIG, seed=s, alpha=ALPHA, order="fixed", purse_scale=scale)
                mine = res["owner"] == focal
                U.append(team_utility(res, M, focal, ALPHA)); F.append(res["v"][mine].sum())
                S.append(res["purse"][focal] - res["b"][focal]); N.append(int(mine.sum()))
                if s < 30: OPT.append(hindsight_opt(res, M, focal, ALPHA))
                won = mine & ~res["fill"]
                WC.append(float((res["price"][won] - res["v"][won]).sum()))
            U = np.array(U)
            cfg[name] = dict(utility=float(U.mean()), utility_se=float(U.std() / np.sqrt(len(U))),
                             squad_value=float(np.mean(F)), spend=float(np.mean(S)), players=float(np.mean(N)),
                             overpay_total=float(np.mean(WC)),
                             opt_ratio=float(np.mean(U[:30]) / np.mean(OPT)), opt=float(np.mean(OPT)),
                             utility_samples=U.tolist())
            print(team, scale, name, round(cfg[name]["utility"], 2), "+-", round(cfg[name]["utility_se"], 2),
                  "spend", round(cfg[name]["spend"], 1), "n", cfg[name]["players"], "OPT ratio", round(cfg[name]["opt_ratio"], 3), flush=True)
        cfg["dp_predicted_V0"] = sol.interp(0, M.m0[focal], B)
        if team == "DC" and scale == 1.0:
            cfg["prop1_check"] = prop1_check(M, sol, pilot, focal)
            print("Prop 1:", cfg["prop1_check"], flush=True)
            # grid-size sensitivity of the DP
            sens = {}
            for g in [0.1, 0.25, 0.5, 1.0]:
                sg = DPSolution(M, pilot, B, M.m0[focal], perm, SIG, ALPHA, grid=g)
                U = [team_utility(run_auction(M, opp()[:focal] + [DPAgent(sg)] + opp()[focal + 1:], sig=SIG, seed=s,
                                              alpha=ALPHA, order="fixed"), M, focal, ALPHA) for s in range(40)]
                sens[g] = (float(np.mean(U)), float(np.std(U) / np.sqrt(40)), sg.interp(0, M.m0[focal], B))
            cfg["grid_sensitivity"] = sens
            print("grid:", sens, flush=True)
            # dual-pacing multiplier trace for one auction
            dpz = DualPace(); ag = opp(); ag[focal] = dpz
            run_auction(M, ag, sig=SIG, seed=0, alpha=ALPHA, order="fixed")
            cfg["dual_trace"] = dpz.trace
        cfg["seconds"] = time.time() - t0
        out[f"{team}|{scale}"] = cfg
        json.dump(out, open("results/rq2_rq4_budget_aware.json", "w"))
    json.dump(out, open("results/rq2_rq4_budget_aware.json", "w"))

if __name__ == "__main__":
    main()
