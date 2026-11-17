"""RQ3: best-response dynamics over pacing multipliers when every team paces (a = vhat / (alpha + lambda)).
Each round, every team in turn picks the lambda on a grid that maximises its mean utility (common random numbers,
other teams fixed). Run from two initial profiles to check convergence / uniqueness."""
import sys, json, time
sys.path.insert(0, "src")
import numpy as np
from sim import Market, Pace, run_auction, team_utility, TEAMS

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
SEEDS = range(20)
SIG, ALPHA = 0.3, 1.0

def utilities(M, lams, seeds=SEEDS):
    U = np.zeros((len(seeds), len(lams)))
    for k, s in enumerate(seeds):
        res = run_auction(M, [Pace(l) for l in lams], sig=SIG, seed=s, alpha=ALPHA, order="fixed")
        U[k] = [team_utility(res, M, i, ALPHA) for i in range(len(lams))]
    return U

def br_dynamics(M, lam0, rounds=8):
    lams = list(lam0); hist = [list(lams)]
    for rd in range(rounds):
        changed = 0
        for i in range(len(lams)):
            best, bu = lams[i], -1e18
            for g in GRID:
                trial = list(lams); trial[i] = g
                u = utilities(M, trial)[:, i].mean()
                if u > bu + 1e-9: best, bu = g, u
            changed += best != lams[i]; lams[i] = best
        hist.append(list(lams))
        print(f"round {rd+1}: {lams}  changes={changed}", flush=True)
        if changed == 0: break
    return lams, hist

def main():
    M = Market()
    t = time.time()
    out = {}
    for lab, init in [("from_0", [0.0] * 10), ("from_4", [4.0] * 10)]:
        lams, hist = br_dynamics(M, init)
        out[lab] = dict(final=lams, history=hist, rounds=len(hist) - 1)
    lam_star = out["from_0"]["final"]
    # welfare comparison on fresh seeds
    ev = range(100, 200)
    U_eq = utilities(M, lam_star, ev); U_nv = utilities(M, [0.0] * 10, ev)
    out["lambda_star"] = lam_star
    out["mean_utility_eq"] = U_eq.mean(0).tolist(); out["mean_utility_naive"] = U_nv.mean(0).tolist()
    out["total_utility_eq"] = float(U_eq.sum(1).mean()); out["total_utility_naive"] = float(U_nv.sum(1).mean())
    out["total_utility_se"] = [float(U_eq.sum(1).std() / 10), float(U_nv.sum(1).std() / 10)]
    # spend map: total spend share as a function of a common lambda
    sp = {}
    for g in GRID:
        S = []
        for s in range(100, 120):
            r = run_auction(M, [Pace(g) for _ in range(10)], sig=SIG, seed=s, order="fixed")
            S.append(((r["purse"] - r["b"]) / r["purse"]).mean())
        sp[g] = float(np.mean(S))
    out["spend_share_vs_common_lambda"] = sp
    out["teams"] = TEAMS; out["seconds"] = time.time() - t
    json.dump(out, open("results/rq3_pacing_equilibrium.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ["lambda_star", "total_utility_eq", "total_utility_naive", "spend_share_vs_common_lambda"]}))

if __name__ == "__main__":
    main()
