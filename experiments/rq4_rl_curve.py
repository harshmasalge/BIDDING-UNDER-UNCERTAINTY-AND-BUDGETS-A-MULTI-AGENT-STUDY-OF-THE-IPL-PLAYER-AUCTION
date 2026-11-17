"""RQ4: learning curve of the Q-learning bidder (DC, real purse) -- is its weak result just under-training?"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from sim import Market, Pace, run_auction, team_utility, TEAMS
from rl import QAgent

def main():
    M = Market(); lam = json.load(open("results/rq3_pacing_equilibrium.json"))["lambda_star"]
    focal = TEAMS.index("DC"); opp = lambda: [Pace(l) for l in lam]
    Q = None; curve = []; E = 6000
    for e in range(E + 1):
        if e % 750 == 0:
            U = []
            for s in range(30):
                ag = opp(); ag[focal] = QAgent(Q=None if Q is None else Q.copy(), train=False)
                U.append(team_utility(run_auction(M, ag, sig=0.3, seed=s, order="fixed"), M, focal))
            curve.append((e, float(np.mean(U)), float(np.std(U) / np.sqrt(30))))
            print(curve[-1], flush=True)
        if e == E: break
        eps = max(0.3 * (1 - e / (0.8 * E)), 0.02)
        ag = opp(); qa = QAgent(Q=Q, eps=eps, train=True, seed=e); ag[focal] = qa
        res = run_auction(M, ag, sig=0.3, seed=50_000 + e, order="fixed")
        f = res["fill"] & (res["owner"] == focal)
        qa.finish(float((res["v"][f] - res["price"][f]).sum())); Q = qa.Q
    json.dump(dict(curve=curve), open("results/rq4_rl_curve.json", "w"), indent=1)

if __name__ == "__main__":
    main()
