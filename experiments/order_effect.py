"""Sale-order experiment (declining-price anomaly): price relative to expected value along the auction,
real set-wise order vs. random order, under the pacing-equilibrium population."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from sim import Market, Pace, run_auction

def main():
    M = Market()
    lam = json.load(open("results/rq3_pacing_equilibrium.json"))["lambda_star"]
    Ev = np.exp(M.mu + M.s ** 2 / 2)
    out = {}
    for order in ["setwise", "random"]:
        bins = [[] for _ in range(10)]; sold_rate = [[] for _ in range(10)]
        for s in range(60):
            r = run_auction(M, [Pace(l) for l in lam], sig=0.3, seed=s, order=order)
            pos = np.empty(M.K, int); pos[r["perm"]] = np.arange(M.K)
            q = pos * 10 // M.K
            au = (r["owner"] >= 0) & ~r["fill"]
            for k in range(10):
                sel = (q == k) & au
                if sel.any(): bins[k].append(r["price"][sel].sum() / r["v"][sel].sum())
                sold_rate[k].append(au[q == k].mean())
        out[order] = dict(price_over_value=[float(np.mean(b)) for b in bins],
                          se=[float(np.std(b) / np.sqrt(len(b))) for b in bins],
                          sold_rate=[float(np.mean(b)) for b in sold_rate])
        print(order, np.round(out[order]["price_over_value"], 3), np.round(out[order]["sold_rate"], 2), flush=True)
    json.dump(out, open("results/order_effect.json", "w"), indent=1)

if __name__ == "__main__":
    main()
