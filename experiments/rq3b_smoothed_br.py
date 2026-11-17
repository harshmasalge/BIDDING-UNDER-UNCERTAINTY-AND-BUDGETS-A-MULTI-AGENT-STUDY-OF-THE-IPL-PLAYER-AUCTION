"""RQ3 (cont.): plain best-response dynamics cycle (see rq3_pacing_equilibrium.json). Here:
(a) smoothed best response (fictitious-play style averaging lambda_i <- (1-1/k) lambda_i + (1/k) BR_i);
(b) epsilon-equilibrium check: the largest gain any team gets from a unilateral deviation on the grid,
    for the smoothed profile and for the time-average of the BR cycle."""
import sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
import numpy as np
from sim import Market
import rq3_pacing_equilibrium as R3
R3.SEEDS = range(12)                                   # cheaper evaluation for the smoothed dynamics
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
utilities = R3.utilities

def br(M, lams, i):
    best, bu = lams[i], -1e18
    for g in GRID:
        trial = list(lams); trial[i] = g
        u = utilities(M, trial)[:, i].mean()
        if u > bu: best, bu = g, u
    return best, bu

def epsilon(M, lams):
    base = utilities(M, lams).mean(0)
    gains = []
    for i in range(len(lams)):
        _, bu = br(M, lams, i)
        gains.append(max(bu - base[i], 0.0))
    return gains, base.tolist()

def main():
    M = Market()
    d = json.load(open("results/rq3_pacing_equilibrium.json"))
    cyc = np.array(d["from_0"]["history"][-4:]).mean(0).round(2).tolist()
    out = {"cycle_average": cyc}
    lam = [2.0] * 10; hist = [lam]
    for k in range(2, 9):
        new = []
        for i in range(10):
            b, _ = br(M, lam, i)
            new.append((1 - 1 / k) * lam[i] + b / k)
        lam = [round(x, 3) for x in new]; hist.append(lam)
        step = max(abs(a - b) for a, b in zip(hist[-1], hist[-2]))
        print(k, lam, "max step", round(step, 3), flush=True)
    out["smoothed_history"] = hist
    out["smoothed_final"] = lam
    for lab, prof in [("smoothed_final", lam), ("cycle_average", cyc)]:
        g, base = epsilon(M, prof)
        out[f"eps_{lab}"] = dict(gains=g, base_utility=base, max_gain=max(g),
                                 max_rel_gain=max(gi / max(abs(b), 1e-9) for gi, b in zip(g, base)))
        print(lab, "max eps gain", round(max(g), 2), "rel", round(out[f'eps_{lab}']['max_rel_gain'], 3), flush=True)
    best = min(["smoothed_final", "cycle_average"], key=lambda l: out[f"eps_{l}"]["max_rel_gain"])
    out["lambda_star"] = out[best] if best == "smoothed_final" else cyc
    out["lambda_star_source"] = best
    d["lambda_star_br_last"] = d["lambda_star"]; d["lambda_star"] = out["lambda_star"]
    d["smoothed"] = out
    json.dump(d, open("results/rq3_pacing_equilibrium.json", "w"), indent=1)

if __name__ == "__main__":
    main()
