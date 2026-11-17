# PROTOTYPE ONLY: synthetic data. Known issues: (1) winner's curse vanishes once budgets bind (calibration needed); (2) hindsight ILP benchmark cost proxy is inconsistent. Do not cite these numbers.
"""Feasibility prototype: sequential common-value auction with budgets and squad constraints.
Synthetic players; real data would replace `make_players`."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N_TEAMS, POOL, BUDGET = 10, 200, 40.0
MIN_SQ, MAX_SQ, MAX_OS = 18, 25, 8
INC = 0.25

def make_players(rng):
    v = rng.lognormal(mean=0.9, sigma=0.8, size=POOL)          # true value (crore-ish)
    os_ = rng.random(POOL) < 0.3                                # overseas flag
    order = rng.permutation(POOL)
    return v[order], os_[order]

class Team:
    def __init__(self, strat, rng):
        self.strat, self.budget, self.n, self.os, self.vals, self.paid = strat, BUDGET, 0, 0, [], []
    def can_take(self, is_os):
        return self.n < MAX_SQ and (not is_os or self.os < MAX_OS)
    def slots_needed(self):
        return max(MIN_SQ - self.n, 0)
    def max_bid(self, signal, is_os, remaining_items, params):
        if not self.can_take(is_os): return 0.0
        # must keep 0.5 per remaining mandatory slot
        reserve = 0.5 * max(self.slots_needed() - 1, 0)
        cap = max(self.budget - reserve, 0)
        s = self.strat
        if s == "naive":   b = signal
        elif s == "shade": b = signal * params["shade"]
        elif s == "pace":  # never spend more than fair share of budget on one player
            per_slot = self.budget / max(self.slots_needed(), 1)
            b = min(signal * params["shade"], 3.0 * per_slot)
        else: b = signal
        if self.slots_needed() > 0 and remaining_items <= self.slots_needed() + 15:
            b = max(b, signal)              # must-fill: stop shading near the end
        return min(b, cap)

def run(strats, sigma, seed, shade=0.85):
    rng = np.random.default_rng(seed)
    v, os_ = make_players(rng)
    teams = [Team(s, rng) for s in strats]
    params = {"shade": shade}
    stats = dict(overpay=[], sold=0)
    for k in range(POOL):
        # private noisy multiplicative signals (common value)
        sig = v[k] * np.exp(rng.normal(0, sigma, N_TEAMS) - sigma**2 / 2)
        bids = np.array([t.max_bid(sig[i], os_[k], POOL - k, params) for i, t in enumerate(teams)])
        order = np.argsort(-bids)
        top, second = bids[order[0]], bids[order[1]]
        if top < 0.2: continue
        price = min(top, second + INC) if second >= 0.2 else 0.2   # ascending-auction outcome
        w = teams[order[0]]
        w.budget -= price; w.n += 1; w.os += int(os_[k]); w.vals.append(v[k]); w.paid.append(price)
        stats["overpay"].append(price - v[k]); stats["sold"] += 1
    return teams, stats, v, os_

def strength(t, top=11):
    return sum(sorted(t.vals, reverse=True)[:top])

def hindsight_ilp(team, v, os_):
    """Best squad from the whole pool under budget/size/overseas constraints at true values
    with cost = the price this team actually paid on won items, else the value (cost proxy)."""
    n = len(v)
    c = -v
    cost = np.maximum(1.3 * v, 0.2)
    A = [cost, np.ones(n), os_.astype(float)]
    lo = [0, MIN_SQ, 0]; hi = [BUDGET, MAX_SQ, MAX_OS]
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    return -res.fun, res.status

if __name__ == "__main__":
    import sys
    print("== Winner's curse vs signal noise (all teams 'naive') ==")
    for sigma in [0.0, 0.1, 0.25, 0.5]:
        op, ratio = [], []
        for seed in range(30):
            teams, st, v, o = run(["naive"] * N_TEAMS, sigma, seed)
            op += st["overpay"]
            ratio += [sum(t.paid) / max(sum(t.vals), 1e-9) for t in teams if t.vals]
        print(f"sigma={sigma:4}: mean overpay/player={np.mean(op):6.3f}  paid/value ratio={np.mean(ratio):5.3f}")
    print("\n== Strategy tournament (1 focal team vs 9 naive), sigma=0.25, 60 seeds ==")
    for s in ["naive", "shade", "pace"]:
        sc, ratio = [], []
        for seed in range(60):
            teams, st, v, o = run([s] + ["naive"] * (N_TEAMS - 1), 0.25, seed)
            sc.append(strength(teams[0])); ratio.append(sum(teams[0].paid) / max(sum(teams[0].vals), 1e-9))
        print(f"{s:6}: top-11 strength {np.mean(sc):6.2f} +- {np.std(sc)/np.sqrt(60):.2f}   paid/value {np.mean(ratio):5.2f}")
    teams, st, v, o = run(["naive"] * N_TEAMS, 0.25, 0)
    best, status = hindsight_ilp(teams[0], v, o)
    print(f"\nILP (200 vars) status={status}, hindsight-optimal value={best:.1f} vs team0 actual={sum(teams[0].vals):.1f}")
    # invariant checks
    ok = all(t.budget >= -1e-9 and t.n <= MAX_SQ and t.os <= MAX_OS for t in teams)
    print("constraints respected:", ok, "| squad sizes:", [t.n for t in teams])
