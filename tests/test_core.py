"""Invariant and sanity tests. Run:  python -m pytest -q tests   (or: python tests/test_core.py)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sim import Market, Agent, Pace, Equilibrium, NeedAware, run_auction, curse_stats, MIN_SQ, MAX_SQ, MAX_OS, TEAMS
from bestxi import best_xi, xi_milp, XITracker, ROLES

M = Market()

def test_squad_and_budget_invariants():
    for pop in [lambda: [Agent() for _ in range(10)], lambda: [Pace(2.0) for _ in range(10)],
                lambda: [NeedAware(1.0) for _ in range(10)]]:
        for s in range(5):
            r = run_auction(M, pop(), sig=0.3, seed=s, order="setwise")
            assert (r["b"] >= -1e-9).all()
            assert ((r["m"] >= MIN_SQ) & (r["m"] <= MAX_SQ)).all()
            assert (r["q"] <= MAX_OS).all()
            sold = r["owner"] >= 0
            assert (r["price"][sold] >= M.r[sold] - 1e-9).all()
            for i in range(10):
                assert abs(r["purse"][i] - r["b"][i] - r["price"][r["owner"] == i].sum()) < 1e-6

def test_no_curse_without_noise():
    r = curse_stats(run_auction(M, [Agent() for _ in range(10)], sig=0.0, seed=1, purse_scale=20, order="fixed"))
    assert abs(r["belief_bias"]) < 1e-9 and r["paid_over_value"] <= 1.0 + 1e-9

def test_curse_with_noise_and_slack_budgets():
    st = [curse_stats(run_auction(M, [Agent() for _ in range(10)], sig=0.3, seed=s, purse_scale=20, order="fixed"))
          for s in range(10)]
    assert np.mean([d["overpay"] for d in st]) > 0

def test_equilibrium_bid_below_naive():
    r = run_auction(M, [Equilibrium() for _ in range(10)], sig=0.3, seed=0, purse_scale=20, order="fixed")
    assert curse_stats(r)["paid_over_value"] < 1.0

def test_best_xi_exact():
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.integers(8, 26); v = rng.lognormal(0, 1, n); o = (rng.random(n) < .3).astype(int)
        role = list(rng.choice(ROLES, n))
        mm = xi_milp(v, o, role)
        if not np.isnan(mm): assert abs(best_xi(v, o, role) - mm) < 1e-6

def test_tracker_matches_batch():
    rng = np.random.default_rng(2)
    v = rng.lognormal(0, 1, 20); o = (rng.random(20) < .3).astype(int); role = list(rng.choice(ROLES, 20))
    t = XITracker(v[:10], o[:10], role[:10])
    for k in range(10, 20): t.add(v[k], o[k], role[k])
    from bestxi import squad_value
    assert abs(t.cur - squad_value(v, o, role)) < 1e-6

def test_real_pre_auction_state():
    assert M.K == 577 and M.m0.sum() == 46 and abs(M.purse.sum() - 641.5) < 1e-9

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
