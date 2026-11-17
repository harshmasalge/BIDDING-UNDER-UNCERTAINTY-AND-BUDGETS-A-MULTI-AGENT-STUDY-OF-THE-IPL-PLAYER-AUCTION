"""Budget-aware bidding by backward induction (Eq. 6) and the indifference drop-out rule (Prop. 1, Eq. 7).

The opponent model G_j is the empirical distribution of the highest rival drop-out price D_j, jointly with the
true value v_j, collected from pilot auctions against the same opponent population (full-information DP).
Overseas counts are enforced by the simulator but not tracked in the DP state (documented approximation)."""
import numpy as np
from sim import (Agent, run_auction, post_mean, beta_eq, MIN_SQ, MAX_SQ, R_MIN, INC)

def collect_pilot(market, make_opponents, focal, n_runs=40, sig=0.3, alpha=1.0, seed0=10_000, purse_scale=1.0,
                  focal_agent=None):
    """Run pilot auctions; record for every lot the true value and the highest rival bid (after caps)."""
    K = market.K
    V = np.zeros((n_runs, K)); D = np.zeros((n_runs, K)); fillv = []
    for rnum in range(n_runs):
        agents = make_opponents()
        if focal_agent is not None: agents[focal] = focal_agent()
        rec = RivalRecorder(agents, focal)
        res = run_auction(market, rec.agents, sig=sig, seed=seed0 + rnum, alpha=alpha, order="fixed",
                          purse_scale=purse_scale)
        V[rnum] = res["v"]; D[rnum] = rec.D
        f = res["fill"] & (res["owner"] == focal)
        fillv += list(res["v"][f])
    v_fill = float(np.mean(fillv)) if fillv else float(np.exp(market.mu).min())
    return dict(V=V, D=D, v_fill=v_fill, sig=sig, alpha=alpha)

class RivalRecorder:
    """Wraps agents so that the highest rival (non-focal) capped bid on each lot is recorded."""
    def __init__(self, agents, focal):
        self.focal = focal; self.agents = agents; self.D = None
        rec = self
        for i, a in enumerate(agents):
            orig = a.bid
            def wrapped(j, x, st, _orig=orig, _i=i):
                out = _orig(j, x, st)
                if rec.D is None: rec.D = np.zeros(len(st["x"]))
                if _i != rec.focal:
                    cap = st["b"][_i] - R_MIN * max(MIN_SQ - st["m"][_i] - 1, 0)
                    rec.D[j] = max(rec.D[j], min(out, cap))
                return out
            a.bid = wrapped

class DPSolution:
    def __init__(self, market, pilot, B, m0, perm, sig, alpha=1.0, grid=0.25, n_samples=None, value="vhat", seed=0):
        rng = np.random.default_rng(seed)
        self.grid, self.alpha, self.B, self.m0 = grid, alpha, B, m0
        self.bgrid = np.arange(0, B + grid / 2, grid)
        nb, nm = len(self.bgrid), MAX_SQ + 1
        Vs, Ds = pilot["V"], pilot["D"]
        S = Vs.shape[0] if n_samples is None else n_samples
        K = market.K
        mm = np.arange(nm)[:, None]
        # terminal value
        k = np.maximum(MIN_SQ - mm, 0)
        bb = self.bgrid[None, :]
        VT = np.where(bb >= R_MIN * k - 1e-9, alpha * (bb - R_MIN * k) + k * pilot["v_fill"], -1e3)
        self.V = np.zeros((K + 1, nm, nb), np.float32)
        self.V[K] = VT
        self.perm = perm
        self.mu, self.s, self.r, self.sig = market.mu, market.s, market.r, sig
        self.value = value
        reserve = R_MIN * np.maximum(MIN_SQ - mm - 1, 0)          # (nm,1)
        for t in range(K - 1, -1, -1):
            j = perm[t]
            Vn = self.V[t + 1].astype(float)
            v = Vs[:S, j]; Dj = Ds[:S, j]
            x = np.log(v) + sig * rng.standard_normal(S)
            vh = self._val(j, x)
            p = np.where(Dj >= self.r[j], np.minimum(Dj + INC, 1e9), self.r[j])          # price if focal wins
            # continuation after a win: V_{t+1}(b - p, m + 1), linear interpolation on the budget grid
            Vup = np.vstack([Vn[1:], np.full((1, nb), -1e3)])                            # m+1 (m=25 infeasible)
            pos = (self.bgrid[None, :] - p[:, None]) / grid                               # (S, nb)
            lo = np.floor(pos).astype(int); fr = pos - lo
            ok = lo >= 0
            lo_c = np.clip(lo, 0, nb - 1); hi_c = np.clip(lo + 1, 0, nb - 1)
            cont = Vup[:, lo_c] * (1 - fr) + Vup[:, hi_c] * fr                            # (nm, S, nb)
            feas = ok[None] & (p[None, :, None] <= self.bgrid[None, None, :] - reserve[:, :, None] + 1e-9)
            feas &= (mm < MAX_SQ)[:, :, None]
            gain_win = vh[None, :, None] - alpha * p[None, :, None] + cont                 # decision uses vhat
            win = feas & (gain_win >= Vn[:, None, :])
            real_win = v[None, :, None] - alpha * p[None, :, None] + cont                  # payoff uses true v
            self.V[t] = np.where(win, real_win, Vn[:, None, :]).mean(1)

    def _val(self, j, x):
        if self.value == "beta":
            return beta_eq(x, self.mu[j], self.s[j], self.sig, 10)
        return post_mean(x, self.mu[j], self.s[j], self.sig)

    def interp(self, t, m, b):
        V = self.V[t, m]
        pos = b / self.grid
        lo = int(np.floor(pos)); fr = pos - lo
        if lo < 0: return -1e3
        if lo >= len(V) - 1: return float(V[-1])
        return float(V[lo] * (1 - fr) + V[lo + 1] * fr)

    def dropout(self, t, j, val, b, m, cap):
        """Eq. 7: highest price p in [r_j, cap] with  val - alpha p >= V_{t+1}(b,m) - V_{t+1}(b-p,m+1)."""
        if m >= MAX_SQ or cap < self.r[j]: return 0.0
        ps = np.arange(self.r[j], cap + 1e-9, 0.05)
        if len(ps) == 0: return 0.0
        stay = self.interp(t + 1, m, b)
        Vn = self.V[t + 1, m + 1]
        pos = (b - ps) / self.grid
        lo = np.floor(pos).astype(int); fr = pos - lo
        lo_c = np.clip(lo, 0, len(Vn) - 1); hi_c = np.clip(lo + 1, 0, len(Vn) - 1)
        cont = np.where(lo >= 0, Vn[lo_c] * (1 - fr) + Vn[hi_c] * fr, -1e3)
        okp = val - self.alpha * ps + cont >= stay
        if not okp[0]: return 0.0
        last = np.argmin(okp) - 1 if not okp.all() else len(ps) - 1
        return float(ps[last])

class DPAgent(Agent):
    def __init__(self, sol, label="dp"):
        self.sol = sol; self.name = label
    def reset(self, ctx, i):
        super().reset(ctx, i)
        self.pos = {j: t for t, j in enumerate(self.sol.perm)}
    def bid(self, j, x, st):
        i = self.i
        cap = st["b"][i] - R_MIN * max(MIN_SQ - st["m"][i] - 1, 0)
        val = float(self.sol._val(j, np.atleast_1d(x))[0])
        return self.sol.dropout(self.pos[j], j, val, st["b"][i], st["m"][i], cap)
