"""Sequential ascending-auction simulator for the 2025 IPL mega auction (real lots, purses, retentions).

Changes w.r.t. the idea-review prototype (feedback point 3, tight budgets):
  * real purses and retentions (B_i, m_i^0, q_i^0) instead of one synthetic budget;
  * feasibility cap (Eq. 2) reserves the minimum base price (0.30 cr) for every still-mandatory slot;
  * the ad-hoc "stop shading near the end" override is removed; teams below 18 players are completed in an
    explicit fill phase at reserve prices (as the accelerated rounds do in reality);
  * the winner's curse is measured two ways -- belief bias E[vhat_w - v] and overpayment E[p - v] -- and every
    sale records whether the winner's bid was capped by its budget, so budget-masking can be separated from
    the curse itself;
  * the hindsight ILP uses realised prices (Eq. 10), not a cost proxy.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from teams import RETAINED, OVERSEAS_RETAINED, ROLE_RETAINED
from bestxi import XITracker, squad_value

MIN_SQ, MAX_SQ, MAX_OS, R_MIN, INC = 18, 25, 8, 0.30, 0.05
TEAMS = list(RETAINED)
ROLE_ORDER = {"BAT": 0, "WK": 1, "AR": 2, "BOWL": 3}

# ---------------------------------------------------------------- data
class Market:
    """Lots, their public value priors (log-normal), and the teams' pre-auction state."""
    def __init__(self, lots_csv="data/processed/lots_2025_valued.csv", prior="mu_log", retained_csv="data/processed/retained_2025.csv"):
        L = pd.read_csv(lots_csv)
        self.L = L
        self.name = L.name.values
        self.mu = L[prior].values.astype(float)
        self.s = L.s_log.values.astype(float)
        self.r = L.base.values.astype(float)
        self.os = L.overseas.values.astype(int)
        self.role = L.role.values
        self.capped = L.capped.values.astype(int)
        self.K = len(L)
        R = pd.read_csv(retained_csv)
        self.purse = np.array([RETAINED[t][0] for t in TEAMS])
        self.ret = {t: R[R.team == t] for t in TEAMS}
        self.m0 = np.array([len(self.ret[t]) for t in TEAMS])
        self.q0 = np.array([int(self.ret[t].overseas.sum()) for t in TEAMS])

    def setwise_order(self, rng):
        """Real-style order: capped before uncapped, higher base-price sets first, roles grouped; shuffled within set."""
        key = np.lexsort((rng.random(self.K), [ROLE_ORDER[r] for r in self.role], -self.r, -self.capped))
        return key

# ---------------------------------------------------------------- posterior helpers (log-normal common value)
def post_mean(x, mu, s, sig):
    """E[v | x] with log v ~ N(mu, s^2), x = log v + N(0, sig^2)."""
    if sig == 0: return np.exp(x)
    w = s**2 / (s**2 + sig**2)
    m = w * x + (1 - w) * mu; var = w * sig**2
    return np.exp(m + var / 2)

_U = np.linspace(-5, 5, 161)
_LOG2PI = 0.5 * np.log(2 * np.pi)
def beta_eq(x, mu, s, sig, n):
    """Symmetric-equilibrium bid (Eq. 5): E[v | x1 = x, max of the other n-1 signals = x], v = exp(u)."""
    from scipy.special import log_ndtr
    if sig == 0: return np.exp(np.atleast_1d(x))
    x = np.atleast_1d(np.asarray(x, float))
    u = mu + s * _U                                  # grid over log value
    z = (x[:, None] - u[None, :]) / sig
    lp = -0.5 * _U[None, :] ** 2 + 2 * (-0.5 * z ** 2) + (n - 2) * log_ndtr(z)
    lp -= lp.max(1, keepdims=True)
    w = np.exp(lp)
    return (w * np.exp(u)[None, :]).sum(1) / w.sum(1)

# ---------------------------------------------------------------- agents
class Agent:
    """Base agent: returns a drop-out price given its private signal and state."""
    name = "naive"
    team_specific = False
    def reset(self, ctx, i): self.ctx, self.i = ctx, i
    def value(self, j, x):
        c = self.ctx
        return post_mean(x, c.mu[j], c.s[j], c.sig)
    def bid(self, j, x, st): return self.value(j, x)
    def observe(self, j, winner, price, st): pass

class Shade(Agent):
    def __init__(self, k=0.85): self.k = k; self.name = f"shade{k}"
    def bid(self, j, x, st): return self.k * self.value(j, x)

class Equilibrium(Agent):
    """Winner's-curse-corrected bid beta(x) of Eq. 5 with n = number of currently eligible teams."""
    name = "equilibrium"
    def bid(self, j, x, st):
        c = self.ctx
        n = max(int(st["eligible"][j]), 2)
        return float(beta_eq(x, c.mu[j], c.s[j], c.sig, n)[0])

class Pace(Agent):
    """Fixed multiplicative pacing (Eq. 8 with omega = 0): a = vhat / (alpha + lambda)."""
    def __init__(self, lam=0.3): self.lam = lam; self.name = f"pace{lam}"
    def bid(self, j, x, st): return self.value(j, x) / (self.ctx.alpha + self.lam)

class DualPace(Agent):
    """Online pacing by projected dual sub-gradient descent (Eq. 9); no opponent model needed."""
    def __init__(self, eta=0.02, lam_max=3.0, lam0=0.0): self.eta, self.lam_max, self.lam0 = eta, lam_max, lam0; self.name = "dualpace"
    def reset(self, ctx, i):
        super().reset(ctx, i); self.lam = self.lam0
        self.rho = ctx.purse[i] / ctx.K
        self.trace = []
    def bid(self, j, x, st): return self.value(j, x) / (self.ctx.alpha + self.lam)
    def observe(self, j, winner, price, st):
        spend = price if winner == self.i else 0.0
        self.lam = float(np.clip(self.lam - self.eta * (self.rho - spend), 0, self.lam_max))
        self.trace.append(self.lam)

class NeedAware(Agent):
    """Team-specific valuation (feedback point 2): bids the marginal value of the player to ITS OWN best XI,
    using its estimates of bought players and the known values of its retained players."""
    team_specific = True
    def __init__(self, lam=0.0): self.lam = lam; self.name = "need" if lam == 0 else f"need_pace{lam}"
    def reset(self, ctx, i):
        super().reset(ctx, i)
        R = ctx.ret[TEAMS[i]]
        self.trk = XITracker(R.value.values, R.overseas.values, R.role.values)
    def bid(self, j, x, st):
        c = self.ctx
        vh = self.value(j, x)
        return self.trk.marginal(vh, c.os[j], c.role[j]) / (c.alpha + self.lam)
    def observe(self, j, winner, price, st):
        if winner == self.i:
            self.trk.add(self.value(j, st["x"][j, self.i]), self.ctx.os[j], self.ctx.role[j])

# ---------------------------------------------------------------- auction
class Ctx:
    pass

def run_auction(market, agents, sig=0.3, seed=0, alpha=1.0, purse_scale=1.0, order="setwise", n_teams=None,
                record=False, v=None):
    rng = np.random.default_rng(seed)
    K = market.K
    n = len(agents)
    idx = np.arange(n)
    if v is None:
        v = np.exp(market.mu + market.s * rng.standard_normal(K))         # true common values
    x = np.log(v)[:, None] + sig * rng.standard_normal((K, n))             # private log-signals
    if order == "fixed":
        perm = market.setwise_order(np.random.default_rng(12345))      # one fixed set-wise order for all seeds
    elif order == "setwise":
        perm = market.setwise_order(rng)
    else:
        perm = rng.permutation(K)
    c = Ctx(); c.mu, c.s, c.sig, c.alpha, c.K = market.mu, market.s, sig, alpha, K
    c.purse = market.purse[:n] * purse_scale; c.ret = market.ret; c.os = market.os; c.role = market.role
    b = c.purse.copy(); m = market.m0[:n].copy(); q = market.q0[:n].copy()
    for i, a in enumerate(agents): a.reset(c, i)
    owner = -np.ones(K, int); price = np.zeros(K); capped = np.zeros(K, bool); vhat_w = np.full(K, np.nan)
    eligible = np.zeros(K)
    st = {"x": x, "eligible": eligible, "b": b, "m": m, "q": q, "v_true": v}   # v_true: only the RL reward uses it
    for t, j in enumerate(perm):
        elig = (m < MAX_SQ) & ((market.os[j] == 0) | (q < MAX_OS))
        cap = b - R_MIN * np.maximum(MIN_SQ - m - 1, 0)
        elig &= cap >= market.r[j]
        eligible[j] = elig.sum()
        if not elig.any(): continue
        raw = np.zeros(n)
        for i in idx[elig]:
            raw[i] = agents[i].bid(j, x[j, i], st)
        a = np.where(elig, np.minimum(np.nan_to_num(raw, nan=0.0), cap), 0.0)
        top = np.flatnonzero(a == a.max())
        w = top[rng.integers(len(top))] if len(top) > 1 else top[0]
        if a[w] < market.r[j]:
            for ag in agents: ag.observe(j, -1, 0.0, st)
            continue
        others = np.delete(a, w)
        second = others.max() if len(others) else 0.0
        p = market.r[j] if second < market.r[j] else min(a[w], second + INC)
        owner[j], price[j] = w, p
        capped[j] = raw[w] > cap[w] + 1e-9
        vhat_w[j] = post_mean(x[j, w], market.mu[j], market.s[j], sig)
        b[w] -= p; m[w] += 1; q[w] += market.os[j]
        for ag in agents: ag.observe(j, w, p, st)
    # accelerated round (as in the real auction): unsold players are offered again at their reserve; teams take
    # turns (round-robin, random start) claiming the unsold player with the largest estimated surplus
    # vhat - alpha*r > 0, subject to squad, overseas and budget caps.  Then a mandatory fill phase completes any
    # squad below 18 at reserve prices.  (Replaces the prototype's "stop shading near the end" override.)
    fill = np.zeros(K, bool)
    def claim(i, need_positive):
        un = np.flatnonzero((owner < 0) & (market.r <= b[i] - R_MIN * max(MIN_SQ - m[i] - 1, 0) + 1e-9)
                            & ((market.os == 0) | (q[i] < MAX_OS)))
        if len(un) == 0 or m[i] >= MAX_SQ: return False
        est = post_mean(x[un, i], market.mu[un], market.s[un], sig) - alpha * market.r[un]
        k = np.argmax(est)
        if need_positive and est[k] <= 0: return False
        j = un[k]
        owner[j], price[j], fill[j] = i, market.r[j], True
        vhat_w[j] = post_mean(x[j, i], market.mu[j], market.s[j], sig)
        b[i] -= market.r[j]; m[i] += 1; q[i] += market.os[j]
        return True
    start = rng.integers(n)
    active = set(idx)
    while active:
        for i in list(np.roll(idx, -start)):
            if i in active and not claim(i, True): active.discard(i)
    while True:
        short = [i for i in idx if m[i] < MIN_SQ]
        if not short or not any([claim(i, False) for i in short]): break
    res = dict(owner=owner, price=price, v=v, capped=capped, vhat_w=vhat_w, fill=fill, b=b, m=m, q=q,
               purse=c.purse, perm=perm, agents=agents)
    return res

# ---------------------------------------------------------------- evaluation
def team_utility(res, market, i, alpha=1.0, team_specific=False):
    mine = res["owner"] == i
    spent = res["purse"][i] - res["b"][i]
    if team_specific:
        R = market.ret[TEAMS[i]]
        vv = np.r_[R.value.values, res["v"][mine]]
        oo = np.r_[R.overseas.values, market.os[mine]]
        rr = list(R.role.values) + list(market.role[mine])
        f = squad_value(vv, oo, rr) - squad_value(R.value.values, R.overseas.values, list(R.role.values))
    else:
        f = res["v"][mine].sum()
    return f - alpha * spent

def curse_stats(res, auction_only=True):
    sold = res["owner"] >= 0
    if auction_only: sold &= ~res["fill"]
    v, p = res["v"][sold], res["price"][sold]
    return dict(overpay=float(np.mean(p - v)) if sold.any() else np.nan,
                belief_bias=float(np.mean(res["vhat_w"][sold] - v)) if sold.any() else np.nan,
                paid_over_value=float(p.sum() / v.sum()) if sold.any() else np.nan,
                capped_share=float(res["capped"][sold].mean()) if sold.any() else np.nan,
                n_sold=int(sold.sum()))

def hindsight_opt(res, market, i, alpha=1.0):
    """Eq. 10 with realised prices (unsold lots at reserve): best squad team i could have bought ex post."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    pt = np.where(res["owner"] >= 0, res["price"], market.r)
    v = res["v"]
    lo_n, hi_n = max(MIN_SQ - market.m0[i], 0), MAX_SQ - market.m0[i]
    A = np.vstack([pt, np.ones(market.K), market.os])
    lo = [0, lo_n, 0]; hi = [res["purse"][i], hi_n, MAX_OS - market.q0[i]]
    obj = -(v - alpha * pt)
    sol = milp(obj, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(market.K), bounds=Bounds(0, 1))
    return -sol.fun if sol.status == 0 else np.nan
