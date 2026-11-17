"""Team-specific squad value f_i(S): value of the best playing XI under IPL selection rules plus a bench term.
XI rules: 11 players, <=4 overseas, >=1 wicket-keeper, >=5 bowling options (BOWL/AR), >=4 batters (BAT/WK)."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

ROLES = ("BAT", "WK", "AR", "BOWL")
BENCH = 0.2   # weight on non-XI players (injury/rotation cover)

def _xi_greedy(v, os_, role):
    """Greedy in value order with feasibility look-ahead; exact on >99% of squads (see tests)."""
    n = len(v)
    if n == 0: return 0.0, []
    order = np.argsort(-np.asarray(v))
    take, cnt = [], dict(os=0, wk=0, bowl=0, bat=0)
    need = lambda c: max(1 - c["wk"], 0) + max(5 - c["bowl"], 0) + max(4 - c["bat"], 0)
    # available supply for each requirement (so we never demand the impossible)
    sup = dict(wk=sum(r == "WK" for r in role), bowl=sum(r in ("BOWL", "AR") for r in role),
               bat=sum(r in ("BAT", "WK") for r in role))
    req = dict(wk=min(1, sup["wk"]), bowl=min(5, sup["bowl"]), bat=min(4, sup["bat"]))
    def deficit(c):
        return max(req["wk"] - c["wk"], 0) + max(req["bowl"] - c["bowl"], 0) + max(req["bat"] - c["bat"], 0)
    k = min(11, n)
    for idx in order:
        if len(take) == k: break
        r, o = role[idx], os_[idx]
        if o and cnt["os"] >= 4: continue
        c = dict(cnt); c["os"] += o
        c["wk"] += r == "WK"; c["bowl"] += r in ("BOWL", "AR"); c["bat"] += r in ("BAT", "WK")
        if deficit(c) <= k - len(take) - 1 or deficit(c) < deficit(cnt):
            take.append(idx); cnt = c
    if len(take) < k:   # relax: fill remaining slots with best leftovers respecting overseas cap
        for idx in order:
            if len(take) == k: break
            if idx in take or (os_[idx] and cnt["os"] >= 4): continue
            take.append(idx); cnt["os"] += os_[idx]
    return float(np.sum(np.asarray(v)[take])), take

# ---- exact best XI by dynamic programming over count states (fast, used in the simulator) ----
_SH = (12, 5, 2, 6, 5)          # picked, overseas, wk(capped 1), bowling options(capped 5), batters(capped 4)
_N = int(np.prod(_SH))
_IDX = np.arange(_N).reshape(_SH)
_DEST = {}
for _o in (0, 1):
    for _r in ROLES:
        d = np.full(_SH, -1)
        wk, bo, ba = int(_r == "WK"), int(_r in ("BOWL", "AR")), int(_r in ("BAT", "WK"))
        for c in range(11):
            for o in range(5 - _o):
                for w in range(2):
                    for b in range(6):
                        for t in range(5):
                            d[c, o, w, b, t] = _IDX[c + 1, o + _o, min(w + wk, 1), min(b + bo, 5), min(t + ba, 4)]
        _DEST[(_o, _r)] = d.ravel()

def best_xi(v, os_, role):
    """Exact value of the best XI (DP over count states)."""
    n = len(v)
    if n == 0: return 0.0
    k = min(11, n)
    wk = sum(r == "WK" for r in role); bo = sum(r in ("BOWL", "AR") for r in role); ba = sum(r in ("BAT", "WK") for r in role)
    req = (min(1, wk), min(5, bo), min(4, ba))
    dp = np.full(_N, -np.inf); dp[0] = 0.0
    for val, o, r in zip(v, os_, role):
        d = _DEST[(int(o), r)]
        m = d >= 0
        cand = dp[m] + val
        new = dp.copy()
        np.maximum.at(new, d[m], cand)
        dp = new
    dp = dp.reshape(_SH)[k, :, req[0]:, req[1]:, req[2]:]
    best = dp.max()
    if not np.isfinite(best):      # requirements infeasible together: best XI under the overseas cap only
        best = np.full(_N, -np.inf)
        return _xi_greedy(v, os_, role)[0]
    return float(best)

def squad_value(v, os_, role, bench=BENCH):
    xi = best_xi(v, os_, role)
    return xi + bench * (float(np.sum(v)) - xi)

def xi_milp(v, os_, role):
    """Exact best XI by integer programming (used to verify the greedy)."""
    v = np.asarray(v, float); n = len(v)
    k = min(11, n)
    os_ = np.asarray(os_, float)
    wk = np.array([r == "WK" for r in role], float); bo = np.array([r in ("BOWL", "AR") for r in role], float)
    ba = np.array([r in ("BAT", "WK") for r in role], float)
    A = np.vstack([np.ones(n), os_, wk, bo, ba])
    lo = [k, 0, min(1, wk.sum()), min(5, bo.sum()), min(4, ba.sum())]
    hi = [k, 4, n, n, n]
    res = milp(-v, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1))
    return -res.fun if res.status == 0 else np.nan


class XITracker:
    """Keeps the DP table for a squad so the marginal best-XI value of one more player costs one DP step."""
    def __init__(self, v=(), os_=(), role=(), bench=BENCH):
        self.bench = bench
        self.dp = np.full(_N, -np.inf); self.dp[0] = 0.0
        self.n = 0; self.total = 0.0; self.cnt = np.zeros(3, int)
        for a, b, c in zip(v, os_, role): self.add(a, b, c)
        self.cur = self._value(self.dp, self.n, self.cnt, self.total)

    @staticmethod
    def _step(dp, val, o, r):
        d = _DEST[(int(o), r)]; m = d >= 0
        new = dp.copy(); np.maximum.at(new, d[m], dp[m] + val)
        return new

    def _value(self, dp, n, cnt, total):
        if n == 0: return 0.0
        k = min(11, n)
        req = (min(1, cnt[0]), min(5, cnt[1]), min(4, cnt[2]))
        best = dp.reshape(_SH)[k, :, req[0]:, req[1]:, req[2]:].max()
        if not np.isfinite(best): best = dp.reshape(_SH)[:k + 1].max()   # overseas cap forces a smaller XI
        return float(best) + self.bench * (total - float(best))

    @staticmethod
    def _cnt(r):
        return np.array([r == "WK", r in ("BOWL", "AR"), r in ("BAT", "WK")], int)

    def marginal(self, val, o, r):
        new = self._step(self.dp, val, o, r)
        return self._value(new, self.n + 1, self.cnt + self._cnt(r), self.total + val) - self.cur

    def add(self, val, o, r):
        self.dp = self._step(self.dp, val, o, r)
        self.n += 1; self.total += val; self.cnt += self._cnt(r)
        self.cur = self._value(self.dp, self.n, self.cnt, self.total)
