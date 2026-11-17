"""Tabular Q-learning bidder (RQ4): learns a shading multiplier kappa from interaction, with no opponent model.
State: (auction phase, budget per mandatory slot, squad need, lot quality); action: kappa in KAPPAS (0 = pass).
Reward on a won lot: v_j - alpha * p_j (true value revealed after the season); unsold/lost lots: 0."""
import numpy as np
from sim import Agent, post_mean, MIN_SQ, R_MIN

KAPPAS = np.array([0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25])

class QAgent(Agent):
    def __init__(self, Q=None, eps=0.1, lr=0.05, train=True, seed=0):
        self.Q = np.zeros((4, 5, 4, 4, len(KAPPAS))) if Q is None else Q
        self.eps, self.lr, self.train = eps, lr, train
        self.rng = np.random.default_rng(seed)
        self.name = "rl"

    def reset(self, ctx, i):
        super().reset(ctx, i)
        self.prev = None; self.t = 0; self.v = None

    def state(self, j, x, st):
        i, c = self.i, self.ctx
        phase = min(int(4 * self.t / c.K), 3)
        need = max(MIN_SQ - st["m"][i], 0)
        bps = st["b"][i] / max(need, 1)
        bb = int(np.digitize(bps, [1, 3, 6, 12]))
        nb = int(np.digitize(need, [1, 4, 9]))
        vh = post_mean(x, c.mu[j], c.s[j], c.sig)
        qb = int(np.digitize(vh, [1, 3, 8]))
        return (phase, bb, nb, qb), vh

    def bid(self, j, x, st):
        s, vh = self.state(j, x, st)
        if self.train and self.rng.random() < self.eps:
            a = self.rng.integers(len(KAPPAS))
        else:
            a = int(np.argmax(self.Q[s]))
        self._pending = (s, a, j)
        return KAPPAS[a] * vh

    def observe(self, j, winner, price, st):
        self.t += 1
        p = getattr(self, "_pending", None)
        if p is None or p[2] != j: return
        s, a, _ = p; self._pending = None
        r = (st["v_true"][j] - self.ctx.alpha * price) if winner == self.i else 0.0
        if self.prev is not None and self.train:
            ps, pa, pr = self.prev
            self.Q[ps + (pa,)] += self.lr * (pr + self.Q[s].max() - self.Q[ps + (pa,)])
        self.prev = (s, a, r)

    def finish(self, terminal_reward):
        if self.prev is not None and self.train:
            ps, pa, pr = self.prev
            self.Q[ps + (pa,)] += self.lr * (pr + terminal_reward - self.Q[ps + (pa,)])
        self.prev = None
