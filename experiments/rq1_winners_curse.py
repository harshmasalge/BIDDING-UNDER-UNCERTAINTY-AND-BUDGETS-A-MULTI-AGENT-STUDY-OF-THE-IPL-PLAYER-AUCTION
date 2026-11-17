"""RQ1 + feedback point 3: winner's curse vs signal noise, budget tightness and number of bidders,
for naive bidders and equilibrium (Eq. 5) bidders; plus a single-lot numerical check of Eq. 5."""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from sim import Market, Agent, Equilibrium, run_auction, curse_stats, beta_eq, post_mean

SEEDS = range(100)

def beta_chunked(z, mu, s, sig, n, ch=20000):
    z = np.asarray(z).ravel()
    return np.concatenate([beta_eq(z[i:i + ch], mu, s, sig, n) for i in range(0, len(z), ch)])

def single_lot_check(n=10, mu=0.0, s=0.9, sig=0.3, N=60_000, seed=0):
    """Second-price auction, one lot. (i) naive winners overpay; (ii) beta is a symmetric equilibrium:
    a single bidder scaling its bid by k gains nothing when the others bid beta."""
    rng = np.random.default_rng(seed)
    u = mu + s * rng.standard_normal(N); v = np.exp(u)
    x = u[:, None] + sig * rng.standard_normal((N, n))
    out = {}
    for lab, f in [("naive", lambda z: post_mean(z, mu, s, sig)), ("beta", lambda z: beta_chunked(z, mu, s, sig, n).reshape(z.shape))]:
        bids = f(x)
        srt = np.sort(bids, 1)
        p = srt[:, -2]; win = bids.argmax(1)
        out[lab] = dict(overpay=float((p - v).mean()), winner_profit=float((v - p).mean()),
                        belief_bias=float((post_mean(x[np.arange(N), win], mu, s, sig) - v).mean()))
    # deviation check for bidder 0 when others bid beta
    others = beta_chunked(x[:, 1:], mu, s, sig, n).reshape(N, n - 1).max(1)
    b0 = beta_chunked(x[:, 0], mu, s, sig, n)
    dev = {}
    for k in [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
        w = k * b0 > others
        pay = np.where(w, v - others, 0.0)
        dev[k] = (float(pay.mean()), float(pay.std() / np.sqrt(N)))
    out["deviation_payoff_bidder0"] = dev
    return out

def main():
    M = Market()
    out = {"single_lot": single_lot_check()}
    print("single lot:", out["single_lot"]["naive"], out["single_lot"]["beta"])
    grid = {}
    for pop in ["naive", "equilibrium"]:
        for scale in [0.5, 1.0, 3.0, 20.0]:
            for sig in [0.0, 0.15, 0.3, 0.5, 0.75]:
                st = []
                for s in SEEDS:
                    ag = [Agent() if pop == "naive" else Equilibrium() for _ in range(10)]
                    st.append(curse_stats(run_auction(M, ag, sig=sig, seed=s, purse_scale=scale, order="fixed")))
                agg = {k: (float(np.nanmean([d[k] for d in st])), float(np.nanstd([d[k] for d in st]) / np.sqrt(len(st))))
                       for k in st[0]}
                grid[f"{pop}|{scale}|{sig}"] = agg
                print(pop, scale, sig, {k: round(v[0], 3) for k, v in agg.items()}, flush=True)
    out["grid"] = grid
    nt = {}
    for pop in ["naive", "equilibrium"]:
        for n in [4, 6, 8, 10]:
            st = [curse_stats(run_auction(M, [Agent() if pop == "naive" else Equilibrium() for _ in range(n)], sig=0.3,
                                          seed=s, purse_scale=20.0, order="fixed")) for s in SEEDS]
            nt[f"{pop}|{n}"] = {k: (float(np.nanmean([d[k] for d in st])), float(np.nanstd([d[k] for d in st]) / 10)) for k in st[0]}
            print(pop, n, {k: round(v[0], 3) for k, v in nt[f'{pop}|{n}'].items()}, flush=True)
    out["n_teams_slack"] = nt
    json.dump(out, open("results/rq1_winners_curse.json", "w"), indent=1)

if __name__ == "__main__":
    main()
