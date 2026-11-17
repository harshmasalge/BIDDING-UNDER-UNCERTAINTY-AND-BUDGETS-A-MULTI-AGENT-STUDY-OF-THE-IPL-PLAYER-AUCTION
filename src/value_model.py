"""Per-ball impact scores and the normal-normal empirical-Bayes model (Eq. 11 of the idea review)."""
import numpy as np, pandas as pd

W_WICKET = 5.0   # run value of a wicket (sensitivity analysed in experiments/fit_money_scale.py)

def ball_impacts(dl, w=W_WICKET):
    """Return batting and bowling impact rows (one per legal ball) with season-level league baselines."""
    legal_bat = dl[dl.wides == 0].copy()
    lg = legal_bat.groupby("year").agg(rpb=("runs_bat", "mean"), opb=("batter_out", "mean"))
    legal_bat = legal_bat.join(lg, on="year")
    legal_bat["z"] = legal_bat.runs_bat - legal_bat.rpb - w * (legal_bat.batter_out - legal_bat.opb)
    bowl = dl.copy()
    bowl["legal"] = ((bowl.wides == 0) & (bowl.noballs == 0)).astype(int)
    # conceded runs exclude byes/leg-byes: batter runs + wides + no-balls
    bowl["conc"] = bowl.runs_bat + bowl.wides + bowl.noballs
    lgb = bowl.groupby("year").apply(lambda g: pd.Series({"cpb": g.conc.sum() / g.legal.sum(),
                                                          "wpb": g.bowler_wkt.sum() / g.legal.sum()}))
    bowl = bowl.join(lgb, on="year")
    # impact per legal ball: runs saved vs league + wickets above league rate
    bowl["z"] = bowl.cpb * bowl.legal - bowl.conc + w * (bowl.bowler_wkt - bowl.wpb * bowl.legal)
    return legal_bat[["match", "year", "batter", "z"]].rename(columns={"batter": "player"}), \
           bowl[["match", "year", "bowler", "z", "legal"]].rename(columns={"bowler": "player"})

def player_window(bat, bowl, years):
    b = bat[bat.year.isin(years)].groupby("player").agg(n_bat=("z", "size"), s_bat=("z", "sum"), ss_bat=("z", lambda x: (x**2).sum()))
    o = bowl[bowl.year.isin(years)].groupby("player").agg(n_bowl=("legal", "sum"), s_bowl=("z", "sum"), ss_bowl=("z", lambda x: (x**2).sum()))
    matches = pd.concat([bat[bat.year.isin(years)][["player", "match"]], bowl[bowl.year.isin(years)][["player", "match"]]]).drop_duplicates()
    m = matches.groupby("player").size().rename("matches")
    return pd.concat([b, o, m], axis=1).fillna(0)

def eb_shrink(n, s, ss, min_n=30):
    """Normal-normal EB for a per-ball mean. Returns posterior mean and variance per player.
    Within-player per-ball variance varsigma^2 pooled; tau^2 by method of moments on players with n>=min_n."""
    n = np.asarray(n, float); s = np.asarray(s, float); ss = np.asarray(ss, float)
    zbar = np.where(n > 0, s / np.maximum(n, 1), 0.0)
    varsig = (ss.sum() - (s**2 / np.maximum(n, 1)).sum()) / max(n.sum() - (n > 0).sum(), 1)
    big = n >= min_n
    mu = np.average(zbar[big], weights=n[big])
    tau2 = max(np.average((zbar[big] - mu) ** 2, weights=n[big]) - varsig * big.sum() / n[big].sum(), 1e-6)
    w = np.where(n > 0, tau2 / (tau2 + varsig / np.maximum(n, 1)), 0.0)
    post = w * zbar + (1 - w) * mu
    pvar = np.where(n > 0, w * varsig / np.maximum(n, 1), tau2)
    return post, pvar, dict(mu=mu, tau2=tau2, varsig=varsig)

def impact_table(bat, bowl, auction_year, window=3):
    """Pre-auction per-match impact I_j = theta_bat*e_bat + theta_bowl*e_bowl with its posterior variance.
    Uses only seasons auction_year-window .. auction_year-1 (no look-ahead)."""
    years = list(range(auction_year - window, auction_year))
    t = player_window(bat, bowl, years)
    tb, vb, pb = eb_shrink(t.n_bat, t.s_bat, t.ss_bat)
    to, vo, po = eb_shrink(t.n_bowl, t.s_bowl, t.ss_bowl)
    m = np.maximum(t.matches.values, 1)
    k = 3.0   # exposure shrinkage: pseudo-matches at zero exposure
    e_bat = t.n_bat.values / (m + k); e_bowl = t.n_bowl.values / (m + k)
    out = pd.DataFrame({"player": t.index, "matches": t.matches.values, "balls_bat": t.n_bat.values,
                        "balls_bowl": t.n_bowl.values, "theta_bat": tb, "theta_bowl": to,
                        "e_bat": e_bat, "e_bowl": e_bowl})
    out["impact"] = tb * e_bat + to * e_bowl
    out["impact_var"] = e_bat**2 * vb + e_bowl**2 * vo
    out["stumpings"] = 0
    return out.set_index("player"), dict(bat=pb, bowl=po)

def season_impact(bat, bowl, year):
    """Realised per-match impact in a single season (no shrinkage) -- used for the ex-post check."""
    t = player_window(bat, bowl, [year])
    out = pd.DataFrame({"player": t.index, "matches": t.matches.values,
                        "real_impact_total": (t.s_bat + t.s_bowl).values})
    out["real_impact_pm"] = out.real_impact_total / np.maximum(out.matches, 1)
    return out.set_index("player")
