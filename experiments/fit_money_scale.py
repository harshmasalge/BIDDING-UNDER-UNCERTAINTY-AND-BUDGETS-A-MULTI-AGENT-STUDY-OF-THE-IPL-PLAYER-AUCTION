"""Feedback point 1: fit the impact->money scale on EARLIER auctions and test it on 2025.

y = log(price / purse) = b0 + b1*impact + b2*log1p(matches) + b3*has_record + b4*overseas + role dummies + e
Primary model: trained on the 2022 mega auction only. Robustness: pooled 2018-2024.
The 2025-in-sample fit is reported only to show how optimistic the non-independent check was."""
import sys, os, json
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy import stats
from value_model import ball_impacts, impact_table

FEATS = ["impact", "lmatch", "has_record", "overseas", "AR", "BOWL", "WK"]

def design(d):
    X = pd.DataFrame({"const": 1.0, "impact": d.impact, "lmatch": np.log1p(d.matches), "has_record": d.has_record,
                      "overseas": d.overseas.astype(float)})
    for r in ["AR", "BOWL", "WK"]: X[r] = (d.role == r).astype(float)
    return X

def ols(X, y):
    b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    res = y.values - X.values @ b
    return b, res

def evaluate(b, d):
    X = design(d); y = np.log(d.price / d.purse)
    pred = X.values @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rho = stats.spearmanr(pred, y).correlation
    return dict(n=int(len(d)), r2_log=float(r2), spearman=float(rho), mae_log=float(np.abs(y - pred).mean()),
                bias_log=float((y - pred).mean()))

def refeature(d, bat, bowl, year):
    it, _ = impact_table(bat, bowl, year)
    f = it.reindex(d.cs.fillna("__none__"))
    d = d.copy(); d["impact"] = f.impact.fillna(0.0).values
    return d

def main():
    tr = pd.read_csv("data/processed/train_sold_2018_2024.csv")
    L = pd.read_csv("data/processed/lots_2025.csv")
    test = L[L.sold == 1].copy()
    out = {}
    # (a) choose the wicket run-value W by leave-one-out fit on the TRAINING auction (2022) only
    dl = pd.read_csv("data/processed/deliveries.csv.gz")
    wsel = {}
    for w in [0, 5, 10, 15, 20, 30]:
        bat, bowl = ball_impacts(dl, w)
        d22 = refeature(tr[tr.year == 2022], bat, bowl, 2022)
        X, y = design(d22), np.log(d22.price / d22.purse)
        H = X.values @ np.linalg.pinv(X.values.T @ X.values) @ X.values.T
        b, res = ols(X, y)
        loo = res / (1 - np.diag(H))
        wsel[w] = float(np.mean(loo ** 2))
    out["wicket_value_LOO_mse_2022"] = wsel
    w_best = min(wsel, key=wsel.get)
    out["wicket_value_selected"] = w_best
    bat, bowl = ball_impacts(dl, w_best)
    tr = pd.concat([refeature(tr[tr.year == y], bat, bowl, y) for y in sorted(tr.year.unique())])
    test = refeature(test, bat, bowl, 2025); L = refeature(L, bat, bowl, 2025)
    # impact posterior variances for the selected W
    it25, _ = impact_table(bat, bowl, 2025)
    iv = it25.reindex(L.cs.fillna("__none__")).impact_var
    L["impact_var"] = iv.fillna(iv.quantile(0.95)).values

    fits = {"train_2022": tr[tr.year == 2022], "train_2018_2024": tr, "insample_2025": test}
    coefs = {}
    for name, d in fits.items():
        X, y = design(d), np.log(d.price / d.purse)
        b, res = ols(X, y)
        coefs[name] = b
        out[name] = dict(coef=dict(zip(X.columns, map(float, b))), resid_sd=float(res.std(ddof=X.shape[1])),
                         train=evaluate(b, d), test_2025=evaluate(b, test))
        print(f"{name:16s} train R2={out[name]['train']['r2_log']:.3f}  ->  2025 R2={out[name]['test_2025']['r2_log']:.3f} "
              f"rho={out[name]['test_2025']['spearman']:.3f} bias={out[name]['test_2025']['bias_log']:+.3f}")
    # value prior for every 2025 lot from the PRIMARY (2022) fit, rescaled to the 2025 purse
    b = coefs["train_2022"]; s_res = out["train_2022"]["resid_sd"]
    d22 = fits["train_2022"]
    it22, _ = impact_table(bat, bowl, 2022)
    v22 = it22.reindex(d22.cs.fillna("__none__")).impact_var
    v22 = v22.fillna(v22.quantile(0.95))
    s0sq = max(s_res ** 2 - b[1] ** 2 * v22.mean(), 0.05)
    for name in ["train_2022", "train_2018_2024", "insample_2025"]:
        bb = coefs[name]
        L[f"mu_log_{name}"] = design(L).values @ bb + np.log(L.purse)
    L["mu_log"] = L["mu_log_train_2022"]
    L["s_log"] = np.sqrt(s0sq + b[1] ** 2 * L.impact_var)
    out["prior_s0"] = float(np.sqrt(s0sq))
    out["prior_s_log_mean"] = float(L.s_log.mean())

    # Ex-post check on REAL data: did buyers pay for more performance than they got?
    s = L[(L.sold == 1) & (L.real_matches_2025 >= 5)].copy()
    X = design(s); other = X.values @ b - b[1] * s.impact
    s["premium_log"] = np.log(s.price) - s.mu_log
    s["surprise"] = s.real_impact_pm_2025 - s.impact          # realised minus pre-auction posterior mean
    q = pd.qcut(s.premium_log, 3, labels=["low", "mid", "high"])
    lr = stats.linregress(s.premium_log, s.surprise)
    out["expost_2025"] = dict(
        n=int(len(s)), mean_surprise=float(s.surprise.mean()), se=float(s.surprise.std() / np.sqrt(len(s))),
        by_premium_tercile={k: dict(surprise=float(g.mean()), se=float(g.std() / np.sqrt(len(g))), n=int(len(g)),
                                    premium=float(s.premium_log[q == k].mean()))
                            for k, g in s.surprise.groupby(q, observed=True)},
        slope=float(lr.slope), slope_ci95=[float(lr.slope - 1.96 * lr.stderr), float(lr.slope + 1.96 * lr.stderr)],
        slope_if_premium_justified=float(1 / b[1]), slope_pvalue=float(lr.pvalue))
    print("ex-post:", json.dumps(out["expost_2025"], indent=1)[:900])
    s.to_csv("results/expost_2025.csv", index=False)
    L.to_csv("data/processed/lots_2025_valued.csv", index=False)
    json.dump(out, open("results/money_scale.json", "w"), indent=2)

if __name__ == "__main__":
    main()
