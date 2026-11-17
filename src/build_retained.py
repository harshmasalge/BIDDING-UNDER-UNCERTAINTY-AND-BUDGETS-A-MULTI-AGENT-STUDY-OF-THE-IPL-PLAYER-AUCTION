"""Value the 46 retained players with the same (2022-fitted) money scale, for the best-XI experiments."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from auctions import Matcher, ALIAS, norm, PURSE
from value_model import ball_impacts, impact_table
from teams import RETAINED, OVERSEAS_RETAINED, ROLE_RETAINED

def main():
    js = json.load(open("results/money_scale.json"))
    co = js["train_2022"]["coef"]
    dl = pd.read_csv("data/processed/deliveries.csv.gz")
    bat, bowl = ball_impacts(dl, js["wicket_value_selected"])
    it, _ = impact_table(bat, bowl, 2025)
    M = Matcher(dl)
    rows = []
    for t, (purse, names) in RETAINED.items():
        for nm in names:
            cs = M.match(nm, 2024)
            f = it.loc[cs] if cs in it.index else None
            imp = f.impact if f is not None else 0.0
            mt = f.matches if f is not None else 0
            os_ = int(nm in OVERSEAS_RETAINED); role = ROLE_RETAINED[nm]
            y = co["const"] + co["impact"] * imp + co["lmatch"] * np.log1p(mt) + co["has_record"] * (mt > 0) \
                + co["overseas"] * os_ + co.get(role, 0.0) * (role in ("AR", "BOWL", "WK"))
            rows.append(dict(team=t, name=nm, cs=cs, impact=imp, matches=mt, overseas=os_, role=role,
                             value=float(np.exp(y + np.log(PURSE[2025])))))
    R = pd.DataFrame(rows)
    R.to_csv("data/processed/retained_2025.csv", index=False)
    print(R.groupby("team").agg(n=("name", "size"), os=("overseas", "sum"), value=("value", "sum")).round(2))
    print("unmatched:", R[R.cs.isna()].name.tolist())

if __name__ == "__main__":
    main()
