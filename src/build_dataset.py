"""Build modelling tables: auction lots with pre-auction value features (no look-ahead)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from auctions import *
from value_model import ball_impacts, impact_table, season_impact

def infer_role(row, stumpers):
    if row.cs in stumpers: return "WK"
    bb, bo = row.get("balls_bat", 0) or 0, row.get("balls_bowl", 0) or 0
    if bb + bo == 0: return None
    f = bo / (bb + bo)
    return "BOWL" if f > 0.7 else ("BAT" if f < 0.15 else "AR")

def main():
    dl = pd.read_csv("data/processed/deliveries.csv.gz")
    M = Matcher(dl)
    bat, bowl = ball_impacts(dl)
    stumpers = set(dl.stumper.dropna()) - {""}
    sold = {y: load_sold(y) for y in range(2014, 2027)}
    # nationality / role lookup from every sold list (latest year wins)
    look = pd.concat([sold[y] for y in sorted(sold)])
    look["k"] = look.name.map(norm)
    nat = look.dropna(subset=["overseas"]).groupby("k").overseas.last().to_dict()
    rol = look.dropna(subset=["role"]).groupby("k").role.last().to_dict()

    def features(df, year):
        it, _ = impact_table(bat, bowl, year)
        df = df.copy()
        df["cs"] = match_list(M, df.name.tolist(), year - 1, df["capped"].values if "capped" in df else None)
        f = it.reindex(df.cs.fillna("__none__"))
        for c in ["matches", "balls_bat", "balls_bowl", "impact", "impact_var"]:
            df[c] = f[c].values
        df[["matches", "balls_bat", "balls_bowl"]] = df[["matches", "balls_bat", "balls_bowl"]].fillna(0)
        miss = df.impact.isna()
        df.loc[miss, "impact"] = 0.0
        # players with no record in the window get the prior variance of a typical player
        df.loc[miss, "impact_var"] = np.nan
        df["impact_var"] = df.impact_var.fillna(df.impact_var.quantile(0.95))
        df["k"] = df.name.map(norm)
        if "overseas" not in df or df.overseas.isna().any():
            o = df.k.map(nat)
            df["overseas"] = df["overseas"].fillna(o) if "overseas" in df else o
        df["role"] = df["role"] if "role" in df else None
        df["role"] = df.role.where(df.role.notna(), df.k.map(rol))
        df["role"] = [r if isinstance(r, str) else infer_role(row, stumpers) for r, (_, row) in zip(df.role, df.iterrows())]
        df["has_record"] = (df.matches > 0).astype(int)
        return df

    os.makedirs("data/processed", exist_ok=True)
    # training auctions (sold players only; prices in crore)
    tr = []
    for y in range(2018, 2025):
        d = sold[y].copy(); d["capped"] = np.nan
        d = features(d, y); d["purse"] = PURSE[y]; tr.append(d)
    tr = pd.concat(tr)
    tr["overseas_missing"] = tr.overseas.isna().astype(int)
    tr["overseas"] = tr.overseas.fillna(0)
    tr.to_csv("data/processed/train_sold_2018_2024.csv", index=False)
    # 2025 full lot list (test)
    L = load_2025_list()
    s25 = sold[2025].copy(); s25["k"] = s25.name.map(norm)
    L["overseas"] = L.name.map(norm).map(dict(zip(s25.k, s25.overseas)))
    L = features(L, 2025)
    # documented imputation for lots absent from every auction list and from Cricsheet:
    #   unknown nationality -> overseas if capped (Indian capped players all have IPL records), else Indian
    L["overseas_imputed"] = L.overseas.isna().astype(int)
    L["overseas"] = L.overseas.fillna(L.capped)
    #   unknown role -> deterministic hash draw from the role mix of known lots
    known = L.role.dropna().value_counts(normalize=True).sort_index()
    import zlib
    def draw(n):
        u = (zlib.crc32(n.encode()) % 10000) / 10000
        return known.index[np.searchsorted(known.cumsum().values, u)]
    L["role_imputed"] = L.role.isna().astype(int)
    L["role"] = [r if isinstance(r, str) else draw(n) for r, n in zip(L.role, L.name)]
    L["purse"] = PURSE[2025]
    # realised 2025-season performance (for the ex-post winner's-curse check only)
    real = season_impact(bat, bowl, 2025)
    cs_all = match_list(M, L.name.tolist(), 2025, L.capped.values)
    r = real.reindex(pd.Series(cs_all).fillna("__none__"))
    L["real_matches_2025"] = r.matches.fillna(0).values
    L["real_impact_pm_2025"] = r.real_impact_pm.values
    L.to_csv("data/processed/lots_2025.csv", index=False)
    print("train rows", len(tr), "by year", tr.year.value_counts().sort_index().to_dict())
    print("2025 lots", len(L), "with record", int(L.has_record.sum()), "| overseas known", int(L.overseas.notna().sum()),
          "| role known", int(L.role.notna().sum()))
    print("2025 roles", L.role.value_counts(dropna=False).to_dict())
    print("train overseas missing", int(tr.overseas.isna().sum()), "role missing", int(tr.role.isna().sum()))

if __name__ == "__main__":
    main()
