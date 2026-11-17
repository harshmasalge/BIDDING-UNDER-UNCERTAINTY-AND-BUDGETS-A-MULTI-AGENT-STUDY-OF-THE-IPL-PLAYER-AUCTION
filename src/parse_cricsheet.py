"""Parse Cricsheet IPL JSON (v1.2.0) into one row per delivery."""
import json, glob, os, sys
import pandas as pd

def parse(json_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(json_dir, "*.json"))):
        m = json.load(open(f))
        info = m["info"]
        date = info["dates"][0]
        year = int(date[:4])
        mid = os.path.basename(f)[:-5]
        for inn_no, inn in enumerate(m["innings"]):
            if inn.get("super_over"):
                continue
            bat_team = inn["team"]
            for ov in inn["overs"]:
                for d in ov["deliveries"]:
                    ex = d.get("extras", {})
                    wides, nb = ex.get("wides", 0), ex.get("noballs", 0)
                    wk = d.get("wickets", [])
                    out_batter = [w["player_out"] for w in wk]
                    bowler_wkt = sum(w["kind"] not in ("run out", "retired hurt", "retired out", "obstructing the field") for w in wk)
                    stumper = [fl.get("name") for w in wk if w["kind"] == "stumped" for fl in w.get("fielders", [])]
                    rows.append((mid, year, inn_no, bat_team, ov["over"], d["batter"], d["bowler"],
                                 d["runs"]["batter"], d["runs"]["total"], wides, nb,
                                 int(d["batter"] in out_batter), bowler_wkt, stumper[0] if stumper else ""))
    cols = ["match", "year", "innings", "bat_team", "over", "batter", "bowler", "runs_bat", "runs_total",
            "wides", "noballs", "batter_out", "bowler_wkt", "stumper"]
    return pd.DataFrame(rows, columns=cols)

if __name__ == "__main__":
    df = parse(sys.argv[1])
    df.to_csv(sys.argv[2], index=False, compression="gzip")
    print(len(df), "deliveries,", df.match.nunique(), "matches,", df.year.min(), "-", df.year.max())
    b = df[df.wides == 0].groupby("batter").runs_bat.sum().sort_values()
    print("top run scorers:", b.tail(3).to_dict())
