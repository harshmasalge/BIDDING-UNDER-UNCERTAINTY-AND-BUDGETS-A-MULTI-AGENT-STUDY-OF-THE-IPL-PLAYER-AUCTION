"""Feedback point 2: is a player worth the same to every team?  Values are made team-specific through each team's
best playing XI (retained players + purchases).  (a) heterogeneity of marginal values at the start of the auction;
(b) does bidding on own needs (need-aware pacing) beat common-value pacing when utility is the team's own XI value?"""
import sys, json
sys.path.insert(0, "src")
import numpy as np
from sim import Market, Pace, NeedAware, run_auction, team_utility, TEAMS
from bestxi import XITracker

SEEDS = range(60)

def heterogeneity(M, lam, seeds=range(20), frac=0.5):
    """Marginal best-XI value of the top-40 lots to each team, given the squads the teams hold after `frac` of
    the auction (pacing-equilibrium play). At the very start every squad has <11 players, so all ratios are 1."""
    Ev = np.exp(M.mu + M.s ** 2 / 2)
    top = np.argsort(-Ev)[:40]
    rel_all, cv_all, spread = [], [], []
    for s in seeds:
        r = run_auction(M, [Pace(l) for l in lam], sig=0.3, seed=s, order="fixed")
        pos = np.empty(M.K, int); pos[r["perm"]] = np.arange(M.K)
        held = (r["owner"] >= 0) & (pos < frac * M.K) & ~r["fill"]
        D = np.zeros((len(top), 10))
        for i, t in enumerate(TEAMS):
            mine = held & (r["owner"] == i)
            R = M.ret[t]
            trk = XITracker(np.r_[R.value.values, r["v"][mine]], np.r_[R.overseas.values, M.os[mine]],
                            list(R.role.values) + list(M.role[mine]))
            D[:, i] = [trk.marginal(Ev[j], M.os[j], M.role[j]) for j in top]
        rel = D / Ev[top, None]
        rel_all.append(rel); cv_all.append((D.std(1) / D.mean(1)).mean()); spread.append((rel.max(1) - rel.min(1)).mean())
    rel = np.mean(rel_all, 0)
    return dict(players=[str(M.name[j]) for j in top], common_value=Ev[top].tolist(), rel_mean=rel.tolist(),
                mean_min_ratio=float(np.mean([x.min(1).mean() for x in rel_all])),
                mean_max_ratio=float(np.mean([x.max(1).mean() for x in rel_all])),
                mean_spread=float(np.mean(spread)), cv_across_teams=float(np.mean(cv_all)), snapshot_fraction=frac)

def main():
    M = Market()
    lam = json.load(open("results/rq3_pacing_equilibrium.json"))["lambda_star"]
    out = {}
    out["heterogeneity"] = heterogeneity(M, lam)
    if "--het-only" in sys.argv:
        old = json.load(open("results/fb2_team_specific.json")); old["heterogeneity"] = out["heterogeneity"]
        json.dump(old, open("results/fb2_team_specific.json", "w"), indent=1); return
    print("heterogeneity", {k: v for k, v in out["heterogeneity"].items() if not isinstance(v, list)}, flush=True)
    res_t = {}
    for i, t in enumerate(TEAMS):
        U = {"pace": [], "need_pace": []}; Uc = {"pace": [], "need_pace": []}
        for s in SEEDS:
            for lab, mk in [("pace", lambda: Pace(lam[i])), ("need_pace", lambda: NeedAware(lam[i]))]:
                ag = [Pace(l) for l in lam]; ag[i] = mk()
                r = run_auction(M, ag, sig=0.3, seed=s, order="fixed")
                U[lab].append(team_utility(r, M, i, 1.0, team_specific=True))
                Uc[lab].append(team_utility(r, M, i, 1.0, team_specific=False))
        d = np.array(U["need_pace"]) - np.array(U["pace"])
        res_t[t] = dict(xi_utility_pace=float(np.mean(U["pace"])), xi_utility_need=float(np.mean(U["need_pace"])),
                        gain=float(d.mean()), gain_se=float(d.std() / np.sqrt(len(d))),
                        common_utility_pace=float(np.mean(Uc["pace"])), common_utility_need=float(np.mean(Uc["need_pace"])))
        print(t, {k: round(v, 2) for k, v in res_t[t].items()}, flush=True)
    out["focal"] = res_t
    # all teams need-aware vs all common-value pacing: league-wide XI utility
    for lab, mk in [("all_pace", lambda: [Pace(l) for l in lam]), ("all_need", lambda: [NeedAware(l) for l in lam])]:
        Ut = []
        for s in SEEDS:
            r = run_auction(M, mk(), sig=0.3, seed=s, order="fixed")
            Ut.append([team_utility(r, M, i, 1.0, team_specific=True) for i in range(10)])
        Ut = np.array(Ut)
        out[lab] = dict(mean_per_team=Ut.mean(0).tolist(), total=float(Ut.sum(1).mean()), total_se=float(Ut.sum(1).std() / np.sqrt(len(Ut))))
        print(lab, out[lab]["total"], flush=True)
    json.dump(out, open("results/fb2_team_specific.json", "w"), indent=1)

if __name__ == "__main__":
    main()
