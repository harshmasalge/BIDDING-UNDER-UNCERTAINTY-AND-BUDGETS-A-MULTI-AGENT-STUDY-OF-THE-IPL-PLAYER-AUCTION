"""Load real auction lists and match auction names to Cricsheet names."""
import re, unicodedata
import numpy as np, pandas as pd

PURSE = {2018: 80, 2019: 82, 2020: 85, 2021: 85, 2022: 90, 2023: 95, 2024: 100, 2025: 120, 2026: 125}  # crore per team
TEAM_ABBR = {"chennai super kings": "CSK", "delhi capitals": "DC", "delhi daredevils": "DC", "gujarat titans": "GT",
             "kolkata knight riders": "KKR", "lucknow super giants": "LSG", "mumbai indians": "MI",
             "punjab kings": "PBKS", "kings xi punjab": "PBKS", "rajasthan royals": "RR",
             "royal challengers bangalore": "RCB", "royal challengers bengaluru": "RCB",
             "sunrisers hyderabad": "SRH"}
# Cricsheet names that cannot be recovered from surname + initial (manual alias table).
ALIAS = {"varun chakaravarthy": "CV Varun", "varun chakravarthy": "CV Varun",
         "ravichandaran ashwin": "R Ashwin", "ravichandran ashwin": "R Ashwin",
         "mohammad siraj": "Mohammed Siraj", "mohammed shami": "Mohammed Shami",
         "t natarajan": "T Natarajan", "phil salt": "PD Salt", "philip salt": "PD Salt",
         "jos buttler": "JC Buttler", "mitchell marsh": "MR Marsh", "mitch marsh": "MR Marsh",
         "abhinav manohar": "A Manohar", "anrich nortje": "A Nortje", "rahul chahar": "RD Chahar",
         "deepak chahar": "DL Chahar", "kl rahul": "KL Rahul", "suryakumar yadav": "SA Yadav",
         "glenn maxwell": "GJ Maxwell", "will jacks": "WG Jacks", "harshal patel": "HV Patel",
         "axar patel": "AR Patel", "kuldeep yadav": "Kuldeep Yadav", "umran malik": "Umran Malik",
         "mohit sharma": "MM Sharma", "ishant sharma": "I Sharma", "krunal pandya": "KH Pandya",
         "hardik pandya": "HH Pandya", "rashid khan": "Rashid Khan", "noor ahmad": "Noor Ahmad",
         "faf du plessis": "F du Plessis", "quinton de kock": "Q de Kock", "rassie van der dussen": "HE van der Dussen",
         "venkatesh iyer": "VR Iyer", "shreyas iyer": "SS Iyer", "arshdeep singh": "Arshdeep Singh",
         "yuzvendra chahal": "YS Chahal", "prasidh krishna": "M Prasidh Krishna", "washington sundar": "Washington Sundar",
         "nitish rana": "N Rana", "devdutt padikkal": "D Padikkal", "kagiso rabada": "K Rabada",
         "bhuvneshwar kumar": "B Kumar", "trent boult": "TA Boult", "josh hazlewood": "JR Hazlewood",
         "lockie ferguson": "LH Ferguson", "jofra archer": "JC Archer", "mitchell starc": "MA Starc",
         "liam livingstone": "LS Livingstone", "marcus stoinis": "MP Stoinis", "david miller": "DA Miller",
         "rishabh pant": "RR Pant", "ishan kishan": "Ishan Kishan", "jitesh sharma": "JM Sharma",
         "avesh khan": "Avesh Khan", "khaleel ahmed": "KK Ahmed", "mukesh kumar": "Mukesh Kumar",
         "tushar deshpande": "TU Deshpande", "akash deep": "Akash Deep", "shahbaz ahmed": "Shahbaz Ahmed",
         "abdul samad": "Abdul Samad", "rahul tripathi": "RA Tripathi", "ajinkya rahane": "AM Rahane",
         "manish pandey": "MK Pandey", "sai kishore": "R Sai Kishore", "r sai kishore": "R Sai Kishore",
         "mayank agarwal": "MA Agarwal", "prithvi shaw": "PP Shaw", "david warner": "DA Warner",
         "kane williamson": "KS Williamson", "jonny bairstow": "JM Bairstow", "steve smith": "SPD Smith",
         "shardul thakur": "SN Thakur", "umesh yadav": "UT Yadav", "piyush chawla": "PP Chawla",
         "rovman powell": "R Powell", "gerald coetzee": "G Coetzee", "marco jansen": "M Jansen",
         "spencer johnson": "SH Johnson", "jake fraser-mcgurk": "J Fraser-McGurk", "tim david": "TH David",
         "nehal wadhera": "N Wadhera", "harpreet brar": "Harpreet Brar", "shahrukh khan": "M Shahrukh Khan",
         "rahul tewatia": "R Tewatia", "abishek porel": "Abishek Porel", "ravi bishnoi": "Ravi Bishnoi",
         "mohsin khan": "Mohsin Khan", "ayush badoni": "A Badoni", "riyan parag": "R Parag",
         "dhruv jurel": "Dhruv Jurel", "sandeep sharma": "Sandeep Sharma", "rinku singh": "RK Singh",
         "harshit rana": "Harshit Rana", "ramandeep singh": "Ramandeep Singh", "yash dayal": "Yash Dayal",
         "rajat patidar": "RM Patidar", "abhishek sharma": "Abhishek Sharma", "nitish kumar reddy": "Nithish Kumar Reddy",
         "shashank singh": "Shashank Singh", "prabhsimran singh": "P Simran Singh", "mayank yadav": "MP Yadav",
         "tilak varma": "Tilak Varma", "sai sudharsan": "B Sai Sudharsan", "shubman gill": "Shubman Gill",
         "ruturaj gaikwad": "RD Gaikwad", "ravindra jadeja": "RA Jadeja", "shivam dube": "S Dube",
         "ms dhoni": "MS Dhoni", "matheesha pathirana": "M Pathirana", "tristan stubbs": "T Stubbs",
         "sunil narine": "SP Narine", "andre russell": "AD Russell", "nicholas pooran": "N Pooran",
         "jasprit bumrah": "JJ Bumrah", "rohit sharma": "RG Sharma", "sanju samson": "SV Samson",
         "yashasvi jaiswal": "YBK Jaiswal", "shimron hetmyer": "SO Hetmyer", "virat kohli": "V Kohli",
         "heinrich klaasen": "H Klaasen",
         "syed khaleel ahmed": "KK Ahmed", "shahbaz ahamad": "Shahbaz Ahmed", "dushmantha chameera": "PVD Chameera",
         "dushmanta chameera": "PVD Chameera", "wanindu hasaranga": "PWH de Silva", "kumar kartikeya singh": "K Kartikeya",
         "vyshak vijaykumar": "Vijaykumar Vyshak", "rasikh dar": "Rasikh Salam", "rasikh salam dar": "Rasikh Salam",
         "dinesh karthik": "KD Karthik", "n tilak varma": "Tilak Varma", "kamindu mendis": "PHKD Mendis",
         "bevan john jacobs": "DJ Jacobs", "bhanuka rajapaksa": "PBB Rajapaksa", "gurkeerat singh mann": "Gurkeerat Singh",
         "mujeeb zadran": "Mujeeb Ur Rahman", "mujeeb ur rahman": "Mujeeb Ur Rahman", "m s washington sundar": "Washington Sundar",
         "mayank agarawal": "MA Agarwal", "naveen ul haq": "Naveen-ul-Haq", "dasun shanaka": "MD Shanaka",
         "mohammad shami": "Mohammed Shami", "kusal mendis": None, "kusal perera": None, "pat cummins": "PJ Cummins", "travis head": "TM Head"}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\- ]", " ", s)).strip().lower()

def money(s):
    return float(str(s).replace(",", "").strip()) / 1e7  # rupees -> crore

def role_code(r):
    r = norm(r)
    if "keeper" in r: return "WK"
    if "all" in r: return "AR"
    if "bowl" in r: return "BOWL"
    if "bat" in r: return "BAT"
    return None

def load_sold(year):
    d = pd.read_csv(f"data/raw/auctions/IPL_Auction_{year}_Sold_Player.csv")
    out = pd.DataFrame({"name": d["Name"].map(lambda s: str(s).strip()), "year": year})
    pc = [c for c in d.columns if "Winning" in c or c.startswith("Price in rs")]
    out["price"] = d[pc[0]].map(money)
    out["team"] = d["TeamName"].map(lambda s: TEAM_ABBR.get(norm(s), norm(s)))
    out["role"] = d["Role"].map(role_code) if "Role" in d else None
    out["overseas"] = d["Nationality"].map(lambda s: int(norm(s) == "overseas")) if "Nationality" in d else np.nan
    if "BasePrices in Rs" in d: out["base"] = d["BasePrices in Rs"].map(money)
    return out

def load_2025_list():
    d = pd.read_csv("data/raw/IPL_2025_Auction.csv")
    return pd.DataFrame({"name": d["Player Name"].str.strip(), "base": d["Base Price"] / 1e7,
                         "price": d["Winning Bid"] / 1e7, "capped": (d["Capped/Uncapped"] == "Capped").astype(int),
                         "team": d["Team"], "sold": (d["Sold/Unsold"] == "Sold").astype(int)})

class Matcher:
    """Map auction-style full names to Cricsheet names (surname + first initial, then alias table)."""
    def __init__(self, deliveries):
        names = pd.unique(pd.concat([deliveries.batter, deliveries.bowler]))
        last_year = pd.concat([deliveries[["batter", "year"]].rename(columns={"batter": "n"}),
                               deliveries[["bowler", "year"]].rename(columns={"bowler": "n"})]).groupby("n").year.max()
        self.last_year = last_year.to_dict()
        self.first_year = pd.concat([deliveries[["batter", "year"]].rename(columns={"batter": "n"}),
                                     deliveries[["bowler", "year"]].rename(columns={"bowler": "n"})]).groupby("n").year.min().to_dict()
        self.names = set(names)
        self.by_surname = {}
        for n in names:
            self.by_surname.setdefault(norm(n).split(" ")[-1], []).append(n)

    def match(self, full, before_year=None):
        k = norm(full)
        if k in ALIAS:
            return ALIAS[k] if ALIAS[k] in self.names else None
        toks = k.split(" ")
        toks = k.split(" ")
        if full.strip() in self.names: return full.strip()
        cands = []
        for n in self.by_surname.get(toks[-1], []):
            nt = n.split(" ")
            first = nt[0]
            if first.isupper() and len(nt) > 1:      # initials, e.g. "RG Sharma"
                ok = first[0].lower() == toks[0][0]
                mids = [norm(t) for t in nt[1:-1]]      # e.g. "R Sanjay Yadav": middle name must appear
                ok = ok and all(m in toks for m in mids)
            else:
                f = norm(first)
                ok = f == toks[0] or (min(len(f), len(toks[0])) >= 5 and f[:5] == toks[0][:5])
            if ok: cands.append(n)
        if before_year is not None and len(cands) > 1:   # keep candidates active before the auction
            pre = [c for c in cands if self.first_year.get(c, 9999) <= before_year]
            cands = pre or cands
        if len(cands) == 1: return cands[0]
        if len(cands) > 1:   # prefer the most recently active candidate
            cands.sort(key=lambda c: -self.last_year.get(c, 0))
            if self.last_year[cands[0]] > self.last_year[cands[1]]: return cands[0]
        return None


def match_list(matcher, names, before_year, capped=None):
    """Match a list of names; a Cricsheet name claimed by several auction names is kept only for
    alias/exact matches (or the single capped claimant), otherwise dropped."""
    out = [matcher.match(n, before_year) for n in names]
    strong = [norm(n) in ALIAS or n.strip() in matcher.names for n in names]
    from collections import defaultdict
    claims = defaultdict(list)
    for i, c in enumerate(out):
        if c is not None: claims[c].append(i)
    for c, idx in claims.items():
        if len(idx) == 1: continue
        keep = [i for i in idx if strong[i]]
        if len(keep) != 1 and capped is not None:
            keep = [i for i in idx if capped[i] == 1]
        for i in idx:
            if len(keep) != 1 or i != keep[0]: out[i] = None
    return out
