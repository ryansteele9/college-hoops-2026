"""
Build master team table: one row per team per tournament year.
Spine: KenPom Barttorvik.csv

Two feature groups:

1. Current-season metrics (KenPom Barttorvik.csv numeric cols):
   Efficiency ratings, four factors, tempo, shooting splits, SOS, etc.
   These are legitimately available before the tournament and carry no leakage.

2. Historical aggregate features (computed from Tournament Matchups.csv):
   PROG_*, SEED_*, CONF_* — computed using only years strictly before each
   team's tournament year to eliminate the all-time static table leakage.

Output: data/processed/master_team_table.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load files ──────────────────────────────────────────────────────
kenpom       = pd.read_csv(RAW / "KenPom Barttorvik.csv")
matchups_raw = pd.read_csv(RAW / "Tournament Matchups.csv")

# ── Step 2: Build spine ─────────────────────────────────────────────────────
SPINE_COLS = ["YEAR", "TEAM NO", "TEAM ID", "TEAM", "SEED", "ROUND", "CONF", "CONF ID"]
spine = kenpom[SPINE_COLS].copy()
CONF_REMAP = {"P10": "P12", "SInd": "Slnd"}
spine["CONF"] = spine["CONF"].replace(CONF_REMAP)

# Current-season KenPom/Barttorvik metrics: all numeric columns except spine
# identifiers and rank columns (rank cols are ordinal transforms of the raw
# values — we keep the raw metrics and let feature selection handle redundancy).
_NON_FEAT = set(SPINE_COLS) | {"QUAD NO", "QUAD ID"}
KENPOM_FEAT_COLS = [
    c for c in kenpom.columns
    if c not in _NON_FEAT
    and "RANK" not in c
    and kenpom[c].dtype in ("int64", "float64")
]

# ── Step 3: Parse Tournament Matchups into per-team game results ─────────────
# Each consecutive pair of rows (sorted by BY YEAR NO desc) is one game.
m = matchups_raw.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
m["GAME_ID"] = m.index // 2
m["SLOT"]    = m.index % 2   # 0 = Team A (higher BY YEAR NO), 1 = Team B

ga = (m[m["SLOT"] == 0]
      [["YEAR", "GAME_ID", "TEAM NO", "SEED", "SCORE", "CURRENT ROUND"]]
      .rename(columns={"TEAM NO": "TNO_A", "SEED": "SEED_A",
                       "SCORE": "SCORE_A", "CURRENT ROUND": "CUR_ROUND"}))
gb = (m[m["SLOT"] == 1]
      [["YEAR", "GAME_ID", "TEAM NO", "SEED", "SCORE"]]
      .rename(columns={"TEAM NO": "TNO_B", "SEED": "SEED_B", "SCORE": "SCORE_B"}))

games = ga.merge(gb, on=["YEAR", "GAME_ID"])
games["A_WIN"] = (games["SCORE_A"] > games["SCORE_B"]).astype(int)
games["B_WIN"] = 1 - games["A_WIN"]

# Stack into one row per (team, game)
rA = (games[["YEAR", "TNO_A", "SEED_A", "A_WIN", "CUR_ROUND"]]
      .rename(columns={"TNO_A": "TEAM_NO", "SEED_A": "SEED", "A_WIN": "WIN"}))
rB = (games[["YEAR", "TNO_B", "SEED_B", "B_WIN", "CUR_ROUND"]]
      .rename(columns={"TNO_B": "TEAM_NO", "SEED_B": "SEED", "B_WIN": "WIN"}))
game_records = pd.concat([rA, rB], ignore_index=True)

# Round-achievement flags
game_records["F4"]    = (game_records["CUR_ROUND"] <= 4).astype(int)
game_records["CHAMP"] = ((game_records["CUR_ROUND"] == 2) & (game_records["WIN"] == 1)).astype(int)

# Attach stable TEAM ID and CONF from kenpom (join on YEAR + TEAM_NO)
team_id_map = (kenpom[["YEAR", "TEAM NO", "TEAM ID", "CONF"]].copy()
               .rename(columns={"TEAM NO": "TEAM_NO"}))
team_id_map["CONF"] = team_id_map["CONF"].replace(CONF_REMAP)
game_records = game_records.merge(team_id_map, on=["YEAR", "TEAM_NO"], how="left")

# ── Step 4: Compute per-(group, year) aggregate stats ───────────────────────

# Program (TEAM ID) — one tournament appearance per year
team_yr = (game_records.groupby(["YEAR", "TEAM ID"])
           .agg(GAMES=("WIN", "count"),
                W    =("WIN", "sum"),
                F4   =("F4",  "max"),   # 1 if reached Final Four this year
                CHAMP=("CHAMP","max"))  # 1 if won championship this year
           .reset_index())
team_yr["L"]   = team_yr["GAMES"] - team_yr["W"]
team_yr["APP"] = 1  # one tournament appearance per row

# Seed — aggregate all teams sharing this seed per year
seed_yr = (game_records.groupby(["YEAR", "SEED"])
           .agg(GAMES=("WIN",    "count"),
                W    =("WIN",    "sum"),
                CHAMP=("CHAMP",  "max"),
                APP  =("TEAM_NO","nunique"))  # distinct teams appearing as this seed
           .reset_index())
seed_yr["L"] = seed_yr["GAMES"] - seed_yr["W"]

# Conference — aggregate all conference teams per year
conf_yr = (game_records.dropna(subset=["CONF"])
           .groupby(["YEAR", "CONF"])
           .agg(GAMES=("WIN",    "count"),
                W    =("WIN",    "sum"),
                CHAMP=("CHAMP",  "max"),
                APP  =("TEAM_NO","nunique"))
           .reset_index())
conf_yr["L"] = conf_yr["GAMES"] - conf_yr["W"]

# ── Step 5: Expanding prior-year cumulative sums ────────────────────────────
# cumsum() includes the current year; subtracting the current year's value
# gives the sum of all strictly prior years — leak-free.

def add_prior_cumstats(df, group_col, stat_cols):
    """Add prev_<col> = cumulative sum using only prior years per group."""
    df = df.sort_values([group_col, "YEAR"]).copy()
    for col in stat_cols:
        cumsum = df.groupby(group_col)[col].cumsum()
        df[f"prev_{col}"] = cumsum - df[col]
    return df

PROG_STATS = ["GAMES", "W", "L", "F4", "CHAMP", "APP"]
SEED_STATS = ["GAMES", "W", "L", "CHAMP", "APP"]
CONF_STATS = ["GAMES", "W", "L", "CHAMP", "APP"]

team_yr = add_prior_cumstats(team_yr, "TEAM ID", PROG_STATS)
seed_yr = add_prior_cumstats(seed_yr, "SEED",    SEED_STATS)
conf_yr = add_prior_cumstats(conf_yr, "CONF",    CONF_STATS)

# ── Step 6: Compute rate features ───────────────────────────────────────────

def safe_div(a, b):
    return np.where(b > 0, a / b, np.nan)

# PROG_
team_yr["PROG_GAMES"]  = team_yr["prev_GAMES"]
team_yr["PROG_W"]      = team_yr["prev_W"]
team_yr["PROG_L"]      = team_yr["prev_L"]
team_yr["PROG_WIN%"]   = safe_div(team_yr["prev_W"],     team_yr["prev_GAMES"])
team_yr["PROG_F4%"]    = safe_div(team_yr["prev_F4"],    team_yr["prev_APP"])
team_yr["PROG_CHAMP%"] = safe_div(team_yr["prev_CHAMP"], team_yr["prev_APP"])

# SEED_
seed_yr["SEED_GAMES"]  = seed_yr["prev_GAMES"]
seed_yr["SEED_W"]      = seed_yr["prev_W"]
seed_yr["SEED_L"]      = seed_yr["prev_L"]
seed_yr["SEED_WIN%"]   = safe_div(seed_yr["prev_W"],     seed_yr["prev_GAMES"])
seed_yr["SEED_CHAMP%"] = safe_div(seed_yr["prev_CHAMP"], seed_yr["prev_APP"])

# CONF_
conf_yr["CONF_GAMES"]  = conf_yr["prev_GAMES"]
conf_yr["CONF_W"]      = conf_yr["prev_W"]
conf_yr["CONF_L"]      = conf_yr["prev_L"]
conf_yr["CONF_WIN%"]   = safe_div(conf_yr["prev_W"],     conf_yr["prev_GAMES"])
conf_yr["CONF_CHAMP%"] = safe_div(conf_yr["prev_CHAMP"], conf_yr["prev_APP"])

# ── Step 7: Build slim lookup tables ────────────────────────────────────────
PROG_FEAT = ["PROG_GAMES", "PROG_W", "PROG_L", "PROG_WIN%", "PROG_F4%", "PROG_CHAMP%"]
SEED_FEAT = ["SEED_GAMES", "SEED_W", "SEED_L", "SEED_WIN%", "SEED_CHAMP%"]
CONF_FEAT = ["CONF_GAMES", "CONF_W", "CONF_L", "CONF_WIN%", "CONF_CHAMP%"]

prog_lookup = team_yr[["YEAR", "TEAM ID"] + PROG_FEAT]
seed_lookup = seed_yr[["YEAR", "SEED"]    + SEED_FEAT]
conf_lookup = conf_yr[["YEAR", "CONF"]    + CONF_FEAT]

# ── Step 8: Join onto spine ──────────────────────────────────────────────────

# 8a. Current-season KenPom metrics (joined on YEAR + TEAM NO)
kenpom_slim = kenpom[["YEAR", "TEAM NO"] + KENPOM_FEAT_COLS].copy()

master = (
    spine
    .merge(kenpom_slim,  on=["YEAR", "TEAM NO"], how="left")
    .merge(prog_lookup,  on=["YEAR", "TEAM ID"], how="left")
    .merge(seed_lookup,  on=["YEAR", "SEED"],    how="left")
    .merge(conf_lookup,  on=["YEAR", "CONF"],    how="left")
)

# ── Step 8b: Barttorvik Away-Neutral ─────────────────────────────────────────
b_neutral = pd.read_csv(RAW / "Barttorvik Away-Neutral.csv",
                        usecols=["YEAR", "TEAM NO", "BADJ EM", "BADJ O", "BADJ D",
                                  "BARTHAG", "EFG%", "EFG%D"])
b_neutral = b_neutral.rename(columns={
    "BADJ EM": "NEUTRAL BADJ EM", "BADJ O": "NEUTRAL BADJ O",
    "BADJ D": "NEUTRAL BADJ D",  "BARTHAG": "NEUTRAL BARTHAG",
    "EFG%":  "NEUTRAL EFG%",     "EFG%D":  "NEUTRAL EFG%D",
})
master = master.merge(b_neutral, on=["YEAR", "TEAM NO"], how="left")

# ── Step 8c: Teamsheet Ranks ──────────────────────────────────────────────────
teamsheet = pd.read_csv(RAW / "Teamsheet Ranks.csv",
                        usecols=["YEAR", "TEAM NO", "QUALITY AVG", "RESUME AVG",
                                  "Q1A W", "Q1 W", "Q2 W", "Q3 L", "Q4 L"])
teamsheet = teamsheet.rename(columns={
    "Q1A W": "TS Q1A W", "Q1 W": "TS Q1 W", "Q2 W": "TS Q2 W",
    "Q3 L":  "TS Q3 L",  "Q4 L": "TS Q4 L",
})
master = master.merge(teamsheet, on=["YEAR", "TEAM NO"], how="left")

# ── Step 8d: TeamRankings ─────────────────────────────────────────────────────
tr = pd.read_csv(RAW / "TeamRankings.csv",
                 usecols=["YEAR", "TEAM NO", "TR RATING", "LAST", "HI", "LO",
                           "CONSISTENCY TR RATING", "LUCK RATING"])
tr = tr.rename(columns={
    "LAST": "TR LAST", "HI": "TR HI", "LO": "TR LO",
    "CONSISTENCY TR RATING": "TR CONSISTENCY", "LUCK RATING": "TR LUCK",
})
master = master.merge(tr, on=["YEAR", "TEAM NO"], how="left")

tr_neutral = pd.read_csv(RAW / "TeamRankings Neutral.csv",
                          usecols=["YEAR", "TEAM NO", "TR RATING", "LAST"])
tr_neutral = tr_neutral.rename(columns={
    "TR RATING": "TR NEUTRAL RATING", "LAST": "TR NEUTRAL LAST"
})
master = master.merge(tr_neutral, on=["YEAR", "TEAM NO"], how="left")

master["PEAK DECLINE"] = master["TR HI"]   - master["TR LAST"]
master["RECOVERY"]     = master["TR LAST"] - master["TR LO"]

# ── Step 8e: Preseason trajectory ────────────────────────────────────────────
preseason = pd.read_csv(RAW / "KenPom Preseason.csv",
                         usecols=["YEAR", "TEAM NO", "KADJ EM CHANGE", "KADJ EM RANK CHANGE"])
master = master.merge(preseason, on=["YEAR", "TEAM NO"], how="left")

rppf_pre = pd.read_csv(RAW / "RPPF Preseason Ratings.csv",
                        usecols=["YEAR", "TEAM NO", "RPPF RATING CHANGE"])
master = master.merge(rppf_pre, on=["YEAR", "TEAM NO"], how="left")

# ── Step 8f: Tournament location ──────────────────────────────────────────────
loc = pd.read_csv(RAW / "Tournament Locations.csv",
                  usecols=["YEAR", "TEAM NO", "CURRENT ROUND",
                            "DISTANCE (MI)", "TIME ZONES CROSSED", "TIME ZONES CROSSED VALUE"])
loc_r64 = loc[loc["CURRENT ROUND"] == 64].drop(columns="CURRENT ROUND")
loc_r64 = loc_r64.rename(columns={
    "DISTANCE (MI)":            "LOC DISTANCE MI",
    "TIME ZONES CROSSED":       "LOC TIME ZONES CROSSED",
    "TIME ZONES CROSSED VALUE": "LOC TIME ZONES CROSSED VALUE",
})
loc_r64["PROXIMITY ADVANTAGE"] = loc_r64["LOC DISTANCE MI"].apply(
    lambda d: 1 if d < 1000 else (-1 if d > 1000 else 0)
).astype(int)
master = master.merge(loc_r64, on=["YEAR", "TEAM NO"], how="left")

# ── Step 8g: Resumes ──────────────────────────────────────────────────────────
resumes = pd.read_csv(RAW / "Resumes.csv",
                      usecols=["YEAR", "TEAM NO", "Q1 W", "Q3 Q4 L", "R SCORE", "BID TYPE"])
resumes = resumes.rename(columns={
    "Q1 W": "RES Q1 W", "Q3 Q4 L": "RES Q3 Q4 L", "R SCORE": "RES R SCORE"
})
resumes["RES BID TYPE AUTO"] = (resumes["BID TYPE"] == "Auto").astype(int)
resumes = resumes.drop(columns="BID TYPE")
master = master.merge(resumes, on=["YEAR", "TEAM NO"], how="left")

# ── Step 8h: Shooting Splits ──────────────────────────────────────────────
ss = pd.read_csv(RAW / "Shooting Splits.csv",
                 usecols=["YEAR", "TEAM NO",
                           "THREES SHARE", "THREES FG%", "THREES FG%D",
                           "CLOSE TWOS SHARE", "CLOSE TWOS FG%",
                           "DUNKS SHARE", "DUNKS FG%D"])
master = master.merge(ss, on=["YEAR", "TEAM NO"], how="left")

# ── Step 9: Save and report ──────────────────────────────────────────────────
out_path = PROCESSED / "master_team_table.csv"
master.to_csv(out_path, index=False)

print(f"Shape: {master.shape}")
print(f"Saved to {out_path}")
print(f"  Current-season KenPom features: {len(KENPOM_FEAT_COLS)}")
print(f"  Historical PROG_ features:      {len(PROG_FEAT)}")
print(f"  Historical SEED_ features:      {len(SEED_FEAT)}")
print(f"  Historical CONF_ features:      {len(CONF_FEAT)}\n")

# Null counts (non-zero only)
null_counts = master.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if null_counts.empty:
    print("No nulls found.")
else:
    print(f"Null counts per column ({len(null_counts)} columns with nulls):")
    print(null_counts.to_string())

# Sanity check: earliest year should have all NaN PROG_ features (no prior data)
earliest = master["YEAR"].min()
prog_nan = master.loc[master["YEAR"] == earliest, "PROG_WIN%"].isna().all()
print(f"\nSanity check: PROG_WIN% is all NaN for earliest year {earliest}: {prog_nan}")

# Show a sample of PROG_WIN% across years (should increase with more history)
print("\nMedian PROG_WIN% by year (should be NaN for first, then stabilize):")
print(master.groupby("YEAR")["PROG_WIN%"].median().to_string())

print("\nDtype counts:")
print(master.dtypes.value_counts().to_string())
