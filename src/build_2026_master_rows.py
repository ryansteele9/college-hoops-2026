"""build_2026_master_rows.py

Merges 7 manually-collected 2026 data files into 64 new YEAR=2026 rows
and appends them to data/processed/master_team_table.csv.

Sources:
  barttorvik_2026_mapped.csv  — Barttorvik ratings, four factors, shooting splits,
                                 FT%, AST%/BLK%, PROG_GAMES/WIN%, CONF_WIN%
  kenpom_2026_main.csv        — KADJ EM/O/D/T, K OFF/DEF/TEMPO
  kenpom_2026_preseason.csv   — KADJ EM CHANGE, KADJ EM RANK CHANGE
  kenpom_2026_factors.csv     — four factors cross-check (used if barto NaN)
  teamrankings_2026.csv       — TR RATING/HI/LO/LAST → PEAK DECLINE, RECOVERY
  barttorvik_2026_teamsheet.csv — RESUME AVG, QUALITY AVG, TS Q* wins/losses
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")
MASTER = PROC / "master_team_table.csv"

# ──────────────────────────────────────────────────────────────
# Name normalisation helpers
# ──────────────────────────────────────────────────────────────

def strip_seed(name: str) -> str:
    """Remove ~N or \xa0N suffix used by KenPom/Barttorvik teamsheet files."""
    name = str(name)
    name = re.sub(r"~\d+$", "", name)        # "Duke~1"
    name = re.sub(r"\xa0\d+$", "", name)     # "Duke\xa01"
    return name.strip()

def strip_record(name: str) -> str:
    """Remove \xa0(W-L) suffix used by TeamRankings."""
    name = str(name)
    name = re.sub(r"\xa0\([\d\-]+\)", "", name)
    name = re.sub(r"\s+\([\d\-]+\)", "", name)
    return name.strip()

# Canonical remap: <source name> → <bracket name>
KENPOM_REMAP = {
    "Connecticut": "UConn",
    "N.C. State":  "NC State",
    "Queens":       "Queens NC",
    "LIU":          "Long Island",
    "St. John's (NY)": "St. John's",
}

TR_REMAP = {
    "Iowa St":        "Iowa St.",
    "Michigan St":    "Michigan St.",
    "St John's":      "St. John's",
    "Queens":         "Queens NC",
    "LIU":            "Long Island",
    "N.C. State":     "NC State",
    "Ohio St":        "Ohio St.",
    "Tennessee St":   "Tennessee St.",
    "Utah St":        "Utah St.",
    "Wright St":      "Wright St.",
    "Kennesaw St":    "Kennesaw St.",
    "Miami":          "Miami FL",   # only D1 Miami in tournament context
}

def canon_kenpom(name: str) -> str:
    name = strip_seed(name)
    return KENPOM_REMAP.get(name, name)

def canon_tr(name: str) -> str:
    name = strip_record(name)
    return TR_REMAP.get(name, name)

def canon_barto_sheet(name: str) -> str:
    return strip_seed(name)

# ──────────────────────────────────────────────────────────────
# Parse W-L strings like "17-2", "Jun-00" (pandas date-parsed "6-0"),
# "2-Sep" (pandas date-parsed "2-9")
# ──────────────────────────────────────────────────────────────
MONTH_TO_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

def parse_wl(s, want="wins"):
    """
    Extract wins or losses from strings like:
      "17-2"   → wins=17, losses=2
      "Jun-00" → wins=6, losses=0   (pandas date-parsed "6-0")
      "2-Sep"  → wins=2, losses=9   (pandas date-parsed "2-9")
      "00-00"  → wins=0, losses=0
    """
    s = str(s).strip()
    if s in ("nan", "NaN", "", "-"):
        return np.nan
    # Try "Month-DD" format (pandas auto-parsed dates)
    m = re.match(r"([A-Za-z]+)-(\d+)", s)
    if m:
        month_str, day_str = m.group(1), m.group(2)
        month_num = MONTH_TO_NUM.get(month_str.capitalize(), None)
        day_num = int(day_str)
        if month_num is not None:
            wins, losses = month_num, day_num
            return float(wins if want == "wins" else losses)
    # Try "DD-Month" format (e.g. "2-Sep")
    m = re.match(r"(\d+)-([A-Za-z]+)", s)
    if m:
        day_str, month_str = m.group(1), m.group(2)
        month_num = MONTH_TO_NUM.get(month_str.capitalize(), None)
        day_num = int(day_str)
        if month_num is not None:
            wins, losses = day_num, month_num
            return float(wins if want == "wins" else losses)
    # Try plain "W-L"
    m = re.match(r"(\d+)-(\d+)", s)
    if m:
        wins, losses = int(m.group(1)), int(m.group(2))
        return float(wins if want == "wins" else losses)
    return np.nan


# ──────────────────────────────────────────────────────────────
# Load bracket (ground truth for 64 teams)
# ──────────────────────────────────────────────────────────────
bracket = pd.read_csv(RAW / "bracket_2026.csv")
bracket_teams = set(bracket["TEAM"].tolist())
print(f"Bracket: {len(bracket)} teams")


# ──────────────────────────────────────────────────────────────
# 1. Start from barttorvik_2026_mapped.csv
# ──────────────────────────────────────────────────────────────
barto = pd.read_csv(RAW / "barttorvik_2026_mapped.csv")
# Rename any extra columns to avoid duplicates later
BARTO_COLS = {
    "TEAM": "TEAM", "CONF": "CONF", "CONF ID": "CONF ID",
    "YEAR": "YEAR", "TEAM NO": "TEAM NO", "TEAM ID": "TEAM ID",
    "SEED": "SEED", "ROUND": "ROUND",
    "BADJ EM": "BADJ EM", "BADJ O": "BADJ O", "BADJ D": "BADJ D",
    "BARTHAG": "BARTHAG", "GAMES": "GAMES", "W": "W", "L": "L",
    "WIN%": "WIN%", "EFG%": "EFG%", "EFG%D": "EFG%D",
    "FTR": "FTR", "FTRD": "FTRD", "TOV%": "TOV%", "TOV%D": "TOV%D",
    "OREB%": "OREB%", "DREB%": "DREB%",
    "OP OREB%": "OP OREB%", "OP DREB%": "OP DREB%",
    "RAW T": "RAW T", "2PT%": "2PT%", "2PT%D": "2PT%D",
    "3PT%": "3PT%", "3PT%D": "3PT%D",
    "BLK%": "BLK%", "BLKED%": "BLKED%", "AST%": "AST%", "OP AST%": "OP AST%",
    "2PTR": "2PTR", "3PTR": "3PTR", "2PTRD": "2PTRD", "3PTRD": "3PTRD",
    "BADJ T": "BADJ T", "AVG HGT": "AVG HGT", "EFF HGT": "EFF HGT",
    "EXP": "EXP", "TALENT": "TALENT",
    "FT%": "FT%", "OP FT%": "OP FT%", "PPPO": "PPPO", "PPPD": "PPPD",
    "ELITE SOS": "ELITE SOS", "WAB": "WAB",
    "PROG_GAMES": "PROG_GAMES", "PROG_WIN%": "PROG_WIN%", "CONF_WIN%": "CONF_WIN%",
    "DUNKS FG%": "DUNKS FG%", "DUNKS SHARE": "DUNKS SHARE", "DUNKS FG%D": "DUNKS FG%D",
    "CLOSE TWOS FG%": "CLOSE TWOS FG%", "CLOSE TWOS SHARE": "CLOSE TWOS SHARE",
    "CLOSE TWOS FG%D": "CLOSE TWOS FG%D",
    "THREES FG%": "THREES FG%", "THREES SHARE": "THREES SHARE", "THREES FG%D": "THREES FG%D",
    "FARTHER TWOS FG%": "FARTHER TWOS FG%", "FARTHER TWOS SHARE": "FARTHER TWOS SHARE",
    "FARTHER TWOS FG%D": "FARTHER TWOS FG%D",
}
# Keep only the columns we care about
barto_keep = {k: v for k, v in BARTO_COLS.items() if k in barto.columns}
df = barto[list(barto_keep.keys())].rename(columns=barto_keep).copy()
df["YEAR"] = 2026

# Fill TEAM NO and SEED from bracket where missing
for _, br in bracket.iterrows():
    mask = df["TEAM"] == br["TEAM"]
    if mask.any():
        df.loc[mask, "TEAM NO"] = br["TEAM_NO"]
        df.loc[mask, "SEED"] = br["SEED"]
        df.loc[mask, "ROUND"] = 1

print(f"Base rows: {len(df)} (barttorvik_2026_mapped)")


# ──────────────────────────────────────────────────────────────
# 2. KenPom main: KADJ EM/O/D/T, K OFF/DEF/TEMPO
# ──────────────────────────────────────────────────────────────
kp = pd.read_csv(RAW / "kenpom_2026_main.csv")
kp["_team"] = kp["TeamName"].apply(canon_kenpom)

KP_MAP = {
    "Tempo":    "K TEMPO",
    "AdjTempo": "KADJ T",
    "OE":       "K OFF",
    "AdjOE":    "KADJ O",
    "DE":       "K DEF",
    "AdjDE":    "KADJ D",
    "AdjEM":    "KADJ EM",
}
kp_sub = kp[["_team"] + list(KP_MAP.keys())].rename(columns=KP_MAP)
df = df.merge(kp_sub, left_on="TEAM", right_on="_team", how="left")
df.drop(columns=["_team"], inplace=True)
n_matched = df["KADJ EM"].notna().sum()
print(f"KenPom main matched: {n_matched}/64")


# ──────────────────────────────────────────────────────────────
# 3. KenPom preseason: KADJ EM CHANGE, KADJ EM RANK CHANGE
#    File has garbage pandas header; row 0 is the real sub-header.
#    Pandas named the last two useful cols 'Rk' and 'EM' (change cols).
# ──────────────────────────────────────────────────────────────
kpp_raw = pd.read_csv(RAW / "kenpom_2026_preseason.csv", header=0, dtype=str)
# Row 0 is the real sub-header row — drop it
kpp = kpp_raw.iloc[1:].reset_index(drop=True)
# Column 'Unnamed: 1' = Team name; 'EM' = EM change; 'Rk' = Rk change
kpp["_team"] = kpp["Unnamed: 1"].apply(canon_kenpom)
kpp["KADJ EM CHANGE"] = pd.to_numeric(kpp["EM"], errors="coerce")
kpp["KADJ EM RANK CHANGE"] = pd.to_numeric(kpp["Rk"], errors="coerce")
kpp_sub = kpp[["_team", "KADJ EM CHANGE", "KADJ EM RANK CHANGE"]]
df = df.merge(kpp_sub, left_on="TEAM", right_on="_team", how="left")
df.drop(columns=["_team"], inplace=True)
n_matched = df["KADJ EM CHANGE"].notna().sum()
print(f"KenPom preseason matched: {n_matched}/64")


# ──────────────────────────────────────────────────────────────
# 4. TeamRankings: TR RATING, TR HI, TR LO, TR LAST
#    → compute PEAK DECLINE = TR HI - TR LAST
#    → compute RECOVERY    = TR LAST - TR LO
# ──────────────────────────────────────────────────────────────
tr_raw = pd.read_csv(RAW / "teamrankings_2026.csv", header=None, dtype=str)
tr_raw.columns = tr_raw.iloc[8].tolist()
tr = tr_raw.iloc[9:].reset_index(drop=True)
tr["_team"] = tr["Team"].apply(canon_tr)
for col in ["Rating", "Hi", "Lo", "Last"]:
    tr[col] = pd.to_numeric(tr[col], errors="coerce")
tr["TR RATING"] = tr["Rating"]
tr["TR HI"]     = tr["Hi"]
tr["TR LO"]     = tr["Lo"]
tr["TR LAST"]   = tr["Last"]
tr["PEAK DECLINE"] = tr["TR HI"] - tr["TR LAST"]
tr["RECOVERY"]     = tr["TR LAST"] - tr["TR LO"]
tr_sub = tr[["_team", "TR RATING", "TR HI", "TR LO", "TR LAST", "PEAK DECLINE", "RECOVERY"]]
df = df.merge(tr_sub, left_on="TEAM", right_on="_team", how="left")
df.drop(columns=["_team"], inplace=True)
n_matched = df["TR RATING"].notna().sum()
print(f"TeamRankings matched: {n_matched}/64")


# ──────────────────────────────────────────────────────────────
# 5. Barttorvik teamsheet: RESUME AVG, QUALITY AVG, TS Q* wins/losses
#    File has garbage pandas header; row 0 = real header data.
# ──────────────────────────────────────────────────────────────
ts_raw = pd.read_csv(RAW / "barttorvik_2026_teamsheet.csv", header=0, dtype=str)
# Row 0 is real column headers
ts = ts_raw.iloc[1:].reset_index(drop=True)
# Pandas column names: Unnamed:0=Rk, Unnamed:1=Team, Unnamed:2=NET, Resum?=KPI,
#   Unnamed:4=SOR, Unnamed:5=WAB, Unnamed:6=Avg(Resume), Quality=BPI,
#   Unnamed:8=KP, Unnamed:9=TRK, Unnamed:10=Avg(Quality),
#   Current Quadrant Records=Q1A, Unnamed:12=Q1, Unnamed:13=Q2,
#   Unnamed:14=Q1&2, Unnamed:15=Q3, Unnamed:16=Q4
ts["_team"] = ts["Unnamed: 1"].apply(canon_barto_sheet)
ts["RESUME AVG"]  = pd.to_numeric(ts["Unnamed: 6"], errors="coerce")
ts["QUALITY AVG"] = pd.to_numeric(ts["Unnamed: 10"], errors="coerce")
ts["TS Q1A W"] = ts["Current Quadrant Records"].apply(lambda x: parse_wl(x, "wins"))
ts["TS Q1 W"]  = ts["Unnamed: 12"].apply(lambda x: parse_wl(x, "wins"))
ts["TS Q2 W"]  = ts["Unnamed: 13"].apply(lambda x: parse_wl(x, "wins"))
ts["TS Q3 L"]  = ts["Unnamed: 15"].apply(lambda x: parse_wl(x, "losses"))
ts["TS Q4 L"]  = ts["Unnamed: 16"].apply(lambda x: parse_wl(x, "losses"))
ts_sub = ts[["_team", "RESUME AVG", "QUALITY AVG",
             "TS Q1A W", "TS Q1 W", "TS Q2 W", "TS Q3 L", "TS Q4 L"]]
df = df.merge(ts_sub, left_on="TEAM", right_on="_team", how="left")
df.drop(columns=["_team"], inplace=True)
n_matched = df["RESUME AVG"].notna().sum()
print(f"Barttorvik teamsheet matched: {n_matched}/64")


# ──────────────────────────────────────────────────────────────
# 6. Load master table to get exact column order + CONF ID mapping
# ──────────────────────────────────────────────────────────────
master = pd.read_csv(MASTER)
master_cols = master.columns.tolist()
print(f"\nMaster table: {len(master)} rows, {len(master_cols)} cols")

# Build CONF → CONF ID mapping from existing data
conf_id_map = (master.dropna(subset=["CONF", "CONF ID"])
               .groupby("CONF")["CONF ID"]
               .agg(lambda x: x.mode().iloc[0])
               .to_dict())

# Fill CONF ID from map
df["CONF ID"] = df["CONF"].map(conf_id_map)

# Build TEAM ID mapping (TEAM → stable TEAM ID used across years)
team_id_map = (master.dropna(subset=["TEAM", "TEAM ID"])
               .groupby("TEAM")["TEAM ID"]
               .agg(lambda x: x.mode().iloc[0])
               .to_dict())
df["TEAM ID"] = df["TEAM"].map(team_id_map)


# ──────────────────────────────────────────────────────────────
# 7. Add NaN placeholder columns for anything we can't source
# ──────────────────────────────────────────────────────────────
for col in master_cols:
    if col not in df.columns:
        df[col] = np.nan

# Reorder to match master schema exactly
df = df[master_cols]

print(f"\n2026 rows built: {len(df)}")
# Coverage report for selected features
sel = [l.strip() for l in open(PROC / "selected_features.txt") if l.strip()]
# Selected features are DIFF_* pairs — check base column (strip DIFF_ prefix)
base_cols = [f.replace("DIFF_", "") if f.startswith("DIFF_") else f for f in sel]
covered = [sel[i] for i, b in enumerate(base_cols) if b in df.columns and df[b].notna().any()]
missing = [sel[i] for i, b in enumerate(base_cols) if b not in df.columns or df[b].isna().all()]
print(f"Selected features covered: {len(covered)}/{len(sel)}")
if missing:
    print(f"  NaN (will be median-imputed): {missing}")


# ──────────────────────────────────────────────────────────────
# 8. Append to master table (skip if already appended)
# ──────────────────────────────────────────────────────────────
already = (master["YEAR"] == 2026).sum()
if already > 0:
    print(f"\nWARNING: {already} YEAR=2026 rows already exist in master — replacing them.")
    master = master[master["YEAR"] != 2026]

master_new = pd.concat([master, df], ignore_index=True)
master_new.to_csv(MASTER, index=False)
print(f"Saved {len(master_new)} rows ({len(df)} new 2026) -> {MASTER}")

# Sanity check
sample = df[["TEAM", "SEED", "KADJ EM", "BADJ EM", "KADJ EM CHANGE",
             "TR RATING", "PEAK DECLINE", "RESUME AVG",
             "PROG_GAMES", "PROG_WIN%", "CONF_WIN%"]].copy()
print("\nSample rows:")
print(sample[sample["TEAM"].isin(["Duke", "Florida", "Michigan", "Iowa St.",
                                   "Houston", "Arizona"])].to_string(index=False))
