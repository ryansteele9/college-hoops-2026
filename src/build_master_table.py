"""
Build master team table: one row per team per tournament year.
Spine: KenPom Barttorvik.csv
Lookups joined: Seed Results, Team Results, Conference Results
Output: data/processed/master_team_table.csv
"""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Step 1: Load files ──────────────────────────────────────────────────────
kenpom   = pd.read_csv(RAW / "KenPom Barttorvik.csv")
seed_res = pd.read_csv(RAW / "Seed Results.csv")
team_res = pd.read_csv(RAW / "Team Results.csv")
conf_res = pd.read_csv(RAW / "Conference Results.csv")

# ── Step 2: Build spine ─────────────────────────────────────────────────────
SPINE_COLS = ["YEAR", "TEAM NO", "TEAM ID", "TEAM", "SEED", "ROUND", "CONF", "CONF ID"]
spine = kenpom[SPINE_COLS].copy()

# ── Step 3: Clean percentage columns ────────────────────────────────────────
def clean_pct(series: pd.Series) -> pd.Series:
    """Strip % sign, cast to float. Divide by 100 if the raw string had a % sign."""
    s = series.astype(str).str.strip()
    has_pct = s.str.endswith("%")
    numeric = s.str.rstrip("%").str.strip().astype(float)
    # Only divide rows that had a literal % sign
    numeric = numeric.where(~has_pct, numeric / 100)
    return numeric

PCT_COLS = {
    "seed_res": ["WIN%", "CHAMP%"],
    "team_res": ["WIN%", "F4%", "CHAMP%"],
    "conf_res": ["WIN%", "CHAMP%"],
}

for col in PCT_COLS["seed_res"]:
    seed_res[col] = clean_pct(seed_res[col])

for col in PCT_COLS["team_res"]:
    team_res[col] = clean_pct(team_res[col])

for col in PCT_COLS["conf_res"]:
    conf_res[col] = clean_pct(conf_res[col])

# ── Step 4: Prefix and prepare lookup tables ─────────────────────────────────

def prefix_cols(df: pd.DataFrame, join_key: str, prefix: str) -> pd.DataFrame:
    """Rename all columns except join_key with prefix."""
    rename = {c: f"{prefix}{c}" for c in df.columns if c != join_key}
    return df.rename(columns=rename)

# Seed Results: join on SEED, prefix SEED_
seed_lookup = prefix_cols(seed_res, "SEED", "SEED_")

# Team Results: drop TEAM (conflicts with spine), join on TEAM ID, prefix PROG_
team_res_clean = team_res.drop(columns=["TEAM"])
team_lookup = prefix_cols(team_res_clean, "TEAM ID", "PROG_")

# Conference Results: drop CONF ID (conflicts with spine), join on CONF, prefix CONF_
conf_res_clean = conf_res.drop(columns=["CONF ID"])
conf_lookup = prefix_cols(conf_res_clean, "CONF", "CONF_")

# ── Step 5: Left-join all onto spine ─────────────────────────────────────────
CONF_REMAP = {"P10": "P12", "SInd": "Slnd"}
spine["CONF"] = spine["CONF"].replace(CONF_REMAP)

master = (
    spine
    .merge(seed_lookup, on="SEED",    how="left")
    .merge(team_lookup, on="TEAM ID", how="left")
    .merge(conf_lookup, on="CONF",    how="left")
)

# ── Step 6: Save and report ──────────────────────────────────────────────────
out_path = PROCESSED / "master_team_table.csv"
master.to_csv(out_path, index=False)

print(f"Shape: {master.shape}")
print(f"Saved to {out_path}\n")

# Null counts (non-zero only)
null_counts = master.isnull().sum()
null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
if null_counts.empty:
    print("No nulls found.")
else:
    print(f"Null counts per column ({len(null_counts)} columns with nulls):")
    print(null_counts.to_string())

# Dtype summary
print("\nDtype counts:")
print(master.dtypes.value_counts().to_string())
