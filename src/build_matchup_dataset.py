"""
Build matchup-level training dataset from historical tournament games.

Each row = one head-to-head game. Dataset is mirrored (symmetric).
Target: TEAM_A_WIN (1 if Team A won, 0 if lost).
"""

import pandas as pd
from pathlib import Path

RAW       = Path("data/raw")
PROCESSED = Path("data/processed")

# ── Load data ────────────────────────────────────────────────────────────────
matchups = pd.read_csv(RAW / "Tournament Matchups.csv")
master   = pd.read_csv(PROCESSED / "master_team_table.csv")

# ── Pair games ───────────────────────────────────────────────────────────────
# Each game is two consecutive rows sorted by BY YEAR NO descending.
# Row with higher BY YEAR NO = Team A, lower = Team B.
matchups = matchups.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
matchups["GAME_ID"] = matchups.index // 2
matchups["SLOT"]    = matchups.index % 2   # 0 = Team A, 1 = Team B

team_a = matchups[matchups["SLOT"] == 0].rename(columns={
    "TEAM NO": "TEAM_NO_A", "TEAM": "TEAM_A", "SEED": "SEED_A",
    "SCORE": "SCORE_A", "ROUND": "ROUND_A",
})[["YEAR", "GAME_ID", "TEAM_NO_A", "TEAM_A", "SEED_A", "SCORE_A", "ROUND_A"]]

team_b = matchups[matchups["SLOT"] == 1].rename(columns={
    "TEAM NO": "TEAM_NO_B", "TEAM": "TEAM_B", "SEED": "SEED_B",
    "SCORE": "SCORE_B", "ROUND": "ROUND_B",
})[["YEAR", "GAME_ID", "TEAM_NO_B", "TEAM_B", "SEED_B", "SCORE_B", "ROUND_B"]]

games = team_a.merge(team_b, on=["YEAR", "GAME_ID"])
games["TEAM_A_WIN"] = (games["SCORE_A"] > games["SCORE_B"]).astype(int)

# ── Identify numeric feature columns in master table ─────────────────────────
meta_cols = {"YEAR", "TEAM NO", "TEAM ID", "TEAM", "SEED", "ROUND", "CONF", "CONF ID"}
num_cols  = [c for c in master.columns if c not in meta_cols and
             master[c].dtype in ("int64", "float64")]

master_slim = master[["YEAR", "TEAM NO"] + num_cols].copy()

# ── Join master features for Team A and Team B ───────────────────────────────
games = games.merge(
    master_slim.rename(columns={"TEAM NO": "TEAM_NO_A"}).add_suffix("_A")
               .rename(columns={"YEAR_A": "YEAR", "TEAM_NO_A_A": "TEAM_NO_A"}),
    on=["YEAR", "TEAM_NO_A"], how="left"
)
games = games.merge(
    master_slim.rename(columns={"TEAM NO": "TEAM_NO_B"}).add_suffix("_B")
               .rename(columns={"YEAR_B": "YEAR", "TEAM_NO_B_B": "TEAM_NO_B"}),
    on=["YEAR", "TEAM_NO_B"], how="left"
)

# ── Compute differentials (Team A minus Team B) ───────────────────────────────
games["DIFF_SEED"] = games["SEED_A"] - games["SEED_B"]

diff_cols = ["DIFF_SEED"]
for col in num_cols:
    diff_name = f"DIFF_{col}"
    games[diff_name] = games[f"{col}_A"] - games[f"{col}_B"]
    diff_cols.append(diff_name)

# ── Assemble final columns ────────────────────────────────────────────────────
id_cols = ["YEAR", "GAME_ID",
           "TEAM_NO_A", "TEAM_A", "SEED_A", "SCORE_A",
           "TEAM_NO_B", "TEAM_B", "SEED_B", "SCORE_B",
           "TEAM_A_WIN"]
feat_a   = [f"{c}_A" for c in num_cols]
feat_b   = [f"{c}_B" for c in num_cols]

games = games[id_cols + feat_a + feat_b + diff_cols]

# ── Mirror dataset (swap A/B, flip target) ────────────────────────────────────
mirror = games.copy()
# Swap identity columns
mirror[["TEAM_NO_A","TEAM_A","SEED_A","SCORE_A",
        "TEAM_NO_B","TEAM_B","SEED_B","SCORE_B"]] = \
    games[["TEAM_NO_B","TEAM_B","SEED_B","SCORE_B",
           "TEAM_NO_A","TEAM_A","SEED_A","SCORE_A"]].values
# Swap feature columns
mirror[feat_a] = games[feat_b].values
mirror[feat_b] = games[feat_a].values
# Flip differentials and target
for col in diff_cols:
    mirror[col] = -games[col]
mirror["TEAM_A_WIN"] = 1 - games["TEAM_A_WIN"]

dataset = pd.concat([games, mirror], ignore_index=True)
dataset = dataset.sort_values(["YEAR", "GAME_ID"]).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = PROCESSED / "matchup_dataset.csv"
dataset.to_csv(out_path, index=False)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"Shape: {dataset.shape}")

null_counts = dataset.isnull().sum()
null_counts = null_counts[null_counts > 0]
print(f"\nNull counts ({len(null_counts)} columns with nulls):")
print(null_counts.to_string() if len(null_counts) > 0 else "  None")

print("\nSample row (first game, Team A perspective):")
sample = dataset.iloc[0]
print(f"  YEAR={sample['YEAR']:.0f}  GAME_ID={sample['GAME_ID']:.0f}")
print(f"  Team A: {sample['TEAM_A']} (seed {sample['SEED_A']:.0f}, score {sample['SCORE_A']:.0f})")
print(f"  Team B: {sample['TEAM_B']} (seed {sample['SEED_B']:.0f}, score {sample['SCORE_B']:.0f})")
print(f"  TEAM_A_WIN: {sample['TEAM_A_WIN']:.0f}")
print(f"  Sample diffs — DIFF_SEED_WIN%: {sample.get('DIFF_SEED_WIN%', 'N/A'):.4f}, "
      f"DIFF_CONF_WIN%: {sample.get('DIFF_CONF_WIN%', 'N/A'):.4f}")
