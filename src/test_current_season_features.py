"""
Test whether adding individual KenPom/Barttorvik current-season metrics
improves walk-forward AUC on top of the 9 selected historical features.

Candidates:
  DIFF_BARTHAG  (Barttorvik win probability metric)
  DIFF_ADJOE    (KenPom adjusted offensive efficiency: KADJ O)
  DIFF_ADJDE    (KenPom adjusted defensive efficiency: KADJ D)
  DIFF_EFG_O    (effective FG% offense: EFG%)
  DIFF_EFG_D    (effective FG% defense: EFG%D)
  DIFF_TEMPO    (KenPom tempo: K TEMPO)

Each is tested individually added to the 9-feature base set.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

PROCESSED = Path("data/processed")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]

# ── Load matchup dataset + selected features ──────────────────────────────────
matchup = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
TARGET   = "TEAM_A_WIN"

print(f"Base feature set: {len(selected)} features")
print(f"  {selected}\n")

# ── Pull current-season metrics from KenPom Barttorvik raw file ───────────────
kb = pd.read_csv("data/raw/KenPom Barttorvik.csv",
                 usecols=["YEAR", "TEAM NO",
                          "BARTHAG", "KADJ O", "KADJ D", "EFG%", "EFG%D", "K TEMPO"])

kb = kb.rename(columns={
    "KADJ O":  "ADJOE",
    "KADJ D":  "ADJDE",
    "EFG%":    "EFG_O",
    "EFG%D":   "EFG_D",
    "K TEMPO": "TEMPO",
})
CS_COLS = ["BARTHAG", "ADJOE", "ADJDE", "EFG_O", "EFG_D", "TEMPO"]

# Join current-season metrics for Team A and Team B into matchup dataset
for side, id_col in [("A", "TEAM_NO_A"), ("B", "TEAM_NO_B")]:
    kb_side = kb.rename(columns={c: f"{c}_{side}" for c in CS_COLS})
    matchup = matchup.merge(
        kb_side.rename(columns={"TEAM NO": id_col}),
        on=["YEAR", id_col], how="left"
    )

# Compute DIFF_ for each candidate
for c in CS_COLS:
    matchup[f"DIFF_{c}"] = matchup[f"{c}_A"] - matchup[f"{c}_B"]

CANDIDATE_DIFFS = [f"DIFF_{c}" for c in CS_COLS]

# Coverage check
print("Coverage of current-season DIFF_ cols (non-null rows / total):")
for col in CANDIDATE_DIFFS:
    nn = matchup[col].notna().sum()
    print(f"  {col:<20s}: {nn}/{len(matchup)} ({100*nn/len(matchup):.1f}%)")
print()

# ── Walk-forward AUC helper ───────────────────────────────────────────────────
def walk_forward_auc(df, feature_cols, label):
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        Xtr = tr[feature_cols].values.astype(np.float32)
        Xte = te[feature_cols].values.astype(np.float32)
        ytr = tr[TARGET].values
        yte = te[TARGET].values
        imp = SimpleImputer(strategy="median").fit(Xtr)
        rf  = RandomForestClassifier(n_estimators=500, max_depth=8,
                                     min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf.fit(imp.transform(Xtr), ytr)
        prob = rf.predict_proba(imp.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(yte, prob))
    arr = np.array(aucs)
    return arr, arr.mean(), arr.std()

# ── Baseline: 9 selected features ────────────────────────────────────────────
print("Running walk-forward CV...")
base_aucs, base_mean, base_std = walk_forward_auc(matchup, selected, "Base (9 features)")
print(f"  {'Base (9 features)':<32s}  mean={base_mean:.4f}  std={base_std:.4f}")

# ── Test each candidate addition ──────────────────────────────────────────────
results = [{"Feature added": "(none — base 9)", "Mean AUC": base_mean,
            "Std AUC": base_std, "Delta": 0.0}]

for diff_col in CANDIDATE_DIFFS:
    augmented = selected + [diff_col]
    fold_aucs, mean_auc, std_auc = walk_forward_auc(matchup, augmented, diff_col)
    delta = mean_auc - base_mean
    results.append({
        "Feature added": diff_col,
        "Mean AUC":      round(mean_auc, 4),
        "Std AUC":       round(std_auc, 4),
        "Delta":         round(delta, 4),
    })
    print(f"  {'+ ' + diff_col:<32s}  mean={mean_auc:.4f}  std={std_auc:.4f}  "
          f"delta={delta:+.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("CURRENT-SEASON FEATURE ADDITIONS — Walk-Forward AUC Summary")
print("="*62)
res_df = pd.DataFrame(results).set_index("Feature added")
res_df["Mean AUC"] = res_df["Mean AUC"].round(4)
res_df["Std AUC"]  = res_df["Std AUC"].round(4)
res_df["Delta"]    = res_df["Delta"].round(4)
print(res_df.to_string())

# ── Per-fold breakdown for candidates that help ────────────────────────────────
gainers = [r["Feature added"] for r in results[1:]
           if r["Delta"] > 0.001]  # meaningful positive delta

if gainers:
    print(f"\nPer-fold AUC for helpful additions (delta > +0.001):")
    header = f"  {'Year':<6s}  {'Base':>8s}" + "".join(f"  {g.replace('DIFF_',''):>10s}" for g in gainers)
    print(header)
    for i, s in enumerate(FOLD_YEARS):
        row = f"  {s:<6d}  {base_aucs[i]:>8.4f}"
        for diff_col in gainers:
            fold_aucs, _, _ = walk_forward_auc(matchup, selected + [diff_col], diff_col)
            row += f"  {fold_aucs[i]:>10.4f}"
        print(row)
else:
    print("\nNo candidate adds meaningful AUC (delta > +0.001).")
