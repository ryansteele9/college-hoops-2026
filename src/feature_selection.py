"""
Feature selection via three approaches, all anchored to the walk-forward CV framework:

  1. Permutation importance  — RF trained per fold, importance measured on test fold
  2. SHAP values             — XGBoost trained per fold, mean |SHAP| on test fold
  3. Correlation pruning     — drop one of each pair with |corr| > 0.85, keeping
                               the member with higher combined importance score

Final selection = features surviving correlation pruning that also rank in the
top half by combined importance.

Output:
  data/processed/selected_features.txt  — one feature name per line
  data/processed/feature_scores.csv     — all 53 features with all scores
  models/feature_selection_scores.png   — ranked importance bar chart
  models/rolling_cv_reduced_vs_full.png — walk-forward AUC: full vs reduced RF
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import xgboost as xgb

PROCESSED = Path("data/processed")
MODELS    = Path("models")

FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
DIFF_COLS = [c for c in df.columns
             if c.startswith("DIFF_") or c == "SEED_MATCHUP_UPSET_RATE"]
TARGET    = "TEAM_A_WIN"
N_FEAT    = len(DIFF_COLS)

print(f"Features: {N_FEAT}  |  Folds: {len(FOLD_YEARS)}")

# Accumulators — one score per feature per fold
perm_scores  = np.zeros((len(FOLD_YEARS), N_FEAT))
shap_scores  = np.zeros((len(FOLD_YEARS), N_FEAT))

# ── Walk-forward loop ─────────────────────────────────────────────────────────
for fi, s in enumerate(FOLD_YEARS):
    print(f"  Fold {s}...", flush=True)

    train_df = df[df["YEAR"] < s]
    test_df  = df[df["YEAR"] == s]

    X_train = train_df[DIFF_COLS].values.astype(np.float32)
    X_test  = test_df[DIFF_COLS].values.astype(np.float32)
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    imp = SimpleImputer(strategy="median").fit(X_train)
    Xtr = imp.transform(X_train)
    Xte = imp.transform(X_test)

    # ── 1. Permutation importance (RF) ────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(Xtr, y_train)
    pi = permutation_importance(rf, Xte, y_test, n_repeats=20,
                                random_state=42, n_jobs=-1, scoring="roc_auc")
    perm_scores[fi] = pi.importances_mean

    # ── 2. SHAP (XGBoost) ────────────────────────────────────────────────────
    xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr, y_train)
    explainer = shap.TreeExplainer(xgb_m)
    sv = explainer.shap_values(Xte)
    shap_scores[fi] = np.abs(sv).mean(axis=0)

# ── Average across folds ──────────────────────────────────────────────────────
mean_perm  = perm_scores.mean(axis=0)
mean_shap  = shap_scores.mean(axis=0)

# Normalise each to [0,1] then average for a combined score
def norm01(x):
    r = x - x.min()
    return r / r.max() if r.max() > 0 else r

norm_perm = norm01(mean_perm)
norm_shap = norm01(mean_shap)
combined  = (norm_perm + norm_shap) / 2

scores_df = pd.DataFrame({
    "feature":    DIFF_COLS,
    "perm_imp":   mean_perm,
    "shap_imp":   mean_shap,
    "norm_perm":  norm_perm,
    "norm_shap":  norm_shap,
    "combined":   combined,
}).sort_values("combined", ascending=False).reset_index(drop=True)

scores_df.to_csv(PROCESSED / "feature_scores.csv", index=False)
print(f"\nFeature scores saved -> {PROCESSED / 'feature_scores.csv'}")

# ── 3. Correlation pruning ────────────────────────────────────────────────────
print("\nRunning correlation pruning (threshold = 0.85)...")

X_all = df[DIFF_COLS].fillna(0).values
corr  = np.corrcoef(X_all.T)

# Map feature name -> combined score for tie-breaking
feat_score = dict(zip(DIFF_COLS, combined))

to_drop = set()
for i in range(N_FEAT):
    if DIFF_COLS[i] in to_drop:
        continue
    for j in range(i + 1, N_FEAT):
        if DIFF_COLS[j] in to_drop:
            continue
        if abs(corr[i, j]) > 0.85:
            # Drop the lower-importance one
            loser = DIFF_COLS[i] if feat_score[DIFF_COLS[i]] < feat_score[DIFF_COLS[j]] \
                    else DIFF_COLS[j]
            to_drop.add(loser)

after_corr = [f for f in DIFF_COLS if f not in to_drop]
print(f"  Dropped {len(to_drop)} highly correlated features -> {len(after_corr)} remaining")

# ── Final selection: top half by combined importance among post-corr survivors ─
survivors_df = scores_df[scores_df["feature"].isin(after_corr)].reset_index(drop=True)
cutoff       = survivors_df["combined"].median()
selected     = survivors_df[survivors_df["combined"] >= cutoff]["feature"].tolist()

print(f"  Kept top half by combined score -> {len(selected)} features selected")
print("\nSelected features:")
for f in selected:
    row = scores_df[scores_df["feature"] == f].iloc[0]
    print(f"  {f:<30s}  perm={row['perm_imp']:.4f}  shap={row['shap_imp']:.4f}  "
          f"combined={row['combined']:.4f}")

# Save selected feature list
out_feats = PROCESSED / "selected_features.txt"
out_feats.write_text("\n".join(selected))
print(f"\nSelected feature list saved -> {out_feats}")

# ── Importance bar chart ──────────────────────────────────────────────────────
top_n  = 30
plot_df = scores_df.head(top_n).iloc[::-1]   # bottom = highest for horizontal bar
sel_set = set(selected)

fig, axes = plt.subplots(1, 2, figsize=(14, 9))
for ax, col, label, color_col in [
    (axes[0], "norm_perm", "Norm. Permutation Importance (RF)", "perm"),
    (axes[1], "norm_shap", "Norm. Mean |SHAP| (XGBoost)",       "shap"),
]:
    colors = ["tab:blue" if f in sel_set else "tab:gray" for f in plot_df["feature"]]
    ax.barh(plot_df["feature"], plot_df[col], color=colors)
    ax.set_xlabel(label, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)

axes[0].set_title(f"Top {top_n} — blue = selected", fontsize=10)
axes[1].set_title(f"Top {top_n} — blue = selected", fontsize=10)
fig.suptitle("Feature Importance: Permutation (RF) vs SHAP (XGBoost)\naveraged across walk-forward folds",
             fontsize=11)
fig.tight_layout()
bar_path = MODELS / "feature_selection_scores.png"
fig.savefig(bar_path, dpi=150)
print(f"Importance chart saved -> {bar_path}")

# ── Walk-forward AUC: full vs reduced RF ─────────────────────────────────────
print("\nWalk-forward AUC comparison: full vs reduced feature set...")

full_aucs    = []
reduced_aucs = []

for s in FOLD_YEARS:
    train_df = df[df["YEAR"] < s]
    test_df  = df[df["YEAR"] == s]
    y_test   = test_df[TARGET].values

    def fold_auc(cols):
        X_train = train_df[cols].values.astype(np.float32)
        X_test  = test_df[cols].values.astype(np.float32)
        y_train = train_df[TARGET].values
        imp = SimpleImputer(strategy="median").fit(X_train)
        rf  = RandomForestClassifier(n_estimators=500, max_depth=8,
                                     min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf.fit(imp.transform(X_train), y_train)
        prob = rf.predict_proba(imp.transform(X_test))[:, 1]
        return roc_auc_score(y_test, prob)

    full_aucs.append(fold_auc(DIFF_COLS))
    reduced_aucs.append(fold_auc(selected))
    print(f"  {s}: full={full_aucs[-1]:.4f}  reduced={reduced_aucs[-1]:.4f}  "
          f"delta={reduced_aucs[-1]-full_aucs[-1]:+.4f}")

full_arr    = np.array(full_aucs)
reduced_arr = np.array(reduced_aucs)
print(f"\n  Full RF    mean AUC: {full_arr.mean():.4f} +/- {full_arr.std():.4f}")
print(f"  Reduced RF mean AUC: {reduced_arr.mean():.4f} +/- {reduced_arr.std():.4f}")
print(f"  Delta:               {(reduced_arr - full_arr).mean():+.4f}")

# Chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(FOLD_YEARS, full_aucs,    "o--", color="tab:gray",  label=f"Full ({N_FEAT} features)")
ax2.plot(FOLD_YEARS, reduced_aucs, "o-",  color="tab:blue",  label=f"Reduced ({len(selected)} features)")
ax2.axhline(full_arr.mean(),    color="tab:gray", lw=0.8, linestyle=":")
ax2.axhline(reduced_arr.mean(), color="tab:blue", lw=0.8, linestyle=":")
ax2.set_xlabel("Test Year")
ax2.set_ylabel("ROC-AUC")
ax2.set_title("Walk-Forward AUC: Full vs Reduced Feature Set (Random Forest)")
ax2.legend()
ax2.set_xticks(FOLD_YEARS)
ax2.grid(alpha=0.3)
fig2.tight_layout()
comp_path = MODELS / "rolling_cv_reduced_vs_full.png"
fig2.savefig(comp_path, dpi=150)
print(f"Comparison chart saved -> {comp_path}")
