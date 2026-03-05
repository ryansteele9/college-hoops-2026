"""
Walk-forward (rolling) cross-validation for tournament prediction.

For each fold year s in 2015-2025 (excl. 2020):
  - Train on YEAR < s
  - Test  on YEAR == s
  - MLP early-stopping validation: last training year (s-1, or s-2 if s==2021)

Models evaluated each fold:
  1. Logistic Regression  (DIFF_SEED only)
  2. Random Forest        (all DIFF_ cols)
  3. XGBoost              (all DIFF_ cols)
  4. LightGBM             (all DIFF_ cols)
  5. PyTorch MLP          (all DIFF_ cols)

Output:
  - data/processed/rolling_cv_results.csv  (fold-by-fold)
  - printed summary table (mean +/- std per model)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

PROCESSED = Path("data/processed")
MODELS    = Path("models")

FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
MODEL_NAMES = ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP"]

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
DIFF_COLS = [c for c in df.columns
             if c.startswith("DIFF_") or c == "SEED_MATCHUP_UPSET_RATE"]
TARGET    = "TEAM_A_WIN"

# ── MLP ───────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, 1),    nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_mlp(Xtr_sc, ytr, Xva_sc, yva):
    n_feat = Xtr_sc.shape[1]
    t_Xtr = torch.tensor(Xtr_sc, dtype=torch.float32)
    t_ytr = torch.tensor(ytr,    dtype=torch.float32)
    t_Xva = torch.tensor(Xva_sc, dtype=torch.float32)
    t_yva = torch.tensor(yva,    dtype=torch.float32)
    dl   = DataLoader(TensorDataset(t_Xtr, t_ytr), batch_size=64,
                      shuffle=True, drop_last=True)
    m    = MLP(n_feat)
    opt  = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCELoss()
    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(300):
        m.train()
        for Xb, yb in dl:
            opt.zero_grad(); crit(m(Xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(t_Xva), t_yva).item()
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xva_sc if Xva_sc is None else
                              # reuse tensor already built above
                              Xva_sc, dtype=torch.float32)).numpy()

def _mlp_predict(model_state, n_feat, X_sc):
    m = MLP(n_feat)
    m.load_state_dict(model_state)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(X_sc, dtype=torch.float32)).numpy()

# ── Rolling CV ────────────────────────────────────────────────────────────────
records = []

print(f"Walk-forward CV: {len(FOLD_YEARS)} folds ({FOLD_YEARS[0]}-{FOLD_YEARS[-1]}, excl 2020)\n")

for s in FOLD_YEARS:
    train_mask = df["YEAR"] < s
    test_mask  = df["YEAR"] == s

    train_fold = df[train_mask].reset_index(drop=True)
    test_fold  = df[test_mask].reset_index(drop=True)

    # MLP early-stop val = last training year
    # 2021 has no 2020 data, so use 2019 as val year
    val_year  = (s - 2) if s == 2021 else (s - 1)
    val_mask  = train_fold["YEAR"] == val_year
    fit_mask  = train_fold["YEAR"] != val_year   # train minus val year

    X_fit  = train_fold.loc[fit_mask,  DIFF_COLS].values.astype(np.float32)
    X_val  = train_fold.loc[val_mask,  DIFF_COLS].values.astype(np.float32)
    X_test = test_fold[DIFF_COLS].values.astype(np.float32)
    y_fit  = train_fold.loc[fit_mask,  TARGET].values
    y_val  = train_fold.loc[val_mask,  TARGET].values
    y_test = test_fold[TARGET].values

    X_train_full = train_fold[DIFF_COLS].values.astype(np.float32)
    y_train_full = train_fold[TARGET].values

    X_seed_train = train_fold[["DIFF_SEED"]].values.astype(np.float32)
    X_seed_test  = test_fold[["DIFF_SEED"]].values.astype(np.float32)

    # Impute + scale on full training window
    imp = SimpleImputer(strategy="median").fit(X_train_full)
    sc  = StandardScaler().fit(imp.transform(X_train_full))
    seed_sc = StandardScaler().fit(X_seed_train)

    Xtr_imp  = imp.transform(X_train_full)
    Xte_imp  = imp.transform(X_test)
    Xtr_sc   = sc.transform(Xtr_imp)
    Xte_sc   = sc.transform(Xte_imp)
    Xte_seed = seed_sc.transform(X_seed_test)

    # MLP uses fit/val split (already raw — impute/scale separately)
    imp_fit = SimpleImputer(strategy="median").fit(X_fit)
    sc_fit  = StandardScaler().fit(imp_fit.transform(X_fit))
    Xfit_sc = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc = sc_fit.transform(imp_fit.transform(X_val))
    Xte_mlp = sc_fit.transform(imp_fit.transform(X_test))

    probs = {}

    # 1. LR (seed only)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(seed_sc.transform(X_seed_train), y_train_full)
    probs["LR (seed)"] = lr.predict_proba(Xte_seed)[:, 1]

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(Xtr_imp, y_train_full)
    probs["Random Forest"] = rf.predict_proba(Xte_imp)[:, 1]

    # 3. XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_imp, y_train_full)
    probs["XGBoost"] = xgb_m.predict_proba(Xte_imp)[:, 1]

    # 4. LightGBM
    lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=10, random_state=42,
                                n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_imp, y_train_full)
    probs["LightGBM"] = lgb_m.predict_proba(Xte_imp)[:, 1]

    # 5. MLP (early-stop on last training year)
    mlp_oof = train_mlp(Xfit_sc, y_fit, Xval_sc, y_val)
    # Retrain on full training window with same architecture for test prediction
    n_feat = Xtr_sc.shape[1]
    t_Xtr = torch.tensor(Xtr_sc, dtype=torch.float32)
    t_ytr = torch.tensor(y_train_full, dtype=torch.float32)
    t_Xva = torch.tensor(Xval_sc,  dtype=torch.float32)
    t_yva = torch.tensor(y_val,    dtype=torch.float32)
    dl_full = DataLoader(TensorDataset(t_Xtr, t_ytr), batch_size=64,
                         shuffle=True, drop_last=True)
    mlp_full = MLP(n_feat)
    opt_full = torch.optim.Adam(mlp_full.parameters(), lr=1e-3, weight_decay=1e-4)
    crit     = nn.BCELoss()
    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(300):
        mlp_full.train()
        for Xb, yb in dl_full:
            opt_full.zero_grad(); crit(mlp_full(Xb), yb).backward(); opt_full.step()
        mlp_full.eval()
        with torch.no_grad():
            vl = crit(mlp_full(t_Xva), t_yva).item()
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in mlp_full.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break
    mlp_full.load_state_dict(best_state); mlp_full.eval()
    with torch.no_grad():
        probs["MLP"] = mlp_full(
            torch.tensor(Xte_mlp, dtype=torch.float32)).numpy()

    # Score
    row = {"YEAR": s, "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum())}
    for name, p in probs.items():
        row[f"{name}_acc"]  = round(accuracy_score(y_test, (p >= 0.5).astype(int)), 4)
        row[f"{name}_ll"]   = round(log_loss(y_test, p), 4)
        row[f"{name}_auc"]  = round(roc_auc_score(y_test, p), 4)

    records.append(row)
    print(f"  {s}  train={row['train_rows']:4d}  "
          + "  ".join(f"{m}: acc={row[f'{m}_acc']:.3f} auc={row[f'{m}_auc']:.3f}"
                      for m in MODEL_NAMES))

# ── Save ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
out_path = PROCESSED / "rolling_cv_results.csv"
results_df.to_csv(out_path, index=False)
print(f"\nFold results saved -> {out_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("WALK-FORWARD CV SUMMARY — mean +/- std across folds (2015-2025 excl 2020)")
print("="*72)

summary_rows = []
for m in MODEL_NAMES:
    accs = results_df[f"{m}_acc"].values
    lls  = results_df[f"{m}_ll"].values
    aucs = results_df[f"{m}_auc"].values
    summary_rows.append({
        "Model":       m,
        "Acc":         f"{accs.mean():.4f} +/- {accs.std():.4f}",
        "Log Loss":    f"{lls.mean():.4f}  +/- {lls.std():.4f}",
        "ROC-AUC":     f"{aucs.mean():.4f} +/- {aucs.std():.4f}",
    })

summary_df = pd.DataFrame(summary_rows).set_index("Model")
print(summary_df.to_string())

# ── Per-fold AUC chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"LR (seed)": "tab:gray", "Random Forest": "tab:orange",
          "XGBoost": "tab:green", "LightGBM": "tab:red", "MLP": "tab:blue"}
for m in MODEL_NAMES:
    ax.plot(results_df["YEAR"], results_df[f"{m}_auc"],
            marker="o", label=m, color=colors[m])
ax.set_xlabel("Test Year")
ax.set_ylabel("ROC-AUC")
ax.set_title("Walk-Forward CV — ROC-AUC by Fold Year")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xticks(results_df["YEAR"])
fig.tight_layout()
chart_path = MODELS / "rolling_cv_auc.png"
fig.savefig(chart_path, dpi=150)
print(f"\nAUC-by-year chart saved -> {chart_path}")
