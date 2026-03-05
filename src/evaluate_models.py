"""
Evaluate all five trained models:
  1. Calibration plot (holdout 2023-2025) — saved to models/calibration_plot.png
  2. 5-fold stratified CV on training set (YEAR <= 2022) — accuracy, log loss, ROC-AUC
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

PROCESSED = Path("data/processed")
MODELS    = Path("models")

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
DIFF_COLS = [c for c in df.columns if c.startswith("DIFF_")]
TARGET    = "TEAM_A_WIN"

train = df[df["YEAR"] <= 2022].copy()
hold  = df[df["YEAR"].between(2023, 2025)].copy()

X_train_full = train[DIFF_COLS].values.astype(np.float32)
X_hold_full  = hold[DIFF_COLS].values.astype(np.float32)
y_train      = train[TARGET].values
y_hold       = hold[TARGET].values

X_train_seed = train[["DIFF_SEED"]].values.astype(np.float32)
X_hold_seed  = hold[["DIFF_SEED"]].values.astype(np.float32)

# Fit global imputer/scaler on full training set (matches train_models.py)
imputer = SimpleImputer(strategy="median").fit(X_train_full)
scaler  = StandardScaler().fit(imputer.transform(X_train_full))
seed_scaler = StandardScaler().fit(X_train_seed)

X_train_imp = imputer.transform(X_train_full)
X_hold_imp  = imputer.transform(X_hold_full)
X_train_sc  = scaler.transform(X_train_imp)
X_hold_sc   = scaler.transform(X_hold_imp)
X_hold_seed_sc = seed_scaler.transform(X_hold_seed)

# ── Load saved models & generate holdout probs for calibration ────────────────
print("Loading saved models for calibration plot...")

lr_bundle  = joblib.load(MODELS / "logistic_regression.pkl")
rf_bundle  = joblib.load(MODELS / "random_forest.pkl")
xgb_bundle = joblib.load(MODELS / "xgboost.pkl")
lgb_bundle = joblib.load(MODELS / "lightgbm.pkl")

lr_prob  = lr_bundle["model"].predict_proba(
               lr_bundle["scaler"].transform(X_hold_seed))[:, 1]
rf_prob  = rf_bundle["model"].predict_proba(X_hold_imp)[:, 1]
xgb_prob = xgb_bundle["model"].predict_proba(X_hold_imp)[:, 1]
lgb_prob = lgb_bundle["model"].predict_proba(X_hold_imp)[:, 1]

mlp_bundle = torch.load(MODELS / "mlp.pt", weights_only=False)

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

mlp = MLP(mlp_bundle["n_feat"])
mlp.load_state_dict(mlp_bundle["model_state"])
mlp.eval()
with torch.no_grad():
    mlp_prob = mlp(torch.tensor(X_hold_sc, dtype=torch.float32)).numpy()

holdout_probs = {
    "Logistic Regression (seed only)": lr_prob,
    "Random Forest":                   rf_prob,
    "XGBoost":                         xgb_prob,
    "LightGBM":                        lgb_prob,
    "MLP (PyTorch)":                   mlp_prob,
}

# ── Calibration plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")

colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
markers = ["o", "s", "^", "D", "v"]

for (name, prob), color, marker in zip(holdout_probs.items(), colors, markers):
    frac_pos, mean_pred = calibration_curve(y_hold, prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker=marker, color=color, label=name, lw=1.5)

ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title("Calibration Plot — Holdout 2023-2025", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(MODELS / "calibration_plot.png", dpi=150)
print("Calibration plot saved -> models/calibration_plot.png")

# ── 5-Fold Stratified CV ──────────────────────────────────────────────────────
print("\nRunning 5-fold stratified CV on training set (YEAR <= 2022)...")
print(f"Training rows: {len(y_train)} | Folds: 5\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def score_fold(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return (
        accuracy_score(y_true, y_pred),
        log_loss(y_true, y_prob),
        roc_auc_score(y_true, y_prob),
    )

def cv_sklearn(model_fn, X, y, scale=False):
    accs, lls, aucs = [], [], []
    for tr_idx, va_idx in skf.split(X, y):
        imp = SimpleImputer(strategy="median").fit(X[tr_idx])
        Xtr = imp.transform(X[tr_idx])
        Xva = imp.transform(X[va_idx])
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        m = model_fn()
        m.fit(Xtr, y[tr_idx])
        prob = m.predict_proba(Xva)[:, 1]
        a, l, r = score_fold(y[va_idx], prob)
        accs.append(a); lls.append(l); aucs.append(r)
    return np.array(accs), np.array(lls), np.array(aucs)

def cv_seed_lr(X_seed, y):
    accs, lls, aucs = [], [], []
    for tr_idx, va_idx in skf.split(X_seed, y):
        sc = StandardScaler().fit(X_seed[tr_idx])
        Xtr = sc.transform(X_seed[tr_idx])
        Xva = sc.transform(X_seed[va_idx])
        m = LogisticRegression(max_iter=1000, random_state=42)
        m.fit(Xtr, y[tr_idx])
        prob = m.predict_proba(Xva)[:, 1]
        a, l, r = score_fold(y[va_idx], prob)
        accs.append(a); lls.append(l); aucs.append(r)
    return np.array(accs), np.array(lls), np.array(aucs)

def cv_mlp(X_sc_full, y):
    """Re-impute and re-scale per fold, then train MLP."""
    accs, lls, aucs = [], [], []
    n_feat = X_sc_full.shape[1]
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sc_full, y), 1):
        # X_sc_full is already imputed+scaled on full train — redo per fold
        imp = SimpleImputer(strategy="median").fit(X_train_full[tr_idx])
        sc  = StandardScaler()
        Xtr_sc = sc.fit_transform(imp.transform(X_train_full[tr_idx]))
        Xva_sc = sc.transform(imp.transform(X_train_full[va_idx]))
        ytr, yva = y[tr_idx], y[va_idx]

        t_Xtr = torch.tensor(Xtr_sc, dtype=torch.float32)
        t_ytr = torch.tensor(ytr,    dtype=torch.float32)
        t_Xva = torch.tensor(Xva_sc, dtype=torch.float32)

        dl = DataLoader(TensorDataset(t_Xtr, t_ytr), batch_size=64, shuffle=True, drop_last=True)
        m  = MLP(n_feat)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss()

        best_loss, wait, best_state = np.inf, 0, None
        for epoch in range(300):
            m.train()
            for Xb, yb in dl:
                opt.zero_grad(); criterion(m(Xb), yb).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                vl = criterion(m(t_Xva), torch.tensor(yva, dtype=torch.float32)).item()
            if vl < best_loss - 1e-4:
                best_loss, wait = vl, 0
                best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                wait += 1
                if wait >= 20:
                    break

        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            prob = m(t_Xva).numpy()
        a, l, r = score_fold(yva, prob)
        accs.append(a); lls.append(l); aucs.append(r)
        print(f"    fold {fold}: acc={a:.4f}  ll={l:.4f}  auc={r:.4f}")
    return np.array(accs), np.array(lls), np.array(aucs)

cv_results = []

def record(name, accs, lls, aucs):
    cv_results.append({
        "Model":          name,
        "Acc mean":       round(accs.mean(), 4),
        "Acc std":        round(accs.std(),  4),
        "LogLoss mean":   round(lls.mean(),  4),
        "LogLoss std":    round(lls.std(),   4),
        "AUC mean":       round(aucs.mean(), 4),
        "AUC std":        round(aucs.std(),  4),
    })

print("[1/5] Logistic Regression (seed only)...")
accs, lls, aucs = cv_seed_lr(X_train_seed, y_train)
record("Logistic Regression (seed only)", accs, lls, aucs)

print("[2/5] Random Forest...")
accs, lls, aucs = cv_sklearn(
    lambda: RandomForestClassifier(n_estimators=500, max_depth=8,
                                   min_samples_leaf=10, random_state=42, n_jobs=-1),
    X_train_full, y_train
)
record("Random Forest", accs, lls, aucs)

print("[3/5] XGBoost...")
accs, lls, aucs = cv_sklearn(
    lambda: xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, n_jobs=-1),
    X_train_full, y_train
)
record("XGBoost", accs, lls, aucs)

print("[4/5] LightGBM...")
accs, lls, aucs = cv_sklearn(
    lambda: lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=10, random_state=42, n_jobs=-1,
                                verbose=-1),
    X_train_full, y_train
)
record("LightGBM", accs, lls, aucs)

print("[5/5] MLP (PyTorch) — training 5 folds...")
accs, lls, aucs = cv_mlp(X_train_sc, y_train)
record("MLP (PyTorch)", accs, lls, aucs)

# ── Print CV table ─────────────────────────────────────────────────────────────
print("\n" + "="*78)
print("5-FOLD CV RESULTS — Training Set (YEAR <= 2022)")
print("="*78)
cv_df = pd.DataFrame(cv_results).set_index("Model")
print(cv_df.to_string())

# ── Holdout reminder ──────────────────────────────────────────────────────────
print("\n" + "="*62)
print("HOLDOUT RESULTS (2023-2025) — reference")
print("="*62)
holdout_rows = []
for name, prob in holdout_probs.items():
    y_pred = (prob >= 0.5).astype(int)
    holdout_rows.append({
        "Model":    name,
        "Accuracy": round(accuracy_score(y_hold, y_pred), 4),
        "Log Loss": round(log_loss(y_hold, prob), 4),
        "ROC-AUC":  round(roc_auc_score(y_hold, prob), 4),
    })
print(pd.DataFrame(holdout_rows).set_index("Model").to_string())
