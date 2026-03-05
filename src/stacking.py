"""
Stacking ensemble: logistic regression meta-learner trained on out-of-fold
predictions from all five base models.

Level-1 features: [lr_prob, rf_prob, xgb_prob, lgb_prob, mlp_prob]
Meta-learner: LogisticRegression (no regularisation bias; C=1)
Holdout evaluation vs. MLP alone.
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

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
DIFF_COLS = [c for c in df.columns if c.startswith("DIFF_")]
TARGET    = "TEAM_A_WIN"

train_df = df[df["YEAR"] <= 2022].reset_index(drop=True)
hold_df  = df[df["YEAR"].between(2023, 2025)].reset_index(drop=True)

X_train_raw = train_df[DIFF_COLS].values.astype(np.float32)
X_hold_raw  = hold_df[DIFF_COLS].values.astype(np.float32)
y_train     = train_df[TARGET].values
y_hold      = hold_df[TARGET].values

X_train_seed = train_df[["DIFF_SEED"]].values.astype(np.float32)
X_hold_seed  = hold_df[["DIFF_SEED"]].values.astype(np.float32)

# Global imputer/scaler fitted on full train (for holdout inference only)
g_imp = SimpleImputer(strategy="median").fit(X_train_raw)
g_sc  = StandardScaler().fit(g_imp.transform(X_train_raw))
g_seed_sc = StandardScaler().fit(X_train_seed)

X_hold_imp = g_imp.transform(X_hold_raw)
X_hold_sc  = g_sc.transform(X_hold_imp)
X_hold_seed_sc = g_seed_sc.transform(X_hold_seed)

# ── MLP definition (must match train_models.py) ───────────────────────────────
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
    dl = DataLoader(TensorDataset(t_Xtr, t_ytr), batch_size=64,
                    shuffle=True, drop_last=True)
    m   = MLP(n_feat)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCELoss()
    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(300):
        m.train()
        for Xb, yb in dl:
            opt.zero_grad(); crit(m(Xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = crit(m(t_Xva), torch.tensor(yva, dtype=torch.float32)).item()
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 20:
                break
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        return m(t_Xva).numpy()

# ── Generate out-of-fold predictions ─────────────────────────────────────────
N_MODELS = 5
oof = np.zeros((len(y_train), N_MODELS), dtype=np.float32)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Generating out-of-fold predictions...")
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_raw, y_train), 1):
    print(f"  Fold {fold}/5", flush=True)

    # Per-fold preprocessing
    imp = SimpleImputer(strategy="median").fit(X_train_raw[tr_idx])
    Xtr_imp = imp.transform(X_train_raw[tr_idx])
    Xva_imp = imp.transform(X_train_raw[va_idx])

    sc = StandardScaler().fit(Xtr_imp)
    Xtr_sc  = sc.transform(Xtr_imp)
    Xva_sc  = sc.transform(Xva_imp)

    seed_sc = StandardScaler().fit(X_train_seed[tr_idx])
    Xtr_seed = seed_sc.transform(X_train_seed[tr_idx])
    Xva_seed = seed_sc.transform(X_train_seed[va_idx])

    ytr, yva = y_train[tr_idx], y_train[va_idx]

    # 0: Logistic Regression (seed only)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xtr_seed, ytr)
    oof[va_idx, 0] = lr.predict_proba(Xva_seed)[:, 1]

    # 1: Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=8,
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(Xtr_imp, ytr)
    oof[va_idx, 1] = rf.predict_proba(Xva_imp)[:, 1]

    # 2: XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_imp, ytr)
    oof[va_idx, 2] = xgb_m.predict_proba(Xva_imp)[:, 1]

    # 3: LightGBM
    lgb_m = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=10, random_state=42,
                                n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_imp, ytr)
    oof[va_idx, 3] = lgb_m.predict_proba(Xva_imp)[:, 1]

    # 4: MLP
    oof[va_idx, 4] = train_mlp(Xtr_sc, ytr, Xva_sc, yva)

# ── Train meta-learner on OOF predictions ────────────────────────────────────
print("\nTraining meta-learner on OOF predictions...")
meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
meta.fit(oof, y_train)

print("Meta-learner weights (LR -> RF -> XGB -> LGB -> MLP):")
for name, w in zip(["LR(seed)", "RF", "XGBoost", "LightGBM", "MLP"], meta.coef_[0]):
    print(f"  {name:10s}: {w:+.4f}")

joblib.dump({"meta": meta}, MODELS / "stacking_meta.pkl")
print("Saved -> models/stacking_meta.pkl")

# ── Holdout: level-1 predictions from full trained models ────────────────────
print("\nGenerating holdout level-1 predictions from saved models...")

lr_b  = joblib.load(MODELS / "logistic_regression.pkl")
rf_b  = joblib.load(MODELS / "random_forest.pkl")
xgb_b = joblib.load(MODELS / "xgboost.pkl")
lgb_b = joblib.load(MODELS / "lightgbm.pkl")

h_lr  = lr_b["model"].predict_proba(lr_b["scaler"].transform(X_hold_seed))[:, 1]
h_rf  = rf_b["model"].predict_proba(X_hold_imp)[:, 1]
h_xgb = xgb_b["model"].predict_proba(X_hold_imp)[:, 1]
h_lgb = lgb_b["model"].predict_proba(X_hold_imp)[:, 1]

mlp_bundle = torch.load(MODELS / "mlp.pt", weights_only=False)
mlp_full = MLP(mlp_bundle["n_feat"])
mlp_full.load_state_dict(mlp_bundle["model_state"])
mlp_full.eval()
with torch.no_grad():
    h_mlp = mlp_full(torch.tensor(X_hold_sc, dtype=torch.float32)).numpy()

L1_hold = np.column_stack([h_lr, h_rf, h_xgb, h_lgb, h_mlp])
stack_prob = meta.predict_proba(L1_hold)[:, 1]

# ── Evaluate ──────────────────────────────────────────────────────────────────
def metrics(name, y_true, y_prob):
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 4),
        "Log Loss": round(log_loss(y_true, y_prob), 4),
        "ROC-AUC":  round(roc_auc_score(y_true, y_prob), 4),
    }

rows = [
    metrics("MLP (PyTorch)",  y_hold, h_mlp),
    metrics("Stack (LR meta)", y_hold, stack_prob),
]
results = pd.DataFrame(rows).set_index("Model")
print("\n" + "="*50)
print("HOLDOUT 2023-2025 — Stack vs. MLP")
print("="*50)
print(results.to_string())

# ── Calibration plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")
for name, prob, color, marker, ls in [
    ("MLP (PyTorch)",   h_mlp,      "tab:purple", "v", "--"),
    ("Stack (LR meta)", stack_prob, "tab:blue",   "o", "-"),
]:
    fp, mp = calibration_curve(y_hold, prob, n_bins=10, strategy="uniform")
    ax.plot(mp, fp, marker=marker, color=color, linestyle=ls, lw=2, label=name)

ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title("Calibration — Stack vs. MLP (Holdout 2023-2025)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
out = MODELS / "calibration_stacking.png"
fig.savefig(out, dpi=150)
print(f"\nCalibration plot saved -> {out}")
