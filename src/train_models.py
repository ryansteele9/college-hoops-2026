"""
Train five models on the matchup dataset and evaluate on holdout (2023-2025).

Models:
  1. Logistic Regression  — DIFF_SEED only (baseline)
  2. Random Forest        — all DIFF_ features
  3. XGBoost              — all DIFF_ features
  4. LightGBM             — all DIFF_ features
  5. PyTorch MLP          — all DIFF_ features

Metrics: accuracy, log loss, ROC-AUC
Output: calibration plot, feature importance table, model comparison table
Saved:  models/*.pkl / models/mlp.pt
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROCESSED = Path("data/processed")
MODELS    = Path("models")
MODELS.mkdir(exist_ok=True)

# ── Load & split ──────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")

DIFF_COLS   = [c for c in df.columns if c.startswith("DIFF_")]
TARGET      = "TEAM_A_WIN"

train = df[df["YEAR"] <= 2022].copy()
hold  = df[df["YEAR"].between(2023, 2025)].copy()

X_train_full = train[DIFF_COLS].values.astype(np.float32)
X_hold_full  = hold[DIFF_COLS].values.astype(np.float32)
y_train      = train[TARGET].values
y_hold       = hold[TARGET].values

X_train_seed = train[["DIFF_SEED"]].values.astype(np.float32)
X_hold_seed  = hold[["DIFF_SEED"]].values.astype(np.float32)

print(f"Train: {len(train)} rows | Holdout: {len(hold)} rows | Features: {len(DIFF_COLS)}")

# ── Imputer + scaler pipelines ────────────────────────────────────────────────
# Median impute the 18 DIFF_PROG_* cols that have nulls for first-time programs
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train_full)
X_hold_imp  = imputer.transform(X_hold_full)

X_train_sc  = scaler.fit_transform(X_train_imp)
X_hold_sc   = scaler.transform(X_hold_imp)

# Baseline uses only DIFF_SEED — no nulls, but scale for consistency
seed_scaler      = StandardScaler()
X_train_seed_sc  = seed_scaler.fit_transform(X_train_seed)
X_hold_seed_sc   = seed_scaler.transform(X_hold_seed)

# ── Helper ────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Log Loss": round(log_loss(y_true, y_prob), 4),
        "ROC-AUC":  round(roc_auc_score(y_true, y_prob), 4),
    }

results   = []
all_probs = {}   # for calibration plot

# ══════════════════════════════════════════════════════════════════════════════
# 1. Logistic Regression baseline (DIFF_SEED only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Logistic Regression (baseline)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_seed_sc, y_train)
lr_prob = lr.predict_proba(X_hold_seed_sc)[:, 1]

joblib.dump({"model": lr, "scaler": seed_scaler}, MODELS / "logistic_regression.pkl")
results.append(evaluate("Logistic Regression (seed only)", y_hold, lr_prob))
all_probs["Logistic Regression"] = lr_prob
print(f"  Saved -> models/logistic_regression.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Random Forest...")
rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train_imp, y_train)
rf_prob = rf.predict_proba(X_hold_imp)[:, 1]

joblib.dump({"model": rf, "imputer": imputer}, MODELS / "random_forest.pkl")
results.append(evaluate("Random Forest", y_hold, rf_prob))
all_probs["Random Forest"] = rf_prob
print(f"  Saved -> models/random_forest.pkl")

# ── RF feature importances ────────────────────────────────────────────────────
rf_imp = pd.Series(rf.feature_importances_, index=DIFF_COLS).sort_values(ascending=False)
print("  Top-10 features:")
print(rf_imp.head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 3. XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="logloss",
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train_imp, y_train,
              eval_set=[(X_hold_imp, y_hold)], verbose=False)
xgb_prob = xgb_model.predict_proba(X_hold_imp)[:, 1]

joblib.dump({"model": xgb_model, "imputer": imputer}, MODELS / "xgboost.pkl")
results.append(evaluate("XGBoost", y_hold, xgb_prob))
all_probs["XGBoost"] = xgb_prob
print(f"  Saved -> models/xgboost.pkl")

xgb_imp = pd.Series(xgb_model.feature_importances_, index=DIFF_COLS).sort_values(ascending=False)
print("  Top-10 features:")
print(xgb_imp.head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 4. LightGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=10, random_state=42, n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_imp, y_train,
              eval_set=[(X_hold_imp, y_hold)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=-1)])
lgb_prob = lgb_model.predict_proba(X_hold_imp)[:, 1]

joblib.dump({"model": lgb_model, "imputer": imputer}, MODELS / "lightgbm.pkl")
results.append(evaluate("LightGBM", y_hold, lgb_prob))
all_probs["LightGBM"] = lgb_prob
print(f"  Saved -> models/lightgbm.pkl")

lgb_imp = pd.Series(lgb_model.feature_importances_, index=DIFF_COLS).sort_values(ascending=False)
print("  Top-10 features:")
print(lgb_imp.head(10).to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 5. PyTorch MLP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] PyTorch MLP...")

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

device = torch.device("cpu")
n_feat = X_train_sc.shape[1]

t_Xtr = torch.tensor(X_train_sc, dtype=torch.float32)
t_ytr = torch.tensor(y_train,    dtype=torch.float32)
t_Xho = torch.tensor(X_hold_sc,  dtype=torch.float32)

train_ds = TensorDataset(t_Xtr, t_ytr)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

mlp = MLP(n_feat).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCELoss()

best_loss, patience, wait = np.inf, 20, 0
best_state = None

for epoch in range(300):
    mlp.train()
    for Xb, yb in train_dl:
        opt.zero_grad()
        criterion(mlp(Xb), yb).backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        val_loss = criterion(mlp(t_Xho), torch.tensor(y_hold, dtype=torch.float32)).item()
    if val_loss < best_loss - 1e-4:
        best_loss, wait = val_loss, 0
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    else:
        wait += 1
        if wait >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

mlp.load_state_dict(best_state)
mlp.eval()
with torch.no_grad():
    mlp_prob = mlp(t_Xho).numpy()

torch.save({"model_state": best_state, "n_feat": n_feat,
            "imputer": imputer, "scaler": scaler},
           MODELS / "mlp.pt")
results.append(evaluate("MLP (PyTorch)", y_hold, mlp_prob))
all_probs["MLP (PyTorch)"] = mlp_prob
print(f"  Saved -> models/mlp.pt")

# ══════════════════════════════════════════════════════════════════════════════
# Results table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*62)
print("MODEL COMPARISON — Holdout 2023–2025")
print("="*62)
results_df = pd.DataFrame(results).set_index("Model")
print(results_df.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# Calibration plot
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

for name, prob in all_probs.items():
    frac_pos, mean_pred = calibration_curve(y_hold, prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=name)

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Plot — Holdout 2023–2025")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
plot_path = MODELS / "calibration_plot.png"
fig.savefig(plot_path, dpi=150)
print(f"\nCalibration plot saved -> {plot_path}")
