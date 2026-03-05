"""
Ensemble: average win probabilities from MLP + Random Forest.
Evaluate on holdout (2023-2025). Calibration plot vs. both individuals.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

PROCESSED = Path("data/processed")
MODELS    = Path("models")

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
DIFF_COLS = [c for c in df.columns if c.startswith("DIFF_")]
TARGET    = "TEAM_A_WIN"

train = df[df["YEAR"] <= 2022]
hold  = df[df["YEAR"].between(2023, 2025)]

X_train = train[DIFF_COLS].values.astype(np.float32)
X_hold  = hold[DIFF_COLS].values.astype(np.float32)
y_hold  = hold[TARGET].values

imputer = SimpleImputer(strategy="median").fit(X_train)
scaler  = StandardScaler().fit(imputer.transform(X_train))

X_hold_imp = imputer.transform(X_hold)
X_hold_sc  = scaler.transform(X_hold_imp)

# ── Load Random Forest ────────────────────────────────────────────────────────
rf_bundle = joblib.load(MODELS / "random_forest.pkl")
rf_prob   = rf_bundle["model"].predict_proba(X_hold_imp)[:, 1]

# ── Load MLP ─────────────────────────────────────────────────────────────────
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

bundle = torch.load(MODELS / "mlp.pt", weights_only=False)
mlp = MLP(bundle["n_feat"])
mlp.load_state_dict(bundle["model_state"])
mlp.eval()
with torch.no_grad():
    mlp_prob = mlp(torch.tensor(X_hold_sc, dtype=torch.float32)).numpy()

# ── Ensemble ──────────────────────────────────────────────────────────────────
ens_prob = (rf_prob + mlp_prob) / 2

# ── Evaluate ──────────────────────────────────────────────────────────────────
def metrics(name, y_true, y_prob):
    return {
        "Model":    name,
        "Accuracy": round(accuracy_score(y_true, (y_prob >= 0.5).astype(int)), 4),
        "Log Loss": round(log_loss(y_true, y_prob), 4),
        "ROC-AUC":  round(roc_auc_score(y_true, y_prob), 4),
    }

rows = [
    metrics("Random Forest",    y_hold, rf_prob),
    metrics("MLP (PyTorch)",    y_hold, mlp_prob),
    metrics("Ensemble (avg)",   y_hold, ens_prob),
]
results = pd.DataFrame(rows).set_index("Model")

print("="*52)
print("HOLDOUT 2023-2025 — Ensemble vs. Individuals")
print("="*52)
print(results.to_string())

# ── Calibration plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")

models_to_plot = [
    ("Random Forest",  rf_prob,  "tab:orange", "s", "--"),
    ("MLP (PyTorch)",  mlp_prob, "tab:purple",  "v", "--"),
    ("Ensemble (avg)", ens_prob, "tab:blue",    "o", "-"),
]
for name, prob, color, marker, ls in models_to_plot:
    frac_pos, mean_pred = calibration_curve(y_hold, prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker=marker, color=color,
            linestyle=ls, lw=2, label=name)

ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title("Calibration — Ensemble vs. Individuals (Holdout 2023-2025)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
out = MODELS / "calibration_ensemble.png"
fig.savefig(out, dpi=150)
print(f"\nCalibration plot saved -> {out}")
