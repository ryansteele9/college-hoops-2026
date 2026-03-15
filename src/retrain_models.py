"""
Retrain all 6 production models on the current 21-feature set and overwrite
the *_calibrated.pkl bundle files.

Run after any feature set change to ensure imputer/model dimensions match.
Usage:
    source venv/Scripts/activate
    python src/retrain_models.py
"""

import warnings, json, sys
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED = Path("data/processed")
MODELS    = Path("models")
TARGET    = "TEAM_A_WIN"

df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

print(f"Features ({len(selected)}): {selected}")
print(f"Training rows: {len(df)}\n")

# Keys added by post-processing that are not sklearn constructor args
_META_KEYS = {"decay_rate", "calibration"}

def model_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in _META_KEYS}


# ── MLP helpers ───────────────────────────────────────────────────────────────
def build_mlp(n_in, hidden, n_layers, dropout):
    layers, in_dim = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


def train_mlp(Xtr_sc, ytr, Xva_sc, yva, p, max_epochs=200, patience=15):
    net  = build_mlp(Xtr_sc.shape[1], p["hidden_size"], p["n_layers"], p["dropout"])
    opt  = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    crit = nn.BCELoss()
    dl   = DataLoader(TensorDataset(torch.tensor(Xtr_sc, dtype=torch.float32),
                                    torch.tensor(ytr,    dtype=torch.float32)),
                      batch_size=64, shuffle=True, drop_last=True)
    t_Xva = torch.tensor(Xva_sc, dtype=torch.float32)
    t_yva = torch.tensor(yva,    dtype=torch.float32)
    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        net.train()
        for Xb, yb in dl:
            opt.zero_grad(); crit(net(Xb).squeeze(1), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = crit(net(t_Xva).squeeze(1), t_yva).item()
        if vl < best_loss - 1e-4:
            best_loss, wait, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    return net


# ── Build full-dataset matrices ───────────────────────────────────────────────
X_all     = df[selected].values.astype(np.float32)
y_all     = df[TARGET].values
imp_final = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_all)
X_all_imp = imp_final.transform(X_all)

# ── LR (seed only) ────────────────────────────────────────────────────────────
print("Training LR (seed)...")
X_seed_all     = df[["DIFF_SEED"]].values.astype(np.float32)
imp_seed_final = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_seed_all)
p_lr = model_params(params["logistic_regression"])
lr_final = LogisticRegression(C=p_lr["C"], penalty=p_lr["penalty"],
                               solver="liblinear", max_iter=1000, random_state=42)
lr_final.fit(imp_seed_final.transform(X_seed_all), y_all)
joblib.dump({
    "model_type": "logistic_regression",
    "model":      lr_final,
    "imputer":    imp_seed_final,
    "features":   ["DIFF_SEED"],
}, MODELS / "lr_seed_calibrated.pkl")
print("  Saved: lr_seed_calibrated.pkl")

# ── Random Forest ─────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf_final = RandomForestClassifier(**model_params(params["random_forest"]), random_state=42, n_jobs=-1)
rf_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "random_forest",
    "model":      rf_final,
    "imputer":    imp_final,
    "features":   selected,
}, MODELS / "random_forest_calibrated.pkl")
print("  Saved: random_forest_calibrated.pkl")

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("Training XGBoost...")
xgb_final = xgb.XGBClassifier(**model_params(params["xgboost"]), eval_metric="logloss",
                                random_state=42, n_jobs=-1)
xgb_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "xgboost",
    "model":      xgb_final,
    "imputer":    imp_final,
    "features":   selected,
}, MODELS / "xgboost_calibrated.pkl")
print("  Saved: xgboost_calibrated.pkl")

# ── LightGBM ──────────────────────────────────────────────────────────────────
print("Training LightGBM...")
lgb_final = lgb.LGBMClassifier(**model_params(params["lightgbm"]), random_state=42,
                                 n_jobs=-1, verbose=-1)
lgb_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "lightgbm",
    "model":      lgb_final,
    "imputer":    imp_final,
    "features":   selected,
}, MODELS / "lightgbm_calibrated.pkl")
print("  Saved: lightgbm_calibrated.pkl")

# ── MLP — train on all years except 2024 (held out for early-stop validation) ─
print("Training MLP...")
val_yr     = 2024
fit_mask   = df["YEAR"] != val_yr
val_mask   = df["YEAR"] == val_yr
X_fit_mlp  = df.loc[fit_mask, selected].values.astype(np.float32)
X_val_mlp  = df.loc[val_mask, selected].values.astype(np.float32)
y_fit_mlp  = df.loc[fit_mask, TARGET].values
y_val_mlp  = df.loc[val_mask, TARGET].values

imp_mlp    = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit_mlp)
sc_mlp     = StandardScaler().fit(imp_mlp.transform(X_fit_mlp))
Xfit_sc    = sc_mlp.transform(imp_mlp.transform(X_fit_mlp))
Xval_sc    = sc_mlp.transform(imp_mlp.transform(X_val_mlp))

net_final  = train_mlp(Xfit_sc, y_fit_mlp, Xval_sc, y_val_mlp, params["mlp"])
joblib.dump({
    "model_type":     "mlp",
    "net_state_dict": net_final.state_dict(),
    "net_params":     {k: params["mlp"][k] for k in ["hidden_size", "n_layers", "dropout"]},
    "n_features":     len(selected),
    "imputer":        imp_mlp,
    "scaler":         sc_mlp,
    "features":       selected,
}, MODELS / "mlp_calibrated.pkl")
print("  Saved: mlp_calibrated.pkl")

# ── Ensemble (RF + XGB + LGB equal weight) ────────────────────────────────────
print("Saving Ensemble bundle...")
joblib.dump({
    "model_type": "ensemble",
    "components": ["random_forest", "xgboost", "lightgbm"],
    "weights":    [1/3, 1/3, 1/3],
    "models": {
        "random_forest": rf_final,
        "xgboost":       xgb_final,
        "lightgbm":      lgb_final,
    },
    "imputer":  imp_final,
    "features": selected,
}, MODELS / "ensemble_rf_xgb_lgbm_calibrated.pkl")
print("  Saved: ensemble_rf_xgb_lgbm_calibrated.pkl")

print("\nAll 6 models retrained on 21-feature set. Ready for run_2026.py.")
