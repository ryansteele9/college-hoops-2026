"""
Retune MLP hyperparameters on the current 21-feature set.

Optuna (50 trials) searches architecture + optimisation hyperparameters:
  hidden_size, n_layers, dropout, learning_rate, weight_decay

Objective: mean walk-forward AUC (2015-2025, excl 2020).
Reporting: AUC before/after tuning (deduplicated), then triggers full
           ESPN backtest by importing src/historical_backtest utilities
           to produce a clean before/after ESPN comparison for MLP.

Updates models/best_params.json with new MLP params (all other model
params are preserved unchanged).
"""

import warnings, json, sys, time
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
N_TRIALS   = 50

# ── Load data ─────────────────────────────────────────────────────────────────
df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

print("=" * 66)
print(f"MLP Hyperparameter Retuning — {N_TRIALS} Optuna trials")
print(f"Features ({len(selected)}): {selected}")
print("=" * 66)

# Show old MLP params
_old = {k: v for k, v in params["mlp"].items()
        if k not in {"decay_rate", "calibration"}}
print(f"\nCurrent MLP params: {_old}\n")


# ── MLP helpers ───────────────────────────────────────────────────────────────
def build_mlp(n_in, hidden, n_layers, dropout):
    layers, in_dim = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


def train_mlp_fold(Xfit_sc, y_fit, Xval_sc, y_val,
                   hidden, n_layers, dropout, lr, wd,
                   max_epochs=200, patience=15):
    net  = build_mlp(Xfit_sc.shape[1], hidden, n_layers, dropout)
    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCELoss()
    dl   = DataLoader(
        TensorDataset(torch.tensor(Xfit_sc, dtype=torch.float32),
                      torch.tensor(y_fit,   dtype=torch.float32)),
        batch_size=64, shuffle=True, drop_last=True)
    t_Xva = torch.tensor(Xval_sc, dtype=torch.float32)
    t_yva = torch.tensor(y_val,   dtype=torch.float32)
    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        net.train()
        for Xb, yb in dl:
            opt.zero_grad(); crit(net(Xb).squeeze(1), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = crit(net(t_Xva).squeeze(1), t_yva).item()
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def wf_auc_mlp(hidden, n_layers, dropout, lr, wd, dedup=False):
    """Walk-forward AUC. dedup=True filters test fold to TEAM_NO_A < TEAM_NO_B."""
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s].reset_index(drop=True)
        te = df[df["YEAR"] == s].reset_index(drop=True)
        if dedup:
            te = te[te["TEAM_NO_A"] < te["TEAM_NO_B"]].reset_index(drop=True)

        val_year = (s - 2) if s == 2021 else (s - 1)
        fit_mask = tr["YEAR"] != val_year
        val_mask = tr["YEAR"] == val_year

        X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
        X_val = tr.loc[val_mask, selected].values.astype(np.float32)
        X_te  = te[selected].values.astype(np.float32)
        y_fit = tr.loc[fit_mask, TARGET].values
        y_val = tr.loc[val_mask, TARGET].values
        y_te  = te[TARGET].values

        imp = SimpleImputer(strategy="median").fit(X_fit)
        sc  = StandardScaler().fit(imp.transform(X_fit))

        Xfit_sc = sc.transform(imp.transform(X_fit))
        Xval_sc = sc.transform(imp.transform(X_val))
        Xte_sc  = sc.transform(imp.transform(X_te))

        net = train_mlp_fold(Xfit_sc, y_fit, Xval_sc, y_val,
                             hidden, n_layers, dropout, lr, wd)
        with torch.no_grad():
            probs = net(torch.tensor(Xte_sc, dtype=torch.float32)).squeeze(1).numpy()
        aucs.append(roc_auc_score(y_te, probs))
    return np.array(aucs)


# ── Baseline: current MLP params ──────────────────────────────────────────────
print("Computing baseline AUC with current MLP params (mirrored + dedup)...")
p_old = _old
t0 = time.time()
aucs_before_mirror = wf_auc_mlp(
    p_old["hidden_size"], p_old["n_layers"],
    p_old["dropout"], p_old["lr"], p_old["weight_decay"])
aucs_before_dedup = wf_auc_mlp(
    p_old["hidden_size"], p_old["n_layers"],
    p_old["dropout"], p_old["lr"], p_old["weight_decay"], dedup=True)
print(f"  AUC (mirror): {aucs_before_mirror.mean():.4f} +/- {aucs_before_mirror.std():.4f}")
print(f"  AUC (dedup):  {aucs_before_dedup.mean():.4f} +/- {aucs_before_dedup.std():.4f}")
print(f"  ({time.time()-t0:.0f}s)\n")


# ── Optuna tuning ─────────────────────────────────────────────────────────────
print(f"Running Optuna ({N_TRIALS} trials)...")

def mlp_objective(trial):
    hidden   = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    dropout  = trial.suggest_float("dropout", 0.0, 0.5)
    lr       = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    try:
        aucs = wf_auc_mlp(hidden, n_layers, dropout, lr, wd)
        return float(aucs.mean())
    except Exception:
        return 0.0

t0 = time.time()
study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(mlp_objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"  Done in {time.time()-t0:.0f}s")
print(f"  Best AUC (mirror): {study.best_value:.4f}")
print(f"  Best params: {study.best_params}\n")


# ── After: evaluate tuned params (dedup) ──────────────────────────────────────
p_new = study.best_params
print("Computing tuned AUC (mirrored + dedup)...")
t0 = time.time()
aucs_after_mirror = wf_auc_mlp(
    p_new["hidden_size"], p_new["n_layers"],
    p_new["dropout"], p_new["lr"], p_new["weight_decay"])
aucs_after_dedup = wf_auc_mlp(
    p_new["hidden_size"], p_new["n_layers"],
    p_new["dropout"], p_new["lr"], p_new["weight_decay"], dedup=True)
print(f"  AUC (mirror): {aucs_after_mirror.mean():.4f} +/- {aucs_after_mirror.std():.4f}")
print(f"  AUC (dedup):  {aucs_after_dedup.mean():.4f} +/- {aucs_after_dedup.std():.4f}")
print(f"  ({time.time()-t0:.0f}s)\n")


# ── Before/after table ────────────────────────────────────────────────────────
print("=" * 66)
print("MLP BEFORE vs AFTER RETUNING")
print("=" * 66)
print(f"  {'Metric':<28}  {'Before':>8}  {'After':>8}  {'Delta':>8}")
print(f"  {'-'*56}")
print(f"  {'AUC (mirror, mean)':<28}  "
      f"{aucs_before_mirror.mean():>8.4f}  {aucs_after_mirror.mean():>8.4f}  "
      f"{aucs_after_mirror.mean()-aucs_before_mirror.mean():>+8.4f}")
print(f"  {'AUC (dedup, mean)':<28}  "
      f"{aucs_before_dedup.mean():>8.4f}  {aucs_after_dedup.mean():>8.4f}  "
      f"{aucs_after_dedup.mean()-aucs_before_dedup.mean():>+8.4f}")
print(f"\n  Old arch: hidden={p_old['hidden_size']}, n_layers={p_old['n_layers']}, "
      f"dropout={p_old['dropout']:.3f}, lr={p_old['lr']:.6f}, "
      f"wd={p_old['weight_decay']:.6f}")
print(f"  New arch: hidden={p_new['hidden_size']}, n_layers={p_new['n_layers']}, "
      f"dropout={p_new['dropout']:.3f}, lr={p_new['lr']:.6f}, "
      f"wd={p_new['weight_decay']:.6f}")


# ── Update best_params.json (MLP only; preserve other model params) ───────────
old_mlp_meta = {k: params["mlp"][k] for k in ("decay_rate", "calibration")
                if k in params["mlp"]}
params["mlp"] = {**p_new, **old_mlp_meta}

params_path = MODELS / "best_params.json"
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"\nUpdated {params_path}  (MLP params only; LR/RF/XGB/LGB unchanged)")
print("\nNext: run  python src/historical_backtest.py  for updated ESPN scores.")
