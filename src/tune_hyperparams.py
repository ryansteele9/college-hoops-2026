"""
Hyperparameter tuning for all five models using Optuna (50 trials each).
Objective: mean ROC-AUC across walk-forward CV folds (2015-2025, excl 2020).
Features:  selected features from data/processed/selected_features.txt.

MLP search space includes architecture (layers, hidden size up to 256, dropout).
Best params saved to models/best_params.json.
Final table: tuned vs untuned walk-forward AUC for all five models.
"""

import warnings, json, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
N_TRIALS   = 50

# ── Load data ─────────────────────────────────────────────────────────────────
df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
print(f"Features ({len(selected)}): {selected}")
print(f"Folds: {FOLD_YEARS}\n")

# ── Shared walk-forward evaluator ─────────────────────────────────────────────
def wf_auc(df, feature_cols, model_fn, seed_col_only=False):
    """Return per-fold AUC list. model_fn(Xtr, ytr, Xte) -> proba array."""
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        cols = ["DIFF_SEED"] if seed_col_only else feature_cols
        Xtr = tr[cols].values.astype(np.float32)
        Xte = te[cols].values.astype(np.float32)
        ytr = tr[TARGET].values
        yte = te[TARGET].values
        imp = SimpleImputer(strategy="median").fit(Xtr)
        prob = model_fn(imp.transform(Xtr), ytr, imp.transform(Xte))
        aucs.append(roc_auc_score(yte, prob))
    return np.array(aucs)

# ── MLP helpers ───────────────────────────────────────────────────────────────
def build_mlp(n_in, hidden, n_layers, dropout):
    layers = []
    in_dim = n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden),
                   nn.BatchNorm1d(hidden),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)

def mlp_forward(model, X_np):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_np, dtype=torch.float32)).squeeze(1).numpy()

def train_mlp_fold(Xtr_sc, ytr, Xva_sc, yva,
                   hidden, n_layers, dropout, lr, wd,
                   max_epochs=200, patience=15):
    n_in = Xtr_sc.shape[1]
    net  = build_mlp(n_in, hidden, n_layers, dropout)
    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCELoss()
    dl   = DataLoader(
        TensorDataset(torch.tensor(Xtr_sc, dtype=torch.float32),
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
            best_loss, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net

def wf_auc_mlp(df, feature_cols, hidden, n_layers, dropout, lr, wd):
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        val_year = (s - 2) if s == 2021 else (s - 1)
        fit_mask = tr["YEAR"] != val_year
        val_mask = tr["YEAR"] == val_year

        X_fit = tr.loc[fit_mask, feature_cols].values.astype(np.float32)
        X_val = tr.loc[val_mask, feature_cols].values.astype(np.float32)
        X_te  = te[feature_cols].values.astype(np.float32)
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
        aucs.append(roc_auc_score(y_te, mlp_forward(net, Xte_sc)))
    return np.array(aucs)

# ═════════════════════════════════════════════════════════════════════════════
# Untuned baselines (10 selected features, default params)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Computing untuned baselines ({len(selected)} features, default params)...")
print("=" * 60)

untuned = {}

# LR (seed only — stays seed-only as its natural baseline)
a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: LogisticRegression(max_iter=1000, random_state=42)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
           seed_col_only=True)
untuned["LR (seed only)"] = a.mean()
print(f"  LR (seed only)  : {a.mean():.4f} +/- {a.std():.4f}")

a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: RandomForestClassifier(
               n_estimators=500, max_depth=8, min_samples_leaf=10,
               random_state=42, n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
untuned["Random Forest"] = a.mean()
print(f"  Random Forest   : {a.mean():.4f} +/- {a.std():.4f}")

a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: xgb.XGBClassifier(
               n_estimators=500, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8,
               eval_metric="logloss", random_state=42, n_jobs=-1)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
untuned["XGBoost"] = a.mean()
print(f"  XGBoost         : {a.mean():.4f} +/- {a.std():.4f}")

a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: lgb.LGBMClassifier(
               n_estimators=500, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8,
               min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
untuned["LightGBM"] = a.mean()
print(f"  LightGBM        : {a.mean():.4f} +/- {a.std():.4f}")

a = wf_auc_mlp(df, selected, hidden=128, n_layers=2, dropout=0.3, lr=1e-3, wd=1e-4)
untuned["MLP"] = a.mean()
print(f"  MLP (default)   : {a.mean():.4f} +/- {a.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Optuna tuning
# ═════════════════════════════════════════════════════════════════════════════
best_params = {}

# ── 1. Logistic Regression ────────────────────────────────────────────────────
print(f"\n[1/5] Tuning Logistic Regression ({N_TRIALS} trials)...")

def lr_objective(trial):
    C       = trial.suggest_float("C", 1e-3, 100.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver  = "liblinear"   # supports both l1 and l2
    try:
        aucs = wf_auc(df, selected,
                      lambda Xtr, ytr, Xte: LogisticRegression(
                          C=C, penalty=penalty, solver=solver,
                          max_iter=1000, random_state=42)
                          .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
                      seed_col_only=True)
        return aucs.mean()
    except Exception:
        return 0.0

study_lr = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
study_lr.optimize(lr_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params["logistic_regression"] = study_lr.best_params
print(f"  Best AUC: {study_lr.best_value:.4f}  params: {study_lr.best_params}")

# ── 2. Random Forest ──────────────────────────────────────────────────────────
print(f"\n[2/5] Tuning Random Forest ({N_TRIALS} trials)...")

def rf_objective(trial):
    p = dict(
        n_estimators    = trial.suggest_int("n_estimators", 100, 600),
        max_depth       = trial.suggest_int("max_depth", 3, 15),
        min_samples_leaf= trial.suggest_int("min_samples_leaf", 3, 30),
        max_features    = trial.suggest_categorical("max_features",
                              ["sqrt", "log2", 0.5, 0.8]),
    )
    aucs = wf_auc(df, selected,
                  lambda Xtr, ytr, Xte: RandomForestClassifier(
                      **p, random_state=42, n_jobs=-1)
                      .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return aucs.mean()

study_rf = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(rf_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params["random_forest"] = study_rf.best_params
print(f"  Best AUC: {study_rf.best_value:.4f}  params: {study_rf.best_params}")

# ── 3. XGBoost ────────────────────────────────────────────────────────────────
print(f"\n[3/5] Tuning XGBoost ({N_TRIALS} trials)...")

def xgb_objective(trial):
    p = dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 600),
        max_depth         = trial.suggest_int("max_depth", 2, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma             = trial.suggest_float("gamma", 0.0, 5.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    )
    aucs = wf_auc(df, selected,
                  lambda Xtr, ytr, Xte: xgb.XGBClassifier(
                      **p, eval_metric="logloss",
                      random_state=42, n_jobs=-1)
                      .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return aucs.mean()

study_xgb = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params["xgboost"] = study_xgb.best_params
print(f"  Best AUC: {study_xgb.best_value:.4f}  params: {study_xgb.best_params}")

# ── 4. LightGBM ───────────────────────────────────────────────────────────────
print(f"\n[4/5] Tuning LightGBM ({N_TRIALS} trials)...")

def lgb_objective(trial):
    p = dict(
        n_estimators     = trial.suggest_int("n_estimators", 100, 600),
        num_leaves       = trial.suggest_int("num_leaves", 8, 64),
        max_depth        = trial.suggest_int("max_depth", 2, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    )
    aucs = wf_auc(df, selected,
                  lambda Xtr, ytr, Xte: lgb.LGBMClassifier(
                      **p, random_state=42, n_jobs=-1, verbose=-1)
                      .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    return aucs.mean()

study_lgb = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params["lightgbm"] = study_lgb.best_params
print(f"  Best AUC: {study_lgb.best_value:.4f}  params: {study_lgb.best_params}")

# ── 5. MLP ────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Tuning MLP ({N_TRIALS} trials)...")

def mlp_objective(trial):
    hidden   = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout  = trial.suggest_float("dropout", 0.0, 0.5)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd       = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    try:
        aucs = wf_auc_mlp(df, selected, hidden, n_layers, dropout, lr, wd)
        return aucs.mean()
    except Exception:
        return 0.0

study_mlp = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=42))
study_mlp.optimize(mlp_objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params["mlp"] = study_mlp.best_params
print(f"  Best AUC: {study_mlp.best_value:.4f}  params: {study_mlp.best_params}")

# ── Save best params ──────────────────────────────────────────────────────────
params_path = MODELS / "best_params.json"
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
print(f"\nBest params saved -> {params_path}")

# ═════════════════════════════════════════════════════════════════════════════
# Final tuned AUC (re-evaluate with best params for clean per-fold numbers)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRe-evaluating tuned models for final comparison table...")

tuned = {}

p = best_params["logistic_regression"]
a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: LogisticRegression(
               C=p["C"], penalty=p["penalty"], solver="liblinear",
               max_iter=1000, random_state=42)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
           seed_col_only=True)
tuned["LR (seed only)"] = (a.mean(), a.std())

p = best_params["random_forest"]
a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: RandomForestClassifier(
               **p, random_state=42, n_jobs=-1)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
tuned["Random Forest"] = (a.mean(), a.std())

p = best_params["xgboost"]
a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: xgb.XGBClassifier(
               **p, eval_metric="logloss", random_state=42, n_jobs=-1)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
tuned["XGBoost"] = (a.mean(), a.std())

p = best_params["lightgbm"]
a = wf_auc(df, selected,
           lambda Xtr, ytr, Xte: lgb.LGBMClassifier(
               **p, random_state=42, n_jobs=-1, verbose=-1)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1])
tuned["LightGBM"] = (a.mean(), a.std())

p = best_params["mlp"]
a = wf_auc_mlp(df, selected,
               hidden=p["hidden_size"], n_layers=p["n_layers"],
               dropout=p["dropout"], lr=p["lr"], wd=p["weight_decay"])
tuned["MLP"] = (a.mean(), a.std())

# ── Comparison table ──────────────────────────────────────────────────────────
print("\n" + "=" * 66)
print(f"TUNED vs UNTUNED — Walk-Forward AUC ({len(selected)} selected features)")
print("=" * 66)
rows = []
for name in ["LR (seed only)", "Random Forest", "XGBoost", "LightGBM", "MLP"]:
    un = untuned[name]
    tu_mean, tu_std = tuned[name]
    rows.append({
        "Model":        name,
        "Untuned AUC":  round(un, 4),
        "Tuned AUC":    round(tu_mean, 4),
        "Tuned Std":    round(tu_std, 4),
        "Delta":        round(tu_mean - un, 4),
    })
cmp_df = pd.DataFrame(rows).set_index("Model")
print(cmp_df.to_string())
