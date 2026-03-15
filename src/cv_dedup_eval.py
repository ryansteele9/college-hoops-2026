"""
Walk-forward CV with deduplicated scoring.

The matchup dataset is mirrored — each game appears twice with A/B flipped.
Scoring on both orientations double-counts games, inflating statistical
confidence and distorting accuracy when the threshold hits near 0.5.

Fix: filter test fold to TEAM_NO_A < TEAM_NO_B (one orientation per game)
before computing metrics.

Reports corrected AUC, log loss, and accuracy for all six models.
"""

import warnings, json, sys
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import xgboost as xgb
import lightgbm as lgb

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"

df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

_META_KEYS = {"decay_rate", "calibration"}


def model_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in _META_KEYS}


# ── MLP helpers ────────────────────────────────────────────────────────────────
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


MODEL_NAMES = ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP", "Ensemble"]

print("=" * 68)
print("Walk-Forward CV — Mirrored vs Deduplicated Scoring")
print("=" * 68)
print(f"Features: {len(selected)}, Folds: {FOLD_YEARS}\n")

# Dedup check
n_dedup = (df["TEAM_NO_A"] < df["TEAM_NO_B"]).sum()
print(f"Total rows: {len(df)}, Dedup rows: {n_dedup} "
      f"(exactly one orientation per game)\n")

# Accumulators: mirrored vs deduped metrics per fold
results = {
    n: {"mirror": {"auc": [], "ll": [], "acc": []},
        "dedup":  {"auc": [], "ll": [], "acc": []}}
    for n in MODEL_NAMES
}

for s in FOLD_YEARS:
    print(f"  Fold {s}...", end=" ", flush=True)
    tr      = df[df["YEAR"] < s].reset_index(drop=True)
    te      = df[df["YEAR"] == s].reset_index(drop=True)
    te_dd   = te[te["TEAM_NO_A"] < te["TEAM_NO_B"]].reset_index(drop=True)

    val_year = (s - 2) if s == 2021 else (s - 1)
    fit_mask = tr["YEAR"] != val_year
    val_mask = tr["YEAR"] == val_year

    X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
    X_val = tr.loc[val_mask, selected].values.astype(np.float32)
    X_tr  = tr[selected].values.astype(np.float32)
    X_te  = te[selected].values.astype(np.float32)
    X_te_dd = te_dd[selected].values.astype(np.float32)
    y_fit = tr.loc[fit_mask, TARGET].values
    y_val = tr.loc[val_mask, TARGET].values
    y_tr  = tr[TARGET].values
    y_te  = te[TARGET].values
    y_te_dd = te_dd[TARGET].values

    # Tree imputer (full train window)
    imp     = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    Xtr_i   = imp.transform(X_tr)
    Xte_i   = imp.transform(X_te)
    Xte_i_dd = imp.transform(X_te_dd)

    # MLP imputer + scaler (fit split only)
    imp_fit  = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit)
    sc_fit   = StandardScaler().fit(imp_fit.transform(X_fit))
    Xfit_sc  = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc  = sc_fit.transform(imp_fit.transform(X_val))
    Xte_sc   = sc_fit.transform(imp_fit.transform(X_te))
    Xte_sc_dd = sc_fit.transform(imp_fit.transform(X_te_dd))

    # ── LR (seed only) ────────────────────────────────────────────────────────
    X_seed_tr = tr[["DIFF_SEED"]].values.astype(np.float32)
    X_seed_te = te[["DIFF_SEED"]].values.astype(np.float32)
    X_seed_dd = te_dd[["DIFF_SEED"]].values.astype(np.float32)
    imp_seed  = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_seed_tr)
    p_lr = model_params(params["logistic_regression"])
    lr   = LogisticRegression(C=p_lr["C"], penalty=p_lr["penalty"],
                              solver="liblinear", max_iter=1000, random_state=42)
    lr.fit(imp_seed.transform(X_seed_tr), y_tr)
    probs_lr    = lr.predict_proba(imp_seed.transform(X_seed_te))[:, 1]
    probs_lr_dd = lr.predict_proba(imp_seed.transform(X_seed_dd))[:, 1]

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(**model_params(params["random_forest"]),
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr_i, y_tr)
    probs_rf    = rf.predict_proba(Xte_i)[:, 1]
    probs_rf_dd = rf.predict_proba(Xte_i_dd)[:, 1]

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_m = xgb.XGBClassifier(**model_params(params["xgboost"]),
                                eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_i, y_tr)
    probs_xgb    = xgb_m.predict_proba(Xte_i)[:, 1]
    probs_xgb_dd = xgb_m.predict_proba(Xte_i_dd)[:, 1]

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgb_m = lgb.LGBMClassifier(**model_params(params["lightgbm"]),
                                 random_state=42, n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_i, y_tr)
    probs_lgb    = lgb_m.predict_proba(Xte_i)[:, 1]
    probs_lgb_dd = lgb_m.predict_proba(Xte_i_dd)[:, 1]

    # ── MLP ───────────────────────────────────────────────────────────────────
    net = train_mlp(Xfit_sc, y_fit, Xval_sc, y_val, params["mlp"])
    with torch.no_grad():
        probs_mlp    = net(torch.tensor(Xte_sc,    dtype=torch.float32)).squeeze(1).numpy()
        probs_mlp_dd = net(torch.tensor(Xte_sc_dd, dtype=torch.float32)).squeeze(1).numpy()

    # ── Ensemble ──────────────────────────────────────────────────────────────
    probs_ens    = (probs_rf    + probs_xgb    + probs_lgb)    / 3
    probs_ens_dd = (probs_rf_dd + probs_xgb_dd + probs_lgb_dd) / 3

    fold_probs = {
        "LR (seed)":     (probs_lr,    probs_lr_dd),
        "Random Forest": (probs_rf,    probs_rf_dd),
        "XGBoost":       (probs_xgb,   probs_xgb_dd),
        "LightGBM":      (probs_lgb,   probs_lgb_dd),
        "MLP":           (probs_mlp,   probs_mlp_dd),
        "Ensemble":      (probs_ens,   probs_ens_dd),
    }
    for name, (p_m, p_dd) in fold_probs.items():
        results[name]["mirror"]["auc"].append(roc_auc_score(y_te,    p_m))
        results[name]["mirror"]["ll"].append( log_loss(y_te,    p_m))
        results[name]["mirror"]["acc"].append(accuracy_score(y_te,    (p_m    >= 0.5).astype(int)))
        results[name]["dedup"]["auc"].append( roc_auc_score(y_te_dd, p_dd))
        results[name]["dedup"]["ll"].append(  log_loss(y_te_dd, p_dd))
        results[name]["dedup"]["acc"].append( accuracy_score(y_te_dd, (p_dd >= 0.5).astype(int)))

    print(f"Ens(mirror)={roc_auc_score(y_te, probs_ens):.4f}  "
          f"Ens(dedup)={roc_auc_score(y_te_dd, probs_ens_dd):.4f}"
          f"  ({len(te)} -> {len(te_dd)} games)", flush=True)

# ── Summary tables ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("CORRECTED CV SUMMARY — Mirror vs Deduplicated Scoring (10 folds)")
print("=" * 78)
print(f"  {'Model':<18}  {'AUC(mir)':>9}  {'AUC(ded)':>9}  "
      f"{'LL(mir)':>8}  {'LL(ded)':>8}  {'Acc(mir)':>9}  {'Acc(ded)':>9}")
print(f"  {'-'*76}")

for name in MODEL_NAMES:
    m = results[name]["mirror"]
    d = results[name]["dedup"]
    print(f"  {name:<18}  "
          f"{np.mean(m['auc']):>9.4f}  {np.mean(d['auc']):>9.4f}  "
          f"{np.mean(m['ll']):>8.4f}  {np.mean(d['ll']):>8.4f}  "
          f"{np.mean(m['acc']):>9.4f}  {np.mean(d['acc']):>9.4f}")

print()
print("Std dev across folds (dedup — honest with N=63 unique games per fold):")
print(f"  {'Model':<18}  {'AUC std':>9}  {'LL std':>9}  {'Acc std':>9}")
print(f"  {'-'*52}")
for name in MODEL_NAMES:
    d = results[name]["dedup"]
    print(f"  {name:<18}  "
          f"{np.std(d['auc']):>9.4f}  {np.std(d['ll']):>9.4f}  "
          f"{np.std(d['acc']):>9.4f}")

print()
print("Canonical dedup results (used for reporting):")
print(f"  {'Model':<18}  {'AUC':>8} +/- {'std':>6}  {'Log Loss':>10}  {'Accuracy':>10}")
print(f"  {'-'*62}")
for name in MODEL_NAMES:
    d = results[name]["dedup"]
    print(f"  {name:<18}  "
          f"{np.mean(d['auc']):>8.4f} +/- {np.std(d['auc']):>6.4f}  "
          f"{np.mean(d['ll']):>10.4f}  {np.mean(d['acc']):>10.4f}")
