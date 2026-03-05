"""
Ensemble of tuned RF + XGBoost + LightGBM (equal-weight average).

Walk-forward CV (2015-2025, excl 2020) on the 20 selected features.
Hyperparameters loaded from models/best_params.json.

Three ensemble combinations are compared:
  1. RF + XGB + LightGBM   (primary — saved to disk)
  2. RF + MLP
  3. RF + XGB + LightGBM + MLP (all four)

Output:
  models/ensemble_rf_xgb_lgbm.pkl   — component models + weights, trained on all data
  models/best_params.json            — appends 'ensemble_rf_xgb_lgbm' entry

Existing model artifacts (random_forest.pkl, xgboost.pkl, etc.) are NOT modified.
"""

import warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"

# ── Load config ───────────────────────────────────────────────────────────────
df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

print(f"Features ({len(selected)}): {selected}")
print(f"Folds: {FOLD_YEARS}\n")

# ── MLP helpers (tuned architecture) ─────────────────────────────────────────
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

def mlp_predict(net, X_sc):
    with torch.no_grad():
        return net(torch.tensor(X_sc, dtype=torch.float32)).squeeze(1).numpy()

# ── Walk-forward CV ───────────────────────────────────────────────────────────
# Accumulators: one list of per-fold metrics per model/combo
model_names = ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP",
               "Ens: RF+XGB+LGB", "Ens: RF+MLP", "Ens: RF+XGB+LGB+MLP"]

records = []  # one dict per fold

print("Running walk-forward CV...")
for s in FOLD_YEARS:
    tr = df[df["YEAR"] < s].reset_index(drop=True)
    te = df[df["YEAR"] == s].reset_index(drop=True)

    # MLP uses fit/val split (last training year as validation)
    val_year = (s - 2) if s == 2021 else (s - 1)
    fit_mask = tr["YEAR"] != val_year
    val_mask = tr["YEAR"] == val_year

    X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
    X_val = tr.loc[val_mask, selected].values.astype(np.float32)
    X_tr  = tr[selected].values.astype(np.float32)
    X_te  = te[selected].values.astype(np.float32)
    y_fit = tr.loc[fit_mask, TARGET].values
    y_val = tr.loc[val_mask, TARGET].values
    y_tr  = tr[TARGET].values
    y_te  = te[TARGET].values

    # Imputer for tree models (fitted on full training window)
    imp   = SimpleImputer(strategy="median").fit(X_tr)
    Xtr_i = imp.transform(X_tr)
    Xte_i = imp.transform(X_te)

    # Scaler for MLP (fitted on fit split only)
    imp_fit  = SimpleImputer(strategy="median").fit(X_fit)
    sc_fit   = StandardScaler().fit(imp_fit.transform(X_fit))
    Xfit_sc  = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc  = sc_fit.transform(imp_fit.transform(X_val))
    Xte_sc   = sc_fit.transform(imp_fit.transform(X_te))

    probs = {}

    # 1. LR (seed only) — needs its own 1-feature imputer
    X_seed_tr = tr[["DIFF_SEED"]].values.astype(np.float32)
    X_seed_te = te[["DIFF_SEED"]].values.astype(np.float32)
    imp_seed  = SimpleImputer(strategy="median").fit(X_seed_tr)
    p_lr = params["logistic_regression"]
    lr = LogisticRegression(C=p_lr["C"], penalty=p_lr["penalty"], solver="liblinear",
                            max_iter=1000, random_state=42)
    lr.fit(imp_seed.transform(X_seed_tr), y_tr)
    probs["LR (seed)"] = lr.predict_proba(imp_seed.transform(X_seed_te))[:, 1]

    # 2. Random Forest
    p_rf = params["random_forest"]
    rf = RandomForestClassifier(**p_rf, random_state=42, n_jobs=-1)
    rf.fit(Xtr_i, y_tr)
    probs["Random Forest"] = rf.predict_proba(Xte_i)[:, 1]

    # 3. XGBoost
    p_xgb = params["xgboost"]
    xgb_m = xgb.XGBClassifier(**p_xgb, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_i, y_tr)
    probs["XGBoost"] = xgb_m.predict_proba(Xte_i)[:, 1]

    # 4. LightGBM
    p_lgb = params["lightgbm"]
    lgb_m = lgb.LGBMClassifier(**p_lgb, random_state=42, n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_i, y_tr)
    probs["LightGBM"] = lgb_m.predict_proba(Xte_i)[:, 1]

    # 5. MLP (tuned params, early-stop on val year)
    p_mlp = params["mlp"]
    net = train_mlp(Xfit_sc, y_fit, Xval_sc, y_val, p_mlp)
    probs["MLP"] = mlp_predict(net, Xte_sc)

    # Ensemble combinations
    probs["Ens: RF+XGB+LGB"]     = (probs["Random Forest"] + probs["XGBoost"] + probs["LightGBM"]) / 3
    probs["Ens: RF+MLP"]         = (probs["Random Forest"] + probs["MLP"]) / 2
    probs["Ens: RF+XGB+LGB+MLP"] = (probs["Random Forest"] + probs["XGBoost"] +
                                     probs["LightGBM"] + probs["MLP"]) / 4

    row = {"YEAR": s}
    for name, p in probs.items():
        row[f"{name}_auc"] = roc_auc_score(y_te, p)
        row[f"{name}_ll"]  = log_loss(y_te, p)
        row[f"{name}_acc"] = accuracy_score(y_te, (p >= 0.5).astype(int))
    records.append(row)

    ens_auc = row["Ens: RF+XGB+LGB_auc"]
    rf_auc  = row["Random Forest_auc"]
    xgb_auc = row["XGBoost_auc"]
    lgb_auc = row["LightGBM_auc"]
    print(f"  {s}  RF={rf_auc:.4f}  XGB={xgb_auc:.4f}  LGB={lgb_auc:.4f}  "
          f"Ens(3)={ens_auc:.4f}")

cv = pd.DataFrame(records)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 76)
print("WALK-FORWARD CV SUMMARY — 20 selected features, tuned params")
print("=" * 76)
print(f"{'Model':<26}  {'Mean AUC':>9}  {'Std AUC':>8}  {'Mean LL':>8}  {'Delta vs RF':>12}")
print("-" * 76)

rf_mean = cv["Random Forest_auc"].mean()
summary_rows = []
for name in model_names:
    aucs = cv[f"{name}_auc"].values
    lls  = cv[f"{name}_ll"].values
    delta = aucs.mean() - rf_mean
    delta_str = f"{delta:+.4f}" if name != "Random Forest" else "  baseline"
    print(f"  {name:<24}  {aucs.mean():>9.4f}  {aucs.std():>8.4f}  "
          f"{lls.mean():>8.4f}  {delta_str:>12}")
    summary_rows.append({
        "model": name,
        "mean_auc": round(aucs.mean(), 4),
        "std_auc":  round(aucs.std(),  4),
        "mean_ll":  round(lls.mean(),  4),
    })

print()
ens3_auc  = cv["Ens: RF+XGB+LGB_auc"].mean()
rfmlp_auc = cv["Ens: RF+MLP_auc"].mean()
all4_auc  = cv["Ens: RF+XGB+LGB+MLP_auc"].mean()
best_combo = max(
    ("RF+XGB+LGB", ens3_auc),
    ("RF+MLP",     rfmlp_auc),
    ("All-4",      all4_auc),
    key=lambda x: x[1]
)
print(f"Best ensemble combination: {best_combo[0]} (AUC={best_combo[1]:.4f})")

# ── Train final ensemble on all available data ────────────────────────────────
print("\nTraining final ensemble on all data for deployment...")

X_all = df[selected].values.astype(np.float32)
y_all = df[TARGET].values
imp_final  = SimpleImputer(strategy="median").fit(X_all)
X_all_imp  = imp_final.transform(X_all)

rf_final = RandomForestClassifier(**params["random_forest"], random_state=42, n_jobs=-1)
rf_final.fit(X_all_imp, y_all)

xgb_final = xgb.XGBClassifier(**params["xgboost"], eval_metric="logloss",
                                random_state=42, n_jobs=-1)
xgb_final.fit(X_all_imp, y_all)

lgb_final = lgb.LGBMClassifier(**params["lightgbm"], random_state=42, n_jobs=-1, verbose=-1)
lgb_final.fit(X_all_imp, y_all)

ensemble_bundle = {
    "components":  ["random_forest", "xgboost", "lightgbm"],
    "weights":     [1/3, 1/3, 1/3],
    "models": {
        "random_forest": rf_final,
        "xgboost":       xgb_final,
        "lightgbm":      lgb_final,
    },
    "imputer":  imp_final,
    "features": selected,
    "cv_mean_auc": round(ens3_auc, 4),
    "cv_std_auc":  round(cv["Ens: RF+XGB+LGB_auc"].std(), 4),
}

ens_path = MODELS / "ensemble_rf_xgb_lgbm.pkl"
joblib.dump(ensemble_bundle, ens_path)
print(f"Ensemble saved -> {ens_path}")

# ── Update best_params.json ───────────────────────────────────────────────────
params["ensemble_rf_xgb_lgbm"] = {
    "components":  ["random_forest", "xgboost", "lightgbm"],
    "weights":     [1/3, 1/3, 1/3],
    "combination": "average",
    "cv_mean_auc": round(ens3_auc, 4),
    "cv_std_auc":  round(cv["Ens: RF+XGB+LGB_auc"].std(), 4),
}
params_path = MODELS / "best_params.json"
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"best_params.json updated -> {params_path}")

# ── Final six-model comparison table ─────────────────────────────────────────
print("\n" + "=" * 76)
print("FINAL MODEL COMPARISON — Walk-Forward AUC (20 features, tuned params)")
print("=" * 76)
print(f"{'Model':<26}  {'Mean AUC':>9}  {'Std':>6}  {'Mean LL':>8}  {'Mean Acc':>9}")
print("-" * 76)
for name in ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP",
             "Ens: RF+XGB+LGB"]:
    aucs = cv[f"{name}_auc"].values
    lls  = cv[f"{name}_ll"].values
    accs = cv[f"{name}_acc"].values
    marker = "  <-- ensemble" if name == "Ens: RF+XGB+LGB" else ""
    print(f"  {name:<24}  {aucs.mean():>9.4f}  {aucs.std():>6.4f}  "
          f"{lls.mean():>8.4f}  {accs.mean():>9.4f}{marker}")

print("\nAlternative ensemble combinations:")
for name, label in [("Ens: RF+MLP", "RF + MLP"),
                     ("Ens: RF+XGB+LGB+MLP", "RF + XGB + LGB + MLP (all-4)")]:
    aucs = cv[f"{name}_auc"].values
    lls  = cv[f"{name}_ll"].values
    print(f"  {label:<32}  AUC={aucs.mean():.4f} ±{aucs.std():.4f}  LL={lls.mean():.4f}")
