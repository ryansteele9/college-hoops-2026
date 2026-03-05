"""
Recency weighting: exponential decay for training samples.

Weight formula: w_i = decay_rate ^ (s - YEAR_i)
  s      = test fold year
  YEAR_i = tournament year of training sample i
  decay_rate = 1.0 → uniform weights (no effect)
  decay_rate < 1.0 → older seasons down-weighted

Optuna (20 trials per model) finds optimal decay_rate ∈ [0.7, 1.0]:
  RF, XGBoost, LightGBM  → sample_weight passed to .fit()
  MLP                    → per-sample weighted BCE (training only);
                           unweighted BCE used for val / early-stop since
                           all val samples share the same year (constant factor).

Before/after AUC reported for each model.
Best decay_rate written into models/best_params.json under each model's key.
"""

import warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED  = Path("data/processed")
MODELS     = Path("models")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
N_TRIALS   = 20

# ── Load config ───────────────────────────────────────────────────────────────
df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

print(f"Features ({len(selected)})")
print(f"Folds: {FOLD_YEARS}")
print(f"Trials per model: {N_TRIALS}\n")


# ── Weight helper ─────────────────────────────────────────────────────────────
def sample_weights(train_years: np.ndarray, test_year: int, decay_rate: float) -> np.ndarray:
    """w_i = decay_rate ^ (test_year - YEAR_i).  All weights > 0."""
    return decay_rate ** (test_year - train_years).astype(np.float32)


# ── Walk-forward evaluators ───────────────────────────────────────────────────
def wf_auc_rf(decay_rate: float) -> np.ndarray:
    p = params["random_forest"]
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        X_tr = tr[selected].values.astype(np.float32)
        X_te = te[selected].values.astype(np.float32)
        y_tr = tr[TARGET].values
        y_te = te[TARGET].values
        w_tr = sample_weights(tr["YEAR"].values, s, decay_rate)
        imp  = SimpleImputer(strategy="median").fit(X_tr)
        rf   = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
        rf.fit(imp.transform(X_tr), y_tr, sample_weight=w_tr)
        aucs.append(roc_auc_score(y_te, rf.predict_proba(imp.transform(X_te))[:, 1]))
    return np.array(aucs)


def wf_auc_xgb(decay_rate: float) -> np.ndarray:
    p = params["xgboost"]
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        X_tr = tr[selected].values.astype(np.float32)
        X_te = te[selected].values.astype(np.float32)
        y_tr = tr[TARGET].values
        y_te = te[TARGET].values
        w_tr = sample_weights(tr["YEAR"].values, s, decay_rate)
        imp  = SimpleImputer(strategy="median").fit(X_tr)
        m    = xgb.XGBClassifier(**p, eval_metric="logloss", random_state=42, n_jobs=-1)
        m.fit(imp.transform(X_tr), y_tr, sample_weight=w_tr)
        aucs.append(roc_auc_score(y_te, m.predict_proba(imp.transform(X_te))[:, 1]))
    return np.array(aucs)


def wf_auc_lgb(decay_rate: float) -> np.ndarray:
    p = params["lightgbm"]
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s]
        te = df[df["YEAR"] == s]
        X_tr = tr[selected].values.astype(np.float32)
        X_te = te[selected].values.astype(np.float32)
        y_tr = tr[TARGET].values
        y_te = te[TARGET].values
        w_tr = sample_weights(tr["YEAR"].values, s, decay_rate)
        imp  = SimpleImputer(strategy="median").fit(X_tr)
        m    = lgb.LGBMClassifier(**p, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(imp.transform(X_tr), y_tr, sample_weight=w_tr)
        aucs.append(roc_auc_score(y_te, m.predict_proba(imp.transform(X_te))[:, 1]))
    return np.array(aucs)


# ── MLP helpers ───────────────────────────────────────────────────────────────
def build_mlp(n_in: int, hidden: int, n_layers: int, dropout: float) -> nn.Sequential:
    layers, in_dim = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


def train_mlp_weighted(Xfit_sc, yfit, w_fit, Xval_sc, yval, p,
                       max_epochs=200, patience=15):
    """Train MLP with per-sample weighted BCE on fit set; unweighted BCE on val."""
    net  = build_mlp(Xfit_sc.shape[1], p["hidden_size"], p["n_layers"], p["dropout"])
    opt  = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    bce  = nn.BCELoss(reduction="none")

    t_Xfit = torch.tensor(Xfit_sc, dtype=torch.float32)
    t_yfit = torch.tensor(yfit,    dtype=torch.float32)
    t_wfit = torch.tensor(w_fit,   dtype=torch.float32)
    t_Xval = torch.tensor(Xval_sc, dtype=torch.float32)
    t_yval = torch.tensor(yval,    dtype=torch.float32)

    dl = DataLoader(TensorDataset(t_Xfit, t_yfit, t_wfit),
                    batch_size=64, shuffle=True, drop_last=True)

    best_loss, wait, best_state = np.inf, 0, None
    for _ in range(max_epochs):
        net.train()
        for Xb, yb, wb in dl:
            opt.zero_grad()
            loss = (wb * bce(net(Xb).squeeze(1), yb)).sum() / wb.sum()
            loss.backward()
            opt.step()
        # Val uses unweighted BCE (all val samples same year → constant weight factor)
        net.eval()
        with torch.no_grad():
            vl = nn.BCELoss()(net(t_Xval).squeeze(1), t_yval).item()
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


def wf_auc_mlp(decay_rate: float) -> np.ndarray:
    p = params["mlp"]
    aucs = []
    for s in FOLD_YEARS:
        tr = df[df["YEAR"] < s].reset_index(drop=True)
        te = df[df["YEAR"] == s].reset_index(drop=True)
        val_year = (s - 2) if s == 2021 else (s - 1)
        fit_mask = tr["YEAR"] != val_year
        val_mask = tr["YEAR"] == val_year

        X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
        X_val = tr.loc[val_mask, selected].values.astype(np.float32)
        X_te  = te[selected].values.astype(np.float32)
        y_fit = tr.loc[fit_mask, TARGET].values
        y_val = tr.loc[val_mask, TARGET].values
        y_te  = te[TARGET].values
        w_fit = sample_weights(tr.loc[fit_mask, "YEAR"].values, s, decay_rate)

        imp = SimpleImputer(strategy="median").fit(X_fit)
        sc  = StandardScaler().fit(imp.transform(X_fit))
        Xfit_sc = sc.transform(imp.transform(X_fit))
        Xval_sc = sc.transform(imp.transform(X_val))
        Xte_sc  = sc.transform(imp.transform(X_te))

        net = train_mlp_weighted(Xfit_sc, y_fit, w_fit, Xval_sc, y_val, p)
        with torch.no_grad():
            prob = net(torch.tensor(Xte_sc, dtype=torch.float32)).squeeze(1).numpy()
        aucs.append(roc_auc_score(y_te, prob))
    return np.array(aucs)


# ═════════════════════════════════════════════════════════════════════════════
# Untuned baselines (decay_rate = 1.0, i.e., existing tuned params, no recency)
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 62)
print("Step 1: Unweighted baselines (decay_rate=1.0, tuned params)")
print("=" * 62)

before = {}
for name, fn in [("Random Forest", wf_auc_rf),
                 ("XGBoost",       wf_auc_xgb),
                 ("LightGBM",      wf_auc_lgb),
                 ("MLP",           wf_auc_mlp)]:
    a = fn(1.0)
    before[name] = a
    print(f"  {name:<18}: {a.mean():.4f} +/- {a.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Optuna: tune decay_rate per model
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print(f"Step 2: Optuna decay_rate search ({N_TRIALS} trials per model)")
print(f"{'=' * 62}")

best_decay = {}

model_configs = [
    ("Random Forest", wf_auc_rf),
    ("XGBoost",       wf_auc_xgb),
    ("LightGBM",      wf_auc_lgb),
    ("MLP",           wf_auc_mlp),
]

for name, fn in model_configs:
    print(f"\n  Tuning {name}...")

    def objective(trial, _fn=fn):
        dr = trial.suggest_float("decay_rate", 0.7, 1.0)
        return _fn(dr).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    dr_star = study.best_params["decay_rate"]
    best_decay[name] = dr_star
    print(f"  Best decay_rate: {dr_star:.4f}  (trial AUC: {study.best_value:.4f})")

    # Show top-5 trials to illustrate the decay landscape
    trials_df = study.trials_dataframe()[["params_decay_rate", "value"]].sort_values(
        "value", ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"    decay={row['params_decay_rate']:.4f}  AUC={row['value']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# After: evaluate with optimal decay_rate per model
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 62}")
print("Step 3: Weighted evaluation with optimal decay_rate")
print(f"{'=' * 62}")

after = {}
for name, fn in model_configs:
    dr = best_decay[name]
    a  = fn(dr)
    after[name] = a
    print(f"  {name:<18}: {a.mean():.4f} +/- {a.std():.4f}  (decay={dr:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# Comparison table
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("RECENCY WEIGHTING — Before vs After (walk-forward AUC, 10 folds)")
print(f"{'=' * 70}")
print(f"  {'Model':<18}  {'Before':>8}  {'After':>8}  {'Delta':>7}  {'Decay':>6}")
print(f"  {'-'*64}")
for name in ["Random Forest", "XGBoost", "LightGBM", "MLP"]:
    b_mean = before[name].mean()
    a_mean = after[name].mean()
    delta  = a_mean - b_mean
    dr     = best_decay[name]
    marker = " *" if delta > 0 else ""
    print(f"  {name:<18}  {b_mean:>8.4f}  {a_mean:>8.4f}  {delta:>+7.4f}  {dr:>6.4f}{marker}")

# ═════════════════════════════════════════════════════════════════════════════
# Save to best_params.json
# ═════════════════════════════════════════════════════════════════════════════
key_map = {
    "Random Forest": "random_forest",
    "XGBoost":       "xgboost",
    "LightGBM":      "lightgbm",
    "MLP":           "mlp",
}
for name, key in key_map.items():
    params[key]["decay_rate"] = round(best_decay[name], 6)

params_path = MODELS / "best_params.json"
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"\nDecay rates saved -> {params_path}")
print("\nFinal decay rates per model:")
for name, key in key_map.items():
    print(f"  {name:<18}: {params[key]['decay_rate']}")
