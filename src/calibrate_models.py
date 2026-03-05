"""
Platt scaling (sigmoid calibration) for all six models.

Calibration is fit on out-of-fold walk-forward CV predictions — never on
the test fold directly.

Evaluation protocol: leave-one-fold-out calibration
  For each test fold s: fit calibrator on OOF predictions from the 9 other folds,
  apply to fold s, measure ECE and log loss before/after.

Deployment: final calibrator fit on all 10 pooled OOF predictions.

Outputs:
  models/calibration_plots/{model}_reliability.png  — reliability diagrams
  models/lr_seed_calibrated.pkl
  models/random_forest_calibrated.pkl
  models/xgboost_calibrated.pkl
  models/lightgbm_calibrated.pkl
  models/mlp_calibrated.pkl
  models/ensemble_rf_xgb_lgbm_calibrated.pkl
  models/best_params.json  — calibration metrics appended per model
"""

import warnings, json, sys
warnings.filterwarnings("ignore")
# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED  = Path("data/processed")
MODELS     = Path("models")
PLOTS_DIR  = MODELS / "calibration_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
N_BINS     = 10

df       = pd.read_csv(PROCESSED / "matchup_dataset.csv")
selected = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
params   = json.loads((MODELS / "best_params.json").read_text())

print(f"Features ({len(selected)}), Folds: {FOLD_YEARS}\n")

MODEL_NAMES = ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP", "Ensemble"]

# Keys added by post-processing steps that are not sklearn constructor args
_META_KEYS = {"decay_rate", "calibration"}


def model_params(p: dict) -> dict:
    """Return only the constructor-compatible keys from a params dict."""
    return {k: v for k, v in p.items() if k not in _META_KEYS}


# ── ECE ───────────────────────────────────────────────────────────────────────
def ece_score(y_true, y_prob, n_bins=N_BINS):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)


# ── Platt calibration helpers ─────────────────────────────────────────────────
def fit_calibrator(probs: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit sigmoid (Platt) calibration: logistic on raw prob as single feature."""
    cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10_000)
    cal.fit(probs.reshape(-1, 1), labels)
    return cal


def apply_calibrator(cal: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    return cal.predict_proba(probs.reshape(-1, 1))[:, 1]


# ── MLP helpers (mirrors build_ensemble.py) ───────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Collect OOF predictions for all 6 models via walk-forward CV
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("Step 1: Walk-forward CV — collecting OOF predictions")
print("=" * 66)

oof = {name: {"y_true": [], "y_prob": [], "year": []} for name in MODEL_NAMES}

for s in FOLD_YEARS:
    print(f"  Fold {s}...", end=" ", flush=True)
    tr = df[df["YEAR"] < s].reset_index(drop=True)
    te = df[df["YEAR"] == s].reset_index(drop=True)

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

    # Shared imputer for tree models (fit on full train window)
    imp   = SimpleImputer(strategy="median").fit(X_tr)
    Xtr_i = imp.transform(X_tr)
    Xte_i = imp.transform(X_te)

    # MLP: separate imputer + scaler on fit split
    imp_fit = SimpleImputer(strategy="median").fit(X_fit)
    sc_fit  = StandardScaler().fit(imp_fit.transform(X_fit))
    Xfit_sc = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc = sc_fit.transform(imp_fit.transform(X_val))
    Xte_sc  = sc_fit.transform(imp_fit.transform(X_te))

    # LR (seed only)
    X_seed_tr = tr[["DIFF_SEED"]].values.astype(np.float32)
    X_seed_te = te[["DIFF_SEED"]].values.astype(np.float32)
    imp_seed  = SimpleImputer(strategy="median").fit(X_seed_tr)
    p_lr = model_params(params["logistic_regression"])
    lr   = LogisticRegression(C=p_lr["C"], penalty=p_lr["penalty"],
                              solver="liblinear", max_iter=1000, random_state=42)
    lr.fit(imp_seed.transform(X_seed_tr), y_tr)
    probs_lr = lr.predict_proba(imp_seed.transform(X_seed_te))[:, 1]

    # Random Forest
    rf = RandomForestClassifier(**model_params(params["random_forest"]), random_state=42, n_jobs=-1)
    rf.fit(Xtr_i, y_tr)
    probs_rf = rf.predict_proba(Xte_i)[:, 1]

    # XGBoost
    xgb_m = xgb.XGBClassifier(**model_params(params["xgboost"]), eval_metric="logloss",
                                random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_i, y_tr)
    probs_xgb = xgb_m.predict_proba(Xte_i)[:, 1]

    # LightGBM
    lgb_m = lgb.LGBMClassifier(**model_params(params["lightgbm"]), random_state=42,
                                n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_i, y_tr)
    probs_lgb = lgb_m.predict_proba(Xte_i)[:, 1]

    # MLP
    net      = train_mlp(Xfit_sc, y_fit, Xval_sc, y_val, params["mlp"])
    probs_mlp = mlp_predict(net, Xte_sc)

    # Ensemble (equal-weight average)
    probs_ens = (probs_rf + probs_xgb + probs_lgb) / 3

    fold_probs = {
        "LR (seed)":     probs_lr,
        "Random Forest": probs_rf,
        "XGBoost":       probs_xgb,
        "LightGBM":      probs_lgb,
        "MLP":           probs_mlp,
        "Ensemble":      probs_ens,
    }
    for name, probs in fold_probs.items():
        oof[name]["y_true"].extend(y_te.tolist())
        oof[name]["y_prob"].extend(probs.tolist())
        oof[name]["year"].extend([s] * len(y_te))

    print(f"RF={roc_auc_score(y_te, probs_rf):.4f}  "
          f"Ens={roc_auc_score(y_te, probs_ens):.4f}", flush=True)

# Consolidate to arrays
for name in MODEL_NAMES:
    oof[name]["y_true"] = np.array(oof[name]["y_true"])
    oof[name]["y_prob"] = np.array(oof[name]["y_prob"])
    oof[name]["year"]   = np.array(oof[name]["year"])

print("\nOOF collection complete.\n")


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Leave-one-fold-out calibration evaluation
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("Step 2: Leave-one-fold-out calibration evaluation")
print("=" * 66)

results = {
    name: {"before": {"ece": [], "ll": [], "auc": []},
           "after":  {"ece": [], "ll": [], "auc": []}}
    for name in MODEL_NAMES
}

for s in FOLD_YEARS:
    for name in MODEL_NAMES:
        d          = oof[name]
        test_mask  = d["year"] == s
        train_mask = ~test_mask

        y_te_cal  = d["y_true"][test_mask]
        p_te_raw  = d["y_prob"][test_mask]
        y_tr_cal  = d["y_true"][train_mask]
        p_tr_cal  = d["y_prob"][train_mask]

        cal      = fit_calibrator(p_tr_cal, y_tr_cal)
        p_te_cal = apply_calibrator(cal, p_te_raw)

        results[name]["before"]["ece"].append(ece_score(y_te_cal, p_te_raw))
        results[name]["before"]["ll"].append(log_loss(y_te_cal, p_te_raw))
        results[name]["before"]["auc"].append(roc_auc_score(y_te_cal, p_te_raw))
        results[name]["after"]["ece"].append(ece_score(y_te_cal, p_te_cal))
        results[name]["after"]["ll"].append(log_loss(y_te_cal, p_te_cal))
        results[name]["after"]["auc"].append(roc_auc_score(y_te_cal, p_te_cal))

print("Done.\n")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Reliability diagrams
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("Step 3: Plotting reliability diagrams")
print("=" * 66)


def reliability_data(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    mean_conf, mean_acc, counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            mean_conf.append(y_prob[mask].mean())
            mean_acc.append(y_true[mask].mean())
            counts.append(mask.sum())
    return np.array(mean_conf), np.array(mean_acc), np.array(counts)


# Fit final calibrators on all pooled OOF data (for plotting and deployment)
final_cals = {
    name: fit_calibrator(oof[name]["y_prob"], oof[name]["y_true"])
    for name in MODEL_NAMES
}

for name in MODEL_NAMES:
    y_true = oof[name]["y_true"]
    p_raw  = oof[name]["y_prob"]
    p_cal  = apply_calibrator(final_cals[name], p_raw)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    safe_name = (name.replace(" ", "_").replace("(", "").replace(")", "")
                     .replace("+", "plus"))
    fig.suptitle(f"{name} — Reliability Diagram", fontsize=13, fontweight="bold")

    for ax, probs, title, color in [
        (axes[0], p_raw, "Before Calibration",  "steelblue"),
        (axes[1], p_cal, "After Platt Scaling",  "darkorange"),
    ]:
        conf, acc, cnts = reliability_data(y_true, probs)
        ece_val = ece_score(y_true, probs)
        ll_val  = log_loss(y_true, probs)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration",
                zorder=1)
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

        # Reliability curve with error bars sized by bin count
        ax.plot(conf, acc, "o-", color=color, linewidth=2.5, markersize=8,
                label="Model", zorder=3)

        # Shade the gap between ideal and actual
        for c, a in zip(conf, acc):
            ax.plot([c, c], [c, a], color=color, alpha=0.4, linewidth=1.5, zorder=2)

        ax.set_title(f"{title}\nECE = {ece_val:.4f}   Log Loss = {ll_val:.4f}",
                     fontsize=11)
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of Positives",      fontsize=10)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)

        # Inset: predicted probability histogram
        ax_in = ax.inset_axes([0.57, 0.07, 0.40, 0.28])
        ax_in.hist(probs, bins=20, color=color, alpha=0.65,
                   edgecolor="white", linewidth=0.3)
        ax_in.set_xlim(0, 1); ax_in.set_yticks([])
        ax_in.set_xlabel("Predicted prob.", fontsize=7)
        ax_in.set_title("Distribution", fontsize=7)
        ax_in.tick_params(labelsize=6)

    plt.tight_layout()
    out_path = PLOTS_DIR / f"{safe_name}_reliability.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")

# Combined 6-panel summary figure
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Reliability Diagrams — All Models (Before vs After Calibration)",
             fontsize=13, fontweight="bold")

for ax, name in zip(axes.flat, MODEL_NAMES):
    y_true = oof[name]["y_true"]
    p_raw  = oof[name]["y_prob"]
    p_cal  = apply_calibrator(final_cals[name], p_raw)

    conf_b, acc_b, _ = reliability_data(y_true, p_raw)
    conf_a, acc_a, _ = reliability_data(y_true, p_cal)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal", zorder=1)
    ax.plot(conf_b, acc_b, "o-", color="steelblue",   linewidth=2, markersize=6,
            label=f"Raw  (ECE={ece_score(y_true, p_raw):.3f})")
    ax.plot(conf_a, acc_a, "s-", color="darkorange",  linewidth=2, markersize=6,
            label=f"Cal  (ECE={ece_score(y_true, p_cal):.3f})")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Fraction positives",    fontsize=9)

plt.tight_layout()
combined_path = PLOTS_DIR / "all_models_reliability.png"
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {combined_path.name}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Summary table
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 76)
print("CALIBRATION SUMMARY — Leave-One-Fold-Out (10 folds)")
print("=" * 76)
print(f"  {'Model':<18}  {'ECE Bef':>8}  {'ECE Aft':>8}  "
      f"{'LL Bef':>8}  {'LL Aft':>8}  {'dLL':>8}  {'dECE':>8}")
print(f"  {'-'*70}")

summary_rows = {}
for name in MODEL_NAMES:
    b = results[name]["before"]
    a = results[name]["after"]
    ece_b = np.mean(b["ece"]); ece_a = np.mean(a["ece"])
    ll_b  = np.mean(b["ll"]);  ll_a  = np.mean(a["ll"])
    auc_b = np.mean(b["auc"]); auc_a = np.mean(a["auc"])
    d_ll  = ll_a  - ll_b
    d_ece = ece_a - ece_b
    marker = "  *" if d_ll < -0.003 else ""
    sign_ll  = "+" if d_ll  >= 0 else ""
    sign_ece = "+" if d_ece >= 0 else ""
    print(f"  {name:<18}  {ece_b:>8.4f}  {ece_a:>8.4f}  "
          f"{ll_b:>8.4f}  {ll_a:>8.4f}  {sign_ll}{d_ll:.4f}  {sign_ece}{d_ece:.4f}{marker}")
    summary_rows[name] = {
        "ece_before": round(ece_b, 4), "ece_after":  round(ece_a, 4),
        "ll_before":  round(ll_b, 4),  "ll_after":   round(ll_a, 4),
        "auc_before": round(auc_b, 4), "auc_after":  round(auc_a, 4),
    }

print()


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: Train final models and save calibrated wrappers
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 66)
print("Step 5: Training final models + saving calibrated wrappers")
print("=" * 66)

X_all     = df[selected].values.astype(np.float32)
y_all     = df[TARGET].values
imp_final = SimpleImputer(strategy="median").fit(X_all)
X_all_imp = imp_final.transform(X_all)

# LR (seed)
X_seed_all     = df[["DIFF_SEED"]].values.astype(np.float32)
imp_seed_final = SimpleImputer(strategy="median").fit(X_seed_all)
p_lr_p = model_params(params["logistic_regression"])
lr_final = LogisticRegression(C=p_lr_p["C"], penalty=p_lr_p["penalty"],
                               solver="liblinear", max_iter=1000, random_state=42)
lr_final.fit(imp_seed_final.transform(X_seed_all), y_all)
joblib.dump({
    "model_type": "logistic_regression",
    "model":      lr_final,
    "imputer":    imp_seed_final,
    "features":   ["DIFF_SEED"],
    "calibrator": final_cals["LR (seed)"],
    **{f"cv_{k}": v for k, v in summary_rows["LR (seed)"].items()},
}, MODELS / "lr_seed_calibrated.pkl")
print("  Saved: lr_seed_calibrated.pkl")

# Random Forest
rf_final = RandomForestClassifier(**model_params(params["random_forest"]), random_state=42, n_jobs=-1)
rf_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "random_forest",
    "model":      rf_final,
    "imputer":    imp_final,
    "features":   selected,
    "calibrator": final_cals["Random Forest"],
    **{f"cv_{k}": v for k, v in summary_rows["Random Forest"].items()},
}, MODELS / "random_forest_calibrated.pkl")
print("  Saved: random_forest_calibrated.pkl")

# XGBoost
xgb_final = xgb.XGBClassifier(**model_params(params["xgboost"]), eval_metric="logloss",
                                random_state=42, n_jobs=-1)
xgb_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "xgboost",
    "model":      xgb_final,
    "imputer":    imp_final,
    "features":   selected,
    "calibrator": final_cals["XGBoost"],
    **{f"cv_{k}": v for k, v in summary_rows["XGBoost"].items()},
}, MODELS / "xgboost_calibrated.pkl")
print("  Saved: xgboost_calibrated.pkl")

# LightGBM
lgb_final = lgb.LGBMClassifier(**model_params(params["lightgbm"]), random_state=42,
                                 n_jobs=-1, verbose=-1)
lgb_final.fit(X_all_imp, y_all)
joblib.dump({
    "model_type": "lightgbm",
    "model":      lgb_final,
    "imputer":    imp_final,
    "features":   selected,
    "calibrator": final_cals["LightGBM"],
    **{f"cv_{k}": v for k, v in summary_rows["LightGBM"].items()},
}, MODELS / "lightgbm_calibrated.pkl")
print("  Saved: lightgbm_calibrated.pkl")

# MLP — train on all data except 2024 held out for early-stop val
val_yr_final = 2024
fit_mask_all = df["YEAR"] != val_yr_final
val_mask_all = df["YEAR"] == val_yr_final
X_fit_all    = df.loc[fit_mask_all, selected].values.astype(np.float32)
X_val_all    = df.loc[val_mask_all, selected].values.astype(np.float32)
y_fit_all    = df.loc[fit_mask_all, TARGET].values
y_val_all    = df.loc[val_mask_all, TARGET].values

imp_mlp  = SimpleImputer(strategy="median").fit(X_fit_all)
sc_mlp   = StandardScaler().fit(imp_mlp.transform(X_fit_all))
Xfit_sc2 = sc_mlp.transform(imp_mlp.transform(X_fit_all))
Xval_sc2 = sc_mlp.transform(imp_mlp.transform(X_val_all))

net_final = train_mlp(Xfit_sc2, y_fit_all, Xval_sc2, y_val_all, params["mlp"])
joblib.dump({
    "model_type":      "mlp",
    "net_state_dict":  net_final.state_dict(),
    "net_params":      {k: params["mlp"][k]
                        for k in ["hidden_size", "n_layers", "dropout"]},
    "n_features":      len(selected),
    "imputer":         imp_mlp,
    "scaler":          sc_mlp,
    "features":        selected,
    "calibrator":      final_cals["MLP"],
    **{f"cv_{k}": v for k, v in summary_rows["MLP"].items()},
}, MODELS / "mlp_calibrated.pkl")
print("  Saved: mlp_calibrated.pkl")

# Ensemble (component models already trained above)
joblib.dump({
    "model_type": "ensemble",
    "components": ["random_forest", "xgboost", "lightgbm"],
    "weights":    [1/3, 1/3, 1/3],
    "models": {
        "random_forest": rf_final,
        "xgboost":       xgb_final,
        "lightgbm":      lgb_final,
    },
    "imputer":    imp_final,
    "features":   selected,
    "calibrator": final_cals["Ensemble"],
    **{f"cv_{k}": v for k, v in summary_rows["Ensemble"].items()},
}, MODELS / "ensemble_rf_xgb_lgbm_calibrated.pkl")
print("  Saved: ensemble_rf_xgb_lgbm_calibrated.pkl")


# ── Update best_params.json ────────────────────────────────────────────────────
key_map = {
    "LR (seed)":     "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost":       "xgboost",
    "LightGBM":      "lightgbm",
    "MLP":           "mlp",
    "Ensemble":      "ensemble_rf_xgb_lgbm",
}
for name, key in key_map.items():
    params[key]["calibration"] = summary_rows[name]

with open(MODELS / "best_params.json", "w") as f:
    json.dump(params, f, indent=2)
print(f"\nbest_params.json updated with calibration metrics.")
print("\nDone.")
