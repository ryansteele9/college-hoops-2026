"""
src/ensemble_search.py

Tests 3 new ensemble combinations vs. the current RF+XGB+LGB baseline.
Uses the same walk-forward loop as historical_backtest.py — per-fold models
are trained identically, then probabilities are averaged differently.
No new model training beyond what the baseline evaluation already requires.

Combos (equal-weight averaging within each combo):
  Current:  RF + XGB + LGB                    (1/3 each)
  Combo 1:  MLP + Ensemble  = MLP + RF+XGB+LGB (MLP=0.5, each tree=1/6)
  Combo 2:  MLP + RF                           (0.5 each)
  Combo 3:  MLP + XGB                          (0.5 each)

Reports walk-forward AUC (dedup) and ESPN avg/yr for each combo.
Saves the best-by-ESPN combo as ensemble_rf_xgb_lgbm_calibrated.pkl,
using the existing "ensemble" format or a new "ensemble_with_mlp" format.
"""

import warnings, json, sys, time
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED  = Path("data/processed")
MODELS     = Path("models")
RAW        = Path("data/raw")
FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
ESPN_PTS     = [10, 20, 40, 80, 160, 320]
ROUND_RANGES = [(0,32),(32,48),(48,56),(56,60),(60,62),(62,63)]

COMBO_NAMES = [
    "Current (RF+XGB+LGB)",
    "Combo1 (MLP+XGB+LGB)",
    "Combo2 (MLP+RF+XGB)",
    "Combo3 (MLP+RF+LGB)",
]

matchup_df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
master_df  = pd.read_csv(PROCESSED / "master_team_table.csv")
tourney_df = pd.read_csv(RAW / "Tournament Matchups.csv")
selected   = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
raw_feats  = [f.replace("DIFF_", "") for f in selected]
params     = json.loads((MODELS / "best_params.json").read_text())

_META_KEYS = {"decay_rate", "calibration"}
_SEED_MATCH_WR_IDX = selected.index("DIFF_SEED_MATCHUP_WINRATE") \
                     if "DIFF_SEED_MATCHUP_WINRATE" in selected else None
_SEED_MATCH_UR_IDX = selected.index("SEED_MATCHUP_UPSET_RATE") \
                     if "SEED_MATCHUP_UPSET_RATE" in selected else None


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
    net.load_state_dict(best_state)
    net.eval()
    return net


# ── Seed table helpers ────────────────────────────────────────────────────────
def build_seed_table(df: pd.DataFrame) -> dict:
    dedup = df[df["TEAM_NO_A"] < df["TEAM_NO_B"]]
    lo = dedup[["SEED_A", "SEED_B"]].min(axis=1).astype(int)
    hi = dedup[["SEED_A", "SEED_B"]].max(axis=1).astype(int)
    lo_won = (
        ((dedup["SEED_A"] <= dedup["SEED_B"]) & (dedup["TEAM_A_WIN"] == 1)) |
        ((dedup["SEED_A"] >  dedup["SEED_B"]) & (dedup["TEAM_A_WIN"] == 0))
    )
    tmp   = pd.DataFrame({"lo": lo, "hi": hi, "lo_won": lo_won})
    stats = tmp.groupby(["lo", "hi"])["lo_won"].agg(["sum", "count"])
    return (stats["sum"] / stats["count"]).to_dict()


def seed_match_vals(sa: int, sb: int, seed_table: dict) -> tuple:
    lo, hi  = min(sa, sb), max(sa, sb)
    lo_wr   = seed_table.get((lo, hi), 0.5)
    diff_wr = (2 * lo_wr - 1) if sa <= sb else -(2 * lo_wr - 1)
    return diff_wr, 1.0 - lo_wr


def patch_seed(X: np.ndarray, sa: int, sb: int, seed_table: dict) -> None:
    if _SEED_MATCH_WR_IDX is None and _SEED_MATCH_UR_IDX is None:
        return
    dw, ur = seed_match_vals(sa, sb, seed_table)
    if _SEED_MATCH_WR_IDX is not None:
        X[0, _SEED_MATCH_WR_IDX] = dw
    if _SEED_MATCH_UR_IDX is not None:
        X[0, _SEED_MATCH_UR_IDX] = ur


# ── Bracket helpers ───────────────────────────────────────────────────────────
_NAN_FEATS = {f: np.nan for f in raw_feats}
_NAN_FEATS["_SEED"] = 8


def get_year_features(year: int) -> dict:
    yr_df = master_df[master_df["YEAR"] == year]
    lookup = {}
    for _, row in yr_df.iterrows():
        tn = int(row["TEAM NO"])
        d  = {f: row.get(f, np.nan) for f in raw_feats}
        d["_SEED"] = int(row["SEED"])
        lookup[tn] = d
    return lookup


def get_bracket_games(year: int) -> list:
    yr = (tourney_df[tourney_df["YEAR"] == year]
          .sort_values("BY YEAR NO", ascending=False).reset_index(drop=True))
    games = []
    for i in range(0, len(yr), 2):
        r1, r2    = yr.iloc[i], yr.iloc[i + 1]
        winner_no = int(r1["TEAM NO"]) if r1["ROUND"] < r2["ROUND"] else int(r2["TEAM NO"])
        games.append((int(r1["TEAM NO"]), int(r2["TEAM NO"]), winner_no))
    if len(games) < 63:
        n_r64     = len(yr[yr["CURRENT ROUND"] == 64]) // 2
        r64_teams = {tn for g in games[:n_r64] for tn in (g[0], g[1])}
        r32_teams = {tn for g in games[n_r64:n_r64+16] for tn in (g[0], g[1])}
        for bye_tn in (r32_teams - r64_teams):
            r32_game_idx = next(gi for gi, g in enumerate(games[n_r64:n_r64+16])
                                if bye_tn in (g[0], g[1]))
            exp_a, exp_b = 2 * r32_game_idx, 2 * r32_game_idx + 1
            r32_g        = games[n_r64 + r32_game_idx]
            partner_tn   = r32_g[1] if r32_g[0] == bye_tn else r32_g[0]
            partner_pos  = next(i for i, g in enumerate(games[:n_r64])
                                if partner_tn in (g[0], g[1]))
            ins = exp_b if partner_pos < exp_b else exp_a
            games.insert(ins, (bye_tn, -1, bye_tn))
            n_r64 += 1
    return games


def simulate_bracket(r64_pairs: list, predictor, feat_lookup: dict) -> list:
    slots, predicted = list(r64_pairs), []
    while slots:
        winners = []
        for a_no, b_no in slots:
            if a_no == -1:   w = b_no
            elif b_no == -1: w = a_no
            else:
                fa = feat_lookup.get(a_no, _NAN_FEATS)
                fb = feat_lookup.get(b_no, _NAN_FEATS)
                w  = a_no if predictor(fa, fb) >= 0.5 else b_no
            predicted.append(w)
            winners.append(w)
        if len(winners) == 1:
            break
        slots = [(winners[k], winners[k+1]) for k in range(0, len(winners), 2)]
    return predicted


def score_espn(predicted: list, actual: list) -> int:
    total = 0
    for rnd, (start, end) in enumerate(ROUND_RANGES):
        for g in range(start, end):
            if predicted[g] == actual[g]:
                total += ESPN_PTS[rnd]
    return total


# ── Walk-forward loop ─────────────────────────────────────────────────────────
print(f"Ensemble search — walk-forward {FOLD_YEARS[0]}–{FOLD_YEARS[-1]} (excl 2020)")
print(f"Features: {len(selected)}  |  Combos: {len(COMBO_NAMES)}\n")

fold_aucs  = {c: [] for c in COMBO_NAMES}
fold_espns = {c: [] for c in COMBO_NAMES}
fold_espn_table = []  # per-year rows for display

t_total = time.time()

for year in FOLD_YEARS:
    t0 = time.time()
    print(f"  Fold {year}", end="  ", flush=True)

    tr      = matchup_df[matchup_df["YEAR"] < year].reset_index(drop=True)
    te      = matchup_df[matchup_df["YEAR"] == year].reset_index(drop=True)
    te_dd   = te[te["TEAM_NO_A"] < te["TEAM_NO_B"]].reset_index(drop=True)

    val_year  = (year - 2) if year == 2021 else (year - 1)
    fit_mask  = tr["YEAR"] != val_year
    val_mask  = tr["YEAR"] == val_year

    X_tr  = tr[selected].values.astype(np.float32)
    X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
    X_val = tr.loc[val_mask, selected].values.astype(np.float32)
    X_te  = te_dd[selected].values.astype(np.float32)
    y_tr  = tr[TARGET].values
    y_fit = tr.loc[fit_mask, TARGET].values
    y_val = tr.loc[val_mask, TARGET].values
    y_te  = te_dd[TARGET].values

    imp     = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    imp_fit = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit)
    sc_fit  = StandardScaler().fit(imp_fit.transform(X_fit))

    Xtr_imp = imp.transform(X_tr)
    Xte_imp = imp.transform(X_te)
    Xte_sc  = sc_fit.transform(imp_fit.transform(X_te))
    Xfit_sc = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc = sc_fit.transform(imp_fit.transform(X_val))

    # Train models (identical to historical_backtest.py per-fold training)
    rf_m  = RandomForestClassifier(**model_params(params["random_forest"]),
                                   random_state=42, n_jobs=-1)
    rf_m.fit(Xtr_imp, y_tr)

    xgb_m = xgb.XGBClassifier(**model_params(params["xgboost"]),
                                eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_m.fit(Xtr_imp, y_tr)

    lgb_m = lgb.LGBMClassifier(**model_params(params["lightgbm"]),
                                 random_state=42, n_jobs=-1, verbose=-1)
    lgb_m.fit(Xtr_imp, y_tr)

    mlp_net = train_mlp(Xfit_sc, y_fit, Xval_sc, y_val, params["mlp"])

    print(f"trained ({time.time()-t0:.0f}s)", end="  ", flush=True)

    # ── AUC on dedup test fold ────────────────────────────────────────────────
    p_rf  = rf_m.predict_proba(Xte_imp)[:, 1]
    p_xgb = xgb_m.predict_proba(Xte_imp)[:, 1]
    p_lgb = lgb_m.predict_proba(Xte_imp)[:, 1]
    with torch.no_grad():
        p_mlp = mlp_net(torch.tensor(Xte_sc, dtype=torch.float32)).squeeze().numpy()

    p_ens = (p_rf + p_xgb + p_lgb) / 3

    combos_proba = {
        "Current (RF+XGB+LGB)": p_ens,
        "Combo1 (MLP+XGB+LGB)": (p_mlp + p_xgb + p_lgb) / 3,
        "Combo2 (MLP+RF+XGB)":  (p_mlp + p_rf  + p_xgb) / 3,
        "Combo3 (MLP+RF+LGB)":  (p_mlp + p_rf  + p_lgb) / 3,
    }
    for c, proba in combos_proba.items():
        fold_aucs[c].append(roc_auc_score(y_te, proba))

    # ── ESPN bracket simulation ───────────────────────────────────────────────
    seed_table  = build_seed_table(tr)
    feat_lookup = get_year_features(year)
    all_games   = get_bracket_games(year)
    r64_pairs   = [(a, b) for a, b, _ in all_games[:32]]
    actual_wins = [w for _, _, w in all_games]

    def _make_pred(combo):
        def pred(fa, fb):
            X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in raw_feats]],
                         dtype=np.float32)
            patch_seed(X, fa["_SEED"], fb["_SEED"], seed_table)
            Xi  = imp.transform(X)
            Xs  = sc_fit.transform(imp_fit.transform(X))
            prf  = float(rf_m.predict_proba(Xi)[:, 1][0])
            pxgb = float(xgb_m.predict_proba(Xi)[:, 1][0])
            plgb = float(lgb_m.predict_proba(Xi)[:, 1][0])
            with torch.no_grad():
                pmlp = float(mlp_net(torch.tensor(Xs, dtype=torch.float32)).squeeze())
            pens = (prf + pxgb + plgb) / 3
            if   combo == "Current (RF+XGB+LGB)": return pens
            elif combo == "Combo1 (MLP+XGB+LGB)": return (pmlp + pxgb + plgb) / 3
            elif combo == "Combo2 (MLP+RF+XGB)":  return (pmlp + prf  + pxgb) / 3
            else:                                   return (pmlp + prf  + plgb) / 3
        return pred

    year_row = {"YEAR": year}
    for c in COMBO_NAMES:
        pb    = simulate_bracket(r64_pairs, _make_pred(c), feat_lookup)
        score = score_espn(pb, actual_wins)
        fold_espns[c].append(score)
        year_row[c] = score
    fold_espn_table.append(year_row)

    best_yr = max(COMBO_NAMES, key=lambda c: year_row[c])
    scores_str = "  ".join(f"{c[:6]}={year_row[c]}" for c in COMBO_NAMES)
    print(f"| {scores_str}  best={best_yr}  ({time.time()-t0:.0f}s total)", flush=True)

print(f"\nTotal time: {time.time()-t_total:.0f}s\n")


# ── Summary table ─────────────────────────────────────────────────────────────
COL_W = 24

print("=" * 80)
print("ENSEMBLE SEARCH RESULTS  (walk-forward, 10 folds 2015-2025 excl 2020)")
print("=" * 80)
print(f"  {'Combo':<{COL_W}}  {'AUC (dedup)':>12}  {'ESPN avg/yr':>12}  {'ESPN total':>12}")
print("  " + "-" * 64)
for c in COMBO_NAMES:
    auc_mean   = np.mean(fold_aucs[c])
    espn_mean  = np.mean(fold_espns[c])
    espn_total = int(np.sum(fold_espns[c]))
    marker = " *" if c == max(COMBO_NAMES, key=lambda x: np.mean(fold_espns[x])) else "  "
    print(f"  {c:<{COL_W}}  {auc_mean:>12.4f}  {espn_mean:>12.1f}  {espn_total:>12}{marker}")

print()
print("Per-year ESPN scores:")
hdr = f"  {'Year':<6}" + "".join(f"  {c[:14]:>14}" for c in COMBO_NAMES)
print(hdr)
print("  " + "-" * (6 + len(COMBO_NAMES) * 16))
for row in fold_espn_table:
    best = max(COMBO_NAMES, key=lambda c: row[c])
    line = f"  {row['YEAR']:<6}"
    for c in COMBO_NAMES:
        marker = " *" if c == best else "  "
        line += f"  {row[c]:>14}{marker}"
    print(line)

best_combo = max(COMBO_NAMES, key=lambda c: np.mean(fold_espns[c]))
best_espn  = np.mean(fold_espns[best_combo])
print(f"\nWinner: {best_combo}  (ESPN avg/yr: {best_espn:.1f})")

baseline_espn = np.mean(fold_espns["Current (RF+XGB+LGB)"])
print(f"Baseline: Current (RF+XGB+LGB)  (ESPN avg/yr: {baseline_espn:.1f})")
print(f"Delta vs baseline: {best_espn - baseline_espn:+.1f}")


# ── Save best combo as new ensemble pkl ───────────────────────────────────────
print(f"\nSaving {best_combo} as ensemble_rf_xgb_lgbm_calibrated.pkl ...")

rf_prod  = joblib.load(MODELS / "random_forest_calibrated.pkl")
xgb_prod = joblib.load(MODELS / "xgboost_calibrated.pkl")
lgb_prod = joblib.load(MODELS / "lightgbm_calibrated.pkl")
mlp_prod = joblib.load(MODELS / "mlp_calibrated.pkl")

if best_combo == "Current (RF+XGB+LGB)":
    print("  Best is current ensemble — existing pkl unchanged.")

else:
    # Determine tree components for this combo (each model gets equal weight = 1/3)
    if best_combo == "Combo1 (MLP+XGB+LGB)":
        tree_names   = ["xgboost", "lightgbm"]
        tree_weights = [1/3, 1/3]
        tree_models  = {"xgboost": xgb_prod["model"], "lightgbm": lgb_prod["model"]}
    elif best_combo == "Combo2 (MLP+RF+XGB)":
        tree_names   = ["random_forest", "xgboost"]
        tree_weights = [1/3, 1/3]
        tree_models  = {"random_forest": rf_prod["model"], "xgboost": xgb_prod["model"]}
    else:  # Combo3 MLP+RF+LGB
        tree_names   = ["random_forest", "lightgbm"]
        tree_weights = [1/3, 1/3]
        tree_models  = {"random_forest": rf_prod["model"], "lightgbm": lgb_prod["model"]}

    bundle = {
        "model_type":      "ensemble_with_mlp",
        "components":      ["mlp"] + tree_names,
        "mlp_weight":      1/3,
        "mlp_state_dict":  mlp_prod["net_state_dict"],
        "mlp_net_params":  mlp_prod["net_params"],
        "mlp_n_features":  mlp_prod["n_features"],
        "mlp_imputer":     mlp_prod["imputer"],
        "mlp_scaler":      mlp_prod["scaler"],
        "tree_components": tree_names,
        "tree_weights":    tree_weights,
        "tree_models":     tree_models,
        "tree_imputer":    rf_prod["imputer"],
        "features":        selected,
    }
    joblib.dump(bundle, MODELS / "ensemble_rf_xgb_lgbm_calibrated.pkl")
    print(f"  Saved: ensemble_with_mlp  components={['mlp'] + tree_names}")
    print(f"  mlp_weight=0.5  tree_weights={tree_weights}")
    print(f"  NOTE: bracket_simulator.py must handle 'ensemble_with_mlp' — "
          f"run this script first, then patch is auto-applied.")

print("\nDone.")
