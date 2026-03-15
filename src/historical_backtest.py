"""
src/historical_backtest.py

Walk-forward ESPN bracket backtest: for each tournament year 2015-2025
(excl 2020), train each of the 6 models on strictly prior years, simulate
the full 63-game bracket deterministically, and score against actual results
using standard ESPN bracket scoring.

ESPN scoring:
  R64=10  R32=20  S16=40  E8=80  F4=160  Championship=320
  Max possible: 32×10 + 16×20 + 8×40 + 4×80 + 2×160 + 1×320 = 1920 pts

Also runs 10,000 Monte Carlo trials per year with the Ensemble model
and reports the average simulated ESPN score across trials.

Outputs:
  data/processed/backtest_results.csv   — per-model per-year ESPN scores
  models/backtest_espn_scores.png       — grouped bar chart by year
"""

import warnings, json, sys, time
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
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
import xgboost as xgb
import lightgbm as lgb

# ── Paths and constants ───────────────────────────────────────────────────────
PROCESSED  = Path("data/processed")
MODELS     = Path("models")
RAW        = Path("data/raw")

FOLD_YEARS = [y for y in range(2015, 2026) if y != 2020]
TARGET     = "TEAM_A_WIN"
MC_TRIALS  = 10_000

# ESPN bracket scoring: index = round (0=R64 … 5=Championship)
ESPN_PTS        = [10, 20, 40, 80, 160, 320]
ROUND_RANGES    = [(0,32),(32,48),(48,56),(56,60),(60,62),(62,63)]
ROUND_LABELS    = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
MAX_ESPN        = sum(ESPN_PTS[r] * (ROUND_RANGES[r][1] - ROUND_RANGES[r][0])
                      for r in range(6))   # 1920

MODEL_NAMES     = ["LR (seed)", "Random Forest", "XGBoost",
                   "LightGBM", "MLP", "Ensemble"]
ALL_COLS        = MODEL_NAMES + ["Ensemble MC (mean)"]

_META_KEYS      = {"decay_rate", "calibration"}

# ── Data loading ──────────────────────────────────────────────────────────────
print("Loading data...", flush=True)
matchup_df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
master_df  = pd.read_csv(PROCESSED / "master_team_table.csv")
tourney_df = pd.read_csv(RAW / "Tournament Matchups.csv")
selected   = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
raw_feats  = [f.replace("DIFF_", "") for f in selected]

# Indices of seed matchup features in the selected list (None if not present)
_SEED_MATCH_WR_IDX = selected.index("DIFF_SEED_MATCHUP_WINRATE") \
                     if "DIFF_SEED_MATCHUP_WINRATE" in selected else None
_SEED_MATCH_UR_IDX = selected.index("SEED_MATCHUP_UPSET_RATE") \
                     if "SEED_MATCHUP_UPSET_RATE"   in selected else None
params     = json.loads((MODELS / "best_params.json").read_text())


def model_params(p: dict) -> dict:
    return {k: v for k, v in p.items() if k not in _META_KEYS}


def build_seed_table(df: pd.DataFrame) -> dict:
    """Return {(seed_lo, seed_hi): lo_win_rate} from matchup rows (prior years)."""
    dedup = df[df["TEAM_NO_A"] < df["TEAM_NO_B"]]
    lo = dedup[["SEED_A", "SEED_B"]].min(axis=1).astype(int)
    hi = dedup[["SEED_A", "SEED_B"]].max(axis=1).astype(int)
    lo_won = (
        ((dedup["SEED_A"] <= dedup["SEED_B"]) & (dedup["TEAM_A_WIN"] == 1)) |
        ((dedup["SEED_A"] >  dedup["SEED_B"]) & (dedup["TEAM_A_WIN"] == 0))
    )
    tmp = pd.DataFrame({"lo": lo, "hi": hi, "lo_won": lo_won})
    stats = tmp.groupby(["lo", "hi"])["lo_won"].agg(["sum", "count"])
    return (stats["sum"] / stats["count"]).to_dict()


def seed_match_vals(sa: int, sb: int, seed_table: dict) -> tuple:
    """Return (diff_wr, upset_rate) for seeds sa vs sb given historical table."""
    lo, hi = min(sa, sb), max(sa, sb)
    lo_wr   = seed_table.get((lo, hi), 0.5)
    diff_wr = (2 * lo_wr - 1) if sa <= sb else -(2 * lo_wr - 1)
    return diff_wr, 1.0 - lo_wr


def _patch_seed_feats_vec(X: np.ndarray, sa: int, sb: int,
                          seed_table: dict) -> None:
    """Patch seed matchup features into X[0, :] in-place."""
    if _SEED_MATCH_WR_IDX is None and _SEED_MATCH_UR_IDX is None:
        return
    dw, ur = seed_match_vals(sa, sb, seed_table)
    if _SEED_MATCH_WR_IDX is not None:
        X[0, _SEED_MATCH_WR_IDX] = dw
    if _SEED_MATCH_UR_IDX is not None:
        X[0, _SEED_MATCH_UR_IDX] = ur


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
    opt  = torch.optim.Adam(net.parameters(), lr=p["lr"],
                             weight_decay=p["weight_decay"])
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
            opt.zero_grad()
            crit(net(Xb).squeeze(1), yb).backward()
            opt.step()
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


# ── Predictor factories ───────────────────────────────────────────────────────
def make_lr_predictor(lr_model, imp_seed):
    def pred(fa, fb):
        X = np.array([[fa.get("_SEED", 8) - fb.get("_SEED", 8)]], dtype=np.float32)
        return float(lr_model.predict_proba(imp_seed.transform(X))[:, 1][0])
    return pred


def make_tree_predictor(model, imputer, seed_table=None):
    def pred(fa, fb):
        X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in raw_feats]],
                     dtype=np.float32)
        if seed_table is not None:
            _patch_seed_feats_vec(X, fa["_SEED"], fb["_SEED"], seed_table)
        return float(model.predict_proba(imputer.transform(X))[:, 1][0])
    return pred


def make_mlp_predictor(net, imp_fit, sc_fit, seed_table=None):
    def pred(fa, fb):
        X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in raw_feats]],
                     dtype=np.float32)
        if seed_table is not None:
            _patch_seed_feats_vec(X, fa["_SEED"], fb["_SEED"], seed_table)
        X = sc_fit.transform(imp_fit.transform(X))
        with torch.no_grad():
            return float(net(torch.tensor(X, dtype=torch.float32)).squeeze())
    return pred


def make_ensemble_predictor(rf_m, xgb_m, lgb_m, imputer, seed_table=None):
    """Ensemble predictor with batch_predict for fast MC matrix build."""
    def pred(fa, fb):
        X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in raw_feats]],
                     dtype=np.float32)
        if seed_table is not None:
            _patch_seed_feats_vec(X, fa["_SEED"], fb["_SEED"], seed_table)
        X_imp = imputer.transform(X)
        return (float(rf_m.predict_proba(X_imp)[:, 1][0]) +
                float(xgb_m.predict_proba(X_imp)[:, 1][0]) +
                float(lgb_m.predict_proba(X_imp)[:, 1][0])) / 3

    def batch_pred(X_diffs: np.ndarray) -> np.ndarray:
        """Seed matchup features must be pre-patched by caller before this call."""
        X_imp = imputer.transform(X_diffs.astype(np.float32))
        return (rf_m.predict_proba(X_imp)[:, 1] +
                xgb_m.predict_proba(X_imp)[:, 1] +
                lgb_m.predict_proba(X_imp)[:, 1]) / 3

    pred.batch_predict = batch_pred
    return pred


# ── Feature lookup ────────────────────────────────────────────────────────────
def get_year_features(year: int) -> dict:
    """Return {team_no: {raw_feat: value, '_SEED': seed}} for all teams in year."""
    yr_df = master_df[master_df["YEAR"] == year]
    lookup = {}
    for _, row in yr_df.iterrows():
        tn = int(row["TEAM NO"])
        d  = {f: row.get(f, np.nan) for f in raw_feats}
        d["_SEED"] = int(row["SEED"])
        lookup[tn] = d
    return lookup


# ── Bracket reconstruction ────────────────────────────────────────────────────
def get_bracket_games(year: int) -> list:
    """
    Return list of 63 (team_no_a, team_no_b, actual_winner_no) in bracket order:
      games 0-31  = R64,  32-47 = R32,  48-55 = S16,  56-59 = E8,
      60-61 = F4,  62 = Championship.
    Consecutive row-pairs sorted by BY YEAR NO desc are game pairs;
    the team with the smaller ROUND value is the actual winner.

    2021 special case: VCU withdrew (COVID), giving Oregon a bye into R32.
    A synthetic game (bye_team, -1, bye_team) is inserted at the correct R64
    bracket slot so the downstream tree structure stays valid.  -1 = ghost/BYE.
    """
    yr = (tourney_df[tourney_df["YEAR"] == year]
          .sort_values("BY YEAR NO", ascending=False)
          .reset_index(drop=True))
    games = []
    for i in range(0, len(yr), 2):
        r1, r2 = yr.iloc[i], yr.iloc[i + 1]
        winner_no = int(r1["TEAM NO"]) if r1["ROUND"] < r2["ROUND"] \
                    else int(r2["TEAM NO"])
        games.append((int(r1["TEAM NO"]), int(r2["TEAM NO"]), winner_no))

    # Fix incomplete R64 (bye team — in R32 but never played R64)
    if len(games) < 63:
        n_r64 = len(yr[yr["CURRENT ROUND"] == 64]) // 2   # 31 for 2021
        r64_teams = {tn for g in games[:n_r64] for tn in (g[0], g[1])}
        r32_teams = {tn for g in games[n_r64 : n_r64 + 16] for tn in (g[0], g[1])}
        for bye_tn in (r32_teams - r64_teams):
            # Which R32 game (0-indexed within the 16-game R32 block) has bye_tn?
            r32_game_idx = next(
                gi for gi, g in enumerate(games[n_r64 : n_r64 + 16])
                if bye_tn in (g[0], g[1]))
            exp_a = 2 * r32_game_idx          # expected R64 slot A
            exp_b = 2 * r32_game_idx + 1      # expected R64 slot B
            # Partner = the real R64 winner that feeds this R32 game
            r32_g = games[n_r64 + r32_game_idx]
            partner_tn = r32_g[1] if r32_g[0] == bye_tn else r32_g[0]
            partner_pos = next(i for i, g in enumerate(games[:n_r64])
                               if partner_tn in (g[0], g[1]))
            # Insert synthetic bye at the slot that isn't the partner's
            ins = exp_b if partner_pos < exp_b else exp_a
            games.insert(ins, (bye_tn, -1, bye_tn))   # -1 = BYE ghost
            n_r64 += 1

    return games  # 63 triples


# ── Bracket simulation ────────────────────────────────────────────────────────
_NAN_FEATS = {f: np.nan for f in raw_feats}
_NAN_FEATS["_SEED"] = 8


def simulate_bracket(r64_pairs: list, predictor, feat_lookup: dict) -> list:
    """
    Deterministic bracket simulation.
    r64_pairs: 32 (team_no_a, team_no_b) in bracket order.
    -1 denotes a BYE ghost — the paired real team always advances.
    Returns list of 63 predicted winner team_nos (games 0-62).
    """
    slots = list(r64_pairs)
    predicted = []
    while slots:
        winners = []
        for a_no, b_no in slots:
            if a_no == -1:
                w = b_no
            elif b_no == -1:
                w = a_no
            else:
                fa = feat_lookup.get(a_no, _NAN_FEATS)
                fb = feat_lookup.get(b_no, _NAN_FEATS)
                p  = predictor(fa, fb)
                w  = a_no if p >= 0.5 else b_no
            predicted.append(w)
            winners.append(w)
        if len(winners) == 1:
            break
        slots = [(winners[k], winners[k + 1]) for k in range(0, len(winners), 2)]
    return predicted  # 63 items


# ── ESPN scoring ──────────────────────────────────────────────────────────────
def score_espn(predicted: list, actual: list) -> int:
    """ESPN points for a predicted bracket vs actual game-by-game winners."""
    total = 0
    for rnd, (start, end) in enumerate(ROUND_RANGES):
        pts = ESPN_PTS[rnd]
        for g in range(start, end):
            if predicted[g] == actual[g]:
                total += pts
    return total


# ── Monte Carlo ESPN scoring ──────────────────────────────────────────────────
def simulate_mc_espn(r64_pairs: list, feat_lookup: dict,
                     ens_pred, actual_winners: list,
                     n_trials: int = MC_TRIALS,
                     seed_table: dict | None = None) -> float:
    """
    Run n_trials stochastic bracket simulations using precomputed prob matrix.
    Returns mean ESPN score across trials.
    """
    # Build ordered team list from all R64 teams (exclude -1 BYE ghost)
    all_team_nos = []
    seen = set()
    for a, b in r64_pairs:
        for tn in (a, b):
            if tn != -1 and tn not in seen:
                all_team_nos.append(tn)
                seen.add(tn)
    n = len(all_team_nos)
    idx = {tn: i for i, tn in enumerate(all_team_nos)}

    # Precompute prob matrix via batched predict
    all_feats = np.array(
        [[feat_lookup.get(tn, _NAN_FEATS).get(f, np.nan) for f in raw_feats]
         for tn in all_team_nos], dtype=np.float32)

    prob_matrix = np.zeros((n, n), dtype=np.float32)
    if hasattr(ens_pred, "batch_predict"):
        pairs     = [(i, j) for i in range(n) for j in range(n) if i != j]
        X_diffs   = all_feats[[p[0] for p in pairs]] - all_feats[[p[1] for p in pairs]]
        # Patch seed matchup features for each pair before calling batch predict
        if seed_table is not None and \
                (_SEED_MATCH_WR_IDX is not None or _SEED_MATCH_UR_IDX is not None):
            all_seeds = [feat_lookup.get(tn, _NAN_FEATS).get("_SEED", 8)
                         for tn in all_team_nos]
            for k, (i, j) in enumerate(pairs):
                dw, ur = seed_match_vals(int(all_seeds[i]), int(all_seeds[j]),
                                         seed_table)
                if _SEED_MATCH_WR_IDX is not None:
                    X_diffs[k, _SEED_MATCH_WR_IDX] = dw
                if _SEED_MATCH_UR_IDX is not None:
                    X_diffs[k, _SEED_MATCH_UR_IDX] = ur
        probs_flat = ens_pred.batch_predict(X_diffs)
        for k, (i, j) in enumerate(pairs):
            prob_matrix[i, j] = probs_flat[k]
    else:
        for i, tn_a in enumerate(all_team_nos):
            for j, tn_b in enumerate(all_team_nos):
                if i != j:
                    fa = feat_lookup.get(tn_a, _NAN_FEATS)
                    fb = feat_lookup.get(tn_b, _NAN_FEATS)
                    prob_matrix[i, j] = ens_pred(fa, fb)

    # Build initial slots; normalize so BYE (-1) is always the second element
    init_slots = []
    for a, b in r64_pairs:
        if a == -1:
            init_slots.append((idx[b], -1))
        elif b == -1:
            init_slots.append((idx[a], -1))
        else:
            init_slots.append((idx[a], idx[b]))

    rng          = np.random.default_rng(42)
    all_preds    = np.zeros((n_trials, 63), dtype=np.int32)
    actual_arr   = np.array(actual_winners, dtype=np.int32)

    for trial in range(n_trials):
        slots = list(init_slots)
        game  = 0
        while slots:
            winners_i = []
            for ai, bi in slots:
                if bi == -1:          # BYE: real team always wins
                    w = ai
                else:
                    w = ai if rng.random() < prob_matrix[ai, bi] else bi
                all_preds[trial, game] = all_team_nos[w]
                game += 1
                winners_i.append(w)
            if len(winners_i) == 1:
                break
            slots = [(winners_i[k], winners_i[k + 1])
                     for k in range(0, len(winners_i), 2)]

    # Batch ESPN scoring across all trials
    scores = np.zeros(n_trials, dtype=np.int32)
    for rnd, (start, end) in enumerate(ROUND_RANGES):
        correct  = (all_preds[:, start:end] == actual_arr[start:end]).sum(axis=1)
        scores  += ESPN_PTS[rnd] * correct

    return float(np.mean(scores))


# ── Main backtest loop ────────────────────────────────────────────────────────
print(f"Walk-forward ESPN bracket backtest: {FOLD_YEARS[0]}-{FOLD_YEARS[-1]} "
      f"(excl 2020)  |  max ESPN = {MAX_ESPN}\n", flush=True)

all_results = []
t_total = time.time()

for year in FOLD_YEARS:
    t0 = time.time()
    print(f"Year {year}", end="  ", flush=True)

    # ── Training data ────────────────────────────────────────────────────────
    tr = matchup_df[matchup_df["YEAR"] < year].reset_index(drop=True)

    val_year = (year - 2) if year == 2021 else (year - 1)
    fit_mask = tr["YEAR"] != val_year
    val_mask = tr["YEAR"] == val_year

    X_fit = tr.loc[fit_mask, selected].values.astype(np.float32)
    X_val = tr.loc[val_mask, selected].values.astype(np.float32)
    X_tr  = tr[selected].values.astype(np.float32)
    y_fit = tr.loc[fit_mask, TARGET].values
    y_val = tr.loc[val_mask, TARGET].values
    y_tr  = tr[TARGET].values

    X_seed_tr = tr[["DIFF_SEED"]].values.astype(np.float32)

    # ── Imputers / scalers ───────────────────────────────────────────────────
    imp      = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_tr)
    imp_seed = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_seed_tr)
    imp_fit  = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_fit)
    sc_fit   = StandardScaler().fit(imp_fit.transform(X_fit))

    Xtr_imp  = imp.transform(X_tr)
    Xfit_sc  = sc_fit.transform(imp_fit.transform(X_fit))
    Xval_sc  = sc_fit.transform(imp_fit.transform(X_val))

    # ── Train models ─────────────────────────────────────────────────────────
    p_lr  = model_params(params["logistic_regression"])
    lr_m  = LogisticRegression(C=p_lr["C"], penalty=p_lr["penalty"],
                                solver="liblinear", max_iter=1000, random_state=42)
    lr_m.fit(imp_seed.transform(X_seed_tr), y_tr)

    rf_m = RandomForestClassifier(**model_params(params["random_forest"]),
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

    # ── Seed matchup table (years < current year) ────────────────────────────
    seed_table = build_seed_table(tr)

    # ── Build bracket ────────────────────────────────────────────────────────
    feat_lookup = get_year_features(year)
    all_games   = get_bracket_games(year)          # 63 triples
    r64_pairs   = [(a, b) for a, b, _ in all_games[:32]]
    actual_wins = [w for _, _, w in all_games]     # 63 actual winners

    # ── Deterministic simulations ─────────────────────────────────────────────
    ens_pred = make_ensemble_predictor(rf_m, xgb_m, lgb_m, imp,
                                       seed_table=seed_table)
    predictors = {
        "LR (seed)":     make_lr_predictor(lr_m, imp_seed),
        "Random Forest": make_tree_predictor(rf_m, imp, seed_table=seed_table),
        "XGBoost":       make_tree_predictor(xgb_m, imp, seed_table=seed_table),
        "LightGBM":      make_tree_predictor(lgb_m, imp, seed_table=seed_table),
        "MLP":           make_mlp_predictor(mlp_net, imp_fit, sc_fit,
                                            seed_table=seed_table),
        "Ensemble":      ens_pred,
    }

    row = {"YEAR": year}
    for mname, pred_fn in predictors.items():
        pred_bracket = simulate_bracket(r64_pairs, pred_fn, feat_lookup)
        row[mname]   = score_espn(pred_bracket, actual_wins)

    # ── Monte Carlo (Ensemble) ────────────────────────────────────────────────
    mc_mean       = simulate_mc_espn(r64_pairs, feat_lookup, ens_pred,
                                     actual_wins, n_trials=MC_TRIALS,
                                     seed_table=seed_table)
    row["Ensemble MC (mean)"] = round(mc_mean, 1)

    all_results.append(row)
    best_m = max(MODEL_NAMES, key=lambda m: row[m])
    scores_str = "  ".join(f"{m[:4]}={row[m]}" for m in MODEL_NAMES)
    print(f"| {scores_str}  MC={row['Ensemble MC (mean)']:.0f}  "
          f"best={best_m}  ({time.time()-t0:.0f}s total)", flush=True)

print(f"\nTotal backtest time: {time.time()-t_total:.0f}s\n", flush=True)


# ── Build results DataFrame ───────────────────────────────────────────────────
results_df = pd.DataFrame(all_results).set_index("YEAR")

# Add totals and averages
results_df.loc["Total"]   = results_df[MODEL_NAMES].sum().tolist() + \
                             [round(results_df["Ensemble MC (mean)"].sum(), 1)]
results_df.loc["Average"] = results_df.loc[FOLD_YEARS, MODEL_NAMES].mean().round(1).tolist() + \
                             [round(results_df.loc[FOLD_YEARS, "Ensemble MC (mean)"].mean(), 1)]

out_csv = PROCESSED / "backtest_results.csv"
results_df.to_csv(out_csv)
print(f"Results saved -> {out_csv}")


# ── Summary table ─────────────────────────────────────────────────────────────
# Per-year best model
year_bests = {yr: max(MODEL_NAMES, key=lambda m: results_df.loc[yr, m])
              for yr in FOLD_YEARS}
best_counts = {m: sum(1 for b in year_bests.values() if b == m) for m in MODEL_NAMES}

print()
print("=" * 90)
print("ESPN BRACKET BACKTEST SUMMARY  (walk-forward, all years 2015-2025 excl 2020)")
print(f"Max possible per year: {MAX_ESPN} pts")
print("=" * 90)

# Header
hdr = f"  {'Year':<6}" + "".join(f"  {m[:14]:>14}" for m in ALL_COLS)
print(hdr)
print("  " + "-" * 88)

for yr in FOLD_YEARS:
    best = year_bests[yr]
    line = f"  {yr:<6}"
    for m in ALL_COLS:
        val = results_df.loc[yr, m]
        marker = " *" if m == best else "  "
        line += f"  {val:>12.0f}{marker}"
    print(line)

print("  " + "-" * 88)
for label in ["Total", "Average"]:
    line = f"  {label:<6}"
    for m in ALL_COLS:
        val = results_df.loc[label, m]
        line += f"  {val:>14.1f}"
    print(line)

print()
print("  * = best model for that year")
print()
print("  Wins per model (best ESPN score in year):")
for m in MODEL_NAMES:
    bar = "█" * best_counts[m]
    print(f"    {m:<18}  {best_counts[m]:2d}  {bar}")

print()
print(f"  Best overall (total ESPN): "
      f"{max(MODEL_NAMES, key=lambda m: results_df.loc['Total', m])}")
print(f"  Avg ESPN across all years: "
      + "  ".join(f"{m[:4]}={results_df.loc['Average', m]:.0f}" for m in MODEL_NAMES))


# ── Bar chart ──────────────────────────────────────────────────────────────────
colors = {
    "LR (seed)":     "#9E9E9E",
    "Random Forest": "#FF7043",
    "XGBoost":       "#4CAF50",
    "LightGBM":      "#F44336",
    "MLP":           "#2196F3",
    "Ensemble":      "#9C27B0",
}

fig, ax = plt.subplots(figsize=(14, 6))

year_labels = [str(y) for y in FOLD_YEARS]
n_years     = len(FOLD_YEARS)
n_models    = len(MODEL_NAMES)
bar_width   = 0.13
offsets     = np.arange(n_models) * bar_width - (n_models - 1) * bar_width / 2

for mi, mname in enumerate(MODEL_NAMES):
    scores = [results_df.loc[yr, mname] for yr in FOLD_YEARS]
    xs     = np.arange(n_years) + offsets[mi]
    bars   = ax.bar(xs, scores, width=bar_width, label=mname,
                    color=colors[mname], alpha=0.85, edgecolor="white", linewidth=0.5)

# Highlight best model per year with a gold star above the tallest bar
for yi, yr in enumerate(FOLD_YEARS):
    best = year_bests[yr]
    best_score = results_df.loc[yr, best]
    bi = MODEL_NAMES.index(best)
    ax.text(yi + offsets[bi], best_score + 12, "★",
            ha="center", va="bottom", fontsize=9, color="#FFD700")

# Monte Carlo line (Ensemble MC mean)
mc_scores = [results_df.loc[yr, "Ensemble MC (mean)"] for yr in FOLD_YEARS]
ax.plot(np.arange(n_years), mc_scores, "k--o", linewidth=1.5, markersize=5,
        label="Ensemble MC (mean)", zorder=5, alpha=0.7)

ax.axhline(MAX_ESPN, color="gray", linestyle=":", linewidth=1, alpha=0.5,
           label=f"Max ({MAX_ESPN})")

ax.set_xticks(np.arange(n_years))
ax.set_xticklabels(year_labels, fontsize=10)
ax.set_xlabel("Tournament Year", fontsize=11)
ax.set_ylabel("ESPN Bracket Score", fontsize=11)
ax.set_title("Walk-Forward ESPN Bracket Backtest (2015–2025, excl 2020)\n"
             "★ = best model for year  |  dashed line = Ensemble MC mean",
             fontsize=12)
ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
ax.set_ylim(0, MAX_ESPN * 1.08)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
chart_path = MODELS / "backtest_espn_scores.png"
fig.savefig(chart_path, dpi=150)
print(f"\nChart saved -> {chart_path}")
