"""
NCAA Tournament Bracket Simulator

Accepts a bracket CSV with columns:
    TEAM_NO   — integer, matches master_team_table "TEAM NO"
    TEAM      — display name
    SEED      — integer 1-16
    REGION    — string (4 distinct values; Final Four pairs regions[0]/[1] and [2]/[3])
    YEAR      — (optional) which season's features to use; defaults to most recent available

Outputs (data/processed/simulator_outputs/):
    deterministic_{model}.csv  — round-by-round bracket picks per model
    montecarlo_probs.csv       — per-team probability of reaching each round (MLP Monte Carlo)
    sample_bracket_2025.csv    — ready-to-use 2025 bracket for backtesting

Usage:
    python src/bracket_simulator.py                          # uses sample 2025 bracket
    python src/bracket_simulator.py --bracket path/to/bracket.csv
"""

import argparse, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

PROCESSED   = Path("data/processed")
MODELS      = Path("models")
RAW         = Path("data/raw")
OUT_DIR     = PROCESSED / "simulator_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED    = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
# Raw column names (strip DIFF_ prefix)
RAW_FEATS   = [f.replace("DIFF_", "") for f in SELECTED]
TARGET      = "TEAM_A_WIN"

# Standard seed bracket order within a region
SEED_BRACKET_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
ROUND_NAMES = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "F2", 1: "CHAMP"}

# ── MLP definition ────────────────────────────────────────────────────────────
def build_mlp(n_in, hidden, n_layers, dropout):
    layers, dim = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(dim, hidden), nn.BatchNorm1d(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        dim = hidden
    layers += [nn.Linear(dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)

def train_mlp_full(X_sc, y, hidden, n_layers, dropout, lr, wd,
                   X_val_sc, y_val, max_epochs=300, patience=20):
    n_in = X_sc.shape[1]
    net  = build_mlp(n_in, hidden, n_layers, dropout)
    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCELoss()
    dl   = DataLoader(TensorDataset(torch.tensor(X_sc, dtype=torch.float32),
                                    torch.tensor(y,   dtype=torch.float32)),
                      batch_size=64, shuffle=True, drop_last=True)
    t_Xv = torch.tensor(X_val_sc, dtype=torch.float32)
    t_yv = torch.tensor(y_val,    dtype=torch.float32)
    best, wait, state = np.inf, 0, None
    for _ in range(max_epochs):
        net.train()
        for Xb, yb in dl:
            opt.zero_grad(); crit(net(Xb).squeeze(1), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = crit(net(t_Xv).squeeze(1), t_yv).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    net.load_state_dict(state); return net

# ── Feature lookup ─────────────────────────────────────────────────────────────
def build_feature_lookup():
    """Return {(team_no, year): {raw_feat: value}} from master + KenPom."""
    master = pd.read_csv(PROCESSED / "master_team_table.csv")
    kenpom = pd.read_csv(RAW / "KenPom Barttorvik.csv",
                         usecols=["YEAR", "TEAM NO", "KADJ O"]
                         ).rename(columns={"KADJ O": "ADJOE"})
    merged = master.merge(kenpom, on=["YEAR", "TEAM NO"], how="left")

    lookup = {}
    for _, row in merged.iterrows():
        key = (int(row["TEAM NO"]), int(row["YEAR"]))
        lookup[key] = {f: row.get(f, np.nan) for f in RAW_FEATS}
    return lookup

def get_team_features(team_no, year, lookup):
    """Return feature dict for a team; fall back to most recent year, then NaN."""
    if (team_no, year) in lookup:
        return lookup[(team_no, year)]
    # Most recent year available for this team
    years = [y for (tn, y) in lookup if tn == team_no]
    if years:
        return lookup[(team_no, max(years))]
    return {f: np.nan for f in RAW_FEATS}

def matchup_features(feat_a, feat_b):
    """Compute DIFF_ feature vector (Team A − Team B)."""
    return np.array([[feat_a.get(f, np.nan) - feat_b.get(f, np.nan)
                      for f in RAW_FEATS]], dtype=np.float32)

# ── Model training ─────────────────────────────────────────────────────────────
def fit_all_models(best_params):
    """Train all five models on the full historical matchup dataset."""
    df = pd.read_csv(PROCESSED / "matchup_dataset.csv")
    X_raw = df[SELECTED].values.astype(np.float32)
    y     = df[TARGET].values

    imp  = SimpleImputer(strategy="median").fit(X_raw)
    X_imp = imp.transform(X_raw)
    sc    = StandardScaler().fit(X_imp)
    X_sc  = sc.transform(X_imp)

    # Val split for MLP early stopping: last available year
    last_year = df["YEAR"].max()
    val_mask  = df["YEAR"] == last_year
    fit_mask  = ~val_mask
    imp_fit = SimpleImputer(strategy="median").fit(X_raw[fit_mask])
    sc_fit  = StandardScaler().fit(imp_fit.transform(X_raw[fit_mask]))
    X_fit_sc = sc_fit.transform(imp_fit.transform(X_raw[fit_mask]))
    X_val_sc = sc_fit.transform(imp_fit.transform(X_raw[val_mask]))

    # LR: seed-only baseline (DIFF_SEED is not in SELECTED; load separately)
    X_seed  = df[["DIFF_SEED"]].values.astype(np.float32)
    seed_sc = StandardScaler().fit(X_seed)
    p = best_params["logistic_regression"]
    lr_model = LogisticRegression(C=p["C"], solver="liblinear",
                                   max_iter=1000, random_state=42)
    lr_model.fit(seed_sc.transform(X_seed), y)

    p  = best_params["random_forest"]
    rf = RandomForestClassifier(**p, random_state=42, n_jobs=-1)
    rf.fit(X_imp, y)

    p    = best_params["xgboost"]
    xgbm = xgb.XGBClassifier(**p, eval_metric="logloss",
                               random_state=42, n_jobs=-1)
    xgbm.fit(X_imp, y)

    p    = best_params["lightgbm"]
    lgbm = lgb.LGBMClassifier(**p, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(X_imp, y)

    mp = best_params["mlp"]
    mlp = train_mlp_full(X_fit_sc, y[fit_mask], mp["hidden_size"], mp["n_layers"],
                          mp["dropout"], mp["lr"], mp["weight_decay"],
                          X_val_sc, y[val_mask])

    return {
        "LR (seed)":    ("lr",  lr_model,  imp, seed_sc),
        "Random Forest":("rf",  rf,        imp, None),
        "XGBoost":      ("xgb", xgbm,      imp, None),
        "LightGBM":     ("lgb", lgbm,      imp, None),
        "MLP":          ("mlp", mlp,       imp, sc),
    }

def predict_prob(feat_a, feat_b, model_bundle):
    """Return P(Team A beats Team B) for a given model."""
    kind, model, imp, sc = model_bundle
    X = matchup_features(feat_a, feat_b)

    if kind == "lr":
        diff_seed = feat_a.get("_SEED", 0) - feat_b.get("_SEED", 0)
        X_seed = np.array([[diff_seed]], dtype=np.float32)
        return float(model.predict_proba(sc.transform(X_seed))[:, 1][0])

    X_imp = imp.transform(X)
    if kind == "mlp":
        X_sc = sc.transform(X_imp)
        model.eval()
        with torch.no_grad():
            return float(model(torch.tensor(X_sc, dtype=torch.float32)).squeeze())
    return float(model.predict_proba(X_imp)[:, 1][0])

# ── Bracket simulation ────────────────────────────────────────────────────────
def play_game(team_a, team_b, p_a_wins, stochastic):
    """Return winner (team dict). stochastic=True: sample; False: deterministic."""
    if stochastic:
        return team_a if np.random.random() < p_a_wins else team_b
    return team_a if p_a_wins >= 0.5 else team_b

def simulate_region(teams, model_bundle, stochastic, game_log):
    """Simulate one 16-team region. Returns regional champion team dict."""
    ordered = sorted(teams, key=lambda t: SEED_BRACKET_ORDER.index(t["seed"]))
    current = ordered
    round_size = 64

    for _ in range(4):  # R64 -> R32 -> S16 -> E8
        round_size //= 2
        next_round = []
        for i in range(0, len(current), 2):
            ta, tb = current[i], current[i + 1]
            p = predict_prob(ta["feats"], tb["feats"], model_bundle)
            winner = play_game(ta, tb, p, stochastic)
            loser  = tb if winner is ta else ta
            if game_log is not None:
                game_log.append({
                    "round": ROUND_NAMES[round_size * 2],
                    "team_a": ta["team"], "seed_a": ta["seed"],
                    "team_b": tb["team"], "seed_b": tb["seed"],
                    "p_a_wins": round(p, 4),
                    "winner": winner["team"], "winner_seed": winner["seed"],
                })
            winner["reached"][ROUND_NAMES[round_size]] = True
            next_round.append(winner)
        current = next_round

    return current[0]

def simulate_tournament(bracket, model_bundle, stochastic):
    """Run one tournament simulation. Returns (game_log, round_results dict)."""
    regions = list(bracket.keys())
    game_log = [] if not stochastic else None

    regional_winners = []
    for region in regions:
        for team in bracket[region]:
            team["reached"] = {r: False for r in ROUND_NAMES.values()}
            team["reached"]["R64"] = True
        winner = simulate_region(bracket[region], model_bundle, stochastic, game_log)
        regional_winners.append(winner)

    # Final Four: pair (0,1) and (2,3)
    finalists = []
    for i in range(0, 4, 2):
        ta, tb = regional_winners[i], regional_winners[i + 1]
        p = predict_prob(ta["feats"], tb["feats"], model_bundle)
        w = play_game(ta, tb, p, stochastic)
        l = tb if w is ta else ta
        if game_log is not None:
            game_log.append({
                "round": "F4",
                "team_a": ta["team"], "seed_a": ta["seed"],
                "team_b": tb["team"], "seed_b": tb["seed"],
                "p_a_wins": round(p, 4),
                "winner": w["team"], "winner_seed": w["seed"],
            })
        w["reached"]["F4"]  = True
        w["reached"]["F2"]  = False
        finalists.append(w)

    # Championship
    ta, tb = finalists[0], finalists[1]
    p = predict_prob(ta["feats"], tb["feats"], model_bundle)
    champ = play_game(ta, tb, p, stochastic)
    ru    = tb if champ is ta else ta
    champ["reached"]["F2"]    = True
    champ["reached"]["CHAMP"] = True
    ru["reached"]["F2"]       = True
    if game_log is not None:
        game_log.append({
            "round": "CHAMP",
            "team_a": ta["team"], "seed_a": ta["seed"],
            "team_b": tb["team"], "seed_b": tb["seed"],
            "p_a_wins": round(p, 4),
            "winner": champ["team"], "winner_seed": champ["seed"],
        })
    return game_log, champ

# ── Sample bracket generator ───────────────────────────────────────────────────
def generate_sample_bracket(year=2025):
    """Extract bracket from historical data using Tournament Simulation order."""
    sim = pd.read_csv(RAW / "Tournament Simulation.csv")
    sim = sim[(sim["YEAR"] == year) & (sim["CURRENT ROUND"] == 64)].copy()
    sim = sim.sort_values("BY ROUND NO", ascending=False).reset_index(drop=True)

    region_labels = ["East", "West", "South", "Midwest"]
    sim["REGION"] = [region_labels[i // 16] for i in range(len(sim))]
    sim["FEATURE_YEAR"] = year
    bracket_df = sim[["TEAM NO", "TEAM", "SEED", "REGION", "FEATURE_YEAR"]].rename(
        columns={"TEAM NO": "TEAM_NO", "FEATURE_YEAR": "YEAR"})
    out = OUT_DIR / f"sample_bracket_{year}.csv"
    bracket_df.to_csv(out, index=False)
    print(f"Sample bracket saved -> {out}")
    return bracket_df

# ── Main ──────────────────────────────────────────────────────────────────────
def main(bracket_path=None):
    print("Loading best params and fitting models on full history...")
    with open(MODELS / "best_params.json") as f:
        best_params = json.load(f)

    models = fit_all_models(best_params)
    print("All five models trained.\n")

    # Feature lookup
    print("Building team feature lookup...")
    lookup = build_feature_lookup()

    # Load bracket
    if bracket_path is None:
        bracket_df = generate_sample_bracket(2025)
    else:
        bracket_df = pd.read_csv(bracket_path)
        if "YEAR" not in bracket_df.columns:
            bracket_df["YEAR"] = bracket_df["TEAM_NO"].map(
                lambda tn: max((y for (t, y) in lookup if t == tn), default=2025))

    print(f"Bracket loaded: {len(bracket_df)} teams across "
          f"{bracket_df['REGION'].nunique()} regions\n")

    # Build bracket structure: {region: [team_dict, ...]}
    regions = bracket_df["REGION"].unique().tolist()
    bracket = {}
    for region in regions:
        rteams = bracket_df[bracket_df["REGION"] == region].to_dict("records")
        team_list = []
        for row in rteams:
            tn   = int(row["TEAM_NO"])
            yr   = int(row.get("YEAR", 2025))
            feats = get_team_features(tn, yr, lookup)
            feats["_SEED"] = int(row["SEED"])  # store seed for LR model
            team_list.append({
                "team_no": tn, "team": row["TEAM"],
                "seed": int(row["SEED"]), "region": region,
                "feats": feats, "reached": {},
            })
        bracket[region] = team_list

    # ── Deterministic simulation for each model ───────────────────────────────
    all_team_list = [t for teams in bracket.values() for t in teams]
    print("Running deterministic simulations...")
    for model_name, bundle in models.items():
        # Deep-copy bracket for this model
        import copy
        b = copy.deepcopy(bracket)
        game_log, champ = simulate_tournament(b, bundle, stochastic=False)

        # Save game log
        log_df = pd.DataFrame(game_log)
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path  = OUT_DIR / f"deterministic_{safe_name}.csv"
        log_df.to_csv(out_path, index=False)
        print(f"  {model_name:<22s} -> champion: {champ['team']} "
              f"(seed {champ['seed']})")

    # ── Monte Carlo simulation (MLP) ──────────────────────────────────────────
    N_TRIALS = 10_000
    print(f"\nRunning Monte Carlo simulation ({N_TRIALS:,} trials, MLP)...")

    mlp_bundle = models["MLP"]
    round_cols = ["R64", "R32", "S16", "E8", "F4", "F2", "CHAMP"]
    counts = {t["team_no"]: {r: 0 for r in round_cols}
              for t in all_team_list}

    import copy
    for trial in range(N_TRIALS):
        b = copy.deepcopy(bracket)
        _, _ = simulate_tournament(b, mlp_bundle, stochastic=True)
        for teams in b.values():
            for team in teams:
                for rnd in round_cols:
                    if team["reached"].get(rnd, False):
                        counts[team["team_no"]][rnd] += 1

    # Build output DataFrame
    mc_rows = []
    for t in all_team_list:
        tn   = t["team_no"]
        row  = {"TEAM_NO": tn, "TEAM": t["team"],
                "SEED": t["seed"], "REGION": t["region"]}
        for rnd in round_cols:
            row[rnd] = round(counts[tn][rnd] / N_TRIALS, 4)
        mc_rows.append(row)

    mc_df = pd.DataFrame(mc_rows).sort_values(["SEED", "REGION"])
    mc_path = OUT_DIR / "montecarlo_probs.csv"
    mc_df.to_csv(mc_path, index=False)
    print(f"Monte Carlo results saved -> {mc_path}")

    # Print summary
    print(f"\n{'='*65}")
    print(f"MONTE CARLO CHAMPIONSHIP PROBABILITIES (top 16, {N_TRIALS:,} trials)")
    print(f"{'='*65}")
    top = mc_df.sort_values("CHAMP", ascending=False).head(16)
    print(top[["TEAM", "SEED", "REGION", "F4", "F2", "CHAMP"]].to_string(index=False))

    print(f"\n{'='*65}")
    print("DETERMINISTIC BRACKET PREDICTIONS — CHAMPIONS")
    print(f"{'='*65}")
    for model_name, bundle in models.items():
        b = copy.deepcopy(bracket)
        _, champ = simulate_tournament(b, bundle, stochastic=False)
        print(f"  {model_name:<22s}: {champ['team']} (seed {champ['seed']}, "
              f"region {champ['region']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracket", default=None,
                        help="Path to bracket CSV (default: 2025 sample)")
    args = parser.parse_args()
    main(args.bracket)
