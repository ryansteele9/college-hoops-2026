"""
NCAA Tournament Bracket Simulator — Production Version

Loads trained model pkl files from models/ (no retraining at runtime).
Supports all six models: LR (seed), RF, XGBoost, LightGBM, MLP, Ensemble.

Monte Carlo uses Ensemble as the primary probability source (best log loss).
Deterministic simulation runs all six models and saves one CSV per model.

Bracket CSV columns:
    TEAM_NO   — integer matching master_team_table "TEAM NO"
    TEAM      — display name
    SEED      — integer 1-16
    REGION    — string (4 distinct regions; F4 pairs regions[0]/[1] and [2]/[3])
    YEAR      — season whose features to use (optional; defaults to most recent)

Outputs (data/processed/simulator_outputs/):
    deterministic_{model}.csv  — game-by-game picks for each of the 6 models
    montecarlo_probs.csv       — per-team round probabilities (Ensemble MC)
"""

import argparse, copy, json, warnings, sys
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer

PROCESSED = Path("data/processed")
MODELS    = Path("models")
RAW       = Path("data/raw")
OUT_DIR   = PROCESSED / "simulator_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SELECTED   = (PROCESSED / "selected_features.txt").read_text().strip().splitlines()
RAW_FEATS  = [f.replace("DIFF_", "") for f in SELECTED]  # without DIFF_ prefix

# Seed matchup feature positions in SELECTED (None if not present)
_SEED_MATCH_WR_IDX = SELECTED.index("DIFF_SEED_MATCHUP_WINRATE") \
                     if "DIFF_SEED_MATCHUP_WINRATE" in SELECTED else None
_SEED_MATCH_UR_IDX = SELECTED.index("SEED_MATCHUP_UPSET_RATE") \
                     if "SEED_MATCHUP_UPSET_RATE" in SELECTED else None


def _build_seed_table() -> dict:
    """Compute {(seed_lo, seed_hi): lo_win_rate} from full historical matchup data."""
    if not (PROCESSED / "matchup_dataset.csv").exists():
        return {}
    _df = pd.read_csv(PROCESSED / "matchup_dataset.csv",
                      usecols=["TEAM_NO_A", "TEAM_NO_B", "SEED_A", "SEED_B", "TEAM_A_WIN"])
    _df = _df[_df["TEAM_NO_A"] < _df["TEAM_NO_B"]]  # one orientation per game
    _lo = _df[["SEED_A", "SEED_B"]].min(axis=1).astype(int)
    _hi = _df[["SEED_A", "SEED_B"]].max(axis=1).astype(int)
    _lo_won = (
        ((_df["SEED_A"] <= _df["SEED_B"]) & (_df["TEAM_A_WIN"] == 1)) |
        ((_df["SEED_A"] >  _df["SEED_B"]) & (_df["TEAM_A_WIN"] == 0))
    )
    _tmp = pd.DataFrame({"lo": _lo, "hi": _hi, "lo_won": _lo_won})
    _stats = _tmp.groupby(["lo", "hi"])["lo_won"].agg(["sum", "count"])
    return (_stats["sum"] / _stats["count"]).to_dict()


_SEED_TABLE = _build_seed_table()


def _seed_match_vals(sa: int, sb: int) -> tuple:
    """Return (diff_wr, upset_rate) for seeds sa vs sb from global _SEED_TABLE."""
    lo, hi = min(sa, sb), max(sa, sb)
    lo_wr   = _SEED_TABLE.get((lo, hi), 0.5)
    diff_wr = (2 * lo_wr - 1) if sa <= sb else -(2 * lo_wr - 1)
    return diff_wr, 1.0 - lo_wr


def _patch_seed_feats(X: np.ndarray, sa: int, sb: int) -> None:
    """Patch seed matchup features in feature vector X[0, :] in-place."""
    if _SEED_MATCH_WR_IDX is None and _SEED_MATCH_UR_IDX is None:
        return
    diff_wr, upset_rate = _seed_match_vals(sa, sb)
    if _SEED_MATCH_WR_IDX is not None:
        X[0, _SEED_MATCH_WR_IDX] = diff_wr
    if _SEED_MATCH_UR_IDX is not None:
        X[0, _SEED_MATCH_UR_IDX] = upset_rate

SEED_BRACKET_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
ROUND_NAMES = {64: "R64", 32: "R32", 16: "S16", 8: "E8", 4: "F4", 2: "F2", 1: "CHAMP"}
ROUND_LIST  = ["R64", "R32", "S16", "E8", "F4", "F2", "CHAMP"]
MODEL_KEYS  = ["LR (seed)", "Random Forest", "XGBoost", "LightGBM", "MLP", "Ensemble"]


# ── MLP architecture (must match training) ────────────────────────────────────
def build_mlp(n_in, hidden, n_layers, dropout):
    layers, in_dim = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                   nn.ReLU(), nn.Dropout(dropout)]
        in_dim = hidden
    layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
    return nn.Sequential(*layers)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models() -> dict:
    """Load all six calibrated model bundles. Returns {name: predictor_fn}."""
    bundle_files = {
        "LR (seed)":     "lr_seed_calibrated.pkl",
        "Random Forest": "random_forest_calibrated.pkl",
        "XGBoost":       "xgboost_calibrated.pkl",
        "LightGBM":      "lightgbm_calibrated.pkl",
        "MLP":           "mlp_calibrated.pkl",
        "Ensemble":      "ensemble_rf_xgb_lgbm_calibrated.pkl",
    }
    predictors = {}
    for name, fname in bundle_files.items():
        bundle = joblib.load(MODELS / fname)
        predictors[name] = make_predictor(bundle)
        print(f"  Loaded: {fname}")
    return predictors


def make_predictor(bundle):
    """
    Return a predict(feat_a, feat_b) -> float function for a given bundle.

    feat_a / feat_b are dicts of {raw_feature_name: value}.
    feat_a["_SEED"] holds the integer seed (set by build_bracket).
    Probabilities are raw (calibration layer bypassed).
    """
    mt = bundle["model_type"]

    if mt == "logistic_regression":
        imp = bundle["imputer"]
        mdl = bundle["model"]
        def pred(fa, fb):
            X = np.array([[fa.get("_SEED", 0) - fb.get("_SEED", 0)]], dtype=np.float32)
            return float(mdl.predict_proba(imp.transform(X))[:, 1][0])

    elif mt in ("random_forest", "xgboost", "lightgbm"):
        imp = bundle["imputer"]
        mdl = bundle["model"]
        def pred(fa, fb):
            X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in RAW_FEATS]],
                         dtype=np.float32)
            _patch_seed_feats(X, int(fa.get("_SEED", 8)), int(fb.get("_SEED", 8)))
            return float(mdl.predict_proba(imp.transform(X))[:, 1][0])

    elif mt == "ensemble":
        imp     = bundle["imputer"]
        weights = bundle["weights"]
        models  = list(bundle["models"].values())
        def pred(fa, fb):
            X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in RAW_FEATS]],
                         dtype=np.float32)
            _patch_seed_feats(X, int(fa.get("_SEED", 8)), int(fb.get("_SEED", 8)))
            X_imp = imp.transform(X)
            return sum(w * float(m.predict_proba(X_imp)[:, 1][0])
                       for m, w in zip(models, weights))

        def batch_pred(X_diffs: np.ndarray) -> np.ndarray:
            """Batch predict. X_diffs shape (n_pairs, n_feats). Returns (n_pairs,).
            Seed matchup features must be pre-patched into X_diffs by the caller."""
            X_imp = imp.transform(X_diffs.astype(np.float32))
            return sum(w * m.predict_proba(X_imp)[:, 1]
                       for m, w in zip(models, weights))

        pred.batch_predict = batch_pred

    elif mt == "mlp":
        imp   = bundle["imputer"]
        sc    = bundle["scaler"]
        np_   = bundle["net_params"]
        net   = build_mlp(bundle["n_features"], np_["hidden_size"],
                          np_["n_layers"], np_["dropout"])
        net.load_state_dict(bundle["net_state_dict"])
        net.eval()
        def pred(fa, fb):
            X = np.array([[fa.get(f, np.nan) - fb.get(f, np.nan) for f in RAW_FEATS]],
                         dtype=np.float32)
            _patch_seed_feats(X, int(fa.get("_SEED", 8)), int(fb.get("_SEED", 8)))
            X = sc.transform(imp.transform(X))
            with torch.no_grad():
                return float(net(torch.tensor(X, dtype=torch.float32)).squeeze())

    else:
        raise ValueError(f"Unknown model_type: {mt}")

    return pred


# ── Feature lookup ────────────────────────────────────────────────────────────
def build_feature_lookup() -> dict:
    """Return {(team_no, year): {raw_feat: value}} from master table."""
    master = pd.read_csv(PROCESSED / "master_team_table.csv")
    # Also pull ADJOE (= KADJ O) which is one of the raw features
    kenpom = pd.read_csv(RAW / "KenPom Barttorvik.csv",
                         usecols=["YEAR", "TEAM NO", "KADJ O"]
                         ).rename(columns={"KADJ O": "ADJOE"})
    merged = master.merge(kenpom, on=["YEAR", "TEAM NO"], how="left")

    lookup = {}
    for _, row in merged.iterrows():
        key = (int(row["TEAM NO"]), int(row["YEAR"]))
        lookup[key] = {f: row.get(f, np.nan) for f in RAW_FEATS}
    return lookup


def get_team_features(team_no: int, year: int, lookup: dict) -> dict:
    """Return feature dict; fall back to most recent available year if needed."""
    if (team_no, year) in lookup:
        return lookup[(team_no, year)]
    years = [y for (tn, y) in lookup if tn == team_no]
    if years:
        return lookup[(team_no, max(years))]
    return {f: np.nan for f in RAW_FEATS}


# ── Bracket construction ──────────────────────────────────────────────────────
def build_bracket(bracket_df: pd.DataFrame, lookup: dict) -> dict:
    """
    Convert bracket DataFrame to {region: [team_dict, ...]} structure.
    Each team_dict: {team_no, team, seed, region, feats, reached}
    """
    bracket = {}
    for region in bracket_df["REGION"].unique():
        rdf = bracket_df[bracket_df["REGION"] == region]
        teams = []
        for _, row in rdf.iterrows():
            tn  = int(row["TEAM_NO"])
            yr  = int(row.get("YEAR", 2026))
            fts = get_team_features(tn, yr, lookup)
            fts["_SEED"] = int(row["SEED"])   # used by LR predictor
            teams.append({
                "team_no": tn,
                "team":    str(row["TEAM"]),
                "seed":    int(row["SEED"]),
                "region":  region,
                "feats":   fts,
                "reached": {r: False for r in ROUND_LIST},
            })
        bracket[region] = teams
    return bracket


def load_bracket_df(path: str, lookup: dict) -> pd.DataFrame:
    """Load bracket CSV and fill in YEAR from lookup if not provided."""
    df = pd.read_csv(path)
    if "YEAR" not in df.columns:
        def best_year(tn):
            ys = [y for (t, y) in lookup if t == int(tn)]
            return max(ys) if ys else 2026
        df["YEAR"] = df["TEAM_NO"].apply(best_year)
    return df


def generate_sample_bracket(year: int = 2025) -> pd.DataFrame:
    """Extract bracket from historical Tournament Simulation data."""
    sim = pd.read_csv(RAW / "Tournament Simulation.csv")
    sim = sim[(sim["YEAR"] == year) & (sim["CURRENT ROUND"] == 64)].copy()
    sim = sim.sort_values("BY ROUND NO", ascending=False).reset_index(drop=True)
    region_labels = ["East", "West", "South", "Midwest"]
    sim["REGION"] = [region_labels[i // 16] for i in range(len(sim))]
    df = sim[["TEAM NO", "TEAM", "SEED", "REGION"]].rename(columns={"TEAM NO": "TEAM_NO"})
    df["YEAR"] = year
    out = OUT_DIR / f"sample_bracket_{year}.csv"
    df.to_csv(out, index=False)
    print(f"Sample bracket saved -> {out}")
    return df


# ── Simulation engine ─────────────────────────────────────────────────────────
def play_game(team_a: dict, team_b: dict, p_a: float, stochastic: bool) -> tuple:
    """Return (winner, loser). stochastic=True samples; False uses argmax."""
    if stochastic:
        a_wins = np.random.random() < p_a
    else:
        a_wins = p_a >= 0.5
    return (team_a, team_b) if a_wins else (team_b, team_a)


def simulate_region(teams: list, predictor, stochastic: bool,
                    game_log: list | None) -> dict:
    """Simulate one 16-team region. Returns regional champion."""
    current = sorted(teams, key=lambda t: SEED_BRACKET_ORDER.index(t["seed"]))
    round_size = 64

    for _ in range(4):   # R64 -> R32 -> S16 -> E8
        round_size //= 2
        next_round = []
        for i in range(0, len(current), 2):
            ta, tb = current[i], current[i + 1]
            p = predictor(ta["feats"], tb["feats"])
            winner, loser = play_game(ta, tb, p, stochastic)
            if game_log is not None:
                game_log.append({
                    "round":       ROUND_NAMES[round_size * 2],
                    "team_a":      ta["team"], "seed_a": ta["seed"],
                    "team_b":      tb["team"], "seed_b": tb["seed"],
                    "p_a_wins":    round(p, 4),
                    "winner":      winner["team"],
                    "winner_seed": winner["seed"],
                })
            winner["reached"][ROUND_NAMES[round_size]] = True
            next_round.append(winner)
        current = next_round

    return current[0]


def simulate_tournament(bracket: dict, predictor, stochastic: bool):
    """
    Run one full tournament.
    Returns (game_log, champion_dict).
    game_log is populated only for deterministic runs (stochastic=False).
    """
    regions = list(bracket.keys())
    game_log = [] if not stochastic else None

    # Regional rounds
    regional_champs = []
    for region in regions:
        for t in bracket[region]:
            t["reached"]["R64"] = True
        champ = simulate_region(bracket[region], predictor, stochastic, game_log)
        regional_champs.append(champ)

    # Final Four: pair (0,1) and (2,3)
    finalists = []
    for i in range(0, 4, 2):
        ta, tb = regional_champs[i], regional_champs[i + 1]
        p = predictor(ta["feats"], tb["feats"])
        winner, loser = play_game(ta, tb, p, stochastic)
        if game_log is not None:
            game_log.append({
                "round": "F4",
                "team_a": ta["team"], "seed_a": ta["seed"],
                "team_b": tb["team"], "seed_b": tb["seed"],
                "p_a_wins": round(p, 4),
                "winner": winner["team"], "winner_seed": winner["seed"],
            })
        winner["reached"]["F4"] = True
        finalists.append(winner)

    # Championship
    ta, tb = finalists[0], finalists[1]
    p = predictor(ta["feats"], tb["feats"])
    champ, runner_up = play_game(ta, tb, p, stochastic)
    champ["reached"]["F2"]    = True
    champ["reached"]["CHAMP"] = True
    runner_up["reached"]["F2"] = True
    if game_log is not None:
        game_log.append({
            "round": "CHAMP",
            "team_a": ta["team"], "seed_a": ta["seed"],
            "team_b": tb["team"], "seed_b": tb["seed"],
            "p_a_wins": round(p, 4),
            "winner": champ["team"], "winner_seed": champ["seed"],
        })

    return game_log, champ


# ── High-level simulation runners ─────────────────────────────────────────────
def run_deterministic(bracket: dict, predictors: dict, verbose: bool = True) -> dict:
    """
    Run deterministic simulation for all models.
    Returns {model_name: (game_log_df, champion_dict)}.
    Saves deterministic_{model}.csv per model.
    """
    det_results = {}
    for name, predictor in predictors.items():
        b = copy.deepcopy(bracket)
        game_log, champ = simulate_tournament(b, predictor, stochastic=False)
        log_df = pd.DataFrame(game_log)
        safe  = name.replace(" ", "_").replace("(", "").replace(")", "")
        path  = OUT_DIR / f"deterministic_{safe}.csv"
        log_df.to_csv(path, index=False)
        det_results[name] = (log_df, champ)
        if verbose:
            print(f"  {name:<22s}: {champ['team']} (seed {champ['seed']}, "
                  f"{champ['region']})")
    return det_results


def run_monte_carlo(bracket: dict, ens_predictor,
                    n_trials: int = 10_000, verbose: bool = True) -> pd.DataFrame:
    """
    Monte Carlo simulation using a precomputed win-probability matrix.

    Performance fix: calling the ensemble predictor (RF+XGB+LGB) 630K times
    (10K trials × 63 games) with single-sample predict_proba takes ~30 minutes.
    Instead, precompute all 64×64 pairwise probabilities once (~4K model calls,
    ~5-10s), then run trials using pure numpy array lookups.
    """
    import time
    t0 = time.time()

    all_teams = [t for teams in bracket.values() for t in teams]
    n = len(all_teams)
    team_no_to_idx = {t["team_no"]: i for i, t in enumerate(all_teams)}

    # ── Precompute 64×64 probability matrix ──────────────────────────────────
    if verbose:
        print(f"  Precomputing {n}×{n} win-probability matrix...", flush=True)

    all_feats = np.array(
        [[t["feats"].get(f, np.nan) for f in RAW_FEATS] for t in all_teams],
        dtype=np.float32,
    )
    prob_matrix = np.zeros((n, n), dtype=np.float32)

    if hasattr(ens_predictor, "batch_predict"):
        # Build all (i,j) i≠j pairs in one shot and call predict_proba once per model
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        X_diffs = all_feats[[p[0] for p in pairs]] - all_feats[[p[1] for p in pairs]]
        # Patch seed matchup features for each pair using team seeds
        if _SEED_MATCH_WR_IDX is not None or _SEED_MATCH_UR_IDX is not None:
            all_seeds = [t["seed"] for t in all_teams]
            for k, (i, j) in enumerate(pairs):
                dw, ur = _seed_match_vals(all_seeds[i], all_seeds[j])
                if _SEED_MATCH_WR_IDX is not None:
                    X_diffs[k, _SEED_MATCH_WR_IDX] = dw
                if _SEED_MATCH_UR_IDX is not None:
                    X_diffs[k, _SEED_MATCH_UR_IDX] = ur
        probs_flat = ens_predictor.batch_predict(X_diffs)
        for k, (i, j) in enumerate(pairs):
            prob_matrix[i, j] = probs_flat[k]
    else:
        # Fallback: one-at-a-time (slow — use ensemble predictor for MC)
        for i, ta in enumerate(all_teams):
            for j, tb in enumerate(all_teams):
                if i != j:
                    prob_matrix[i, j] = ens_predictor(ta["feats"], tb["feats"])

    if verbose:
        print(f"  Matrix built in {time.time()-t0:.1f}s", flush=True)

    # ── Build per-region seeding order (list of 4 × 16 team indices) ─────────
    regions = list(bracket.keys())
    seeded_regions = []
    for region in regions:
        ordered = sorted(bracket[region],
                         key=lambda t: SEED_BRACKET_ORDER.index(t["seed"]))
        seeded_regions.append([team_no_to_idx[t["team_no"]] for t in ordered])

    # ── Run trials with numpy random + array lookups (no model calls) ────────
    rnd_idx = {r: i for i, r in enumerate(ROUND_LIST)}
    counts  = np.zeros((n, len(ROUND_LIST)), dtype=np.int32)
    rng     = np.random.default_rng(42)

    if verbose:
        print(f"  Running {n_trials:,} trials...", flush=True)

    for trial in range(n_trials):
        # All teams participate in R64
        for reg in seeded_regions:
            for idx in reg:
                counts[idx, rnd_idx["R64"]] += 1

        # Regional rounds: 16→8 (R32), 8→4 (S16), 4→2 (E8), 2→1 (F4)
        regional_champs = []
        for reg_teams in seeded_regions:
            current = list(reg_teams)
            for rnd_name in ["R32", "S16", "E8", "F4"]:
                nxt = []
                for k in range(0, len(current), 2):
                    a, b = current[k], current[k + 1]
                    winner = a if rng.random() < prob_matrix[a, b] else b
                    counts[winner, rnd_idx[rnd_name]] += 1
                    nxt.append(winner)
                current = nxt
            regional_champs.append(current[0])

        # Final Four → 2 finalists
        finalists = []
        for k in range(0, 4, 2):
            a, b = regional_champs[k], regional_champs[k + 1]
            winner = a if rng.random() < prob_matrix[a, b] else b
            finalists.append(winner)

        # Both finalists reach F2
        for f in finalists:
            counts[f, rnd_idx["F2"]] += 1

        # Championship → 1 champion
        a, b = finalists[0], finalists[1]
        winner = a if rng.random() < prob_matrix[a, b] else b
        counts[winner, rnd_idx["CHAMP"]] += 1

        if verbose and (trial + 1) % 2000 == 0:
            print(f"    {trial+1:,}/{n_trials:,} trials "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)

    # ── Build output DataFrame ────────────────────────────────────────────────
    rows = []
    for i, t in enumerate(all_teams):
        row = {"TEAM_NO": t["team_no"], "TEAM": t["team"],
               "SEED": t["seed"], "REGION": t["region"]}
        for rnd in ROUND_LIST:
            row[rnd] = round(counts[i, rnd_idx[rnd]] / n_trials, 4)
        rows.append(row)

    mc_df = pd.DataFrame(rows).sort_values(["CHAMP", "F2"], ascending=False)
    mc_path = OUT_DIR / "montecarlo_probs.csv"
    mc_df.to_csv(mc_path, index=False)
    if verbose:
        print(f"  Monte Carlo complete in {time.time()-t0:.1f}s  ->  {mc_path}")
    return mc_df


# ── Console summary ───────────────────────────────────────────────────────────
def print_summary(det_results: dict, mc_df: pd.DataFrame,
                  n_trials: int) -> None:
    """Print readable bracket summary to console."""
    print()
    print("=" * 70)
    print("DETERMINISTIC CHAMPION PICKS — ALL SIX MODELS")
    print("=" * 70)
    for name, (_, champ) in det_results.items():
        marker = "  <-- MC primary" if name == "Ensemble" else ""
        print(f"  {name:<22s}: {champ['team']} "
              f"(seed {champ['seed']}, {champ['region']}){marker}")

    print()
    print(f"{'=' * 70}")
    print(f"MONTE CARLO CHAMPIONSHIP PROBABILITIES  ({n_trials:,} trials, Ensemble)")
    print(f"{'=' * 70}")
    print(f"  {'Team':<22}  {'Seed':>4}  {'Region':<10}  "
          f"{'F4':>6}  {'F2':>6}  {'Champ':>7}")
    print(f"  {'-'*65}")
    for _, row in mc_df.head(16).iterrows():
        print(f"  {row['TEAM']:<22}  {int(row['SEED']):>4}  {row['REGION']:<10}  "
              f"{row['F4']:>6.3f}  {row['F2']:>6.3f}  {row['CHAMP']:>7.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import time
    t_start = time.time()

    parser = argparse.ArgumentParser(
        description="NCAA tournament bracket simulator")
    parser.add_argument("--bracket", default=None,
                        help="Path to bracket CSV (default: 2025 sample)")
    parser.add_argument("--monte-carlo-trials", type=int, default=10_000)
    args = parser.parse_args()

    t0 = time.time()
    print("Loading models from pkl files...")
    predictors = load_models()
    print(f"  Models loaded in {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print("Building team feature lookup...")
    lookup = build_feature_lookup()
    print(f"  Lookup built in {time.time()-t0:.1f}s ({len(lookup)} entries)\n")

    if args.bracket is None:
        bracket_df = generate_sample_bracket(2025)
    else:
        bracket_df = load_bracket_df(args.bracket, lookup)

    print(f"Bracket: {len(bracket_df)} teams, "
          f"{bracket_df['REGION'].nunique()} regions\n")

    bracket = build_bracket(bracket_df, lookup)

    t0 = time.time()
    print("Running deterministic simulations (all 6 models)...")
    det_results = run_deterministic(bracket, predictors, verbose=True)
    print(f"  Deterministic done in {time.time()-t0:.1f}s\n")

    t0 = time.time()
    print(f"Running Monte Carlo ({args.monte_carlo_trials:,} trials, Ensemble)...")
    mc_df = run_monte_carlo(
        bracket,
        predictors["Ensemble"],
        n_trials=args.monte_carlo_trials,
        verbose=True,
    )

    print_summary(det_results, mc_df, args.monte_carlo_trials)
    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
