"""
run_2026.py — 2026 NCAA Tournament Prediction Entry Point

Usage:
    python run_2026.py --bracket data/raw/bracket_2026.csv
    python run_2026.py --bracket data/raw/bracket_2026.csv --monte-carlo-trials 10000
    python run_2026.py  # uses 2025 sample bracket for validation

Bracket CSV format:
    TEAM_NO   — integer matching master_team_table "TEAM NO"
    TEAM      — display name
    SEED      — integer 1-16
    REGION    — one of four region names (e.g., East, West, South, Midwest)
    YEAR      — (optional) season whose KenPom/Barttorvik features to use;
                omit to auto-detect from master table (defaults to most recent)

Prerequisites:
    1. Activate venv:  source venv/Scripts/activate
    2. If 2026 data is available, update data/raw/KenPom Barttorvik.csv and
       run: python src/build_master_table.py && python src/build_matchup_dataset.py
    3. Then:  python run_2026.py --bracket <bracket_csv>

Outputs:
    data/processed/simulator_outputs/deterministic_{model}.csv  — 6 files
    data/processed/simulator_outputs/montecarlo_probs.csv
"""

import argparse
import copy
import sys
from pathlib import Path

# Allow importing from src/ without installing as package
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from bracket_simulator import (
    build_bracket,
    build_feature_lookup,
    generate_sample_bracket,
    load_bracket_df,
    load_models,
    print_summary,
    run_deterministic,
    run_monte_carlo,
    OUT_DIR,
)


# ── Enhanced console output ────────────────────────────────────────────────────
def print_bracket_tree(det_results: dict, bracket: dict) -> None:
    """Print a readable region-by-region bracket view for the Ensemble model."""
    if "Ensemble" not in det_results:
        return

    log_df = det_results["Ensemble"][0]   # game log DataFrame

    print()
    print("=" * 70)
    print("ENSEMBLE DETERMINISTIC BRACKET — GAME-BY-GAME")
    print("=" * 70)

    for rnd in ["R64", "R32", "S16", "E8"]:
        rnd_games = log_df[log_df["round"] == rnd]
        if rnd_games.empty:
            continue
        rnd_labels = {"R64": "Round of 64", "R32": "Round of 32",
                      "S16": "Sweet 16",    "E8":  "Elite 8"}
        print(f"\n  {rnd_labels.get(rnd, rnd)}")
        print(f"  {'Team A':<22} vs {'Team B':<22}  {'Prob':>5}  Winner")
        print(f"  {'-'*68}")
        for _, g in rnd_games.iterrows():
            marker = " *" if g["winner"] == g["team_a"] else ""
            prob_str = f"{g['p_a_wins']:.3f}"
            print(f"  ({g['seed_a']:>2}) {g['team_a']:<18} vs "
                  f"({g['seed_b']:>2}) {g['team_b']:<18}  {prob_str}  "
                  f"({g['winner_seed']}) {g['winner']}")

    # Final Four and Championship
    for rnd in ["F4", "CHAMP"]:
        rnd_games = log_df[log_df["round"] == rnd]
        if rnd_games.empty:
            continue
        label = "Final Four" if rnd == "F4" else "Championship"
        print(f"\n  {label}")
        print(f"  {'-'*68}")
        for _, g in rnd_games.iterrows():
            print(f"  ({g['seed_a']:>2}) {g['team_a']:<22} vs "
                  f"({g['seed_b']:>2}) {g['team_b']:<22} -> "
                  f"({g['winner_seed']}) {g['winner']}")


def print_mc_detail(mc_df: pd.DataFrame, n_trials: int) -> None:
    """Print full Monte Carlo table for all 64 teams."""
    print()
    print("=" * 70)
    print(f"MONTE CARLO ROUND PROBABILITIES — ALL TEAMS ({n_trials:,} trials)")
    print("=" * 70)
    print(f"  {'Team':<22}  {'Seed':>4}  {'Region':<10}  "
          f"{'R32':>6}  {'S16':>6}  {'E8':>6}  {'F4':>6}  {'F2':>6}  {'Champ':>7}")
    print(f"  {'-'*75}")

    # Group by region for readability
    for region in mc_df["REGION"].unique():
        region_df = mc_df[mc_df["REGION"] == region].sort_values("SEED")
        print(f"\n  -- {region} --")
        for _, row in region_df.iterrows():
            print(f"  {row['TEAM']:<22}  {int(row['SEED']):>4}  {row['REGION']:<10}  "
                  f"{row['R32']:>6.3f}  {row['S16']:>6.3f}  {row['E8']:>6.3f}  "
                  f"{row['F4']:>6.3f}  {row['F2']:>6.3f}  {row['CHAMP']:>7.3f}")


def print_header(bracket_df: pd.DataFrame, year: str) -> None:
    print()
    print("=" * 70)
    print(f"  NCAA TOURNAMENT {year} — ML BRACKET PREDICTIONS")
    print(f"  6 Models: LR (seed), RF, XGBoost, LightGBM, MLP, Ensemble")
    print(f"  Teams: {len(bracket_df)}  |  "
          f"Regions: {', '.join(bracket_df['REGION'].unique())}")
    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="2026 NCAA Tournament ML Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bracket", default=None,
        help="Path to bracket CSV. Omit to use 2025 sample for validation.",
    )
    parser.add_argument(
        "--monte-carlo-trials", type=int, default=10_000,
        metavar="N",
        help="Number of Monte Carlo trials for round probabilities (default: 10000)",
    )
    parser.add_argument(
        "--full-table", action="store_true",
        help="Print full 64-team Monte Carlo table (default: top-16 only)",
    )
    parser.add_argument(
        "--bracket-tree", action="store_true",
        help="Print game-by-game bracket tree for the Ensemble model",
    )
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models...")
    predictors = load_models()
    print()

    # ── Build feature lookup ─────────────────────────────────────────────────
    print("Building team feature lookup from master table...")
    lookup = build_feature_lookup()
    print()

    # ── Load bracket ─────────────────────────────────────────────────────────
    if args.bracket is None:
        print("No bracket specified — using 2025 sample bracket for validation.")
        bracket_df = generate_sample_bracket(2025)
        year_label = "2025 (validation)"
    else:
        bracket_df = load_bracket_df(args.bracket, lookup)
        year_label = Path(args.bracket).stem

    print_header(bracket_df, year_label)

    # Warn if features may be stale
    if "YEAR" in bracket_df.columns:
        feature_year = bracket_df["YEAR"].max()
        from bracket_simulator import PROCESSED
        master = pd.read_csv(PROCESSED / "master_team_table.csv")
        latest_master = master["YEAR"].max()
        if feature_year > latest_master:
            print(f"\n  WARNING: Bracket requests YEAR={feature_year} features but "
                  f"master table only has data through {latest_master}.")
            print(f"  Run src/build_master_table.py after updating KenPom data.\n")

    bracket = build_bracket(bracket_df, lookup)

    # ── Deterministic simulations ─────────────────────────────────────────────
    print("\nRunning deterministic simulations (all 6 models)...")
    det_results = run_deterministic(bracket, predictors, verbose=True)

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f"\nRunning Monte Carlo ({args.monte_carlo_trials:,} trials, Ensemble)...")
    mc_df = run_monte_carlo(
        bracket,
        predictors["Ensemble"],
        n_trials=args.monte_carlo_trials,
        verbose=True,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(det_results, mc_df, args.monte_carlo_trials)

    if args.bracket_tree:
        print_bracket_tree(det_results, bracket)

    if args.full_table:
        print_mc_detail(mc_df, args.monte_carlo_trials)

    print()
    print("=" * 70)
    print("Output files:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
