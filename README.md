# 🏀 college-hoops-2026

An end-to-end machine learning pipeline for predicting the 2026 NCAA Men's Basketball Tournament. Six optimized models produce deterministic bracket predictions and Monte Carlo championship probabilities for all 64 teams.

**Validation (2025 tournament):** Ensemble model achieved **0.805 AUC** on a fully leak-free walk-forward cross-validation framework. Historical ESPN bracket backtest (2015–2025, excl. 2020) shows Ensemble averaging **954 pts/year** and MLP **934 pts/year** vs. a theoretical max of 1,920.

---

## Overview

This project builds a matchup-based prediction system — rather than predicting how far a team will advance, each model predicts the winner of a specific head-to-head game. The full bracket is simulated round-by-round, and 10,000 Monte Carlo trials produce win probabilities for every team at every round.

---

## Models

Six models are trained, tuned, and preserved for bracket simulation:

| Model | Walk-Forward AUC | ESPN pts/yr | ESPN rank |
|-------|-----------------|-------------|-----------|
| LR (seed only) | 0.779 | 870 | 6th |
| Random Forest | 0.804 | 904 | 5th |
| XGBoost | 0.803 | 923 | 4th |
| LightGBM | 0.803 | 924 | 3rd |
| MLP (PyTorch) | 0.801 | 934 | 2nd |
| **Ensemble (RF+XGB+LGB)** | **0.805** | **954** | **1st** |

AUC computed using **deduplicated walk-forward cross-validation** (test years 2015–2025, excl. 2020) with no data leakage. ESPN pts/yr from deterministic bracket simulation averaged over 10 historical years.

---

## Prediction Approach

**Unit of prediction: the individual game, not the team.**

Each training row is a head-to-head matchup between two teams. Features are differentials between the two teams — the model learns relative strength rather than absolute quality:

```
DIFF_BADJ_EM = Team A adjusted efficiency margin - Team B adjusted efficiency margin
DIFF_SEED    = Team A seed - Team B seed
...
```

Every historical game is mirrored (Team A vs B and B vs A) to ensure symmetry. To predict a full bracket, games are predicted round-by-round with winners advancing.

---

## Data

**Source:** Kaggle NCAA Men's Basketball dataset (~38 CSVs, 2008–2025, excluding 2020)

**Primary sources within dataset:** KenPom, Barttorvik, ESPN, 538, College Poll Archive

**Common spine across all CSVs:**

| Column | Description |
|--------|-------------|
| `YEAR` | Tournament year |
| `TEAM_ID` | Consistent team identifier across years |
| `TEAM_NO` | Unique identifier per team per tournament |
| `TEAM` | College name |
| `SEED` | Tournament seed (1–16) |
| `ROUND` | Round eliminated: 68=First Four → 1=Champion |

**Key data decisions:**
- Historical aggregate features (program win rate, conference history) are computed using **expanding windows** — only data from years prior to the test year is used, preventing leakage
- 2020 excluded from all training and evaluation (tournament canceled)
- No static all-time lookup tables used in training — all historical features are dynamically computed per fold

---

## Feature Engineering

**21 features selected** via permutation importance, SHAP analysis, and correlation pruning (from an initial 68 DIFF_ columns):

**Current season metrics (from KenPom/Barttorvik):**
- `DIFF_BADJ_EM` — Barttorvik adjusted efficiency margin *(dominant signal)*
- `DIFF_WAB` — Wins Above Bubble (strength-of-schedule adjusted)
- `DIFF_TALENT` — Roster talent rating
- `DIFF_BARTHAG` — Barttorvik win probability composite
- `DIFF_FTR` — Free throw rate
- `DIFF_AST%` — Assist rate
- `DIFF_TOV%D` — Defensive turnover forced rate
- `DIFF_OREB%` — Offensive rebounding rate
- + 3 additional efficiency metrics

**Historical aggregate features (leak-free expanding window):**
- `DIFF_PROG_GAMES` — Program tournament appearances (years < test year)
- `DIFF_CONF_WIN%` — Conference tournament win rate (years < test year)
- + 5 additional historical metrics

**Seed matchup prior:**
- `DIFF_SEED_MATCHUP_WINRATE` — Historical win rate for Team A's seed vs Team B's seed (e.g. 5-vs-12), computed as a 16×16 lookup using only years < test year. Encoded as `2*WR_A - 1` so standard mirror negation gives the correct flipped perspective.

**Feature selection finding:** Reducing from 68 → 21 features improved Random Forest AUC by +0.006, confirming that correlated KenPom/Barttorvik metrics introduce noise rather than signal.

---

## Pipeline Architecture

```
data/raw/                        # 38 original CSVs — never modified
    ↓
src/build_master_table.py        # Join CSVs → one row per team per year
    ↓                            # Dynamic expanding-window historical features
src/build_matchup_dataset.py     # Convert to head-to-head matchup rows
    ↓                            # Mirror every game, compute DIFF_ features
src/rolling_cv.py                # Walk-forward CV evaluation framework
    ↓
src/tune_hyperparams.py          # Optuna (50 trials per model, tree models)
src/tune_mlp.py                  # Optuna (50 trials, MLP architecture search)
    ↓
src/bracket_simulator.py         # Round-by-round simulation + Monte Carlo
src/historical_backtest.py       # Walk-forward ESPN bracket scoring (2015–2025)
    ↓
run_2026.py                      # Single entry point for predictions
```

---

## Hyperparameter Tuning

All models tuned using **Optuna (50 trials)** optimizing mean walk-forward AUC across 10 folds:

| Model | Key tuned parameters | AUC gain |
|-------|---------------------|----------|
| Random Forest | max_depth=4, max_features=0.5 | +0.002 |
| XGBoost | max_depth=2, lr=0.011, gamma=4.7, reg_alpha=5.45 | +0.020 |
| LightGBM | num_leaves=11, lr=0.010, reg_alpha=3.92 | +0.020 |
| MLP | hidden=16, n_layers=3, dropout=0.33, lr=0.00129 | +0.001 |

XGBoost and LightGBM benefited most from tuning — both required heavy regularization (shallow trees + high L1) for a dataset of this size (~1,760 training rows). MLP required retuning after each feature set change; the 21-feature set favored a deeper (3-layer) architecture over the original single-layer network.

---

## Monte Carlo Simulation

The Ensemble model's win probabilities drive 10,000 bracket simulations:

1. Precompute a 64×64 win-probability matrix for all possible matchups (0.1s)
2. Each trial randomly resolves games using probabilities as weighted coin flips
3. Aggregate results to produce round-reaching probabilities for every team

**Performance:** ~1 second for 10,000 trials via batched probability matrix precomputation.

**Sample output (2025 validation):**

```
Team              Seed  Region     F4      F2    Champ
Duke                 1  South    55.7%   35.8%   25.4%
Houston              1  Midwest  43.0%   27.8%   20.7%
Florida              1  West     38.0%   23.7%   11.0%  ← actual winner
Auburn               1  East     33.0%   20.3%    9.8%
```

---

## Project Structure

```
college-hoops-2026/
├── data/
│   ├── raw/                    # Original CSVs (gitignored)
│   └── processed/
│       ├── master_team_table.csv
│       ├── matchup_dataset.csv
│       ├── selected_features.txt
│       ├── rolling_cv_results.csv
│       ├── backtest_results.csv  # ESPN scores per model per year
│       └── simulator_outputs/  # Bracket predictions per model
├── models/
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── mlp.pt
│   ├── ensemble_rf_xgb_lgbm.pkl
│   ├── best_params.json
│   └── backtest_espn_scores.png  # Bar chart: ESPN score by model & year
├── src/
│   ├── audit_csvs.py
│   ├── build_master_table.py
│   ├── build_matchup_dataset.py
│   ├── rolling_cv.py
│   ├── cv_dedup_eval.py
│   ├── train_models.py
│   ├── tune_hyperparams.py
│   ├── tune_mlp.py
│   ├── bracket_simulator.py
│   └── historical_backtest.py
├── notebooks/
├── run_2026.py                 # ← Entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/college-hoops-2026
cd college-hoops-2026

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Rebuild the data pipeline
```bash
python src/build_master_table.py
python src/build_matchup_dataset.py
```

### Run walk-forward CV evaluation
```bash
python src/rolling_cv.py
python src/cv_dedup_eval.py    # deduplicated metrics
```

### Retrain and tune models
```bash
python src/train_models.py
python src/tune_hyperparams.py   # tree models (RF, XGB, LGB)
python src/tune_mlp.py           # MLP architecture search
```

### Run historical ESPN backtest
```bash
python src/historical_backtest.py
# Outputs: data/processed/backtest_results.csv, models/backtest_espn_scores.png
```

### Run bracket predictions
```bash
# With a bracket CSV
python run_2026.py --bracket data/raw/bracket_2026.csv

# Full output with bracket tree and Monte Carlo table
python run_2026.py --bracket data/raw/bracket_2026.csv --bracket-tree --full-table

# Custom Monte Carlo trials
python run_2026.py --bracket data/raw/bracket_2026.csv --monte-carlo-trials 50000
```

### Bracket CSV format
```
TEAM_NO,TEAM,SEED,REGION
1234,Duke,1,South
5678,Auburn,1,East
...
```
`TEAM_NO` must match the `TEAM NO` column in `master_team_table.csv` for the target year.

---

## Key Findings

**Leakage was a significant issue.** Initial models used all-time historical aggregate features (program win rates, seed line statistics) computed from the full dataset including future years. Fixing this with dynamic expanding-window aggregation dropped AUC by ~0.040 — those gains were entirely artificial. The honest post-fix AUC of 0.794 was then improved legitimately to 0.806 by adding current season features.

**DIFF_BADJ_EM is the dominant feature.** Barttorvik's adjusted efficiency margin has a SHAP score 10x higher than the next feature. A single number summarizing net efficiency relative to the field captures most of the predictable variance in tournament outcomes.

**Historical program pedigree adds real signal beyond efficiency ratings.** Even after controlling for current season metrics, a team's all-time tournament win rate and conference tournament history add measurable predictive value — tournament experience and program culture are real factors.

**Recency weighting is unnecessary.** Optuna found optimal decay rates near 1.0 for all tree models, meaning older seasons are as informative as recent ones. KenPom/Barttorvik efficiency margins are already era-normalized, so 2012 data is genuinely comparable to 2024 data.

**The realistic AUC ceiling for public-data tournament prediction is ~0.84–0.85.** Beyond that, you need private information (injury reports, line movement, scouting intel) that isn't available at bracket time.

**The Ensemble model wins the ESPN bracket scoring contest.** Walk-forward backtest across 10 years (2015–2025, excl. 2020) using ESPN scoring (R64=10 → Championship=320 pts, max 1,920/year): Ensemble averaged **954 pts/year** (best in 3 of 10 years), MLP 934 (best in 2), LightGBM 924 (best in 2), XGBoost 923 (best in 2), Random Forest 904 (best in 1), LR (seed) 870. AUC and ESPN ranking are correlated but not perfectly aligned — the Ensemble's blend of RF+XGB+LGB consistently produces well-calibrated favorites without any single model's noise. The Ensemble Monte Carlo mean was ~650 pts/year — conservative spread costs ESPN points relative to picking definitive winners.

**The 2021 VCU withdrawal is handled.** Oregon received an automatic bye into the Round of 32 after VCU withdrew due to COVID. The simulator detects this case and inserts a synthetic bye game to maintain the full 63-game bracket structure for scoring.

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
lightgbm
torch
optuna
matplotlib
seaborn
jupyter
```

---

## Acknowledgements

Data sourced from the Kaggle NCAA Men's Basketball dataset, which aggregates from:
[KenPom](https://kenpom.com/) · [Barttorvik](https://www.barttorvik.com/) · [ESPN](https://www.espn.com/) · [538](https://abcnews.go.com/538) · [College Poll Archive](https://www.collegepollarchive.com/)
