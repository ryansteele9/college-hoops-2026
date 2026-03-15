# college-hoops-2026

End-to-end ML pipeline to predict NCAA tournament game outcomes (2026). Trains and compares six models; primary output is bracket picks and Monte Carlo championship probabilities.

## Environment

- Python venv at `venv/` — activate with `source venv/Scripts/activate` (bash) or `venv\Scripts\Activate.ps1` (PowerShell)
- Run Jupyter: `jupyter lab` from project root

## Key packages

- Data: pandas, numpy, requests, beautifulsoup4
- ML: scikit-learn, xgboost, lightgbm, torch
- Notebooks: jupyterlab

## Project structure

```
data/raw/         # raw scraped/downloaded data
data/processed/   # cleaned, feature-engineered data
models/           # saved model artifacts (*.pkl, best_params.json)
notebooks/        # exploratory analysis and experiments
src/              # production pipeline code
```

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `src/build_master_table.py` | Builds `data/processed/master_team_table.csv` (1147 rows, 114 cols) |
| `src/build_matchup_dataset.py` | Builds `data/processed/matchup_dataset.csv` (2140 rows, 337 cols) |
| `src/feature_selection.py` | Selects features → `data/processed/selected_features.txt` (currently 36) |
| `src/retrain_models.py` | Retrains all 6 models on current feature set; run after any feature changes |
| `src/rolling_cv.py` | Walk-forward CV evaluation |
| `src/cv_dedup_eval.py` | Walk-forward CV with deduplicated scoring (one orientation per game) |
| `src/historical_backtest.py` | ESPN-style bracket backtest across all 10 walk-forward folds |
| `src/ensemble_search.py` | Tests ensemble combinations; saves best as production pkl |
| `src/bracket_simulator.py` | Deterministic + Monte Carlo bracket simulation |
| `src/injury_adjustments.py` | Post-prediction injury adjustment module |
| `run_2026.py` | Main entry point for 2026 predictions |

## Data Sources Used in Pipeline

14 CSV files contribute features to `master_team_table.csv` (joined on YEAR + TEAM NO):

| Source | Master table columns added | Notes |
|--------|---------------------------|-------|
| `KenPom Barttorvik.csv` | KADJ EM/O/D, BADJ EM/O/D, BARTHAG, four factors, tempo, TALENT, AVG HGT, EFF HGT, ELITE SOS, plus 30+ others | Primary stats source; 49 feature cols |
| `Barttorvik Away-Neutral.csv` | NEUTRAL BADJ EM/O/D, NEUTRAL BARTHAG, NEUTRAL EFG%/D | Away/neutral site splits |
| `TeamRankings.csv` | TR RATING, TR LAST/HI/LO, TR CONSISTENCY, TR LUCK, TR NEUTRAL RATING/LAST | Predictive power rating + trajectory |
| `KenPom Preseason.csv` | KADJ EM CHANGE, KADJ EM RANK CHANGE (derived from preseason vs current) | Available 2012–2025; dominant feature group |
| `RPPF Preseason Ratings.csv` | RPPF RATING CHANGE | Preseason-to-current trajectory; 2015–2025 |
| `Tournament Locations.csv` | LOC DISTANCE MI, LOC TIME ZONES CROSSED/VALUE, PROXIMITY ADVANTAGE | First-round travel distance |
| `Resumes.csv` | RES Q1 W, RES Q3 Q4 L, RES R SCORE, RES BID TYPE AUTO | Quad record and résumé score |
| `Teamsheet Ranks.csv` | QUALITY AVG, RESUME AVG, TS Q1A/Q1/Q2 W, TS Q3/Q4 L | Available 2019–2025 only; NaN in earlier folds |
| `Shooting Splits.csv` | THREES SHARE, THREES FG%, THREES FG%D, CLOSE TWOS SHARE, CLOSE TWOS FG%, DUNKS SHARE, DUNKS FG%D | Shot selection style; 2010–2025 |
| `Tournament Matchups.csv` | PROG_GAMES/W/L/WIN%/F4%/CHAMP% (program history) | Expanding window, strictly prior years — no leakage |
| `Tournament Matchups.csv` | SEED_GAMES/W/L/WIN%/CHAMP% (seed history) | Same file, different aggregation |
| `Tournament Matchups.csv` | CONF_GAMES/W/L/WIN%/CHAMP% (conference history) | Same file, different aggregation |
| `TeamRankings.csv` | PEAK DECLINE (TR HI − TR LAST), RECOVERY (TR LAST − TR LO) | Computed momentum features |
| `matchup_dataset.csv` | SEED_MATCHUP_UPSET_RATE, DIFF_SEED_MATCHUP_WINRATE | Computed at matchup time from historical seed pairs |

### Key architecture: leakage prevention

Historical features (PROG_*, SEED_*, CONF_*) use expanding cumsum on Tournament Matchups.csv
with `years < current_year` — no all-time aggregates that would leak future results.
TEAM ID is stable across years; TEAM NO is per-year sequential (different values each season).

### Null coverage by source

| Source | Years available | Null rows in master table |
|--------|----------------|--------------------------|
| Shooting Splits.csv | 2010–2025 | 130 (2008–2009 teams) |
| Teamsheet Ranks.csv | 2019–2025 | 744 |
| KenPom Preseason.csv | 2012–2025 | 264 |
| RPPF Preseason Ratings.csv | 2015–2025 | 467 |
| Tournament Locations.csv | 2008–2025 | 61 (teams with no site data) |

All nulls handled by `SimpleImputer(strategy="median", keep_empty_features=True)` in every
downstream script. `keep_empty_features=True` is required — newer feature groups are entirely
NaN in early CV folds, and sklearn 1.1+ silently drops all-NaN columns otherwise, causing
shape mismatches at predict time.

## Selected Features (36)

Saved to `data/processed/selected_features.txt`. Feature scores in `data/processed/feature_scores.csv`.
Dominant signal: `DIFF_KADJ EM CHANGE` (SHAP 0.393, permutation importance 0.0028).

```
DIFF_KADJ EM CHANGE      DIFF_BADJ EM             DIFF_BADJ D
DIFF_FTR                 DIFF_TALENT              DIFF_THREES FG%D
DIFF_NEUTRAL EFG%D       DIFF_RPPF RATING CHANGE  DIFF_W
DIFF_NEUTRAL BADJ O      DIFF_DUNKS FG%D          DIFF_PEAK DECLINE
DIFF_GAMES               DIFF_OREB%               DIFF_PROG_GAMES
DIFF_RECOVERY            DIFF_TOV%D               DIFF_AST%
DIFF_CLOSE TWOS FG%      DIFF_LOC DISTANCE MI     DIFF_DUNKS SHARE
DIFF_AVG HGT             DIFF_ELITE SOS           DIFF_CLOSE TWOS SHARE
DIFF_NEUTRAL EFG%        DIFF_THREES FG%          DIFF_KADJ EM RANK CHANGE
DIFF_PROG_WIN%           DIFF_DREB%               DIFF_CONF_WIN%
DIFF_2PTRD               DIFF_TR NEUTRAL LAST     DIFF_FT%
DIFF_SEED_MATCHUP_WINRATE  DIFF_RES R SCORE       SEED_MATCHUP_UPSET_RATE
```

Note: `DIFF_SEED_MATCHUP_WINRATE`, `DIFF_RES R SCORE`, and `SEED_MATCHUP_UPSET_RATE` are
force-pinned (not from auto-selection) — they were dropped by correlation pruning but are
bracket-decisive signals that recover ESPN score when included.

`SEED_MATCHUP_UPSET_RATE` and `DIFF_SEED_MATCHUP_WINRATE` are not in `master_team_table.csv`.
They are computed on-the-fly at simulation/backtest time from historical seed pair win rates
and patched into the feature vector before each prediction.

## Model Performance

Walk-forward CV (10 folds, 2015–2025 excl. 2020, deduplicated scoring).
ESPN avg/yr from walk-forward bracket backtest (same folds).

| Model | AUC | ESPN avg/yr | Notes |
|-------|-----|-------------|-------|
| MLP | 0.816 | 1069 | Best in upset-heavy years |
| **Ensemble (RF+XGB+LGB)** | **0.810** | **1055** | **Primary model; most consistent** |
| XGBoost | 0.805 | 1027 | |
| Random Forest | 0.807 | 1008 | |
| LightGBM | 0.807 | 999 | |
| LR (seed baseline) | 0.779 | 870 | Seed differential only |

Max possible ESPN per year: 1920. Ensemble wins 4 of 10 years outright; MLP wins 5.
In chalk/favorite-dominant years the Ensemble leads; in upset-heavy years MLP is stronger.

Models saved as `models/*_calibrated.pkl`. Raw probabilities used — Platt scaling wrappers
exist but hurt log loss (+0.002–0.005) because models are already well-calibrated (ECE 0.07–0.10)
and there are too few samples per fold to reliably fit the sigmoid. Hyperparameters in
`models/best_params.json` (includes `decay_rate` and `calibration` keys that must be stripped
before sklearn constructor calls — use the `model_params()` helper in each script).

Seven ensemble combinations were tested (RF+XGB+LGB, MLP+Ensemble, MLP+RF, MLP+XGB,
MLP+XGB+LGB, MLP+RF+XGB, MLP+RF+LGB). RF+XGB+LGB won on ESPN in all seven comparisons.
`bracket_simulator.py` supports `"ensemble_with_mlp"` model type for future use.

## Monte Carlo

- 100,000 trials by default; completes in ~7 seconds
- Performance: precompute 64×64 win-probability matrix (one batched `predict_proba` call
  per model, ~0.1s), then trials use numpy array lookups only (~6s for 100K trials)
- Primary probability source: Ensemble (RF+XGB+LGB)
- Output: `data/processed/simulator_outputs/montecarlo_probs.csv`

Implementation note: the `batch_predict` attribute on the ensemble predictor object is what
triggers the fast path in `simulate_mc_espn()`. If this attribute is missing (e.g., a custom
predictor), MC falls back to per-game calls (~31 minutes for 100K trials).

## Injury Adjustment System

Post-prediction adjustment module: `src/injury_adjustments.py`

```bash
python run_2026.py --injuries                        # uses data/raw/injuries_2026.csv
python run_2026.py --injuries path/to/other.csv      # custom path
```

**Adjustment formula** (applied to win probabilities, not model weights):
```
base_penalty = 0.06 × (rating / 5.0)
sev_mult:    severe=1.00, moderate=0.75, questionable=0.30
rec_weight:  ≤3 days→1.00, ≤7 days→0.85, >7 days→0.70
adjustment   = base_penalty × sev_mult × rec_weight
p_adjusted   = clip(p − adj_team_a + adj_team_b, 0.05, 0.95)
```

Severity values are lowercased before lookup — any casing in the CSV works.

`injuries_2026.csv` columns: `TEAM, PLAYER, POSITION, RATING, INJURY_TYPE, GAMES_MISSED, SEVERITY`
(`GAMES_MISSED` contains the injury date, e.g. `10-Mar`, used to compute days since injury.)

Adjusted outputs saved with `_adjusted` suffix:
- `deterministic_{model}_adjusted.csv` (6 files)
- `montecarlo_probs_adjusted.csv`

## Selection Sunday Workflow

```bash
source venv/Scripts/activate

# 1. Drop fresh CSVs into data/raw/ (KenPom Barttorvik, Barttorvik Away-Neutral,
#    TeamRankings, KenPom Preseason, RPPF Preseason, Resumes, Tournament Locations,
#    Shooting Splits), then rebuild feature tables
python src/build_master_table.py       # → master_team_table.csv (1147 rows, 114 cols)
python src/build_matchup_dataset.py    # → matchup_dataset.csv (2140 rows, 337 cols)

# 2. Retrain models (only needed if features or data changed)
python src/retrain_models.py

# 3. Baseline predictions
python run_2026.py --bracket data/raw/bracket_2026.csv

# 4. Full output with injury adjustments
python run_2026.py --bracket data/raw/bracket_2026.csv \
    --injuries \
    --bracket-tree \
    --full-table
```

`bracket_2026.csv` required columns: `TEAM_NO, TEAM, SEED, REGION, YEAR`

## Known Data Limitations

- **Shooting Splits**: 2008–2009 teams have NaN for all 7 shooting split features (coverage starts 2010). Median-imputed in training.
- **Teamsheet Ranks**: only 2019–2025. All 744 rows before 2019 are NaN. These columns are entirely absent in early CV folds — `keep_empty_features=True` is the fix.
- **KenPom Preseason**: requires paid subscription (~$20/yr). If 2026 preseason data is unavailable, `KADJ EM CHANGE` and `KADJ EM RANK CHANGE` — the two most important features — will be NaN and median-imputed, meaningfully degrading predictions.
- **RPPF Preseason**: available 2015–2025 only; 467 NaN rows in master table.
- **Tournament Locations**: 61 teams have no location data (no first-round site assigned yet or historical gap). Median-imputed.
- **Seed matchup features**: computed from historical matchup data only. For novel seed matchups with no historical precedent, defaults to 0.5 win rate.
- **2020**: no tournament (COVID). Excluded from all CV folds (`FOLD_YEARS = 2015–2025 excl. 2020`).
- **2008**: first year in dataset; all PROG_* historical features are NaN (no prior tournament data). Median-imputed.
- **Conference remaps**: `P10→P12`, `SInd→Slnd` applied in build scripts for consistency across years.

## Data Files

All CSVs in `data/raw/`. Files actively used by the pipeline are marked with ✓.

| File | Rows | Years | Active |
|------|------|-------|--------|
| KenPom Barttorvik.csv | 1147 | 2008–2025 | ✓ |
| Barttorvik Away-Neutral.csv | 1147 | 2008–2025 | ✓ |
| TeamRankings.csv | 1147 | 2008–2025 | ✓ |
| KenPom Preseason.csv | 884 | 2012–2025 | ✓ |
| RPPF Preseason Ratings.csv | 680 | 2015–2025 | ✓ |
| Tournament Locations.csv | 2140 | 2008–2025 | ✓ |
| Resumes.csv | 1147 | 2008–2025 | ✓ |
| Teamsheet Ranks.csv | 408 | 2019–2025 | ✓ |
| Shooting Splits.csv | 1017 | 2010–2025 | ✓ |
| Tournament Matchups.csv | 2140 | 2008–2025 | ✓ |
| Barttorvik Away.csv | 1147 | 2008–2025 | — |
| Barttorvik Home.csv | 1147 | 2008–2025 | — |
| Barttorvik Neutral.csv | 1147 | 2008–2025 | — |
| TeamRankings Away.csv | 1147 | 2008–2025 | — |
| TeamRankings Home.csv | 1147 | 2008–2025 | — |
| TeamRankings Neutral.csv | 1147 | 2008–2025 | — |
| 538 Ratings.csv | 544 | 2016–2024 | — |
| AP Poll Data.csv | 15372 | 2008–2025 | — |
| Coach Results.csv | 319 | N/A | — |
| Conference Results.csv | 32 | N/A | — |
| Conference Stats Away Neutral.csv | 550 | 2008–2025 | — |
| Conference Stats Away.csv | 550 | 2008–2025 | — |
| Conference Stats Home.csv | 550 | 2008–2025 | — |
| Conference Stats Neutral.csv | 548 | 2008–2025 | — |
| Conference Stats.csv | 550 | 2008–2025 | — |
| EvanMiya.csv | 816 | 2013–2025 | — |
| Heat Check Ratings.csv | 325 | 2013–2025 | — |
| Heat Check Tournament Index.csv | 768 | 2013–2025 | — |
| Public Picks.csv | 64 | 2025 | — |
| RPPF Conference Ratings.csv | 420 | 2012–2025 | — |
| RPPF Ratings.csv | 884 | 2012–2025 | — |
| Seed Results.csv | 16 | N/A | — |
| Team Results.csv | 242 | N/A | — |
| Tournament Simulation.csv | 64 | 2025 | — |
| Upset Count.csv | 17 | 2008–2025 | — |
| Upset Seed Info.csv | 229 | 2008–2025 | — |
| Z Rating Cumulative.csv | 136 | N/A | — |
| Z Rating Teams.csv | 1360 | 2015–2025 | — |
