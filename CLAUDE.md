# college-hoops-2026

End-to-end ML pipeline to predict NCAA tournament game outcomes (2026). Goal: build and compare multiple prediction models.

## Environment

- Python venv at `venv/` — activate with `source venv/Scripts/activate` (bash) or `venv\Scripts\Activate.ps1` (PowerShell)
- Run Jupyter: `jupyter lab` from project root

## Key packages

- Data: pandas, numpy, requests, beautifulsoup4
- ML: scikit-learn, xgboost
- Notebooks: jupyterlab

## Project structure

```
data/raw/         # raw scraped/downloaded data
data/processed/   # cleaned, feature-engineered data
models/           # saved model artifacts
notebooks/        # exploratory analysis and experiments
src/              # production pipeline code
```

## Workflow

1. Collect data -> `data/raw/`
2. Feature engineering -> `data/processed/`
3. Train models in `notebooks/`, promote best to `src/`
4. Evaluate and compare model performance

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `src/build_master_table.py` | Builds `data/processed/master_team_table.csv` (1147 rows, 107 cols) |
| `src/build_matchup_dataset.py` | Builds `data/processed/matchup_dataset.csv` (2140 rows, 314 cols) |
| `src/feature_selection.py` | Selects features → `data/processed/selected_features.txt` |
| `src/retrain_models.py` | Retrains all 6 models on current feature set; run after feature changes |
| `src/rolling_cv.py` | Walk-forward CV evaluation |
| `src/cv_dedup_eval.py` | Walk-forward CV comparing mirrored vs deduplicated scoring |
| `src/historical_backtest.py` | ESPN-style backtest across all 10 folds |
| `src/bracket_simulator.py` | Deterministic + Monte Carlo bracket simulation |
| `src/injury_adjustments.py` | Post-prediction injury adjustment module |
| `run_2026.py` | Main entry point for 2026 predictions |

## Data Sources Used in Pipeline

13 CSV files contribute features to `master_team_table.csv`:

| Source | Key Features |
|--------|-------------|
| `KenPom Barttorvik.csv` | KADJ EM/O/D, BADJ EM/O/D, BARTHAG, four factors, tempo, TALENT, AVG HGT, EFF HGT |
| `Barttorvik Away-Neutral.csv` | NEUTRAL BADJ EM/O/D, NEUTRAL BARTHAG, NEUTRAL EFG%/D |
| `TeamRankings.csv` | TR RATING, TR LAST/HI/LO, TR CONSISTENCY, TR LUCK, TR NEUTRAL RATING/LAST |
| `KenPom Preseason.csv` | PRESEASON KADJ EM → KADJ EM CHANGE, KADJ EM RANK CHANGE |
| `RPPF Preseason Ratings.csv` | RPPF RATING CHANGE (preseason-to-current trajectory) |
| `Tournament Locations.csv` | LOC DISTANCE MI, LOC TIME ZONES CROSSED/VALUE, PROXIMITY ADVANTAGE |
| `Resumes.csv` | RES Q1 W, RES Q3 Q4 L, RES R SCORE, RES BID TYPE AUTO |
| `Teamsheet Ranks.csv` | QUALITY AVG, RESUME AVG, TS Q1A/Q1/Q2 W, TS Q3/Q4 L (2019–2025) |
| `Tournament Matchups.csv` | Historical PROG_*, SEED_*, CONF_* (expanding window, no leakage) |
| `KenPom Barttorvik.csv` | Computed: PEAK DECLINE (TR HI − TR LAST), RECOVERY (TR LAST − TR LO) |

## Selected Features (31)

Saved to `data/processed/selected_features.txt`. Dominant signal: `DIFF_KADJ EM CHANGE`.

```
DIFF_KADJ EM CHANGE    DIFF_BADJ EM           DIFF_FTR
DIFF_NEUTRAL BADJ O    DIFF_RPPF RATING CHANGE DIFF_NEUTRAL EFG%D
DIFF_TALENT            DIFF_TOV%D             DIFF_W
DIFF_PEAK DECLINE      DIFF_AST%              DIFF_BADJ D
DIFF_GAMES             DIFF_FT%               DIFF_KADJ EM RANK CHANGE
DIFF_AVG HGT           DIFF_LOC DISTANCE MI   DIFF_RECOVERY
DIFF_RES R SCORE       DIFF_PROG_W            DIFF_EFF HGT
DIFF_OREB%             DIFF_ELITE SOS         DIFF_CONF_WIN%
DIFF_2PT%D             DIFF_DREB%             SEED_MATCHUP_UPSET_RATE
DIFF_BLKED%            DIFF_TR CONSISTENCY    DIFF_TOV%
DIFF_3PT%D
```

## Model Performance

Walk-forward CV (10 folds, 2015–2025 excl. 2020). ESPN avg/yr from historical backtest.

| Model | AUC | ESPN avg/yr |
|-------|-----|------------|
| MLP | 0.809 | 940 |
| Ensemble (RF+XGB+LGB) | 0.808 | **1052** |
| Random Forest | 0.807 | 1050 |
| LightGBM | 0.807 | 1045 |
| XGBoost | 0.806 | 1035 |
| LR (seed baseline) | 0.779 | 870 |

Ensemble is the primary model for Monte Carlo. Models saved as `models/*_calibrated.pkl`
(raw probabilities used — calibration wrappers exist but slightly hurt log loss).
Hyperparameters in `models/best_params.json`.

## Monte Carlo

- 100,000 trials by default; completes in ~7 seconds
- Performance: precompute 64×64 win-probability matrix (one batched `predict_proba` call
  per model, ~0.1s), then trials use numpy array lookups only (~6s for 100K trials)
- Output: `data/processed/simulator_outputs/montecarlo_probs.csv`

## Injury Adjustment System

Post-prediction adjustment module: `src/injury_adjustments.py`

```bash
python run_2026.py --injuries                          # uses data/raw/injuries_2026.csv
python run_2026.py --injuries path/to/other.csv        # custom path
```

**Adjustment formula** (applied to win probabilities, not model weights):
```
base_penalty = 0.06 × (rating / 5.0)
sev_mult:    severe=1.00, moderate=0.75, questionable=0.30
rec_weight:  ≤3 days→1.00, ≤7 days→0.85, >7 days→0.70
adjustment   = base_penalty × sev_mult × rec_weight
p_adjusted   = clip(p − adj_team_a + adj_team_b, 0.05, 0.95)
```

`injuries_2026.csv` columns: `TEAM, PLAYER, POSITION, RATING, INJURY_TYPE, GAMES_MISSED, SEVERITY`
(GAMES_MISSED column contains the injury date, e.g. `10-Mar`)

Adjusted outputs saved with `_adjusted` suffix:
- `deterministic_{model}_adjusted.csv` (6 files)
- `montecarlo_probs_adjusted.csv`

## Selection Sunday Workflow

```bash
source venv/Scripts/activate

# 1. Rebuild features with fresh KenPom data
python src/build_master_table.py
python src/build_matchup_dataset.py

# 2. Retrain models (if features changed)
python src/retrain_models.py

# 3. Baseline predictions
python run_2026.py --bracket data/raw/bracket_2026.csv

# 4. With injury adjustments + full output
python run_2026.py --bracket data/raw/bracket_2026.csv \
    --injuries \
    --bracket-tree \
    --full-table
```

## Data Files

| File | Rows | Years | Extra Columns | Unique Columns |
|------|------|-------|---------------|----------------|
| 538 Ratings.csv | 544 | 2016–2024 | POWER RATING, POWER RATING RANK | POWER RATING, POWER RATING RANK |
| AP Poll Data.csv | 15372 | 2008–2025 | WEEK, W, L, AP VOTES (+2 more) | WEEK, AP VOTES, AP RANK, RANK? |
| Barttorvik Away-Neutral.csv | 1147 | 2008–2025 | BADJ EM, BADJ O, BADJ D, BARTHAG (+75 more) |  |
| Barttorvik Away.csv | 1147 | 2008–2025 | BADJ EM, BADJ O, BADJ D, BARTHAG (+75 more) |  |
| Barttorvik Home.csv | 1147 | 2008–2025 | BADJ EM, BADJ O, BADJ D, BARTHAG (+75 more) |  |
| Barttorvik Neutral.csv | 1147 | 2008–2025 | BADJ EM, BADJ O, BADJ D, BARTHAG (+75 more) |  |
| Coach Results.csv | 319 | N/A | COACH ID, COACH, PAKE, PAKE RANK (+16 more) | COACH ID, COACH |
| Conference Results.csv | 32 | N/A | CONF ID, CONF, PAKE, PAKE RANK (+15 more) |  |
| Conference Stats Away Neutral.csv | 550 | 2008–2025 | CONF ID, CONF, BADJ EM, BADJ O (+40 more) |  |
| Conference Stats Away.csv | 550 | 2008–2025 | CONF ID, CONF, BADJ EM, BADJ O (+40 more) |  |
| Conference Stats Home.csv | 550 | 2008–2025 | CONF ID, CONF, BADJ EM, BADJ O (+40 more) |  |
| Conference Stats Neutral.csv | 548 | 2008–2025 | CONF ID, CONF, BADJ EM, BADJ O (+40 more) |  |
| Conference Stats.csv | 550 | 2008–2025 | CONF ID, CONF, BADJ EM, BADJ O (+40 more) |  |
| EvanMiya.csv | 816 | 2013–2025 | O RATE, D RATE, RELATIVE RATING, OPPONENT ADJUST (+17 more) | O RATE, D RATE, RELATIVE RATING, OPPONENT ADJUST (+17 more) |
| Heat Check Ratings.csv | 325 | 2013–2025 | EASY DRAW, TOUGH DRAW, DARK HORSE, UPSET ALERT (+1 more) | EASY DRAW, TOUGH DRAW, DARK HORSE, UPSET ALERT (+1 more) |
| Heat Check Tournament Index.csv | 768 | 2013–2025 | POWER, PATH, DRAW, WINS (+5 more) | POWER, PATH, DRAW, WINS (+5 more) |
| KenPom Barttorvik.csv | 1147 | 2008–2025 | CONF, CONF ID, QUAD NO, QUAD ID (+93 more) | QUAD NO, QUAD ID, K TEMPO, K TEMPO RANK (+12 more) |
| KenPom Preseason.csv | 884 | 2012–2025 | PRESEASON KADJ EM RANK, PRESEASON KADJ EM, PRESEASON KADJ O, PRESEASON KADJ O RANK (+7 more) | PRESEASON KADJ EM RANK, PRESEASON KADJ EM, PRESEASON KADJ O, PRESEASON KADJ O RANK (+7 more) |
| Public Picks.csv | 64 | 2025–2025 | R64, R32, S16, E8 (+2 more) | FINALS |
| RPPF Conference Ratings.csv | 420 | 2012–2025 | CONF ID, CONF, CONF RANK, RPPF RATING (+1 more) | CONF RANK |
| RPPF Preseason Ratings.csv | 680 | 2015–2025 | RPPF PRESEASON RANK, PRESEASON RPPF RATING, RPPF RATING CHANGE RANK, RPPF RATING CHANGE (+1 more) | RPPF PRESEASON RANK, PRESEASON RPPF RATING, RPPF RATING CHANGE RANK, RPPF RATING CHANGE (+1 more) |
| RPPF Ratings.csv | 884 | 2012–2025 | RPPF RATING RANK, RPPF RATING, NPB RATING RANK, NPB RATING (+15 more) | RPPF RATING RANK, NPB RATING RANK, RADJ O RANK, RADJ O (+13 more) |
| Resumes.csv | 1147 | 2008–2025 | NET RPI, RESUME, WAB RANK, ELO (+8 more) | NET RPI, RESUME, ELO, B POWER (+5 more) |
| Seed Results.csv | 16 | N/A | PAKE, PAKE RANK, PASE, PASE RANK (+13 more) |  |
| Shooting Splits.csv | 1017 | 2010–2025 | CONF, DUNKS FG%, DUNKS SHARE, DUNKS FG%D (+29 more) | DUNKS FG%, DUNKS SHARE, DUNKS FG%D, DUNKS D SHARE (+28 more) |
| Team Results.csv | 242 | N/A | PAKE, PAKE RANK, PASE, PASE RANK (+14 more) |  |
| TeamRankings Away.csv | 1147 | 2008–2025 | TR RANK, TR RATING, V 1-25 WINS, V 1-25 LOSS (+7 more) |  |
| TeamRankings Home.csv | 1147 | 2008–2025 | TR RANK, TR RATING, V 1-25 WINS, V 1-25 LOSS (+7 more) |  |
| TeamRankings Neutral.csv | 1147 | 2008–2025 | TR RANK, TR RATING, V 1-25 WINS, V 1-25 LOSS (+7 more) |  |
| TeamRankings.csv | 1147 | 2008–2025 | TR RANK, TR RATING, V 1-25 WINS, V 1-25 LOSS (+34 more) | SOS RANK, SOS RATING, SOS HI, SOS LO (+23 more) |
| Teamsheet Ranks.csv | 408 | 2019–2025 | NET, KPI, SOR, WAB RANK (+20 more) | NET, KPI, SOR, RESUME AVG (+17 more) |
| Tournament Locations.csv | 2140 | 2008–2025 | BY YEAR NO, CURRENT ROUND, COLLEGE CITY, COLLEGE STATE (+22 more) | COLLEGE CITY, COLLEGE STATE, COLLEGE LOCATION, COLLEGE TIME ZONE (+20 more) |
| Tournament Matchups.csv | 2140 | 2008–2025 | BY YEAR NO, CURRENT ROUND, SCORE | SCORE |
| Tournament Simulation.csv | 64 | 2025–2025 | BY YEAR NO, BY ROUND NO, CURRENT ROUND | BY ROUND NO |
| Upset Count.csv | 17 | 2008–2025 | FIRST ROUND, SECOND ROUND, SWEET 16, ELITE 8 (+2 more) | TOTAL |
| Upset Seed Info.csv | 229 | 2008–2025 | CURRENT ROUND, SEED WON, SEED LOST, SEED DIFF | SEED WON, SEED LOST, SEED DIFF |
| Z Rating Cumulative.csv | 136 | N/A | Z RATING RANK, CHAMPION, RUNNER UP, FINAL 4 (+5 more) | CHAMPION, RUNNER UP |
| Z Rating Teams.csv | 1360 | 2015–2025 | Z RATING RANK, SEED LIST, Z RATING, TYPE | SEED LIST, Z RATING |
