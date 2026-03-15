# Selection Sunday 2026 — Step-by-Step Guide

Everything you need to run predictions on tournament day. Follow steps 1–6 in order.

---

## Step 1 — Download fresh data from Kaggle

1. Go to the Kaggle dataset and download the latest CSVs.
2. Copy all downloaded files into `data/raw/`, overwriting the existing files.

The files that matter most to update are:
- `KenPom Barttorvik.csv` — primary stats source, must include 2026 season
- `Barttorvik Away-Neutral.csv`
- `TeamRankings.csv`
- `KenPom Preseason.csv`
- `RPPF Preseason Ratings.csv`
- `Tournament Locations.csv`
- `Resumes.csv`
- `Teamsheet Ranks.csv`

You do **not** need to update the historical files (Tournament Matchups, Seed Results, etc.) unless Kaggle has new versions.

---

## Step 2 — Activate the virtual environment

Open a terminal in the project root (`college-hoops-2026/`) and run:

```bash
source venv/Scripts/activate
```

Your prompt should now show `(venv)` at the start. All commands below assume the venv is active.

---

## Step 3 — Rebuild the feature tables

These two scripts read `data/raw/` and produce the processed files the models need.

```bash
python src/build_master_table.py
```

Expected output: `master_team_table.csv written — 1147 rows, 107 cols` (or similar).

```bash
python src/build_matchup_dataset.py
```

Expected output: `matchup_dataset.csv written — 2140 rows` (or similar).

Both scripts print warnings for any teams they cannot match — these are usually fine to ignore unless a 2026 team is listed.

---

## Step 4 — Create the 2026 bracket CSV

Create `data/raw/bracket_2026.csv` with the 64 tournament teams. The file needs these columns:

| Column | Description |
|--------|-------------|
| `TEAM_NO` | Integer — must match the `TEAM NO` value in `master_team_table.csv` |
| `TEAM` | Display name (e.g., `Duke`, `North Carolina`) |
| `SEED` | Integer 1–16 |
| `REGION` | One of four region names (e.g., `East`, `West`, `South`, `Midwest`) |
| `YEAR` | `2026` for all rows |

To find a team's TEAM_NO, search `data/processed/master_team_table.csv` for the team name and use the TEAM NO value from the 2025 row (or 2026 if available).

Example rows:
```
TEAM_NO,TEAM,SEED,REGION,YEAR
1,Duke,1,East,2026
2,North Carolina,4,East,2026
```

---

## Step 5 — Run predictions

```bash
python run_2026.py \
    --bracket data/raw/bracket_2026.csv \
    --bracket-tree \
    --full-table
```

This will:
- Load all 6 trained models
- Run deterministic simulations for every model (picks one champion per model)
- Run 100,000-trial Monte Carlo simulation (~7 seconds)
- Print a game-by-game bracket tree for the Ensemble model
- Print round probabilities for all 64 teams

---

## Step 6 — Run with injury adjustments

First, update `data/raw/injuries_2026.csv` with any players who are injured or questionable for the tournament. The file has these columns:

| Column | Values |
|--------|--------|
| `TEAM` | Abbreviation — must be in the map (see below) |
| `PLAYER` | Name (e.g., `C.Foster`) |
| `POSITION` | G, F, or C |
| `RATING` | 1–5 (5 = star player) |
| `INJURY_TYPE` | Free text description |
| `GAMES_MISSED` | Injury date in `10-Mar` format |
| `SEVERITY` | `Severe`, `Moderate`, or `Questionable` |

Current team abbreviation map (add new entries to `src/injury_adjustments.py` if needed):

| Abbreviation | Full name |
|---|---|
| DUKE | Duke |
| TTU | Texas Tech |
| UNC | North Carolina |
| WISC | Wisconsin |
| LOU | Louisville |
| UCLA | UCLA |
| CLEM | Clemson |
| NOVA | Villanova |

Then run:

```bash
python run_2026.py \
    --bracket data/raw/bracket_2026.csv \
    --bracket-tree \
    --full-table \
    --injuries
```

The console will print:
1. A table of all injury adjustments applied and team penalty totals
2. Adjusted champion picks for all 6 models
3. Adjusted Monte Carlo championship probabilities
4. A delta table showing how much each injured team's championship odds changed

---

## Step 7 — Find the output files

All output files are saved to `data/processed/simulator_outputs/`.

| File | Contents |
|------|---------|
| `deterministic_{model}.csv` | Game-by-game picks, one file per model (6 files) |
| `montecarlo_probs.csv` | Round-by-round probabilities for all 64 teams |
| `deterministic_{model}_adjusted.csv` | Same as above, injury-adjusted (6 files) |
| `montecarlo_probs_adjusted.csv` | Injury-adjusted Monte Carlo probabilities |

---

## Troubleshooting

### "Feature mismatch" or shape error when running predictions

The saved models expect exactly 31 features. This error means the feature tables were rebuilt but the models were not retrained to match.

Fix:
```bash
python src/retrain_models.py
```

This retrains all 6 models on the current feature set and overwrites the pkl files in `models/`. It takes a few minutes.

---

### A 2026 team is missing or shows all-NaN features

This means the team's TEAM_NO in your bracket CSV doesn't match any row in `master_team_table.csv`.

Fix:
1. Open `data/processed/master_team_table.csv` and search for the team name.
2. Use the TEAM NO from the most recent year available.
3. Update `data/raw/bracket_2026.csv` with the correct TEAM_NO.

If the team genuinely has no data (new program, name change, etc.), the pipeline will impute all features with median values and still run — but predictions for that team will be unreliable.

---

### Models not loading ("No such file or directory" in `models/`)

The pkl files are missing from `models/`. This can happen after a fresh clone or if the models directory was cleared.

Fix: retrain from scratch:
```bash
python src/retrain_models.py
```

If that fails because `data/processed/matchup_dataset.csv` is missing, run Steps 3 and then retry.

---

### Injury team abbreviation not recognized

If a team in `injuries_2026.csv` doesn't match any key in `TEAM_ABBREV_MAP`, the full name defaults to the abbreviation itself (e.g., `"KSU"` stays `"KSU"`), and it will silently apply no adjustment because `"KSU"` doesn't match any bracket team name.

Fix: add the missing abbreviation to the map at the top of `src/injury_adjustments.py`:
```python
TEAM_ABBREV_MAP = {
    ...
    "KSU": "Kansas State",
}
```

---

### Monte Carlo takes more than 30 seconds

This usually means the `batch_predict` optimization is not being used (falling back to one game at a time). Check that `models/ensemble_rf_xgb_lgbm_calibrated.pkl` exists and loaded without errors in the console output.

If the ensemble pkl is missing, run `python src/retrain_models.py` to rebuild it.
