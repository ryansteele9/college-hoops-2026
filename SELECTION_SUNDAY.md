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

Expected output: `master_team_table.csv written — 1147 rows, 114 cols` (or similar).

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

### Model performance reference

Walk-forward ESPN backtest average (2015–2025, excl. 2020):

| Model | ESPN avg/yr | Notes |
|-------|-------------|-------|
| MLP | 1069 | Best in upset-heavy years |
| **Ensemble (RF+XGB+LGB)** | **1055** | **Primary recommendation — best overall** |
| XGBoost | 1027 | |
| Random Forest | 1008 | |
| LightGBM | 999 | |
| LR (seed baseline) | 870 | Seed-only, no features |

**Which model to use for your bracket:** The Ensemble is the primary recommendation and historically the most consistent performer. In chalk/favorite-dominant tournaments it tends to outperform all other models. In upset-heavy tournaments, MLP has historically been the stronger bracket pick — if you expect a chaotic field, consider weighting MLP's champion and Final Four picks more heavily alongside the Ensemble.

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

The saved models expect exactly 36 features. This error means the feature tables were rebuilt but the models were not retrained to match.

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

---

## Manual Data Collection (Kaggle Fallback)

Use this section if the Kaggle dataset does not yet have 2026 data for one or more files. Each subsection covers one source: where to get it, what to download, and exactly which columns the pipeline reads.

**The goal:** add one 2026 row per tournament team to each CSV in `data/raw/`, matching the column names and `TEAM NO` / `TEAM ID` keys already in those files. Then re-run Steps 3–4.

---

### Sources at a glance

| File in `data/raw/` | Source website | Cost | Features it drives |
|---|---|---|---|
| `KenPom Barttorvik.csv` | barttorvik.com | Free | 18 of 31 features |
| `KenPom Preseason.csv` | kenpom.com | $20/yr | KADJ EM CHANGE, KADJ EM RANK CHANGE |
| `Barttorvik Away-Neutral.csv` | barttorvik.com | Free | NEUTRAL BADJ O, NEUTRAL EFG%D |
| `RPPF Preseason Ratings.csv` | barttorvik.com | Free | RPPF RATING CHANGE |
| `TeamRankings.csv` | teamrankings.com | Free | TR CONSISTENCY, PEAK DECLINE, RECOVERY |
| `Tournament Locations.csv` | NCAA bracket / ESPN | Free | LOC DISTANCE MI |
| `Resumes.csv` | barttorvik.com | Free | RES R SCORE |
| `Tournament Matchups.csv` | Already in repo | — | PROG_W, CONF_WIN%, SEED_MATCHUP_UPSET_RATE |

**Three sources you don't need to touch:**

- `Tournament Matchups.csv` — historical game results through 2025 are already in the repo. The pipeline computes PROG_W, CONF_WIN%, and SEED_MATCHUP_UPSET_RATE from this file using only prior-year data, so no 2026 rows are needed until after the tournament concludes.
- `TeamRankings Neutral.csv` — used to build TR NEUTRAL RATING / TR NEUTRAL LAST, but neither of those made the final 31 features. Update it if you like, but it will not affect predictions.
- `Teamsheet Ranks.csv` — no Teamsheet columns appear in the 36 selected features. Skip it.

---

### Source 1 — Barttorvik (barttorvik.com)

**URL:** https://barttorvik.com

Free, no login required. Covers four files.

#### 1a. KenPom Barttorvik.csv — 18 features

This is the most important file. Go to **https://barttorvik.com** and navigate to the full T-Rank table for the 2026 season. Look for a download or export button (usually labeled "CSV" or a down-arrow icon near the table). Download the file for year=2026.

Exact columns the pipeline reads (all others are ignored or are rank columns that get dropped):

| Column in CSV | Feature it becomes | What it measures |
|---|---|---|
| `YEAR` | (join key) | Season year — set to 2026 |
| `TEAM NO` | (join key) | Numeric team ID — must match existing IDs |
| `TEAM ID` | (join key) | Stable team identifier across years |
| `TEAM` | (spine) | Team display name |
| `SEED` | (spine) | Tournament seed |
| `ROUND` | (spine) | Tournament round reached (set to 64 for new bracket) |
| `CONF` | (spine) | Conference abbreviation |
| `CONF ID` | (spine) | Conference numeric ID |
| `BADJ EM` | DIFF_BADJ EM | Barttorvik adjusted efficiency margin |
| `BADJ D` | DIFF_BADJ D | Barttorvik adjusted defensive efficiency |
| `FTR` | DIFF_FTR | Free throw rate (FTA / FGA) |
| `TALENT` | DIFF_TALENT | Roster talent composite |
| `TOV%D` | DIFF_TOV%D | Defensive turnover rate forced |
| `W` | DIFF_W | Regular-season wins |
| `AST%` | DIFF_AST% | Assist rate |
| `GAMES` | DIFF_GAMES | Games played |
| `FT%` | DIFF_FT% | Free throw percentage |
| `AVG HGT` | DIFF_AVG HGT | Average height (inches) |
| `EFF HGT` | DIFF_EFF HGT | Effective height (weighted by role) |
| `OREB%` | DIFF_OREB% | Offensive rebound rate |
| `ELITE SOS` | DIFF_ELITE SOS | Strength of schedule vs elite opponents |
| `2PT%D` | DIFF_2PT%D | Opponent 2-point field goal percentage allowed |
| `DREB%` | DIFF_DREB% | Defensive rebound rate |
| `BLKED%` | DIFF_BLKED% | Rate at which team's shots are blocked |
| `TOV%` | DIFF_TOV% | Offensive turnover rate |
| `3PT%D` | DIFF_3PT%D | Opponent 3-point field goal percentage allowed |

After downloading, append the 2026 rows to the existing `KenPom Barttorvik.csv`. Make sure `TEAM NO` values for 2026 teams match the IDs already in the file for those programs.

#### 1b. Barttorvik Away-Neutral.csv — 2 features

On barttorvik.com, look for the away/neutral site splits table (sometimes labeled "Neutral" or "Away & Neutral"). Download or copy the 2026 data.

Columns the pipeline reads:

| Column in CSV | Renamed to | Feature |
|---|---|---|
| `YEAR` | (join key) | — |
| `TEAM NO` | (join key) | — |
| `BADJ O` | NEUTRAL BADJ O | DIFF_NEUTRAL BADJ O |
| `EFG%D` | NEUTRAL EFG%D | DIFF_NEUTRAL EFG%D |

Only these four columns matter. The pipeline renames them on load; the CSV itself must keep the original names.

#### 1c. RPPF Preseason Ratings.csv — 1 feature

On barttorvik.com, look for the RPPF (Résumé-based Power Rating) preseason section. This shows each team's RPPF rating at the start of the season vs. the current rating.

Column the pipeline reads:

| Column in CSV | Feature |
|---|---|
| `YEAR` | (join key) |
| `TEAM NO` | (join key) |
| `RPPF RATING CHANGE` | DIFF_RPPF RATING CHANGE |

This is the difference between the team's current RPPF rating and its preseason RPPF rating. A positive number means the team improved from where it was picked.

#### 1d. Resumes.csv — 1 feature

On barttorvik.com, look for the Resume scores table (sometimes under "Résumé" in the navigation).

Column the pipeline reads:

| Column in CSV | Renamed to | Feature |
|---|---|---|
| `YEAR` | (join key) | — |
| `TEAM NO` | (join key) | — |
| `R SCORE` | RES R SCORE | DIFF_RES R SCORE |

R SCORE is Barttorvik's composite résumé score combining quad record, SOS, and WAB.

---

### Source 2 — KenPom (kenpom.com)

**URL:** https://kenpom.com

**Requires a paid subscription (~$20/year).** If you don't have one, the two features from this source (KADJ EM CHANGE and KADJ EM RANK CHANGE) will fall back to NaN and be median-imputed. This is the single biggest signal in the model — worth subscribing if you don't have access.

#### KenPom Preseason.csv — 2 features

Log in to kenpom.com. The main ratings page shows both the current efficiency margin and the preseason predicted ranking. The `KADJ EM CHANGE` and `KADJ EM RANK CHANGE` columns represent how much each team has moved from its preseason projection.

Look for a CSV export button on the main page. Download the file, find the 2026 rows, and add them to `KenPom Preseason.csv`.

Columns the pipeline reads:

| Column in CSV | Feature | What it measures |
|---|---|---|
| `YEAR` | (join key) | — |
| `TEAM NO` | (join key) | — |
| `KADJ EM CHANGE` | DIFF_KADJ EM CHANGE | Current KADJ EM minus preseason predicted KADJ EM — **dominant feature** |
| `KADJ EM RANK CHANGE` | DIFF_KADJ EM RANK CHANGE | Rank improvement since preseason (positive = improved) |

`KADJ EM CHANGE` is the single most important feature in the model. A team that was predicted to be mediocre but finished in the top 10 of efficiency margin has a large positive value; a team that underperformed its preseason billing has a negative value.

---

### Source 3 — TeamRankings (teamrankings.com)

**URL:** https://www.teamrankings.com/ncaa-basketball/

Free, no login required.

#### TeamRankings.csv — 3 features

Navigate to the team ratings section. The pipeline uses the predictive power rating and its season high, low, and most-recent values, plus the consistency rating.

Specific pages to look for:
- **Predictive rating:** https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other (gives TR RATING, HI, LO, LAST)
- **Consistency rating:** https://www.teamrankings.com/ncaa-basketball/ranking/consistency-rating (gives CONSISTENCY TR RATING)

Columns the pipeline reads:

| Column in CSV | Renamed to | Feature | What it measures |
|---|---|---|---|
| `YEAR` | — | (join key) | — |
| `TEAM NO` | — | (join key) | — |
| `HI` | TR HI | (used to compute PEAK DECLINE) | Season-high power rating |
| `LO` | TR LO | (used to compute RECOVERY) | Season-low power rating |
| `LAST` | TR LAST | (used to compute both) | Most recent power rating |
| `CONSISTENCY TR RATING` | TR CONSISTENCY | DIFF_TR CONSISTENCY | How consistent the team's performance has been |

The pipeline then computes two derived features:
- **PEAK DECLINE** = TR HI − TR LAST (how far the team has dropped from its peak)
- **RECOVERY** = TR LAST − TR LO (how much the team has bounced back from its worst stretch)

You do not need to add PEAK DECLINE or RECOVERY to the CSV — the pipeline computes them automatically from HI, LO, and LAST.

---

### Source 4 — Tournament Locations (NCAA / ESPN)

**URL:** NCAA bracket official site or https://www.espn.com/mens-college-basketball/bracket

Free. This file tracks where each team plays their first-round games and how far they have to travel. It only needs to be updated with 2026 first-round host sites.

#### Tournament Locations.csv — 1 feature

When the 2026 bracket is announced, find the first-round game locations (8 cities, 2 games each). Then for each of the 64 teams, calculate the distance from the team's home campus to their assigned first-round site.

Column the pipeline reads (only `CURRENT ROUND = 64` rows are used):

| Column in CSV | Feature | Notes |
|---|---|---|
| `YEAR` | (join key) | 2026 |
| `TEAM NO` | (join key) | — |
| `CURRENT ROUND` | (filter) | Must be 64 for first-round rows |
| `DISTANCE (MI)` | LOC DISTANCE MI | Straight-line miles from campus to game site |

For LOC DISTANCE MI, use any online distance calculator between the team's city and the game city. Straight-line distance (not driving) is what the existing data uses — Google Maps "as the crow flies" works.

The other location columns (TIME ZONES CROSSED, TIME ZONES CROSSED VALUE, etc.) are present in the file but did not make the 36 selected features. You only need DISTANCE (MI) to be accurate.

---

### Finding correct TEAM NO values

Every file must use the same `TEAM NO` for a given team-year. These are assigned by Barttorvik and are stable but incrementally updated each year. To find the right value for a 2026 team:

```bash
source venv/Scripts/activate
python -c "
import pandas as pd
master = pd.read_csv('data/processed/master_team_table.csv')
# Find a team by name
print(master[master['TEAM'].str.contains('Duke', case=False)][['YEAR','TEAM NO','TEAM ID','TEAM']].tail(5))
"
```

The TEAM NO from the most recent available year is almost always correct for 2026. Use TEAM ID (not TEAM NO) when joining across years — TEAM ID is stable across years, TEAM NO increments annually.

---

### After manual data collection

Once you have updated one or more files in `data/raw/`, re-run the pipeline from Step 3:

```bash
source venv/Scripts/activate
python src/build_master_table.py
python src/build_matchup_dataset.py
python run_2026.py --bracket data/raw/bracket_2026.csv --injuries
```

If `build_master_table.py` prints many null counts for a file you just updated, the TEAM NO values likely don't match — recheck those join keys first.
