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
