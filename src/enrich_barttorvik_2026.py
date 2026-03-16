"""
Enrich barttorvik_2026_mapped.csv with additional columns from three sources:
  1. teampbp.php      → shooting splits (DUNKS/CLOSE TWOS/FARTHER TWOS/THREES FG%/SHARE)
  2. teamstats.php    → FT%, OP FT%
  3. team.php × 64   → AST%, OP AST%, BLK%, BLKED%
  4. Tournament Matchups.csv (local) → PROG_GAMES, PROG_WIN%, CONF_WIN%

Also reports on neutral site splits (expected empty this early in the cycle).
"""

import requests
import time
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(".")
MAPPED_CSV  = BASE / "data/raw/barttorvik_2026_mapped.csv"
RAW_CSV     = BASE / "data/raw/barttorvik_2026_raw.csv"
MATCHUPS    = BASE / "data/raw/Tournament Matchups.csv"
KENPOM_CSV  = BASE / "data/raw/KenPom Barttorvik.csv"
OUT_CSV     = BASE / "data/raw/barttorvik_2026_mapped.csv"   # overwrite in place

BARTO_BASE  = "https://barttorvik.com"

# ── helpers ───────────────────────────────────────────────────────────────────
def make_session() -> requests.Session:
    """Return a session with the Barttorvik JS-gate cookie."""
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    s = requests.Session()
    seed_url = f"{BARTO_BASE}/trank.php?year=2026&conlimit=All"
    s.get(seed_url, headers=hdrs, timeout=30)
    s.post(seed_url, data={"js_test_submitted": "1"}, headers=hdrs, timeout=30)
    print(f"  Session cookies: {dict(s.cookies)}")
    return s, hdrs


def fetch(s, hdrs, url: str) -> str:
    r = s.get(url, headers=hdrs, timeout=60)
    r.raise_for_status()
    return r.text


def parse_val_rank(td):
    """Extract (float_value, int_rank) from a td with a span.lowrow rank."""
    span = td.find("span", class_="lowrow")
    rank_str = span.get_text(strip=True) if span else ""
    if span:
        span.decompose()
    val_str = td.get_text(strip=True).lstrip("+")
    try:
        val = float(val_str)
    except ValueError:
        val = np.nan
    try:
        rank = int(rank_str)
    except ValueError:
        rank = np.nan
    return val, rank


def barto_team_url_name(raw_name: str) -> str:
    """Convert a Barttorvik team name to its URL-encoded form."""
    return quote_plus(raw_name)


# ── 1. teampbp.php — shooting splits ─────────────────────────────────────────
# Cell class → (column_name, skip_if_mobileonly)
# mobileonly = duplicate of the mobileout cell; skip it.
PBP_CLASS_MAP = {
    "14": "DUNKS FG%",
    "15": "DUNKS SHARE",
    "16": "DUNKS FG%D",    # use mobileout, skip mobileonly
    "17": "DUNKS D SHARE",
    "1":  "CLOSE TWOS FG%",
    "2":  "CLOSE TWOS SHARE",
    "3":  "CLOSE TWOS FG%D",  # use mobileout, skip mobileonly
    "4":  "CLOSE TWOS D SHARE",
    "5":  "FARTHER TWOS FG%",
    "6":  "FARTHER TWOS SHARE",
    "7":  "FARTHER TWOS FG%D",  # use mobileout, skip mobileonly
    "8":  "FARTHER TWOS D SHARE",
    "9":  "THREES FG%",
    "10": "THREES SHARE",
    "11": "THREES FG%D",    # use mobileout, skip mobileonly
    "12": "THREES D SHARE",
}

def scrape_teampbp(s, hdrs) -> pd.DataFrame:
    """Returns DataFrame with columns BARTO_TEAM + shooting split columns."""
    print("  Fetching teampbp.php...")
    html = fetch(s, hdrs, f"{BARTO_BASE}/teampbp.php?year=2026&sort=1")
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    records = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 10:
            continue
        # Team name is in a td with class "teamname" OR a td with an <a> to team.php
        team_cell = row.find("td", class_="teamname")
        if team_cell is None:
            # Try finding a cell with a team.php link
            for c in cells:
                if c.find("a") and "team.php" in str(c.find("a").get("href", "")):
                    team_cell = c
                    break
        if team_cell is None:
            continue

        # Clean team name
        a = team_cell.find("a")
        if not a:
            continue
        for sp in a.find_all("span"):
            sp.decompose()
        team = re.sub(r'\s*(vs\.|def\.|lost|won).*$', '', a.get_text(strip=True),
                      flags=re.IGNORECASE).strip()

        record = {"BARTO_TEAM": team}

        for cell in cells:
            classes = cell.get("class") or []
            class_str = " ".join(classes)
            # Skip mobileonly duplicates
            if "mobileonly" in classes:
                continue
            # Find the leading class number
            for cls in classes:
                if cls in PBP_CLASS_MAP:
                    try:
                        record[PBP_CLASS_MAP[cls]] = float(cell.get_text(strip=True))
                    except ValueError:
                        record[PBP_CLASS_MAP[cls]] = np.nan
                    break

        if len(record) > 1:  # has at least one stat
            records.append(record)

    df = pd.DataFrame(records)
    print(f"    Parsed {len(df)} teams from teampbp.php")
    return df


# ── 2. teamstats.php — FT%, OP FT% ───────────────────────────────────────────
# Multi-row header:  Row0=group names, Row1=Off./Def. sub-labels
# Data rows start at row2. Cell positions (0-indexed after Rk/Team/Conf):
#   3=AdjO, 4=AdjD, 5=EFG%, 6=EFG%D, 7=TOV%, 8=TOV%D,
#   9=OREB%, 10=DREB?, 11=FTR, 12=FTRD, 13=FT%, 14=OP FT%, 15=2PT%, ...

def extract_val_br(td) -> float:
    """For teamstats.php cells: value is text before <br>, rank is in a plain span after."""
    text = td.get_text(separator="|", strip=True)
    val_str = text.split("|")[0].strip().lstrip("+")
    try:
        return float(val_str)
    except ValueError:
        return np.nan


def scrape_teamstats(s, hdrs) -> pd.DataFrame:
    """Returns DataFrame with BARTO_TEAM, FT%, OP FT%."""
    print("  Fetching teamstats.php...")
    html = fetch(s, hdrs, f"{BARTO_BASE}/teamstats.php?year=2026")
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    records = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 15:
            continue
        first_text = cells[0].get_text(strip=True)
        if not first_text.isdigit():
            continue

        # Team name from cell[1]
        team_cell = cells[1]
        a = team_cell.find("a")
        if not a:
            continue
        for sp in a.find_all("span"):
            sp.decompose()
        team = re.sub(r'\s*(vs\.|def\.|lost|won).*$', '', a.get_text(strip=True),
                      flags=re.IGNORECASE).strip()

        # Cell layout: 3=AdjO, 4=AdjD, 5=EFG%, 6=EFG%D, 7=TOV%, 8=TOV%D,
        #              9=OREB%, 10=DRB, 11=FTR, 12=FTRD, 13=FT%, 14=OP FT%
        ft_off = extract_val_br(cells[13])
        ft_def = extract_val_br(cells[14])

        records.append({
            "BARTO_TEAM": team,
            "FT%":    ft_off,
            "OP FT%": ft_def,
        })

    df = pd.DataFrame(records)
    print(f"    Parsed {len(df)} teams from teamstats.php")
    return df


# ── 3. team.php per team — AST%, OP AST%, BLK%, BLKED% ───────────────────────
TEAM_STAT_MAP = {
    "Ast. Rate": ("AST%", "OP AST%"),
    "Block %":   ("BLK%", "BLKED%"),
}

def scrape_team_page(s, hdrs, barto_name: str) -> dict:
    """Fetch a single team page and extract AST%, OP AST%, BLK%, BLKED%."""
    url = f"{BARTO_BASE}/team.php?team={barto_team_url_name(barto_name)}&year=2026"
    try:
        html = fetch(s, hdrs, url)
    except Exception as e:
        print(f"      ERROR fetching {barto_name}: {e}")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return {}

    result = {}
    t0 = tables[0]
    for row in t0.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        first_text = cells[0].get_text(strip=True)
        if first_text in TEAM_STAT_MAP:
            off_col, def_col = TEAM_STAT_MAP[first_text]
            try:
                result[off_col] = float(cells[1].get_text(strip=True))
            except (ValueError, IndexError):
                result[off_col] = np.nan
            try:
                result[def_col] = float(cells[3].get_text(strip=True))
            except (ValueError, IndexError):
                result[def_col] = np.nan

        # Once both found, stop
        if len(result) == 4:
            break

    return result


def scrape_all_team_pages(s, hdrs, teams: list) -> pd.DataFrame:
    """Scrape team.php for each of the 64 bracket teams."""
    print(f"  Fetching team.php for {len(teams)} teams (64 requests)...")
    records = []
    for i, (barto_name, pipeline_name) in enumerate(teams):
        stats = scrape_team_page(s, hdrs, barto_name)
        stats["TEAM"] = pipeline_name
        records.append(stats)
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(teams)} done")
        time.sleep(0.15)   # polite delay

    df = pd.DataFrame(records)
    print(f"    Scraped {len(df)} team pages")
    return df


# ── 4. Tournament Matchups.csv — PROG_GAMES, PROG_WIN%, CONF_WIN% ─────────────
CONF_REMAP = {"P10": "P12", "SInd": "Slnd"}

def compute_historical_features(bracket_teams: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the build_master_table.py logic for PROG_* and CONF_* using
    all matchup data through 2025 (strictly prior to 2026).
    """
    print("  Computing PROG_GAMES, PROG_WIN%, CONF_WIN% from Tournament Matchups.csv...")

    matchups = pd.read_csv(MATCHUPS)
    kenpom   = pd.read_csv(KENPOM_CSV)
    kenpom["CONF"] = kenpom["CONF"].replace(CONF_REMAP)

    # -- replicate game_records construction from build_master_table.py -------
    m = matchups.sort_values("BY YEAR NO", ascending=False).reset_index(drop=True)
    m["GAME_ID"] = m.index // 2
    m["SLOT"]    = m.index % 2

    ga = (m[m["SLOT"] == 0]
          [["YEAR", "GAME_ID", "TEAM NO", "SEED", "SCORE", "CURRENT ROUND"]]
          .rename(columns={"TEAM NO": "TEAM_NO", "SEED": "SEED_A",
                           "SCORE": "SCORE_A", "CURRENT ROUND": "CUR_ROUND"}))
    gb = (m[m["SLOT"] == 1]
          [["YEAR", "GAME_ID", "TEAM NO", "SEED", "SCORE"]]
          .rename(columns={"TEAM NO": "TEAM_NO_B", "SEED": "SEED_B", "SCORE": "SCORE_B"}))

    games = ga.merge(gb, on=["YEAR", "GAME_ID"])
    games["A_WIN"] = (games["SCORE_A"] > games["SCORE_B"]).astype(int)
    games["B_WIN"] = 1 - games["A_WIN"]

    rA = (games[["YEAR", "TEAM_NO", "SEED_A", "A_WIN", "CUR_ROUND"]]
          .rename(columns={"SEED_A": "SEED", "A_WIN": "WIN"}))
    rB = (games[["YEAR", "TEAM_NO_B", "SEED_B", "B_WIN", "CUR_ROUND"]]
          .rename(columns={"TEAM_NO_B": "TEAM_NO", "SEED_B": "SEED", "B_WIN": "WIN"}))
    game_records = pd.concat([rA, rB], ignore_index=True)

    # Attach TEAM ID and CONF
    team_id_map = (kenpom[["YEAR", "TEAM NO", "TEAM ID", "CONF"]].copy()
                   .rename(columns={"TEAM NO": "TEAM_NO"}))
    team_id_map["CONF"] = team_id_map["CONF"].replace(CONF_REMAP)
    game_records = game_records.merge(team_id_map, on=["YEAR", "TEAM_NO"], how="left")

    # -- PROG_ (program history) -----------------------------------------------
    def safe_div(a, b):
        return np.where(b > 0, a / b, np.nan)

    # All games through 2025 only
    hist = game_records[game_records["YEAR"] < 2026].copy()

    team_yr = (hist.groupby(["YEAR", "TEAM ID"])
               .agg(GAMES=("WIN", "count"), W=("WIN", "sum"))
               .reset_index())

    all_team_yr = (hist.groupby("TEAM ID")
                   .agg(PROG_GAMES=("WIN", "count"), PROG_W=("WIN", "sum"))
                   .reset_index())
    all_team_yr["PROG_WIN%"] = safe_div(all_team_yr["PROG_W"], all_team_yr["PROG_GAMES"])

    # -- CONF_ (conference history) -------------------------------------------
    conf_hist = hist.dropna(subset=["CONF"])
    conf_agg = (conf_hist.groupby("CONF")
                .agg(CONF_GAMES=("WIN", "count"), CONF_W=("WIN", "sum"))
                .reset_index())
    conf_agg["CONF_WIN%"] = safe_div(conf_agg["CONF_W"], conf_agg["CONF_GAMES"])

    # -- Look up TEAM ID for 2026 bracket teams --------------------------------
    # Use the most-recent TEAM NO → TEAM ID mapping from kenpom
    latest_ids = (kenpom.sort_values("YEAR")
                  .groupby("TEAM NO")[["TEAM ID"]].last()
                  .reset_index()
                  .rename(columns={"TEAM NO": "TEAM NO"}))

    # bracket_teams has: TEAM, TEAM_NO, CONF (from Barttorvik scrape)
    bt = bracket_teams.copy()
    bt = bt.merge(latest_ids.rename(columns={"TEAM NO": "TEAM_NO"}),
                  on="TEAM_NO", how="left")

    bt = bt.merge(all_team_yr[["TEAM ID", "PROG_GAMES", "PROG_WIN%"]],
                  on="TEAM ID", how="left")
    bt = bt.merge(conf_agg[["CONF", "CONF_WIN%"]], on="CONF", how="left")

    out_cols = ["TEAM", "PROG_GAMES", "PROG_WIN%", "CONF_WIN%"]
    result = bt[out_cols].copy()

    found = result["PROG_GAMES"].notna().sum()
    print(f"    PROG_GAMES populated for {found}/64 teams")
    conf_found = result["CONF_WIN%"].notna().sum()
    print(f"    CONF_WIN% populated for {conf_found}/64 teams")
    return result


# ── neutral site check ────────────────────────────────────────────────────────
def check_neutral(s, hdrs):
    print("  Checking neutral-site trank.php...")
    for venue in ["Neutral", "AN"]:
        url = f"{BARTO_BASE}/trank.php?year=2026&venue={venue}&conlimit=All"
        html = fetch(s, hdrs, url)
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        rows = table.find_all("tr") if table else []
        data = [r for r in rows if r.find("td") and
                len(r.find_all(["th", "td"])) > 3 and
                r.find_all(["th", "td"])[0].get_text(strip=True).isdigit()]
        print(f"    venue={venue}: {len(data)} data rows")
    print("    NOTE: Neutral-site splits are EMPTY — no data available for 2026.")


# ── main ──────────────────────────────────────────────────────────────────────
# Barttorvik name → pipeline name (same map as scrape_barttorvik_2026.py)
BARTO_TO_PIPELINE = {
    "Connecticut":         "UConn",
    "UConn":               "UConn",
    "N.C. State":          "NC State",
    "NC State":            "NC State",
    "North Carolina St.":  "NC State",
    "McNeese St.":         "McNeese",
    "McNeese":             "McNeese",
    "Saint Mary's (CA)":   "Saint Mary's",
    "California Baptist":  "Cal Baptist",
    "LIU":                 "Long Island",
    "LIU Brooklyn":        "Long Island",
    "Queens":              "Queens NC",
    "Tennessee St.":       "Tennessee St.",
    "Tennessee State":     "Tennessee St.",
    "St. Louis":           "Saint Louis",
    "Miami (FL)":          "Miami FL",
    "Miami FL":            "Miami FL",
    "Brigham Young":       "BYU",
}


def main():
    df = pd.read_csv(MAPPED_CSV)
    raw = pd.read_csv(RAW_CSV)

    # Build Barttorvik → pipeline name map using the raw BARTO_TEAM column
    # plus the known remaps
    barto_to_pipeline = {name: BARTO_TO_PIPELINE.get(name, name)
                         for name in raw["BARTO_TEAM"].tolist()}

    # Build reverse map: pipeline_name → barto_name
    pipeline_to_barto = {}
    for bname, pname in barto_to_pipeline.items():
        if pname not in pipeline_to_barto:
            pipeline_to_barto[pname] = bname

    bracket_teams_list = []  # [(barto_name, pipeline_name), ...]
    for _, row in df.iterrows():
        pname = row["TEAM"]
        bname = pipeline_to_barto.get(pname, pname)
        bracket_teams_list.append((bname, pname))

    print("=" * 65)
    print("Establishing Barttorvik session...")
    s, hdrs = make_session()

    # ── Source 1: teampbp.php ─────────────────────────────────────────────────
    print("\n[1/4] Shooting splits (teampbp.php)...")
    pbp_df = scrape_teampbp(s, hdrs)
    # Normalise names the same way as raw
    pbp_df["TEAM"] = pbp_df["BARTO_TEAM"].map(barto_to_pipeline).fillna(pbp_df["BARTO_TEAM"])
    pbp_cols = [c for c in PBP_CLASS_MAP.values()]
    pbp_bracket = pbp_df[pbp_df["TEAM"].isin(df["TEAM"].tolist())]
    print(f"    Matched {len(pbp_bracket)}/64 bracket teams")

    # ── Source 2: teamstats.php ───────────────────────────────────────────────
    print("\n[2/4] FT% splits (teamstats.php)...")
    ts_df = scrape_teamstats(s, hdrs)
    ts_df["TEAM"] = ts_df["BARTO_TEAM"].map(barto_to_pipeline).fillna(ts_df["BARTO_TEAM"])
    ts_bracket = ts_df[ts_df["TEAM"].isin(df["TEAM"].tolist())]
    print(f"    Matched {len(ts_bracket)}/64 bracket teams")

    # ── Source 3: team.php per team ───────────────────────────────────────────
    print("\n[3/4] AST%, BLK% (team.php individual pages)...")
    team_df = scrape_all_team_pages(s, hdrs, bracket_teams_list)

    # ── Neutral site check ────────────────────────────────────────────────────
    print("\n[Neutral check]")
    check_neutral(s, hdrs)

    # ── Source 4: Tournament Matchups.csv ─────────────────────────────────────
    print("\n[4/4] Historical features (Tournament Matchups.csv)...")
    # bracket_teams needs TEAM_NO and CONF for lookup
    bracket_info = df[["TEAM", "TEAM NO", "CONF"]].rename(columns={"TEAM NO": "TEAM_NO"})
    hist_df = compute_historical_features(bracket_info)

    # ── Merge all into mapped CSV ─────────────────────────────────────────────
    print("\nMerging all sources into mapped CSV...")
    out = df.copy()

    # Source 1: shooting splits
    merge1 = pbp_bracket[["TEAM"] + [c for c in pbp_cols if c in pbp_bracket.columns]]
    out = out.merge(merge1, on="TEAM", how="left", suffixes=("", "_pbp"))
    # Copy into proper schema columns
    col_map_pbp = {
        "DUNKS FG%":         "DUNKS FG%",       # Shooting Splits schema cols
        "DUNKS SHARE":       "DUNKS SHARE",
        "DUNKS FG%D":        "DUNKS FG%D",
        "CLOSE TWOS FG%":    "CLOSE TWOS FG%",
        "CLOSE TWOS SHARE":  "CLOSE TWOS SHARE",
        "CLOSE TWOS FG%D":   "CLOSE TWOS FG%D",
        "FARTHER TWOS FG%":  "FARTHER TWOS FG%",
        "FARTHER TWOS SHARE":"FARTHER TWOS SHARE",
        "FARTHER TWOS FG%D": "FARTHER TWOS FG%D",
        "THREES FG%":        "THREES FG%",
        "THREES SHARE":      "THREES SHARE",
        "THREES FG%D":       "THREES FG%D",
    }
    # These are new columns not in the original schema — add them
    for src_col in col_map_pbp:
        if src_col + "_pbp" in out.columns:
            out[src_col] = out[src_col + "_pbp"]
            out.drop(columns=[src_col + "_pbp"], inplace=True)
        elif src_col in out.columns:
            pass  # already set from merge

    # Source 2: FT%, OP FT%
    merge2 = ts_bracket[["TEAM", "FT%", "OP FT%"]]
    out = out.merge(merge2, on="TEAM", how="left", suffixes=("", "_ts"))
    for col in ["FT%", "OP FT%"]:
        ts_col = col + "_ts"
        if ts_col in out.columns:
            # Fill NaN in original with scraped values
            out[col] = out[col].combine_first(out[ts_col])
            out.drop(columns=[ts_col], inplace=True)

    # Source 3: AST%, OP AST%, BLK%, BLKED%
    merge3 = team_df[["TEAM", "AST%", "OP AST%", "BLK%", "BLKED%"]
                     if all(c in team_df.columns for c in ["AST%", "OP AST%", "BLK%", "BLKED%"])
                     else ["TEAM"] + [c for c in ["AST%", "OP AST%", "BLK%", "BLKED%"] if c in team_df.columns]]
    out = out.merge(merge3, on="TEAM", how="left", suffixes=("", "_tm"))
    for col in ["AST%", "OP AST%", "BLK%", "BLKED%"]:
        tm_col = col + "_tm"
        if tm_col in out.columns:
            out[col] = out[col].combine_first(out[tm_col])
            out.drop(columns=[tm_col], inplace=True)

    # Source 4: PROG_GAMES, PROG_WIN%, CONF_WIN%
    merge4 = hist_df[["TEAM", "PROG_GAMES", "PROG_WIN%", "CONF_WIN%"]]
    out = out.merge(merge4, on="TEAM", how="left", suffixes=("", "_hist"))
    for col in ["PROG_GAMES", "PROG_WIN%", "CONF_WIN%"]:
        hcol = col + "_hist"
        if hcol in out.columns:
            out[col] = out[col].combine_first(out[hcol])
            out.drop(columns=[hcol], inplace=True)

    # ── Coverage report ───────────────────────────────────────────────────────
    new_cols = [
        "THREES FG%D", "DUNKS FG%D", "CLOSE TWOS FG%", "DUNKS SHARE",
        "CLOSE TWOS SHARE", "THREES FG%", "THREES SHARE", "FARTHER TWOS FG%",
        "FARTHER TWOS SHARE", "FARTHER TWOS FG%D",
        "FT%", "OP FT%",
        "AST%", "OP AST%", "BLK%", "BLKED%",
        "PROG_GAMES", "PROG_WIN%", "CONF_WIN%",
    ]

    print("\n" + "=" * 65)
    print("COVERAGE REPORT — newly populated columns (64 teams)")
    print("-" * 65)
    for col in new_cols:
        if col in out.columns:
            n = out[col].notna().sum()
            sample = out[col].dropna().iloc[0] if out[col].notna().any() else "N/A"
            print(f"  {col:<25} {n:>3}/64  (sample: {sample:.3f})" if n > 0 else f"  {col:<25}   0/64  EMPTY")
        else:
            print(f"  {col:<25}   --   NOT IN OUTPUT")

    # ── Save ──────────────────────────────────────────────────────────────────
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved enriched file to {OUT_CSV} ({len(out)} rows, {len(out.columns)} cols)")

    # Quick sanity check on key teams
    key_teams = ["Duke", "Florida", "Michigan", "Akron", "Furman"]
    check_cols = ["TEAM", "THREES FG%D", "CLOSE TWOS FG%", "DUNKS FG%D",
                  "FT%", "AST%", "BLK%", "PROG_GAMES", "PROG_WIN%", "CONF_WIN%"]
    avail = [c for c in check_cols if c in out.columns]
    print("\nSanity check — key teams:")
    print(out[out["TEAM"].isin(key_teams)][avail].to_string(index=False))

    return out


if __name__ == "__main__":
    main()
