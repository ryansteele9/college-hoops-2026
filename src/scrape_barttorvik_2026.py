"""
Scrape 2026 Barttorvik T-Rank data and map to KenPom Barttorvik.csv schema.
Usage: python src/scrape_barttorvik_2026.py
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import sys

# ── constants ────────────────────────────────────────────────────────────────
URL = "https://barttorvik.com/trank.php?year=2026&conlimit=All&venue=All"
RAW_OUT  = "data/raw/barttorvik_2026_raw.csv"
SCHEMA_FILE = "data/raw/KenPom Barttorvik.csv"
BRACKET_FILE = "data/raw/bracket_2026.csv"

# Barttorvik team names → bracket/pipeline names
# Only entries that differ are listed; exact matches pass through unchanged.
BARTO_TO_PIPELINE = {
    "UConn":                "UConn",          # keep as UConn (bracket uses UConn)
    "Connecticut":          "UConn",
    "NC State":             "NC State",       # keep
    "N.C. State":           "NC State",
    "North Carolina St.":   "NC State",
    "McNeese St.":          "McNeese",
    "McNeese":              "McNeese",
    "Saint Mary's":         "Saint Mary's",
    "Saint Mary's (CA)":    "Saint Mary's",
    "Cal Baptist":          "Cal Baptist",
    "California Baptist":   "Cal Baptist",
    "LIU":                  "Long Island",
    "LIU Brooklyn":         "Long Island",
    "Long Island":          "Long Island",
    "Queens":               "Queens NC",
    "Queens NC":            "Queens NC",
    "Tennessee St.":        "Tennessee St.",
    "Tennessee State":      "Tennessee St.",
    "Howard":               "Howard",
    "Saint Louis":          "Saint Louis",
    "St. Louis":            "Saint Louis",
    "North Carolina":       "North Carolina",
    "VCU":                  "VCU",
    "Miami (FL)":           "Miami FL",
    "Miami FL":             "Miami FL",
    "Miami (Ohio)":         "Miami OH",       # not in bracket, keep separate
    "BYU":                  "BYU",
    "Brigham Young":        "BYU",
    "TCU":                  "TCU",
    "SMU":                  "SMU",
    "UCF":                  "UCF",
    "Northern Iowa":        "Northern Iowa",
    "North Dakota St.":     "North Dakota St.",
    "High Point":           "High Point",
    "Kennesaw St.":         "Kennesaw St.",
    "Kennesaw State":       "Kennesaw St.",
    "South Florida":        "South Florida",
    "Saint Mary's":         "Saint Mary's",
    "Wright St.":           "Wright St.",
    "Wright State":         "Wright St.",
    "Idaho":                "Idaho",
    "Hofstra":              "Hofstra",
    "Santa Clara":          "Santa Clara",
}

# ── fetch page ────────────────────────────────────────────────────────────────
def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    session = requests.Session()
    session.get(url, headers=headers, timeout=30)          # seed cookies
    r = session.post(url, data={"js_test_submitted": "1"}, # pass JS gate
                     headers=headers, timeout=60)
    r.raise_for_status()
    return r.text

# ── parse table ───────────────────────────────────────────────────────────────
def parse_cell(td):
    """Return (value_str, rank_str) from a td that may contain a .lowrow span."""
    span = td.find("span", class_="lowrow")
    rank_str = span.get_text(strip=True) if span else ""
    if span:
        span.decompose()
    val_str = td.get_text(strip=True)
    return val_str, rank_str

def parse_team_cell(td):
    """Extract clean team name from the teamname cell."""
    a = td.find("a")
    if not a:
        return td.get_text(strip=True)
    # Remove any nested spans (e.g. seeding notes)
    for span in a.find_all("span"):
        span.decompose()
    raw = a.get_text(strip=True)
    # Strip tournament result suffixes like "vs. 8 Purdue (lost)"
    raw = re.sub(r'\s*(vs\.|def\.|lost|won).*$', '', raw, flags=re.IGNORECASE).strip()
    return raw

def parse_record(rec_str):
    """'31-3' -> (31, 3, 91.18)"""
    m = re.match(r'(\d+)-(\d+)', rec_str.strip())
    if m:
        w, l = int(m.group(1)), int(m.group(2))
        games = w + l
        pct = round(100 * w / games, 5) if games > 0 else 0.0
        return w, l, games, pct
    return None, None, None, None

def to_float(s):
    s = s.strip().lstrip('+')
    try:
        return float(s)
    except ValueError:
        return np.nan

def to_int(s):
    try:
        return int(s.strip())
    except ValueError:
        return np.nan

def parse_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    records = []
    for row in rows:
        cells = row.find_all(["th", "td"])
        # Data rows have exactly 24 cells and start with an integer rank
        if len(cells) < 24:
            continue
        rank_text = cells[0].get_text(strip=True)
        if not rank_text.isdigit():
            continue

        # Cell positions in the rendered table:
        # 0=Rank  1=Team  2=Conf  3=G  4=Rec
        # 5=AdjOE  6=AdjDE  7=Barthag
        # 8=EFG%  9=EFGD%  10=TOR  11=TORD
        # 12=ORB  13=DRB  14=FTR  15=FTRD
        # 16=2P%  17=2P%D  18=3P%  19=3P%D
        # 20=3PR  21=3PRD  22=AdjT  23=WAB

        def v(i):  return parse_cell(cells[i])[0]
        def r(i):  return parse_cell(cells[i])[1]

        team  = parse_team_cell(cells[1])
        conf  = cells[2].get_text(strip=True)
        games_raw = cells[3].get_text(strip=True)
        rec_raw   = cells[4].get_text(strip=True)

        w, l, g, pct = parse_record(rec_raw)

        badj_o    = to_float(v(5));  badj_o_rank  = to_int(r(5))
        badj_d    = to_float(v(6));  badj_d_rank  = to_int(r(6))
        barthag   = to_float(v(7));  barthag_rank = to_int(r(7))
        badj_em   = round(badj_o - badj_d, 5) if not (np.isnan(badj_o) or np.isnan(badj_d)) else np.nan

        efg       = to_float(v(8));   efg_rank    = to_int(r(8))
        efgd      = to_float(v(9));   efgd_rank   = to_int(r(9))
        tor       = to_float(v(10));  tor_rank    = to_int(r(10))
        tord      = to_float(v(11));  tord_rank   = to_int(r(11))
        orb       = to_float(v(12));  orb_rank    = to_int(r(12))
        drb       = to_float(v(13));  drb_rank    = to_int(r(13))
        ftr       = to_float(v(14));  ftr_rank    = to_int(r(14))
        ftrd      = to_float(v(15));  ftrd_rank   = to_int(r(15))
        two_p     = to_float(v(16));  two_p_rank  = to_int(r(16))
        two_pd    = to_float(v(17));  two_pd_rank = to_int(r(17))
        three_p   = to_float(v(18));  three_p_rank= to_int(r(18))
        three_pd  = to_float(v(19));  three_pd_rank= to_int(r(19))
        three_pr  = to_float(v(20));  three_pr_rank= to_int(r(20))
        three_prd = to_float(v(21));  three_prd_rank= to_int(r(21))
        badj_t    = to_float(v(22));  badj_t_rank = to_int(r(22))
        wab       = to_float(v(23));  wab_rank    = to_int(r(23))

        # Derived columns
        two_pr    = round(100 - three_pr,  2) if not np.isnan(three_pr)  else np.nan
        two_prd   = round(100 - three_prd, 2) if not np.isnan(three_prd) else np.nan
        op_oreb   = round(100 - drb,  2) if not np.isnan(drb)  else np.nan
        op_dreb   = round(100 - orb,  2) if not np.isnan(orb)  else np.nan

        # Barttorvik rank for BADJ EM is not directly on page; approximate with position
        badj_em_rank = to_int(rank_text)

        records.append({
            # Identification
            "BARTO_TEAM": team,
            "CONF":       conf,
            # Basic record
            "GAMES": g,  "W": w,  "L": l,  "WIN%": pct,
            # Core efficiency
            "BADJ EM": badj_em,   "BADJ EM RANK": badj_em_rank,
            "BADJ O":  badj_o,    "BADJ O RANK":  badj_o_rank,
            "BADJ D":  badj_d,    "BADJ D RANK":  badj_d_rank,
            "BARTHAG": barthag,   "BARTHAG RANK": barthag_rank,
            # Four factors
            "EFG%":   efg,    "EFG% RANK":   efg_rank,
            "EFG%D":  efgd,   "EFGD% RANK":  efgd_rank,
            "TOV%":   tor,    "TOV% RANK":   tor_rank,
            "TOV%D":  tord,   "TOV%D RANK":  tord_rank,
            "OREB%":  orb,    "OREB% RANK":  orb_rank,
            "DREB%":  drb,    "DREB% RANK":  drb_rank,
            "OP OREB%": op_oreb,  "OP OREB% RANK": np.nan,
            "OP DREB%": op_dreb,  "OP DREB% RANK": np.nan,
            "FTR":    ftr,    "FTR RANK":    ftr_rank,
            "FTRD":   ftrd,   "FTRD RANK":   ftrd_rank,
            # Shooting
            "2PT%":   two_p,   "2PT% RANK":   two_p_rank,
            "2PT%D":  two_pd,  "2PT%D RANK":  two_pd_rank,
            "3PT%":   three_p,  "3PT% RANK":   three_p_rank,
            "3PT%D":  three_pd, "3PT%D RANK":  three_pd_rank,
            "2PTR":   two_pr,   "2PTR RANK":   np.nan,
            "3PTR":   three_pr,  "3PTR RANK":  three_pr_rank,
            "2PTRD":  two_prd,  "2PTRD RANK":  np.nan,
            "3PTRD":  three_prd, "3PTRD RANK": three_prd_rank,
            # Tempo
            "BADJ T": badj_t,  "BADJT RANK": badj_t_rank,
            # WAB
            "WAB":    wab,     # no separate rank col in schema
        })

    return pd.DataFrame(records)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Fetching barttorvik.com T-Rank 2026...")
    html = fetch_page(URL)
    print(f"  Page size: {len(html):,} bytes")

    print("Parsing table...")
    raw_df = parse_table(html)
    print(f"  Rows parsed: {len(raw_df)}")

    # Save raw (all teams)
    raw_df.to_csv(RAW_OUT, index=False)
    print(f"  Raw data saved to {RAW_OUT}")

    # ── load bracket & schema ────────────────────────────────────────────────
    bracket = pd.read_csv(BRACKET_FILE)
    schema_df = pd.read_csv(SCHEMA_FILE)
    schema_cols = list(schema_df.columns)

    bracket_teams = set(bracket["TEAM"].tolist())

    # ── normalize Barttorvik names → pipeline names ──────────────────────────
    def normalize(name):
        return BARTO_TO_PIPELINE.get(name, name)

    raw_df["TEAM"] = raw_df["BARTO_TEAM"].apply(normalize)

    # ── filter to bracket teams ──────────────────────────────────────────────
    matched = raw_df[raw_df["TEAM"].isin(bracket_teams)].copy()
    unmatched_barto = raw_df[~raw_df["TEAM"].isin(bracket_teams)]["BARTO_TEAM"].tolist()

    found_teams = set(matched["TEAM"].tolist())
    missing_from_barto = bracket_teams - found_teams
    teams_with_zero_no = bracket[bracket["TEAM_NO"] == 0]["TEAM"].tolist()

    print(f"\n{'='*60}")
    print(f"Bracket teams found in Barttorvik data: {len(found_teams)}/64")
    if missing_from_barto:
        print(f"\nMISSING from Barttorvik scrape ({len(missing_from_barto)}):")
        for t in sorted(missing_from_barto):
            print(f"  - {t}")
    print(f"\nTeams with TEAM_NO=0 in bracket (not in master table): {teams_with_zero_no}")

    # ── build output with full schema ────────────────────────────────────────
    out_rows = []
    for _, brow in matched.iterrows():
        team = brow["TEAM"]
        bkt  = bracket[bracket["TEAM"] == team].iloc[0]

        # Spine columns — placeholders for IDs not yet assigned
        row = {c: np.nan for c in schema_cols}
        row["YEAR"]    = 2026
        row["CONF"]    = brow["CONF"]
        row["CONF ID"] = np.nan   # assigned by build_master_table
        row["QUAD NO"] = np.nan
        row["QUAD ID"] = np.nan
        row["TEAM NO"] = int(bkt["TEAM_NO"]) if bkt["TEAM_NO"] != 0 else np.nan
        row["TEAM ID"] = np.nan
        row["TEAM"]    = team
        row["SEED"]    = int(bkt["SEED"])
        row["ROUND"]   = 64       # first round placeholder

        # KenPom columns — not available without subscription; leave NaN
        for c in ["K TEMPO","K TEMPO RANK","KADJ T","KADJ T RANK",
                  "K OFF","KO RANK","KADJ O","KADJ O RANK",
                  "K DEF","KD RANK","KADJ D","KADJ D RANK",
                  "KADJ EM","KADJ EM RANK"]:
            row[c] = np.nan

        # Barttorvik columns we scraped
        for src, dst in [
            ("BADJ EM", "BADJ EM"),  ("BADJ EM RANK", "BADJ EM RANK"),
            ("BADJ O",  "BADJ O"),   ("BADJ O RANK",  "BADJ O RANK"),
            ("BADJ D",  "BADJ D"),   ("BADJ D RANK",  "BADJ D RANK"),
            ("BARTHAG", "BARTHAG"),  ("BARTHAG RANK", "BARTHAG RANK"),
            ("GAMES",   "GAMES"),    ("W", "W"),  ("L", "L"),  ("WIN%", "WIN%"),
            ("EFG%",    "EFG%"),     ("EFG% RANK",  "EFG% RANK"),
            ("EFG%D",   "EFG%D"),    ("EFGD% RANK", "EFGD% RANK"),
            ("FTR",     "FTR"),      ("FTR RANK",   "FTR RANK"),
            ("FTRD",    "FTRD"),     ("FTRD RANK",  "FTRD RANK"),
            ("TOV%",    "TOV%"),     ("TOV% RANK",  "TOV% RANK"),
            ("TOV%D",   "TOV%D"),    ("TOV%D RANK", "TOV%D RANK"),
            ("OREB%",   "OREB%"),    ("OREB% RANK", "OREB% RANK"),
            ("DREB%",   "DREB%"),    ("DREB% RANK", "DREB% RANK"),
            ("OP OREB%","OP OREB%"), ("OP DREB%",   "OP DREB%"),
            ("2PT%",    "2PT%"),     ("2PT% RANK",  "2PT% RANK"),
            ("2PT%D",   "2PT%D"),    ("2PT%D RANK", "2PT%D RANK"),
            ("3PT%",    "3PT%"),     ("3PT% RANK",  "3PT% RANK"),
            ("3PT%D",   "3PT%D"),    ("3PT%D RANK", "3PT%D RANK"),
            ("2PTR",    "2PTR"),     ("3PTR",       "3PTR"),
            ("3PTR RANK","3PTR RANK"),
            ("2PTRD",   "2PTRD"),    ("3PTRD",      "3PTRD"),
            ("3PTRD RANK","3PTRD RANK"),
            ("BADJ T",  "BADJ T"),   ("BADJT RANK", "BADJT RANK"),
            ("WAB",     "WAB"),
        ]:
            if src in brow.index:
                row[dst] = brow[src]

        # Columns not on trank.php — will remain NaN (to be filled from other sources
        # or median-imputed): RAW T, BLK%, BLKED%, AST%, OP AST%, AVG HGT, EFF HGT,
        # EXP, TALENT, FT%, OP FT%, PPPO, PPPD, ELITE SOS

        out_rows.append(row)

    out_df = pd.DataFrame(out_rows, columns=schema_cols)
    out_df = out_df.sort_values("TEAM").reset_index(drop=True)

    # ── report column coverage ───────────────────────────────────────────────
    print(f"\nColumn coverage in mapped output ({len(out_df)} teams):")
    null_cols = []
    for col in schema_cols:
        n_null = out_df[col].isna().sum()
        if n_null == len(out_df):
            null_cols.append(col)
    print(f"  Fully populated:  {len(schema_cols) - len(null_cols)}/{len(schema_cols)} columns")
    print(f"  All-NaN columns ({len(null_cols)}) — require KenPom or other sources:")
    for c in null_cols:
        print(f"    {c}")

    return out_df, raw_df


if __name__ == "__main__":
    out_df, raw_df = main()

    # Preview
    print(f"\nFirst 5 rows (key cols):")
    preview_cols = ["TEAM","SEED","CONF","BADJ EM","BADJ O","BADJ D",
                    "BARTHAG","EFG%","EFG%D","OREB%","TOV%D","W","GAMES"]
    print(out_df[preview_cols].head(10).to_string(index=False))

    save = input("\nSave mapped output? (y/n): ").strip().lower()
    if save == 'y':
        out_path = "data/raw/barttorvik_2026_mapped.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    else:
        print("Not saved.")
