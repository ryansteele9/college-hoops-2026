"""
injury_adjustments.py — Post-prediction injury adjustment module.

Applies win-probability adjustments based on player ratings, injury severity,
and recency. Does not modify any model pkl files.
"""

import numpy as np
import pandas as pd
from datetime import datetime

REFERENCE_DATE = datetime(2026, 3, 15)

TEAM_ABBREV_MAP = {
    "DUKE": "Duke",
    "TTU":  "Texas Tech",
    "UNC":  "North Carolina",
    "WISC": "Wisconsin",
    "LOU":  "Louisville",
    "UCLA": "UCLA",
    "CLEM": "Clemson",
    "NOVA": "Villanova",
}

SEVERITY_MULT = {"severe": 1.00, "moderate": 0.75, "questionable": 0.30}


def _parse_injury_date(s) -> datetime:
    """Parse injury date string in '10-Mar' or 'Mar 10' format. Sets year=2026."""
    s = str(s).strip()
    for fmt in ("%d-%b", "%b %d", "%B %d", "%d-%B"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(year=2026)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse injury date: {s!r}")


def load_adjustments(csv_path, reference_date=None, abbrev_map=None):
    """
    Load injury data and compute per-team probability adjustments.

    Returns:
        team_adjustments: {full_name: sum_of_adjustments}
        detail_rows: [per-player dict with adjustment details]
    """
    if reference_date is None:
        reference_date = REFERENCE_DATE
    if abbrev_map is None:
        abbrev_map = TEAM_ABBREV_MAP

    df = pd.read_csv(csv_path)
    # Drop rows where TEAM is blank/NaN (CSV has trailing empty rows)
    df = df[df["TEAM"].notna() & (df["TEAM"].astype(str).str.strip() != "")]

    team_adjustments = {}
    detail_rows = []

    for _, row in df.iterrows():
        abbrev    = str(row["TEAM"]).strip()
        full_name = abbrev_map.get(abbrev, abbrev)
        player    = str(row["PLAYER"]).strip()
        rating    = float(row["RATING"])
        severity_str = str(row["SEVERITY"]).strip().lower()
        date_str  = row["GAMES_MISSED"]   # column contains injury date

        # Parse date and compute days since injury
        inj_date   = _parse_injury_date(date_str)
        days_since = (reference_date - inj_date).days

        # Compute adjustment
        base_penalty = 0.06 * (rating / 5.0)
        sev_mult     = SEVERITY_MULT.get(severity_str, 0.50)
        if days_since <= 3:
            rec_weight = 1.00
        elif days_since <= 7:
            rec_weight = 0.85
        else:
            rec_weight = 0.70

        adjustment = base_penalty * sev_mult * rec_weight

        team_adjustments[full_name] = team_adjustments.get(full_name, 0.0) + adjustment
        detail_rows.append({
            "team":       full_name,
            "player":     player,
            "rating":     rating,
            "severity":   severity_str,
            "days_since": days_since,
            "adj":        round(adjustment, 4),
        })

    return team_adjustments, detail_rows


def wrap_predictor(base_pred, team_adjustments):
    """
    Wrap a predictor function to apply injury adjustments.
    Reads _TEAM key from feats dicts (set by build_bracket).
    """
    def adjusted_pred(fa, fb):
        p     = base_pred(fa, fb)
        adj_a = team_adjustments.get(fa.get("_TEAM", ""), 0.0)
        adj_b = team_adjustments.get(fb.get("_TEAM", ""), 0.0)
        return float(np.clip(p - adj_a + adj_b, 0.05, 0.95))

    # Forward batch_predict unchanged — MC uses matrix-level adjustment instead
    if hasattr(base_pred, "batch_predict"):
        adjusted_pred.batch_predict = base_pred.batch_predict

    return adjusted_pred


def adjust_prob_matrix(prob_matrix, team_names, team_adjustments):
    """
    Apply injury adjustments to a precomputed win-probability matrix.
    Returns a copy with adjustments applied. Symmetry preserved: p[i,j]+p[j,i]=1.
    """
    adj_vec  = np.array(
        [team_adjustments.get(n, 0.0) for n in team_names], dtype=np.float32
    )
    adjusted = prob_matrix.copy()
    adjusted -= adj_vec[:, None]   # row team injured → win prob goes down
    adjusted += adj_vec[None, :]   # col team injured → opponent's win prob goes up
    np.fill_diagonal(adjusted, 0.0)
    np.clip(adjusted, 0.05, 0.95, out=adjusted)
    return adjusted


def print_summary(detail_rows, team_adjustments):
    """Print per-player table and team totals."""
    print()
    print("=" * 70)
    print("INJURY ADJUSTMENTS APPLIED")
    print("=" * 70)
    print(f"  {'Team':<18}  {'Player':<12}  {'Rat':>3}  {'Severity':<12}  "
          f"{'Days':>4}  {'Adj':>6}")
    print(f"  {'-'*62}")
    for r in detail_rows:
        print(f"  {r['team']:<18}  {r['player']:<12}  {int(r['rating']):>3}  "
              f"{r['severity']:<12}  {r['days_since']:>4}  {r['adj']:>6.4f}")

    print()
    print("  Team penalty totals:")
    for team, total in sorted(team_adjustments.items(), key=lambda x: -x[1]):
        print(f"    {team:<20}  {total:.4f}")


def print_delta_table(mc_before, mc_after, team_adjustments):
    """Print CHAMP probability delta for each affected team."""
    print()
    print("=" * 70)
    print("WIN PROBABILITY DELTA  (Injury-adjusted vs Baseline)")
    print("=" * 70)
    print(f"  {'Team':<22}  {'Before':>7}  {'After':>7}  {'Delta':>7}")
    print(f"  {'-'*48}")

    for team in sorted(team_adjustments.keys(), key=lambda t: -team_adjustments[t]):
        before_row = mc_before[mc_before["TEAM"] == team]
        after_row  = mc_after[mc_after["TEAM"] == team]
        if before_row.empty or after_row.empty:
            continue
        b     = float(before_row["CHAMP"].iloc[0])
        a     = float(after_row["CHAMP"].iloc[0])
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        print(f"  {team:<22}  {b:>7.4f}  {a:>7.4f}  {sign}{delta:>6.4f}")
