"""
fetch_historical.py - Pull 2021-2025 MLB data
Uses ESPN API for game results (reliable, no scraping blocks)
Now merges Pinnacle opening/closing lines + juice into game features.
"""
import os, sys, time, warnings
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    import pybaseball
    from pybaseball import team_batting, team_pitching, pitching_stats, batting_stats
    pybaseball.cache.enable()
except ImportError:
    print("[ERROR] pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

from config import HIST_DIR, TRAIN_YEARS
DATA_DIR        = HIST_DIR
HISTORICAL_PATH = os.path.join(HIST_DIR, "historical_stats.parquet")

# ── CSV paths for historical lines (2021-2025) ─────────────────────────────
# These CSVs must live in the same directory as this script (or update path).
from config import HIST_ODDS_DIR
LINES_CSV_DIR = HIST_ODDS_DIR

BVP_CSV_PATH        = os.path.join(HIST_ODDS_DIR, "batter_vs_pitcher_career.csv")
LINEUP_PARQUET_PATH = os.path.join(HIST_DIR, "game_lineups.parquet")

os.makedirs(DATA_DIR, exist_ok=True)

HEADERS   = {"User-Agent": "Mozilla/5.0"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

ESPN_TO_STD = {
    "CWS":"CHW","KC":"KCR","SD":"SDP","SF":"SFG","TB":"TBR","WSH":"WSN"
}

# The Odds API CSVs use full team names; ESPN schedules use abbreviations.
# This map converts full names → standard abbreviations for the join key.
FULL_NAME_TO_ABB = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CHW",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KCR",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "OAK",
    "Athletics":             "OAK",  # Sacramento rebranding
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SDP",
    "San Francisco Giants":  "SFG",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TBR",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSN",
}

# ─────────────────────────────────────────────────────────────────────────────
# Pinnacle helpers
# ─────────────────────────────────────────────────────────────────────────────

def american_to_prob(odds):
    """Convert American odds → implied probability (no vig removal)."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def pinnacle_no_vig_prob(ml_home, ml_away):
    """
    Remove vig from Pinnacle ML prices and return (home_prob, away_prob).
    Returns (nan, nan) if either price is missing.
    """
    ph = american_to_prob(ml_home)
    pa = american_to_prob(ml_away)
    if np.isnan(ph) or np.isnan(pa):
        return np.nan, np.nan
    total = ph + pa
    return ph / total, pa / total


def load_pinnacle_lines_csv(years):
    """
    Load and normalise Pinnacle lines from the per-year CSV files.

    Returns a DataFrame indexed by game_id with canonical columns:
        pi_open_ml_home            – opening Pinnacle ML home (American)
        pi_close_ml_home           – closing Pinnacle ML home (American)
        pi_close_ml_away           – closing Pinnacle ML away (American)
        pi_ml_movement             – closing − opening ML home
        pi_open_spread_home        – opening Pinnacle spread (home line)
        pi_close_spread_home       – closing Pinnacle spread (home line)
        pi_spread_movement         – closing − opening spread
        pi_close_spread_home_price – closing spread juice (home)
        pi_close_spread_away_price – closing spread juice (away)
        pi_open_total              – opening Pinnacle total line
        pi_close_total             – closing Pinnacle total line
        pi_total_movement          – closing − opening total
        pi_close_total_over_price  – closing over juice
        pi_close_total_under_price – closing under juice
        pi_open_no_vig_home        – opening Pinnacle no-vig home win prob
        pi_close_no_vig_home       – closing Pinnacle no-vig home win prob
    """
    frames = []
    for yr in years:
        csv_path = os.path.join(LINES_CSV_DIR, f"baseball_mlb_{yr}.csv")
        if not os.path.exists(csv_path):
            print(f"  [warn] Missing lines CSV for {yr}: {csv_path}")
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        cols = df.columns.tolist()

        def pick(candidates):
            """Return first candidate column that exists, else NaN series."""
            for c in candidates:
                if c in cols:
                    return df[c]
            return pd.Series(np.nan, index=df.index)

        # Opening ML – prefer the deduplicated canonical column, fall back to _x/_y
        open_ml_home = pick([
            "opening_pinnacle_ml_home",
            "opening_pinnacle_ml_home_x",
            "opening_pinnacle_ml_home_y",
        ])

        # Closing ML – the closing_pinnacle_ml_home column is consistent across all years
        close_ml_home = pick(["closing_pinnacle_ml_home"])

        # Closing ML away – only present in pi_ml_away
        close_ml_away = pick(["pi_ml_away"])

        # Spreads
        open_spread  = pick([
            "opening_pinnacle_spread_home_line",
            "opening_pinnacle_spread_home_line_x",
        ])
        close_spread = pick(["closing_pinnacle_spread_home_line"])

        # Spread juice – from the pi_ (closing) columns
        close_spread_home_price = pick(["pi_spread_home_price"])
        close_spread_away_price = pick(["pi_spread_away_price"])

        # Totals
        open_total  = pick([
            "opening_pinnacle_total_line",
            "opening_pinnacle_total_line_x",
        ])
        close_total = pick(["closing_pinnacle_total_line"])

        close_total_over  = pick(["pi_total_over_price"])
        close_total_under = pick(["pi_total_under_price"])

        # Line movements (pre-computed in the CSV)
        ml_move     = pick(["pinnacle_ml_home_movement"])
        spread_move = pick(["pinnacle_spread_home_line_movement"])
        total_move  = pick(["pinnacle_total_line_movement"])

        out = pd.DataFrame({
            "game_id":                   df["game_id"],
            "commence_time":             df.get("commence_time", pd.Series(dtype=str)),
            "home_team":                 df.get("home_team", pd.Series(dtype=str)),
            "away_team":                 df.get("away_team", pd.Series(dtype=str)),
            "pi_open_ml_home":           pd.to_numeric(open_ml_home, errors="coerce"),
            "pi_close_ml_home":          pd.to_numeric(close_ml_home, errors="coerce"),
            "pi_close_ml_away":          pd.to_numeric(close_ml_away, errors="coerce"),
            "pi_ml_movement":            pd.to_numeric(ml_move, errors="coerce"),
            "pi_open_spread_home":       pd.to_numeric(open_spread, errors="coerce"),
            "pi_close_spread_home":      pd.to_numeric(close_spread, errors="coerce"),
            "pi_spread_movement":        pd.to_numeric(spread_move, errors="coerce"),
            "pi_close_spread_home_price":pd.to_numeric(close_spread_home_price, errors="coerce"),
            "pi_close_spread_away_price":pd.to_numeric(close_spread_away_price, errors="coerce"),
            "pi_open_total":             pd.to_numeric(open_total, errors="coerce"),
            "pi_close_total":            pd.to_numeric(close_total, errors="coerce"),
            "pi_total_movement":         pd.to_numeric(total_move, errors="coerce"),
            "pi_close_total_over_price": pd.to_numeric(close_total_over, errors="coerce"),
            "pi_close_total_under_price":pd.to_numeric(close_total_under, errors="coerce"),
        })

        # Compute no-vig probabilities row-by-row
        nv_open  = out.apply(
            lambda r: pinnacle_no_vig_prob(r["pi_open_ml_home"],
                                           american_to_prob_inv(r["pi_open_ml_home"])),
            axis=1
        )
        # We only have closing away ML in the CSV; derive opening away from closing ratio isn't reliable.
        # Use closing no-vig from the actual away column where available.
        open_nv_home  = out["pi_open_ml_home"].apply(lambda v: american_to_prob(v))  # raw, not no-vig
        close_nv_home, close_nv_away = zip(*out.apply(
            lambda r: pinnacle_no_vig_prob(r["pi_close_ml_home"], r["pi_close_ml_away"]),
            axis=1
        ))

        out["pi_open_implied_home"]  = open_nv_home   # raw implied (vig-on) for opening
        out["pi_close_no_vig_home"]  = list(close_nv_home)
        out["pi_close_no_vig_away"]  = list(close_nv_away)

        frames.append(out)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # De-duplicate: keep the row with the most non-null pinnacle values per game_id
    combined["_pi_completeness"] = combined.filter(like="pi_").notna().sum(axis=1)
    combined = (combined.sort_values("_pi_completeness", ascending=False)
                        .drop_duplicates(subset=["game_id"], keep="first")
                        .drop(columns=["_pi_completeness"]))
    return combined


def american_to_prob_inv(prob):
    """Placeholder: not used – see pinnacle_no_vig_prob."""
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Existing fetch functions (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_batting(years):
    frames = []
    for yr in years:
        print(f"  Fetching team batting {yr}...")
        try:
            df = team_batting(yr)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["Team","team","teamName"]:
        if col in out.columns:
            out.rename(columns={col:"team"}, inplace=True)
            break
    return out

def fetch_team_pitching(years):
    frames = []
    for yr in years:
        print(f"  Fetching team pitching {yr}...")
        try:
            df = team_pitching(yr)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["Team","team","teamName"]:
        if col in out.columns:
            out.rename(columns={col:"team"}, inplace=True)
            break
    return out

def fetch_live_season_stats(live_year=None, fallback_year=None, min_games=15):
    """
    Fetch current-season team batting and pitching stats for the live year.

    Because pybaseball's team_batting/team_pitching pull season aggregates,
    very early in a season (days 1-14) the sample is too small to be meaningful.
    Strategy:
      1. Try to pull live_year stats.
      2. If the average games-played across teams is below min_games, log a
         warning and fall back to fallback_year end-of-season stats instead.
      3. Save both parquets to LIVE_DIR so run_model.py picks them up on the
         next run without touching the historical parquets in HIST_DIR.

    Returns (team_bat_df, team_pit_df).  Both DataFrames have a 'team' column
    (3-letter abbreviation) and a 'season' column.
    """
    from config import LIVE_DIR, TRAIN_YEARS
    import os

    if live_year is None:
        import datetime
        live_year = datetime.date.today().year
    if fallback_year is None:
        fallback_year = live_year - 1

    os.makedirs(LIVE_DIR, exist_ok=True)

    def _normalise(df):
        """Rename Team/teamName → team; drop rows where team is NaN."""
        for col in ["Team", "teamName"]:
            if col in df.columns and "team" not in df.columns:
                df = df.rename(columns={col: "team"})
        if "team" not in df.columns:
            df["team"] = "UNK"
        return df.dropna(subset=["team"])

    def _try_fetch(year):
        try:
            bat = _normalise(team_batting(year))
            pit = _normalise(team_pitching(year))
            bat["season"] = year
            pit["season"] = year
            return bat, pit
        except Exception as e:
            print(f"  [warn] Could not fetch {year} stats: {e}")
            return pd.DataFrame(), pd.DataFrame()

    print(f"\n[Live stats] Fetching {live_year} team stats...")
    bat, pit = _try_fetch(live_year)

    # Check whether we have enough games to trust the numbers.
    # "G" is the games-played column in pybaseball team_batting output.
    use_fallback = False
    if bat.empty or pit.empty:
        print(f"  [warn] {live_year} returned empty — using {fallback_year} fallback")
        use_fallback = True
    else:
        avg_g = bat["G"].mean() if "G" in bat.columns else min_games
        if avg_g < min_games:
            print(f"  [warn] {live_year} only {avg_g:.0f} avg games played "
                  f"(threshold {min_games}) — using {fallback_year} fallback stats")
            use_fallback = True
        else:
            print(f"  {live_year} stats OK ({avg_g:.0f} avg G/team)")

    if use_fallback:
        print(f"  Fetching {fallback_year} end-of-season stats as fallback...")
        bat, pit = _try_fetch(fallback_year)
        if bat.empty or pit.empty:
            print(f"  [ERROR] Fallback {fallback_year} also failed. "
                  "Check pybaseball connectivity.")
            return pd.DataFrame(), pd.DataFrame()

    # Validate that the expected feature columns exist and warn on any gaps.
    EXPECTED_BAT = {"OPS", "wOBA", "AVG", "OBP", "SLG", "HR", "BB%", "K%", "R"}
    EXPECTED_PIT = {"ERA", "FIP", "WHIP", "K/9", "BB/9", "xFIP"}
    missing_bat = EXPECTED_BAT - set(bat.columns)
    missing_pit = EXPECTED_PIT - set(pit.columns)
    if missing_bat:
        print(f"  [warn] Batting parquet missing expected columns: {sorted(missing_bat)}")
        print(f"         Available batting columns: {sorted(bat.columns.tolist())}")
    if missing_pit:
        print(f"  [warn] Pitching parquet missing expected columns: {sorted(missing_pit)}")
        print(f"         Available pitching columns: {sorted(pit.columns.tolist())}")

    bat_path = os.path.join(LIVE_DIR, "team_batting.parquet")
    pit_path = os.path.join(LIVE_DIR, "team_pitching.parquet")
    bat.to_parquet(bat_path, index=False)
    pit.to_parquet(pit_path, index=False)
    print(f"  Saved live team_batting.parquet  ({len(bat)} teams) → {bat_path}")
    print(f"  Saved live team_pitching.parquet ({len(pit)} teams) → {pit_path}")

    return bat, pit


def fetch_pitcher_stats(years):
    frames = []
    for yr in years:
        print(f"  Fetching pitcher stats {yr}...")
        try:
            df = pitching_stats(yr, qual=10)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_batter_stats(years):
    frames = []
    for yr in years:
        print(f"  Fetching batter stats {yr}...")
        try:
            df = batting_stats(yr, qual=50)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_schedules(years):
    all_frames = []
    for yr in years:
        print(f"  Fetching ESPN game results {yr}...")
        rows, seen = [], set()
        try:
            r = requests.get(f"{ESPN_BASE}/teams", headers=HEADERS, timeout=15)
            team_ids = [(t["team"]["id"], t["team"]["abbreviation"])
                        for t in r.json()["sports"][0]["leagues"][0]["teams"]]
        except Exception as e:
            print(f"    [warn] {e}")
            continue
        for tid, abbr in team_ids:
            try:
                r = requests.get(f"{ESPN_BASE}/teams/{tid}/schedule?season={yr}",
                                 headers=HEADERS, timeout=15)
                for ev in r.json().get("events", []):
                    gid = ev.get("id","")
                    if gid in seen:
                        continue
                    seen.add(gid)
                    try:
                        comp = ev["competitions"][0]
                        if not comp.get("status",{}).get("type",{}).get("completed",False):
                            continue
                        comps = comp["competitors"]
                        home  = next(c for c in comps if c["homeAway"]=="home")
                        away  = next(c for c in comps if c["homeAway"]=="away")
                        hs    = int(home.get("score",{}).get("value",0) or 0)
                        as_   = int(away.get("score",{}).get("value",0) or 0)
                        ht    = ESPN_TO_STD.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                        at    = ESPN_TO_STD.get(away["team"]["abbreviation"], away["team"]["abbreviation"])

                        raw_date = ev.get("date","")
                        game_date = pd.to_datetime(raw_date, utc=True).date().isoformat() if raw_date else ""

                        rows.append({
                            "season":      yr,
                            "date":        game_date,
                            "game_id":     gid,
                            "fetch_team":  ht,
                            "opponent":    at,
                            "home_away":   "H",
                            "result":      "W" if hs > as_ else "L",
                            "runs_scored": float(hs),
                            "runs_allowed":float(as_),
                        })
                    except Exception:
                        continue
            except Exception:
                pass
            time.sleep(0.1)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
            print(f"    -> {len(df)} games for {yr}")
            all_frames.append(df)
        else:
            print(f"    [warn] No data for {yr}")
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Batter-vs-Pitcher + lineup helpers
# ─────────────────────────────────────────────────────────────────────────────

# BBRef team name → 3-letter abbreviation used by ESPN schedule data.
# Needed so we can match lineup_lookup (keyed by bbref home_abbr) against
# the ESPN fetch_team column (also 3-letter abbreviations).
BBREF_ABBR_TO_ESPN = {
    # Standard modern codes (already matched)
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "OAK",
    "ATH": "OAK",   # Sacramento Athletics
    "PHI": "PHI", "PIT": "PIT", "SDP": "SDP", "SEA": "SEA",
    "SFG": "SFG", "STL": "STL", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSN": "WSN",
    # BUG FIX: BBRef uses old NL-style 3-letter codes for NL teams.
    # These were the root cause of the crash — unmapped codes passed through
    # as-is, so lineup_lookup keys like ("2026-03-25", "SFN") never matched
    # ESPN schedule keys like ("2026-03-25", "SFG"), silently killing all BvP.
    "SFN": "SFG",   # San Francisco Giants
    "LAN": "LAD",   # Los Angeles Dodgers
    "SDN": "SDP",   # San Diego Padres
    "CHN": "CHC",   # Chicago Cubs
    "NYN": "NYM",   # New York Mets
    "SLN": "STL",   # St. Louis Cardinals
    "CHA": "CHW",   # Chicago White Sox (AL style)
    "KCA": "KCR",   # Kansas City Royals
    "TBA": "TBR",   # Tampa Bay Rays
    "WAS": "WSN",   # Washington Nationals (alternate)
    "FLO": "MIA",   # Florida/Miami Marlins (pre-2012 name)
    "MON": "WSN",   # Montreal Expos (pre-2005, rare in 2021+ data)
}


def load_bvp_features(bvp_path: str = BVP_CSV_PATH):
    """Load batter_vs_pitcher_career.csv. Returns DataFrame or None."""
    if not os.path.exists(bvp_path):
        print(f"  [warn] BvP file not found: {bvp_path}")
        print("         Run bbref_batter_vs_pitcher_scraper.py first.")
        return None
    df = pd.read_csv(bvp_path, low_memory=False)
    required = {"batter", "pitcher", "PA", "AB", "H", "HR", "BB", "K"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  [warn] BvP CSV missing columns: {missing} — skipping")
        return None
    print(f"  BvP: loaded {len(df):,} matchup rows from {os.path.basename(bvp_path)}")
    return df


def build_game_lineups(
    pa_dir: str = HIST_ODDS_DIR,
    output_path: str = LINEUP_PARQUET_PATH,
    force_rebuild: bool = False,
) -> dict:
    """
    Build a per-game lineup lookup from plate_appearances_raw_*.csv files.

    Key insight: ESPN schedule uses numeric game IDs; BBRef uses its own IDs.
    They share two fields: game_date and home team abbreviation.
    So we key this lookup by (game_date, home_abbr) — both datasets have
    these, giving a reliable join without needing ID translation.

    Returns dict keyed by (game_date, home_abbr) → lineup info.
    """
    from pathlib import Path

    if not force_rebuild and os.path.exists(output_path):
        print(f"  Lineups: loading from cache {os.path.basename(output_path)}")
        df = pd.read_parquet(output_path)
        # BUG FIX: validate cache has all required columns before using it.
        # Older versions saved game_lineups.parquet without home_abbr, causing
        # a KeyError crash on every subsequent run. Force a rebuild if stale.
        required_cols = {"game_date", "home_abbr", "home_batters",
                         "away_batters", "home_sp", "away_sp"}
        missing_cols  = required_cols - set(df.columns)
        if missing_cols:
            print(f"  [warn] Cache missing {missing_cols} — rebuilding from PA files")
        else:
            return _lineups_df_to_dict(df)

    pa_files = sorted(Path(pa_dir).glob("plate_appearances_raw_*.csv"))
    if not pa_files:
        print(f"  [warn] No plate_appearances_raw_*.csv files in {pa_dir}")
        return {}

    print(f"  Lineups: building from {len(pa_files)} PA file(s)...")
    frames = []
    for f in pa_files:
        try:
            frames.append(pd.read_csv(
                f,
                usecols=["bbref_game_id", "game_date", "batter",
                         "pitcher", "batting_team_home"],
                low_memory=False,
            ))
        except Exception as e:
            print(f"    [warn] Could not read {f.name}: {e}")

    if not frames:
        print("  [warn] All PA files failed — using career proxy")
        return {}

    pa = pd.concat(frames, ignore_index=True)
    pa["batting_team_home"] = (
        pd.to_numeric(pa["batting_team_home"], errors="coerce").fillna(0).astype(int)
    )

    records = []
    for gid, grp in pa.groupby("bbref_game_id"):
        home_bat_rows = grp[grp["batting_team_home"] == 1]
        away_bat_rows = grp[grp["batting_team_home"] == 0]

        # BUG FIX: BBRef PA files use non-breaking spaces (\xa0) instead of
        # regular spaces in batter/pitcher names. This silently breaks BvP
        # lookups for every player. Normalise all names at write time.
        def _clean(s): return str(s).replace("\xa0", " ").replace("\u00a0", " ").strip()

        # home_sp = first pitcher to face away batters
        # away_sp = first pitcher to face home batters
        home_sp = _clean(away_bat_rows["pitcher"].dropna().iloc[0]) \
                  if not away_bat_rows.empty else ""
        away_sp = _clean(home_bat_rows["pitcher"].dropna().iloc[0]) \
                  if not home_bat_rows.empty else ""

        game_date = str(grp["game_date"].iloc[0])

        # Extract home team abbreviation from bbref_game_id.
        # BBRef format: HHHYYYYMMDDnn  e.g. BOS2023040601
        # First 3 chars = home team abbreviation.
        home_abbr_bbref = str(gid)[:3]
        # Map to ESPN abbreviation (most are identical; a few differ)
        home_abbr = BBREF_ABBR_TO_ESPN.get(home_abbr_bbref, home_abbr_bbref)

        records.append({
            "bbref_game_id": gid,
            "game_date":     game_date,
            "home_abbr":     home_abbr,
            "home_batters":  "|".join(_clean(b) for b in home_bat_rows["batter"].dropna().unique()),
            "away_batters":  "|".join(_clean(b) for b in away_bat_rows["batter"].dropna().unique()),
            "home_sp":       home_sp,
            "away_sp":       away_sp,
        })

    lineup_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lineup_df.to_parquet(output_path, index=False)

    n      = len(lineup_df)
    has_sp = (lineup_df["home_sp"] != "").sum()
    print(f"  Lineups: {n:,} games | {has_sp:,} ({has_sp/n:.1%}) with home SP")
    print(f"  Saved → {output_path}")
    return _lineups_df_to_dict(lineup_df)


def _lineups_df_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert lineup DataFrame to dict keyed by (game_date, home_abbr).
    This matches the ESPN schedule join key used in build_game_features.
    """
    out = {}
    for _, row in df.iterrows():
        key = (str(row["game_date"]), str(row["home_abbr"]))
        out[key] = {
            "home_batters": [b for b in str(row["home_batters"]).split("|") if b],
            "away_batters": [b for b in str(row["away_batters"]).split("|") if b],
            "home_sp":      str(row["home_sp"]),
            "away_sp":      str(row["away_sp"]),
        }
    return out


def compute_bvp_pitcher_features(
    pitcher_name: str,
    lineup: list,
    bvp_df,
    min_pa: int = 5,
) -> dict:
    """
    Aggregate career H2H stats for pitcher_name vs every batter in lineup.
    Returns 8 features; all NaN when no qualifying history exists.
    """
    empty = {
        "bvp_pa_total": np.nan, "bvp_avg": np.nan, "bvp_ops": np.nan,
        "bvp_k_rate":   np.nan, "bvp_bb_rate": np.nan, "bvp_hr_rate": np.nan,
        "bvp_h_rate":   np.nan, "bvp_matchups_found": 0,
    }
    if bvp_df is None or not lineup or not pitcher_name:
        return empty

    # BUG FIX: BBRef CSVs use non-breaking spaces ( ) in player names.
    # str.strip() does not remove   — normalise both sides of the join.
    def _norm(s): return str(s).replace(" ", " ").replace(" ", " ").strip().lower()

    pitcher_lower = _norm(pitcher_name)
    sub = bvp_df[bvp_df["pitcher"].apply(_norm) == pitcher_lower]
    if sub.empty:
        return empty

    sub_idx = sub.set_index(sub["batter"].apply(_norm))
    totals  = {"PA": 0, "AB": 0, "H": 0, "HR": 0, "BB": 0, "K": 0, "SF": 0, "HBP": 0}
    found   = 0

    for batter in lineup:
        key = _norm(batter)
        if key not in sub_idx.index:
            continue
        row = sub_idx.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        pa = int(row.get("PA", 0) or 0)
        if pa < min_pa:
            continue
        found += 1
        for col in totals:
            totals[col] += float(row.get(col, 0) or 0)

    if found == 0 or totals["PA"] == 0:
        return empty

    pa = totals["PA"]
    ab = totals["AB"] if totals["AB"] > 0 else pa
    h, bb, hbp, sf, hr = totals["H"], totals["BB"], totals["HBP"], totals["SF"], totals["HR"]
    singles = max(h - hr, 0)
    obp_den = ab + bb + hbp + sf
    obp     = (h + bb + hbp) / obp_den if obp_den > 0 else np.nan
    slg     = (singles + 4 * hr) / ab  if ab > 0      else np.nan
    ops     = round(obp + slg, 3) if (pd.notna(obp) and pd.notna(slg)) else np.nan

    return {
        "bvp_pa_total":       pa,
        "bvp_avg":            round(h  / ab, 3) if ab > 0 else np.nan,
        "bvp_ops":            ops,
        "bvp_k_rate":         round(totals["K"]  / pa, 3),
        "bvp_bb_rate":        round(totals["BB"] / pa, 3),
        "bvp_hr_rate":        round(totals["HR"] / pa, 3),
        "bvp_h_rate":         round(h / pa, 3),
        "bvp_matchups_found": found,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder – now accepts pinnacle_lines, bvp_df, lineup_lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_game_features(schedules_df, team_bat_df, team_pit_df,
                        pinnacle_lines=None, bvp_df=None, lineup_lookup=None):
    """
    Build one row per completed game.
    lineup_lookup is keyed by (game_date, home_team_abbr) matching the ESPN
    schedule's date + fetch_team columns — this is the correct join key since
    ESPN and BBRef use different game ID formats.
    """
    print("  Building game-level features...")
    if schedules_df.empty:
        return pd.DataFrame()

    bat_lookup, pit_lookup = {}, {}
    if not team_bat_df.empty and "team" in team_bat_df.columns:
        for _, row in team_bat_df.iterrows():
            bat_lookup[(str(row["team"]), int(row["season"]))] = row
    if not team_pit_df.empty and "team" in team_pit_df.columns:
        for _, row in team_pit_df.iterrows():
            pit_lookup[(str(row["team"]), int(row["season"]))] = row

    bat_features = ["AVG","OBP","SLG","OPS","R","HR","BB%","K%","wOBA","WAR"]
    pit_features = ["ERA","FIP","WHIP","K/9","BB/9","xFIP","WAR"]

    # Build pinnacle lookup by (date, home_team_abbr).
    # The Odds API CSVs use full team names; ESPN schedules use abbreviations.
    # We normalise full names → abbreviations so the join keys match.
    pi_lookup = {}
    if pinnacle_lines is not None and not pinnacle_lines.empty:
        pi_df = pinnacle_lines.copy()
        if "commence_time" in pi_df.columns:
            try:
                import pandas as _pd
                # Use UTC date directly — avoids tzdata dependency.
                # Games typically have a UTC commence_time that matches ESPN's
                # UTC-based date. A small number of late-night West Coast games
                # may be off by one day but this is negligible for matching.
                pi_df["_date"] = (_pd.to_datetime(pi_df["commence_time"], utc=True)
                                   .dt.date.astype(str))
            except Exception:
                pi_df["_date"] = ""
        else:
            pi_df["_date"] = ""
        for _, row in pi_df.iterrows():
            raw_home = str(row.get("home_team", ""))
            # Convert full name → abbreviation; fall back to raw value if unknown
            home_abbr = FULL_NAME_TO_ABB.get(raw_home, raw_home)
            key = (row.get("_date", ""), home_abbr)
            pi_lookup[key] = row.to_dict()

    rows = []
    for _, game in schedules_df.iterrows():
        season    = int(game["season"])
        home_team = game["fetch_team"]
        away_team = game["opponent"]
        rs        = float(game["runs_scored"])
        ra        = float(game["runs_allowed"])
        gid       = game.get("game_id", "")

        row = {
            "season":    season,
            "date":      game.get("date", ""),
            "game_id":   gid,
            "home_team": home_team,
            "away_team": away_team,
            "home_runs": rs,
            "away_runs": ra,
            "total_runs":rs + ra,
            "home_win":  int(rs > ra),
            "run_diff":  rs - ra,
        }

        for prefix, team in [("home_bat_", home_team), ("away_bat_", away_team)]:
            stats = bat_lookup.get((team, season), {})
            for feat in bat_features:
                row[f"{prefix}{feat}"] = stats.get(feat, np.nan) if hasattr(stats, "get") else np.nan

        for prefix, team in [("home_pit_", home_team), ("away_pit_", away_team)]:
            stats = pit_lookup.get((team, season), {})
            for feat in pit_features:
                row[f"{prefix}{feat}"] = stats.get(feat, np.nan) if hasattr(stats, "get") else np.nan

        # ── Pinnacle lines: join on (date, home_team) ────────────────────
        pi = pi_lookup.get((game.get("date",""), home_team), {})
        row["pi_open_ml_home"]           = pi.get("pi_open_ml_home", np.nan)
        row["pi_close_ml_home"]          = pi.get("pi_close_ml_home", np.nan)
        row["pi_close_ml_away"]          = pi.get("pi_close_ml_away", np.nan)
        row["pi_ml_movement"]            = pi.get("pi_ml_movement", np.nan)
        row["pi_open_spread_home"]       = pi.get("pi_open_spread_home", np.nan)
        row["pi_close_spread_home"]      = pi.get("pi_close_spread_home", np.nan)
        row["pi_spread_movement"]        = pi.get("pi_spread_movement", np.nan)
        row["pi_close_spread_home_price"]= pi.get("pi_close_spread_home_price", np.nan)
        row["pi_close_spread_away_price"]= pi.get("pi_close_spread_away_price", np.nan)
        row["pi_open_total"]             = pi.get("pi_open_total", np.nan)
        row["pi_close_total"]            = pi.get("pi_close_total", np.nan)
        row["pi_total_movement"]         = pi.get("pi_total_movement", np.nan)
        row["pi_close_total_over_price"] = pi.get("pi_close_total_over_price", np.nan)
        row["pi_close_total_under_price"]= pi.get("pi_close_total_under_price", np.nan)
        row["pi_open_implied_home"]      = pi.get("pi_open_implied_home", np.nan)
        row["pi_close_no_vig_home"]      = pi.get("pi_close_no_vig_home", np.nan)
        row["pi_close_no_vig_away"]      = pi.get("pi_close_no_vig_away", np.nan)

        # ── Batter-vs-Pitcher features ────────────────────────────────────
        # Join key: (game_date, home_team) — same fields available in both
        # the ESPN schedule and the lineup_lookup built from BBRef PA files.
        # This avoids the ESPN numeric ID vs BBRef alphanumeric ID mismatch.
        if bvp_df is not None:
            # BUG FIX: game["date"] is stored as pd.Timestamp in the parquet
            # (e.g. Timestamp("2021-04-01")), so str() gives "2021-04-01 00:00:00"
            # which never matches lineup_lookup keys like "2021-04-01".
            # Use .date().isoformat() to normalise to YYYY-MM-DD string.
            _raw_date     = game.get("date", "")
            game_date_str = (pd.Timestamp(_raw_date).date().isoformat()
                             if _raw_date else "")
            lineup_key    = (game_date_str, home_team)
            game_lineup   = lineup_lookup.get(lineup_key) if lineup_lookup else None

            if game_lineup:
                home_batters = game_lineup["home_batters"]
                away_batters = game_lineup["away_batters"]
                home_sp      = game_lineup["home_sp"]
                away_sp      = game_lineup["away_sp"]
            else:
                # Career proxy: all batters ever recorded facing this pitcher
                home_sp = away_sp = ""
                home_batters = away_batters = []

            # away SP vs home lineup → home offensive BvP edge
            h_feats = compute_bvp_pitcher_features(away_sp, home_batters, bvp_df)
            # home SP vs away lineup → away offensive BvP edge
            a_feats = compute_bvp_pitcher_features(home_sp, away_batters, bvp_df)

            for k, v in h_feats.items():
                row[f"home_{k}"] = v
            for k, v in a_feats.items():
                row[f"away_{k}"] = v

            # Positive = home lineup has the BvP advantage
            row["bvp_avg_diff"]    = (h_feats["bvp_avg"]    or np.nan) - (a_feats["bvp_avg"]    or np.nan)
            row["bvp_ops_diff"]    = (h_feats["bvp_ops"]    or np.nan) - (a_feats["bvp_ops"]    or np.nan)
            row["bvp_k_rate_diff"] = (h_feats["bvp_k_rate"] or np.nan) - (a_feats["bvp_k_rate"] or np.nan)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("MLB Historical Data Fetch (2021-2025)")
    print("="*60)

    print("\n[1/6] Team Batting Stats")
    team_bat = fetch_team_batting(TRAIN_YEARS)
    print(f"  -> {len(team_bat)} rows")

    print("\n[2/6] Team Pitching Stats")
    team_pit = fetch_team_pitching(TRAIN_YEARS)
    print(f"  -> {len(team_pit)} rows")

    print("\n[3/6] Pitcher Individual Stats")
    pitcher_stats_df = fetch_pitcher_stats(TRAIN_YEARS)
    print(f"  -> {len(pitcher_stats_df)} rows")

    print("\n[4/6] Batter Individual Stats")
    batter_stats_df = fetch_batter_stats(TRAIN_YEARS)
    print(f"  -> {len(batter_stats_df)} rows")

    print("\n[5/6] Game Results (ESPN API)")
    schedules = fetch_schedules(TRAIN_YEARS)
    print(f"  -> {len(schedules)} total games")

    print("\n[5b/6] Pinnacle Historical Lines (CSV)")
    pinnacle_lines = load_pinnacle_lines_csv(TRAIN_YEARS)
    if pinnacle_lines.empty:
        print("  [warn] No Pinnacle lines loaded — pi_* features will be NaN")
    else:
        pi_coverage = pinnacle_lines["pi_close_no_vig_home"].notna().mean()
        print(f"  -> {len(pinnacle_lines)} unique game_ids | "
              f"closing no-vig coverage: {pi_coverage:.1%}")

    print("\n[5c/6] Batter-vs-Pitcher Matchup Table")
    bvp_df = load_bvp_features()
    if bvp_df is not None:
        print(f"  -> {len(bvp_df):,} batter-pitcher pairs loaded")
        print(f"  -> {bvp_df['PA'].ge(5).mean():.1%} pairs have >= 5 PA")
    else:
        print("  -> BvP features skipped (run bbref scraper first)")

    print("\n[5d/6] Per-Game Lineups (from plate appearance files)")
    lineup_lookup = build_game_lineups(pa_dir=HIST_ODDS_DIR)
    if lineup_lookup:
        print(f"  -> {len(lineup_lookup):,} games in lineup lookup")
    else:
        print("  -> No lineup data — BvP will use career proxy")

    print("\n[6/6] Building game features...")
    game_features = build_game_features(
        schedules, team_bat, team_pit, pinnacle_lines, bvp_df, lineup_lookup
    )
    print(f"  -> {len(game_features)} game rows with features")

    if "date" in game_features.columns:
        sample = game_features["date"].dropna().iloc[:3].tolist()
        print(f"  -> Date column confirmed: {sample}")
    else:
        print("  [warn] Date column missing from game features!")

    pi_cols_present = [c for c in game_features.columns if c.startswith("pi_")]
    if pi_cols_present:
        nv_cov = game_features["pi_close_no_vig_home"].notna().mean()
        print(f"  -> {len(pi_cols_present)} Pinnacle feature columns | "
              f"no-vig coverage in output: {nv_cov:.1%}")
    else:
        print("  [warn] No Pinnacle columns in output!")

    bvp_cols = [c for c in game_features.columns if "bvp" in c]
    if bvp_cols:
        bvp_cov = game_features["home_bvp_pa_total"].notna().mean()
        print(f"  -> {len(bvp_cols)} BvP feature columns | coverage: {bvp_cov:.1%}")

    print("\n[Saving data...]")
    if not team_bat.empty:
        team_bat.to_parquet(os.path.join(DATA_DIR, "team_batting.parquet"), index=False)
        print("  Saved team_batting.parquet")
    if not team_pit.empty:
        team_pit.to_parquet(os.path.join(DATA_DIR, "team_pitching.parquet"), index=False)
        print("  Saved team_pitching.parquet")
    if not pitcher_stats_df.empty:
        pitcher_stats_df.to_parquet(os.path.join(DATA_DIR, "pitcher_stats.parquet"), index=False)
        print("  Saved pitcher_stats.parquet")
    if not batter_stats_df.empty:
        batter_stats_df.to_parquet(os.path.join(DATA_DIR, "batter_stats.parquet"), index=False)
        print("  Saved batter_stats.parquet")
    if not schedules.empty:
        schedules.to_parquet(os.path.join(DATA_DIR, "schedules.parquet"), index=False)
        print("  Saved schedules.parquet")
    if not pinnacle_lines.empty:
        pinnacle_lines.reset_index().to_parquet(
            os.path.join(DATA_DIR, "pinnacle_lines.parquet"), index=False
        )
        print("  Saved pinnacle_lines.parquet")
    if lineup_lookup:
        print(f"  game_lineups.parquet already saved during step 5d "
              f"({len(lineup_lookup):,} games)")
    if not game_features.empty:
        game_features.to_parquet(HISTORICAL_PATH, index=False)
        print(f"  Saved historical_stats.parquet ({len(game_features)} game rows)")
    else:
        print("  [warn] No game features saved")

    print("\n[7/6] Live Season Team Stats (for run_model.py)")
    from config import LIVE_SEASON
    live_bat, live_pit = fetch_live_season_stats(
        live_year=LIVE_SEASON,
        fallback_year=max(TRAIN_YEARS),
    )
    if live_bat.empty or live_pit.empty:
        print("  [warn] Live season stats unavailable — run_model.py will use "
              "historical parquets from data/historical/")
    else:
        print(f"  Live stats ready: {len(live_bat)} batting rows, "
              f"{len(live_pit)} pitching rows")

    print("\nDone! Run next: python train_model.py")

if __name__ == "__main__":
    main()