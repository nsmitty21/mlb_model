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
# Feature builder – now accepts pinnacle_lines lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_game_features(schedules_df, team_bat_df, team_pit_df, pinnacle_lines=None):
    """
    Build one row per completed game.  If pinnacle_lines (DataFrame indexed by
    game_id) is provided, all pi_* columns are joined in.
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

    print("\n[6/6] Building game features...")
    game_features = build_game_features(schedules, team_bat, team_pit, pinnacle_lines)
    print(f"  -> {len(game_features)} game rows with features")

    # Verify date and pi columns
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