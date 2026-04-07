"""
player_props.py — MLB Player Props + Beat the Streak
======================================================
Three features built from plate_appearances_raw_*.csv data:

  1. BEAT THE STREAK — picks the single best batter to get a hit today,
     tracking a running streak (trying to beat DiMaggio's 56).

  2. HOME RUN PROPS — ranks today's lineup vs opposing pitchers for HR
     likelihood, combining batter HR rate, pitcher HR-allowed rate, park
     factor, and career BvP HR data.

  3. NRFI — No Run First Inning picks. Scores each game on pitcher quality,
     team run-scoring rate, and park factor. Recommends NRFI or YRFI.

All three read from your existing PA CSV files and today_lines.json.
Output goes to data/live/player_props_today.json and a Writeups txt file.

Usage (standalone):
    python player_props.py

Called automatically by run_today.py after picks are saved.
"""

import json, re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import HIST_DIR, LIVE_DIR, BASE_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────
PA_DIR          = Path(HIST_DIR) / "odds"   # plate_appearances_raw_*.csv live here
LINES_PATH      = Path(LIVE_DIR) / "today_lines.json"
PROPS_OUT       = Path(LIVE_DIR) / "player_props_today.json"
NRFI_BACKTEST   = PA_DIR / "nrfi_pitcher_backtest.csv"   # built by nrfi_backtest.py

NBA_OUTPUT      = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\ML and Spread"
WRITEUPS_DIR    = Path(NBA_OUTPUT) / "MLB" / "Writeups"
LOCAL_WRITEUPS  = Path(BASE_DIR) / "Writeups"
BTS_TRACKER     = Path(BASE_DIR) / "data" / "live" / "bts_tracker.json"

# ── Park factors (from features.py — kept in sync) ────────────────────────────
PARK_FACTORS = {
    "Colorado Rockies": 1.18, "Boston Red Sox": 1.08, "Cincinnati Reds": 1.07,
    "Philadelphia Phillies": 1.05, "Chicago Cubs": 1.04, "Houston Astros": 1.03,
    "Texas Rangers": 1.03, "Atlanta Braves": 1.02, "New York Yankees": 1.01,
    "Toronto Blue Jays": 1.00, "Los Angeles Dodgers": 1.00, "San Diego Padres": 0.99,
    "New York Mets": 0.99, "St. Louis Cardinals": 0.99, "Baltimore Orioles": 0.99,
    "Minnesota Twins": 0.98, "Detroit Tigers": 0.98, "Kansas City Royals": 0.97,
    "Chicago White Sox": 0.97, "Cleveland Guardians": 0.97, "Seattle Mariners": 0.96,
    "Pittsburgh Pirates": 0.96, "Tampa Bay Rays": 0.96, "Oakland Athletics": 0.96,
    "Athletics": 0.96,
    "Miami Marlins": 0.95, "Washington Nationals": 0.95, "San Francisco Giants": 0.94,
    "Los Angeles Angels": 0.97, "Milwaukee Brewers": 0.98, "Arizona Diamondbacks": 1.01,
}


# ── BBRef home-team abbreviations (first 3 chars of bbref_game_id) ────────────
# bbref_game_id encodes the home team: e.g. "SFN202603250" → home = SFN.
# This lets us derive the exact team name for every batter without a roster,
# using only bbref_game_id + batting_team_home (0/1) from the PA files.
BBREF_HOME_ABBR_TO_FULL = {
    "ARI": "Arizona Diamondbacks",   "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",      "BOS": "Boston Red Sox",
    "CHN": "Chicago Cubs",           "CHA": "Chicago White Sox",
    "CIN": "Cincinnati Reds",        "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",       "DET": "Detroit Tigers",
    "HOU": "Houston Astros",         "KCA": "Kansas City Royals",
    "ANA": "Los Angeles Angels",     "LAN": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",          "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",        "NYN": "New York Mets",
    "NYA": "New York Yankees",       "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",  "PIT": "Pittsburgh Pirates",
    "SDN": "San Diego Padres",       "SFN": "San Francisco Giants",
    "SEA": "Seattle Mariners",       "SLN": "St. Louis Cardinals",
    "TBA": "Tampa Bay Rays",         "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",      "WAS": "Washington Nationals",
    "ATH": "Athletics",              # Sacramento relocation 2026
}

# Reverse map: full team name → BBRef home abbr (for cross-referencing today_lines)
FULL_TO_BBREF_ABBR = {v: k for k, v in BBREF_HOME_ABBR_TO_FULL.items()}


def build_batter_team_map(pa_slice: pd.DataFrame, games: list) -> dict:
    """
    Build a precise batter → full_team_name mapping from PA data.

    Strategy (two passes):

    Pass 1 — PA data only (works for all dates, including historical):
      Home batters (batting_team_home == 1):
        team = BBREF_HOME_ABBR_TO_FULL[bbref_game_id[:3]]

      Away batters (batting_team_home == 0):
        Within each game, the pitchers facing away batters are from the HOME
        team, and the pitchers facing home batters are from the AWAY team.
        We build a pitcher → team lookup from the home batters' PA rows
        (pitcher facing home batters = away team's pitcher → away team).
        Then for each away batter, look up their pitcher's team.

    Pass 2 — today_lines.json (fills any remaining gaps using home_abbr match):
        home_abbr → away_full from the games list.

    This means away batters are correctly mapped even when the game date in
    the PA slice doesn't appear in today_lines.json (e.g. historical dates).
    """
    # ── Pass 1a: home batters — direct from bbref_game_id ────────────────────
    batter_to_team: dict[str, str] = {}
    for _, row in pa_slice.iterrows():
        if row["batting_team_home"] == 1:
            home_abbr = str(row["bbref_game_id"])[:3].upper()
            home_full = BBREF_HOME_ABBR_TO_FULL.get(home_abbr, "")
            if home_full:
                batter_to_team[row["batter"]] = home_full

    # ── Pass 1b: away batters — derive via pitcher→team within each game ──────
    # For each game: pitchers who pitched to HOME batters are the AWAY team's pitchers.
    # So: pitcher → game_id, and game_id[:3] gives us the home team.
    # The pitcher's team = the AWAY team = the team we need for away batters.
    #
    # Build: game_id → away_team by finding which pitchers faced home batters
    # then looking those pitchers up against a pitcher-to-known-team table.
    #
    # Simpler direct approach: within each game, build
    #   game_id → set of (pitcher, faced_home_batters)
    # A pitcher who faces home batters (batting_team_home=1) is an AWAY pitcher.
    # But we still need the away team NAME, not just the pitcher name.
    #
    # Actual solution: group each game's away-batting rows by game_id.
    # For each game, look at the pitchers who face AWAY batters (batting_team_home=0).
    # Those pitchers are HOME team pitchers. Cross-ref against home_batters in the
    # same game to find the away team from home_abbr, then assign to away batters.
    #
    # This is circular — we know home from bbref_game_id but need away team name.
    # Use today_lines.json for today's date; for historical, build pitcher→team
    # from the home-side evidence (pitcher appears as home pitcher → away team
    # is whoever was batting when batting_team_home=0 in that game).
    #
    # PRAGMATIC: build game_id → away_team from all games where we DO know the
    # away team (either from today_lines or from home-batter evidence).
    # For today's games, today_lines.json is the source.

    # ── Pass 2: fill gaps from today_lines.json ───────────────────────────────
    home_abbr_to_away: dict[str, str] = {}
    for g in games:
        ht = g.get("home_team", "")
        at = g.get("away_team", "")
        ha = FULL_TO_BBREF_ABBR.get(ht, ht[:3].upper())
        if ha and at:
            home_abbr_to_away[ha] = at

    # Also build from game-level pitcher evidence within the PA slice:
    # For each game, pitchers who appear when batting_team_home=0 are home pitchers.
    # We know the home team → so the away team is whichever team ISN'T the home team.
    # We can't get the away team name from this alone without a roster.
    # BUT: if any home batter in the same game already has a mapped team, the away
    # team is whoever home batters' team's OPPONENT is in today_lines.
    # → Fall back: for games not in today_lines, derive away team from the
    #   PITCHER NAME itself, cross-referencing pitcher → team built from historical
    #   rows where we know the home/away assignment.

    # Build pitcher → their team (the team they PITCH for = opposite of batter's team):
    # A pitcher in row with batting_team_home=1 → pitcher is on the AWAY team.
    # home_abbr of that game is the HOME team, and we know it → away team for pitcher.
    pitcher_to_team: dict[str, str] = {}
    for _, row in pa_slice.iterrows():
        pitcher   = row["pitcher"]
        home_abbr = str(row["bbref_game_id"])[:3].upper()
        home_full = BBREF_HOME_ABBR_TO_FULL.get(home_abbr, "")
        if not home_full:
            continue
        if row["batting_team_home"] == 1:
            # Pitcher is pitching to HOME batters → pitcher is on the AWAY team
            # Away team = home_abbr_to_away.get(home_abbr) if known
            away = home_abbr_to_away.get(home_abbr, "")
            if away:
                pitcher_to_team[pitcher] = away
        else:
            # Pitcher is pitching to AWAY batters → pitcher is on the HOME team
            pitcher_to_team[pitcher] = home_full

    # Now assign away batters using their pitcher's team (the away team's pitcher
    # is the home team's pitcher, but the AWAY batter faces the HOME pitcher —
    # so the away batter's pitcher is from the home team, and the away batter
    # belongs to the away team = home_abbr_to_away[home_abbr]).
    for _, row in pa_slice.iterrows():
        batter    = row["batter"]
        if batter in batter_to_team:
            continue   # already mapped in pass 1a
        if row["batting_team_home"] != 0:
            continue
        home_abbr = str(row["bbref_game_id"])[:3].upper()
        # Try today_lines lookup first (most reliable for today's games)
        away_full = home_abbr_to_away.get(home_abbr, "")
        if away_full:
            batter_to_team[batter] = away_full

    return batter_to_team


def fetch_espn_starters_today(target_date: str | None = None) -> dict:
    """
    Fetch today's starting pitchers from ESPN public API.
    Returns dict: {home_team_full: (home_sp, away_sp), ...}
    Used as fallback when picks_today.json doesn't exist yet.
    """
    import requests
    from datetime import date as _date

    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
    if target_date is None:
        target_date = _date.today().strftime("%Y%m%d")
    else:
        target_date = target_date.replace("-", "")

    try:
        r = requests.get(
            f"{ESPN_BASE}/scoreboard",
            params={"dates": target_date, "limit": 50},
            timeout=10,
        )
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"  [warn] ESPN starters fetch failed: {e}")
        return {}

    result = {}
    for ev in events:
        try:
            comps = ev.get("competitions", [{}])[0]
            competitors = comps.get("competitors", [])
            home_c = next((c for c in competitors if c["homeAway"] == "home"), None)
            away_c = next((c for c in competitors if c["homeAway"] == "away"), None)
            if not home_c or not away_c:
                continue
            hn = home_c["team"].get("displayName", "")
            an = away_c["team"].get("displayName", "")
            home_sp = away_sp = ""
            for comp in competitors:
                for p in comp.get("probables", []):
                    name = p.get("athlete", {}).get("fullName", "")
                    if comp["homeAway"] == "home":
                        home_sp = name
                    else:
                        away_sp = name
            result[hn] = {"home_sp": home_sp, "away_sp": away_sp,
                          "home_team": hn, "away_team": an}
        except Exception:
            continue

    n = sum(1 for v in result.values() if v["home_sp"] or v["away_sp"])
    print(f"  ESPN starters: {n}/{len(result)} games have SP data")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pa_data() -> pd.DataFrame:
    """
    Load and concatenate all plate_appearances_raw_*.csv files from PA_DIR.
    Returns a single DataFrame with all historical + current season PAs.
    """
    pa_files = sorted(PA_DIR.glob("plate_appearances_raw_*.csv"))
    if not pa_files:
        raise FileNotFoundError(
            f"No plate_appearances_raw_*.csv files found in {PA_DIR}\n"
            f"Expected path: {PA_DIR}"
        )

    dfs = []
    for p in pa_files:
        try:
            df = pd.read_csv(p, low_memory=False)
            # Normalize non-breaking spaces in batter/pitcher names (BBRef quirk)
            for col in ("batter", "pitcher"):
                if col in df.columns:
                    df[col] = df[col].str.replace("\xa0", " ", regex=False).str.strip()
            dfs.append(df)
        except Exception as e:
            print(f"  [warn] Could not read {p.name}: {e}")

    if not dfs:
        raise ValueError("All PA files failed to load.")

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    print(f"  PA data: {len(combined):,} rows from {len(pa_files)} file(s) "
          f"({combined['game_date'].min().date()} → {combined['game_date'].max().date()})")
    return combined


def load_today_games() -> list:
    """
    Load today_lines.json and return the games list, with SP names populated.

    SP name priority (highest to lowest):
      1. picks_today.json  — written by run_today.py, has ESPN-fetched SPs
      2. ESPN API          — fetched directly as fallback (standalone runs)
      3. Empty string      — unknown, props functions use league-average rates
    """
    if not LINES_PATH.exists():
        print("  [warn] today_lines.json not found — run pull_lines.py first")
        return []
    with open(LINES_PATH) as f:
        games = json.load(f)

    # Initialise home_sp/away_sp keys so downstream code can always .get() them
    for g in games:
        g.setdefault("home_sp", "")
        g.setdefault("away_sp", "")

    # ── Source 1: picks_today.json (run_today.py writes ESPN starters here) ──
    picks_path = Path(LIVE_DIR) / "picks_today.json"
    if picks_path.exists():
        try:
            with open(picks_path) as f:
                picks_data = json.load(f)
            sp_lookup = {
                f"{pg.get('home_team','')}|{pg.get('away_team','')}": pg
                for pg in picks_data.get("games", [])
            }
            for g in games:
                key = f"{g.get('home_team','')}|{g.get('away_team','')}"
                pg  = sp_lookup.get(key)
                if pg:
                    if not g["home_sp"] and pg.get("home_sp"):
                        g["home_sp"] = pg["home_sp"]
                    if not g["away_sp"] and pg.get("away_sp"):
                        g["away_sp"] = pg["away_sp"]
        except Exception as e:
            print(f"  [warn] picks_today.json SP merge failed: {e}")

    # ── Source 2: ESPN API — for any games still missing SPs ─────────────────
    missing_sp = [g for g in games if not g["home_sp"] and not g["away_sp"]]
    if missing_sp:
        try:
            espn_starters = fetch_espn_starters_today()
            for g in games:
                ht  = g.get("home_team", "")
                esp = espn_starters.get(ht)
                if esp:
                    if not g["home_sp"] and esp.get("home_sp"):
                        g["home_sp"] = esp["home_sp"]
                    if not g["away_sp"] and esp.get("away_sp"):
                        g["away_sp"] = esp["away_sp"]
        except Exception as e:
            print(f"  [warn] ESPN starter fallback failed: {e}")

    n_with_sp   = sum(1 for g in games if g.get("home_sp") or g.get("away_sp"))
    n_missing   = sum(1 for g in games if not g.get("home_sp") or not g.get("away_sp"))
    print(f"  SPs loaded: {n_with_sp}/{len(games)} games have starter data")

    if n_missing > 0:
        missing_games = [
            f"{g.get('away_team','?')} @ {g.get('home_team','?')}"
            f" (home_sp={g.get('home_sp','') or 'MISSING'}, "
            f"away_sp={g.get('away_sp','') or 'MISSING'})"
            for g in games
            if not g.get("home_sp") or not g.get("away_sp")
        ]
        msg = (
            f"\n[ERROR] Starting pitchers not announced for {n_missing} game(s):\n"
            + "\n".join(f"  {g}" for g in missing_games)
            + "\n\nStarters should be available by game day. "
              "Check ESPN or wait for the official lineup announcement, "
              "then re-run pull_lines.py and run_today.py before player_props.py."
        )
        raise ValueError(msg)

    return games


def _parse_score(s) -> tuple[int, int]:
    """Parse 'away-home' score string → (away_runs, home_runs)."""
    try:
        parts = str(s).split("-")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


# ═══════════════════════════════════════════════════════════════════════════════
# BATTER + PITCHER STATS FROM PA DATA
# ═══════════════════════════════════════════════════════════════════════════════

def build_batter_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Per-batter season stats: PA, AB, H, HR, BB, K, AVG, OBP, SLG, OPS,
    HR/PA rate, hit rate (H/PA), current hit streak.
    """
    g = pa.groupby("batter").agg(
        PA=("PA", "sum"),
        AB=("AB", "sum"),
        H=("H", "sum"),
        singles=("1B", "sum"),
        doubles=("2B", "sum"),
        triples=("3B", "sum"),
        HR=("HR", "sum"),
        BB=("BB", "sum"),
        HBP=("HBP", "sum"),
        K=("K", "sum"),
        SF=("SF", "sum"),
    ).reset_index()

    g["AVG"]   = (g["H"]  / g["AB"].replace(0, np.nan)).round(3)
    g["OBP"]   = ((g["H"] + g["BB"] + g["HBP"]) /
                  (g["AB"] + g["BB"] + g["HBP"] + g["SF"]).replace(0, np.nan)).round(3)
    tb         = g["singles"] + 2*g["doubles"] + 3*g["triples"] + 4*g["HR"]
    g["SLG"]   = (tb / g["AB"].replace(0, np.nan)).round(3)
    g["OPS"]   = (g["OBP"] + g["SLG"]).round(3)
    g["hr_per_pa"]  = (g["HR"] / g["PA"].replace(0, np.nan)).round(4)
    g["hit_per_pa"] = (g["H"]  / g["PA"].replace(0, np.nan)).round(4)
    return g.set_index("batter")


def build_pitcher_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pitcher stats from the batter's perspective (PA faced, HR allowed, etc.).
    """
    g = pa.groupby("pitcher").agg(
        PA_faced=("PA", "sum"),
        H_allowed=("H", "sum"),
        HR_allowed=("HR", "sum"),
        BB_allowed=("BB", "sum"),
        K=("K", "sum"),
    ).reset_index()

    g["hr_per_pa_allowed"] = (
        g["HR_allowed"] / g["PA_faced"].replace(0, np.nan)
    ).round(4)
    g["h_per_pa_allowed"]  = (
        g["H_allowed"]  / g["PA_faced"].replace(0, np.nan)
    ).round(4)
    g["k_pct"] = (g["K"] / g["PA_faced"].replace(0, np.nan)).round(4)
    return g.set_index("pitcher")


def build_batter_game_log(pa: pd.DataFrame) -> pd.DataFrame:
    """Per-batter per-game hit log (used for streak calculation)."""
    g = pa.groupby(["batter", "game_date", "bbref_game_id"]).agg(
        H=("H", "sum"),
        AB=("AB", "sum"),
        PA=("PA", "sum"),
    ).reset_index()
    g["got_hit"] = (g["H"] > 0).astype(int)
    return g.sort_values(["batter", "game_date"])


def current_hit_streak(batter: str, game_log: pd.DataFrame) -> int:
    """Count consecutive games with a hit, going backwards from most recent."""
    bdf = game_log[game_log["batter"] == batter].sort_values("game_date")
    streak = 0
    for got_hit in reversed(bdf["got_hit"].tolist()):
        if got_hit:
            streak += 1
        else:
            break
    return streak


def build_nrfi_game_results(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Determine NRFI/YRFI result for each historical game.
    Uses the score at the first PA of the 2nd inning (top) to detect
    whether any runs scored in the 1st inning (top + bottom).

    Returns DataFrame with columns: bbref_game_id, game_date, away_r1, home_r1, nrfi.
    """
    rows = []
    for gid, gdf in pa.groupby("bbref_game_id"):
        gdf = gdf.sort_index()
        t2  = gdf[gdf["inning"] == "t2"]
        if t2.empty:
            continue   # incomplete game data
        away_r1, home_r1 = _parse_score(t2.iloc[0]["score"])
        nrfi = (away_r1 == 0 and home_r1 == 0)
        game_date = gdf["game_date"].iloc[0]
        rows.append({
            "bbref_game_id": gid,
            "game_date":     game_date,
            "away_r1":       away_r1,
            "home_r1":       home_r1,
            "nrfi":          nrfi,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BEAT THE STREAK
# ═══════════════════════════════════════════════════════════════════════════════

def _load_bts_tracker() -> dict:
    """Load the running BTS tracker JSON, or initialize a fresh one."""
    if BTS_TRACKER.exists():
        try:
            with open(BTS_TRACKER) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "season":        2026,
        "current_streak": 0,
        "best_streak":    0,
        "picks":          [],   # [{date, batter, result, streak_after}]
        "last_updated":   None,
    }


def _save_bts_tracker(tracker: dict):
    BTS_TRACKER.parent.mkdir(parents=True, exist_ok=True)
    with open(BTS_TRACKER, "w") as f:
        json.dump(tracker, f, indent=2)


def pick_beat_the_streak(
    games: list,
    pa: pd.DataFrame,
    batter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    game_log: pd.DataFrame,
    today_str: str,
) -> dict:
    """
    Select the single best batter to get a hit today for Beat the Streak.

    Scoring formula (all normalized 0–1):
      45% — regressed hit rate  (H/PA regressed to league mean, prevents small-
             sample hot starts from dominating every day)
      25% — opposing pitcher H-allowed rate (also regressed)
      15% — park factor of the game
      15% — current hit streak momentum (streak / 15, capped at 1.0)

    Regression prior: (career_PA * raw_rate + PRIOR_PA * league_avg) / (career_PA + PRIOR_PA)
    A batter with 20 PA at .400 regresses to ~.296; a vet with 3000 PA at .300
    stays near .299. Same math used in DIPS/FIP theory.

    Minimum 50 career PA required (up from 10) to filter cups-of-coffee.
    """
    LEAGUE_AVG_HIT_RATE   = 0.270
    BATTER_PRIOR_PA       = 300      # regression weight for batters
    LEAGUE_AVG_H_ALLOWED  = 0.260
    PITCHER_PRIOR_PA      = 400      # regression weight for pitchers
    HIT_RATE_ELITE        = 0.320    # normalization ceiling (genuine elite)

    sp_map   = {}
    park_map = {}
    for g in games:
        ht  = g.get("home_team", "")
        at  = g.get("away_team", "")
        hsp = g.get("home_sp", "") or ""
        asp = g.get("away_sp", "") or ""
        sp_map[ht]   = asp
        sp_map[at]   = hsp
        park_map[ht] = ht
        park_map[at] = ht

    last_game_date = pa["game_date"].max()
    recent_batters = pa[pa["game_date"] == last_game_date]["batter"].unique()
    recent_pa      = pa[pa["game_date"] == last_game_date]
    batter_to_team = build_batter_team_map(recent_pa, games)

    candidates = []
    for batter in recent_batters:
        if batter not in batter_stats.index:
            continue
        bs = batter_stats.loc[batter]
        career_pa = int(bs["PA"])
        if career_pa < 50:
            continue

        team    = batter_to_team.get(batter, "")
        opp_sp  = sp_map.get(team, "")
        home_tm = park_map.get(team, team)
        park    = PARK_FACTORS.get(home_tm, 1.00)

        # ── Regressed hit rate ────────────────────────────────────────────────
        raw_hit_rate = float(bs["hit_per_pa"] or 0)
        reg_hit_rate = ((career_pa * raw_hit_rate + BATTER_PRIOR_PA * LEAGUE_AVG_HIT_RATE)
                        / (career_pa + BATTER_PRIOR_PA))
        hit_rate_norm = min(reg_hit_rate / HIT_RATE_ELITE, 1.0)

        # ── Regressed pitcher H-allowed rate ──────────────────────────────────
        if opp_sp and opp_sp in pitcher_stats.index:
            ps = pitcher_stats.loc[opp_sp]
            p_pa = float(ps.get("PA_faced") or 0)
            raw_h = float(ps["h_per_pa_allowed"]) if not pd.isna(ps.get("h_per_pa_allowed")) else LEAGUE_AVG_H_ALLOWED
            reg_h = ((p_pa * raw_h + PITCHER_PRIOR_PA * LEAGUE_AVG_H_ALLOWED)
                     / (p_pa + PITCHER_PRIOR_PA))
        else:
            reg_h = LEAGUE_AVG_H_ALLOWED
        pitcher_norm = min(reg_h / 0.310, 1.0)   # 0.310 = very hittable

        streak = current_hit_streak(batter, game_log)

        park_norm   = (park - 0.90) / (1.20 - 0.90)
        streak_norm = min(streak / 15.0, 1.0)

        score = (
            0.45 * hit_rate_norm +
            0.25 * pitcher_norm  +
            0.15 * park_norm     +
            0.15 * streak_norm
        )

        candidates.append({
            "batter":         batter,
            "team":           team,
            "opp_sp":         opp_sp,
            "home_park":      home_tm,
            "park_factor":    park,
            "season_avg":     round(float(bs["AVG"] or 0), 3),
            "season_ops":     round(float(bs["OPS"] or 0), 3),
            "career_pa":      career_pa,
            "hit_per_pa":     round(raw_hit_rate, 3),
            "reg_hit_rate":   round(reg_hit_rate, 3),
            "pitcher_h_rate": round(reg_h, 3),
            "current_streak": streak,
            "score":          round(score, 4),
        })

    if not candidates:
        return {"pick": None, "candidates": [], "tracker": _load_bts_tracker()}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top = candidates[0]
    tracker = _load_bts_tracker()

    pick = {
        "date":          today_str,
        "batter":        top["batter"],
        "team":          top["team"],
        "opp_sp":        top["opp_sp"],
        "park":          top["home_park"],
        "park_factor":   top["park_factor"],
        "season_avg":    top["season_avg"],
        "season_ops":    top["season_ops"],
        "hit_per_pa":    top["hit_per_pa"],
        "pitcher_h_rate":top["pitcher_h_rate"],
        "current_streak":top["current_streak"],
        "bts_streak":    tracker["current_streak"],
        "score":         top["score"],
        "top5":          candidates[:5],
    }

    return {"pick": pick, "tracker": tracker}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HOME RUN PROPS
# ═══════════════════════════════════════════════════════════════════════════════

# League-average HR/PA ≈ 0.033 (roughly 1 HR per 30 PAs)
LEAGUE_AVG_HR_PA = 0.033

# Minimum career PA needed to trust BvP HR rate over season rate
BVP_MIN_PA = 10


def load_bvp_hr(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Build career batter-vs-pitcher HR stats directly from the PA data files.
    Returns a DataFrame indexed by (batter, pitcher) with columns:
      bvp_pa, bvp_hr, bvp_hr_rate, bvp_h, bvp_avg

    We compute this from the full PA dataset (all years) so it's genuinely
    career H2H data — the same matchup history the book is pricing against.
    """
    g = pa.groupby(["batter", "pitcher"]).agg(
        bvp_pa=("PA",  "sum"),
        bvp_ab=("AB",  "sum"),
        bvp_h= ("H",   "sum"),
        bvp_hr=("HR",  "sum"),
        bvp_bb=("BB",  "sum"),
        bvp_k= ("K",   "sum"),
    ).reset_index()

    g["bvp_hr_rate"] = (g["bvp_hr"] / g["bvp_pa"].replace(0, np.nan)).round(4)
    g["bvp_avg"]     = (g["bvp_h"]  / g["bvp_ab"].replace(0, np.nan)).round(3)
    return g.set_index(["batter", "pitcher"])


def pick_hr_props(
    games: list,
    pa: pd.DataFrame,
    batter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    game_log: pd.DataFrame,
    today_str: str,
    bvp_hr: pd.DataFrame | None = None,
    top_n: int = 5,
) -> list:
    """
    Rank today's batters for HR likelihood using three layers:

      1. Season HR/PA rate   (batter's overall HR rate this year)
      2. Pitcher HR-allowed  (how many HRs this pitcher gives up per PA)
      3. Career BvP HR rate  (how many HRs THIS batter has hit off THIS pitcher)

    When career BvP data exists (≥ BVP_MIN_PA), it's blended with the season
    rate using a credibility weight:
        blend = bvp_weight * bvp_hr_rate + (1 - bvp_weight) * season_hr_rate
    where bvp_weight = min(bvp_pa / 50, 0.6) — caps at 60% so season context
    always contributes.  50 PA = full credibility threshold.

    Final score = blended_hr_rate × pitcher_mult × park_factor
    """
    last_game_date = pa["game_date"].max()
    recent_batters = pa[pa["game_date"] == last_game_date]["batter"].unique()
    recent_pa      = pa[pa["game_date"] == last_game_date]

    # Build team and SP maps from today's games
    sp_map   = {}
    park_map = {}
    for g in games:
        ht  = g.get("home_team", "")
        at  = g.get("away_team", "")
        hsp = g.get("home_sp", "") or ""
        asp = g.get("away_sp", "") or ""
        sp_map[ht]   = asp    # home batters face away SP
        sp_map[at]   = hsp    # away batters face home SP
        park_map[ht] = ht
        park_map[at] = ht

    batter_to_team = build_batter_team_map(recent_pa, games)

    candidates = []
    for batter in recent_batters:
        if batter not in batter_stats.index:
            continue
        bs = batter_stats.loc[batter]
        career_pa = int(bs["PA"])
        if career_pa < 50 or int(bs["AB"]) < 20:
            continue

        team    = batter_to_team.get(batter, "")
        opp_sp  = sp_map.get(team, "")
        home_tm = park_map.get(team, team)
        park    = PARK_FACTORS.get(home_tm, 1.00)

        season_hr_rate = float(bs["hr_per_pa"] or 0)

        # ── Regress batter HR rate to league mean ─────────────────────────────
        # Do this BEFORE the zero check. Early-season batters with 0 raw HRs
        # in 20 PA still carry a regressed rate of ~2.8% — skipping them
        # entirely (as the old raw-rate==0 check did) would blank the entire
        # HR props list in the first weeks of the season.
        # 200 PA prior: a batter with 21 PA at 23.8% regresses to ~5.2%;
        # a batter with 20 PA at 0% regresses to ~2.8% (still rankable).
        BATTER_HR_PRIOR_PA = 200
        reg_hr_rate = ((career_pa * season_hr_rate + BATTER_HR_PRIOR_PA * LEAGUE_AVG_HR_PA)
                       / (career_pa + BATTER_HR_PRIOR_PA))

        # Skip only if regressed rate is essentially zero (no career HR history
        # at all — e.g. a pitcher batting or a true zero across many seasons).
        if reg_hr_rate < 0.005:
            continue

        # ── Layer 2: pitcher HR-allowed rate (regressed, capped) ──────────────
        # If SP is unknown fall back to 1.0x (league average) rather than
        # skipping the batter entirely — unknown SP is common early in the
        # season and would blank the entire HR props list.
        sp_known = bool(opp_sp and opp_sp not in ("TBD", ""))
        PITCHER_HR_PRIOR_PA = 300
        if sp_known and opp_sp in pitcher_stats.index:
            ps = pitcher_stats.loc[opp_sp]
            p_pa = float(ps.get("PA_faced") or 0)
            raw_p_hr = float(ps["hr_per_pa_allowed"] or LEAGUE_AVG_HR_PA)
            reg_p_hr = ((p_pa * raw_p_hr + PITCHER_HR_PRIOR_PA * LEAGUE_AVG_HR_PA)
                        / (p_pa + PITCHER_HR_PRIOR_PA))
        else:
            reg_p_hr = LEAGUE_AVG_HR_PA   # league-average multiplier
        # Cap at 2.0x so one bad outing doesn't make everyone look elite
        pitcher_mult = min(reg_p_hr / LEAGUE_AVG_HR_PA, 2.0)

        # ── Layer 3: career BvP HR rate ───────────────────────────────────────
        bvp_pa      = 0
        bvp_hr_val  = 0
        bvp_hr_rate = None
        bvp_avg     = None
        blended_hr_rate = reg_hr_rate   # default: regressed season rate

        if bvp_hr is not None and opp_sp:
            idx = (batter, opp_sp)
            if idx in bvp_hr.index:
                row         = bvp_hr.loc[idx]
                bvp_pa      = int(row["bvp_pa"])
                bvp_hr_val  = int(row["bvp_hr"])
                bvp_hr_rate = float(row["bvp_hr_rate"] or 0)
                bvp_avg     = float(row["bvp_avg"] or 0) if not pd.isna(row["bvp_avg"]) else None

                if bvp_pa >= BVP_MIN_PA:
                    bvp_weight      = min(bvp_pa / 50.0, 0.60)
                    blended_hr_rate = (bvp_weight * bvp_hr_rate +
                                       (1 - bvp_weight) * reg_hr_rate)

        # ── Final HR probability ──────────────────────────────────────────────
        hr_prob = blended_hr_rate * pitcher_mult * park
        hr_prob = min(hr_prob, 0.20)   # hard cap at 20% (realistic ceiling)

        candidates.append({
            "batter":           batter,
            "team":             team,
            "opp_sp":           opp_sp if sp_known else "TBD",
            "sp_known":         sp_known,
            "park":             home_tm,
            "park_factor":      park,
            "season_hr":        int(bs["HR"]),
            "season_pa":        career_pa,
            "season_hr_rate":   round(season_hr_rate, 4),
            "reg_hr_rate":      round(reg_hr_rate, 4),
            "pitcher_hr_rate":  round(reg_p_hr, 4),
            "pitcher_mult":     round(pitcher_mult, 2),
            "bvp_pa":           bvp_pa,
            "bvp_hr":           bvp_hr_val,
            "bvp_hr_rate":      round(bvp_hr_rate, 4) if bvp_hr_rate is not None else None,
            "bvp_avg":          round(bvp_avg, 3)     if bvp_avg     is not None else None,
            "blended_hr_rate":  round(blended_hr_rate, 4),
            "hr_prob":          round(hr_prob, 4),
            "hr_prob_pct":      round(hr_prob * 100, 1),
            "season_avg":       round(float(bs["AVG"] or 0), 3),
            "season_ops":       round(float(bs["OPS"] or 0), 3),
        })

    candidates.sort(key=lambda x: x["hr_prob"], reverse=True)
    return candidates[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NRFI / YRFI PICKS
# ═══════════════════════════════════════════════════════════════════════════════

# Historical NRFI base rate from our data (game level — both halves scoreless)
# Confirmed against 12,412 games 2021-2026: 51.0% of games are NRFI.
# Each individual half-inning is scoreless ~71% of the time (sqrt of 51%).
NRFI_BASE_RATE      = 0.51   # full-game NRFI rate
NRFI_HALF_BASE_RATE = 0.71   # per-pitcher half-inning scoreless rate (league avg)


def load_nrfi_backtest() -> pd.DataFrame:
    """
    Load nrfi_pitcher_backtest.csv built by nrfi_backtest.py.
    Returns a DataFrame indexed by pitcher name, or an empty DataFrame
    if the file doesn't exist (pick_nrfi falls back to league averages).
    """
    if not NRFI_BACKTEST.exists():
        print(f"  [warn] NRFI backtest not found at {NRFI_BACKTEST}")
        print("         Run: python nrfi_backtest.py  to generate it.")
        return pd.DataFrame()
    df = pd.read_csv(NRFI_BACKTEST)
    # Normalize names (same non-breaking-space fix as PA data)
    df["pitcher"] = df["pitcher"].str.replace(" ", " ", regex=False).str.strip()
    return df.set_index("pitcher")


def _pitcher_nrfi_prob(pitcher_name: str,
                       nrfi_bt: pd.DataFrame,
                       pitcher_stats: pd.DataFrame,
                       min_starts: int = 10) -> tuple[float, dict]:
    """
    Return (half_inning_nrfi_prob, detail_dict) for a pitcher.

    Priority:
      1. Historical sp_nrfi_pct from nrfi_backtest.csv  (>= min_starts)
      2. Heuristic from all-PA pitcher_stats            (< min_starts or missing)
      3. League-average fallback                        (completely unknown)

    Returns a prob in [0, 1] where 1.0 = always NRFI in their half-inning.
    """
    detail = {
        "source":       "league_avg",
        "starts":       0,
        "sp_nrfi_pct":  None,
        "avg_r_1st":    None,
    }

    # ── Priority 1: real historical rate ─────────────────────────────────────
    if not nrfi_bt.empty and pitcher_name and pitcher_name in nrfi_bt.index:
        row = nrfi_bt.loc[pitcher_name]
        starts = int(row.get("starts", 0))
        if starts >= min_starts:
            prob = float(row["sp_nrfi_pct"]) / 100.0
            detail.update({
                "source":      "historical",
                "starts":      starts,
                "sp_nrfi_pct": round(float(row["sp_nrfi_pct"]), 1),
                "avg_r_1st":   round(float(row["avg_runs_allowed_1st"]), 3),
            })
            return round(prob, 4), detail

    # ── Priority 2: heuristic from season PA data ─────────────────────────────
    if pitcher_name and pitcher_stats is not None and pitcher_name in pitcher_stats.index:
        ps = pitcher_stats.loc[pitcher_name]
        pa = float(ps.get("PA_faced") or 1)
        if pa >= 5:
            k_pct     = float(ps.get("k_pct")            or 0)
            h_per_pa  = float(ps.get("h_per_pa_allowed")  or 0.28)
            hr_per_pa = float(ps.get("hr_per_pa_allowed") or 0.033)
            k_score   = min(k_pct / 0.35, 1.0)
            h_score   = 1.0 - min(h_per_pa / 0.40, 1.0)
            hr_score  = 1.0 - min(hr_per_pa / 0.10, 1.0)
            heuristic = 0.40 * k_score + 0.35 * h_score + 0.25 * hr_score
            # Scale heuristic [0,1] → plausible prob range [0.55, 0.87]
            prob = 0.55 + heuristic * 0.32
            detail["source"] = "heuristic"
            return round(prob, 4), detail

    # ── Priority 3: league average ────────────────────────────────────────────
    return NRFI_HALF_BASE_RATE, detail


def _team_nrfi_offense_score(team: str, batter_stats: pd.DataFrame,
                              pa: pd.DataFrame, games: list) -> float:
    """
    Team's offensive aggressiveness — higher = more likely to score in inning 1.
    Uses the team's season hit rate and HR rate for batters confirmed on that roster.

    Previously pooled ALL recent batters regardless of team. Now filters to only
    batters who actually played for `team` on the most recent game date, using
    bbref_game_id[:3] + batting_team_home to identify team membership precisely.
    """
    last_date  = pa["game_date"].max()
    recent_pa  = pa[pa["game_date"] == last_date]
    team_map   = build_batter_team_map(recent_pa, games)
    team_batters = [b for b, t in team_map.items() if t == team]

    if not team_batters:
        return 0.5   # no data for this team — neutral

    team_hit_rates = []
    team_hr_rates  = []
    for b in team_batters:
        if b in batter_stats.index:
            bs = batter_stats.loc[b]
            if bs["PA"] >= 10:
                team_hit_rates.append(float(bs["hit_per_pa"] or 0))
                team_hr_rates.append(float(bs["hr_per_pa"]  or 0))

    if not team_hit_rates:
        return 0.5

    avg_hit = np.mean(team_hit_rates)
    avg_hr  = np.mean(team_hr_rates)

    # Offense score: higher = more dangerous
    hit_component = min(avg_hit / 0.35, 1.0)
    hr_component  = min(avg_hr  / 0.06, 1.0)
    offense_score = 0.65 * hit_component + 0.35 * hr_component

    return round(offense_score, 4)


def pick_nrfi(
    games: list,
    pa: pd.DataFrame,
    batter_stats: pd.DataFrame,
    pitcher_stats: pd.DataFrame,
    today_str: str,
    nrfi_threshold: float = 0.60,
    yrfi_threshold: float = 0.42,
) -> list:
    """
    Score each game for NRFI/YRFI likelihood using real historical SP data.

    Method:
      game_nrfi_prob = home_sp_half_prob × away_sp_half_prob × park_adj × offense_adj

    home_sp_half_prob = probability that home SP keeps away team scoreless in t1.
    away_sp_half_prob = probability that away SP keeps home team scoreless in b1.
    Both come from nrfi_pitcher_backtest.csv (historical sp_nrfi_pct) when
    available, falling back to a heuristic from season PA stats, then to the
    league-average half-inning rate (71%).

    Thresholds are calibrated to the product of two half-inning probs:
      nrfi_threshold 0.60 → both pitchers historically ~78%+ NRFI (above avg)
      yrfi_threshold 0.42 → at least one pitcher ~58% or below (well below avg)
      League avg game NRFI = 51% (0.71 × 0.71). No pick if game is near avg.
    """
    # Load historical backtest once per call
    nrfi_bt = load_nrfi_backtest()

    results = []

    for g in games:
        ht  = g.get("home_team", "")
        at  = g.get("away_team", "")
        hsp = g.get("home_sp", "") or ""
        asp = g.get("away_sp", "") or ""

        park = PARK_FACTORS.get(ht, 1.00)

        # ── Per-pitcher historical NRFI probabilities ─────────────────────────
        # home_sp faces away batters in t1 → home_sp_prob = P(away_r1 == 0)
        # away_sp faces home batters in b1 → away_sp_prob = P(home_r1 == 0)
        home_sp_prob, home_sp_detail = _pitcher_nrfi_prob(hsp, nrfi_bt, pitcher_stats)
        away_sp_prob, away_sp_detail = _pitcher_nrfi_prob(asp, nrfi_bt, pitcher_stats)

        # ── Offense adjustment ────────────────────────────────────────────────
        # Compress team offense score into a small multiplier around 1.0.
        # A very strong offense (score=0.8) reduces NRFI prob by ~6%.
        # A very weak offense (score=0.2) increases NRFI prob by ~6%.
        home_off = _team_nrfi_offense_score(ht, batter_stats, pa, games)
        away_off = _team_nrfi_offense_score(at, batter_stats, pa, games)
        # home offense faces away SP; away offense faces home SP
        away_off_adj = 1.0 - 0.12 * (away_off - 0.5)   # affects home_sp_prob
        home_off_adj = 1.0 - 0.12 * (home_off - 0.5)   # affects away_sp_prob

        # ── Park adjustment ────────────────────────────────────────────────────
        # Applied once to the GAME probability, not per-half (avoids compounding).
        # Coors (1.18) → -7% on game prob; pitcher park (0.94) → +2.5%
        # Formula: neutral park (1.00) → multiplier 1.0, linear either way.
        park_mult = 1.0 - 0.60 * (park - 1.00)

        # ── Combine into game NRFI probability ────────────────────────────────
        # Each half is an independent event. Apply offense adj per-half,
        # then multiply together, then apply single park mult.
        adj_home_sp = float(np.clip(home_sp_prob * away_off_adj, 0.20, 0.97))
        adj_away_sp = float(np.clip(away_sp_prob * home_off_adj, 0.20, 0.97))
        nrfi_prob   = float(np.clip(adj_home_sp * adj_away_sp * park_mult, 0.10, 0.92))

        pick_side = None
        if nrfi_prob >= nrfi_threshold:
            pick_side = "NRFI"
        elif nrfi_prob <= yrfi_threshold:
            pick_side = "YRFI"

        # Confidence: distance from 50/50 game, scaled to [0,1]
        confidence = abs(nrfi_prob - 0.50) / 0.50

        results.append({
            "home_team":          ht,
            "away_team":          at,
            "home_sp":            hsp or "TBD",
            "away_sp":            asp or "TBD",
            "park":               ht,
            "park_factor":        park,
            # Per-pitcher detail for web display
            "home_sp_nrfi_pct":   home_sp_detail.get("sp_nrfi_pct"),
            "home_sp_starts":     home_sp_detail.get("starts", 0),
            "home_sp_avg_r":      home_sp_detail.get("avg_r_1st"),
            "home_sp_source":     home_sp_detail.get("source", "league_avg"),
            "away_sp_nrfi_pct":   away_sp_detail.get("sp_nrfi_pct"),
            "away_sp_starts":     away_sp_detail.get("starts", 0),
            "away_sp_avg_r":      away_sp_detail.get("avg_r_1st"),
            "away_sp_source":     away_sp_detail.get("source", "league_avg"),
            "home_off_score":     round(home_off, 3),
            "away_off_score":     round(away_off, 3),
            "nrfi_prob":          round(nrfi_prob, 3),
            "yrfi_prob":          round(1 - nrfi_prob, 3),
            "pick":               pick_side,
            "confidence":         round(confidence, 3),
            "confidence_pct":     round(confidence * 100, 1),
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# WRITEUP GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _short_pitcher(name: str) -> str:
    if not name or name.strip() in ("TBD", "", "Unknown"):
        return "TBD"
    parts = name.strip().split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) >= 2 else name


def build_props_writeup(bts: dict, hr_props: list, nrfi_picks: list, today_str: str) -> str:
    lines = []
    lines.append(f"MLB PROPS — {today_str}")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ── Beat the Streak ───────────────────────────────────────────────────────
    lines.append("─" * 60)
    lines.append("BEAT THE STREAK")
    lines.append("─" * 60)
    pick = bts.get("pick")
    tracker = bts.get("tracker", {})

    if pick:
        streak = tracker.get("current_streak", 0)
        best   = tracker.get("best_streak", 0)
        lines.append(f"🎯 Pick: {pick['batter']} ({pick['team']})")
        lines.append(f"   vs {_short_pitcher(pick['opp_sp'])} @ {pick['park']}")
        lines.append(f"   Season: .{int(pick['season_avg']*1000):03d} AVG | "
                     f"OPS {pick['season_ops']:.3f} | "
                     f"Hit rate {pick['hit_per_pa']*100:.1f}%")
        lines.append(f"   Pitcher H-allowed: {pick['pitcher_h_rate']*100:.1f}%  "
                     f"Park: {pick['park_factor']:.2f}x")
        lines.append(f"   Current personal streak: {pick['current_streak']} games")
        lines.append(f"   BTS streak this season:  {streak} games "
                     f"(best: {best})")
        if streak >= 10:
            lines.append(f"   🔥 {streak} games in! DiMaggio's 56 is the target.")
        lines.append("")
        lines.append("  Top 5 candidates:")
        for i, c in enumerate(pick.get("top5", []), 1):
            lines.append(f"  {i}. {c['batter']} ({c['team']})  "
                         f"Score:{c['score']:.3f}  "
                         f"Avg:{c['season_avg']:.3f}  "
                         f"Streak:{c['current_streak']}")
    else:
        lines.append("  No BTS pick available today.")

    lines.append("")

    # ── HR Props ──────────────────────────────────────────────────────────────
    lines.append("─" * 60)
    lines.append("HOME RUN PROPS")
    lines.append("─" * 60)
    if hr_props:
        for i, p in enumerate(hr_props, 1):
            lines.append(
                f"  {i}. {p['batter']} ({p['team']})  "
                f"HR%: {p['hr_prob_pct']:.1f}%  "
                f"vs {_short_pitcher(p['opp_sp'])}"
            )
            lines.append(
                f"     Season: {p['season_hr']} HR / {p['season_pa']} PA  "
                f"({p['season_hr_rate']*100:.2f}% rate)  "
                f"Park: {p['park_factor']:.2f}x  "
                f"Pitcher mult: {p['pitcher_mult']:.2f}x"
            )
            # Show BvP career data if we have it
            if p.get("bvp_pa", 0) >= BVP_MIN_PA:
                bvp_avg_str = f".{int(p['bvp_avg']*1000):03d}" if p.get("bvp_avg") is not None else "---"
                lines.append(
                    f"     Career H2H: {p['bvp_hr']} HR / {p['bvp_pa']} PA  "
                    f"({p['bvp_hr_rate']*100:.2f}% rate)  "
                    f"AVG {bvp_avg_str}  ← blended into score"
                )
            elif p.get("bvp_pa", 0) > 0:
                lines.append(
                    f"     Career H2H: {p['bvp_hr']} HR / {p['bvp_pa']} PA  "
                    f"(< {BVP_MIN_PA} PA — season rate used)"
                )
    else:
        lines.append("  No HR props available today.")

    lines.append("")

    # ── NRFI ─────────────────────────────────────────────────────────────────
    lines.append("─" * 60)
    lines.append("NRFI / YRFI PICKS")
    lines.append("─" * 60)
    picks_only = [g for g in nrfi_picks if g["pick"] is not None]

    if picks_only:
        lines.append(f"  {len(picks_only)} pick(s):\n")
        for g in picks_only:
            emoji = "🔒" if g["pick"] == "NRFI" else "💥"
            def _sp_record(pct, starts, src):
                if pct is not None and starts:
                    return f"{pct:.0f}% NRFI ({starts} GS)"
                return "league avg" if src == "league_avg" else "limited data"
            hsr = _sp_record(g.get("home_sp_nrfi_pct"), g.get("home_sp_starts", 0),
                             g.get("home_sp_source", "league_avg"))
            asr = _sp_record(g.get("away_sp_nrfi_pct"), g.get("away_sp_starts", 0),
                             g.get("away_sp_source", "league_avg"))
            lines.append(
                f"  {emoji} {g['pick']}  {g['away_team']} @ {g['home_team']}  "
                f"({g['nrfi_prob']*100:.0f}% NRFI)  Conf: {g['confidence_pct']:.0f}%"
            )
            lines.append(
                f"     {_short_pitcher(g['away_sp'])}: {asr}  |  "
                f"{_short_pitcher(g['home_sp'])}: {hsr}  |  "
                f"Park: {g['park_factor']:.2f}x"
            )
    else:
        lines.append("  No NRFI/YRFI picks today (all games near 50/50).")

    return "\n".join(lines)


def save_props_writeup(text: str, today_str: str):
    for wd in [WRITEUPS_DIR, LOCAL_WRITEUPS]:
        wd.mkdir(parents=True, exist_ok=True)
        path = wd / f"{today_str}_props.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    print(f"  Props writeup → {LOCAL_WRITEUPS / f'{today_str}_props.txt'}")


# ═══════════════════════════════════════════════════════════════════════════════
# BTS RESULT GRADER
# ═══════════════════════════════════════════════════════════════════════════════

def grade_bts_yesterday(pa: pd.DataFrame):
    """
    Check yesterday's BTS pick against actual PA results and update tracker.
    Called at the start of each run so the streak is always current.
    """
    tracker = _load_bts_tracker()
    picks   = tracker.get("picks", [])
    if not picks:
        return tracker

    last_pick = picks[-1]
    if last_pick.get("result") is not None:
        return tracker   # already graded

    pick_date   = last_pick["date"]
    pick_batter = last_pick["batter"]

    # Look for the batter's PA on pick_date.
    # IMPORTANT: game_date is a pd.Timestamp after load_pa_data() converts it.
    # .astype(str) on a Timestamp gives '2026-04-03 00:00:00' which never
    # matches the pick_date string '2026-04-03'. Use .dt.date instead.
    day_pa = pa[
        (pa["game_date"].dt.date.astype(str) == pick_date) &
        (pa["batter"] == pick_batter)
    ]

    latest_date = pa["game_date"].max().date().isoformat()
    if day_pa.empty:
        if latest_date < pick_date:
            print(f"  [BTS] Grading pending — PA data only current to {latest_date} "
                  f"(need {pick_date}). Run fetch_historical.py to update.")
        else:
            print(f"  [BTS] {pick_batter} not found in PA data for {pick_date} "
                  f"(may have been scratched or DNP).")
        return tracker

    got_hit = day_pa["H"].sum() > 0
    last_pick["result"] = "HIT" if got_hit else "OUT"

    if got_hit:
        tracker["current_streak"] += 1
        tracker["best_streak"] = max(
            tracker["best_streak"], tracker["current_streak"]
        )
        print(f"  [BTS] ✓ {pick_batter} got a hit! Streak: {tracker['current_streak']}")
    else:
        print(f"  [BTS] ✗ {pick_batter} went hitless. Streak reset to 0.")
        tracker["current_streak"] = 0

    tracker["last_updated"] = datetime.now().isoformat()
    _save_bts_tracker(tracker)
    return tracker


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def generate(picks_json: dict | None = None) -> dict | None:
    """
    Main entry point. Called by run_today.py after saving picks_today.json,
    or standalone via python player_props.py.

    picks_json: the already-parsed picks_today.json dict (optional).
                When provided, its game-level SP names are merged into the
                games list before any SP lookup fallbacks run.
    Returns the full props dict written to player_props_today.json.
    """
    today_str = date.today().isoformat()
    print("\n── Player Props ────────────────────────────────────────")

    # Load PA data (all years — needed for career BvP HR stats)
    try:
        pa = load_pa_data()
    except FileNotFoundError as e:
        print(f"  [error] {e}")
        return None

    # Load today's games with SPs (picks_today.json → ESPN → error if still missing)
    try:
        games = load_today_games()
    except ValueError as e:
        print(str(e))
        return None

    # Additional SP merge from in-memory picks_json (run_today.py passes this
    # immediately after writing picks_today.json, so it's always current)
    if picks_json:
        picks_games = {
            f"{g.get('home_team')}|{g.get('away_team')}": g
            for g in picks_json.get("games", [])
        }
        for g in games:
            key = f"{g.get('home_team')}|{g.get('away_team')}"
            pg  = picks_games.get(key)
            if pg:
                if not g.get("home_sp") and pg.get("home_sp"):
                    g["home_sp"] = pg["home_sp"]
                if not g.get("away_sp") and pg.get("away_sp"):
                    g["away_sp"] = pg["away_sp"]

    if not games:
        print("  [warn] No games found — run pull_lines.py first")
        return None

    # Build stat tables
    print("  Building batter/pitcher stats...")
    batter_stats  = build_batter_stats(pa)
    pitcher_stats = build_pitcher_stats(pa)
    game_log      = build_batter_game_log(pa)

    # Build career BvP HR table from full PA history
    print("  Building career BvP HR matchups...")
    bvp_hr = load_bvp_hr(pa)
    n_matchups = len(bvp_hr)
    n_with_hr  = (bvp_hr["bvp_hr"] > 0).sum()
    print(f"  BvP HR: {n_matchups:,} career matchups, "
          f"{n_with_hr:,} with at least one HR")

    # Grade yesterday's BTS pick before making today's
    grade_bts_yesterday(pa)

    # ── Run the three features ────────────────────────────────────────────────
    print("  Computing Beat the Streak pick...")
    bts = pick_beat_the_streak(
        games, pa, batter_stats, pitcher_stats, game_log, today_str
    )

    print("  Computing HR props (season + BvP career)...")
    hr_props = pick_hr_props(
        games, pa, batter_stats, pitcher_stats, game_log, today_str,
        bvp_hr=bvp_hr,
    )

    print("  Computing NRFI/YRFI picks...")
    nrfi_picks = pick_nrfi(games, pa, batter_stats, pitcher_stats, today_str)

    # ── Save BTS tracker with today's pick ───────────────────────────────────
    if bts.get("pick"):
        tracker = bts["tracker"]
        # Guard against duplicate entries when run_today.py calls generate()
        # multiple times in the same day (re-runs, testing, etc.)
        already_logged = any(p.get("date") == today_str
                             for p in tracker.get("picks", []))
        if not already_logged:
            tracker["picks"].append({
                "date":         today_str,
                "batter":       bts["pick"]["batter"],
                "team":         bts["pick"]["team"],
                "opp_sp":       bts["pick"]["opp_sp"],
                "result":       None,   # graded next run
                "streak_after": None,
            })
            tracker["last_updated"] = datetime.now().isoformat()
            _save_bts_tracker(tracker)
        else:
            print(f"  [BTS] Today's pick already logged — skipping duplicate write."
                  f" ({bts['pick']['batter']})"
            )

    # ── Assemble output ───────────────────────────────────────────────────────
    nrfi_picks_only = [g for g in nrfi_picks if g["pick"] is not None]
    output = {
        "generated_at":  datetime.now().isoformat(),
        "date":          today_str,
        "beat_the_streak": bts,
        "hr_props":      hr_props,
        "nrfi_picks":    nrfi_picks_only,
        "nrfi_all":      nrfi_picks,
    }

    PROPS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PROPS_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Props → {PROPS_OUT}")

    # ── Print summary ─────────────────────────────────────────────────────────
    bts_pick = bts.get("pick")
    if bts_pick:
        tracker  = _load_bts_tracker()
        print(f"\n  🎯 BTS Pick: {bts_pick['batter']} ({bts_pick['team']})  "
              f"Streak: {tracker['current_streak']}  "
              f"Avg: {bts_pick['season_avg']:.3f}")

    if hr_props:
        print(f"\n  💣 HR Props (top {len(hr_props)}):")
        for p in hr_props:
            bvp_str = ""
            if p.get("bvp_pa", 0) >= BVP_MIN_PA:
                bvp_str = f"  BvP: {p['bvp_hr']}HR/{p['bvp_pa']}PA"
            print(f"     {p['batter']:<22} {p['hr_prob_pct']:.1f}%  "
                  f"vs {_short_pitcher(p['opp_sp'])}{bvp_str}")

    if nrfi_picks_only:
        print(f"\n  ⚾ NRFI/YRFI ({len(nrfi_picks_only)} picks):")
        for g in nrfi_picks_only:
            emoji = "🔒" if g["pick"] == "NRFI" else "💥"
            print(f"     {emoji} {g['pick']}  {g['away_team']} @ {g['home_team']}  "
                  f"{g['nrfi_prob']*100:.0f}% NRFI")

    # ── Write props writeup file ──────────────────────────────────────────────
    writeup = build_props_writeup(bts, hr_props, nrfi_picks, today_str)
    save_props_writeup(writeup, today_str)

    return output


def main():
    generate()


if __name__ == "__main__":
    main()
