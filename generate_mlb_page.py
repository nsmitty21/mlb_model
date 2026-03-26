"""
generate_mlb_page.py — MLB Picks + Results Bridge
===================================================
Responsibilities:
  1. Convert today's picks_today.json → mlb_picks_YYYYMMDD.csv
     (format that generate_picks_page.py expects)
  2. Fetch yesterday's ESPN scores and grade any pending bets, writing
     WIN / LOSS / PUSH + PNL into bet_results.csv
  3. Call generate_picks_page.py to rebuild index.html and push to GitHub

Directory layout written:
  <BASE_DIR>/Results/
      mlb_picks_YYYYMMDD.csv   <- today's picks (one file per day)
      bet_results.csv          <- all graded bets (appended nightly)

  Also mirrors both files to the NBA Model Files folder so that
  generate_picks_page.py can find them at its expected SPORT_CONFIG["mlb"] paths.

Usage:
    python generate_mlb_page.py              # normal: write picks + grade + push
    python generate_mlb_page.py --grade-only # skip picks, just grade + push
"""

import os, sys, re, json, importlib.util, argparse
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from pathlib import Path

# ── Config paths ───────────────────────────────────────────────────────────────
from config import LIVE_DIR, LOGS_DIR, BET_LOG_CSV, BASE_DIR

# Results folder lives directly inside MLB Model Files
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
PICKS_JSON  = os.path.join(LIVE_DIR, "picks_today.json")

# generate_picks_page.py hub script
NBA_OUTPUT_DIR = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\ML and Spread"
GPP_PATH       = os.path.join(NBA_OUTPUT_DIR, "generate_picks_page.py")

# Where generate_picks_page.py expects MLB data (its SPORT_CONFIG["mlb"] paths)
GPP_MLB_PICKS_DIR    = os.path.join(NBA_OUTPUT_DIR, "MLB", "Daily Picks CSV")
GPP_MLB_RESULTS_FILE = os.path.join(NBA_OUTPUT_DIR, "MLB", "Results", "bet_results.csv")

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(GPP_MLB_PICKS_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(GPP_MLB_RESULTS_FILE)).mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
HEADERS   = {"User-Agent": "Mozilla/5.0"}


# ── Team abbreviation map ──────────────────────────────────────────────────────
FULL_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",  "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",     "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",          "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",       "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",      "Detroit Tigers": "DET",
    "Houston Astros": "HOU",        "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",         "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",       "New York Mets": "NYM",
    "New York Yankees": "NYY",      "Oakland Athletics": "OAK",
    "Athletics": "OAK",             "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",  "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",   "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",         "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}

def _abbr(name: str) -> str:
    """Return 3-letter abbreviation for a full team name."""
    return FULL_TO_ABBR.get(str(name).strip(), str(name).strip()[:3].upper())


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_odds(odds) -> str:
    try:
        v = int(float(odds))
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"


def _units_to_tier(units: int) -> str:
    if units >= 3:   return "HIGH CONFIDENCE"
    if units >= 2:   return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"


def _game_time_et(commence: str) -> str:
    """Convert UTC commence time to ET string, Windows-safe."""
    try:
        utc = pd.to_datetime(commence, utc=True)
        et  = utc.tz_convert("US/Eastern")
        # %-I is Linux-only; strip leading zero manually for Windows
        s = et.strftime("%I:%M %p ET")
        return s.lstrip("0") or s
    except Exception:
        return ""


def _bet_signal(pick: dict, ht: str = "", at: str = "") -> str:
    """
    Build the BET_SIGNAL string generate_picks_page.py parse_signal() expects:
        "BET (NYY) ML (-142) [HIGH CONFIDENCE]"

    The team-tracker regex in _build_analytics_section:
        r'\\(([A-Z]+)\\s'
    requires the abbreviation in parens followed by a space.
    """
    market = str(pick.get("type", pick.get("market", "ML"))).upper()
    units  = int(pick.get("units", 1))
    odds   = _fmt_odds(pick.get("odds"))
    tier   = _units_to_tier(units)

    if market == "ML":
        abbr  = _abbr(pick.get("team", ht))
        label = f"({abbr}) ML"
    elif market in ("SPREAD", "RL"):
        team_str = str(pick.get("team", ht))
        # team may look like "New York Yankees -1.5" — grab first word group
        abbr  = _abbr(team_str.split()[0]) if team_str else _abbr(ht)
        spread = pick.get("spread", "")
        if isinstance(spread, (int, float)):
            label = f"({abbr}) RL {spread:+.1f}"
        else:
            label = f"({abbr}) RL"
    elif market == "TOTAL":
        direction = str(pick.get("direction", pick.get("side", "OVER"))).upper()
        ha = _abbr(pick.get("home_team", ht))
        aa = _abbr(pick.get("away_team", at))
        line = pick.get("total_line", "")
        label = f"({ha}/{aa}) {direction} {line}"
    else:
        abbr  = _abbr(pick.get("team", ht))
        label = f"({abbr}) {market}"

    return f"BET {label} ({odds}) [{tier}]"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WRITE TODAY'S PICKS CSV
# ═══════════════════════════════════════════════════════════════════════════════

def write_picks_csv(games: list, date_str: str) -> str | None:
    """
    Flatten the games list from picks_today.json into one row per pick and
    write to both RESULTS_DIR and the GPP mirror location.
    Returns the local path written, or None if nothing to write.
    """
    rows = []
    for game in games:
        ht    = game.get("home_team", "")
        at    = game.get("away_team", "")
        gtime = _game_time_et(game.get("commence", ""))

        for pick in game.get("picks", []):
            market = str(pick.get("type", "ML")).upper()

            signal = _bet_signal(pick, ht, at)

            # PRED_MARGIN: run diff for RL/ML, predicted total for totals
            try:
                if market == "TOTAL":
                    pred_margin = float(pick.get("pred_total", 0) or 0)
                else:
                    pred_margin = float(pick.get("model_diff", 0) or 0)
            except (TypeError, ValueError):
                pred_margin = 0.0

            # VEGAS_SPREAD column: spread for RL, total line for totals, ML odds otherwise
            if market in ("SPREAD", "RL"):
                vegas = str(pick.get("spread", "—"))
            elif market == "TOTAL":
                vegas = str(pick.get("total_line", "—"))
            else:
                vegas = _fmt_odds(pick.get("odds"))

            rows.append({
                "DATE":          date_str,
                "MATCHUP":       f"{at} @ {ht}",
                "BET_SIGNAL":    signal,
                "VEGAS_SPREAD":  vegas,
                "SPREAD_EDGE":   f"{float(pick.get('edge', 0)):+.1f}%",
                "MODEL_PRED":    f"{pick.get('model_prob', 0):.1f}% (model) vs {pick.get('book_prob', 0):.1f}% (book)",
                "PRED_MARGIN":   round(pred_margin, 2),
                "BOOK":          str(pick.get("book", "")).upper(),
                "ODDS":          _fmt_odds(pick.get("odds")),
                "UNITS":         int(pick.get("units", 1)),
                "MARKET":        market,
                "GAME_TIME":     gtime,
                "HOME_TEAM":     ht,
                "AWAY_TEAM":     at,
                "HOME_INJURIES": "",
                "AWAY_INJURIES": "",
            })

    if not rows:
        print("  No pick rows to write.")
        return None

    df       = pd.DataFrame(rows)
    date_num = date_str.replace("-", "")

    local_path = os.path.join(RESULTS_DIR, f"mlb_picks_{date_num}.csv")
    df.to_csv(local_path, index=False)
    print(f"  Picks  → {local_path}  ({len(df)} picks)")

    gpp_path = os.path.join(GPP_MLB_PICKS_DIR, f"mlb_picks_{date_num}.csv")
    df.to_csv(gpp_path, index=False)
    print(f"  Mirror → {gpp_path}")

    return local_path


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FETCH ESPN SCORES & GRADE YESTERDAY'S BETS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_espn_scores(game_date: str) -> list:
    """Return list of completed game result dicts for YYYY-MM-DD."""
    ds = game_date.replace("-", "")
    try:
        r = requests.get(ESPN_BASE, params={"dates": ds, "limit": 50},
                         headers=HEADERS, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"  [warn] ESPN fetch failed for {game_date}: {e}")
        return []

    games = []
    for ev in events:
        try:
            comp = ev.get("competitions", [{}])[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            hs   = int(float(home.get("score", 0)))
            as_  = int(float(away.get("score", 0)))
            games.append({
                "home_team":  home["team"].get("displayName", ""),
                "away_team":  away["team"].get("displayName", ""),
                "home_abbr":  home["team"].get("abbreviation", "").upper(),
                "away_abbr":  away["team"].get("abbreviation", "").upper(),
                "home_score": hs,
                "away_score": as_,
                "total":      hs + as_,
                "run_diff":   hs - as_,
            })
        except Exception:
            continue

    print(f"  ESPN: {len(games)} completed games for {game_date}")
    return games


def _name_match(a: str, b: str) -> bool:
    """Loose match: a is in b, b is in a, or first 3 chars match."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a or (len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3])


def _find_game(matchup: str, results: list) -> dict | None:
    """Parse 'away @ home' matchup string and find matching ESPN result."""
    parts = matchup.split("@")
    if len(parts) != 2:
        return None
    at_name = parts[0].strip()
    ht_name = parts[1].strip()

    for g in results:
        hm = _name_match(ht_name, g["home_team"]) or _name_match(ht_name, g["home_abbr"])
        am = _name_match(at_name, g["away_team"]) or _name_match(at_name, g["away_abbr"])
        if hm and am:
            return g
    return None


def _grade_row(row: pd.Series, results: list) -> dict | None:
    """
    Grade a single picks-CSV row against ESPN results.
    Returns a dict with RESULT, PNL, HOME_SCORE, AWAY_SCORE, or None if
    the game can't be matched / isn't finished.
    """
    matchup = str(row.get("MATCHUP", ""))
    signal  = str(row.get("BET_SIGNAL", ""))
    market  = str(row.get("MARKET", "ML")).upper()

    game = _find_game(matchup, results)
    if game is None:
        return None

    hs, as_ = game["home_score"], game["away_score"]
    tot     = game["total"]
    result  = None

    if market == "ML":
        # Extract abbreviation from "(NYY) ML" in BET_SIGNAL
        m = re.search(r'\(([A-Z]{2,4})\)', signal)
        abbr = m.group(1) if m else ""
        home_abbr = _abbr(game["home_team"])
        away_abbr = _abbr(game["away_team"])
        if abbr == home_abbr or _name_match(abbr, game["home_abbr"]):
            result = "WIN" if hs > as_ else "LOSS"
        elif abbr == away_abbr or _name_match(abbr, game["away_abbr"]):
            result = "WIN" if as_ > hs else "LOSS"
        else:
            # Fallback: try matching against full names in the matchup
            at_name = matchup.split("@")[0].strip()
            ht_name = matchup.split("@")[1].strip() if "@" in matchup else ""
            if _name_match(abbr, ht_name):
                result = "WIN" if hs > as_ else "LOSS"
            else:
                result = "WIN" if as_ > hs else "LOSS"

    elif market in ("SPREAD", "RL"):
        m = re.search(r'\(([A-Z]{2,4})\)', signal)
        abbr = m.group(1) if m else ""
        home_abbr = _abbr(game["home_team"])
        is_home = (abbr == home_abbr or _name_match(abbr, game["home_abbr"]))
        if is_home:
            push   = abs((hs - as_) - 1.5) < 0.01
            result = "PUSH" if push else ("WIN" if (hs - as_) > 1.5 else "LOSS")
        else:
            push   = abs((as_ - hs) - 1.5) < 0.01
            result = "PUSH" if push else ("WIN" if (as_ - hs) > 1.5 else "LOSS")

    elif market == "TOTAL":
        dm = re.search(r'\b(OVER|UNDER)\b', signal, re.IGNORECASE)
        direction = dm.group(1).upper() if dm else "OVER"
        lm = re.search(r'(?:OVER|UNDER)\s+([\d.]+)', signal, re.IGNORECASE)
        try:
            line = float(lm.group(1)) if lm else float(row.get("VEGAS_SPREAD", 9))
        except (TypeError, ValueError):
            line = 9.0
        if direction == "OVER":
            result = "WIN" if tot > line else ("PUSH" if tot == line else "LOSS")
        else:
            result = "WIN" if tot < line else ("PUSH" if tot == line else "LOSS")

    if result is None:
        return None

    # Calculate PNL
    try:
        odds_str = str(row.get("ODDS", "-110")).replace("+", "")
        odds_val = float(odds_str)
        units    = float(row.get("UNITS", 1))
        if result == "WIN":
            pnl = units * (odds_val / 100) if odds_val > 0 else units * (100 / abs(odds_val))
        elif result == "LOSS":
            pnl = -units
        else:
            pnl = 0.0
    except (TypeError, ValueError):
        pnl = 0.0

    return {
        "RESULT":     result,
        "PNL":        round(pnl, 3),
        "HOME_SCORE": hs,
        "AWAY_SCORE": as_,
        "GRADED_AT":  datetime.now().isoformat(),
    }


def grade_yesterdays_bets(yesterday: str) -> pd.DataFrame:
    """
    Load yesterday's picks CSV, fetch ESPN scores, grade each bet.
    Returns DataFrame of graded rows, empty if nothing to grade.
    """
    date_num = yesterday.replace("-", "")

    # Look in local Results dir first, then GPP mirror
    picks_csv = os.path.join(RESULTS_DIR, f"mlb_picks_{date_num}.csv")
    if not os.path.exists(picks_csv):
        picks_csv = os.path.join(GPP_MLB_PICKS_DIR, f"mlb_picks_{date_num}.csv")
    if not os.path.exists(picks_csv):
        print(f"  No picks CSV found for {yesterday} — skipping grading")
        return pd.DataFrame()

    picks_df     = pd.read_csv(picks_csv)
    espn_results = _fetch_espn_scores(yesterday)
    if not espn_results:
        return pd.DataFrame()

    graded_rows = []
    for _, row in picks_df.iterrows():
        grading = _grade_row(row, espn_results)
        if grading is None:
            print(f"  [skip] No match: {row.get('MATCHUP', '?')}  {row.get('BET_SIGNAL','?')[:50]}")
            continue

        graded_rows.append({
            "DATE":       yesterday,
            "MATCHUP":    row.get("MATCHUP", ""),
            # BET field must match regex r'\(([A-Z]+)\s' for team tracker
            "BET":        str(row.get("BET_SIGNAL", "")),
            "RESULT":     grading["RESULT"],
            "PNL":        grading["PNL"],
            "UNITS":      int(row.get("UNITS", 1)),
            "ODDS":       str(row.get("ODDS", "—")),
            "MARKET":     str(row.get("MARKET", "ML")).upper(),
            "BOOK":       str(row.get("BOOK", "")),
            "EDGE":       str(row.get("SPREAD_EDGE", "")),
            "HOME_SCORE": grading["HOME_SCORE"],
            "AWAY_SCORE": grading["AWAY_SCORE"],
            "GRADED_AT":  grading["GRADED_AT"],
        })

        icon = "✓" if grading["RESULT"] == "WIN" else ("~" if grading["RESULT"] == "PUSH" else "✗")
        print(f"  {icon} {grading['RESULT']:4s}  {grading['PNL']:+.2f}u  "
              f"{row.get('MATCHUP','?')}  [{str(row.get('BET_SIGNAL',''))[:40]}]")

    if not graded_rows:
        print("  No bets could be graded.")
    return pd.DataFrame(graded_rows) if graded_rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. APPEND TO bet_results.csv
# ═══════════════════════════════════════════════════════════════════════════════

def append_results(graded_df: pd.DataFrame, yesterday: str):
    """
    Merge newly-graded rows into bet_results.csv (local + GPP mirror).
    Deduplicates by DATE + MATCHUP + BET to prevent double-counting on re-runs.
    Replaces any pending rows for yesterday with the now-graded versions.
    """
    if graded_df.empty:
        return

    dedup_keys = ["DATE", "MATCHUP", "BET"]

    for results_path in [
        os.path.join(RESULTS_DIR, "bet_results.csv"),
        GPP_MLB_RESULTS_FILE,
    ]:
        Path(os.path.dirname(results_path)).mkdir(parents=True, exist_ok=True)

        if os.path.exists(results_path):
            existing = pd.read_csv(results_path)
            # Remove any pending/ungraded rows for yesterday so they get replaced
            is_yesterday = existing.get("DATE", pd.Series(dtype=str)) == yesterday
            is_ungraded  = ~existing.get("RESULT", pd.Series(dtype=str)).isin(["WIN", "LOSS", "PUSH"])
            existing = existing[~(is_yesterday & is_ungraded)]
            # Deduplicate: don't add rows whose key already exists
            ex_keys  = existing[dedup_keys].apply(tuple, axis=1)
            new_keys = graded_df[dedup_keys].apply(tuple, axis=1)
            new_only = graded_df[~new_keys.isin(ex_keys)]
            combined = pd.concat([existing, new_only], ignore_index=True)
        else:
            combined = graded_df.copy()

        combined.to_csv(results_path, index=False)
        wins      = (combined["RESULT"] == "WIN").sum()
        losses    = (combined["RESULT"] == "LOSS").sum()
        total_pnl = combined["PNL"].sum()
        print(f"  Results → {results_path}")
        print(f"    Record: {wins}W-{losses}L  |  Season P&L: {total_pnl:+.2f}u  "
              f"({len(combined)} graded bets total)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRIGGER PAGE REBUILD
# ═══════════════════════════════════════════════════════════════════════════════

def trigger_page_rebuild():
    """Dynamically import and call generate_picks_page.main()."""
    if not os.path.exists(GPP_PATH):
        print(f"  [warn] generate_picks_page.py not found:")
        print(f"    {GPP_PATH}")
        print("  Page rebuild skipped — ensure the NBA Model Files folder is set up.")
        return False
    try:
        spec = importlib.util.spec_from_file_location("generate_picks_page", GPP_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        print("  Page rebuilt and pushed to GitHub → ktobttv.github.io")
        return True
    except Exception as e:
        print(f"  [warn] Page rebuild failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(grade_only: bool = False):
    print("\n── MLB Page Generator ──")

    today     = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # ── Step 1: Write today's picks ───────────────────────────────────────────
    if not grade_only:
        if not os.path.exists(PICKS_JSON):
            print(f"  [warn] picks_today.json not found at:\n    {PICKS_JSON}")
            print("  Run: python run_model.py")
        else:
            with open(PICKS_JSON) as f:
                data = json.load(f)
            games     = data.get("games", [])
            date_str  = data.get("date", today)
            total_p   = data.get("total_picks", 0)
            generated = data.get("generated_at", "")
            print(f"  {total_p} picks for {date_str}  (generated {generated[:16]})")
            write_picks_csv(games, date_str)

    # ── Step 2: Grade yesterday's bets ───────────────────────────────────────
    print(f"\n  Grading bets for {yesterday}...")
    graded_df = grade_yesterdays_bets(yesterday)

    # ── Step 3: Append to results CSV ─────────────────────────────────────────
    if not graded_df.empty:
        print(f"\n  Writing {len(graded_df)} graded bets to results CSV...")
        append_results(graded_df, yesterday)
    else:
        print("  Nothing new to append to results.")

    # ── Step 4: Rebuild the picks page ───────────────────────────────────────
    print("\n  Triggering page rebuild...")
    trigger_page_rebuild()

    print("── MLB Page Generator done ──\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB picks page generator")
    parser.add_argument("--grade-only", action="store_true",
                        help="Skip picks CSV step — only grade yesterday + push")
    args = parser.parse_args()
    main(grade_only=args.grade_only)
