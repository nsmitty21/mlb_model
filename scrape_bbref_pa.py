"""
scrape_bbref_pa.py — MLB plate appearance scraper (MLB Stats API)
=================================================================
Replaces the Baseball Reference scraper. BBRef returns 403 due to
bot-detection. This version uses the official MLB Stats API instead:

    https://statsapi.mlb.com/api/v1/...

The API is completely free, requires no key, is what BBRef itself
pulls from, and explicitly permits non-commercial use. It returns
structured JSON so no HTML parsing is needed.

Output schema
-------------
Identical to your existing plate_appearances_raw_{year}.csv files:
  game_date, year, bbref_game_id, inning, inning_half, score,
  outs_before_play, runners_on_base, batter, pitcher,
  batting_team_home, play_description, batted_ball_type,
  PA, AB, H, 1B, 2B, 3B, HR, BB, IBB, HBP, K, SF, SAC, FC, E, DP

Note on game IDs
----------------
The MLB API uses numeric gamePk values (e.g. 748531), not BBRef-style
IDs (e.g. NYA202604010). This script stores them prefixed with "mlb_"
(e.g. "mlb_748531") so the cache key is always unique. Your existing
BBRef-sourced rows keep their original IDs — no conflict.

Usage
-----
  python scrape_bbref_pa.py                    # fill gaps for 2026
  python scrape_bbref_pa.py --year 2025        # backfill 2025
  python scrape_bbref_pa.py --dry-run          # preview only
  python scrape_bbref_pa.py --full-reset       # re-scrape entire year
  python scrape_bbref_pa.py --start 2026-03-20 # only from a date onward
"""

import argparse
import csv
import os
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

try:
    from config import HIST_ODDS_DIR, LIVE_SEASON
    OUT_DIR    = Path(HIST_ODDS_DIR)
    CUR_SEASON = LIVE_SEASON
except ImportError:
    OUT_DIR    = Path("data/historical/odds")
    CUR_SEASON = date.today().year

OUT_DIR.mkdir(parents=True, exist_ok=True)

MLB_API     = "https://statsapi.mlb.com/api/v1"
DELAY       = 1.0   # seconds between requests
JITTER      = 0.5
MAX_RETRIES = 4

HEADERS = {"User-Agent": "MLB-Model-Research/1.0 (personal project)"}

SEASON_START = {
    2021: "2021-04-01", 2022: "2022-04-07", 2023: "2023-03-30",
    2024: "2024-03-20", 2025: "2025-03-27", 2026: "2026-03-25",
}
SEASON_END = {
    2021: "2021-11-03", 2022: "2022-11-05", 2023: "2023-11-01",
    2024: "2024-10-30", 2025: "2025-10-01",
    2026: date.today().isoformat(),
}

PA_COLS = [
    "game_date", "year", "bbref_game_id",
    "inning", "inning_half", "score",
    "outs_before_play", "runners_on_base",
    "batter", "pitcher", "batting_team_home",
    "play_description", "batted_ball_type",
    "PA", "AB", "H", "1B", "2B", "3B", "HR",
    "BB", "IBB", "HBP", "K", "SF", "SAC", "FC", "E", "DP",
]

_PA_EVENTS = {
    "single", "double", "triple", "home_run",
    "walk", "intent_walk", "hit_by_pitch",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play",
    "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
    "fielders_choice", "fielders_choice_out",
    "field_error", "catcher_interf",
}

_BATTED_BALL_MAP = {
    "ground": "groundball", "fly": "flyball",
    "line":   "line_drive", "popup": "popup",
    "bunt":   "bunt",
}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(url, params=None):
    """Polite GET with exponential back-off. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(max(0.2, DELAY + random.uniform(-JITTER, JITTER)))
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    [rate-limit] sleeping {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            print(f"    [HTTP {r.status_code}] {url}")
            return None
        except requests.RequestException as e:
            wait = 5 * (2 ** attempt)
            print(f"    [error] {e} — retry in {wait}s")
            time.sleep(wait)
    return None


# ── Game discovery ────────────────────────────────────────────────────────────

def get_completed_games(year, start_override=None):
    """
    Query MLB schedule API for all completed regular-season games.
    Returns list of dicts: game_pk, game_date, game_id, home_team, away_team, home_abbr
    """
    season_start = start_override or SEASON_START.get(year, f"{year}-04-01")
    season_end   = min(SEASON_END.get(year, f"{year}-10-01"), date.today().isoformat())

    print(f"  Querying schedule: {season_start} -> {season_end}")

    data = _get(f"{MLB_API}/schedule", params={
        "sportId":   1,
        "startDate": season_start,
        "endDate":   season_end,
        "gameType":  "R",
        "hydrate":   "team",
        "fields": (
            "dates,date,games,gamePk,status,abstractGameState,"
            "teams,home,away,team,name,abbreviation"
        ),
    })

    if data is None:
        print("  [ERROR] Schedule API call failed.")
        return []

    games = []
    for date_entry in data.get("dates", []):
        game_date = date_entry["date"]
        for g in date_entry.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue
            pk = g["gamePk"]
            games.append({
                "game_pk":   pk,
                "game_date": game_date,
                "game_id":   f"mlb_{pk}",
                "home_team": g["teams"]["home"]["team"]["name"],
                "away_team": g["teams"]["away"]["team"]["name"],
                "home_abbr": g["teams"]["home"]["team"].get("abbreviation", ""),
            })

    print(f"  Found {len(games)} completed regular-season games")
    return games


# ── Play-by-play parser ───────────────────────────────────────────────────────

def _batted_ball_type(trajectory, description):
    for src in (trajectory or "", description or ""):
        for k, v in _BATTED_BALL_MAP.items():
            if k in src.lower():
                return v
    return None


def _runners_string(runners_before):
    """Build '1__', '12_', '___' etc from runners-on-base before the play."""
    occupied = set()
    for r in runners_before:
        start = r.get("movement", {}).get("start") or ""
        if start in ("1B", "2B", "3B"):
            occupied.add(start[0])
    return "".join(b if b in occupied else "_" for b in ("1", "2", "3"))


def fetch_game_plays(meta):
    """
    Fetch one game's play-by-play from the MLB API.
    Returns list of row dicts matching PA_COLS.
    """
    pk        = meta["game_pk"]
    game_date = meta["game_date"]
    game_id   = meta["game_id"]

    data = _get(f"{MLB_API}/game/{pk}/playByPlay")
    if data is None:
        return []

    rows = []
    for play in data.get("allPlays", []):
        result = play.get("result", {})
        about  = play.get("about",  {})
        event  = result.get("eventType", "")

        if event not in _PA_EVENTS:
            continue

        matchup = play.get("matchup", {})
        batter  = matchup.get("batter",  {}).get("fullName", "")
        pitcher = matchup.get("pitcher", {}).get("fullName", "")
        if not batter or not pitcher:
            continue

        inning      = str(about.get("inning", 0))
        half        = about.get("halfInning", "top")
        inning_half = "top" if half == "top" else "bot"
        outs_before = play.get("count", {}).get("outs", 0)

        home_score = about.get("homeScore", 0) or 0
        away_score = about.get("awayScore", 0) or 0
        score      = f"{away_score}-{home_score}"

        runners_str       = _runners_string(play.get("runners", []))
        batting_team_home = 0 if half == "top" else 1
        play_desc         = result.get("description", "")

        hit_data  = play.get("hitData", {})
        bb_type   = _batted_ball_type(hit_data.get("trajectory", ""), play_desc)

        is_hit    = event in ("single", "double", "triple", "home_run")
        is_bb     = event in ("walk", "intent_walk")
        is_ibb    = event == "intent_walk"
        is_hbp    = event == "hit_by_pitch"
        is_k      = "strikeout" in event
        is_sf     = "sac_fly"   in event
        is_sac    = "sac_bunt"  in event
        is_fc     = "fielders_choice" in event
        is_error  = event == "field_error"
        is_dp     = "double_play" in event
        is_ab     = not (is_bb or is_ibb or is_hbp or is_sf or is_sac)

        rows.append({
            "game_date":         game_date,
            "year":              int(game_date[:4]),
            "bbref_game_id":     game_id,
            "inning":            inning,
            "inning_half":       inning_half,
            "score":             score,
            "outs_before_play":  outs_before,
            "runners_on_base":   runners_str,
            "batter":            batter,
            "pitcher":           pitcher,
            "batting_team_home": batting_team_home,
            "play_description":  play_desc,
            "batted_ball_type":  bb_type,
            "PA":  1,
            "AB":  int(is_ab),
            "H":   int(is_hit),
            "1B":  int(event == "single"),
            "2B":  int(event == "double"),
            "3B":  int(event == "triple"),
            "HR":  int(event == "home_run"),
            "BB":  int(is_bb),
            "IBB": int(is_ibb),
            "HBP": int(is_hbp),
            "K":   int(is_k),
            "SF":  int(is_sf),
            "SAC": int(is_sac),
            "FC":  int(is_fc),
            "E":   int(is_error),
            "DP":  int(is_dp),
        })

    return rows


# ── Cache ─────────────────────────────────────────────────────────────────────

def load_cached_game_ids(year):
    csv_path = OUT_DIR / f"plate_appearances_raw_{year}.csv"
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["bbref_game_id"], low_memory=False)
        return set(df["bbref_game_id"].dropna().astype(str).unique())
    except Exception as e:
        print(f"  [warn] Could not read cache: {e}")
        return set()


def append_rows(year, rows):
    if not rows:
        return
    csv_path    = OUT_DIR / f"plate_appearances_raw_{year}.csv"
    write_hdr   = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PA_COLS, extrasaction="ignore")
        if write_hdr:
            w.writeheader()
        w.writerows(rows)


def rebuild_csv(year, all_rows):
    csv_path = OUT_DIR / f"plate_appearances_raw_{year}.csv"
    pd.DataFrame(all_rows, columns=PA_COLS).to_csv(csv_path, index=False)
    print(f"  Saved {len(all_rows):,} rows -> {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(year, full_reset=False, dry_run=False, start_override=None):
    print(f"\n{'='*60}")
    print(f"  MLB Stats API PA Scraper -- {year}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Mode:   {'FULL RESET' if full_reset else 'incremental'}")
    print(f"{'='*60}\n")

    all_games = get_completed_games(year, start_override=start_override)
    if not all_games:
        print("[ERROR] No games returned. Check network or add year to SEASON_START dict.")
        return

    if full_reset:
        missing = all_games
        print(f"  Full reset -- will re-scrape all {len(missing)} games")
    else:
        cached  = load_cached_game_ids(year)
        missing = [g for g in all_games if g["game_id"] not in cached]
        print(f"  Cached: {len(all_games)-len(missing)}  "
              f"Missing: {len(missing)}  Total: {len(all_games)}")

    if not missing:
        print("  Cache is complete -- nothing to fetch.")
        return

    if dry_run:
        print(f"\n  [dry-run] Would fetch {len(missing)} games:")
        for g in missing[:15]:
            print(f"    {g['game_date']}  {g['away_team']} @ {g['home_team']}  ({g['game_id']})")
        if len(missing) > 15:
            print(f"    ... and {len(missing)-15} more")
        est = len(missing) * (DELAY + JITTER) / 60
        print(f"\n  Estimated time: ~{est:.0f} min at {DELAY}s/request")
        return

    print(f"\n  Fetching {len(missing)} games ...\n")
    all_new, failed = [], []

    for i, meta in enumerate(missing, 1):
        label = f"{meta['away_team']} @ {meta['home_team']}"
        print(f"  [{i:4d}/{len(missing)}] {meta['game_date']}  {label}", end=" ... ")
        sys.stdout.flush()

        rows = fetch_game_plays(meta)
        if not rows:
            print("FAILED")
            failed.append(meta["game_id"])
            continue

        print(f"{len(rows)} PAs")
        if full_reset:
            all_new.extend(rows)
        else:
            append_rows(year, rows)

    if full_reset and all_new:
        rebuild_csv(year, all_new)

    fetched = len(missing) - len(failed)
    print(f"\n{'='*60}")
    print(f"  Done.  Fetched: {fetched}  Failed: {len(failed)}")
    if failed:
        for gid in failed[:20]:
            print(f"    {gid}")
    csv_path = OUT_DIR / f"plate_appearances_raw_{year}.csv"
    if csv_path.exists():
        n = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        print(f"  Total rows in {csv_path.name}: {n:,}")
    print(f"{'='*60}\n")
    if fetched > 0:
        print("  Run next: python fetch_historical.py  (to rebuild BvP + lineup cache)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape MLB play-by-play via the official MLB Stats API."
    )
    parser.add_argument("--year", type=int, default=CUR_SEASON,
                        help=f"Season year (default: {CUR_SEASON})")
    parser.add_argument("--full-reset", action="store_true",
                        help="Re-scrape entire year, overwriting local CSV.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be fetched without scraping.")
    parser.add_argument("--start", default=None, metavar="YYYY-MM-DD",
                        help="Only fetch games on or after this date.")
    args = parser.parse_args()
    run(year=args.year, full_reset=args.full_reset,
        dry_run=args.dry_run, start_override=args.start)
