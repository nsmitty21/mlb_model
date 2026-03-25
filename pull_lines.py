"""
pull_lines.py - MLB Line Puller (FanDuel + DraftKings)
Fetches today's MLB lines from The Odds API.
Outputs today_lines.json (structured for run_model.py).
Usage:
    python pull_lines.py
    python pull_lines.py --date 2026-03-28
"""

import os, sys, json, argparse
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from config import (ODDS_API_KEY, ODDS_API_BASE, SPORT_KEY,
                    TARGET_BOOKS, LIVE_DIR, MARKETS, LINES_PATH)

Path(LIVE_DIR).mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def american_to_prob(odds):
    if odds is None:
        return None
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def fetch_lines():
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    ",".join(MARKETS),
        "oddsFormat": "american",
        "bookmakers": ",".join(TARGET_BOOKS),
    }

    print("Fetching MLB odds (FanDuel + DraftKings)...")
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] Odds API: {e}")
        sys.exit(1)

    remaining = r.headers.get("x-requests-remaining", "?")
    used      = r.headers.get("x-requests-used", "?")
    print(f"  Credits used: {used}  |  Remaining: {remaining}")

    events = r.json()
    if not events:
        print("[WARN] No events returned from Odds API.")
        return []

    print(f"  {len(events)} games found")
    return events


def parse_events(events, target_date):
    """
    Convert raw Odds API events into the structured game list that
    run_model.py expects:
      { game_id, date, commence, home_team, away_team,
        spread_home, total,
        books: { fanduel: {...}, draftkings: {...} } }
    """
    games = {}

    for ev in events:
        game_id   = ev["id"]
        home_team = ev["home_team"]
        away_team = ev["away_team"]
        commence  = ev.get("commence_time", "")

        # Convert commence to Eastern date
        try:
            game_date = (pd.to_datetime(commence, utc=True)
                         .tz_convert("US/Eastern").date().isoformat())
        except Exception:
            game_date = target_date

        if game_id not in games:
            games[game_id] = {
                "game_id":   game_id,
                "date":      game_date,
                "commence":  commence,
                "home_team": home_team,
                "away_team": away_team,
                "spread_home": None,
                "total":       None,
                "books":       {},
            }

        bm_map = {b["key"]: b for b in ev.get("bookmakers", [])}

        for book in TARGET_BOOKS:
            bm = bm_map.get(book)
            if not bm:
                continue

            entry = {}
            for market in bm.get("markets", []):
                key = market["key"]

                if key == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_team:
                            entry["ml_home"] = o["price"]
                        else:
                            entry["ml_away"] = o["price"]

                elif key == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == home_team:
                            games[game_id]["spread_home"] = o["point"]
                            entry["spread_home"]       = o["point"]
                            entry["spread_juice_home"] = o.get("price")
                        else:
                            entry["spread_juice_away"] = o.get("price")

                elif key == "totals":
                    for o in market["outcomes"]:
                        if o["name"] == "Over":
                            games[game_id]["total"]    = o["point"]
                            entry["total"]             = o["point"]
                            entry["total_over_juice"]  = o.get("price")
                        else:
                            entry["total_under_juice"] = o.get("price")

            games[game_id]["books"][book] = entry

    # Filter to target date
    filtered = [g for g in games.values() if g["date"] == target_date]

    if not filtered:
        all_dates = sorted(set(g["date"] for g in games.values()))
        print(f"  [WARN] No games on {target_date}. Available dates: {all_dates}")
        # Fall back to nearest future date
        future = [d for d in all_dates if d >= target_date]
        if future:
            fallback = future[0]
            print(f"  -> Falling back to {fallback}")
            filtered = [g for g in games.values() if g["date"] == fallback]

    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args, _ = parser.parse_known_args()
    target_date = args.date or date.today().isoformat()

    print(f"MLB Lines Pull — {target_date}")

    events  = fetch_lines()
    if not events:
        print("No events available.")
        sys.exit(0)

    games = parse_events(events, target_date)

    if not games:
        print("No games found for today.")
        sys.exit(0)

    # Save JSON for run_model.py
    with open(LINES_PATH, "w") as f:
        json.dump(games, f, indent=2)

    # Pretty print summary
    print(f"\n  {'AWAY':<25} {'HOME':<25} {'FD ML':>7} {'DK ML':>7} {'SPR':>6} {'TOT':>6}  TIME")
    print("  " + "-" * 95)
    for g in sorted(games, key=lambda x: x["commence"]):
        fd = g["books"].get("fanduel", {})
        dk = g["books"].get("draftkings", {})
        fd_ml = f"{int(fd['ml_home']):+d}" if fd.get("ml_home") is not None else "n/a"
        dk_ml = f"{int(dk['ml_home']):+d}" if dk.get("ml_home") is not None else "n/a"
        spr   = f"{g['spread_home']:+.1f}" if g.get("spread_home") is not None else "n/a"
        tot   = str(g["total"])            if g.get("total")       is not None else "n/a"
        try:
            t = (pd.to_datetime(g["commence"], utc=True)
                 .tz_convert("US/Eastern").strftime("%-I:%M %p ET"))
        except Exception:
            t = ""
        print(f"  {g['away_team']:<25} {g['home_team']:<25} {fd_ml:>7} {dk_ml:>7} {spr:>6} {tot:>6}  {t}")

    print(f"\n  Saved {len(games)} games to today_lines.json")
    print("  Run next: python run_model.py")


if __name__ == "__main__":
    main()