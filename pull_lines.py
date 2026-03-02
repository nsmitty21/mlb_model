"""
pull_lines.py - MLB Line Puller (FanDuel + DraftKings)
Fetches today's MLB lines from The Odds API.
Usage:
    python pull_lines.py
    python pull_lines.py --date 2026-03-28
"""

import os, sys, argparse
import requests
import pandas as pd
from datetime import date
from pathlib import Path
from config import ODDS_API_KEY, ODDS_API_BASE, SPORT_KEY, TARGET_BOOKS, LIVE_DIR, MARKETS

SEASON_START = "2026-03-27"

Path(LIVE_DIR).mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT  = f"{LIVE_DIR}/today_lines.csv"


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

    print(f"Fetching MLB odds (FanDuel + DraftKings)...")
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
        print("[ERROR] No events returned from Odds API.")
        sys.exit(1)

    print(f"  {len(events)} games found")
    rows = []
    today_str = date.today().isoformat()

    for ev in events:
        home_team = ev["home_team"]
        away_team = ev["away_team"]
        game_id   = ev["id"]
        commence  = ev.get("commence_time", "")

        bm_map = {b["key"]: b for b in ev.get("bookmakers", [])}

        for book in TARGET_BOOKS:
            bm = bm_map.get(book)
            if not bm:
                continue

            spread_home = spread_juice_h = spread_juice_a = None
            ml_home = ml_away = None
            total_line = total_over_odds = total_under_odds = None

            for market in bm.get("markets", []):
                key = market["key"]

                if key == "h2h":
                    for o in market["outcomes"]:
                        if o["name"] == home_team:
                            ml_home = o["price"]
                        else:
                            ml_away = o["price"]

                elif key == "spreads":
                    for o in market["outcomes"]:
                        if o["name"] == home_team:
                            spread_home    = o["point"]
                            spread_juice_h = o.get("price")
                        else:
                            spread_juice_a = o.get("price")

                elif key == "totals":
                    for o in market["outcomes"]:
                        if o["name"] == "Over":
                            total_line      = o["point"]
                            total_over_odds = o.get("price")
                        else:
                            total_under_odds = o.get("price")

            rows.append({
                "date":              today_str,
                "commence":          commence,
                "book":              book,
                "away_team":         away_team,
                "home_team":         home_team,
                "spread_home":       spread_home,
                "spread_juice_home": spread_juice_h,
                "spread_juice_away": spread_juice_a,
                "ml_home":           ml_home,
                "ml_away":           ml_away,
                "total_line":        total_line,
                "total_over_odds":   total_over_odds,
                "total_under_odds":  total_under_odds,
                "implied_prob_home": american_to_prob(ml_home),
                "implied_prob_away": american_to_prob(ml_away),
                "game_id":           game_id,
            })

    return pd.DataFrame(rows)


def filter_to_date(df, target_date):
    if df.empty or "commence" not in df.columns:
        return df
    df["_dt"] = pd.to_datetime(df["commence"], utc=True).dt.tz_convert("US/Eastern").dt.date.astype(str)
    counts = df["_dt"].value_counts().sort_index()
    print(f"\n  Games by date:")
    for d, c in counts.items():
        marker = " <-- selected" if d == target_date else ""
        print(f"    {d}  ->  {c // len(TARGET_BOOKS)} games{marker}")

    filtered = df[df["_dt"] == target_date].copy()
    if filtered.empty:
        fallback = counts.idxmax()
        print(f"\n  [warn] No games for {target_date} -- falling back to {fallback}")
        filtered = df[df["_dt"] == fallback].copy()
        filtered["date"] = fallback
    else:
        filtered["date"] = target_date
    return filtered.drop(columns=["_dt"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args, _ = parser.parse_known_args()
    target_date = args.date or date.today().isoformat()

    # Block spring training
    if target_date < SEASON_START:
        print(f"\n  MLB regular season starts {SEASON_START}.")
        print(f"  No lines to pull until opening day.")
        print(f"  Come back March 27!")
        sys.exit(0)

    df = fetch_lines()
    df = filter_to_date(df, target_date)
    df.to_csv(OUTPUT, index=False)

    # Pretty print
    games = df[df["book"] == TARGET_BOOKS[0]].sort_values("commence")
    print(f"\n  {'AWAY':<22} {'HOME':<22} {'FD_ML':>7} {'DK_ML':>7} {'SPREAD':>7} {'TOTAL':>7}  TIME")
    print("  " + "-" * 90)
    for _, r in games.iterrows():
        fd = df[(df["game_id"]==r["game_id"]) & (df["book"]=="fanduel")]
        dk = df[(df["game_id"]==r["game_id"]) & (df["book"]=="draftkings")]
        fd_ml = f"{int(fd['ml_home'].iloc[0]):+d}"  if not fd.empty and pd.notna(fd['ml_home'].iloc[0])  else "n/a"
        dk_ml = f"{int(dk['ml_home'].iloc[0]):+d}"  if not dk.empty and pd.notna(dk['ml_home'].iloc[0])  else "n/a"
        spr   = f"{r['spread_home']:+.1f}"           if pd.notna(r.get('spread_home'))                   else "n/a"
        tot   = f"{r['total_line']}"                 if pd.notna(r.get('total_line'))                    else "n/a"
        try:
            t = pd.to_datetime(r["commence"], utc=True).tz_convert("US/Eastern").strftime("%-I:%M %p")
        except Exception:
            t = ""
        print(f"  {r['away_team']:<22} {r['home_team']:<22} {fd_ml:>7} {dk_ml:>7} {spr:>7} {tot:>7}  {t}")

    print(f"\n  Saved {len(games)} games to {OUTPUT}")
    print("  Run next: python run_model.py")


if __name__ == "__main__":
    main()