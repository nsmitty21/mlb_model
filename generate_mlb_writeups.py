"""
generate_mlb_writeups.py — MLB X (Twitter) Thread Generator
=============================================================
Reads:   data/live/picks_today.json  (written by run_today.py)
Writes:  MLB/Writeups/YYYY-MM-DD.txt

Thread structure:
  Tweet 1  — Opening hook (record, pick count)
  Tweet 2+ — One tweet per GAME (not per pick).
              Each game tweet covers all picks for that matchup:
              teams, starting pitchers, and every bet with edge/odds.
  Last     — Closing CTA

Called automatically by run_today.py after saving picks_today.json,
and also by generate_mlb_page.py.  Can be run standalone:
  python generate_mlb_writeups.py

Voice: confident, transparent, shows the math.  The starting pitcher
matchup and model edge numbers are the differentiator.
"""

import json, os
from datetime import datetime, date
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
from config import LIVE_DIR, BASE_DIR

PICKS_JSON   = Path(LIVE_DIR) / "picks_today.json"
NBA_OUTPUT   = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\ML and Spread"
WRITEUPS_DIR = Path(NBA_OUTPUT) / "MLB" / "Writeups"

# Also mirror into MLB Model Files for local record-keeping
LOCAL_WRITEUPS_DIR = Path(BASE_DIR) / "Writeups"

# ── Helpers ────────────────────────────────────────────────────────────────────

UNIT_STARS = {3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}

def _stars(units: int) -> str:
    return UNIT_STARS.get(int(units), "⭐")

def _fmt_odds(odds) -> str:
    try:
        v = int(float(odds))
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"

def _short_pitcher(name: str) -> str:
    """'Gerrit Cole' → 'G. Cole'"""
    if not name or name.strip() in ("TBD", "", "Unknown"):
        return "TBD"
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return name

def _game_time_et(commence: str) -> str:
    try:
        import pandas as pd
        utc = pd.to_datetime(commence, utc=True)
        et  = utc.tz_convert("US/Eastern")
        s   = et.strftime("%I:%M %p ET")
        return s.lstrip("0") or s
    except Exception:
        return ""

def _market_label(pick: dict) -> str:
    """
    Build a compact pick label for one line of the game tweet.
    Examples:
      RL  ATL -1.5 (+135 DK)   ⭐⭐⭐
      ML  DET (+118 FD)         ⭐⭐
      TOT U 8.5 (-108 FD)       ⭐⭐
    """
    market = str(pick.get("market", "ML")).upper()
    team   = str(pick.get("team", ""))
    odds   = _fmt_odds(pick.get("odds"))
    book   = str(pick.get("book", "")).upper()
    units  = int(pick.get("units", 1))
    edge   = float(pick.get("edge", 0)) * 100
    stars  = _stars(units)

    # Shorten book name
    book_short = "DK" if "DRAFT" in book else ("FD" if "FAN" in book else book[:2])

    if market == "ML":
        # team is full name e.g. "Detroit Tigers"
        name_parts = team.split()
        short = name_parts[-1] if name_parts else team  # "Tigers"
        return f"ML  {short} ({odds} {book_short})  {stars}  Edge:{edge:.1f}%"

    elif market == "RL":
        # team field is e.g. "Atlanta Braves -1.5"
        parts = team.rsplit(" ", 1)
        if len(parts) == 2 and (parts[1].startswith("-") or parts[1].startswith("+")):
            name_parts = parts[0].split()
            short = name_parts[-1] if name_parts else parts[0]
            spread = parts[1]
        else:
            name_parts = team.split()
            short  = name_parts[-1] if name_parts else team
            spread = str(pick.get("spread_line", ""))
        return f"RL  {short} {spread} ({odds} {book_short})  {stars}  Edge:{edge:.1f}%"

    elif market == "TOTAL":
        side = str(pick.get("side", "")).upper()
        line = pick.get("total_line", "")
        prefix = "O" if side == "OVER" else "U"
        return f"TOT {prefix} {line} ({odds} {book_short})  {stars}  Edge:{edge:.1f}%"

    else:
        return f"{market}  {team} ({odds} {book_short})  {stars}  Edge:{edge:.1f}%"


# ── Game tweet builder ─────────────────────────────────────────────────────────

def build_game_tweet(game: dict, game_num: int, total_games: int) -> str:
    """
    One tweet per game — covers ALL picks for that matchup.

    Format:
      ⚾ Game N/N  — AWAY @ HOME  (HH:MM ET)
      🔱 SP: A. Pitcher vs G. Cole
      ─────────────────────────
      RL  Braves -1.5 (+135 DK)  ⭐⭐⭐  Edge:19.0%
      ML  Red Sox (+130 FD)       ⭐⭐⭐  Edge:15.5%
    """
    X_LIMIT = 275

    home  = game.get("home_team", "")
    away  = game.get("away_team", "")
    hsp   = _short_pitcher(game.get("home_sp", "TBD"))
    asp   = _short_pitcher(game.get("away_sp", "TBD"))
    gtime = _game_time_et(game.get("commence", ""))
    picks = game.get("picks", [])

    time_str = f"  ({gtime})" if gtime else ""
    lines = [
        f"⚾ Game {game_num}/{total_games} — {away} @ {home}{time_str}",
        f"🔱 {asp} vs {hsp}",
        "─" * 28,
    ]

    for pick in picks:
        lines.append(_market_label(pick))

    tweet = "\n".join(lines)

    # Trim if over limit — drop game time first, then pitcher line
    if len(tweet) > X_LIMIT:
        lines[0] = f"⚾ Game {game_num}/{total_games} — {away} @ {home}"
        tweet = "\n".join(lines)

    if len(tweet) > X_LIMIT:
        lines.pop(1)   # drop pitcher line
        tweet = "\n".join(lines)

    return tweet


# ── Opening tweet ──────────────────────────────────────────────────────────────

def _load_record() -> dict:
    """Load W-L record from MLB bet_results.csv if available."""
    result = {
        "wins": 0, "losses": 0, "units": 0.0,
        "month_wins": 0, "month_losses": 0, "month_units": 0.0,
        "month_name": datetime.now().strftime("%B"),
        "streak_n": 0, "streak_dir": "W",
    }
    results_paths = [
        Path(NBA_OUTPUT) / "MLB" / "Results" / "bet_results.csv",
        Path(BASE_DIR) / "Results" / "bet_results.csv",
    ]
    for rp in results_paths:
        if not rp.exists():
            continue
        try:
            import pandas as pd
            df = pd.read_csv(rp)
            df = df[df["RESULT"].isin(["WIN", "LOSS", "PUSH"])].copy()
            if df.empty:
                break

            result["wins"]   = int((df["RESULT"] == "WIN").sum())
            result["losses"] = int((df["RESULT"] == "LOSS").sum())
            result["units"]  = round(float(df["PNL"].sum()), 2)

            # Streak
            res_list = df["RESULT"].tolist()
            last_res = res_list[-1]
            streak = sum(1 for r in reversed(res_list) if r == last_res)
            # stop at first non-match
            streak = 0
            for r in reversed(res_list):
                if r == last_res: streak += 1
                else: break
            result["streak_n"]   = streak
            result["streak_dir"] = last_res[0]

            # This month
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            month_start = datetime.now().replace(
                day=1, hour=0, minute=0, second=0, microsecond=0)
            this_month = df[df["DATE"] >= month_start]
            result["month_wins"]   = int((this_month["RESULT"] == "WIN").sum())
            result["month_losses"] = int((this_month["RESULT"] == "LOSS").sum())
            result["month_units"]  = round(float(this_month["PNL"].sum()), 2)
            break
        except Exception as e:
            print(f"  [writeup] load_record: {e}")

    return result


def build_opening_tweet(games_with_picks: list, record: dict, date_str: str) -> str:
    total_picks = sum(len(g.get("picks", [])) for g in games_with_picks)
    total_games = len(games_with_picks)

    w   = record["wins"]
    l   = record["losses"]
    u   = record["units"]
    mw  = record["month_wins"]
    ml  = record["month_losses"]
    mu  = record["month_units"]
    mon = record["month_name"]
    sn  = record["streak_n"]
    sd  = record["streak_dir"]

    u_str  = f"{u:+.1f}u" if u != 0 else "0.0u"
    mu_str = f"{mu:+.1f}u" if mu != 0 else "0.0u"

    if sn > 1:
        streak_str = f"On a {sn}{sd} streak"
    elif sd == "W":
        streak_str = "Fresh off a win"
    else:
        streak_str = "Looking to bounce back"

    pick_word  = "play"  if total_picks == 1 else "plays"
    game_word  = "game"  if total_games == 1 else "games"

    return (
        f"⚾ MLB model locked in — {total_picks} {pick_word} across "
        f"{total_games} {game_word} ({date_str}).\n\n"
        f"📋 All-time: {w}W-{l}L ({u_str})\n"
        f"📅 {mon}: {mw}W-{ml}L ({mu_str})\n"
        f"🔥 {streak_str}\n\n"
        f"Full breakdown below 🧵👇"
    )


# ── Closing tweet ──────────────────────────────────────────────────────────────

def build_closing_tweet(total_picks: int) -> str:
    pick_word = "pick" if total_picks == 1 else "picks"
    return (
        f"That's the card — {total_picks} {pick_word} today. "
        f"Every bet gets graded, wins and losses both posted.\n\n"
        f"Model shows the edge %, starting pitcher matchup, and book odds. "
        f"If you want this every day → follow @iMightBetThat.\n\n"
        f"#MLBBetting #MLBpicks #SportsBetting"
    )


# ── Thread assembler ───────────────────────────────────────────────────────────

def build_thread(picks_json: dict) -> list[str]:
    """
    Build the full thread from picks_today.json.
    Returns a list of tweet strings (opening, one per game, closing).
    """
    games_all = picks_json.get("games", [])
    # Only games that have at least one pick
    games_with_picks = [g for g in games_all if g.get("picks")]

    if not games_with_picks:
        return []

    record       = _load_record()
    date_str     = picks_json.get("date", date.today().isoformat())
    total_games  = len(games_with_picks)
    total_picks  = sum(len(g["picks"]) for g in games_with_picks)

    tweets = []

    # Tweet 1: opening hook
    tweets.append(build_opening_tweet(games_with_picks, record, date_str))

    # One tweet per game
    for i, game in enumerate(games_with_picks, start=1):
        tweets.append(build_game_tweet(game, i, total_games))

    # Final: CTA
    tweets.append(build_closing_tweet(total_picks))

    return tweets


# ── File writer ────────────────────────────────────────────────────────────────

def save_writeup(tweets: list[str], date_str: str) -> Path:
    for wd in [WRITEUPS_DIR, LOCAL_WRITEUPS_DIR]:
        wd.mkdir(parents=True, exist_ok=True)

    filename = WRITEUPS_DIR / f"{date_str}.txt"
    local_fn = LOCAL_WRITEUPS_DIR / f"{date_str}.txt"

    def _write(path: Path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"X THREAD — MLB — {date_str}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Tweets in thread: {len(tweets)}\n")
            f.write("=" * 60 + "\n\n")
            f.write("HOW TO POST:\n")
            f.write("  1. Copy Tweet 1 → post it\n")
            f.write("  2. Reply to your own tweet with each subsequent tweet\n")
            f.write("  3. Last tweet is always the follow CTA\n")
            f.write("\n" + "=" * 60 + "\n\n")

            for i, tweet in enumerate(tweets, start=1):
                if i == 1:
                    label = "TWEET 1 (post this first)"
                elif i == len(tweets):
                    label = f"TWEET {i} — FINAL (follow CTA)"
                else:
                    label = f"TWEET {i}"

                f.write(f"{'─' * 60}\n")
                f.write(f"{label}  [{len(tweet)} chars]\n")
                f.write(f"{'─' * 60}\n")
                f.write(tweet)
                f.write("\n\n")

    _write(filename)
    _write(local_fn)
    print(f"  Writeup → {filename}")
    print(f"  Mirror  → {local_fn}")
    return filename


# ── Preview printer ────────────────────────────────────────────────────────────

def print_preview(tweets: list[str]):
    print(f"\n  WRITEUP PREVIEW ({len(tweets)} tweets):")
    print("  " + "─" * 56)
    for i, tweet in enumerate(tweets, start=1):
        if i == 1:
            label = "Tweet 1 (opening)"
        elif i == len(tweets):
            label = f"Tweet {i} (closing CTA)"
        else:
            label = f"Tweet {i}"
        print(f"\n  ── {label} ──")
        for line in tweet.split("\n"):
            print(f"  {line}")
    print("\n  " + "─" * 56)


# ── Entry points ───────────────────────────────────────────────────────────────

def generate(picks_json: dict | None = None) -> Path | None:
    """
    Main entry point called by run_today.py and generate_mlb_page.py.
    Accepts the already-parsed picks_today.json dict (avoids re-reading the file),
    or reads PICKS_JSON from disk if not provided.
    """
    if picks_json is None:
        if not PICKS_JSON.exists():
            print("  [writeup] picks_today.json not found — skipping writeup")
            return None
        with open(PICKS_JSON) as f:
            picks_json = json.load(f)

    tweets = build_thread(picks_json)
    if not tweets:
        print("  [writeup] No picks with bets — no thread generated")
        return None

    date_str = picks_json.get("date", date.today().isoformat())
    path     = save_writeup(tweets, date_str)
    print_preview(tweets)
    return path


def main():
    generate()


if __name__ == "__main__":
    main()
