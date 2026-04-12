"""
generate_mlb_page.py — MLB Picks + Results Bridge
===================================================
Reads:   data/live/picks_today.json         (written by run_today.py)
Writes:  MLB/Daily Picks CSV/mlb_picks_YYYYMMDD.csv   (read by generate_picks_page.py)
         MLB/Results/bet_results.csv                  (read by generate_picks_page.py)
Then:    calls generate_picks_page.py to rebuild index.html and push to GitHub

Daily workflow:
  python pull_lines.py
  python run_today.py
  python generate_mlb_page.py       ← this file

Optional flags:
  python generate_mlb_page.py --grade-only   # skip writing picks CSV, just grade + push
  python generate_mlb_page.py --no-push      # write everything but skip the git push
"""

import argparse, importlib.util, json, os, re, sys
import numpy as np
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
from config import LIVE_DIR, BASE_DIR

PICKS_JSON   = Path(LIVE_DIR) / "picks_today.json"

# Output dirs — where generate_picks_page.py looks for MLB data
NBA_OUTPUT   = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\ML and Spread"
MLB_PICKS_DIR   = Path(NBA_OUTPUT) / "MLB" / "Daily Picks CSV"
MLB_RESULTS_DIR = Path(NBA_OUTPUT) / "MLB" / "Results"
MLB_RESULTS_CSV = MLB_RESULTS_DIR / "bet_results.csv"

# Also mirror into MLB Model Files/Results for local record-keeping
LOCAL_RESULTS_DIR = Path(BASE_DIR) / "Results"
LOCAL_RESULTS_CSV = LOCAL_RESULTS_DIR / "bet_results.csv"

# Path to generate_picks_page.py
GPP_PATH = Path(NBA_OUTPUT) / "generate_picks_page.py"

# ESPN for grading
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

# Team abbreviation map (full name → 3-letter)
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
    return FULL_TO_ABBR.get(str(name).strip(), str(name).strip()[:3].upper())


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONVERT picks_today.json → mlb_picks_YYYYMMDD.csv
# ═══════════════════════════════════════════════════════════════════════════════

def _units_to_tier(units: int) -> str:
    if units >= 3: return "HIGH CONFIDENCE"
    if units >= 2: return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"

def _fmt_odds(odds) -> str:
    try:
        v = int(float(odds))
        return f"+{v}" if v > 0 else str(v)
    except (TypeError, ValueError):
        return "—"

def _game_time_et(commence: str) -> str:
    try:
        utc = pd.to_datetime(commence, utc=True)
        et  = utc.tz_convert("US/Eastern")
        s   = et.strftime("%I:%M %p ET")
        return s.lstrip("0") or s
    except Exception:
        return ""

def _bet_signal(pick: dict, ht: str, at: str) -> str:
    """
    Build the BET_SIGNAL string parse_signal() in generate_picks_page.py expects.
    Format: "BET (ABB) ML (-142) [HIGH CONFIDENCE]"
    The team-tracker regex needs the abbreviation in parens followed by a space.
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
        # Team field may be "Atlanta Braves -1.5" — grab name portion
        abbr     = _abbr(" ".join(team_str.split()[:-1]) if team_str.split() else team_str)
        spread   = pick.get("spread", pick.get("spread_line", ""))
        try:
            label = f"({abbr}) RL {float(spread):+.1f}"
        except (TypeError, ValueError):
            label = f"({abbr}) RL"

    elif market == "TOTAL":
        direction = str(pick.get("direction", pick.get("side", "OVER"))).upper()
        ha    = _abbr(pick.get("home_team", ht))
        aa    = _abbr(pick.get("away_team", at))
        line  = pick.get("total_line", "")
        label = f"({ha}/{aa}) {direction} {line}"

    else:
        abbr  = _abbr(pick.get("team", ht))
        label = f"({abbr}) {market}"

    return f"BET {label} ({odds}) [{tier}]"


def write_picks_csv(games: list, date_str: str) -> Path | None:
    """
    Flatten picks_today.json games list → one CSV row per pick.
    Returns the path written, or None if nothing to write.
    """
    MLB_PICKS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for game in games:
        ht    = game.get("home_team", "")
        at    = game.get("away_team", "")
        gtime = _game_time_et(game.get("commence", ""))

        for pick in game.get("picks", []):
            market = str(pick.get("type", pick.get("market", "ML"))).upper()
            signal = _bet_signal(pick, ht, at)

            # model_prob, book_prob, and edge are already percentages in
            # picks_today.json (run_today.py _picks_to_game_format multiplies
            # by 100 before writing). Do NOT multiply by 100 here again.
            try:
                model_pct   = float(pick.get("model_prob", 0))
                book_pct    = float(pick.get("book_prob",  0))
                pred_margin = round(model_pct - book_pct, 1)
            except (TypeError, ValueError):
                model_pct   = 0.0
                book_pct    = 0.0
                pred_margin = 0.0

            edge_pct = float(pick.get("edge", 0))  # already a percentage

            # VEGAS_SPREAD: the line being bet
            if market in ("SPREAD", "RL"):
                vegas = str(pick.get("spread", pick.get("spread_line", "—")))
            elif market == "TOTAL":
                vegas = str(pick.get("total_line", "—"))
            else:
                vegas = _fmt_odds(pick.get("odds"))

            rows.append({
                "DATE":        date_str,
                "MATCHUP":     f"{at} @ {ht}",
                "BET_SIGNAL":  signal,
                "VEGAS_SPREAD": vegas,
                "SPREAD_EDGE": f"{edge_pct:+.1f}%",
                "MODEL_PRED":  f"{model_pct:.1f}% (model) vs {book_pct:.1f}% (book)",
                "PRED_MARGIN": pred_margin,
                "BOOK":        str(pick.get("book", "")).upper(),
                "ODDS":        _fmt_odds(pick.get("odds")),
                "UNITS":       int(pick.get("units", 1)),
                "MARKET":      market,
                "GAME_TIME":   gtime,
            })

    if not rows:
        print("  No picks to write.")
        return None

    date_num  = date_str.replace("-", "")
    out_path  = MLB_PICKS_DIR / f"mlb_picks_{date_num}.csv"
    df        = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Picks CSV → {out_path}  ({len(rows)} rows)")

    # Mirror to local Results dir
    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mirror = LOCAL_RESULTS_DIR / f"mlb_picks_{date_num}.csv"
    df.to_csv(mirror, index=False)
    print(f"  Mirror    → {mirror}")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GRADE YESTERDAY'S BETS via ESPN
# ═══════════════════════════════════════════════════════════════════════════════

# ESPN uses non-standard abbreviations for 6 teams. Normalise them to the
# same abbreviations the model uses so frozenset lookup keys always match.
# Without this, every game involving KC/SD/SF/TB/WSH/CWS silently fails
# to grade because frozenset(['MIN','KC']) != frozenset(['MIN','KCR']).
_ESPN_ABBR_FIX = {
    "KC":  "KCR",   # Kansas City Royals
    "SD":  "SDP",   # San Diego Padres
    "SF":  "SFG",   # San Francisco Giants
    "TB":  "TBR",   # Tampa Bay Rays
    "WSH": "WSN",   # Washington Nationals
    "CWS": "CHW",   # Chicago White Sox
}

def _norm_espn_abbr(abbr: str) -> str:
    return _ESPN_ABBR_FIX.get(abbr.upper(), abbr.upper())


def _fetch_espn_scores(date_str: str) -> dict:
    """
    Fetch ESPN MLB scores for a given date.
    Returns dict keyed by frozenset({away_abbr, home_abbr}) →
        {home_score, away_score, home_abbr, away_abbr}
    """
    date_num = date_str.replace("-", "")
    try:
        r = requests.get(
            f"{ESPN_BASE}/scoreboard",
            params={"dates": date_num, "limit": 30},
            headers=HEADERS, timeout=12,
        )
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as e:
        print(f"  [warn] ESPN fetch failed for {date_str}: {e}")
        return {}

    results = {}
    for ev in events:
        try:
            comp  = ev["competitions"][0]
            teams = comp["competitors"]
            home  = next(t for t in teams if t["homeAway"] == "home")
            away  = next(t for t in teams if t["homeAway"] == "away")
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            ha = _norm_espn_abbr(home["team"]["abbreviation"])
            aa = _norm_espn_abbr(away["team"]["abbreviation"])
            hs = int(float(home.get("score", 0) or 0))
            as_ = int(float(away.get("score", 0) or 0))
            key = frozenset([ha, aa])
            results[key] = {"home_abbr": ha, "away_abbr": aa,
                            "home_score": hs, "away_score": as_}
        except Exception:
            continue

    print(f"  ESPN scores fetched: {len(results)} completed games on {date_str}")
    return results


def _grade_pick(row: pd.Series, espn_results: dict) -> dict | None:
    """
    Grade one row from the picks CSV against ESPN results.
    Returns grading dict or None if the game can't be matched.
    """
    matchup = str(row.get("MATCHUP", ""))       # "KC @ ATL"
    signal  = str(row.get("BET_SIGNAL", ""))
    market  = str(row.get("MARKET", "ML")).upper()

    # Parse teams from MATCHUP string "Away @ Home"
    parts = [p.strip() for p in matchup.split("@")]
    if len(parts) != 2:
        return None
    away_full, home_full = parts

    # Convert full names to abbreviations for ESPN lookup
    # MATCHUP may already contain abbreviations (3 chars) or full names
    def _to_abbr(name):
        if name in FULL_TO_ABBR:
            return FULL_TO_ABBR[name]
        if len(name) <= 3:
            return name.upper()
        # Try suffix matching
        for full, abbr in FULL_TO_ABBR.items():
            if full.endswith(name) or name.endswith(full.split()[-1]):
                return abbr
        return name[:3].upper()

    ha = _to_abbr(home_full)
    aa = _to_abbr(away_full)
    key = frozenset([ha, aa])

    game = espn_results.get(key)
    if game is None:
        return None

    hs, as_ = game["home_score"], game["away_score"]
    result  = None

    if market == "ML":
        # Find the team being bet on from the signal: "(ATL) ML"
        m = re.search(r'\(([A-Z/]+)\)', signal)
        if not m:
            return None
        bet_abbr = m.group(1)
        if "/" in bet_abbr:
            return None   # totals signal slipped through — skip
        bet_home = (bet_abbr == ha)
        won = (hs > as_) if bet_home else (as_ > hs)
        result = "WIN" if won else ("PUSH" if hs == as_ else "LOSS")

    elif market in ("SPREAD", "RL"):
        # Run line is always ±1.5; home covers if wins by 2+
        m = re.search(r'\(([A-Z]+)\)\s+RL\s+([-+]?\d+\.?\d*)', signal)
        if not m:
            return None
        bet_abbr = m.group(1)
        line     = float(m.group(2))
        bet_home = (bet_abbr == ha)
        margin   = hs - as_
        if bet_home:
            # Home team at 'line': covers if (home_margin + line) > 0
            # e.g. home -1.5: needs margin > 1.5 (win by 2+)
            # e.g. home +1.5: covers if margin > -1.5 (lose by 1 or win)
            covered = (margin + line) > 0
            push    = (margin + line) == 0
        else:
            # Away team at 'line': away margin = -margin (home perspective)
            # Away -1.5: away must win by 2+  →  -margin > 1.5  →  (-margin + line) > 0
            #            where line = -1.5  →  -margin - 1.5 > 0  →  margin < -1.5  ✓
            # Away +1.5: away covers losing by 1 →  -margin > -1.5  →  (-margin + line) > 0
            #            where line = +1.5  →  -margin + 1.5 > 0  →  margin < 1.5  ✓
            covered = (-margin + line) > 0
            push    = (-margin + line) == 0
        result = "WIN" if covered else ("PUSH" if push else "LOSS")

    elif market == "TOTAL":
        m = re.search(r'\(([A-Z/]+)\)\s+(OVER|UNDER)\s+([\d.]+)', signal, re.IGNORECASE)
        if not m:
            return None
        direction = m.group(2).upper()
        line      = float(m.group(3))
        total     = hs + as_
        if direction == "OVER":
            result = "WIN" if total > line else ("PUSH" if total == line else "LOSS")
        else:
            result = "WIN" if total < line else ("PUSH" if total == line else "LOSS")

    if result is None:
        return None

    # PNL calculation
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
    """Load yesterday's picks CSV, fetch ESPN scores, grade every bet."""
    date_num  = yesterday.replace("-", "")
    picks_csv = MLB_PICKS_DIR / f"mlb_picks_{date_num}.csv"

    # Fallback: check local mirror
    if not picks_csv.exists():
        picks_csv = LOCAL_RESULTS_DIR / f"mlb_picks_{date_num}.csv"
    if not picks_csv.exists():
        print(f"  No picks CSV found for {yesterday} — skipping grading")
        return pd.DataFrame()

    picks_df     = pd.read_csv(picks_csv)
    espn_results = _fetch_espn_scores(yesterday)
    if not espn_results:
        return pd.DataFrame()

    graded_rows = []
    for _, row in picks_df.iterrows():
        grading = _grade_pick(row, espn_results)
        if grading is None:
            print(f"  [skip] No match: {row.get('MATCHUP','?')}  "
                  f"{str(row.get('BET_SIGNAL',''))[:50]}")
            continue

        icon = "✓" if grading["RESULT"] == "WIN" else \
               ("~" if grading["RESULT"] == "PUSH" else "✗")
        print(f"  {icon} {grading['RESULT']:4s}  {grading['PNL']:+.3f}u  "
              f"{row.get('MATCHUP','?')}  {str(row.get('BET_SIGNAL',''))[:45]}")

        graded_rows.append({
            "DATE":       yesterday,
            "MATCHUP":    row.get("MATCHUP", ""),
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

    if not graded_rows:
        print("  Nothing graded.")
    return pd.DataFrame(graded_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. APPEND TO bet_results.csv
# ═══════════════════════════════════════════════════════════════════════════════

def _sync_results_files():
    """
    Ensure MLB_RESULTS_CSV (NBA_OUTPUT path — what the page reads) and
    LOCAL_RESULTS_CSV (MLB Model Files path — the authoritative copy) are
    in sync.  The local copy is the source of truth: if it's newer or has
    more rows, copy it to the NBA_OUTPUT path.

    Called at the top of main() so the page always reads current data even
    if a previous run failed to write one of the two paths.
    """
    import shutil

    if not LOCAL_RESULTS_CSV.exists():
        return   # nothing to sync from

    MLB_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not MLB_RESULTS_CSV.exists():
        shutil.copy2(LOCAL_RESULTS_CSV, MLB_RESULTS_CSV)
        print(f"  [sync] Copied results → {MLB_RESULTS_CSV}")
        return

    # Both exist — use whichever has more rows (more complete)
    try:
        local_df  = pd.read_csv(LOCAL_RESULTS_CSV)
        remote_df = pd.read_csv(MLB_RESULTS_CSV)
        if len(local_df) >= len(remote_df):
            shutil.copy2(LOCAL_RESULTS_CSV, MLB_RESULTS_CSV)
            print(f"  [sync] Results synced ({len(local_df)} rows → NBA_OUTPUT path)")
        else:
            # Remote has more rows — sync the other direction
            shutil.copy2(MLB_RESULTS_CSV, LOCAL_RESULTS_CSV)
            print(f"  [sync] Results synced ({len(remote_df)} rows ← NBA_OUTPUT path)")
    except Exception as e:
        print(f"  [warn] Results sync failed: {e}")


def append_results(graded_df: pd.DataFrame, yesterday: str):
    """Merge newly-graded rows into both bet_results.csv locations."""
    if graded_df.empty:
        return

    dedup_keys = ["DATE", "MATCHUP", "BET"]

    for results_path in [MLB_RESULTS_CSV, LOCAL_RESULTS_CSV]:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        if results_path.exists():
            existing = pd.read_csv(results_path)
            # Remove any ungraded rows from yesterday (allow re-grading)
            is_yesterday = existing.get("DATE", pd.Series(dtype=str)) == yesterday
            is_ungraded  = ~existing.get("RESULT", pd.Series(dtype=str)).isin(
                ["WIN", "LOSS", "PUSH"]
            )
            existing = existing[~(is_yesterday & is_ungraded)]
            # Deduplicate
            ex_keys  = existing[dedup_keys].apply(tuple, axis=1)
            new_keys = graded_df[dedup_keys].apply(tuple, axis=1)
            new_only = graded_df[~new_keys.isin(ex_keys)]
            combined = pd.concat([existing, new_only], ignore_index=True)
        else:
            combined = graded_df.copy()

        combined.to_csv(results_path, index=False)
        wins  = (combined["RESULT"] == "WIN").sum()
        losses= (combined["RESULT"] == "LOSS").sum()
        pnl   = combined["PNL"].sum()
        print(f"  Results → {results_path}")
        print(f"    {wins}W-{losses}L  |  Season P&L: {pnl:+.2f}u  "
              f"({len(combined)} graded bets)")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REBUILD PAGE + GIT PUSH
# ═══════════════════════════════════════════════════════════════════════════════

def trigger_page_rebuild() -> bool:
    if not GPP_PATH.exists():
        print(f"  [warn] generate_picks_page.py not found at {GPP_PATH}")
        print("  Page rebuild skipped.")
        return False
    try:
        spec = importlib.util.spec_from_file_location("generate_picks_page", GPP_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        print("  Page rebuilt + pushed → ktobttv.github.io (live in ~60s)")
        return True
    except Exception as e:
        print(f"  [warn] Page rebuild failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(grade_only: bool = False, no_push: bool = False):
    print("\n── MLB Page Generator ──────────────────────────────────")

    today     = date.today().isoformat()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    # ── Step 0: Sync results files so the page always reads current data ─────
    # The page reads from NBA_OUTPUT/MLB/Results/bet_results.csv but grading
    # also writes to MLB Model Files/Results/bet_results.csv (the local copy).
    # If these drift out of sync (e.g. a previous run failed mid-write), the
    # page will show stale stats. This sync step runs before any grading so
    # the authoritative local copy is always propagated to the page's path.
    _sync_results_files()

    # ── Step 1: Write today's picks CSV + X thread writeup ───────────────────
    if not grade_only:
        if not PICKS_JSON.exists():
            print(f"  [warn] picks_today.json not found at {PICKS_JSON}")
            print("  Run: python run_today.py")
        else:
            with open(PICKS_JSON) as f:
                data = json.load(f)
            games    = data.get("games", [])
            date_str = data.get("date", today)
            n_picks  = data.get("total_picks", 0)
            gen_at   = data.get("generated_at", "")[:16]
            print(f"  {n_picks} picks for {date_str}  (generated {gen_at})")
            write_picks_csv(games, date_str)

            # Generate X thread writeup (skip if run_today already did it)
            try:
                from generate_mlb_writeups import generate as _gen_writeup
                from pathlib import Path as _Path
                from config import BASE_DIR as _BASE_DIR
                _wd = _Path(_BASE_DIR) / "Writeups" / f"{date_str}.txt"
                if not _wd.exists():
                    print("\n── Generating X thread writeup ─────────────────────────")
                    _gen_writeup(data)
                else:
                    print(f"  Writeup already exists: {_wd.name} — skipping")
            except Exception as e:
                print(f"  [warn] Writeup generation failed: {e}")

    # ── Step 2: Grade yesterday ───────────────────────────────────────────────
    print(f"\n  Grading bets for {yesterday}...")
    graded_df = grade_yesterdays_bets(yesterday)

    # ── Step 3: Append results ────────────────────────────────────────────────
    if not graded_df.empty:
        print(f"\n  Writing {len(graded_df)} graded bets...")
        append_results(graded_df, yesterday)
    else:
        print("  Nothing new to append.")

    # ── Step 4: Rebuild page ──────────────────────────────────────────────────
    if not no_push:
        print("\n  Rebuilding picks page...")
        trigger_page_rebuild()

    print("── MLB Page Generator done ──────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB picks page generator")
    parser.add_argument("--grade-only", action="store_true",
                        help="Skip picks CSV step — only grade + push")
    parser.add_argument("--no-push",   action="store_true",
                        help="Write files but skip the git push")
    parser.add_argument("--sync-only", action="store_true",
                        help="Only sync results files then push — skip grading and picks")
    args = parser.parse_args()
    if args.sync_only:
        _sync_results_files()
        trigger_page_rebuild()
    else:
        main(grade_only=args.grade_only, no_push=args.no_push)
