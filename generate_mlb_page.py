"""
generate_mlb_page.py — MLB → Public Picks Page Bridge
=======================================================
Reads the MLB model's picks JSON + bet log CSV and converts them
into the format generate_picks_page.py expects, then triggers
the page rebuild and GitHub push.

Runs automatically after run_model.py and after nightly grading.

Usage:
    python generate_mlb_page.py

What it writes:
    MLB/Daily Picks CSV/mlb_picks_YYYYMMDD.csv   <- today's picks in NBA format
    MLB/Results/bet_results.csv                   <- all graded bets in NBA format
Then calls generate_picks_page.py to rebuild index.html and push to GitHub.
"""

import os, sys, json, importlib.util
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
from config import LIVE_DIR, LOGS_DIR, BET_LOG_CSV, BASE_DIR

NBA_OUTPUT_DIR   = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\ML and Spread"
MLB_PICKS_DIR    = os.path.join(NBA_OUTPUT_DIR, "MLB", "Daily Picks CSV")
MLB_RESULTS_DIR  = os.path.join(NBA_OUTPUT_DIR, "MLB", "Results")
PICKS_JSON       = os.path.join(LIVE_DIR, "picks_today.json")
GPP_PATH         = os.path.join(NBA_OUTPUT_DIR, "generate_picks_page.py")

Path(MLB_PICKS_DIR).mkdir(parents=True, exist_ok=True)
Path(MLB_RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# ── Format helpers ────────────────────────────────────────────────────────────

def odds_to_american_str(odds):
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return "—"
    odds = float(odds)
    return f"+{int(odds)}" if odds > 0 else str(int(odds))

def units_to_tier(units):
    if units >= 3:   return "HIGH CONFIDENCE"
    elif units >= 2: return "MEDIUM CONFIDENCE"
    return "LOW CONFIDENCE"

def market_label(pick):
    mkt  = pick.get("market","")
    side = pick.get("side","")
    team = pick.get("team","")
    if mkt == "moneyline":
        return f"{team} ML"
    elif mkt == "run_line":
        spread = pick.get("spread","")
        return f"{team} {spread} RL"
    elif mkt == "totals":
        return f"{team}"
    return team

def build_bet_signal(pick):
    """Format pick as BET_SIGNAL string matching NBA model format."""
    label = market_label(pick)
    tier  = units_to_tier(pick.get("units", 1))
    odds  = odds_to_american_str(pick.get("odds"))
    return f"BET {label} ({odds}) [{tier}]"

def matchup_str(pick):
    return f"{pick.get('away_team','')} @ {pick.get('home_team','')}"

def game_time_str(pick):
    try:
        return pd.to_datetime(pick.get("commence",""), utc=True)\
                 .tz_convert("US/Eastern")\
                 .strftime("%-I:%M %p ET")
    except Exception:
        return ""


# ── Convert picks JSON → CSV ──────────────────────────────────────────────────

def write_picks_csv(picks, date_str):
    """Write picks in the format generate_picks_page.py expects."""
    if not picks:
        print("  No picks to write")
        return

    rows = []
    for p in picks:
        rows.append({
            "DATE":         date_str,
            "MATCHUP":      matchup_str(p),
            "BET_SIGNAL":   build_bet_signal(p),
            "VEGAS_SPREAD": p.get("spread", p.get("line", "—")),
            "SPREAD_EDGE":  f"{p.get('edge', 0):+.1f}%",
            "MODEL_PRED":   f"{p.get('model_prob', 0):.1f}% (model) vs {p.get('book_prob', 0):.1f}% (book)",
            "BOOK":         p.get("book","").upper(),
            "ODDS":         odds_to_american_str(p.get("odds")),
            "UNITS":        p.get("units", 1),
            "MARKET":       p.get("market",""),
            "GAME_TIME":    game_time_str(p),
            "HOME_INJURIES":"",
            "AWAY_INJURIES":"",
        })

    df = pd.DataFrame(rows)
    today_num = date_str.replace("-","")
    out_path  = os.path.join(MLB_PICKS_DIR, f"mlb_picks_{today_num}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Wrote {len(df)} picks → {out_path}")
    return out_path


# ── Convert bet log CSV → results CSV ────────────────────────────────────────

def write_results_csv():
    """Convert MLB bet_log.csv into the format generate_picks_page.py expects."""
    if not Path(BET_LOG_CSV).exists():
        print("  No bet log found — skipping results CSV")
        return

    bet_log = pd.read_csv(BET_LOG_CSV)
    if bet_log.empty:
        print("  Bet log is empty")
        return

    rows = []
    for _, b in bet_log.iterrows():
        result = str(b.get("result","")).strip().upper()
        if result not in ("W","L","PUSH"):
            result_mapped = "pending"
        else:
            result_mapped = "WIN" if result == "W" else ("LOSS" if result == "L" else "PUSH")

        # Calculate PNL in units (already stored in bet_log)
        pnl = float(b.get("pnl_units", 0) or 0)

        rows.append({
            "DATE":     str(b.get("date",""))[:10],
            "MATCHUP":  f"{b.get('away_team','')} @ {b.get('home_team','')}",
            "BET":      build_bet_signal_from_log(b),
            "RESULT":   result_mapped,
            "PNL":      round(pnl, 3),
            "UNITS":    int(b.get("units", 1)),
            "ODDS":     odds_to_american_str(b.get("odds")),
            "MARKET":   str(b.get("market","")),
            "BOOK":     str(b.get("book","")).upper(),
            "EDGE":     float(b.get("edge", 0) or 0),
        })

    df = pd.DataFrame(rows)
    # Only include graded bets for the results page
    graded = df[df["RESULT"].isin(["WIN","LOSS","PUSH"])].copy()

    out_path = os.path.join(MLB_RESULTS_DIR, "bet_results.csv")
    graded.to_csv(out_path, index=False)
    print(f"  Wrote {len(graded)} graded bets → {out_path}")
    return out_path


def build_bet_signal_from_log(row):
    """Build BET_SIGNAL from a bet log row for team tracker parsing."""
    team   = str(row.get("team","")).strip()
    mkt    = str(row.get("market","")).strip()
    odds   = odds_to_american_str(row.get("odds"))
    tier   = units_to_tier(int(row.get("units",1)))
    ht     = str(row.get("home_team",""))
    at     = str(row.get("away_team",""))

    # Format: "BET (TEAM) ML (+150) [HIGH CONFIDENCE]"
    # Team tracker in generate_picks_page.py uses regex: \(([A-Z]+)\s
    if mkt == "moneyline":
        label = f"({team}) ML"
    elif mkt == "run_line":
        label = f"({team}) RL"
    elif mkt == "totals":
        direction = str(row.get("side","")).upper()
        line = str(row.get("line",""))
        label = f"({ht}/{at}) {direction} {line}"
    else:
        label = f"({team})"

    return f"BET {label} ({odds}) [{tier}]"


# ── Trigger page rebuild ──────────────────────────────────────────────────────

def trigger_page_rebuild():
    """Call generate_picks_page.py main() to rebuild index.html and push."""
    if not os.path.exists(GPP_PATH):
        print(f"  [warn] generate_picks_page.py not found at {GPP_PATH}")
        print("  Page rebuild skipped — place generate_picks_page.py in the NBA Model Files folder")
        return False

    try:
        spec = importlib.util.spec_from_file_location("generate_picks_page", GPP_PATH)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        print("  ✓ Page rebuilt and pushed to GitHub")
        return True
    except Exception as e:
        print(f"  [warn] Page rebuild failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── MLB Page Generator ──")

    # Load today's picks
    if not Path(PICKS_JSON).exists():
        print("  No picks_today.json found — run python run_model.py first")
        return

    with open(PICKS_JSON) as f:
        data = json.load(f)

    picks    = data.get("picks", [])
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    print(f"  {len(picks)} picks for {date_str}")

    # Write picks CSV
    write_picks_csv(picks, date_str)

    # Write results CSV
    write_results_csv()

    # Trigger page rebuild
    trigger_page_rebuild()

    print("── MLB Page Generator done ──\n")


if __name__ == "__main__":
    main()
