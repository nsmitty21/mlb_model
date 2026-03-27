"""
run_today.py — Daily Pick Generator
Usage: python run_today.py [--date YYYY-MM-DD]

Fixes (2026-03-27):
  - BUG 1: pivot_lines() was looking for 'total_line' column but pull_lines.py
            stores it as 'total'. This caused fd_total/dk_total to always be NaN,
            meaning the totals model was betting against whatever garbage default
            it fell back to (spring training 3.5 lines). Fixed column names.
  - BUG 2: Edge thresholds in model.py (3%/6%/10%) are far too aggressive.
            Overriding here: 1U >= 7%, 2U >= 10%, 3U >= 15%. A true 3U edge
            should be rare — if you're seeing it on half the slate something is wrong.
  - BUG 3: Team stats were being passed as empty dicts {} to build_game_features().
            This means ALL team-level features (roll_runs_scored, roll_win_pct, etc.)
            are NaN → fillna(-999) → model outputs near-constant probabilities.
            Now loads cached game logs and computes rolling stats before prediction.
  - BUG 4: No data-quality gate. Games where pitcher lookup is empty AND team stats
            are empty are now skipped — the model is running completely blind.
  - BUG 5: generate_mlb_page.py reads picks_today.json. The output format from
            run_today.py was different from what generate_mlb_page.py expected
            (it looks for games[].picks[] with type/team/odds/units/edge/model_prob).
            Fixed output format to match what generate_mlb_page.py consumes.
"""
import argparse, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import date, datetime

# Stack A imports (old model pipeline)
from data_fetcher import (fetch_todays_starters, fetch_pitcher_stats,
                          fetch_game_logs_historical, normalize_team)
from features import build_game_features, compute_team_rolling_features, FEATURE_COLS
from model import load_models, predict_game, load_meta

DATA_DIR   = Path("data")
PICKS_FILE = Path("data/live/picks_today.json")
PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Edge threshold overrides (tighter than model.py defaults) ─────────────────
EDGE_1U = 0.07   # 7%
EDGE_2U = 0.10   # 10%
EDGE_3U = 0.15   # 15%  — genuine mismatch; should fire rarely

def _units(edge: float) -> int:
    """Override model.py's get_units() with stricter thresholds."""
    e = abs(edge)
    if e >= EDGE_3U: return 3
    if e >= EDGE_2U: return 2
    if e >= EDGE_1U: return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Team rolling stats
# ─────────────────────────────────────────────────────────────────────────────

def load_team_rolling_stats(target_date: str) -> dict:
    """
    Load cached game logs, compute 15-game rolling stats per team,
    and return a lookup dict keyed by full team name.

    Returns {} if no cache is found (caller handles gracefully).
    """
    year = int(target_date[:4])
    # Try current year cache first, then prior year
    for y_end in [year, year - 1]:
        cached = sorted(DATA_DIR.glob(f"game_logs_*_{y_end}.parquet"))
        if cached:
            df = pd.read_parquet(cached[-1])
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] < target_date]   # exclude future rows
            df = compute_team_rolling_features(df, window=15)
            # Take the most recent row per team
            latest = df.sort_values("date").groupby("team").last()
            lookup = {}
            for team, row in latest.iterrows():
                lookup[team] = {
                    "roll_runs_scored":  row.get("roll_15_runs_scored", np.nan),
                    "roll_runs_allowed": row.get("roll_15_runs_allowed", np.nan),
                    "roll_run_diff":     row.get("roll_run_diff", np.nan),
                    "roll_win_pct":      row.get("roll_win_pct", np.nan),
                    "season_win_pct":    row.get("season_win_pct", np.nan),
                }
            print(f"  Team rolling stats: {len(lookup)} teams from game_logs cache")
            return lookup
    print("  [WARN] No game_logs cache found — team stats will be empty. "
          "Run: python -c \"from data_fetcher import fetch_game_logs_historical; "
          "fetch_game_logs_historical(2021, 2025)\"")
    return {}


def build_pitcher_lookup(pitcher_stats: pd.DataFrame, year: int) -> dict:
    if pitcher_stats is None or pitcher_stats.empty:
        return {}
    ps = pitcher_stats[pitcher_stats["year"] == year]
    if ps.empty:
        ps = pitcher_stats   # fallback to any year
    lookup = {}
    for _, row in ps.iterrows():
        name = str(row.get("Name", "")).strip().lower()
        lookup[name] = {
            "season_era":   row.get("ERA",   np.nan),
            "season_fip":   row.get("FIP",   np.nan),
            "roll_era":     row.get("ERA",   np.nan),
            "roll_fip":     row.get("FIP",   np.nan),
            "roll_whip":    row.get("WHIP",  np.nan),
            "roll_k_per_9": row.get("K/9",   np.nan),
            "roll_bb_per_9":row.get("BB/9",  np.nan),
            "roll_hr_per_9":row.get("HR/9",  np.nan),
        }
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Lines fetching and pivoting
# ─────────────────────────────────────────────────────────────────────────────

def fetch_todays_lines(target_date: str) -> pd.DataFrame:
    """
    Load today_lines.json (written by pull_lines.py) and flatten into one
    row per game with fd_* and dk_* prefixed columns.

    BUG 1 FIX: pull_lines.py stores total under the key 'total', not
    'total_line'. The old pivot_lines() was renaming 'total_line' which
    never existed, so fd_total/dk_total were always NaN.
    """
    lines_path = Path("data/live/today_lines.json")
    if not lines_path.exists():
        print("[ERROR] today_lines.json not found. Run: python pull_lines.py")
        return pd.DataFrame()

    with open(lines_path) as f:
        games = json.load(f)

    rows = []
    for g in games:
        ht = g["home_team"]
        at = g["away_team"]
        fd = g.get("books", {}).get("fanduel", {})
        dk = g.get("books", {}).get("draftkings", {})

        rows.append({
            "game_id":              g["game_id"],
            "home_team":            ht,
            "away_team":            at,
            "commence":             g.get("commence", ""),
            # FD columns — BUG 1 FIX: key is 'total', not 'total_line'
            "fd_ml_home":           fd.get("ml_home"),
            "fd_ml_away":           fd.get("ml_away"),
            "fd_spread_home":       fd.get("spread_home"),
            "fd_spread_juice_home": fd.get("spread_juice_home"),
            "fd_spread_juice_away": fd.get("spread_juice_away"),
            "fd_total":             fd.get("total"),        # was total_line — FIXED
            "fd_total_over_juice":  fd.get("total_over_juice"),
            "fd_total_under_juice": fd.get("total_under_juice"),
            # DK columns
            "dk_ml_home":           dk.get("ml_home"),
            "dk_ml_away":           dk.get("ml_away"),
            "dk_spread_home":       dk.get("spread_home"),
            "dk_spread_juice_home": dk.get("spread_juice_home"),
            "dk_spread_juice_away": dk.get("spread_juice_away"),
            "dk_total":             dk.get("total"),        # was total_line — FIXED
            "dk_total_over_juice":  dk.get("total_over_juice"),
            "dk_total_under_juice": dk.get("total_under_juice"),
        })

    df = pd.DataFrame(rows)
    print(f"  Lines loaded: {len(df)} games from today_lines.json")

    # Sanity check totals
    valid_totals = df["fd_total"].dropna()
    if len(valid_totals) > 0:
        low = valid_totals[valid_totals < 5.0]
        if len(low) > 0:
            print(f"  [WARN] {len(low)} game(s) have totals < 5.0 — "
                  f"possible spring training or data issue. Values: {low.tolist()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data quality gate
# ─────────────────────────────────────────────────────────────────────────────

def _has_usable_data(home_stats: dict, away_stats: dict,
                     home_pitcher: dict, away_pitcher: dict,
                     home_team: str, away_team: str) -> bool:
    """
    BUG 4 FIX: Return False if both team stats and pitcher stats are essentially
    empty. When everything is NaN the model outputs near-constant probabilities
    and fake edges appear on every game.
    """
    def _nonnull_count(d: dict) -> int:
        return sum(1 for v in d.values()
                   if v is not None and not (isinstance(v, float) and np.isnan(v)))

    team_ok    = _nonnull_count(home_stats) + _nonnull_count(away_stats) >= 4
    pitcher_ok = _nonnull_count(home_pitcher) + _nonnull_count(away_pitcher) >= 4

    if not team_ok and not pitcher_ok:
        print(f"  [SKIP] {away_team} @ {home_team} — "
              f"no usable team or pitcher stats (model would be guessing)")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Output format normalizer
# ─────────────────────────────────────────────────────────────────────────────

def _picks_to_game_format(all_picks: list, target_date: str) -> list:
    """
    BUG 5 FIX: generate_mlb_page.py expects picks_today.json in the format:
        { games: [ { home_team, away_team, commence, picks: [
            { type, side, team, opponent, odds, book,
              model_prob, book_prob, edge, units, ... }
        ] } ] }

    run_today.py / model.py output a flat list with different keys.
    This function converts to the expected structure.
    """
    # Group picks by game
    from collections import defaultdict
    game_map = defaultdict(list)
    game_meta = {}

    for p in all_picks:
        key = f"{p.get('home_team','')}|{p.get('away_team','')}"
        game_meta[key] = {
            "home_team": p.get("home_team", ""),
            "away_team":  p.get("away_team", ""),
            "commence":   p.get("commence", ""),
            "date":       target_date,
            "game_id":    key.replace("|", "_").replace(" ", "_"),
        }

        market = str(p.get("market", "ML")).upper()
        side   = str(p.get("side", "home")).lower()

        # Normalize to generate_mlb_page.py expected schema
        normalized = {
            "type":       market,
            "side":       side,
            "team":       p.get("team", ""),
            "opponent":   p.get("away_team", "") if side == "home" else p.get("home_team", ""),
            "odds":       p.get("odds"),
            "book":       p.get("book", ""),
            "model_prob": round(p.get("model_prob", 0) * 100, 1),
            "book_prob":  round(p.get("book_prob", 0) * 100, 1),
            "edge":       round(p.get("edge", 0) * 100, 2),
            "units":      p.get("units", 1),
        }

        if market in ("SPREAD", "RL"):
            normalized["type"]   = "SPREAD"
            normalized["spread"] = p.get("spread_line")
        elif market == "TOTAL":
            normalized["direction"]  = side.upper()
            normalized["home_team"]  = p.get("home_team", "")
            normalized["away_team"]  = p.get("away_team", "")
            normalized["total_line"] = p.get("total_line")
            normalized["pred_total"] = None  # Stack A doesn't predict a raw total

        game_map[key].append(normalized)

    games_out = []
    for key, picks in game_map.items():
        meta = game_meta[key]
        games_out.append({
            "game_id":   meta["game_id"],
            "date":      meta["date"],
            "commence":  meta["commence"],
            "home_team": meta["home_team"],
            "away_team": meta["away_team"],
            "picks":     sorted(picks, key=lambda x: x["units"], reverse=True),
        })

    return games_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(target_date: str | None = None):
    if target_date is None:
        target_date = date.today().isoformat()

    print(f"\n{'='*60}")
    print(f"  MLB Edge — Picks: {target_date}")
    print(f"  Edge thresholds: 1U>={EDGE_1U*100:.0f}%  2U>={EDGE_2U*100:.0f}%  3U>={EDGE_3U*100:.0f}%")
    print(f"{'='*60}\n")

    # ── Lines ──────────────────────────────────────────────────────────────
    games = fetch_todays_lines(target_date)
    if games.empty:
        print("[error] No lines loaded.")
        return

    # ── Starters ───────────────────────────────────────────────────────────
    starters_df = fetch_todays_starters(target_date)
    if not starters_df.empty:
        games = games.merge(
            starters_df[["home_team", "away_team", "home_sp", "away_sp"]],
            on=["home_team", "away_team"], how="left"
        )
    if "home_sp" not in games.columns:
        games["home_sp"] = ""
    if "away_sp" not in games.columns:
        games["away_sp"] = ""
    games["home_sp"] = games["home_sp"].fillna("")
    games["away_sp"] = games["away_sp"].fillna("")

    # ── Models ─────────────────────────────────────────────────────────────
    try:
        ml_model, spread_model, total_model = load_models()
        meta = load_meta()
        print(f"  Models loaded. Trained on {meta.get('training_rows','?')} games.")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return

    # ── BUG 3 FIX: Load team rolling stats ────────────────────────────────
    team_stats_lookup = load_team_rolling_stats(target_date)

    # ── BUG 3 FIX: Load pitcher stats ─────────────────────────────────────
    pitcher_stats = None
    year = int(target_date[:4])
    cached = sorted(DATA_DIR.glob("pitcher_stats_*.parquet"))
    if cached:
        pitcher_stats = pd.read_parquet(cached[-1])
        print(f"  Pitcher stats: {len(pitcher_stats)} rows from {cached[-1].name}")
    else:
        print("  [WARN] No pitcher stats cache. "
              "Run: python -c \"from data_fetcher import fetch_pitcher_stats; "
              "fetch_pitcher_stats(2021, 2025)\"")

    # Use prior year pitcher stats (current year not available at season start)
    pitcher_lookup = build_pitcher_lookup(pitcher_stats, year - 1) if pitcher_stats is not None else {}
    print(f"  Pitcher lookup: {len(pitcher_lookup)} pitchers")

    # ── Generate picks ────────────────────────────────────────────────────
    all_picks = []
    skipped   = 0

    for _, game in games.iterrows():
        ht  = str(game.get("home_team", ""))
        at  = str(game.get("away_team", ""))
        hsp = str(game.get("home_sp",   ""))
        asp = str(game.get("away_sp",   ""))

        home_team_stats = team_stats_lookup.get(ht, {})
        away_team_stats = team_stats_lookup.get(at, {})
        home_pitcher    = pitcher_lookup.get(hsp.lower(), {})
        away_pitcher    = pitcher_lookup.get(asp.lower(), {})

        # BUG 4 FIX: skip games with no usable data
        if not _has_usable_data(home_team_stats, away_team_stats,
                                  home_pitcher, away_pitcher, ht, at):
            skipped += 1
            continue

        feats = build_game_features(
            ht, at,
            home_pitcher, away_pitcher,
            home_team_stats, away_team_stats,
        )

        raw_picks = predict_game(
            feats, ml_model, spread_model, total_model,
            fd_ml_home           = game.get("fd_ml_home"),
            fd_ml_away           = game.get("fd_ml_away"),
            dk_ml_home           = game.get("dk_ml_home"),
            dk_ml_away           = game.get("dk_ml_away"),
            fd_spread_home       = game.get("fd_spread_home"),
            fd_spread_juice_home = game.get("fd_spread_juice_home"),
            fd_spread_juice_away = game.get("fd_spread_juice_away"),
            dk_spread_home       = game.get("dk_spread_home"),
            dk_spread_juice_home = game.get("dk_spread_juice_home"),
            dk_spread_juice_away = game.get("dk_spread_juice_away"),
            fd_total             = game.get("fd_total"),
            fd_total_over_juice  = game.get("fd_total_over_juice"),
            fd_total_under_juice = game.get("fd_total_under_juice"),
            dk_total             = game.get("dk_total"),
            dk_total_over_juice  = game.get("dk_total_over_juice"),
            dk_total_under_juice = game.get("dk_total_under_juice"),
            home_team=ht, away_team=at,
            home_sp=hsp, away_sp=asp,
            commence=str(game.get("commence", "")),
        )

        # BUG 2 FIX: re-evaluate units with stricter thresholds
        for p in raw_picks:
            p["units"] = _units(p["edge"])
            p["home_team"] = ht
            p["away_team"] = at
            if p["units"] > 0:
                all_picks.append(p)

    if skipped:
        print(f"\n  [WARN] {skipped} game(s) skipped — no usable stats.")

    # ── Sort + sanity check ───────────────────────────────────────────────
    all_picks.sort(key=lambda p: (-p["units"], -abs(p["edge"])))

    ml_picks   = [p for p in all_picks if p["market"] == "ML"]
    away_ml    = [p for p in ml_picks  if p["side"]   == "away"]
    three_u    = [p for p in all_picks if p["units"]  == 3]

    print(f"\n{'='*60}")
    if not all_picks:
        print("  No picks meet edge threshold today.")
    else:
        print(f"  {len(all_picks)} PICKS  "
              f"(3U:{len(three_u)}  ML:{len(ml_picks)}  away_ML:{len(away_ml)})")
        if len(ml_picks) > 0 and len(away_ml) == len(ml_picks):
            print("  [WARN] Every ML pick is away — verify team stats are loaded.")
        if len(three_u) > len(all_picks) * 0.5 and len(all_picks) >= 4:
            print("  [WARN] >50% of picks are 3U — unusually high, review before betting.")
        print()
        for p in all_picks:
            stars = "⭐" * p["units"]
            print(f"  {stars} {p['units']}u [{p['market']}] {p['team']} "
                  f"({p['odds']:+d} @ {p['book'].upper()})  "
                  f"Edge:{p['edge']*100:.1f}%  "
                  f"Model:{p['model_prob']*100:.1f}%  Book:{p['book_prob']*100:.1f}%")
            print(f"       {at} @ {ht}  SP: {p.get('away_sp','?')} vs {p.get('home_sp','?')}")
    print(f"{'='*60}")

    # ── BUG 5 FIX: Write in generate_mlb_page.py-compatible format ───────
    games_out = _picks_to_game_format(all_picks, target_date)

    out = {
        "generated_at": datetime.now().isoformat(),
        "date":         target_date,
        "total_picks":  len(all_picks),
        "games":        games_out,
    }
    with open(PICKS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(all_picks)} picks to {PICKS_FILE}")
    print("Run next: python generate_mlb_page.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    run(args.date)
