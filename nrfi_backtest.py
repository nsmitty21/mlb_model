"""
nrfi_backtest.py — SP NRFI/YRFI Career Backtest
=================================================
Reads all plate_appearances_raw_*.csv files (2021–present) and builds
a per-pitcher NRFI record table showing exactly how often each starting
pitcher kept the first inning scoreless.

Method
------
1. For each bbref_game_id, identify the starting pitchers for each half-inning:
     - away SP  = pitcher facing the first batter in t1 (top of 1st)
     - home SP  = pitcher facing the first batter in b1 (bottom of 1st)
2. Determine whether the away team scored in t1 and/or the home team scored
   in b1 using the score column at the START of t2 (top of 2nd inning).
   The score at t2 start = cumulative runs through end of full 1st inning.
3. Build per-SP records:
     SP_NRFI   = starts where they personally kept their half-inning scoreless
     GAME_NRFI = starts where the whole game (both halves of 1st) was NRFI

Output
------
  data/historical/odds/nrfi_pitcher_backtest.csv

Columns:
  pitcher, starts, sp_nrfi, sp_yrfi, sp_nrfi_pct,
  game_nrfi, game_yrfi, game_nrfi_pct,
  avg_runs_allowed_1st, years_active, last_seen

Usage:
  python nrfi_backtest.py
  python nrfi_backtest.py --min-starts 10
  python nrfi_backtest.py --output my_folder/nrfi_results.csv
"""

import argparse
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

# ── Paths (mirrors config.py BASE_DIR logic) ─────────────────────────────────
BASE_DIR = Path(r"C:\Users\Kctob\OneDrive\Documents\MLB Model Files")
PA_DIR   = BASE_DIR / "data" / "historical" / "odds"
OUT_PATH = PA_DIR / "nrfi_pitcher_backtest.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pa_data(pa_dir: Path) -> pd.DataFrame:
    files = sorted(pa_dir.glob("plate_appearances_raw_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No plate_appearances_raw_*.csv files found in:\n  {pa_dir}"
        )
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            for col in ("batter", "pitcher"):
                if col in df.columns:
                    df[col] = (df[col]
                               .str.replace("\xa0", " ", regex=False)
                               .str.strip())
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  [warn] Could not read {f.name}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])

    # Normalise inning column — some files store "t1"/"b1", others "1"/"1"
    # Keep as-is; we filter on exact string values below.
    print(f"\n  Total: {len(combined):,} PAs | "
          f"{combined['bbref_game_id'].nunique():,} games | "
          f"{combined['game_date'].min().date()} → {combined['game_date'].max().date()}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Score parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_score(s) -> tuple[int, int]:
    """'away-home' string → (away_runs, home_runs). Returns (0,0) on failure."""
    try:
        a, h = str(s).split("-")
        return int(a), int(h)
    except Exception:
        return 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# Core builder
# ─────────────────────────────────────────────────────────────────────────────

def build_nrfi_records(pa: pd.DataFrame) -> pd.DataFrame:
    """
    For every game in the PA data, identify:
      - away_sp  : pitcher who started t1 (top of 1st)
      - home_sp  : pitcher who started b1 (bottom of 1st)
      - away_r1  : runs scored by away team in top of 1st
      - home_r1  : runs scored by home team in bottom of 1st
      - nrfi     : True if both away_r1 == 0 and home_r1 == 0

    Returns a game-level DataFrame.
    """
    records = []

    # Group by game — process one game at a time
    for gid, gdf in pa.groupby("bbref_game_id"):
        gdf = gdf.sort_index()   # preserve original row order within the game

        game_date = gdf["game_date"].iloc[0]

        # ── Identify starting pitchers ────────────────────────────────────
        # Away SP = first pitcher listed in t1 (top of 1st, inning_half == 'top')
        t1 = gdf[(gdf["inning"] == "t1") | 
                 ((gdf["inning"] == "1") & (gdf["inning_half"] == "top"))]
        b1 = gdf[(gdf["inning"] == "b1") |
                 ((gdf["inning"] == "1") & (gdf["inning_half"] == "bot"))]

        if t1.empty or b1.empty:
            continue   # incomplete game

        # First pitcher to face a batter in each half-inning gets the credit.
        # No minimum PA filter — openers, short starts, and blowouts all
        # count equally. The only thing that matters for NRFI betting is
        # who threw the first pitch of each half-inning and whether they
        # gave up a run. A Skenes 6-batter blowout inning is just as real
        # as a Skenes 9-inning gem.
        away_sp = t1.iloc[0]["pitcher"]
        home_sp = b1.iloc[0]["pitcher"]

        # ── Determine runs scored in each half-inning ─────────────────────
        # Strategy: look at the score at the FIRST PA of t2 (top of 2nd).
        # That score reflects runs through the end of the full 1st inning.
        t2 = gdf[(gdf["inning"] == "t2") |
                 ((gdf["inning"] == "2") & (gdf["inning_half"] == "top"))]

        if not t2.empty:
            # Score at start of t2 = total runs through 1 full inning
            away_r1, home_r1 = _parse_score(t2.iloc[0]["score"])
        else:
            # Game ended after 1 inning (walk-off, weather, etc.) — skip
            continue

        # ── Away SP half-inning: did they allow runs in t1? ───────────────
        # The score before the LAST PA of t1 gives home runs only if the
        # bottom hasn't been played, so we use the t2-open score for home.
        # For away: runs allowed = away_r1 (what the home team gave up in b1
        # is irrelevant; what matters for the away SP is what the away team
        # scored in t1 = away_r1).
        #
        # Wait — the "away SP" faces the AWAY team's batters in t1.
        # Runs in t1 = runs scored BY the away team = away_r1.
        # So: away_sp allowed away_r1? No — the away team's batters hit
        # AGAINST the HOME SP in t1. Let's be precise:
        #
        #   t1 = top of 1st = AWAY batters vs HOME pitcher (home_sp)
        #   b1 = bottom of 1st = HOME batters vs AWAY pitcher (away_sp)
        #
        #   home_sp faced: away team's bats in t1 → allowed away_r1 runs
        #   away_sp faced: home team's bats in b1 → allowed home_r1 runs
        #
        # NRFI for home_sp = (away_r1 == 0)
        # NRFI for away_sp = (home_r1 == 0)
        home_sp_nrfi = (away_r1 == 0)   # home pitcher kept away team scoreless
        away_sp_nrfi = (home_r1 == 0)   # away pitcher kept home team scoreless
        game_nrfi    = (away_r1 == 0 and home_r1 == 0)

        records.append({
            "bbref_game_id": gid,
            "game_date":     game_date,
            "home_sp":       home_sp,
            "away_sp":       away_sp,
            "away_r1":       away_r1,   # runs in top-1 (against home_sp)
            "home_r1":       home_r1,   # runs in bot-1 (against away_sp)
            "home_sp_nrfi":  home_sp_nrfi,
            "away_sp_nrfi":  away_sp_nrfi,
            "game_nrfi":     game_nrfi,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate to per-pitcher stats
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_pitcher_records(games: pd.DataFrame) -> pd.DataFrame:
    """
    Stack home_sp and away_sp into one long table, then aggregate per pitcher.

    Each pitcher gets credit for:
      sp_nrfi     = starts where THEY kept their half-inning scoreless
      game_nrfi   = starts where the full game 1st inning was NRFI
                    (only meaningful when combined with the other SP)
      runs_allowed = runs scored against them specifically in the 1st inning
    """
    # Home pitcher rows (faced the away team in t1)
    home_rows = games[["game_date", "home_sp", "home_sp_nrfi",
                        "game_nrfi", "away_r1"]].copy()
    home_rows.columns = ["game_date", "pitcher", "sp_nrfi",
                         "game_nrfi", "runs_allowed_1st"]

    # Away pitcher rows (faced the home team in b1)
    away_rows = games[["game_date", "away_sp", "away_sp_nrfi",
                        "game_nrfi", "home_r1"]].copy()
    away_rows.columns = ["game_date", "pitcher", "sp_nrfi",
                         "game_nrfi", "runs_allowed_1st"]

    long = pd.concat([home_rows, away_rows], ignore_index=True)

    agg = long.groupby("pitcher").agg(
        starts              = ("sp_nrfi",          "count"),
        sp_nrfi             = ("sp_nrfi",          "sum"),
        game_nrfi           = ("game_nrfi",        "sum"),
        total_runs_1st      = ("runs_allowed_1st", "sum"),
        first_seen          = ("game_date",        "min"),
        last_seen           = ("game_date",        "max"),
    ).reset_index()

    agg["sp_yrfi"]              = agg["starts"] - agg["sp_nrfi"]
    agg["game_yrfi"]            = agg["starts"] - agg["game_nrfi"]
    agg["sp_nrfi_pct"]          = (agg["sp_nrfi"]   / agg["starts"] * 100).round(1)
    agg["game_nrfi_pct"]        = (agg["game_nrfi"] / agg["starts"] * 100).round(1)
    agg["avg_runs_allowed_1st"] = (agg["total_runs_1st"] / agg["starts"]).round(3)

    # Year range (e.g. "2021-2026")
    agg["years_active"] = (
        agg["first_seen"].dt.year.astype(str) + "–" +
        agg["last_seen"].dt.year.astype(str)
    )
    agg["last_seen"] = agg["last_seen"].dt.date

    return agg[[
        "pitcher", "starts",
        "sp_nrfi", "sp_yrfi", "sp_nrfi_pct",
        "game_nrfi", "game_yrfi", "game_nrfi_pct",
        "avg_runs_allowed_1st",
        "years_active", "last_seen",
    ]].sort_values("starts", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, min_starts: int, top_n: int = 30):
    filtered = df[df["starts"] >= min_starts].copy()
    total_games = len(df)   # game-level before aggregation — approximated

    print(f"\n{'='*75}")
    print(f"  NRFI PITCHER BACKTEST  |  min {min_starts} starts  |  "
          f"{len(filtered)} qualifying pitchers")
    print(f"{'='*75}")

    # ── Best NRFI pitchers ──────────────────────────────────────────────────
    print(f"\n  TOP {top_n} — Best SP NRFI% (own half-inning scoreless rate)\n")
    print(f"  {'Pitcher':<25} {'GS':>4} {'NRFI':>5} {'YRFI':>5} {'NRFI%':>6}  "
          f"{'GameNRFI%':>9}  {'AvgR/1st':>8}  {'Span'}")
    print("  " + "-" * 73)
    for _, r in filtered.nlargest(top_n, "sp_nrfi_pct").iterrows():
        print(f"  {r['pitcher']:<25} {r['starts']:>4} "
              f"{int(r['sp_nrfi']):>5} {int(r['sp_yrfi']):>5} "
              f"{r['sp_nrfi_pct']:>5.1f}%  "
              f"{r['game_nrfi_pct']:>8.1f}%  "
              f"{r['avg_runs_allowed_1st']:>8.3f}  "
              f"{r['years_active']}")

    # ── Worst NRFI pitchers (bet YRFI against them) ─────────────────────────
    print(f"\n  BOTTOM {top_n} — Worst SP NRFI% (best YRFI targets)\n")
    print(f"  {'Pitcher':<25} {'GS':>4} {'NRFI':>5} {'YRFI':>5} {'NRFI%':>6}  "
          f"{'GameNRFI%':>9}  {'AvgR/1st':>8}  {'Span'}")
    print("  " + "-" * 73)
    for _, r in filtered.nsmallest(top_n, "sp_nrfi_pct").iterrows():
        print(f"  {r['pitcher']:<25} {r['starts']:>4} "
              f"{int(r['sp_nrfi']):>5} {int(r['sp_yrfi']):>5} "
              f"{r['sp_nrfi_pct']:>5.1f}%  "
              f"{r['game_nrfi_pct']:>8.1f}%  "
              f"{r['avg_runs_allowed_1st']:>8.3f}  "
              f"{r['years_active']}")

    # ── Overall league stats from the data ─────────────────────────────────
    print(f"\n  {'─'*73}")
    league_nrfi = filtered["sp_nrfi"].sum()
    league_gs   = filtered["starts"].sum()
    print(f"  League-wide SP NRFI%:  {league_nrfi/league_gs*100:.1f}% "
          f"({int(league_nrfi):,} NRFI / {int(league_gs):,} weighted starts)")
    print(f"  League-wide avg runs allowed in 1st: "
          f"{filtered['avg_runs_allowed_1st'].mean():.3f}")
    print(f"  {'─'*73}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Search helper (call from Python or add --search CLI arg)
# ─────────────────────────────────────────────────────────────────────────────

def lookup_pitcher(df: pd.DataFrame, name: str) -> None:
    """Print the NRFI record for a specific pitcher (partial name match)."""
    hits = df[df["pitcher"].str.contains(name, case=False, na=False)]
    if hits.empty:
        print(f"  No pitcher found matching '{name}'")
        return
    print(f"\n  NRFI record for '{name}':\n")
    print(f"  {'Pitcher':<25} {'GS':>4} {'NRFI':>5} {'YRFI':>5} {'NRFI%':>6}  "
          f"{'GameNRFI%':>9}  {'AvgR/1st':>8}  {'Span'}")
    print("  " + "-" * 73)
    for _, r in hits.iterrows():
        print(f"  {r['pitcher']:<25} {r['starts']:>4} "
              f"{int(r['sp_nrfi']):>5} {int(r['sp_yrfi']):>5} "
              f"{r['sp_nrfi_pct']:>5.1f}%  "
              f"{r['game_nrfi_pct']:>8.1f}%  "
              f"{r['avg_runs_allowed_1st']:>8.3f}  "
              f"{r['years_active']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-pitcher NRFI backtest from plate appearance CSVs."
    )
    parser.add_argument(
        "--min-starts", type=int, default=20,
        help="Minimum starts to appear in the summary (default: 20)"
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of pitchers to show in each table (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUT_PATH),
        help=f"CSV output path (default: {OUT_PATH})"
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help="Look up a specific pitcher by (partial) name after building"
    )
    args = parser.parse_args()

    out_path = Path(args.output)

    print(f"\nNRFI Pitcher Backtest — {date.today()}")
    print(f"Reading PA files from: {PA_DIR}\n")

    # 1. Load
    pa = load_pa_data(PA_DIR)

    # 2. Build game-level NRFI records
    print("\nBuilding game-level NRFI records...")
    games = build_nrfi_records(pa)
    print(f"  {len(games):,} games with complete 1st-inning data")

    nrfi_rate = games["game_nrfi"].mean() * 100
    print(f"  Overall game NRFI rate: {nrfi_rate:.1f}%")

    # 3. Aggregate per pitcher
    print("\nAggregating per-pitcher records...")
    pitcher_df = aggregate_pitcher_records(games)
    print(f"  {len(pitcher_df):,} unique pitchers in dataset")

    # 4. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pitcher_df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    # 5. Print summary
    print_summary(pitcher_df, min_starts=args.min_starts, top_n=args.top)

    # 6. Optional pitcher search
    if args.search:
        lookup_pitcher(pitcher_df, args.search)

    print(f"Done. Full table saved to:\n  {out_path}\n")


if __name__ == "__main__":
    main()
