"""
fetch_historical.py - Pull 2021-2025 MLB data
Uses ESPN API for game results (reliable, no scraping blocks)
"""
import os, sys, time, warnings
import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    import pybaseball
    from pybaseball import team_batting, team_pitching, pitching_stats, batting_stats
    pybaseball.cache.enable()
except ImportError:
    print("[ERROR] pybaseball not installed. Run: pip install pybaseball")
    sys.exit(1)

from config import HIST_DIR, TRAIN_YEARS
DATA_DIR        = HIST_DIR
HISTORICAL_PATH = os.path.join(HIST_DIR, "historical_stats.parquet")

os.makedirs(DATA_DIR, exist_ok=True)

HEADERS   = {"User-Agent": "Mozilla/5.0"}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

ESPN_TO_STD = {
    "CWS":"CHW","KC":"KCR","SD":"SDP","SF":"SFG","TB":"TBR","WSH":"WSN"
}

def fetch_team_batting(years):
    frames = []
    for yr in years:
        print(f"  Fetching team batting {yr}...")
        try:
            df = team_batting(yr)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["Team","team","teamName"]:
        if col in out.columns:
            out.rename(columns={col:"team"}, inplace=True)
            break
    return out

def fetch_team_pitching(years):
    frames = []
    for yr in years:
        print(f"  Fetching team pitching {yr}...")
        try:
            df = team_pitching(yr)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    for col in ["Team","team","teamName"]:
        if col in out.columns:
            out.rename(columns={col:"team"}, inplace=True)
            break
    return out

def fetch_pitcher_stats(years):
    frames = []
    for yr in years:
        print(f"  Fetching pitcher stats {yr}...")
        try:
            df = pitching_stats(yr, qual=10)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_batter_stats(years):
    frames = []
    for yr in years:
        print(f"  Fetching batter stats {yr}...")
        try:
            df = batting_stats(yr, qual=50)
            df["season"] = yr
            frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"    [warn] {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fetch_schedules(years):
    all_frames = []
    for yr in years:
        print(f"  Fetching ESPN game results {yr}...")
        rows, seen = [], set()
        try:
            r = requests.get(f"{ESPN_BASE}/teams", headers=HEADERS, timeout=15)
            team_ids = [(t["team"]["id"], t["team"]["abbreviation"])
                        for t in r.json()["sports"][0]["leagues"][0]["teams"]]
        except Exception as e:
            print(f"    [warn] {e}")
            continue
        for tid, abbr in team_ids:
            try:
                r = requests.get(f"{ESPN_BASE}/teams/{tid}/schedule?season={yr}",
                                 headers=HEADERS, timeout=15)
                for ev in r.json().get("events", []):
                    gid = ev.get("id","")
                    if gid in seen:
                        continue
                    seen.add(gid)
                    try:
                        comp = ev["competitions"][0]
                        if not comp.get("status",{}).get("type",{}).get("completed",False):
                            continue
                        comps = comp["competitors"]
                        home  = next(c for c in comps if c["homeAway"]=="home")
                        away  = next(c for c in comps if c["homeAway"]=="away")
                        hs    = int(home.get("score",{}).get("value",0) or 0)
                        as_   = int(away.get("score",{}).get("value",0) or 0)
                        ht    = ESPN_TO_STD.get(home["team"]["abbreviation"], home["team"]["abbreviation"])
                        at    = ESPN_TO_STD.get(away["team"]["abbreviation"], away["team"]["abbreviation"])

                        # ── Pull the actual game date from the event ──────────
                        raw_date = ev.get("date","")
                        game_date = pd.to_datetime(raw_date, utc=True).date().isoformat() if raw_date else ""

                        rows.append({
                            "season":      yr,
                            "date":        game_date,   # <-- preserved here
                            "game_id":     gid,
                            "fetch_team":  ht,
                            "opponent":    at,
                            "home_away":   "H",
                            "result":      "W" if hs > as_ else "L",
                            "runs_scored": float(hs),
                            "runs_allowed":float(as_),
                        })
                    except Exception:
                        continue
            except Exception:
                pass
            time.sleep(0.1)
        if rows:
            df = pd.DataFrame(rows).drop_duplicates(subset=["game_id"])
            print(f"    -> {len(df)} games for {yr}")
            all_frames.append(df)
        else:
            print(f"    [warn] No data for {yr}")
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()

def build_game_features(schedules_df, team_bat_df, team_pit_df):
    print("  Building game-level features...")
    if schedules_df.empty:
        return pd.DataFrame()

    bat_lookup, pit_lookup = {}, {}
    if not team_bat_df.empty and "team" in team_bat_df.columns:
        for _, row in team_bat_df.iterrows():
            bat_lookup[(str(row["team"]), int(row["season"]))] = row
    if not team_pit_df.empty and "team" in team_pit_df.columns:
        for _, row in team_pit_df.iterrows():
            pit_lookup[(str(row["team"]), int(row["season"]))] = row

    bat_features = ["AVG","OBP","SLG","OPS","R","HR","BB","SO","wRC+","wOBA","WAR"]
    pit_features = ["ERA","FIP","WHIP","K/9","BB/9","HR/9","K%","BB%","xFIP","WAR"]

    rows = []
    for _, game in schedules_df.iterrows():
        season    = int(game["season"])
        home_team = game["fetch_team"]
        away_team = game["opponent"]
        rs        = float(game["runs_scored"])
        ra        = float(game["runs_allowed"])

        row = {
            "season":    season,
            "date":      game.get("date",""),   # <-- carried through to features
            "game_id":   game.get("game_id",""),
            "home_team": home_team,
            "away_team": away_team,
            "home_runs": rs,
            "away_runs": ra,
            "total_runs":rs + ra,
            "home_win":  int(rs > ra),
            "run_diff":  rs - ra,
        }

        for prefix, team in [("home_bat_", home_team), ("away_bat_", away_team)]:
            stats = bat_lookup.get((team, season), {})
            for feat in bat_features:
                row[f"{prefix}{feat}"] = stats.get(feat, np.nan) if hasattr(stats,"get") else np.nan

        for prefix, team in [("home_pit_", home_team), ("away_pit_", away_team)]:
            stats = pit_lookup.get((team, season), {})
            for feat in pit_features:
                row[f"{prefix}{feat}"] = stats.get(feat, np.nan) if hasattr(stats,"get") else np.nan

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()

def main():
    print("="*60)
    print("MLB Historical Data Fetch (2021-2025)")
    print("="*60)

    print("\n[1/5] Team Batting Stats")
    team_bat = fetch_team_batting(TRAIN_YEARS)
    print(f"  -> {len(team_bat)} rows")

    print("\n[2/5] Team Pitching Stats")
    team_pit = fetch_team_pitching(TRAIN_YEARS)
    print(f"  -> {len(team_pit)} rows")

    print("\n[3/5] Pitcher Individual Stats")
    pitcher_stats_df = fetch_pitcher_stats(TRAIN_YEARS)
    print(f"  -> {len(pitcher_stats_df)} rows")

    print("\n[4/5] Batter Individual Stats")
    batter_stats_df = fetch_batter_stats(TRAIN_YEARS)
    print(f"  -> {len(batter_stats_df)} rows")

    print("\n[5/5] Game Results (ESPN API)")
    schedules = fetch_schedules(TRAIN_YEARS)
    print(f"  -> {len(schedules)} total games")

    print("\n[Building game features...]")
    game_features = build_game_features(schedules, team_bat, team_pit)
    print(f"  -> {len(game_features)} game rows with features")

    # Verify date column made it through
    if "date" in game_features.columns:
        sample = game_features["date"].dropna().iloc[:3].tolist()
        print(f"  -> Date column confirmed: {sample}")
    else:
        print("  [warn] Date column missing from game features!")

    print("\n[Saving data...]")
    if not team_bat.empty:
        team_bat.to_parquet(os.path.join(DATA_DIR,"team_batting.parquet"), index=False)
        print("  Saved team_batting.parquet")
    if not team_pit.empty:
        team_pit.to_parquet(os.path.join(DATA_DIR,"team_pitching.parquet"), index=False)
        print("  Saved team_pitching.parquet")
    if not pitcher_stats_df.empty:
        pitcher_stats_df.to_parquet(os.path.join(DATA_DIR,"pitcher_stats.parquet"), index=False)
        print("  Saved pitcher_stats.parquet")
    if not batter_stats_df.empty:
        batter_stats_df.to_parquet(os.path.join(DATA_DIR,"batter_stats.parquet"), index=False)
        print("  Saved batter_stats.parquet")
    if not schedules.empty:
        schedules.to_parquet(os.path.join(DATA_DIR,"schedules.parquet"), index=False)
        print("  Saved schedules.parquet")
    if not game_features.empty:
        game_features.to_parquet(HISTORICAL_PATH, index=False)
        print(f"  Saved historical_stats.parquet ({len(game_features)} game rows)")
    else:
        print("  [warn] No game features saved")

    print("\nDone! Run next: python train_model.py")

if __name__ == "__main__":
    main()
