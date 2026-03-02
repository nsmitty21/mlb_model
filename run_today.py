"""
run_today.py — Daily Pick Generator
Usage: python run_today.py [--date YYYY-MM-DD]
"""
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from datetime import date, datetime
from pull_lines import fetch_lines
from data_fetcher import fetch_todays_starters, fetch_pitcher_stats
from features import build_game_features, FEATURE_COLS
from model import load_models, predict_game, load_meta

DATA_DIR   = Path("data")
BET_LOG    = Path("bet_log.csv")
PICKS_FILE = Path("today_picks.json")

def build_pitcher_lookup(pitcher_stats, year=None):
    if pitcher_stats is None or pitcher_stats.empty: return {}
    if year: pitcher_stats = pitcher_stats[pitcher_stats["year"]==year]
    lookup = {}
    for _, row in pitcher_stats.iterrows():
        name = str(row.get("Name","")).strip().lower()
        lookup[name] = {"season_era":row.get("ERA",np.nan),"season_fip":row.get("FIP",np.nan),"roll_era":row.get("ERA",np.nan),"roll_fip":row.get("FIP",np.nan),"roll_whip":row.get("WHIP",np.nan),"roll_k_per_9":row.get("K/9",np.nan),"roll_bb_per_9":row.get("BB/9",np.nan),"roll_hr_per_9":row.get("HR/9",np.nan)}
    return lookup

def pivot_lines(df):
    fd = df[df["book"]=="fanduel"].copy()
    dk = df[df["book"]=="draftkings"].copy()
    def ren(d, p):
        return d.rename(columns={"ml_home":f"{p}_ml_home","ml_away":f"{p}_ml_away","spread_home":f"{p}_spread_home","spread_juice_home":f"{p}_spread_juice_home","spread_juice_away":f"{p}_spread_juice_away","total_line":f"{p}_total","total_over_juice":f"{p}_total_over_juice","total_under_juice":f"{p}_total_under_juice"})[
            ["home_team","away_team","commence","game_id",f"{p}_ml_home",f"{p}_ml_away",f"{p}_spread_home",f"{p}_spread_juice_home",f"{p}_spread_juice_away",f"{p}_total",f"{p}_total_over_juice",f"{p}_total_under_juice"]]
    return ren(fd,"fd").merge(ren(dk,"dk"), on=["home_team","away_team","commence","game_id"], how="outer")

def run(target_date=None):
    if target_date is None: target_date = date.today().isoformat()
    print(f"\n{'='*60}\n  MLB Edge — Picks: {target_date}\n{'='*60}\n")

    lines_df = fetch_lines(target_date)
    if lines_df.empty: print("[error] No lines."); return

    starters_df = fetch_todays_starters(target_date)

    try:
        ml_model, spread_model, total_model = load_models()
        meta = load_meta(); print(f"  Models loaded. Trained on {meta.get('training_rows','?')} games.")
    except FileNotFoundError as e:
        print(f"[error] {e}"); return

    pitcher_stats = None
    try:
        year = int(target_date[:4])
        cached = sorted(DATA_DIR.glob("pitcher_stats_*.parquet"))
        if cached: pitcher_stats = pd.read_parquet(cached[-1])
    except: pass
    pitcher_lookup = build_pitcher_lookup(pitcher_stats, year=int(target_date[:4])-1) if pitcher_stats is not None else {}

    games = pivot_lines(lines_df)
    if not starters_df.empty:
        games = games.merge(starters_df[["home_team","away_team","home_sp","away_sp"]], on=["home_team","away_team"], how="left")
    games["home_sp"] = games.get("home_sp", pd.Series([""]*len(games))).fillna("")
    games["away_sp"] = games.get("away_sp", pd.Series([""]*len(games))).fillna("")

    all_picks = []
    for _, game in games.iterrows():
        ht = str(game.get("home_team","")); at = str(game.get("away_team",""))
        hsp = str(game.get("home_sp","")); asp = str(game.get("away_sp",""))
        feats = build_game_features(ht, at, pitcher_lookup.get(hsp.lower(),{}), pitcher_lookup.get(asp.lower(),{}), {}, {})
        picks = predict_game(feats, ml_model, spread_model, total_model,
            fd_ml_home=game.get("fd_ml_home"), fd_ml_away=game.get("fd_ml_away"),
            dk_ml_home=game.get("dk_ml_home"), dk_ml_away=game.get("dk_ml_away"),
            fd_spread_home=game.get("fd_spread_home"), fd_spread_juice_home=game.get("fd_spread_juice_home"), fd_spread_juice_away=game.get("fd_spread_juice_away"),
            dk_spread_home=game.get("dk_spread_home"), dk_spread_juice_home=game.get("dk_spread_juice_home"), dk_spread_juice_away=game.get("dk_spread_juice_away"),
            fd_total=game.get("fd_total"), fd_total_over_juice=game.get("fd_total_over_juice"), fd_total_under_juice=game.get("fd_total_under_juice"),
            dk_total=game.get("dk_total"), dk_total_over_juice=game.get("dk_total_over_juice"), dk_total_under_juice=game.get("dk_total_under_juice"),
            home_team=ht, away_team=at, home_sp=hsp, away_sp=asp, commence=str(game.get("commence","")))
        all_picks.extend(picks)

    all_picks.sort(key=lambda p: (-p["units"], -p["edge"]))

    print(f"\n{'='*60}")
    if not all_picks:
        print("  No picks meet edge threshold today.")
    else:
        print(f"  {len(all_picks)} PICKS")
        for p in all_picks:
            print(f"  {'⭐'*p['units']} {p['units']}u [{p['market']}] {p['team']} ({p['odds']:+d} @ {p['book'].upper()}) Edge:{p['edge']*100:.1f}%")
            print(f"       {p['away_team']} @ {p['home_team']}  SP: {p.get('away_sp','?')} vs {p.get('home_sp','?')}")
    print(f"{'='*60}")

    picks_out = [{**p,"date":target_date,"result":"","pnl":None,"id":f"{target_date}_{p['home_team']}_{p['market']}_{p['side']}"} for p in all_picks]
    with open(PICKS_FILE,"w") as f: json.dump({"date":target_date,"picks":picks_out}, f, indent=2)
    print(f"\nSaved {len(picks_out)} picks to {PICKS_FILE}")

    if picks_out:
        new_bets = pd.DataFrame(picks_out)
        new_bets["result"] = ""; new_bets["pnl"] = np.nan; new_bets["home_score"] = ""; new_bets["away_score"] = ""
        cols = ["id","date","market","side","team","home_team","away_team","home_sp","away_sp","odds","units","book","edge","model_prob","book_prob","commence","result","pnl","home_score","away_score"]
        new_bets = new_bets[[c for c in cols if c in new_bets.columns]]
        if BET_LOG.exists():
            ex = pd.read_csv(BET_LOG); ex = ex[~ex["id"].isin(new_bets["id"])]
            pd.concat([ex, new_bets], ignore_index=True).to_csv(BET_LOG, index=False)
        else:
            new_bets.to_csv(BET_LOG, index=False)
        print(f"Logged {len(new_bets)} bets to {BET_LOG}")

    print(f"\n✓ Done. Dashboard: python app.py → http://localhost:5000")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    run(args.date)
