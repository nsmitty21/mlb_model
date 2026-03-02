"""
update_model.py — Nightly: grade yesterday's bets + retrain
Usage: python update_model.py [--date YYYY-MM-DD]
"""
import argparse, numpy as np, pandas as pd
from pathlib import Path
from datetime import date, timedelta
from data_fetcher import fetch_yesterday_results
from features import build_game_features, FEATURE_COLS
from model import train_models

DATA_DIR    = Path("data")
BET_LOG     = Path("bet_log.csv")
TRAIN_CACHE = DATA_DIR / "training_dataset.parquet"

def american_pnl(odds, units):
    if odds > 0: return units * odds / 100
    return units * 100 / abs(odds)

def grade(pick, result, market):
    hw = result.get("home_win",-1); hr = result.get("home_runs",np.nan); ar = result.get("away_runs",np.nan); tr = result.get("total_runs",np.nan)
    if market == "ML":
        if hw < 0: return "void"
        return "win" if (pick.get("side")=="home") == (hw==1) else "loss"
    elif market == "RL":
        if pd.isna(hr) or pd.isna(ar): return "void"
        d = hr - ar
        if pick.get("side")=="home": return "win" if d > 1.5 else "loss"
        return "win" if d < -1.5 else "loss"
    elif market == "Total":
        if pd.isna(tr): return "void"
        try: line = float(str(pick.get("team","")).split()[-1])
        except: return "void"
        if pick.get("side")=="over": return "win" if tr > line else "loss"
        return "win" if tr < line else "loss"
    return "void"

def update(target_date=None):
    if target_date is None: target_date = (date.today()-timedelta(days=1)).isoformat()
    print(f"\n=== Nightly Update: {target_date} ===")
    results_df = fetch_yesterday_results(target_date)
    if results_df.empty: print("[warn] No results."); return
    results_map = {(r["home_team"].lower(),r["away_team"].lower()): r.to_dict() for _,r in results_df.iterrows()}

    # Grade bet log
    if BET_LOG.exists():
        bl = pd.read_csv(BET_LOG)
        pending = bl[(bl["date"]==target_date) & (bl["result"].isna() | (bl["result"]==""))]
        graded = 0
        for idx, row in pending.iterrows():
            key = (str(row.get("home_team","")).lower(), str(row.get("away_team","")).lower())
            result_data = results_map.get(key)
            if result_data is None: continue
            outcome = grade(row.to_dict(), result_data, row.get("market",""))
            pnl = american_pnl(int(row.get("odds",-110)), int(row.get("units",1))) if outcome=="win" else (-float(row.get("units",1)) if outcome=="loss" else 0.0)
            bl.at[idx,"result"] = outcome; bl.at[idx,"pnl"] = round(pnl,2)
            bl.at[idx,"home_score"] = result_data.get("home_runs",""); bl.at[idx,"away_score"] = result_data.get("away_runs","")
            graded += 1
        bl.to_csv(BET_LOG, index=False); print(f"  Graded {graded} bets")

    # Append to training data
    new_rows = []
    for _, r in results_df.iterrows():
        feats = build_game_features(r.get("home_team",""),r.get("away_team",""),{},{},{},{})
        feats.update({"date":target_date,"home_team":r.get("home_team"),"away_team":r.get("away_team"),
                      "home_win":int(r.get("home_win",0)),"home_runs":r.get("home_runs"),"away_runs":r.get("away_runs"),
                      "total_runs":r.get("total_runs"),"home_covered_rl":int(r.get("home_covered_rl",0)),"went_over":np.nan})
        new_rows.append(feats)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([pd.read_parquet(TRAIN_CACHE), new_df], ignore_index=True) if TRAIN_CACHE.exists() else new_df
        combined.to_parquet(TRAIN_CACHE); print(f"  Added {len(new_rows)} games (total: {len(combined)})")
        combined = combined.dropna(subset=["home_win"])
        if "total_runs" in combined.columns:
            combined["went_over"] = (combined["total_runs"] > combined["total_runs"].median()).astype(int)
        print("\n  Retraining..."); train_models(combined); print(f"\n✓ Update complete for {target_date}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args = parser.parse_args()
    update(args.date)
