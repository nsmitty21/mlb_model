"""
backtest.py - MLB Model Backtest Against Real Historical Lines
"""
import os, sys, json, warnings
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from config import HIST_DIR, MODEL_DIR, TRAIN_YEARS, HIST_ODDS_DIR

HISTORICAL_PATH = os.path.join(HIST_DIR, "historical_stats.parquet")
REPORT_PATH     = "backtest_report.txt"
RESULTS_PATH    = "backtest_results.json"

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CHW",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Cleveland Indians":"CLE",
    "Colorado Rockies":"COL","Detroit Tigers":"DET","Houston Astros":"HOU",
    "Kansas City Royals":"KCR","Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD",
    "Miami Marlins":"MIA","Milwaukee Brewers":"MIL","Minnesota Twins":"MIN",
    "New York Mets":"NYM","New York Yankees":"NYY","Oakland Athletics":"OAK",
    "Athletics":"OAK","Philadelphia Phillies":"PHI","Pittsburgh Pirates":"PIT",
    "San Diego Padres":"SDP","San Francisco Giants":"SFG","Seattle Mariners":"SEA",
    "St. Louis Cardinals":"STL","Tampa Bay Rays":"TBR","Texas Rangers":"TEX",
    "Toronto Blue Jays":"TOR","Washington Nationals":"WSN",
}

def name_to_abbr(name):
    return TEAM_NAME_TO_ABBR.get(str(name).strip(), str(name)[:3].upper())

def american_to_prob(odds):
    if odds is None or (isinstance(odds, float) and np.isnan(odds)): return None
    odds = float(odds)
    if odds > 0: return 100/(odds+100)
    return abs(odds)/(abs(odds)+100)

def remove_vig(p1, p2):
    if p1 is None or p2 is None: return p1, p2
    total = p1+p2
    if total <= 0: return p1, p2
    return p1/total, p2/total

def pnl_from_odds(odds, won, units=1):
    if won:
        if odds > 0: return units*(odds/100)
        return units*(100/abs(odds))
    return -units

def load_historical_odds():
    print("Loading historical odds CSVs...")
    frames = []
    for yr in TRAIN_YEARS:
        path = os.path.join(HIST_ODDS_DIR, f"baseball_mlb_{yr}.csv")
        if not os.path.exists(path):
            print(f"  [warn] Not found: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["season"] = yr
        df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
        df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
        df = df.sort_values("snapshot_time").drop_duplicates(subset=["game_id"], keep="last")
        frames.append(df)
        print(f"  {yr}: {len(df)} games")
    if not frames:
        print(f"[ERROR] No odds CSVs at: {HIST_ODDS_DIR}")
        sys.exit(1)
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(combined)} games")
    return combined

def load_intraday_odds():
    print("Loading intraday closing line CSVs...")
    frames = []
    for yr in TRAIN_YEARS:
        path = os.path.join(HIST_ODDS_DIR, f"baseball_mlb_{yr}_intraday.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path, low_memory=False)
        df["season"] = yr
        df["hours_to_tip"] = pd.to_numeric(df.get("hours_to_tip", pd.Series([99]*len(df))), errors="coerce")
        closing = df[df["hours_to_tip"] <= 2].copy()
        if closing.empty:
            closing = df.sort_values("hours_to_tip").drop_duplicates(subset=["game_id"], keep="first")
        else:
            closing = closing.sort_values("hours_to_tip").drop_duplicates(subset=["game_id"], keep="first")
        frames.append(closing)
        print(f"  {yr}: {len(closing)} closing snapshots")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_model():
    model_path  = os.path.join(MODEL_DIR, "mlb_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    feat_path   = os.path.join(MODEL_DIR, "features.json")
    for p in [model_path, scaler_path, feat_path]:
        if not Path(p).exists():
            print(f"[ERROR] Missing: {p} — run python train_model.py first.")
            sys.exit(1)
    models = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feat_path) as f: feat_cols = json.load(f)
    return models, scaler, feat_cols

def load_game_features():
    if not Path(HISTORICAL_PATH).exists():
        print(f"[ERROR] Missing {HISTORICAL_PATH} — run fetch_historical.py first.")
        sys.exit(1)
    return pd.read_parquet(HISTORICAL_PATH)

def generate_predictions(game_df, models, scaler, feat_cols):
    print(f"Generating predictions for {len(game_df)} games...")
    for c in feat_cols:
        if c not in game_df.columns: game_df[c] = 0.0
    game_df[feat_cols] = game_df[feat_cols].fillna(game_df[feat_cols].median())
    X        = game_df[feat_cols].values
    X_scaled = scaler.transform(X)
    game_df["pred_home_win_prob"] = models["ml"].predict_proba(X_scaled)[:, 1]
    game_df["pred_run_diff"]      = models["rl"].predict(X_scaled)
    game_df["pred_total_runs"]    = models["totals"].predict(X_scaled)
    return game_df

def merge_predictions_with_odds(game_df, odds_df):
    print("Merging predictions with historical odds...")
    odds_df = odds_df.copy()
    odds_df["home_abbr"] = odds_df["home_team"].apply(name_to_abbr)
    odds_df["away_abbr"] = odds_df["away_team"].apply(name_to_abbr)
    odds_df["game_date"] = pd.to_datetime(odds_df["commence_time"], utc=True, errors="coerce").dt.date.astype(str)
    odds_clean = odds_df.drop_duplicates(subset=["game_id"]).copy()

    game_df = game_df.copy()
    if "date" in game_df.columns:
        game_df["game_date"] = pd.to_datetime(game_df["date"], errors="coerce").dt.date.astype(str)

    print(f"  Sample game_df  home: {game_df['home_team'].dropna().iloc[:3].tolist()}")
    print(f"  Sample odds     home: {odds_clean['home_abbr'].dropna().iloc[:3].tolist()}")
    print(f"  Sample game_df  date: {game_df['game_date'].dropna().iloc[:3].tolist() if 'game_date' in game_df.columns else 'no date col'}")
    print(f"  Sample odds     date: {odds_clean['game_date'].dropna().iloc[:3].tolist()}")

    # Try date + team merge first
    if "game_date" in game_df.columns:
        merged = pd.merge(
            game_df, odds_clean,
            left_on=["home_team","away_team","game_date"],
            right_on=["home_abbr","away_abbr","game_date"],
            how="inner", suffixes=("","_odds")
        )
        if not merged.empty:
            print(f"  Matched {len(merged)} games (date + team merge)")
            return merged

    # Fall back to season + team merge
    print("  Trying season + team merge...")
    odds_clean["season"] = pd.to_datetime(odds_clean["commence_time"], utc=True, errors="coerce").dt.year
    merged = pd.merge(
        game_df, odds_clean,
        left_on=["home_team","away_team","season"],
        right_on=["home_abbr","away_abbr","season"],
        how="inner", suffixes=("","_odds")
    )
    merged = merged.drop_duplicates(subset=["home_team","away_team","season","home_win"])
    print(f"  Matched {len(merged)} games (season + team merge)")
    return merged

def run_backtest(merged_df, closing_df=None, edge_threshold=3.0):
    from scipy import stats as scipy_stats
    bets = []
    for _, row in merged_df.iterrows():
        home_win   = int(row.get("home_win", 0))
        pred_home  = float(row.get("pred_home_win_prob", 0.5))
        pred_away  = 1 - pred_home
        pred_total = float(row.get("pred_total_runs", 9.0))
        pred_diff  = float(row.get("pred_run_diff", 0.0))
        season     = int(row.get("season", 2024))
        game_id    = row.get("game_id_odds", row.get("game_id", ""))

        # Moneyline
        for side, model_p, ml_col, opp_col, won in [
            ("home", pred_home, "fa_ml_home", "fa_ml_away", home_win==1),
            ("away", pred_away, "fa_ml_away", "fa_ml_home", home_win==0),
        ]:
            odds = row.get(ml_col); opp_odds = row.get(opp_col)
            if pd.isna(odds) or pd.isna(opp_odds): continue
            bp, op = american_to_prob(odds), american_to_prob(opp_odds)
            if bp is None or op is None: continue
            cp, _ = remove_vig(bp, op)
            edge  = (model_p - cp)*100
            if edge >= edge_threshold:
                units = 3 if edge>=9 else 2 if edge>=6 else 1
                clv = None
                if closing_df is not None and not closing_df.empty and game_id:
                    cr = closing_df[closing_df["game_id"]==game_id]
                    if not cr.empty:
                        cc = "fd_ml_home" if side=="home" else "fd_ml_away"
                        if cc in cr.columns and pd.notna(cr[cc].iloc[0]):
                            cp2 = american_to_prob(float(cr[cc].iloc[0]))
                            if cp2: clv = round((model_p-cp2)*100, 2)
                bets.append({"season":season,"market":"moneyline","side":side,
                    "home":row.get("home_team",""),"away":row.get("away_team",""),
                    "odds":float(odds),"edge":round(edge,2),"units":units,
                    "won":won,"pnl":round(pnl_from_odds(float(odds),won,units),3),"clv":clv})

        # Totals
        tl = row.get("fa_total_line"); oo = row.get("fa_total_over_price")
        uo = row.get("fa_total_under_price"); at = row.get("total_runs")
        if all(pd.notna(x) for x in [tl, oo, uo, at]):
            line = float(tl)
            op2  = 1 - scipy_stats.norm.cdf(line, loc=pred_total, scale=2.8)
            up2  = 1 - op2
            ob, ub = american_to_prob(oo), american_to_prob(uo)
            if ob and ub:
                co, _ = remove_vig(ob, ub); cu, _ = remove_vig(ub, ob)
                for direction, mp, cp2, ov, won in [
                    ("over",  op2, co, float(oo), float(at)>line),
                    ("under", up2, cu, float(uo), float(at)<line),
                ]:
                    edge  = (mp-cp2)*100
                    units = 3 if edge>=9 else 2 if edge>=6 else 1 if edge>=edge_threshold else 0
                    if units > 0:
                        bets.append({"season":season,"market":"totals","side":direction,
                            "home":row.get("home_team",""),"away":row.get("away_team",""),
                            "odds":ov,"edge":round(edge,2),"units":units,
                            "won":won,"pnl":round(pnl_from_odds(ov,won,units),3),"clv":None})

        # Run Line
        rho = row.get("fa_spread_home_price"); rao = row.get("fa_spread_away_price")
        rl  = float(row.get("fa_spread_home_line",-1.5)) if pd.notna(row.get("fa_spread_home_line")) else -1.5
        if pd.notna(rho) and pd.notna(rao):
            rd = float(row.get("run_diff", row.get("home_runs",5)-row.get("away_runs",5)))
            hc = (rd+rl)>0
            for side, ov, opv, won in [("home",rho,rao,hc),("away",rao,rho,not hc)]:
                cp2 = 1-scipy_stats.norm.cdf(0,loc=pred_diff+(rl if side=="home" else -rl),scale=2.5)
                bp, op = american_to_prob(ov), american_to_prob(opv)
                if bp is None or op is None: continue
                cpv, _ = remove_vig(bp, op)
                edge   = (cp2-cpv)*100
                units  = 3 if edge>=9 else 2 if edge>=6 else 1 if edge>=edge_threshold else 0
                if units > 0:
                    bets.append({"season":season,"market":"run_line","side":side,
                        "home":row.get("home_team",""),"away":row.get("away_team",""),
                        "odds":float(ov),"edge":round(edge,2),"units":units,
                        "won":won,"pnl":round(pnl_from_odds(float(ov),won,units),3),"clv":None})
    return bets

def analyze_bets(bets, label=""):
    if not bets:
        return {"label":label,"n":0,"wins":0,"losses":0,"win_pct":0,"total_pnl":0,
                "total_wagered":0,"roi":0,"avg_edge":0,"avg_clv":None,
                "clv_positive_pct":None,"clv_n":0,"by_market":{},"by_season":{},"by_units":{}}
    df = pd.DataFrame(bets)
    wins = int(df["won"].sum()); losses = len(df)-wins
    total_pnl = float(df["pnl"].sum()); total_wagered = float(df["units"].sum())
    roi = (total_pnl/total_wagered*100) if total_wagered else 0
    clv_vals = df["clv"].dropna()
    by_market = {mkt:{"n":len(g),"wins":int(g["won"].sum()),"losses":len(g)-int(g["won"].sum()),
        "pnl":round(float(g["pnl"].sum()),2),"roi":round(float(g["pnl"].sum()/g["units"].sum()*100) if g["units"].sum() else 0,1)}
        for mkt, g in df.groupby("market")}
    by_season = {int(yr):{"n":len(g),"wins":int(g["won"].sum()),"losses":len(g)-int(g["won"].sum()),
        "pnl":round(float(g["pnl"].sum()),2),"roi":round(float(g["pnl"].sum()/g["units"].sum()*100) if g["units"].sum() else 0,1)}
        for yr, g in df.groupby("season")}
    by_units = {int(u):{"n":len(g),"wins":int(g["won"].sum()),"losses":len(g)-int(g["won"].sum()),
        "pnl":round(float(g["pnl"].sum()),2),"roi":round(float(g["pnl"].sum()/g["units"].sum()*100) if g["units"].sum() else 0,1)}
        for u, g in df.groupby("units")}
    return {
        "label":label,"n":len(df),"wins":wins,"losses":losses,
        "win_pct":round(wins/len(df)*100,1),"total_pnl":round(total_pnl,2),
        "total_wagered":round(total_wagered,1),"roi":round(roi,1),
        "avg_edge":round(float(df["edge"].mean()),2),
        "avg_clv":round(float(clv_vals.mean()),2) if len(clv_vals) else None,
        "clv_positive_pct":round(float((clv_vals>0).sum()/len(clv_vals)*100),1) if len(clv_vals) else None,
        "clv_n":int(len(clv_vals)),"by_market":by_market,"by_season":by_season,"by_units":by_units,
    }

def sweep_edge_thresholds(merged_df, closing_df):
    print("\nSweeping edge thresholds...")
    results = []
    for t in [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]:
        bets  = run_backtest(merged_df, closing_df, edge_threshold=t)
        stats = analyze_bets(bets)
        results.append({"threshold":t,"n_bets":stats["n"],"roi":stats["roi"],
                         "pnl":stats["total_pnl"],"win_pct":stats["win_pct"]})
        print(f"  Edge >= {t:.1f}%:  {stats['n']:5d} bets  ROI: {stats['roi']:+.1f}%  "
              f"PnL: {stats['total_pnl']:+.2f}u  Win%: {stats['win_pct']:.1f}%")
    return results

def write_report(stats, sweep):
    lines = ["="*65,"  MLB MODEL BACKTEST REPORT",
             f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}","="*65,"",
             "OVERALL RESULTS","─"*65,
             f"  Total bets     : {stats['n']}",
             f"  Record         : {stats['wins']}W - {stats['losses']}L  ({stats['win_pct']}%)",
             f"  Total P&L      : {stats['total_pnl']:+.2f} units",
             f"  Units wagered  : {stats['total_wagered']:.1f}",
             f"  ROI            : {stats['roi']:+.1f}%",
             f"  Avg edge       : {stats['avg_edge']:+.2f}%"]
    if stats.get("avg_clv") is not None:
        lines += [f"  Avg CLV        : {stats['avg_clv']:+.2f}%  ({stats['clv_n']} picks)",
                  f"  CLV positive   : {stats['clv_positive_pct']}% beat closing line"]
    lines += ["","BY MARKET","─"*65]
    for mkt,d in stats.get("by_market",{}).items():
        lines.append(f"  {mkt:<12}: {d['n']:5d} bets  {d['wins']}W-{d['losses']}L  PnL:{d['pnl']:+.2f}u  ROI:{d['roi']:+.1f}%")
    lines += ["","BY SEASON","─"*65]
    for yr,d in sorted(stats.get("by_season",{}).items()):
        lines.append(f"  {yr}: {d['n']:5d} bets  {d['wins']}W-{d['losses']}L  PnL:{d['pnl']:+.2f}u  ROI:{d['roi']:+.1f}%")
    lines += ["","BY UNIT SIZE","─"*65]
    for u,d in sorted(stats.get("by_units",{}).items()):
        lines.append(f"  {u}u plays : {d['n']:5d} bets  {d['wins']}W-{d['losses']}L  PnL:{d['pnl']:+.2f}u  ROI:{d['roi']:+.1f}%")
    lines += ["","EDGE THRESHOLD SWEEP","─"*65,""]
    lines.append(f"  {'Threshold':>10}  {'Bets':>6}  {'ROI':>8}  {'PnL':>9}  {'Win%':>7}")
    lines.append("  "+"─"*50)
    best = max(sweep, key=lambda x: x["roi"]) if sweep else None
    for s in sweep:
        marker = "  <-- OPTIMAL" if best and s["threshold"]==best["threshold"] else ""
        lines.append(f"  >= {s['threshold']:>5.1f}%    {s['n_bets']:>6}  {s['roi']:>+7.1f}%  {s['pnl']:>+8.2f}u  {s['win_pct']:>6.1f}%{marker}")
    if best:
        lines += ["",f"  Recommended: EDGE_1_UNIT = {best['threshold']} in config.py"]
    lines += ["","="*65,"HOW TO INTERPRET","─"*65,
              "  ROI > 0% = model has historical edge",
              "  CLV > 0  = picks beat closing lines (real edge signal)",
              "  Win% > 54% = ML model is directionally accurate","="*65]
    report = "\n".join(lines)
    print("\n"+report)
    with open(REPORT_PATH,"w",encoding="utf-8") as f: f.write(report)
    print(f"\n  Report saved: {REPORT_PATH}")

def main():
    print("="*65)
    print("  MLB MODEL BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)
    odds_df    = load_historical_odds()
    closing_df = load_intraday_odds()
    game_df    = load_game_features()
    models, scaler, feat_cols = load_model()
    game_df = generate_predictions(game_df, models, scaler, feat_cols)
    merged  = merge_predictions_with_odds(game_df, odds_df)
    if merged.empty:
        print("\n[ERROR] 0 games matched. Check HIST_ODDS_DIR in config.py.")
        sys.exit(1)
    print("\nRunning backtest...")
    bets  = run_backtest(merged, closing_df, edge_threshold=3.0)
    stats = analyze_bets(bets, label="default")
    print(f"  {len(bets)} qualifying bets found")
    sweep = sweep_edge_thresholds(merged, closing_df)
    all_results = {"stats":stats,"sweep":sweep,"generated_at":datetime.now().isoformat()}
    with open(RESULTS_PATH,"w") as f: json.dump(all_results,f,indent=2)
    write_report(stats, sweep)

if __name__ == "__main__":
    main()
