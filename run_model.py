"""
run_model.py — Generate Today's MLB Picks
Loads lines, runs 3 models, applies edge thresholds, writes today_picks.json
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
from datetime import date, datetime
warnings.filterwarnings("ignore")

from config import (DATA_DIR, MODELS_DIR, LINES_PATH, PICKS_PATH, MODEL_PATH,
    EDGE_1U, EDGE_2U, EDGE_3U, HOME_FIELD_WIN_PCT_BOOST, LOGS_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BATTING_FEATS  = ["OPS","wOBA","AVG","OBP","SLG","HR","BB%","K%","R"]
PITCHING_FEATS = ["ERA","FIP","WHIP","K/9","BB/9","xFIP"]
FEATURE_COLS   = ([f"home_bat_{f}" for f in BATTING_FEATS]+[f"away_bat_{f}" for f in BATTING_FEATS]+
                  [f"home_pit_{f}" for f in PITCHING_FEATS]+[f"away_pit_{f}" for f in PITCHING_FEATS]+
                  [f"diff_bat_{f}" for f in BATTING_FEATS]+[f"diff_pit_{f}" for f in PITCHING_FEATS]+["home_field"])

def american_to_prob(o):
    if o is None or (isinstance(o,float) and np.isnan(o)): return None
    return (100/(o+100)) if o>0 else (abs(o)/(abs(o)+100))

def calc_edge(mp, bp): return (mp-bp)*100 if mp and bp else 0.0
def units_for_edge(e):
    if e>=EDGE_3U: return 3
    if e>=EDGE_2U: return 2
    if e>=EDGE_1U: return 1
    return 0

def load_team_stats():
    bat,pit={},{}
    bp=os.path.join(DATA_DIR,"team_batting.parquet")
    pp=os.path.join(DATA_DIR,"team_pitching.parquet")
    if os.path.exists(bp):
        df=pd.read_parquet(bp)
        if "team" in df.columns:
            for _,r in df.sort_values("season" if "season" in df.columns else df.columns[0]).groupby("team").last().reset_index().iterrows():
                bat[r["team"]]=r.to_dict()
    if os.path.exists(pp):
        df=pd.read_parquet(pp)
        if "team" in df.columns:
            for _,r in df.sort_values("season" if "season" in df.columns else df.columns[0]).groupby("team").last().reset_index().iterrows():
                pit[r["team"]]=r.to_dict()
    return bat,pit

def build_row(home,away,bat,pit):
    row={}
    for pfx,tm in [("home_bat_",home),("away_bat_",away)]:
        s=bat.get(tm,{})
        for f in BATTING_FEATS: row[f"{pfx}{f}"]=s.get(f,np.nan)
    for pfx,tm in [("home_pit_",home),("away_pit_",away)]:
        s=pit.get(tm,{})
        for f in PITCHING_FEATS: row[f"{pfx}{f}"]=s.get(f,np.nan)
    row["home_field"]=1
    for f in BATTING_FEATS:
        hv,av=row.get(f"home_bat_{f}",np.nan),row.get(f"away_bat_{f}",np.nan)
        row[f"diff_bat_{f}"]=(hv-av) if not(np.isnan(hv) or np.isnan(av)) else np.nan
    for f in PITCHING_FEATS:
        hv,av=row.get(f"home_pit_{f}",np.nan),row.get(f"away_pit_{f}",np.nan)
        if not(np.isnan(hv) or np.isnan(av)):
            row[f"diff_pit_{f}"]=(av-hv) if f in ("ERA","FIP","WHIP","BB/9") else (hv-av)
        else: row[f"diff_pit_{f}"]=np.nan
    return row

def predict(row,models):
    X=pd.DataFrame([row])
    for c in FEATURE_COLS:
        if c not in X.columns: X[c]=np.nan
    X=X[FEATURE_COLS].astype(float).fillna(0)
    res={}
    if models.get("win"):
        try:
            p=models["win"].predict_proba(X)[0]; res["hwp"]=float(p[1]); res["awp"]=float(p[0])
        except: res["hwp"]=res["awp"]=None
    else: res["hwp"]=res["awp"]=None
    for k,nm in [("diff","pred_run_diff"),("total","pred_total")]:
        if models.get(k):
            try: res[nm]=float(models[k].predict(X)[0])
            except: res[nm]=None
        else: res[nm]=None
    return res

def load_models():
    m={}
    for s in ["win","diff","total"]:
        p=MODEL_PATH.replace(".joblib",f"_{s}.joblib")
        m[s]=joblib.load(p) if os.path.exists(p) else None
    return m

def no_vig(mh,ma):
    ph,pa=american_to_prob(mh),american_to_prob(ma)
    if not ph or not pa: return None,None
    t=ph+pa; return ph/t,pa/t

def generate_picks(games,models,bat,pit):
    picks=[]
    for g in games:
        ht,at=g["home_team"],g["away_team"]
        books=g.get("books",{})
        row=build_row(ht,at,bat,pit)
        preds=predict(row,models)
        hwp,awp=preds.get("hwp"),preds.get("awp")
        pdiff,ptot=preds.get("pred_run_diff"),preds.get("pred_total")
        if hwp: hwp=min(0.95,hwp+HOME_FIELD_WIN_PCT_BOOST); awp=1.0-hwp
        gpicks=[]
        # ML picks
        for side,team,mp in [("home",ht,hwp),("away",at,awp)]:
            if mp is None: continue
            best_ml,best_bk=None,None
            for bk in ["fanduel","draftkings"]:
                if bk in books:
                    ml=books[bk].get(f"ml_{side}")
                    if ml is not None and (best_ml is None or ml>best_ml):
                        best_ml,best_bk=ml,bk
            if best_ml is None: continue
            bih,bia=no_vig(books.get("fanduel",{}).get("ml_home") or books.get("draftkings",{}).get("ml_home"),
                           books.get("fanduel",{}).get("ml_away") or books.get("draftkings",{}).get("ml_away"))
            bp=bih if side=="home" else bia
            if bp is None: continue
            edge=calc_edge(mp,bp); units=units_for_edge(edge)
            if units>0:
                gpicks.append({"type":"ML","side":side,"team":team,"opponent":at if side=="home" else ht,
                    "odds":best_ml,"book":best_bk,"model_prob":round(mp*100,1),"book_prob":round(bp*100,1),
                    "edge":round(edge,2),"units":units})
        # Spread
        sh=g.get("spread_home")
        if sh is not None and pdiff is not None:
            for side,spread,md in [("home",sh,pdiff),("away",-sh,-pdiff)]:
                team=ht if side=="home" else at
                best_j,best_bk=None,None
                for bk in ["fanduel","draftkings"]:
                    if bk in books:
                        j=books[bk].get(f"spread_juice_{side}")
                        if j is not None and (best_j is None or j>best_j):
                            best_j,best_bk=j,bk
                if best_j is None: best_j,best_bk=-110,"fanduel"
                cp=1/(1+np.exp(-(md+spread)*0.3))
                bp=american_to_prob(best_j) or 0.524
                edge=calc_edge(cp,bp); units=units_for_edge(edge)
                if units>0:
                    gpicks.append({"type":"SPREAD","side":side,"team":team,"opponent":at if side=="home" else ht,
                        "spread":spread,"odds":best_j,"book":best_bk,"model_diff":round(md,2),
                        "cover_prob":round(cp*100,1),"book_prob":round(bp*100,1),"edge":round(edge,2),"units":units})
        # Total
        tl=g.get("total")
        if tl is not None and ptot is not None:
            diff_from_line=ptot-tl
            for direction,diff in [("over",diff_from_line),("under",-diff_from_line)]:
                op=1/(1+np.exp(-diff*0.25))
                best_j,best_bk=None,None
                for bk in ["fanduel","draftkings"]:
                    if bk in books:
                        j=books[bk].get(f"total_{direction}_juice")
                        if j is not None and (best_j is None or j>best_j):
                            best_j,best_bk=j,bk
                if best_j is None: best_j,best_bk=-110,"fanduel"
                bp=american_to_prob(best_j) or 0.524
                edge=calc_edge(op,bp); units=units_for_edge(edge)
                if units>0:
                    gpicks.append({"type":"TOTAL","direction":direction.upper(),"home_team":ht,"away_team":at,
                        "total_line":tl,"pred_total":round(ptot,2),"odds":best_j,"book":best_bk,
                        "book_prob":round(bp*100,1),"edge":round(edge,2),"units":units})
        if gpicks:
            picks.append({"game_id":g["game_id"],"date":g["date"],"commence":g["commence"],
                "home_team":ht,"away_team":at,"picks":sorted(gpicks,key=lambda x:x["units"],reverse=True)})
    return picks

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",default=None)
    args,_=parser.parse_known_args()
    td=args.date or date.today().isoformat()
    print(f"MLB Model Run — {td}")
    if not os.path.exists(LINES_PATH):
        print("[ERROR] Run: python pull_lines.py"); sys.exit(1)
    with open(LINES_PATH) as f: games=json.load(f)
    print(f"  {len(games)} games loaded")
    models=load_models()
    loaded=[k for k,v in models.items() if v]
    print(f"  Models: {', '.join(loaded) if loaded else 'NONE'}")
    bat,pit=load_team_stats()
    print(f"  Teams — bat:{len(bat)} pit:{len(pit)}")
    picks=generate_picks(games,models,bat,pit)
    all_p=[p for g in picks for p in g["picks"]]
    print(f"\n  3U: {sum(1 for p in all_p if p['units']==3)}  2U: {sum(1 for p in all_p if p['units']==2)}  1U: {sum(1 for p in all_p if p['units']==1)}")
    out={"generated_at":datetime.now().isoformat(),"date":td,"total_picks":len(all_p),"games":picks}
    with open(PICKS_PATH,"w") as f: json.dump(out,f,indent=2)
    print(f"Saved to today_picks.json | Run: python app.py")

if __name__=="__main__":
    main()