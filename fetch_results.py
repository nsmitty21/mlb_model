"""
fetch_results.py — Pull MLB results from ESPN, grade bets, save for retraining
Usage: python fetch_results.py [--date YYYY-MM-DD]
"""
import os, sys, json, requests
import pandas as pd
from datetime import date, datetime, timedelta
from config import BET_LOG_PATH, DATA_DIR, LOGS_DIR, SEASON_PATH

os.makedirs(LOGS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
ESPN = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_scores(game_date):
    ds = game_date.replace("-","")
    try:
        r = requests.get(ESPN, params={"dates":ds,"limit":100}, headers=HEADERS, timeout=15)
        r.raise_for_status(); data=r.json()
    except Exception as e:
        print(f"  [ERROR] ESPN fetch: {e}"); return []
    games=[]
    for ev in data.get("events",[]):
        try:
            comp=ev.get("competitions",[{}])[0]
            if not ev.get("status",{}).get("type",{}).get("completed",False): continue
            home=away=None
            for c in comp.get("competitors",[]):
                if c.get("homeAway")=="home": home=c
                else: away=c
            if not home or not away: continue
            hs,as_=int(home.get("score",0)),int(away.get("score",0))
            games.append({"espn_id":ev.get("id"),"date":game_date,
                "home_team":home.get("team",{}).get("shortDisplayName",""),
                "away_team":away.get("team",{}).get("shortDisplayName",""),
                "home_abbr":home.get("team",{}).get("abbreviation",""),
                "away_abbr":away.get("team",{}).get("abbreviation",""),
                "home_score":hs,"away_score":as_,"total":hs+as_,
                "home_win":int(hs>as_),"run_diff":hs-as_})
        except: continue
    print(f"  {len(games)} completed games for {game_date}")
    return games


def name_match(a,b):
    a,b=a.lower().strip(),b.lower().strip()
    return a in b or b in a


def grade_bet(bet,results):
    bet=bet.copy()
    if bet.get("result") in ("WIN","LOSS","PUSH"): return bet
    bd,bt=bet.get("date",""),bet.get("type","")
    match=None
    for r in results:
        if r["date"]!=bd: continue
        hm=name_match(bet.get("home_team",""),r["home_team"]) or name_match(bet.get("home_team",""),r["home_abbr"])
        am=name_match(bet.get("away_team",""),r["away_team"]) or name_match(bet.get("away_team",""),r["away_abbr"])
        tm=name_match(bet.get("team",""),r["home_team"]) or name_match(bet.get("team",""),r["away_team"]) or name_match(bet.get("team",""),r["home_abbr"]) or name_match(bet.get("team",""),r["away_abbr"])
        if hm or am or tm: match=r; break
    if not match: return bet
    hs,as_,tot=match["home_score"],match["away_score"],match["total"]
    result=None
    if bt=="ML":
        side=bet.get("side","home")
        result="WIN" if (side=="home" and hs>as_) or (side=="away" and as_>hs) else "LOSS"
    elif bt=="SPREAD":
        side,spread=bet.get("side","home"),bet.get("spread",0) or 0
        if side=="home": cv=(hs+spread)>as_; pu=(hs+spread)==as_
        else: cv=(as_+spread)>hs; pu=(as_+spread)==hs
        result="PUSH" if pu else ("WIN" if cv else "LOSS")
    elif bt=="TOTAL":
        direction,tl=bet.get("direction","OVER"),bet.get("total_line",0) or 0
        if direction=="OVER": result="WIN" if tot>tl else ("PUSH" if tot==tl else "LOSS")
        else: result="WIN" if tot<tl else ("PUSH" if tot==tl else "LOSS")
    if not result: return bet
    odds=bet.get("odds",-110); bu=bet.get("units",1)
    if result=="WIN": uw=bu*(odds/100) if odds>0 else bu*(100/abs(odds))
    elif result=="LOSS": uw=-bu
    else: uw=0.0
    bet.update({"result":result,"units_won":round(uw,3),"home_score":hs,"away_score":as_,"graded_at":datetime.now().isoformat()})
    return bet


def update_bet_log(results):
    if not os.path.exists(BET_LOG_PATH):
        print("  No bet log"); return 0
    with open(BET_LOG_PATH) as f: bl=json.load(f)
    bets=bl.get("bets",[])
    graded=0
    for i,b in enumerate(bets):
        if b.get("result") in ("WIN","LOSS","PUSH"): continue
        gb=grade_bet(b,results)
        if gb.get("result") in ("WIN","LOSS","PUSH"): bets[i]=gb; graded+=1
    bl["bets"]=bets
    tu=sum(b.get("units_won",0) for b in bets if b.get("result") in ("WIN","LOSS","PUSH"))
    bl.update({"total_units":round(tu,3),"total_bets":len(bets),
        "graded_bets":sum(1 for b in bets if b.get("result") in ("WIN","LOSS","PUSH")),
        "wins":sum(1 for b in bets if b.get("result")=="WIN"),
        "losses":sum(1 for b in bets if b.get("result")=="LOSS"),
        "pushes":sum(1 for b in bets if b.get("result")=="PUSH"),
        "last_updated":datetime.now().isoformat()})
    with open(BET_LOG_PATH,"w") as f: json.dump(bl,f,indent=2)
    print(f"  Graded {graded} | P&L: {tu:+.2f}u")
    return graded


def save_for_training(results,target_date):
    if not results: return
    rows=[{"date":r["date"],"season":2026,"home_team":r["home_team"],"away_team":r["away_team"],
           "home_runs":r["home_score"],"away_runs":r["away_score"],"total_runs":r["total"],
           "home_win":r["home_win"],"run_diff":r["run_diff"],"home_field":1} for r in results]
    ndf=pd.DataFrame(rows)
    if os.path.exists(SEASON_PATH):
        ex=pd.read_parquet(SEASON_PATH)
        ndf=pd.concat([ex,ndf],ignore_index=True).drop_duplicates(subset=["date","home_team","away_team"])
    ndf.to_parquet(SEASON_PATH,index=False)
    print(f"  Saved {len(rows)} results (total:{len(ndf)})")


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--date",default=None)
    args,_=parser.parse_known_args()
    td=args.date or (date.today()-timedelta(days=1)).isoformat()
    print(f"Fetching results for {td}...")
    results=fetch_scores(td)
    if results:
        update_bet_log(results)
        save_for_training(results,td)
        for r in results:
            print(f"  {r['away_team']} {r['away_score']} @ {r['home_team']} {r['home_score']}")
    print("Done.")

if __name__=="__main__":
    main()