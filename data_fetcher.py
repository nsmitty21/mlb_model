"""
data_fetcher.py — MLB Data Fetcher
ESPN free API + pybaseball (Statcast/FanGraphs)
"""
import time, requests, numpy as np, pandas as pd
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

ABBREV_MAP = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI",
    "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
    "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
    "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
}
DISPLAY_TO_FULL = {v: k for k, v in ABBREV_MAP.items()}

def normalize_team(name):
    name = name.strip()
    for k in ABBREV_MAP:
        if k.lower() in name.lower() or name.lower() in k.lower(): return k
    return name

def _espn_schedule(team, year):
    abbrev = ABBREV_MAP.get(team)
    if not abbrev: return []
    url = f"{ESPN_BASE}/teams/{abbrev}/schedule"
    try:
        r = requests.get(url, params={"season":year,"seasontype":2}, timeout=10)
        r.raise_for_status(); data = r.json()
    except: return []
    logs = []
    for event in data.get("events",[]):
        try:
            comps = event.get("competitions",[{}])[0]
            if not comps.get("status",{}).get("type",{}).get("completed",False): continue
            competitors = comps.get("competitors",[])
            hc = next((c for c in competitors if c["homeAway"]=="home"),None)
            ac = next((c for c in competitors if c["homeAway"]=="away"),None)
            if not hc or not ac: continue
            home_name = normalize_team(hc["team"].get("displayName",""))
            away_name = normalize_team(ac["team"].get("displayName",""))
            hs = int(hc.get("score",0)); as_ = int(ac.get("score",0))
            is_home = (home_name == team)
            logs.append({"date":event.get("date","")[:10],"team":team,"opponent":away_name if is_home else home_name,
                         "is_home":int(is_home),"runs_scored":hs if is_home else as_,
                         "runs_allowed":as_ if is_home else hs,"win":int(hs>as_ if is_home else as_>hs),"year":year})
        except: continue
    return logs

def fetch_game_logs_historical(start_year=2020, end_year=2024):
    cache = DATA_DIR / f"game_logs_{start_year}_{end_year}.parquet"
    if cache.exists(): print(f"[data] Loading cached game logs"); return pd.read_parquet(cache)
    print("[data] Fetching team game logs via ESPN...")
    all_logs = []
    for year in range(start_year, end_year+1):
        print(f"  → {year}")
        for team in ABBREV_MAP:
            all_logs.extend(_espn_schedule(team, year)); time.sleep(0.2)
    df = pd.DataFrame(all_logs)
    if not df.empty: df["date"] = pd.to_datetime(df["date"]); df.to_parquet(cache); print(f"[data] Saved {len(df)} rows")
    return df

def fetch_pitcher_stats(start_year=2020, end_year=2024):
    cache = DATA_DIR / f"pitcher_stats_{start_year}_{end_year}.parquet"
    if cache.exists(): return pd.read_parquet(cache)
    print("[data] Fetching pitcher stats via pybaseball...")
    try:
        import pybaseball; pybaseball.cache.enable()
    except ImportError:
        print("[data] Install pybaseball: pip install pybaseball"); return pd.DataFrame()
    dfs = []
    for year in range(start_year, end_year+1):
        try:
            df = pybaseball.pitching_stats(year, qual=5); df["year"]=year; dfs.append(df); time.sleep(1)
            print(f"  → {year}: {len(df)} pitchers")
        except Exception as e: print(f"  [warn] {year}: {e}")
    if not dfs: return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True); combined.to_parquet(cache); return combined

def fetch_yesterday_results(target_date=None):
    if target_date is None:
        d = date.today()-timedelta(days=1); target_date = d.strftime("%Y%m%d")
    else:
        target_date = target_date.replace("-","")
    url = f"{ESPN_BASE}/scoreboard"
    try:
        r = requests.get(url, params={"dates":target_date,"limit":50}, timeout=10)
        r.raise_for_status(); data = r.json()
    except Exception as e:
        print(f"[data] ESPN error: {e}"); return pd.DataFrame()
    rows = []
    for event in data.get("events",[]):
        try:
            comps = event.get("competitions",[{}])[0]
            if not comps.get("status",{}).get("type",{}).get("completed",False): continue
            competitors = comps.get("competitors",[])
            hc = next((c for c in competitors if c["homeAway"]=="home"),None)
            ac = next((c for c in competitors if c["homeAway"]=="away"),None)
            if not hc or not ac: continue
            hn = normalize_team(hc["team"].get("displayName","")); an = normalize_team(ac["team"].get("displayName",""))
            hs = int(hc.get("score",0)); as_ = int(ac.get("score",0))
            rows.append({"game_date":f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}",
                         "home_team":hn,"away_team":an,"home_runs":hs,"away_runs":as_,
                         "total_runs":hs+as_,"home_win":int(hs>as_),
                         "home_covered_rl":int((hs-as_)>1.5),"espn_id":event.get("id","")})
        except: continue
    df = pd.DataFrame(rows); print(f"[data] {len(df)} results for {target_date}"); return df

def fetch_todays_starters(target_date=None):
    if target_date is None: target_date = date.today().strftime("%Y%m%d")
    else: target_date = target_date.replace("-","")
    url = f"{ESPN_BASE}/scoreboard"
    try:
        r = requests.get(url, params={"dates":target_date,"limit":50}, timeout=10)
        r.raise_for_status(); data = r.json()
    except Exception as e:
        print(f"[data] Starters error: {e}"); return pd.DataFrame()
    rows = []
    for event in data.get("events",[]):
        try:
            comps = event.get("competitions",[{}])[0]
            competitors = comps.get("competitors",[])
            hc = next((c for c in competitors if c["homeAway"]=="home"),None)
            ac = next((c for c in competitors if c["homeAway"]=="away"),None)
            if not hc or not ac: continue
            hn = normalize_team(hc["team"].get("displayName","")); an = normalize_team(ac["team"].get("displayName",""))
            home_sp = away_sp = ""
            for comp in competitors:
                for p in comp.get("probables",[]):
                    name = p.get("athlete",{}).get("fullName","")
                    if comp["homeAway"]=="home": home_sp=name
                    else: away_sp=name
            rows.append({"home_team":hn,"away_team":an,"home_sp":home_sp,"away_sp":away_sp,"commence":event.get("date",""),"espn_id":event.get("id","")})
        except: continue
    df = pd.DataFrame(rows); print(f"[data] {len(df)} games with starters"); return df
