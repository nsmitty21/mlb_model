"""
run_today.py — Daily Pick Generator
Usage: python run_today.py [--date YYYY-MM-DD]
       python run_today.py [--skip-injuries]

Architecture
------------
model.py     = prediction engine  (predict_game, load_models, edge thresholds)
run_today.py = daily orchestrator (fetch lines, build features, call model, write JSON)

Feature alignment
-----------------
train_model.py trained on 65 features (46 team-stat + 6 Pinnacle + 13 BvP).
This file builds those EXACT 65 features at inference time. The feature list
is read from meta.json so there is no drift if train_model.py changes.
The old features.py (35 park-factor/rolling-stat features) is no longer used.

Injury logic
------------
Injuries are a POST-MODEL probability adjustment, not a raw feature.
The model was never trained on injury data so feeding it as a feature is noise.
After the model scores a game we:
  1. Fetch today's injury report from the ESPN public API (no key needed).
  2. Compute an injury severity score: each missing starter is weighted by
     position (SP = 3pp, catcher = 0.8pp, bench bat = 0.3pp, etc.).
  3. Nudge home_win_prob by up to ±MAX_INJURY_ADJUSTMENT (3pp) before
     comparing to the book probability.
This is deliberately conservative — injuries are noisy and are already
partially priced in by the time you're seeing the line.
"""
import argparse, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import date, datetime

from model import load_models, predict_game, load_meta, _units

DATA_DIR   = Path("data")
PICKS_FILE = Path("data/live/picks_today.json")
PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_INJURY_ADJUSTMENT = 0.03   # 3 percentage points max nudge


# ─────────────────────────────────────────────────────────────────────────────
# Team name helpers
# ─────────────────────────────────────────────────────────────────────────────

FULL_NAME_TO_ABB = {
    "Arizona Diamondbacks": "ARI",  "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",     "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",          "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",       "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",      "Detroit Tigers": "DET",
    "Houston Astros": "HOU",        "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",         "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",       "New York Mets": "NYM",
    "New York Yankees": "NYY",      "Oakland Athletics": "OAK",
    "Athletics": "OAK",             "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",  "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",   "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",         "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}

BATTING_FEATS  = ["OPS", "wOBA", "AVG", "OBP", "SLG", "HR", "BB%", "K%", "R"]
PITCHING_FEATS = ["ERA", "FIP", "WHIP", "K/9", "BB/9", "xFIP"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_live_team_stats(year: int) -> tuple[dict, dict]:
    """
    Load team batting/pitching stats for inference.
    Returns (bat_lookup, pit_lookup) keyed by both (full_team_name, season)
    and (abbr, season) so build_inference_row() can look up either form.

    Strategy (season-start aware):
      1. Always load data/historical/team_{batting|pitching}.parquet first.
         This has full-season 2025 stats for every team and is indexed under
         (team, 2025) — used as the year-1 fallback automatically.
      2. Then load data/live/team_{batting|pitching}.parquet and overlay any
         teams that have played >= MIN_GAMES_CURRENT games this season.
         Teams below that threshold keep their prior-year stats.

    This means for the first ~2 weeks of the season every team uses 2025
    full-season stats rather than a 3-game sample, which the model was never
    trained on (it learned on full-season averages).
    """
    ABB_TO_FULL = {v: k for k, v in FULL_NAME_TO_ABB.items()}
    # pybaseball 2026 uses "ATH" for the Sacramento Athletics.
    # Neither ATH nor OAK auto-maps to "Athletics" (the name in
    # today_lines.json) without these explicit overrides.
    ABB_TO_FULL["ATH"] = "Athletics"
    ABB_TO_FULL["OAK"] = "Athletics"
    bat, pit = {}, {}
    # Batting stats need more games to stabilise (OPS, wOBA require ~30+ PA).
    # Pitching ERA/FIP are meaningful after just 4–5 starts (~25–30 IP), so
    # use a lower threshold to pick up early-season changes (new team, injury
    # return, velocity change) before the market fully prices them in.
    MIN_GAMES_BATTING  = 15
    MIN_GAMES_PITCHING = 5

    for filename, lookup, label in [
        ("team_batting.parquet",  bat, "batting"),
        ("team_pitching.parquet", pit, "pitching"),
    ]:
        min_games = MIN_GAMES_BATTING if label == "batting" else MIN_GAMES_PITCHING
        # ── Step 1: load historical (prior seasons) as fallback baseline ──
        hist_path = DATA_DIR / "historical" / filename
        hist_loaded = 0
        if hist_path.exists():
            df_hist = pd.read_parquet(hist_path)
            if "team" in df_hist.columns:
                # If multiple seasons present, index each under its own year
                # so the year-1 fallback in build_inference_row() works.
                if "season" in df_hist.columns:
                    for _, row in df_hist.iterrows():
                        abbr      = str(row["team"]).strip().upper()
                        full_name = ABB_TO_FULL.get(abbr, abbr)
                        yr        = int(row["season"])
                        lookup[(full_name, yr)] = row.to_dict()
                        lookup[(abbr,      yr)] = row.to_dict()
                        hist_loaded += 1
                else:
                    for _, row in df_hist.iterrows():
                        abbr      = str(row["team"]).strip().upper()
                        full_name = ABB_TO_FULL.get(abbr, abbr)
                        lookup[(full_name, year - 1)] = row.to_dict()
                        lookup[(abbr,      year - 1)] = row.to_dict()
                        hist_loaded += 1

        # ── Step 2: overlay live current-season stats for teams with enough games ──
        live_path = DATA_DIR / "live" / filename
        live_loaded = 0
        live_skipped = 0
        if live_path.exists():
            df_live = pd.read_parquet(live_path)
            if "team" in df_live.columns:
                if "season" in df_live.columns:
                    latest = int(df_live["season"].max())
                    df_live = df_live[df_live["season"] == latest].copy()

                for _, row in df_live.iterrows():
                    abbr         = str(row["team"]).strip().upper()
                    full_name    = ABB_TO_FULL.get(abbr, abbr)
                    games_played = float(row.get("G", 0) or 0)

                    if games_played < min_games:
                        live_skipped += 1
                        continue  # too early — keep prior-year fallback

                    lookup[(full_name, year)] = row.to_dict()
                    lookup[(abbr,      year)] = row.to_dict()
                    live_loaded += 1

        # ── Reporting ─────────────────────────────────────────────────────
        if live_loaded > 0:
            print(f"  {label.capitalize()} stats: {live_loaded} teams from "
                  f"live/{filename} (season {year})"
                  + (f" + {live_skipped} teams using prior-year fallback"
                     if live_skipped else ""))
        elif hist_loaded > 0:
            prior = year - 1
            print(f"  {label.capitalize()} stats: 0 teams with {min_games}+ "
                  f"games in {year} — using {prior} full-season stats for all teams")
        else:
            print(f"  [WARN] {filename} not found in data/live/ or data/historical/ — "
                  "run fetch_historical.py")

    return bat, pit


def _load_bvp_data() -> pd.DataFrame | None:
    # Check all known locations for the BvP file.
    # The odds folder contains batter_vs_pitcher_career.csv (36MB per screenshot).
    candidates = [
        DATA_DIR / "historical" / "odds" / "batter_vs_pitcher_career.csv",
        DATA_DIR / "historical" / "batter_vs_pitcher_career.csv",
        DATA_DIR / "historical" / "odds" / "bvp_career.csv",
    ]
    bvp_path = next((c for c in candidates if c.exists()), None)
    if bvp_path is None:
        print("  BvP data: not found (expected batter_vs_pitcher_career.csv in "
              "data/historical/odds/)")
        return None
    df = pd.read_csv(bvp_path, low_memory=False)
    required = {"batter", "pitcher", "PA", "AB", "H", "HR", "BB", "K"}
    if required - set(df.columns):
        print(f"  BvP data: missing columns {required - set(df.columns)} — skipping")
        return None
    print(f"  BvP data: {len(df):,} rows from {bvp_path.name}")
    return df


def _load_lineup_lookup() -> dict:
    """Load game_lineups.parquet → dict keyed by (game_date, home_abbr).

    fetch_historical.py saves this to data/historical/game_lineups.parquet.
    Keys are (YYYY-MM-DD string, 3-letter home abbr).
    """
    candidates = [
        DATA_DIR / "historical" / "game_lineups.parquet",
        DATA_DIR / "historical" / "odds" / "game_lineups.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        # game_lineups.parquet is built from plate_appearances_raw_*.csv files
        # by fetch_historical.py step 5d. Run fetch_historical.py to populate it.
        return {}
    df = pd.read_parquet(path)
    required = {"game_date", "home_abbr", "home_batters", "away_batters",
                "home_sp", "away_sp"}
    if required - set(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        # Normalise game_date: Timestamp → YYYY-MM-DD string
        gd = row["game_date"]
        try:
            gd = pd.Timestamp(gd).date().isoformat()
        except Exception:
            gd = str(gd)[:10]
        key = (gd, str(row["home_abbr"]))
        out[key] = {
            "home_batters": [b for b in str(row["home_batters"]).split("|") if b],
            "away_batters": [b for b in str(row["away_batters"]).split("|") if b],
            "home_sp":      str(row["home_sp"]),
            "away_sp":      str(row["away_sp"]),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BvP computation (mirrors fetch_historical.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return str(s).replace("\xa0", " ").replace("\u00a0", " ").strip().lower()


def _compute_bvp_features(pitcher_name: str, lineup: list,
                           bvp_df: pd.DataFrame | None,
                           min_pa: int = 5) -> dict:
    empty = {"bvp_avg": np.nan, "bvp_ops": np.nan, "bvp_k_rate": np.nan,
             "bvp_bb_rate": np.nan, "bvp_hr_rate": np.nan}

    if bvp_df is None or not lineup or not pitcher_name:
        return empty

    sub = bvp_df[bvp_df["pitcher"].apply(_norm) == _norm(pitcher_name)]
    if sub.empty:
        return empty

    sub_idx = sub.set_index(sub["batter"].apply(_norm))
    totals  = {"PA": 0, "AB": 0, "H": 0, "HR": 0, "BB": 0,
               "K": 0, "SF": 0, "HBP": 0, "2B": 0, "3B": 0}
    found   = 0

    for batter in lineup:
        key = _norm(batter)
        if key not in sub_idx.index:
            continue
        row = sub_idx.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        pa = int(row.get("PA", 0) or 0)
        if pa < min_pa:
            continue
        found += 1
        for col in totals:
            totals[col] += float(row.get(col, 0) or 0)

    if found == 0 or totals["PA"] == 0:
        return empty

    pa  = totals["PA"]
    ab  = totals["AB"] if totals["AB"] > 0 else pa
    h, bb, hbp, sf, hr = (totals["H"], totals["BB"], totals["HBP"],
                           totals["SF"], totals["HR"])
    doubles, triples = totals["2B"], totals["3B"]
    # BUG FIX: previously singles + 4*hr — doubles and triples were excluded,
    # causing gap hitters to have the same SLG as singles hitters in BvP data.
    singles = max(h - doubles - triples - hr, 0)
    obp_den = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_den if obp_den > 0 else np.nan
    slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / ab if ab > 0 else np.nan
    ops = round(obp + slg, 3) if (pd.notna(obp) and pd.notna(slg)) else np.nan

    return {
        "bvp_avg":     round(h / ab, 3) if ab > 0 else np.nan,
        "bvp_ops":     ops,
        "bvp_k_rate":  round(totals["K"]  / pa, 3),
        "bvp_bb_rate": round(totals["BB"] / pa, 3),
        "bvp_hr_rate": round(totals["HR"] / pa, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder — must match train_model.py FEATURE_COLS exactly
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_row(home_team: str, away_team: str,
                        bat_lookup: dict, pit_lookup: dict,
                        game_date: str, lineup_lookup: dict,
                        bvp_df: pd.DataFrame | None,
                        home_sp: str = "", away_sp: str = "",
                        pinnacle: dict | None = None) -> dict:
    """
    Build a 65-feature row matching train_model.py's FEATURE_COLS:
      46 team-stat  (home_bat_*, away_bat_*, diff_bat_*,
                     home_pit_*, away_pit_*, diff_pit_*, home_field)
       6 Pinnacle
      13 BvP
    """
    row  = {}
    year = int(game_date[:4]) if game_date else date.today().year

    # ── Team batting / pitching (FanGraphs season stats) ──────────────────
    for pfx, team in [("home_bat_", home_team), ("away_bat_", away_team)]:
        stats = bat_lookup.get((team, year),
                bat_lookup.get((team, year - 1), {}))
        for f in BATTING_FEATS:
            row[f"{pfx}{f}"] = stats.get(f, np.nan) if stats else np.nan

    for pfx, team in [("home_pit_", home_team), ("away_pit_", away_team)]:
        stats = pit_lookup.get((team, year),
                pit_lookup.get((team, year - 1), {}))
        for f in PITCHING_FEATS:
            row[f"{pfx}{f}"] = stats.get(f, np.nan) if stats else np.nan

    row["home_field"] = 1

    for f in BATTING_FEATS:
        hv, av = row.get(f"home_bat_{f}", np.nan), row.get(f"away_bat_{f}", np.nan)
        row[f"diff_bat_{f}"] = (hv - av) if pd.notna(hv) and pd.notna(av) else np.nan

    for f in PITCHING_FEATS:
        hv, av = row.get(f"home_pit_{f}", np.nan), row.get(f"away_pit_{f}", np.nan)
        if pd.notna(hv) and pd.notna(av):
            row[f"diff_pit_{f}"] = (av - hv) if f in ("ERA","FIP","WHIP","BB/9") \
                                   else (hv - av)
        else:
            row[f"diff_pit_{f}"] = np.nan

    # ── Pinnacle features ─────────────────────────────────────────────────
    pi = pinnacle or {}
    row["pi_close_no_vig_home"] = pi.get("pi_close_no_vig_home") or 0.5
    row["pi_open_implied_home"] = pi.get("pi_open_implied_home") or 0.5
    row["pi_ml_movement"]       = pi.get("pi_ml_movement")       or 0.0
    row["pi_spread_movement"]   = pi.get("pi_spread_movement")   or 0.0
    row["pi_total_movement"]    = pi.get("pi_total_movement")    or 0.0
    row["pi_close_total"]       = pi.get("pi_close_total")       or 8.5

    # ── BvP features ──────────────────────────────────────────────────────
    home_abbr   = FULL_NAME_TO_ABB.get(home_team, home_team[:3].upper())
    game_lineup = lineup_lookup.get((game_date, home_abbr))

    if game_lineup:
        home_batters = game_lineup["home_batters"]
        away_batters = game_lineup["away_batters"]
        home_sp = game_lineup["home_sp"] or home_sp
        away_sp = game_lineup["away_sp"] or away_sp
    else:
        home_batters = []
        away_batters = []

    h_bvp = _compute_bvp_features(away_sp, home_batters, bvp_df)
    a_bvp = _compute_bvp_features(home_sp, away_batters, bvp_df)

    row["home_bvp_avg"]     = h_bvp["bvp_avg"]
    row["home_bvp_ops"]     = h_bvp["bvp_ops"]
    row["home_bvp_k_rate"]  = h_bvp["bvp_k_rate"]
    row["home_bvp_bb_rate"] = h_bvp["bvp_bb_rate"]
    row["home_bvp_hr_rate"] = h_bvp["bvp_hr_rate"]
    row["away_bvp_avg"]     = a_bvp["bvp_avg"]
    row["away_bvp_ops"]     = a_bvp["bvp_ops"]
    row["away_bvp_k_rate"]  = a_bvp["bvp_k_rate"]
    row["away_bvp_bb_rate"] = a_bvp["bvp_bb_rate"]
    row["away_bvp_hr_rate"] = a_bvp["bvp_hr_rate"]

    def _diff(a, b):
        return (a - b) if pd.notna(a) and pd.notna(b) else np.nan

    row["bvp_avg_diff"]    = _diff(h_bvp["bvp_avg"],    a_bvp["bvp_avg"])
    row["bvp_ops_diff"]    = _diff(h_bvp["bvp_ops"],    a_bvp["bvp_ops"])
    row["bvp_k_rate_diff"] = _diff(h_bvp["bvp_k_rate"], a_bvp["bvp_k_rate"])

    # Fill any remaining BvP NaNs with neutral values.
    # During training, NaN BvP features were filled with their column medians
    # (via build_features in train_model.py). At inference, predict_game()
    # would fill them with -999 — far outside the training distribution,
    # which corrupts every prediction. We fill with known neutral constants
    # so the model sees values consistent with what it learned on.
    BVP_NEUTRAL = {
        "home_bvp_avg":     0.250,   # league-average batting average
        "home_bvp_ops":     0.720,   # league-average OPS
        "home_bvp_k_rate":  0.225,   # league-average strikeout rate
        "home_bvp_bb_rate": 0.085,   # league-average walk rate
        "home_bvp_hr_rate": 0.032,   # league-average HR rate
        "away_bvp_avg":     0.250,
        "away_bvp_ops":     0.720,
        "away_bvp_k_rate":  0.225,
        "away_bvp_bb_rate": 0.085,
        "away_bvp_hr_rate": 0.032,
        "bvp_avg_diff":     0.0,     # neutral: no advantage either way
        "bvp_ops_diff":     0.0,
        "bvp_k_rate_diff":  0.0,
    }
    for col, neutral in BVP_NEUTRAL.items():
        if col in row and (row[col] is None or (isinstance(row[col], float) and np.isnan(row[col]))):
            row[col] = neutral

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Injury adjustment
# ─────────────────────────────────────────────────────────────────────────────

_POSITION_WEIGHTS = {
    "SP": 0.030,   # ace scratch = full 3pp
    "RP": 0.004,
    "C":  0.008,   "SS": 0.008,
    "2B": 0.006,   "3B": 0.006,
    "1B": 0.005,   "LF": 0.005,
    "CF": 0.005,   "RF": 0.005,
    "DH": 0.004,
}
_OUT_STATUSES = {
    "Out", "Doubtful", "IR", "IL",
    "10-Day IL", "15-Day IL", "60-Day IL", "Suspended",
}
# Questionable players have ~50% chance of missing the game — weight at 40%
# of their position value so the nudge reflects real uncertainty rather than
# ignoring them entirely (as was the case before this fix).
_QUESTIONABLE_STATUSES = {"Questionable"}
_QUESTIONABLE_WEIGHT   = 0.40


def _fetch_injury_report(team_abbr: str) -> list[dict]:
    """ESPN public injuries endpoint — no API key required."""
    import requests
    try:
        url = (f"https://site.api.espn.com/apis/site/v2/sports/baseball/"
               f"mlb/teams/{team_abbr}/injuries")
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        out = []
        for item in r.json().get("injuries", []):
            athlete = item.get("athlete", {})
            out.append({
                "player":   athlete.get("displayName", ""),
                "status":   item.get("status", ""),
                "position": athlete.get("position", {}).get("abbreviation", ""),
            })
        return out
    except Exception:
        return []


def compute_injury_adjustment(home_abbr: str, away_abbr: str) -> float:
    """
    Returns a signed float in [-MAX_INJURY_ADJUSTMENT, +MAX_INJURY_ADJUSTMENT].
    Positive  = away team more injured → nudge home win prob up.
    Negative  = home team more injured → nudge home win prob down.
    """
    def _severity(injuries: list[dict]) -> float:
        total = 0.0
        for inj in injuries:
            pos_weight = _POSITION_WEIGHTS.get(inj["position"], 0.003)
            if inj["status"] in _OUT_STATUSES:
                total += pos_weight
            elif inj["status"] in _QUESTIONABLE_STATUSES:
                # ~50% game-time decision → apply fractional weight
                total += pos_weight * _QUESTIONABLE_WEIGHT
        return min(total, MAX_INJURY_ADJUSTMENT)

    home_hit = _severity(_fetch_injury_report(home_abbr))
    away_hit = _severity(_fetch_injury_report(away_abbr))
    return float(np.clip(away_hit - home_hit,
                         -MAX_INJURY_ADJUSTMENT, MAX_INJURY_ADJUSTMENT))


# ─────────────────────────────────────────────────────────────────────────────
# Lines loading
# ─────────────────────────────────────────────────────────────────────────────

def fetch_todays_lines(target_date: str) -> pd.DataFrame:
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
        fd = g.get("books", {}).get("fanduel",    {})
        dk = g.get("books", {}).get("draftkings",  {})
        pi = g.get("books", {}).get("pinnacle",    {})

        rows.append({
            "game_id":               g["game_id"],
            "home_team":             ht,
            "away_team":             at,
            "commence":              g.get("commence", ""),
            "fd_ml_home":            fd.get("ml_home"),
            "fd_ml_away":            fd.get("ml_away"),
            "fd_spread_home":        fd.get("spread_home"),
            "fd_spread_juice_home":  fd.get("spread_juice_home"),
            "fd_spread_juice_away":  fd.get("spread_juice_away"),
            "fd_total":              fd.get("total"),
            "fd_total_over_juice":   fd.get("total_over_juice"),
            "fd_total_under_juice":  fd.get("total_under_juice"),
            "dk_ml_home":            dk.get("ml_home"),
            "dk_ml_away":            dk.get("ml_away"),
            "dk_spread_home":        dk.get("spread_home"),
            "dk_spread_juice_home":  dk.get("spread_juice_home"),
            "dk_spread_juice_away":  dk.get("spread_juice_away"),
            "dk_total":              dk.get("total"),
            "dk_total_over_juice":   dk.get("total_over_juice"),
            "dk_total_under_juice":  dk.get("total_under_juice"),
            "_pi_close_no_vig_home": pi.get("pi_close_no_vig_home"),
            "_pi_open_implied_home": pi.get("pi_open_implied_home"),
            "_pi_ml_movement":       pi.get("pi_ml_movement",     0.0),
            "_pi_spread_movement":   pi.get("pi_spread_movement", 0.0),
            "_pi_total_movement":    pi.get("pi_total_movement",  0.0),
            "_pi_close_total":       pi.get("pi_close_total"),
        })

    df = pd.DataFrame(rows)
    print(f"  Lines loaded: {len(df)} games from today_lines.json")

    low = df["fd_total"].dropna()
    low = low[low < 5.0]
    if len(low):
        print(f"  [WARN] {len(low)} game(s) with total < 5.0: {low.tolist()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data quality gate
# ─────────────────────────────────────────────────────────────────────────────

def _has_usable_data(row: dict, ht: str, at: str,
                     feature_cols: list) -> bool:
    non_null = sum(
        1 for c in feature_cols
        if c in row and row[c] is not None
        and not (isinstance(row[c], float) and np.isnan(row[c]))
    )
    frac = non_null / len(feature_cols) if feature_cols else 0.0
    if frac < 0.50:
        print(f"  [SKIP] {at} @ {ht} — only {frac:.0%} of features populated "
              f"({non_null}/{len(feature_cols)})")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Output format normalizer
# ─────────────────────────────────────────────────────────────────────────────

def _picks_to_game_format(all_picks: list, target_date: str) -> list:
    from collections import defaultdict
    game_map  = defaultdict(list)
    game_meta = {}

    for p in all_picks:
        key = f"{p.get('home_team','')}|{p.get('away_team','')}"
        game_meta[key] = {
            "home_team": p.get("home_team", ""),
            "away_team": p.get("away_team", ""),
            "home_sp":   p.get("home_sp",   ""),
            "away_sp":   p.get("away_sp",   ""),
            "commence":  p.get("commence",  ""),
            "date":      target_date,
            "game_id":   key.replace("|", "_").replace(" ", "_"),
        }

        market = str(p.get("market", "ML")).upper()
        side   = str(p.get("side",   "home")).lower()

        norm = {
            "type":       market,
            "side":       side,
            "team":       p.get("team", ""),
            "opponent":   p.get("away_team", "") if side == "home"
                          else p.get("home_team", ""),
            "odds":       p.get("odds"),
            "book":       p.get("book", ""),
            "model_prob": round(float(p.get("model_prob", 0)) * 100, 1),
            "book_prob":  round(float(p.get("book_prob",  0)) * 100, 1),
            "edge":       round(float(p.get("edge",       0)) * 100, 2),
            "units":      p.get("units", 1),
        }

        if market in ("SPREAD", "RL"):
            norm["type"]   = "SPREAD"
            norm["spread"] = p.get("spread_line")
        elif market == "TOTAL":
            norm["direction"]  = side.upper()
            norm["home_team"]  = p.get("home_team", "")
            norm["away_team"]  = p.get("away_team", "")
            norm["total_line"] = p.get("total_line")
            norm["pred_total"] = None

        if p.get("injury_adj"):
            norm["injury_adj_pp"] = round(p["injury_adj"] * 100, 1)

        game_map[key].append(norm)

    return [
        {
            "game_id":   game_meta[k]["game_id"],
            "date":      game_meta[k]["date"],
            "commence":  game_meta[k]["commence"],
            "home_team": game_meta[k]["home_team"],
            "away_team": game_meta[k]["away_team"],
            "home_sp":   game_meta[k]["home_sp"],
            "away_sp":   game_meta[k]["away_sp"],
            "picks":     sorted(v, key=lambda x: x["units"], reverse=True),
        }
        for k, v in game_map.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(target_date: str | None = None, skip_injuries: bool = False):
    if target_date is None:
        target_date = date.today().isoformat()

    meta         = load_meta()
    feature_cols = meta.get("feature_cols", [])

    print(f"\n{'='*60}")
    print(f"  MLB Edge — Picks: {target_date}")
    print(f"  Features: {len(feature_cols)}  (from meta.json)")
    print(f"  Injury adjustment: {'OFF' if skip_injuries else 'ON (±3pp max)'}")
    print(f"{'='*60}\n")

    if not feature_cols:
        print("[ERROR] meta.json missing feature_cols — re-run train_model.py.")
        return

    # Lines
    games = fetch_todays_lines(target_date)
    if games.empty:
        return

    # Starters
    try:
        from data_fetcher import fetch_todays_starters
        starters_df = fetch_todays_starters(target_date)
        if not starters_df.empty:
            games = games.merge(
                starters_df[["home_team", "away_team", "home_sp", "away_sp"]],
                on=["home_team", "away_team"], how="left"
            )
    except Exception as e:
        print(f"  [WARN] Starters fetch failed: {e}")

    for col in ("home_sp", "away_sp"):
        if col not in games.columns:
            games[col] = ""
        games[col] = games[col].fillna("")

    # Models
    try:
        ml_model, spread_model, total_model = load_models()
        print(f"  Models loaded (trained on {meta.get('n_games','?')} games).")
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return

    # Stats
    year = int(target_date[:4])
    bat_lookup, pit_lookup = _load_live_team_stats(year)
    print(f"  Live team stats: {len(bat_lookup)} batting / {len(pit_lookup)} pitching")

    # ── Lookup diagnostic — verify team name resolution ───────────────────
    # Checks that full team names from today_lines.json resolve to real stats.
    # If OPS shows MISSING the ABB_TO_FULL map has a gap for that team.
    print()
    _diag_teams  = sorted(set(games["home_team"].tolist() + games["away_team"].tolist()))
    _any_missing = False
    _missing_teams = []
    _fallback_teams = []
    for _t in _diag_teams:
        _brow_cur  = bat_lookup.get((_t, year), {})
        _brow_prev = bat_lookup.get((_t, year - 1), {})
        _prow_cur  = pit_lookup.get((_t, year), {})
        _prow_prev = pit_lookup.get((_t, year - 1), {})

        _using_cur  = bool(_brow_cur)
        _using_prev = bool(_brow_prev) and not _using_cur
        _brow = _brow_cur or _brow_prev
        _prow = _prow_cur or pit_lookup.get((_t, year - 1), {})

        _ops = _brow.get("OPS", None)
        _era = _prow.get("ERA", None)

        if not _brow and not _prow:
            _any_missing = True
            _missing_teams.append(_t)
        elif _using_prev:
            _fallback_teams.append(_t)

    if _missing_teams:
        print(f"  [WARN] No stats found for: {', '.join(_missing_teams)}")
        print(f"         These games will be hard-skipped. Run fetch_historical.py to populate.")
    if _fallback_teams:
        print(f"  [INFO] Using {year-1} full-season stats for {len(_fallback_teams)} teams "
              f"(< 15 games in {year}): {', '.join(_fallback_teams)}")
    if not _any_missing and not _fallback_teams:
        print(f"  [OK] All {len(_diag_teams)} teams have {year} stats.")
    # Build a hard-exclusion set so no pick is ever generated for a game
    # where either team has completely missing stats. Previously _has_usable_data()
    # could pass (50% threshold) even when the bet team itself had zero data,
    # because BvP / Pinnacle columns filled the non-null count above 50%.
    _no_stats_teams: set[str] = set(_missing_teams)
    print()

    bvp_df        = _load_bvp_data()
    lineup_lookup = _load_lineup_lookup()
    print(f"  BvP data: {len(bvp_df):,} rows" if bvp_df is not None
          else "  BvP data: not found")
    print(f"  Lineup lookup: {len(lineup_lookup):,} historical games")

    # Generate picks
    all_picks, skipped = [], 0

    for _, game in games.iterrows():
        ht  = str(game.get("home_team", ""))
        at  = str(game.get("away_team", ""))
        hsp = str(game.get("home_sp",   ""))
        asp = str(game.get("away_sp",   ""))

        # Hard-skip: if either team has no stats at all, skip before building
        # features. _has_usable_data() alone is not sufficient — BvP / Pinnacle
        # neutral fills can push non-null fraction above 50% even when all
        # batting and pitching features for the missing team are NaN.
        if ht in _no_stats_teams or at in _no_stats_teams:
            print(f"  [SKIP] {at} @ {ht} — no stats for "
                  f"{', '.join(t for t in (ht, at) if t in _no_stats_teams)}")
            skipped += 1
            continue

        pinnacle_ctx = {
            "pi_close_no_vig_home": game.get("_pi_close_no_vig_home"),
            "pi_open_implied_home": game.get("_pi_open_implied_home"),
            "pi_ml_movement":       game.get("_pi_ml_movement",     0.0),
            "pi_spread_movement":   game.get("_pi_spread_movement", 0.0),
            "pi_total_movement":    game.get("_pi_total_movement",  0.0),
            "pi_close_total":       game.get("_pi_close_total"),
        }

        feat_row = build_inference_row(
            ht, at, bat_lookup, pit_lookup,
            target_date, lineup_lookup, bvp_df,
            home_sp=hsp, away_sp=asp, pinnacle=pinnacle_ctx,
        )

        if not _has_usable_data(feat_row, ht, at, feature_cols):
            skipped += 1
            continue

        # Injury nudge (post-model, not a feature)
        injury_adj = 0.0
        if not skip_injuries:
            ha = FULL_NAME_TO_ABB.get(ht, ht[:3].upper())
            aa = FULL_NAME_TO_ABB.get(at, at[:3].upper())
            injury_adj = compute_injury_adjustment(ha, aa)
            if abs(injury_adj) >= 0.005:
                arrow = "↑ home" if injury_adj > 0 else "↓ home"
                print(f"  [INJ] {at} @ {ht}: {injury_adj*100:+.1f}pp {arrow}")

        raw_picks = predict_game(
            feat_row, feature_cols,
            ml_model, spread_model, total_model,
            injury_adj=injury_adj,
            fd_ml_home=game.get("fd_ml_home"),         fd_ml_away=game.get("fd_ml_away"),
            dk_ml_home=game.get("dk_ml_home"),         dk_ml_away=game.get("dk_ml_away"),
            fd_spread_home=game.get("fd_spread_home"),
            fd_spread_juice_home=game.get("fd_spread_juice_home"),
            fd_spread_juice_away=game.get("fd_spread_juice_away"),
            dk_spread_home=game.get("dk_spread_home"),
            dk_spread_juice_home=game.get("dk_spread_juice_home"),
            dk_spread_juice_away=game.get("dk_spread_juice_away"),
            fd_total=game.get("fd_total"),
            fd_total_over_juice=game.get("fd_total_over_juice"),
            fd_total_under_juice=game.get("fd_total_under_juice"),
            dk_total=game.get("dk_total"),
            dk_total_over_juice=game.get("dk_total_over_juice"),
            dk_total_under_juice=game.get("dk_total_under_juice"),
            home_team=ht, away_team=at,
            home_sp=hsp, away_sp=asp,
            commence=str(game.get("commence", "")),
        )

        for p in raw_picks:
            p["home_team"]  = ht
            p["away_team"]  = at
            p["injury_adj"] = injury_adj
            all_picks.append(p)

    if skipped:
        print(f"\n  [WARN] {skipped} game(s) skipped — insufficient features.")

    all_picks.sort(key=lambda p: (-p["units"], -abs(p["edge"])))

    # ── Conflict resolution ───────────────────────────────────────────────────
    # Remove picks that bet opposing sides of the same game, which creates a
    # scenario where only a 1-run win by the specific team produces any profit,
    # and both bets can lose simultaneously.
    #
    # Conflicts we resolve (keep the higher-edge pick, drop the other):
    #   • Both sides of ML in the same game (home ML + away ML)
    #   • ML on Team A + RL -1.5 on Team B (opposing teams, same game)
    #     → only profitable if A wins by exactly 1; 1-run loss means both lose
    #
    # Safe combinations we leave alone:
    #   • Same team on ML + RL (correlated upside, not conflicting)
    #   • RL -1.5 + RL +1.5 on opposing teams (non-conflicting outcomes)
    #   • Any combination involving Totals (independent of side)
    def _resolve_conflicts(picks: list) -> list:
        from collections import defaultdict

        # Group picks by game key (home_team|away_team)
        by_game: dict[str, list] = defaultdict(list)
        for p in picks:
            key = f"{p.get('home_team','')}|{p.get('away_team','')}"
            by_game[key].append(p)

        keep = []
        for game_key, game_picks in by_game.items():
            # Separate into market buckets
            ml_picks    = [p for p in game_picks if p["market"] == "ML"]
            rl_picks    = [p for p in game_picks if p["market"] == "RL"]
            total_picks = [p for p in game_picks if p["market"] == "Total"]

            # Totals never conflict with anything — always keep
            keep.extend(total_picks)

            if not ml_picks and not rl_picks:
                continue

            # Identify which team each pick is on (home or away)
            home_team = game_key.split("|")[0]
            away_team = game_key.split("|")[1]

            def _team_side(p):
                """Return 'home' or 'away' based on which team the pick is on."""
                team = str(p.get("team", ""))
                side = str(p.get("side", ""))
                if side in ("home", "away"):
                    return side
                # Fall back to team name matching
                if home_team and home_team in team:
                    return "home"
                if away_team and away_team in team:
                    return "away"
                return side  # use as-is if can't determine

            # ── Check for ML vs ML conflict (both sides) ──────────────────────
            ml_home = [p for p in ml_picks if _team_side(p) == "home"]
            ml_away = [p for p in ml_picks if _team_side(p) == "away"]

            if ml_home and ml_away:
                # Both sides of ML — keep the higher-edge pick
                all_ml = ml_home + ml_away
                best_ml = max(all_ml, key=lambda p: abs(p["edge"]))
                dropped = [p for p in all_ml if p is not best_ml]
                for d in dropped:
                    print(f"  [CONFLICT] Dropping {d['market']} {d.get('team','')} "
                          f"(edge {d['edge']*100:.1f}%) — conflicts with "
                          f"{best_ml['market']} {best_ml.get('team','')} "
                          f"(edge {best_ml['edge']*100:.1f}%)")
                ml_picks = [best_ml]
                ml_home  = [p for p in ml_picks if _team_side(p) == "home"]
                ml_away  = [p for p in ml_picks if _team_side(p) == "away"]

            # ── Check for ML vs opposing RL -1.5 conflict ─────────────────────
            # In baseball the run line is always ±1.5.
            # A -1.5 pick means that team must WIN by 2+.
            # Conflict: ML on Team A + RL -1.5 on Team B (opposing teams)
            #   → only profitable if A wins by exactly 1; a 1-run loss = both lose.
            # Safe: same team on ML + RL -1.5 (correlated upside).

            def _spread_line(p) -> float:
                """Return the numeric spread line, defaulting to +1.5 if unknown."""
                raw = p.get("spread_line")
                if raw is not None:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        pass
                # Fall back: parse last token of team field e.g. "Texas Rangers -1.5"
                parts = str(p.get("team", "")).rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
                return 1.5  # default +1.5 (won't false-positive as conflict)

            rl_home_minus = [p for p in rl_picks
                             if _team_side(p) == "home" and _spread_line(p) < 0]
            rl_away_minus = [p for p in rl_picks
                             if _team_side(p) == "away" and _spread_line(p) < 0]

            conflicts_to_resolve = []
            # ML on home + RL -1.5 on away = conflict
            for ml in ml_home:
                for rl in rl_away_minus:
                    conflicts_to_resolve.append((ml, rl))
            # ML on away + RL -1.5 on home = conflict
            for ml in ml_away:
                for rl in rl_home_minus:
                    conflicts_to_resolve.append((ml, rl))

            resolved_drops = set()
            for ml_p, rl_p in conflicts_to_resolve:
                if id(ml_p) in resolved_drops or id(rl_p) in resolved_drops:
                    continue
                # Keep higher edge, drop the other
                if abs(ml_p["edge"]) >= abs(rl_p["edge"]):
                    resolved_drops.add(id(rl_p))
                    print(f"  [CONFLICT] Dropping RL {rl_p.get('team','')} "
                          f"(edge {rl_p['edge']*100:.1f}%) — opposes "
                          f"ML {ml_p.get('team','')} "
                          f"(edge {ml_p['edge']*100:.1f}%)")
                else:
                    resolved_drops.add(id(ml_p))
                    print(f"  [CONFLICT] Dropping ML {ml_p.get('team','')} "
                          f"(edge {ml_p['edge']*100:.1f}%) — opposes "
                          f"RL {rl_p.get('team','')} "
                          f"(edge {rl_p['edge']*100:.1f}%)")

            kept_ml = [p for p in ml_picks if id(p) not in resolved_drops]
            kept_rl = [p for p in rl_picks if id(p) not in resolved_drops]
            keep.extend(kept_ml)
            keep.extend(kept_rl)

        return keep

    all_picks = _resolve_conflicts(all_picks)
    # Re-sort after conflict removal
    all_picks.sort(key=lambda p: (-p["units"], -abs(p["edge"])))

    ml_picks = [p for p in all_picks if p["market"] == "ML"]
    away_ml  = [p for p in ml_picks  if p["side"]   == "away"]
    three_u  = [p for p in all_picks if p["units"]  == 3]

    print(f"\n{'='*60}")
    if not all_picks:
        print("  No picks meet edge threshold today.")
    else:
        print(f"  {len(all_picks)} PICKS  "
              f"(3U:{len(three_u)}  ML:{len(ml_picks)}  away_ML:{len(away_ml)})")
        if ml_picks and len(away_ml) == len(ml_picks):
            print("  [WARN] Every ML pick is away — verify live team stats.")
        if len(three_u) > len(all_picks) * 0.5 and len(all_picks) >= 4:
            print("  [WARN] >50% of picks are 3U — review before betting.")
        print()
        for p in all_picks:
            inj = (f"  INJ:{p['injury_adj']*100:+.1f}pp"
                   if p.get("injury_adj") else "")
            print(f"  {'⭐'*p['units']} {p['units']}u [{p['market']}] "
                  f"{p['team']} ({p['odds']:+d} @ {p['book'].upper()})  "
                  f"Edge:{p['edge']*100:.1f}%  "
                  f"Model:{p['model_prob']*100:.1f}%  "
                  f"Book:{p['book_prob']*100:.1f}%{inj}")
    print(f"{'='*60}")

    games_out = _picks_to_game_format(all_picks, target_date)
    out = {
        "generated_at": datetime.now().isoformat(),
        "date":         target_date,
        "total_picks":  len(all_picks),
        "games":        games_out,
    }
    with open(PICKS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(all_picks)} picks → {PICKS_FILE}")

    # ── Auto-generate X thread writeup ───────────────────────────────────────
    try:
        from generate_mlb_writeups import generate as _gen_writeup
        print("\n── Generating X thread writeup ─────────────────────────")
        _gen_writeup(out)
    except Exception as e:
        print(f"  [warn] Writeup generation failed: {e}")

    # ── Auto-generate player props (BTS / HR / NRFI) ─────────────────────────
    try:
        from player_props import generate as _gen_props
        _gen_props(out)
    except Exception as e:
        print(f"  [warn] Player props generation failed: {e}")

    print("Run next: python generate_mlb_page.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--skip-injuries", action="store_true",
                        help="Disable injury nudge (useful for backtesting)")
    args = parser.parse_args()
    run(args.date, skip_injuries=args.skip_injuries)
