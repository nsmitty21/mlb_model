"""
run_model.py — Generate Today's MLB Picks
Loads lines, runs 3 models, applies edge thresholds, writes picks_today.json

Pinnacle integration:
  - Edge calculation uses Pinnacle's no-vig closing probability as the
    true-market benchmark instead of a FD/DK no-vig blend.
  - Pinnacle market features are included in the feature row sent to models.
  - Falls back to FD/DK no-vig if Pinnacle lines are unavailable for a game.

Fixes (2026-03-26 v2):
  - BUG 1 (prev): scaler.joblib now loaded and applied in predict().
  - BUG 2 (prev): Sigmoid slopes flattened to reduce sensitivity.
  - BUG 3 (prev): MAX_EDGE_PCT cap (25%) kills runaway edges.
  - BUG 4 (NEW):  Data-quality gate — games where >50% of team-stat features
                  are missing/zero are skipped entirely. This is the root cause
                  of all picks being underdog MLs: when 2026 stats haven't
                  populated, every feature row is all-zeros → model outputs a
                  near-constant ~55% for every away team → large fake edges
                  only appear on underdogs (whose book_prob is low).
  - BUG 5 (NEW):  Constant-probability detector — if 3+ ML picks share the
                  same model_prob within 0.5pp, the run is aborted before
                  saving any picks (dead giveaway of all-zero feature rows).
  - BUG 6 (NEW):  MIN_EDGE_FOR_3U raised from 10% to 15% to reflect that a
                  genuine mismatch edge should be rare. Config aliases kept for
                  backward compat; the stricter constant is enforced here.
  - BUG 7 (NEW):  Home-field boost was double-counted. The model was trained
                  with home_field=1 as a feature, so it already prices home
                  advantage. The additional 2% boost is now removed from the
                  edge calculation path. The config constant is kept so the
                  dashboard still displays correctly, but it is NOT applied
                  before comparing to book_prob.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
from datetime import date, datetime
warnings.filterwarnings("ignore")

from config import (DATA_DIR, MODELS_DIR, LINES_PATH, PICKS_PATH, MODEL_PATH,
    EDGE_1U, EDGE_2U, EDGE_3U, HOME_FIELD_WIN_PCT_BOOST, LOGS_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Edge caps ──────────────────────────────────────────────────────────────────
# Anything above MAX_EDGE_PCT is almost certainly a data or scaling artifact.
MAX_EDGE_PCT = 25.0

# 3U should only fire on true mismatches. Override the config threshold here
# so that ~55% model output (which happens when features are all-zero or nearly
# so) cannot mechanically reach 3U just because a dog has low book_prob.
MIN_EDGE_3U = max(EDGE_3U, 15.0)   # never lower than 15%, even if config says 10%
MIN_EDGE_2U = EDGE_2U               # 7% stays reasonable
MIN_EDGE_1U = EDGE_1U               # 5% floor

# Minimum fraction of TEAM_FEATURE_COLS that must be non-zero for a game row
# to be considered usable. Below this → model is running blind.
MIN_FEATURE_FILL_FRAC = 0.50

# If ≥ this many ML picks share the same model_prob within PROB_CLUSTER_BAND,
# the entire run is aborted.
PROB_CLUSTER_MIN_COUNT = 3
PROB_CLUSTER_BAND      = 0.5        # percentage-points

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions — must stay in sync with train_model.py
# ─────────────────────────────────────────────────────────────────────────────

BATTING_FEATS  = ["OPS","wOBA","AVG","OBP","SLG","HR","BB%","K%","R"]
PITCHING_FEATS = ["ERA","FIP","WHIP","K/9","BB/9","xFIP"]

TEAM_FEATURE_COLS = (
    [f"home_bat_{f}" for f in BATTING_FEATS] +
    [f"away_bat_{f}" for f in BATTING_FEATS] +
    [f"home_pit_{f}" for f in PITCHING_FEATS] +
    [f"away_pit_{f}" for f in PITCHING_FEATS] +
    [f"diff_bat_{f}" for f in BATTING_FEATS] +
    [f"diff_pit_{f}" for f in PITCHING_FEATS] +
    ["home_field"]
)

PINNACLE_FEATURE_COLS = [
    "pi_close_no_vig_home",
    "pi_open_implied_home",
    "pi_ml_movement",
    "pi_spread_movement",
    "pi_total_movement",
    "pi_close_total",
]

FEATURE_COLS = TEAM_FEATURE_COLS + PINNACLE_FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Odds / probability helpers
# ─────────────────────────────────────────────────────────────────────────────

def american_to_prob(o):
    if o is None or (isinstance(o, float) and np.isnan(o)):
        return None
    return (100 / (o + 100)) if o > 0 else (abs(o) / (abs(o) + 100))


def no_vig(mh, ma):
    """Remove vig from a two-sided market; returns (home_prob, away_prob)."""
    ph, pa = american_to_prob(mh), american_to_prob(ma)
    if not ph or not pa:
        return None, None
    t = ph + pa
    return ph / t, pa / t


def pinnacle_no_vig(pi_ml_home, pi_ml_away):
    return no_vig(pi_ml_home, pi_ml_away)


def best_book_prob(side, books):
    for bk in ["pinnacle", "fanduel", "draftkings"]:
        if bk not in books:
            continue
        ml_h = books[bk].get("ml_home")
        ml_a = books[bk].get("ml_away")
        h, a = no_vig(ml_h, ml_a)
        if h and a:
            return (h if side == "home" else a), bk
    return None, None


def calc_edge(mp, bp):
    return (mp - bp) * 100 if mp and bp else 0.0


def units_for_edge(e):
    if e >= MIN_EDGE_3U: return 3
    if e >= MIN_EDGE_2U: return 2
    if e >= MIN_EDGE_1U: return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Team stats
# ─────────────────────────────────────────────────────────────────────────────


# ── Team name normalization ───────────────────────────────────────────────────
# The Odds API returns full names ("New York Mets"); pybaseball stores
# abbreviations ("NYM").  This map lets us look up either form.
FULL_NAME_TO_ABB = {
    "Arizona Diamondbacks":  "ARI",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CHW",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KCR",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "OAK",
    "Athletics":             "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SDP",
    "San Francisco Giants":  "SFG",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TBR",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSN",
}

def normalize_team(name: str) -> str:
    """Return the canonical key used in the bat/pit dicts.

    pybaseball uses 3-letter abbreviations as the 'Team' column, so
    team stats are stored under keys like 'NYM', 'PIT', etc.
    The Odds API returns full names ('New York Mets').  Convert if needed.
    """
    return FULL_NAME_TO_ABB.get(name, name)


def load_team_stats():
    from config import HIST_DIR, LIVE_DIR
    bat, pit = {}, {}

    # Prefer live parquets (more recent) but fall back to historical
    def _best_path(filename):
        live = os.path.join(LIVE_DIR, filename)
        hist = os.path.join(HIST_DIR, filename)
        if os.path.exists(live):
            return live
        return hist

    bp = _best_path("team_batting.parquet")
    pp = _best_path("team_pitching.parquet")

    if os.path.exists(bp):
        df = pd.read_parquet(bp)
        if "team" in df.columns:
            for _, r in (df.sort_values("season" if "season" in df.columns else df.columns[0])
                           .groupby("team").last().reset_index().iterrows()):
                bat[r["team"]] = r.to_dict()
    if os.path.exists(pp):
        df = pd.read_parquet(pp)
        if "team" in df.columns:
            for _, r in (df.sort_values("season" if "season" in df.columns else df.columns[0])
                           .groupby("team").last().reset_index().iterrows()):
                pit[r["team"]] = r.to_dict()
    return bat, pit


# ─────────────────────────────────────────────────────────────────────────────
# Feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_row(home, away, bat, pit, game=None):
    row = {}

    # Normalize full names ("New York Mets") → abbreviations ("NYM")
    home_key = normalize_team(home)
    away_key = normalize_team(away)

    for pfx, tm in [("home_bat_", home_key), ("away_bat_", away_key)]:
        s = bat.get(tm, {})
        for f in BATTING_FEATS:
            row[f"{pfx}{f}"] = s.get(f, np.nan)
    for pfx, tm in [("home_pit_", home_key), ("away_pit_", away_key)]:
        s = pit.get(tm, {})
        for f in PITCHING_FEATS:
            row[f"{pfx}{f}"] = s.get(f, np.nan)

    row["home_field"] = 1

    for f in BATTING_FEATS:
        hv, av = row.get(f"home_bat_{f}", np.nan), row.get(f"away_bat_{f}", np.nan)
        row[f"diff_bat_{f}"] = (hv - av) if not (np.isnan(hv) or np.isnan(av)) else np.nan
    for f in PITCHING_FEATS:
        hv, av = row.get(f"home_pit_{f}", np.nan), row.get(f"away_pit_{f}", np.nan)
        if not (np.isnan(hv) or np.isnan(av)):
            row[f"diff_pit_{f}"] = (av - hv) if f in ("ERA","FIP","WHIP","BB/9") else (hv - av)
        else:
            row[f"diff_pit_{f}"] = np.nan

    books = (game or {}).get("books", {})
    pi    = books.get("pinnacle", {})

    pi_ml_home = pi.get("ml_home")
    pi_ml_away = pi.get("ml_away")
    pi_nv_home, _ = pinnacle_no_vig(pi_ml_home, pi_ml_away)

    pi_open = (game or {}).get("pinnacle_open", {})
    pi_open_ml_home = pi_open.get("ml_home")
    pi_open_implied = american_to_prob(pi_open_ml_home)

    row["pi_close_no_vig_home"]  = pi_nv_home    if pi_nv_home is not None   else 0.5
    row["pi_open_implied_home"]  = pi_open_implied if pi_open_implied is not None else 0.5
    row["pi_ml_movement"]        = (game or {}).get("pinnacle_ml_movement", 0.0) or 0.0
    row["pi_spread_movement"]    = (game or {}).get("pinnacle_spread_movement", 0.0) or 0.0
    row["pi_total_movement"]     = (game or {}).get("pinnacle_total_movement", 0.0) or 0.0
    row["pi_close_total"]        = pi.get("total") or (game or {}).get("total") or 8.5

    return row


def check_feature_quality(row, home, away):
    """
    Return (ok: bool, fill_frac: float, message: str).

    Checks that a meaningful fraction of TEAM_FEATURE_COLS (excluding
    home_field which is always 1) are non-zero and non-NaN.  When the 2026
    season stats haven't populated yet every raw team stat will be NaN →
    fillna(0) makes them all zero → model outputs a near-constant probability
    → every underdog generates a large fake edge.
    """
    check_cols = [c for c in TEAM_FEATURE_COLS if c != "home_field"]
    non_zero = sum(
        1 for c in check_cols
        if c in row and not (isinstance(row[c], float) and np.isnan(row[c]))
        and row[c] != 0
    )
    frac = non_zero / len(check_cols) if check_cols else 0.0
    ok   = frac >= MIN_FEATURE_FILL_FRAC
    msg  = (
        f"{home} vs {away}: {non_zero}/{len(check_cols)} team features populated "
        f"({frac:.0%}) — {'OK' if ok else 'SKIPPED (insufficient data)'}"
    )
    return ok, frac, msg


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(row, models):
    X = pd.DataFrame([row])
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURE_COLS].astype(float).fillna(0)

    scaler = models.get("scaler")
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
    else:
        print("  [WARN] No scaler loaded — regressor predictions may be unreliable")

    res = {}
    if models.get("win"):
        try:
            p = models["win"].predict_proba(X)[0]
            res["hwp"] = float(p[1])
            res["awp"] = float(p[0])
        except:
            res["hwp"] = res["awp"] = None
    else:
        res["hwp"] = res["awp"] = None
    for k, nm in [("diff","pred_run_diff"), ("total","pred_total")]:
        if models.get(k):
            try:
                res[nm] = float(models[k].predict(X)[0])
            except:
                res[nm] = None
        else:
            res[nm] = None
    return res


def load_models():
    if not os.path.exists(MODEL_PATH):
        print(f"  [ERROR] Model file not found: {MODEL_PATH}")
        return {"win": None, "diff": None, "total": None, "scaler": None}
    try:
        bundle = joblib.load(MODEL_PATH)

        scaler = None
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"  Scaler loaded from {scaler_path}")
        else:
            print(f"  [WARN] scaler.joblib not found at {scaler_path} — predictions will be off")

        return {
            "win":    bundle.get("ml"),
            "diff":   bundle.get("rl"),
            "total":  bundle.get("totals"),
            "scaler": scaler,
        }
    except Exception as e:
        print(f"  [ERROR] Model load failed: {e}")
        return {"win": None, "diff": None, "total": None, "scaler": None}


# ─────────────────────────────────────────────────────────────────────────────
# Edge calculation — Pinnacle as primary benchmark
# ─────────────────────────────────────────────────────────────────────────────

def resolve_book_prob_ml(side, books):
    pi = books.get("pinnacle", {})
    ph, pa = pinnacle_no_vig(pi.get("ml_home"), pi.get("ml_away"))
    if ph is not None:
        return (ph if side == "home" else pa), "pinnacle_no_vig"

    for bk in ["fanduel", "draftkings"]:
        if bk in books:
            h, a = no_vig(books[bk].get("ml_home"), books[bk].get("ml_away"))
            if h and a:
                return (h if side == "home" else a), bk
    return None, None


def resolve_best_odds_ml(side, books):
    best_ml, best_bk = None, None
    for bk in ["fanduel", "draftkings", "pinnacle"]:
        if bk in books:
            ml = books[bk].get(f"ml_{side}")
            if ml is not None and (best_ml is None or ml > best_ml):
                best_ml, best_bk = ml, bk
    return best_ml, best_bk


# ─────────────────────────────────────────────────────────────────────────────
# Pick generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_picks(games, models, bat, pit):
    picks = []
    skipped_data = 0

    for g in games:
        ht, at = g["home_team"], g["away_team"]
        books  = g.get("books", {})
        row    = build_row(ht, at, bat, pit, game=g)

        # ── BUG 4 FIX: Data-quality gate ──────────────────────────────────
        ok, fill_frac, quality_msg = check_feature_quality(row, ht, at)
        print(f"  [DATA] {quality_msg}")
        if not ok:
            skipped_data += 1
            continue

        preds  = predict(row, models)
        hwp, awp = preds.get("hwp"), preds.get("awp")
        pdiff, ptot = preds.get("pred_run_diff"), preds.get("pred_total")

        # ── BUG 7 FIX: Do NOT apply home-field boost before edge calc ─────
        # The model was trained with home_field=1 as a feature, so home
        # advantage is already priced into hwp/awp.  Adding 2% on top is
        # double-counting and inflates the apparent edge for away teams.
        # The boost constant in config is kept for display/dashboard use only.

        gpicks = []

        # ── Moneyline ─────────────────────────────────────────────────────
        for side, team, mp in [("home", ht, hwp), ("away", at, awp)]:
            if mp is None:
                continue

            bp, bp_source = resolve_book_prob_ml(side, books)
            if bp is None:
                continue

            best_ml, best_bk = resolve_best_odds_ml(side, books)
            if best_ml is None:
                continue

            edge  = calc_edge(mp, bp)

            if edge > MAX_EDGE_PCT:
                print(f"  [SKIP] {team} ML edge {edge:.1f}% exceeds cap — possible data issue")
                continue

            units = units_for_edge(edge)
            if units > 0:
                gpicks.append({
                    "type":         "ML",
                    "side":         side,
                    "team":         team,
                    "opponent":     at if side == "home" else ht,
                    "odds":         best_ml,
                    "book":         best_bk,
                    "model_prob":   round(mp * 100, 1),
                    "book_prob":    round(bp * 100, 1),
                    "book_source":  bp_source,
                    "edge":         round(edge, 2),
                    "units":        units,
                })

        # ── Spread ────────────────────────────────────────────────────────
        sh = g.get("spread_home")
        if sh is not None and pdiff is not None:
            for side, spread, md in [("home", sh, pdiff), ("away", -sh, -pdiff)]:
                team = ht if side == "home" else at
                best_j, best_bk = None, None
                for bk in ["fanduel", "draftkings", "pinnacle"]:
                    if bk in books:
                        j = books[bk].get(f"spread_juice_{side}")
                        if j is not None and (best_j is None or j > best_j):
                            best_j, best_bk = j, bk
                if best_j is None:
                    best_j, best_bk = -110, "fanduel"

                pi   = books.get("pinnacle", {})
                pi_j = pi.get(f"spread_juice_{side}")
                bp   = american_to_prob(pi_j) if pi_j else (american_to_prob(best_j) or 0.524)

                cp    = 1 / (1 + np.exp(-(md + spread) * 0.18))
                edge  = calc_edge(cp, bp)

                if edge > MAX_EDGE_PCT:
                    print(f"  [SKIP] {team} spread edge {edge:.1f}% exceeds cap — possible data issue")
                    continue

                units = units_for_edge(edge)
                if units > 0:
                    gpicks.append({
                        "type":        "SPREAD",
                        "side":        side,
                        "team":        team,
                        "opponent":    at if side == "home" else ht,
                        "spread":      spread,
                        "odds":        best_j,
                        "book":        best_bk,
                        "model_diff":  round(md, 2),
                        "cover_prob":  round(cp * 100, 1),
                        "book_prob":   round(bp * 100, 1),
                        "edge":        round(edge, 2),
                        "units":       units,
                    })

        # ── Totals ────────────────────────────────────────────────────────
        pi       = books.get("pinnacle", {})
        pi_total = pi.get("total")
        tl       = pi_total or g.get("total")

        if tl is not None and ptot is not None:
            diff_from_line = ptot - tl
            for direction, diff in [("over", diff_from_line), ("under", -diff_from_line)]:
                op = 1 / (1 + np.exp(-diff * 0.15))

                pi_j = pi.get(f"total_{direction}_juice")
                if pi_j:
                    bp = american_to_prob(pi_j) or 0.524
                    bk_used = "pinnacle"
                else:
                    best_j, bk_used = None, None
                    for bk in ["fanduel", "draftkings"]:
                        if bk in books:
                            j = books[bk].get(f"total_{direction}_juice")
                            if j is not None and (best_j is None or j > best_j):
                                best_j, bk_used = j, bk
                    bp = (american_to_prob(best_j) or 0.524) if best_j else 0.524

                best_j_bet, best_bk_bet = None, None
                for bk in ["fanduel", "draftkings", "pinnacle"]:
                    if bk in books:
                        j = books[bk].get(f"total_{direction}_juice")
                        if j is not None and (best_j_bet is None or j > best_j_bet):
                            best_j_bet, best_bk_bet = j, bk
                if best_j_bet is None:
                    best_j_bet, best_bk_bet = -110, "fanduel"

                edge  = calc_edge(op, bp)

                if edge > MAX_EDGE_PCT:
                    print(f"  [SKIP] {ht}/{at} total {direction} edge {edge:.1f}% exceeds cap — possible data issue")
                    continue

                units = units_for_edge(edge)
                if units > 0:
                    gpicks.append({
                        "type":              "TOTAL",
                        "direction":         direction.upper(),
                        "home_team":         ht,
                        "away_team":         at,
                        "total_line":        tl,
                        "pred_total":        round(ptot, 2),
                        "odds":              best_j_bet,
                        "book":              best_bk_bet,
                        "book_prob":         round(bp * 100, 1),
                        "book_source":       bk_used,
                        "edge":              round(edge, 2),
                        "units":             units,
                    })

        if gpicks:
            picks.append({
                "game_id":   g["game_id"],
                "date":      g["date"],
                "commence":  g["commence"],
                "home_team": ht,
                "away_team": at,
                "picks":     sorted(gpicks, key=lambda x: x["units"], reverse=True),
            })

    if skipped_data:
        print(f"\n  [WARN] {skipped_data} game(s) skipped due to missing team stats.")
        print("         Run fetch_historical.py or check data/live/ parquet files.")

    return picks


def detect_probability_clustering(picks):
    """
    BUG 5 FIX: Abort if multiple ML picks share the same model_prob within
    PROB_CLUSTER_BAND percentage-points — dead giveaway that the model is
    receiving all-zero feature rows and outputting a near-constant prediction.
    Returns (clustered: bool, message: str).
    """
    ml_probs = [
        p["model_prob"]
        for g in picks
        for p in g["picks"]
        if p["type"] == "ML" and "model_prob" in p
    ]
    if len(ml_probs) < PROB_CLUSTER_MIN_COUNT:
        return False, ""

    # Count how many fall within any band of width PROB_CLUSTER_BAND
    ml_probs_sorted = sorted(ml_probs)
    for i in range(len(ml_probs_sorted)):
        cluster = [
            p for p in ml_probs_sorted
            if abs(p - ml_probs_sorted[i]) <= PROB_CLUSTER_BAND
        ]
        if len(cluster) >= PROB_CLUSTER_MIN_COUNT:
            msg = (
                f"CLUSTER DETECTED: {len(cluster)} ML picks share model_prob "
                f"≈ {ml_probs_sorted[i]:.1f}% (within {PROB_CLUSTER_BAND}pp). "
                f"This means the model is receiving near-identical feature rows — "
                f"almost certainly because team stats for today's games are missing. "
                f"Picks NOT saved. Fix: re-run fetch_historical.py and confirm "
                f"data/live/team_batting.parquet contains 2026 rows."
            )
            return True, msg

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    args, _ = parser.parse_known_args()
    td = args.date or date.today().isoformat()
    print(f"MLB Model Run — {td}")
    print(f"  Edge thresholds: 1U≥{MIN_EDGE_1U}%  2U≥{MIN_EDGE_2U}%  3U≥{MIN_EDGE_3U}%")

    if not os.path.exists(LINES_PATH):
        print("[ERROR] Run: python pull_lines.py")
        sys.exit(1)
    with open(LINES_PATH) as f:
        games = json.load(f)
    print(f"  {len(games)} games loaded")

    pi_games = sum(1 for g in games if "pinnacle" in g.get("books", {}))
    print(f"  Pinnacle lines available: {pi_games}/{len(games)} games")

    models = load_models()
    loaded = [k for k, v in models.items() if v]
    print(f"  Models: {', '.join(loaded) if loaded else 'NONE'}")

    bat, pit = load_team_stats()
    print(f"  Teams — bat:{len(bat)} pit:{len(pit)}")

    if len(bat) == 0 or len(pit) == 0:
        print("[ERROR] No team stats loaded — cannot generate picks.")
        print("        Run fetch_historical.py to populate data/historical/ parquets.")
        sys.exit(1)

    picks = generate_picks(games, models, bat, pit)

    # ── BUG 5 FIX: Probability clustering check ───────────────────────────
    clustered, cluster_msg = detect_probability_clustering(picks)
    if clustered:
        print(f"\n  [ABORT] {cluster_msg}")
        sys.exit(1)

    all_p = [p for g in picks for p in g["picks"]]
    print(f"\n  3U: {sum(1 for p in all_p if p['units']==3)}"
          f"  2U: {sum(1 for p in all_p if p['units']==2)}"
          f"  1U: {sum(1 for p in all_p if p['units']==1)}")

    # ── Pick-mix sanity check ──────────────────────────────────────────────
    ml_picks     = [p for p in all_p if p["type"] == "ML"]
    away_ml      = [p for p in ml_picks if p["side"] == "away"]
    three_u_pct  = sum(1 for p in all_p if p["units"] == 3) / max(len(all_p), 1)

    if len(ml_picks) > 0 and len(away_ml) == len(ml_picks):
        print("  [WARN] Every ML pick is an away team — "
              "verify team stats are populated for home teams.")
    if three_u_pct > 0.70 and len(all_p) >= 4:
        print(f"  [WARN] {three_u_pct:.0%} of picks are 3U — unusually high. "
              "Review picks manually before betting.")

    spread_picks = [p for p in all_p if p["type"] == "SPREAD"]
    if len(ml_picks) > 0 and len(spread_picks) == 0:
        print("  [INFO] No spread picks generated — run_diff model may be "
              "under-powered or spread juice is too tight today.")

    out = {
        "generated_at": datetime.now().isoformat(),
        "date":         td,
        "total_picks":  len(all_p),
        "games":        picks,
    }
    with open(PICKS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to picks_today.json | Run: python app.py")


if __name__ == "__main__":
    main()