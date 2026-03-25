"""
run_model.py — Generate Today's MLB Picks
Loads lines, runs 3 models, applies edge thresholds, writes today_picks.json

Pinnacle integration:
  - Edge calculation now uses Pinnacle's no-vig closing probability as the
    true-market benchmark instead of a FD/DK no-vig blend.
  - Pinnacle market features are included in the feature row sent to models
    (pi_close_no_vig_home, pi_open_implied_home, movement columns,
     pi_close_total) — must match the feature order used in train_model.py.
  - Falls back to FD/DK no-vig if Pinnacle lines are unavailable for a game.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
from datetime import date, datetime
warnings.filterwarnings("ignore")

from config import (DATA_DIR, MODELS_DIR, LINES_PATH, PICKS_PATH, MODEL_PATH,
    EDGE_1U, EDGE_2U, EDGE_3U, HOME_FIELD_WIN_PCT_BOOST, LOGS_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

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
    """
    Derive Pinnacle no-vig probabilities.
    Preferred over FD/DK because Pinnacle carries minimal vig and reflects
    sharp-money consensus more accurately.
    """
    return no_vig(pi_ml_home, pi_ml_away)


def best_book_prob(side, books):
    """
    Return the best (lowest-vig) implied probability for a side.
    Priority: Pinnacle > FanDuel > DraftKings.
    Used as fallback when Pinnacle no-vig can't be computed.
    """
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
    if e >= EDGE_3U: return 3
    if e >= EDGE_2U: return 2
    if e >= EDGE_1U: return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Team stats
# ─────────────────────────────────────────────────────────────────────────────

def load_team_stats():
    from config import HIST_DIR
    bat, pit = {}, {}
    bp = os.path.join(HIST_DIR, "team_batting.parquet")
    pp = os.path.join(HIST_DIR, "team_pitching.parquet")
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
    """
    Build a feature row for one game.

    game (dict): the game object from lines JSON.  If it contains Pinnacle
                 fields, they are added to the row so the model can use them.
    """
    row = {}

    # ── Team batting / pitching features ──────────────────────────────────
    for pfx, tm in [("home_bat_", home), ("away_bat_", away)]:
        s = bat.get(tm, {})
        for f in BATTING_FEATS:
            row[f"{pfx}{f}"] = s.get(f, np.nan)
    for pfx, tm in [("home_pit_", home), ("away_pit_", away)]:
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

    # ── Pinnacle features ─────────────────────────────────────────────────
    # Extracted from the lines JSON; pull_lines.py should populate these
    # from the Pinnacle book entry.  Neutral defaults used when unavailable.
    books = (game or {}).get("books", {})
    pi    = books.get("pinnacle", {})

    pi_ml_home = pi.get("ml_home")
    pi_ml_away = pi.get("ml_away")
    pi_nv_home, _ = pinnacle_no_vig(pi_ml_home, pi_ml_away)

    # Opening lines — pull_lines.py should store these under game["pinnacle_open"]
    pi_open = (game or {}).get("pinnacle_open", {})
    pi_open_ml_home = pi_open.get("ml_home")
    pi_open_implied = american_to_prob(pi_open_ml_home)  # raw, vig-on

    row["pi_close_no_vig_home"]  = pi_nv_home   if pi_nv_home is not None  else 0.5
    row["pi_open_implied_home"]  = pi_open_implied if pi_open_implied is not None else 0.5
    row["pi_ml_movement"]        = (game or {}).get("pinnacle_ml_movement", 0.0) or 0.0
    row["pi_spread_movement"]    = (game or {}).get("pinnacle_spread_movement", 0.0) or 0.0
    row["pi_total_movement"]     = (game or {}).get("pinnacle_total_movement", 0.0) or 0.0
    row["pi_close_total"]        = pi.get("total") or (game or {}).get("total") or 8.5

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict(row, models):
    X = pd.DataFrame([row])
    for c in FEATURE_COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEATURE_COLS].astype(float).fillna(0)
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
        return {"win": None, "diff": None, "total": None}
    try:
        bundle = joblib.load(MODEL_PATH)
        return {
            "win":   bundle.get("ml"),
            "diff":  bundle.get("rl"),
            "total": bundle.get("totals"),
        }
    except Exception as e:
        print(f"  [ERROR] Model load failed: {e}")
        return {"win": None, "diff": None, "total": None}


# ─────────────────────────────────────────────────────────────────────────────
# Edge calculation — Pinnacle as primary benchmark
# ─────────────────────────────────────────────────────────────────────────────

def resolve_book_prob_ml(side, books):
    """
    Return (benchmark_prob, source_label) for a ML side.

    Hierarchy:
      1. Pinnacle no-vig (sharpest, lowest vig)
      2. FanDuel/DraftKings no-vig blend
    """
    pi = books.get("pinnacle", {})
    ph, pa = pinnacle_no_vig(pi.get("ml_home"), pi.get("ml_away"))
    if ph is not None:
        return (ph if side == "home" else pa), "pinnacle_no_vig"

    # Fallback: FD/DK no-vig
    for bk in ["fanduel", "draftkings"]:
        if bk in books:
            h, a = no_vig(books[bk].get("ml_home"), books[bk].get("ml_away"))
            if h and a:
                return (h if side == "home" else a), bk
    return None, None


def resolve_best_odds_ml(side, books):
    """Return (best_odds, book) — highest payout for this side."""
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
    for g in games:
        ht, at = g["home_team"], g["away_team"]
        books  = g.get("books", {})
        row    = build_row(ht, at, bat, pit, game=g)
        preds  = predict(row, models)
        hwp, awp = preds.get("hwp"), preds.get("awp")
        pdiff, ptot = preds.get("pred_run_diff"), preds.get("pred_total")

        if hwp:
            hwp = min(0.95, hwp + HOME_FIELD_WIN_PCT_BOOST)
            awp = 1.0 - hwp

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

        # ── Spread ───────────────────────────────────────────────────────
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

                # Use Pinnacle spread juice as benchmark when available
                pi = books.get("pinnacle", {})
                pi_j = pi.get(f"spread_juice_{side}")
                bp   = american_to_prob(pi_j) if pi_j else (american_to_prob(best_j) or 0.524)

                cp    = 1 / (1 + np.exp(-(md + spread) * 0.3))
                edge  = calc_edge(cp, bp)
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
        # Use Pinnacle's closing total as the line benchmark when available
        pi       = books.get("pinnacle", {})
        pi_total = pi.get("total")
        tl       = pi_total or g.get("total")   # prefer Pinnacle line

        if tl is not None and ptot is not None:
            diff_from_line = ptot - tl
            for direction, diff in [("over", diff_from_line), ("under", -diff_from_line)]:
                op = 1 / (1 + np.exp(-diff * 0.25))

                # Benchmark: Pinnacle juice > FD/DK juice
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

                # Best available odds for the bet itself
                best_j_bet, best_bk_bet = None, None
                for bk in ["fanduel", "draftkings", "pinnacle"]:
                    if bk in books:
                        j = books[bk].get(f"total_{direction}_juice")
                        if j is not None and (best_j_bet is None or j > best_j_bet):
                            best_j_bet, best_bk_bet = j, bk
                if best_j_bet is None:
                    best_j_bet, best_bk_bet = -110, "fanduel"

                edge  = calc_edge(op, bp)
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
    return picks


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

    if not os.path.exists(LINES_PATH):
        print("[ERROR] Run: python pull_lines.py")
        sys.exit(1)
    with open(LINES_PATH) as f:
        games = json.load(f)
    print(f"  {len(games)} games loaded")

    # Report Pinnacle availability in today's lines
    pi_games = sum(1 for g in games if "pinnacle" in g.get("books", {}))
    print(f"  Pinnacle lines available: {pi_games}/{len(games)} games")

    models = load_models()
    loaded = [k for k, v in models.items() if v]
    print(f"  Models: {', '.join(loaded) if loaded else 'NONE'}")

    bat, pit = load_team_stats()
    print(f"  Teams — bat:{len(bat)} pit:{len(pit)}")

    picks   = generate_picks(games, models, bat, pit)
    all_p   = [p for g in picks for p in g["picks"]]
    print(f"\n  3U: {sum(1 for p in all_p if p['units']==3)}"
          f"  2U: {sum(1 for p in all_p if p['units']==2)}"
          f"  1U: {sum(1 for p in all_p if p['units']==1)}")

    out = {
        "generated_at": datetime.now().isoformat(),
        "date":         td,
        "total_picks":  len(all_p),
        "games":        picks,
    }
    with open(PICKS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to today_picks.json | Run: python app.py")


if __name__ == "__main__":
    main()