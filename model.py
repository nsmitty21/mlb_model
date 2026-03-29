"""
model.py — MLB Betting Model (Gradient Boosting)

Relationship to run_today.py
-----------------------------
model.py  = prediction engine. Owns predict_game(), load_models(), _units().
run_today.py = orchestrator. Builds features, calls predict_game(), writes JSON.

predict_game() changes
-----------------------
  - Now accepts feature_cols (list) from meta.json so the correct 65-column
    order is always used, regardless of what order keys appear in feature_dict.
  - Now accepts injury_adj (float) — a post-model probability nudge computed
    by run_today.py from the ESPN injury API. Applied to home_win_prob BEFORE
    comparing to book probability. Does not affect the spread/total models
    since those use symmetric cover probabilities.
"""
import json, numpy as np, pandas as pd, joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

ML_PKL     = MODEL_DIR / "ml_model.pkl"
SPREAD_PKL = MODEL_DIR / "spread_model.pkl"
TOTAL_PKL  = MODEL_DIR / "total_model.pkl"
META_JSON  = MODEL_DIR / "meta.json"

# Edge thresholds — used when called standalone. run_today.py reads these
# from meta.json so they stay in sync with whatever train_model.py used.
EDGE_THRESHOLDS = [(0.15, 3), (0.10, 2), (0.07, 1)]

# Sanity bounds: model_prob outside this range almost certainly means
# the feature row is all-NaN (fillna(-999) artifact). Skip the game.
MAX_MODEL_PROB = 0.85
MIN_MODEL_PROB = 0.15


def _units(edge: float) -> int:
    # Only fire on POSITIVE edges. abs() was previously used here, which caused
    # negative-edge picks to trigger (e.g. Edge:-27.7% producing a 3U pick).
    # A negative edge means the model thinks the book has the right side — skip.
    if edge <= 0:
        return 0
    for threshold, u in EDGE_THRESHOLDS:
        if edge >= threshold:
            return u
    return 0


def load_models():
    for p in (ML_PKL, SPREAD_PKL, TOTAL_PKL):
        if not p.exists():
            raise FileNotFoundError(
                f"Model file not found: {p}\nRun: python train_model.py"
            )
    return joblib.load(ML_PKL), joblib.load(SPREAD_PKL), joblib.load(TOTAL_PKL)


def load_meta() -> dict:
    return json.load(open(META_JSON)) if META_JSON.exists() else {}


def save_models(ml, spread, total, meta=None):
    if ml:     joblib.dump(ml,     ML_PKL)
    if spread: joblib.dump(spread, SPREAD_PKL)
    if total:  joblib.dump(total,  TOTAL_PKL)
    if meta:
        with open(META_JSON, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"[model] Saved to {MODEL_DIR}/")


def train_models(training_df):
    """Thin wrapper kept for backward compatibility. train_model.py is preferred."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import warnings; warnings.filterwarnings("ignore")

    meta      = load_meta()
    feat_cols = meta.get("feature_cols", list(training_df.columns))

    print(f"[train] Training on {len(training_df)} games, {len(feat_cols)} features...")
    feature_df = training_df[feat_cols].copy()
    valid_mask = feature_df.notna().sum(axis=1) > (len(feat_cols) * 0.5)
    feature_df = feature_df[valid_mask]
    print(f"[train] {valid_mask.sum()} games after filtering")

    params = {
        "max_iter": 300, "learning_rate": 0.05, "max_depth": 4,
        "min_samples_leaf": 20, "l2_regularization": 0.1, "random_state": 42,
    }
    targets = {
        "ml":     ("home_win",        "ML"),
        "spread": ("home_covered_rl", "Spread"),
        "total":  ("went_over",       "Total"),
    }
    models = {}
    for key, (col, label) in targets.items():
        if col not in training_df.columns:
            models[key] = None
            continue
        y   = training_df.loc[valid_mask, col].fillna(0).astype(int)
        X   = feature_df.fillna(-999)
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        print(f"  {label}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        models[key] = clf

    save_models(
        models.get("ml"), models.get("spread"), models.get("total"),
        meta={"training_rows": int(valid_mask.sum()), "feature_cols": feat_cols},
    )
    return models.get("ml"), models.get("spread"), models.get("total")


def _american_to_prob(odds):
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    return (100 / (odds + 100)) if odds > 0 else (abs(odds) / (abs(odds) + 100))


def _remove_vig(p1, p2):
    if not p1 or not p2:
        return p1, p2
    total = p1 + p2
    return (p1 / total, p2 / total) if total > 0 else (p1, p2)


def _best_odds(book_odds: list) -> tuple:
    """Return (book_name, odds) for the highest-payout option."""
    valid = [(b, o) for b, o in book_odds
             if o is not None and not (isinstance(o, float) and np.isnan(o))]
    if not valid:
        return None, None
    def _payout(odds):
        return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)
    return max(valid, key=lambda x: _payout(x[1]))


def predict_game(feature_dict: dict,
                 feature_cols: list,
                 ml_model, spread_model, total_model,
                 injury_adj: float = 0.0,
                 fd_ml_home=None, fd_ml_away=None,
                 dk_ml_home=None, dk_ml_away=None,
                 fd_spread_home=None,
                 fd_spread_juice_home=None, fd_spread_juice_away=None,
                 dk_spread_home=None,
                 dk_spread_juice_home=None, dk_spread_juice_away=None,
                 fd_total=None, fd_total_over_juice=None, fd_total_under_juice=None,
                 dk_total=None, dk_total_over_juice=None, dk_total_under_juice=None,
                 home_team="", away_team="", home_sp="", away_sp="",
                 commence="") -> list:
    """
    Score one game and return a list of pick dicts (may be empty).

    feature_cols must be the same list used during training (from meta.json).
    injury_adj is a signed float (±MAX_INJURY_ADJUSTMENT) computed by
    run_today.py from the ESPN injury API and added to home_win_prob before
    comparing to the book probability.
    """
    # Build feature matrix in the exact column order used during training
    X = pd.DataFrame([feature_dict])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols].fillna(-999)

    picks = []
    base  = dict(home_team=home_team, away_team=away_team,
                 home_sp=home_sp, away_sp=away_sp, commence=commence)

    # ── Moneyline ─────────────────────────────────────────────────────────
    if ml_model:
        raw_hp = ml_model.predict_proba(X)[0][1]

        if raw_hp < MIN_MODEL_PROB or raw_hp > MAX_MODEL_PROB:
            print(f"  [SKIP ML] {away_team}@{home_team} — "
                  f"model_prob={raw_hp:.3f} out of [{MIN_MODEL_PROB},{MAX_MODEL_PROB}] "
                  f"(likely all-NaN features)")
        else:
            # Apply injury nudge to home win probability
            hp = float(np.clip(raw_hp + injury_adj, 0.05, 0.95))
            ap = 1.0 - hp

            for side, mp, fd_o, dk_o, team in [
                ("home", hp, fd_ml_home, dk_ml_home, home_team),
                ("away", ap, fd_ml_away, dk_ml_away, away_team),
            ]:
                book, odds = _best_odds([("fanduel", fd_o), ("draftkings", dk_o)])
                if odds is None:
                    continue
                opp = (dk_ml_away if book == "draftkings" else fd_ml_away) \
                      if side == "home" \
                      else (dk_ml_home if book == "draftkings" else fd_ml_home)
                b1 = _american_to_prob(odds)
                b2 = _american_to_prob(opp) if opp else 0.5
                # BUG A FIX: b1 is always the bet side's own odds (home or away),
                # b2 is always the opponent's. _remove_vig(b1, b2) returns
                # (bet_side_nv, opponent_nv). fp must always be fh (the first
                # return = the bet team's own no-vig prob), never fa.
                # The old code used fa for away picks, which returned the
                # *opponent's* prob — generating massive fake edges on all
                # away favorites (e.g. TBR -464 away showed book_prob=22.7%
                # instead of 78.7%, producing a fraudulent 42pp "edge").
                bet_nv, _ = _remove_vig(b1, b2)
                fp    = bet_nv
                edge  = mp - fp
                units = _units(edge)
                if units > 0:
                    picks.append({
                        **base,
                        "market":     "ML",
                        "side":       side,
                        "team":       team,
                        "model_prob": round(mp,   4),
                        "book_prob":  round(fp,   4),
                        "edge":       round(edge, 4),
                        "units":      units,
                        "book":       book,
                        "odds":       int(odds),
                    })

    # ── Run line / Spread ─────────────────────────────────────────────────
    if spread_model:
        cp = spread_model.predict_proba(X)[0][1]

        if cp < MIN_MODEL_PROB or cp > MAX_MODEL_PROB:
            print(f"  [SKIP RL] {away_team}@{home_team} — "
                  f"cover_prob={cp:.3f} out of range")
        else:
            for side, mp, j1, j2, opp_j, spread_line, team_label in [
                ("home", cp,   fd_spread_juice_home, dk_spread_juice_home,
                 fd_spread_juice_away, fd_spread_home,
                 f"{home_team} {fd_spread_home:+.1f}"
                 if fd_spread_home else home_team),
                ("away", 1-cp, fd_spread_juice_away, dk_spread_juice_away,
                 fd_spread_juice_home, None, f"{away_team} +1.5"),
            ]:
                book, odds = _best_odds([("fanduel", j1), ("draftkings", j2)])
                if odds is None:
                    continue
                b1 = _american_to_prob(odds)
                b2 = _american_to_prob(opp_j) if opp_j else 0.5
                # BUG A FIX: same as ML — b1 is always the bet side's juice,
                # b2 is the opponent's. Use first return value only.
                bet_nv, _ = _remove_vig(b1, b2)
                fp    = bet_nv
                edge  = mp - fp
                units = _units(edge)
                if units > 0:
                    picks.append({
                        **base,
                        "market":      "RL",
                        "side":        side,
                        "team":        team_label,
                        "spread_line": fd_spread_home if side == "home"
                                       else (-fd_spread_home
                                             if fd_spread_home else 1.5),
                        "model_prob":  round(mp,   4),
                        "book_prob":   round(fp,   4),
                        "edge":        round(edge, 4),
                        "units":       units,
                        "book":        book,
                        "odds":        int(odds),
                    })

    # ── Totals ────────────────────────────────────────────────────────────
    if total_model:
        op = total_model.predict_proba(X)[0][1]

        if op < MIN_MODEL_PROB or op > MAX_MODEL_PROB:
            print(f"  [SKIP TOT] {away_team}@{home_team} — "
                  f"over_prob={op:.3f} out of range")
        else:
            total_line = fd_total or dk_total
            if total_line is not None and total_line < 5.0:
                print(f"  [SKIP TOT] {away_team}@{home_team} — "
                      f"total line {total_line} implausibly low")
            else:
                for side, mp, fd_j, dk_j, opp_fd, opp_dk, line_fd, line_dk in [
                    ("over",  op,   fd_total_over_juice,  dk_total_over_juice,
                     fd_total_under_juice, dk_total_under_juice, fd_total, dk_total),
                    ("under", 1-op, fd_total_under_juice, dk_total_under_juice,
                     fd_total_over_juice,  dk_total_over_juice,  fd_total, dk_total),
                ]:
                    book, odds = _best_odds([("fanduel", fd_j), ("draftkings", dk_j)])
                    if odds is None:
                        continue
                    opp_j = opp_dk if book == "draftkings" else opp_fd
                    line  = line_dk if book == "draftkings" else line_fd
                    b1 = _american_to_prob(odds)
                    b2 = _american_to_prob(opp_j) if opp_j else 0.5
                    fh, _  = _remove_vig(b1, b2)
                    edge   = mp - fh
                    units  = _units(edge)
                    prefix = "O" if side == "over" else "U"
                    if units > 0:
                        picks.append({
                            **base,
                            "market":     "Total",
                            "side":       side,
                            "team":       f"{prefix} {line}" if line else side.upper(),
                            "total_line": line,
                            "model_prob": round(mp,   4),
                            "book_prob":  round(fh,   4),
                            "edge":       round(edge, 4),
                            "units":      units,
                            "book":       book,
                            "odds":       int(odds),
                        })

    return picks
