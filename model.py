"""
model.py — MLB Betting Model (Gradient Boosting)
Edge thresholds: >=3% = 1u, >=6% = 2u, >=10% = 3u
"""
import json, numpy as np, pandas as pd, joblib
from pathlib import Path
from features import FEATURE_COLS, american_to_prob, remove_vig

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
EDGE_THRESHOLDS = [(0.10, 3), (0.06, 2), (0.03, 1)]

def get_units(edge):
    for t, u in EDGE_THRESHOLDS:
        if abs(edge) >= t: return u
    return 0

def save_models(ml, spread, total, meta=None):
    if ml:     joblib.dump(ml,     MODEL_DIR / "ml_model.pkl")
    if spread: joblib.dump(spread, MODEL_DIR / "spread_model.pkl")
    if total:  joblib.dump(total,  MODEL_DIR / "total_model.pkl")
    if meta:
        with open(MODEL_DIR / "meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"[model] Saved models to {MODEL_DIR}/")

def load_models():
    paths = [MODEL_DIR / "ml_model.pkl", MODEL_DIR / "spread_model.pkl", MODEL_DIR / "total_model.pkl"]
    if not all(p.exists() for p in paths):
        raise FileNotFoundError("Models not found. Run train_model.py first.")
    return joblib.load(paths[0]), joblib.load(paths[1]), joblib.load(paths[2])

def load_meta():
    p = MODEL_DIR / "meta.json"
    return json.load(open(p)) if p.exists() else {}

def train_models(training_df):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    import warnings; warnings.filterwarnings("ignore")
    print(f"[train] Training on {len(training_df)} games...")
    feature_df = training_df[FEATURE_COLS].copy()
    valid_mask = feature_df.notna().sum(axis=1) > (len(FEATURE_COLS) * 0.5)
    feature_df = feature_df[valid_mask]
    print(f"[train] {valid_mask.sum()} games after filtering")
    params = {"max_iter":300,"learning_rate":0.05,"max_depth":4,"min_samples_leaf":20,"l2_regularization":0.1,"random_state":42}
    targets = {"ml":("home_win","ML"),"spread":("home_covered_rl","Spread"),"total":("went_over","Total")}
    models = {}
    for key,(col,label) in targets.items():
        if col not in training_df.columns: models[key]=None; continue
        y = training_df.loc[valid_mask, col].fillna(0).astype(int)
        X = feature_df.fillna(-999)
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        print(f"  {label}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        models[key] = clf
    save_models(models.get("ml"), models.get("spread"), models.get("total"),
                meta={"training_rows":int(valid_mask.sum()),"feature_cols":FEATURE_COLS})
    return models.get("ml"), models.get("spread"), models.get("total")

def _best_odds(book_odds):
    valid = [(b,o) for b,o in book_odds if o is not None and not (isinstance(o,float) and np.isnan(o))]
    if not valid: return None, None
    def score(odds): return 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    return max(valid, key=lambda x: score(x[1]))

def predict_game(feature_dict, ml_model, spread_model, total_model,
                 fd_ml_home=None, fd_ml_away=None, dk_ml_home=None, dk_ml_away=None,
                 fd_spread_home=None, fd_spread_juice_home=None, fd_spread_juice_away=None,
                 dk_spread_home=None, dk_spread_juice_home=None, dk_spread_juice_away=None,
                 fd_total=None, fd_total_over_juice=None, fd_total_under_juice=None,
                 dk_total=None, dk_total_over_juice=None, dk_total_under_juice=None,
                 home_team="", away_team="", home_sp="", away_sp="", commence=""):
    X = pd.DataFrame([feature_dict])[FEATURE_COLS].fillna(-999)
    picks = []
    base = dict(home_team=home_team, away_team=away_team, home_sp=home_sp, away_sp=away_sp, commence=commence)

    # ML
    if ml_model:
        hp = ml_model.predict_proba(X)[0][1]
        for side, mp, fd_o, dk_o, team in [("home",hp,fd_ml_home,dk_ml_home,home_team),("away",1-hp,fd_ml_away,dk_ml_away,away_team)]:
            book, odds = _best_odds([("fanduel",fd_o),("draftkings",dk_o)])
            if odds is None: continue
            opp = (dk_ml_away if book=="draftkings" else fd_ml_away) if side=="home" else (dk_ml_home if book=="draftkings" else fd_ml_home)
            b1 = american_to_prob(odds); b2 = american_to_prob(opp) if opp else 0.5
            fh, fa = remove_vig(b1, b2)
            fp = fh if side=="home" else fa
            edge = mp - fp; units = get_units(edge)
            if units > 0:
                picks.append({**base,"market":"ML","side":side,"team":team,"model_prob":round(mp,4),"book_prob":round(fp,4),"edge":round(edge,4),"units":units,"book":book,"odds":int(odds)})

    # Run line
    if spread_model:
        cp = spread_model.predict_proba(X)[0][1]
        for side, mp, j1, j2, opp_j, spread_line, team_label in [
            ("home", cp, fd_spread_juice_home, dk_spread_juice_home, fd_spread_juice_away, fd_spread_home, f"{home_team} {fd_spread_home or -1.5:+.1f}"),
            ("away", 1-cp, fd_spread_juice_away, dk_spread_juice_away, fd_spread_juice_home, None, f"{away_team} +1.5"),
        ]:
            book, odds = _best_odds([("fanduel",j1),("draftkings",j2)])
            if odds is None: continue
            b1 = american_to_prob(odds); b2 = american_to_prob(opp_j) if opp_j else 0.5
            fh, fa = remove_vig(b1, b2)
            fp = fh if side=="home" else fa
            edge = mp - fp; units = get_units(edge)
            if units > 0:
                picks.append({**base,"market":"RL","side":side,"team":team_label,"model_prob":round(mp,4),"book_prob":round(fp,4),"edge":round(edge,4),"units":units,"book":book,"odds":int(odds)})

    # Totals
    if total_model:
        op = total_model.predict_proba(X)[0][1]
        for side, mp, fd_j, dk_j, opp_fd, opp_dk, line_fd, line_dk in [
            ("over", op, fd_total_over_juice, dk_total_over_juice, fd_total_under_juice, dk_total_under_juice, fd_total, dk_total),
            ("under", 1-op, fd_total_under_juice, dk_total_under_juice, fd_total_over_juice, dk_total_over_juice, fd_total, dk_total),
        ]:
            book, odds = _best_odds([("fanduel",fd_j),("draftkings",dk_j)])
            if odds is None: continue
            opp_j = opp_dk if book=="draftkings" else opp_fd
            line  = line_dk if book=="draftkings" else line_fd
            b1 = american_to_prob(odds); b2 = american_to_prob(opp_j) if opp_j else 0.5
            fh, fa = remove_vig(b1, b2)
            fp = fh
            edge = mp - fp; units = get_units(edge)
            prefix = "O" if side=="over" else "U"
            if units > 0:
                picks.append({**base,"market":"Total","side":side,"team":f"{prefix} {line}" if line else side.upper(),"model_prob":round(mp,4),"book_prob":round(fp,4),"edge":round(edge,4),"units":units,"book":book,"odds":int(odds)})

    return picks
