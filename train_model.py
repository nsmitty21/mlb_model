"""
train_model.py - Train / Retrain the MLB Betting Model
Trains 3 XGBoost models: win probability, run differential, total runs

Pinnacle features added:
  pi_open_implied_home   – Pinnacle opening implied home win prob (raw, vig-on)
  pi_close_no_vig_home   – Pinnacle closing no-vig home win prob  ← sharpest signal
  pi_ml_movement         – Opening-to-closing ML home movement (American odds)
  pi_spread_movement     – Opening-to-closing spread movement
  pi_total_movement      – Opening-to-closing total movement
  pi_close_total         – Pinnacle closing total line

Usage: python train_model.py
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from config import HIST_DIR, MODEL_DIR, TRAIN_YEARS

HISTORICAL_PATH = HIST_DIR + "/historical_stats.parquet"
MODELS_DIR      = MODEL_DIR
MODEL_PATH      = MODEL_DIR + "/mlb_model.joblib"
SCALER_PATH     = MODEL_DIR + "/scaler.joblib"
FEATURES_PATH   = MODEL_DIR + "/features.json"
SEASON_PATH     = "data/live/season_2026.parquet"

os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────────────────────────────────────

BATTING_FEATS  = ["OPS","wOBA","AVG","OBP","SLG","HR","BB%","K%","R"]
PITCHING_FEATS = ["ERA","FIP","WHIP","K/9","BB/9","xFIP"]

# Core team-stat features (unchanged from original)
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(HISTORICAL_PATH):
        print("[ERROR] No historical data found. Run fetch_historical.py first.")
        sys.exit(1)
    df = pd.read_parquet(HISTORICAL_PATH)
    print(f"Loaded {len(df)} games from historical data")

    if os.path.exists(SEASON_PATH):
        live = pd.read_parquet(SEASON_PATH)
        df = pd.concat([df, live], ignore_index=True)
        print(f"+ {len(live)} live season games -> {len(df)} total")

    return df


def build_features(df):
    """
    Ensure all feature columns exist, filling missing Pinnacle data with the
    median so training rows without PI lines are still usable.
    """
    # BUG 2 FIX: Validate that the raw batting/pitching columns from pybaseball
    # actually exist before we try to compute diffs from them.  pybaseball has
    # historically renamed columns (e.g. "BB%" → "BB_pct") between versions.
    # If a column is missing we fill with NaN (not 0) so the diff columns also
    # become NaN and get caught by the median-fill step below, rather than
    # silently computing 0-based diffs that look like valid data.
    missing_bat = [f for f in BATTING_FEATS
                   if f"home_bat_{f}" not in df.columns and f"away_bat_{f}" not in df.columns]
    missing_pit = [f for f in PITCHING_FEATS
                   if f"home_pit_{f}" not in df.columns and f"away_pit_{f}" not in df.columns]
    if missing_bat:
        print(f"  [warn] Batting features absent from training data: {missing_bat}")
        print(f"         Likely pybaseball column-name drift. "
              f"Available cols: {[c for c in df.columns if 'bat' in c][:10]}")
        for f in missing_bat:
            for pfx in ("home_bat_", "away_bat_"):
                if f"{pfx}{f}" not in df.columns:
                    df[f"{pfx}{f}"] = np.nan
    if missing_pit:
        print(f"  [warn] Pitching features absent from training data: {missing_pit}")
        print(f"         Likely pybaseball column-name drift. "
              f"Available cols: {[c for c in df.columns if 'pit' in c][:10]}")
        for f in missing_pit:
            for pfx in ("home_pit_", "away_pit_"):
                if f"{pfx}{f}" not in df.columns:
                    df[f"{pfx}{f}"] = np.nan

    missing_pi = [c for c in PINNACLE_FEATURE_COLS if c not in df.columns]
    if missing_pi:
        print(f"  [warn] {len(missing_pi)} Pinnacle features missing from parquet: {missing_pi}")
        for c in missing_pi:
            df[c] = np.nan

    # Report Pinnacle coverage before filling
    pi_avail = [c for c in PINNACLE_FEATURE_COLS if c in df.columns]
    for c in pi_avail:
        cov = df[c].notna().mean()
        if cov < 0.50:
            print(f"  [warn] {c} only {cov:.1%} non-null — PI signal will be weak")

    # ── Compute diff_ and home_field FIRST, before any missing-col fill ──────
    # This ensures diffs are computed from real data, not zeros.
    df["home_field"] = 1
    for f in BATTING_FEATS:
        hv = df.get(f"home_bat_{f}", pd.Series(dtype=float))
        av = df.get(f"away_bat_{f}", pd.Series(dtype=float))
        df[f"diff_bat_{f}"] = hv - av
    for f in PITCHING_FEATS:
        hv = df.get(f"home_pit_{f}", pd.Series(dtype=float))
        av = df.get(f"away_pit_{f}", pd.Series(dtype=float))
        # For ERA/FIP/WHIP/BB9: lower is better, so away - home = positive means home advantage
        if f in ("ERA", "FIP", "WHIP", "BB/9"):
            df[f"diff_pit_{f}"] = av - hv
        else:
            df[f"diff_pit_{f}"] = hv - av

    # Now fill any remaining missing raw team-stat columns
    missing_team = [c for c in TEAM_FEATURE_COLS if c not in df.columns]
    if missing_team:
        print(f"  [warn] {len(missing_team)} team-stat features still missing, filling with 0")
        for c in missing_team:
            df[c] = 0.0

    # Fill team stats with median (original behaviour)
    df[TEAM_FEATURE_COLS] = df[TEAM_FEATURE_COLS].fillna(df[TEAM_FEATURE_COLS].median())

    # Fill Pinnacle features with neutral values
    movement_cols = ["pi_ml_movement", "pi_spread_movement", "pi_total_movement"]
    prob_cols     = ["pi_close_no_vig_home", "pi_open_implied_home"]
    total_col     = ["pi_close_total"]

    for c in movement_cols:
        df[c] = df[c].fillna(0.0)
    for c in prob_cols:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 0.5)
    for c in total_col:
        df[c] = df[c].fillna(df[c].median() if df[c].notna().any() else 8.5)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 55)
    print("MLB Model Training")
    print("=" * 55)

    df = load_data()
    df = build_features(df)

    X      = df[FEATURE_COLS].values
    y_win  = df["home_win"].values.astype(int)
    y_runs = df["total_runs"].values.astype(float)
    y_diff = df["run_diff"].values.astype(float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    print("\nTraining moneyline model (win probability)...")
    ml_model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )
    ml_model.fit(X_scaled, y_win)
    scores = cross_val_score(ml_model, X_scaled, y_win, cv=tscv, scoring="roc_auc")
    print(f"  AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    _report_pi_importance(ml_model, "ML")

    print("\nTraining run line model (run differential)...")
    rl_model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    rl_model.fit(X_scaled, y_diff)
    scores = cross_val_score(rl_model, X_scaled, y_diff, cv=tscv,
                             scoring="neg_mean_absolute_error")
    print(f"  MAE: {-scores.mean():.3f} +/- {scores.std():.3f}")
    _report_pi_importance(rl_model, "RunLine")

    print("\nTraining totals model (total runs)...")
    tot_model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    tot_model.fit(X_scaled, y_runs)
    scores = cross_val_score(tot_model, X_scaled, y_runs, cv=tscv,
                             scoring="neg_mean_absolute_error")
    print(f"  MAE: {-scores.mean():.3f} +/- {scores.std():.3f}")
    _report_pi_importance(tot_model, "Totals")

    joblib.dump({"ml": ml_model, "rl": rl_model, "totals": tot_model}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    meta = {
        "trained_at":          pd.Timestamp.now().isoformat(),
        "n_games":             len(df),
        "n_features":          len(FEATURE_COLS),
        "n_team_features":     len(TEAM_FEATURE_COLS),
        "n_pinnacle_features": len(PINNACLE_FEATURE_COLS),
        "pinnacle_features":   PINNACLE_FEATURE_COLS,
        "home_win_pct":        round(float(y_win.mean()), 3),
        "avg_total":           round(float(y_runs.mean()), 2),
    }
    with open(MODEL_DIR + "/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}/")
    print(f"  Home win pct:     {meta['home_win_pct']}")
    print(f"  Avg total runs:   {meta['avg_total']}")
    print(f"  Feature count:    {len(FEATURE_COLS)} "
          f"({len(TEAM_FEATURE_COLS)} team + {len(PINNACLE_FEATURE_COLS)} Pinnacle)")
    print("\nDone! Run next: python pull_lines.py")


def _report_pi_importance(model, label):
    """Print Pinnacle feature importances as a quick sanity check."""
    try:
        fi = model.feature_importances_
        pi_indices = [FEATURE_COLS.index(c) for c in PINNACLE_FEATURE_COLS
                      if c in FEATURE_COLS]
        total_pi_imp = sum(fi[i] for i in pi_indices)
        print(f"  Pinnacle feature importance share ({label}): {total_pi_imp:.1%}")
        for c in PINNACLE_FEATURE_COLS:
            if c in FEATURE_COLS:
                idx = FEATURE_COLS.index(c)
                print(f"    {c}: {fi[idx]:.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    train()