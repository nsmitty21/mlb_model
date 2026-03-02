"""
train_model.py - Train / Retrain the MLB Betting Model
Trains 3 XGBoost models: win probability, run differential, total runs
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

BATTING_FEATS  = ["OPS","wOBA","AVG","OBP","SLG","HR","BB%","K%","R"]
PITCHING_FEATS = ["ERA","FIP","WHIP","K/9","BB/9","xFIP"]

FEATURE_COLS = (
    [f"home_bat_{f}" for f in BATTING_FEATS] +
    [f"away_bat_{f}" for f in BATTING_FEATS] +
    [f"home_pit_{f}" for f in PITCHING_FEATS] +
    [f"away_pit_{f}" for f in PITCHING_FEATS]
)


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
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [warn] {len(missing)} features missing, filling with 0")
    for c in missing:
        df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return df


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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    # --- Moneyline model ---
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

    # --- Run diff model (run line) ---
    print("\nTraining run line model (run differential)...")
    rl_model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    rl_model.fit(X_scaled, y_diff)
    scores = cross_val_score(rl_model, X_scaled, y_diff, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  MAE: {-scores.mean():.3f} +/- {scores.std():.3f}")

    # --- Totals model ---
    print("\nTraining totals model (total runs)...")
    tot_model = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    tot_model.fit(X_scaled, y_runs)
    scores = cross_val_score(tot_model, X_scaled, y_runs, cv=tscv, scoring="neg_mean_absolute_error")
    print(f"  MAE: {-scores.mean():.3f} +/- {scores.std():.3f}")

    # --- Save ---
    joblib.dump({"ml": ml_model, "rl": rl_model, "totals": tot_model}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_COLS, f)

    meta = {
        "trained_at":   pd.Timestamp.now().isoformat(),
        "n_games":      len(df),
        "n_features":   len(FEATURE_COLS),
        "home_win_pct": round(float(y_win.mean()), 3),
        "avg_total":    round(float(y_runs.mean()), 2),
    }
    with open(MODEL_DIR + "/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}/")
    print(f"  Home win pct: {meta['home_win_pct']}")
    print(f"  Avg total runs: {meta['avg_total']}")
    print("\nDone! Run next: python pull_lines.py")


if __name__ == "__main__":
    train()