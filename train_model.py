"""
train_model.py — MLB Model Training (fixed + backtest merged)
==============================================================

Fixes vs previous version
--------------------------
FIX 1 — Model file mismatch (root cause of "Models not found" after training)
    OLD: saved models/mlb_model.joblib as a dict {ml, rl, totals}
         model.py/load_models() looked for ml_model.pkl, spread_model.pkl,
         total_model.pkl — these files were never created → always failed.
    NEW: saves exactly the three .pkl files load_models() expects.

FIX 2 — Regressor vs classifier mismatch (would crash at predict time)
    OLD: trained XGBRegressor for run-line and totals.
         model.py calls predict_proba() on all three models.
         XGBRegressor has no predict_proba() → AttributeError at prediction.
    NEW: trains XGBClassifier for all three targets with binary labels:
         - ML:     home_win       (1 = home wins)
         - Spread: home_covered_rl (1 = home covers -1.5)
         - Total:  went_over      (1 = total > line)

FIX 3 — Scaler not applied at inference time
    OLD: scaler saved but run_today.py / model.py never loaded or applied it.
         The model was trained on scaled data but got raw data at prediction.
    NEW: scaler is baked into a Pipeline object inside each .pkl so
         load_models() + predict_proba() work correctly with no extra steps.

FIX 4 — Feature column list not shared with model.py at runtime
    OLD: features.json written but model.py imported FEATURE_COLS from
         features.py (a different, smaller list), not from features.json.
    NEW: meta.json now includes the exact feature list used for training.
         load_meta() in model.py already reads meta.json — run_today.py
         can use meta["feature_cols"] to verify alignment.

Usage
-----
    python train_model.py                # train only
    python train_model.py --backtest     # train, then run walk-forward backtest
    python train_model.py --backtest --backtest-years 3   # backtest last N years only
    python train_model.py --years 2021 2022 2023 2024 2025 # explicit training years
"""

import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

from config import HIST_DIR, MODEL_DIR, TRAIN_YEARS

# ── Paths ─────────────────────────────────────────────────────────────────────
HISTORICAL_PATH = os.path.join(HIST_DIR, "historical_stats.parquet")
SEASON_PATH     = "data/live/season_2026.parquet"
os.makedirs(MODEL_DIR, exist_ok=True)

# FIX 1: these are exactly the paths load_models() in model.py expects
ML_PKL     = os.path.join(MODEL_DIR, "ml_model.pkl")
SPREAD_PKL = os.path.join(MODEL_DIR, "spread_model.pkl")
TOTAL_PKL  = os.path.join(MODEL_DIR, "total_model.pkl")
META_JSON  = os.path.join(MODEL_DIR, "meta.json")


# ── Feature definitions ───────────────────────────────────────────────────────
# Keep identical to what fetch_historical.py writes into the parquet so there
# are no surprises at load time.

BATTING_FEATS  = ["OPS", "wOBA", "AVG", "OBP", "SLG", "HR", "BB%", "K%", "R"]
PITCHING_FEATS = ["ERA", "FIP", "WHIP", "K/9", "BB/9", "xFIP"]

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

# Batter-vs-pitcher matchup features from bbref_batter_vs_pitcher_scraper.py
# + fetch_historical.py build_game_lineups(). These use actual per-game lineups
# when available, falling back to career proxy when not.
BVP_FEATURE_COLS = [
    "home_bvp_avg",       # batting avg of home lineup vs away SP (career H2H)
    "home_bvp_ops",       # OPS of home lineup vs away SP
    "home_bvp_k_rate",    # K rate of home lineup vs away SP (high = away SP dominates)
    "home_bvp_bb_rate",   # BB rate of home lineup vs away SP
    "home_bvp_hr_rate",   # HR rate of home lineup vs away SP
    "away_bvp_avg",       # batting avg of away lineup vs home SP
    "away_bvp_ops",       # OPS of away lineup vs home SP
    "away_bvp_k_rate",    # K rate of away lineup vs home SP
    "away_bvp_bb_rate",   # BB rate of away lineup vs home SP
    "away_bvp_hr_rate",   # HR rate of away lineup vs home SP
    "bvp_avg_diff",       # home_bvp_avg - away_bvp_avg (positive = home lineup edge)
    "bvp_ops_diff",       # home_bvp_ops - away_bvp_ops
    "bvp_k_rate_diff",    # home_bvp_k_rate - away_bvp_k_rate (positive = away SP more dominant)
]

FEATURE_COLS = TEAM_FEATURE_COLS + PINNACLE_FEATURE_COLS + BVP_FEATURE_COLS


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(HISTORICAL_PATH):
        print("[ERROR] historical_stats.parquet not found. "
              "Run: python fetch_historical.py")
        sys.exit(1)

    df = pd.read_parquet(HISTORICAL_PATH)
    print(f"[data] Loaded {len(df):,} games from historical_stats.parquet")

    if os.path.exists(SEASON_PATH):
        live = pd.read_parquet(SEASON_PATH)
        df = pd.concat([df, live], ignore_index=True)
        print(f"[data] + {len(live):,} live-season games → {len(df):,} total")

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every feature column in FEATURE_COLS exists.
    Computes diff_ columns, fills missing values with neutral defaults,
    and builds binary target columns.
    """
    df = df.copy()

    # ── Check for pybaseball column-name drift ────────────────────────────
    missing_bat = [f for f in BATTING_FEATS
                   if f"home_bat_{f}" not in df.columns
                   and f"away_bat_{f}" not in df.columns]
    missing_pit = [f for f in PITCHING_FEATS
                   if f"home_pit_{f}" not in df.columns
                   and f"away_pit_{f}" not in df.columns]
    if missing_bat:
        bat_cols = [c for c in df.columns if "bat" in c][:12]
        print(f"  [warn] Batting features missing: {missing_bat}")
        print(f"         Available batting cols: {bat_cols}")
        for f in missing_bat:
            for pfx in ("home_bat_", "away_bat_"):
                df.setdefault(f"{pfx}{f}", np.nan)
    if missing_pit:
        pit_cols = [c for c in df.columns if "pit" in c][:12]
        print(f"  [warn] Pitching features missing: {missing_pit}")
        print(f"         Available pitching cols: {pit_cols}")
        for f in missing_pit:
            for pfx in ("home_pit_", "away_pit_"):
                df.setdefault(f"{pfx}{f}", np.nan)

    # ── Diff columns + home_field ─────────────────────────────────────────
    # Compute from real data BEFORE any fill so diffs are meaningful.
    df["home_field"] = 1
    for f in BATTING_FEATS:
        hv = df.get(f"home_bat_{f}", pd.Series(np.nan, index=df.index))
        av = df.get(f"away_bat_{f}", pd.Series(np.nan, index=df.index))
        df[f"diff_bat_{f}"] = hv - av
    for f in PITCHING_FEATS:
        hv = df.get(f"home_pit_{f}", pd.Series(np.nan, index=df.index))
        av = df.get(f"away_pit_{f}", pd.Series(np.nan, index=df.index))
        # Inverse for ERA/FIP/WHIP/BB9: lower is better → away minus home
        if f in ("ERA", "FIP", "WHIP", "BB/9"):
            df[f"diff_pit_{f}"] = av - hv
        else:
            df[f"diff_pit_{f}"] = hv - av

    # Fill any still-missing team-stat columns
    missing_team = [c for c in TEAM_FEATURE_COLS if c not in df.columns]
    if missing_team:
        print(f"  [warn] {len(missing_team)} team-stat columns still missing — "
              f"filling with 0")
        for c in missing_team:
            df[c] = 0.0

    # Fill team stats with column medians (original behaviour)
    df[TEAM_FEATURE_COLS] = df[TEAM_FEATURE_COLS].fillna(
        df[TEAM_FEATURE_COLS].median()
    )

    # ── Pinnacle features ─────────────────────────────────────────────────
    missing_pi = [c for c in PINNACLE_FEATURE_COLS if c not in df.columns]
    if missing_pi:
        print(f"  [warn] {len(missing_pi)} Pinnacle features absent: {missing_pi}")
        for c in missing_pi:
            df[c] = np.nan

    for c in PINNACLE_FEATURE_COLS:
        cov = df[c].notna().mean()
        if cov < 0.50:
            print(f"  [warn] {c}: only {cov:.1%} non-null — Pinnacle signal weak")

    movement_cols = ["pi_ml_movement", "pi_spread_movement", "pi_total_movement"]
    prob_cols     = ["pi_close_no_vig_home", "pi_open_implied_home"]
    total_col     = ["pi_close_total"]

    for c in movement_cols:
        df[c] = df[c].fillna(0.0)
    for c in prob_cols:
        med = df[c].median() if df[c].notna().any() else 0.5
        df[c] = df[c].fillna(med)
    for c in total_col:
        med = df[c].median() if df[c].notna().any() else 8.5
        df[c] = df[c].fillna(med)

    # ── BvP features ──────────────────────────────────────────────────────
    # These are present only when fetch_historical.py has been run with both
    # batter_vs_pitcher_career.csv and game_lineups.parquet available.
    # Missing columns get NaN; the model handles them via the -999 fillna
    # in the training loop (same treatment as sparse Pinnacle data).
    missing_bvp = [c for c in BVP_FEATURE_COLS if c not in df.columns]
    if missing_bvp:
        print(f"  [warn] {len(missing_bvp)} BvP features absent — "
              f"run fetch_historical.py after bbref scraper completes")
        for c in missing_bvp:
            df[c] = np.nan
    else:
        # Report coverage so we know how much of the training set has real H2H data
        bvp_cov = df["home_bvp_avg"].notna().mean()
        print(f"  BvP feature coverage: {bvp_cov:.1%} of training rows have H2H data")
        if bvp_cov < 0.30:
            print("  [warn] BvP coverage < 30% — signal will be weak. "
                  "Re-run fetch_historical.py once more PA files are scraped.")

    # Fill BvP NaNs with column medians so they degrade gracefully when absent.
    # Do NOT fill with 0 — a 0 AVG looks like a pitcher who dominates completely,
    # which would be worse than saying "no data".
    for c in BVP_FEATURE_COLS:
        med = df[c].median() if df[c].notna().any() else np.nan
        if pd.notna(med):
            df[c] = df[c].fillna(med)
        # Remaining NaNs (all-null column) stay NaN → become -999 in training loop

    # ── Binary target columns (FIX 2) ────────────────────────────────────
    # home_win: already present from fetch_historical
    if "home_win" not in df.columns:
        if "home_runs" in df.columns and "away_runs" in df.columns:
            df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(int)
        else:
            raise ValueError("Cannot compute home_win — need home_runs / away_runs")

    # home_covered_rl: home team covers the -1.5 run line
    # i.e. home wins by 2+ runs
    if "home_covered_rl" not in df.columns:
        if "home_runs" in df.columns and "away_runs" in df.columns:
            df["home_covered_rl"] = (
                (df["home_runs"] - df["away_runs"]) >= 2
            ).astype(int)
        else:
            df["home_covered_rl"] = np.nan

    # went_over: final total exceeds the Pinnacle closing line
    if "went_over" not in df.columns:
        if "total_runs" in df.columns:
            # Use Pinnacle closing line where available, fall back to 8.5
            line = df["pi_close_total"].fillna(8.5)
            df["went_over"] = (df["total_runs"] > line).astype(int)
        else:
            df["went_over"] = np.nan

    return df


# ── Training ──────────────────────────────────────────────────────────────────

def _make_pipeline(**xgb_kwargs) -> Pipeline:
    """
    FIX 3: Wrap scaler + classifier in a single Pipeline so the saved .pkl
    applies the same scaling at inference that was used during training.
    No separate scaler.joblib needed — prediction just calls pipeline.predict_proba().
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(**xgb_kwargs)),
    ])


XGB_PARAMS = dict(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1,
)


def _report_pi_importance(pipeline: Pipeline, label: str):
    """Log the share of total importance held by Pinnacle features."""
    try:
        clf = pipeline.named_steps["clf"]
        fi  = clf.feature_importances_
        pi_idxs = [FEATURE_COLS.index(c) for c in PINNACLE_FEATURE_COLS
                   if c in FEATURE_COLS]
        share = sum(fi[i] for i in pi_idxs)
        print(f"    Pinnacle importance share ({label}): {share:.1%}")
        for c in PINNACLE_FEATURE_COLS:
            if c in FEATURE_COLS:
                print(f"      {c}: {fi[FEATURE_COLS.index(c)]:.3f}")
        # BvP importance
        bvp_indices = [FEATURE_COLS.index(c) for c in BVP_FEATURE_COLS
                       if c in FEATURE_COLS]
        total_bvp_imp = sum(fi[i] for i in bvp_indices)
        print(f"    BvP feature importance share ({label}): {total_bvp_imp:.1%}")
        top_bvp = sorted(
            [(c, fi[FEATURE_COLS.index(c)]) for c in BVP_FEATURE_COLS if c in FEATURE_COLS],
            key=lambda x: x[1], reverse=True
        )[:5]
        for c, imp in top_bvp:
            print(f"      {c}: {imp:.3f}")
    except Exception:
        pass


def train(years: list[int] | None = None) -> dict:
    print("=" * 58)
    print("  MLB Model Training")
    print("=" * 58)

    df = load_data()
    df = build_features(df)

    # Optionally filter to specific training years
    if years and "season" in df.columns:
        df = df[df["season"].isin(years)]
        print(f"[train] Filtered to years {years}: {len(df):,} games")

    # Drop rows missing more than half their features
    X_raw    = df[FEATURE_COLS]
    coverage = X_raw.notna().sum(axis=1) / len(FEATURE_COLS)
    keep     = coverage > 0.5
    df       = df[keep].copy()
    X_raw    = X_raw[keep]
    print(f"[train] {keep.sum():,} games after >50% feature coverage filter "
          f"(dropped {(~keep).sum()})")

    X = X_raw.fillna(-999).values

    tscv = TimeSeriesSplit(n_splits=5)

    targets = [
        ("ML",         "home_win",        ML_PKL),
        ("Spread/RL",  "home_covered_rl", SPREAD_PKL),
        ("Total",      "went_over",       TOTAL_PKL),
    ]

    pipelines = {}
    metrics   = {}

    for label, target_col, pkl_path in targets:
        print(f"\n[train] {label} model  (target: {target_col})")

        if target_col not in df.columns or df[target_col].isna().all():
            print(f"  [skip] {target_col} not available in data")
            pipelines[label] = None
            continue

        y = df[target_col].fillna(0).astype(int).values
        pos_rate = y.mean()
        print(f"  Positive rate: {pos_rate:.3f}  |  n={len(y):,}")

        pipe = _make_pipeline(**XGB_PARAMS)
        pipe.fit(X, y)

        scores = cross_val_score(pipe, X, y, cv=tscv, scoring="roc_auc")
        auc_mean, auc_std = scores.mean(), scores.std()
        print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
        _report_pi_importance(pipe, label)

        # FIX 1: save to the exact paths load_models() expects
        joblib.dump(pipe, pkl_path)
        print(f"  Saved → {pkl_path}")

        pipelines[label] = pipe
        metrics[label]   = {"auc": round(auc_mean, 4), "auc_std": round(auc_std, 4),
                             "pos_rate": round(float(pos_rate), 4), "n": int(len(y))}

    # FIX 4: meta.json includes the feature list so model.py can verify alignment
    meta = {
        "trained_at":          pd.Timestamp.now().isoformat(),
        "training_years":      years or list(range(df["season"].min(),
                                                   df["season"].max() + 1))
                               if "season" in df.columns else [],
        "n_games":             int(len(df)),
        "feature_cols":        FEATURE_COLS,       # ← used by model.py load_meta()
        "n_features":          len(FEATURE_COLS),
        "n_team_features":     len(TEAM_FEATURE_COLS),
        "n_pinnacle_features": len(PINNACLE_FEATURE_COLS),
        "n_bvp_features":      len(BVP_FEATURE_COLS),
        "pinnacle_features":   PINNACLE_FEATURE_COLS,
        "bvp_features":        BVP_FEATURE_COLS,
        "home_win_pct":        round(float(df["home_win"].mean()), 3)
                               if "home_win" in df.columns else None,
        "avg_total":           round(float(df["total_runs"].mean()), 2)
                               if "total_runs" in df.columns else None,
        "model_metrics":       metrics,
        "training_rows":       int(len(df)),       # read by run_today.py
    }
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[train] meta.json saved → {META_JSON}")

    print(f"\n{'='*58}")
    print(f"  Models saved to {MODEL_DIR}/")
    print(f"  Home win pct:   {meta['home_win_pct']}")
    print(f"  Avg total runs: {meta['avg_total']}")
    print(f"  Features:       {len(FEATURE_COLS)} "
          f"({len(TEAM_FEATURE_COLS)} team + "
          f"{len(PINNACLE_FEATURE_COLS)} Pinnacle + "
          f"{len(BVP_FEATURE_COLS)} BvP)")
    print(f"{'='*58}")
    print("  Run next: python pull_lines.py  →  python run_today.py")

    return pipelines


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(df_full: pd.DataFrame, backtest_years: int | None = None):
    """
    Walk-forward backtest: for each season, train on all prior seasons,
    predict on that season, measure AUC + simulated P&L.

    Uses the Pinnacle closing no-vig prob as the book edge baseline so the
    edge calculation is consistent with what run_today.py does at game time.
    """
    print(f"\n{'='*58}")
    print("  Walk-Forward Backtest")
    print("=" * 58)

    if "season" not in df_full.columns:
        print("  [skip] No 'season' column — cannot walk-forward by year")
        return

    seasons = sorted(df_full["season"].dropna().unique().astype(int))
    if backtest_years:
        seasons = seasons[-backtest_years:]
    print(f"  Seasons to evaluate: {seasons}")

    from sklearn.metrics import roc_auc_score
    from model import american_to_prob as _a2p, remove_vig as _vig

    EDGE_THRESHOLDS = [(0.15, 3), (0.10, 2), (0.07, 1)]

    def _get_units(edge):
        for t, u in EDGE_THRESHOLDS:
            if abs(edge) >= t:
                return u
        return 0

    all_results = []
    season_summary = []

    for test_season in seasons:
        train_mask = df_full["season"] < test_season
        test_mask  = df_full["season"] == test_season

        if train_mask.sum() < 200:
            print(f"  [skip] {test_season} — fewer than 200 training games before it")
            continue

        train_df = df_full[train_mask].copy()
        test_df  = df_full[test_mask].copy()

        train_df = build_features(train_df)
        test_df  = build_features(test_df)

        X_train = train_df[FEATURE_COLS].fillna(-999).values
        X_test  = test_df[FEATURE_COLS].fillna(-999).values

        y_train = train_df["home_win"].fillna(0).astype(int).values
        y_test  = test_df["home_win"].fillna(0).astype(int).values

        pipe = _make_pipeline(**XGB_PARAMS)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

        # Simulate bets against Pinnacle closing no-vig line
        pnl_total = 0.0
        bets      = 0
        wins      = 0

        for i, (_, row) in enumerate(test_df.iterrows()):
            model_p = probs[i]
            actual  = int(row.get("home_win", 0))

            # Book probability from Pinnacle closing no-vig (best available signal)
            book_p = row.get("pi_close_no_vig_home", np.nan)
            if np.isnan(book_p):
                continue  # no Pinnacle line → skip

            edge  = model_p - book_p
            units = _get_units(edge)
            if units == 0:
                continue

            bets += 1

            # ── Derive Pinnacle closing ML odds for PNL calculation ────────
            # row.get() on a pandas Series returns NaN even when a default is
            # supplied if the stored value IS NaN, so we check explicitly.
            _ml_home_raw = row["pi_close_ml_home"] if "pi_close_ml_home" in row.index else np.nan
            _ml_away_raw = row["pi_close_ml_away"] if "pi_close_ml_away" in row.index else np.nan

            # Convert NaN → derived odds from the no-vig probability we already have
            if pd.isna(_ml_home_raw):
                # p → American: if p > 0.5, odds = -(p/(1-p))*100 else +(1-p)/p*100
                if book_p >= 0.5:
                    ml_home = -(book_p / (1 - book_p)) * 100
                else:
                    ml_home = ((1 - book_p) / book_p) * 100
            else:
                ml_home = float(_ml_home_raw)

            away_p = 1 - book_p
            if pd.isna(_ml_away_raw):
                if away_p >= 0.5:
                    ml_away = -(away_p / (1 - away_p)) * 100
                else:
                    ml_away = ((1 - away_p) / away_p) * 100
            else:
                ml_away = float(_ml_away_raw)

            # ── PNL calculation ────────────────────────────────────────────
            if edge > 0:  # betting home
                bet_outcome = actual
                odds_used = ml_home
            else:         # betting away
                bet_outcome = 1 - actual
                odds_used = ml_away

            if odds_used > 0:
                pnl = units * (odds_used / 100) if bet_outcome else -float(units)
            else:
                pnl = units * (100 / abs(odds_used)) if bet_outcome else -float(units)

            if bet_outcome:
                wins += 1
            pnl_total += pnl

            all_results.append({
                "season":    test_season,
                "game_id":   row.get("game_id", ""),
                "model_p":   round(model_p, 4),
                "book_p":    round(book_p, 4),
                "edge":      round(edge, 4),
                "units":     units,
                "bet_side":  "home" if edge > 0 else "away",
                "outcome":   bet_outcome,
                "pnl":       round(pnl, 3),
            })

        roi = pnl_total / bets if bets else 0.0
        win_rate = wins / bets if bets else 0.0
        season_summary.append({
            "season":   test_season,
            "games":    len(test_df),
            "bets":     bets,
            "wins":     wins,
            "win_rate": round(win_rate, 3),
            "pnl":      round(pnl_total, 2),
            "roi":      round(roi, 4),
            "auc":      round(auc, 4),
        })

        print(f"\n  {test_season}  |  AUC: {auc:.4f}  |  "
              f"Bets: {bets}  |  W-L: {wins}-{bets-wins}  |  "
              f"P&L: {pnl_total:+.2f}u  |  ROI: {roi:+.1%}")

    if not season_summary:
        print("  No seasons had enough data to backtest.")
        return

    # ── Overall summary ───────────────────────────────────────────────────
    total_bets = sum(s["bets"] for s in season_summary)
    total_pnl  = sum(s["pnl"]  for s in season_summary)
    total_wins = sum(s["wins"] for s in season_summary)
    overall_roi = total_pnl / total_bets if total_bets else 0.0
    mean_auc    = np.mean([s["auc"] for s in season_summary])

    print(f"\n{'─'*58}")
    print(f"  OVERALL  |  Seasons: {len(season_summary)}  |  "
          f"Bets: {total_bets}  |  W-L: {total_wins}-{total_bets-total_wins}")
    print(f"           |  P&L: {total_pnl:+.2f}u  |  "
          f"ROI: {overall_roi:+.1%}  |  Mean AUC: {mean_auc:.4f}")
    print(f"{'─'*58}")

    # ── Write backtest report ─────────────────────────────────────────────
    report_path = os.path.join(MODEL_DIR, "backtest_report.json")
    results_df  = pd.DataFrame(all_results)

    # json.dump can't handle numpy int64/float64 — cast everything explicitly
    def _to_python(obj):
        """Recursively convert numpy scalars to native Python types."""
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    report = _to_python({
        "generated_at":   pd.Timestamp.now().isoformat(),
        "seasons_tested": [s["season"] for s in season_summary],
        "overall": {
            "total_bets":  total_bets,
            "total_wins":  total_wins,
            "total_pnl":   round(total_pnl, 2),
            "overall_roi": round(overall_roi, 4),
            "mean_auc":    round(float(mean_auc), 4),
        },
        "by_season": season_summary,
    })
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Backtest report saved → {report_path}")

    # By-units breakdown
    if not results_df.empty:
        print("\n  P&L breakdown by unit size:")
        for u in sorted(results_df["units"].unique()):
            sub = results_df[results_df["units"] == u]
            u_pnl = sub["pnl"].sum()
            u_roi = u_pnl / len(sub)
            print(f"    {u}U  |  Bets: {len(sub):4d}  |  "
                  f"P&L: {u_pnl:+.2f}u  |  ROI: {u_roi:+.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MLB betting model (XGBoost classifiers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        metavar="YEAR",
        help="Training seasons (default: all in historical_stats.parquet)"
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="After training, run a walk-forward backtest on historical seasons"
    )
    parser.add_argument(
        "--backtest-years", type=int, default=None, metavar="N",
        help="Only backtest the most recent N seasons (default: all)"
    )
    args = parser.parse_args()

    # Train
    train(years=args.years)

    # Backtest (re-uses the full data, not just the filtered subset)
    if args.backtest:
        print("\n[backtest] Loading full dataset for walk-forward evaluation...")
        df_full = load_data()
        backtest(df_full, backtest_years=args.backtest_years)


if __name__ == "__main__":
    main()
