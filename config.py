import os
from dotenv import load_dotenv
load_dotenv()

# ── Auto-detect machine ───────────────────────────────────────────────────────
USER = os.getenv("USERNAME", os.getenv("USER", "")).lower()

if "kctob" in USER:
    BASE_DIR = r"C:\Users\Kctob\OneDrive\Documents\MLB Model Files"
elif "natha" in USER:
    BASE_DIR = r"C:\Users\natha\mlb_model\mlb_model"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── API Keys ──────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "baseball_mlb"
TARGET_BOOKS  = ["fanduel", "draftkings"]

# ── Edge Thresholds ───────────────────────────────────────────────────────────
# Backtest sweep found 7% is the ROI-optimal entry point (see backtest_report.txt).
# run_model.py enforces a hard floor of 15% for 3U regardless of what's set here,
# because a true mismatch edge that big should be rare — if the model is outputting
# 3U plays on half the slate, it's almost certainly a data problem not an edge.
EDGE_1_UNIT  = 7.0    # >= 7% edge  -> 1 unit  (raised from 5% per backtest)
EDGE_2_UNIT  = 10.0   # >= 10% edge -> 2 units  (raised from 7%)
EDGE_3_UNIT  = 15.0   # >= 15% edge -> 3 units  (raised from 10%; hard floor in run_model.py)

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_YEARS  = [2021, 2022, 2023, 2024, 2025]
LIVE_SEASON  = 2026
RETRAIN_HOUR = 3

# ── Paths ─────────────────────────────────────────────────────────────────────
HIST_DIR    = os.path.join(BASE_DIR, "data", "historical")
LIVE_DIR    = os.path.join(BASE_DIR, "data", "live")
LOGS_DIR    = os.path.join(BASE_DIR, "data", "logs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
BET_LOG_CSV = os.path.join(BASE_DIR, "data", "logs", "bet_log.csv")
MARKETS     = ["h2h", "spreads", "totals"]

# ── Aliases — all scripts import from here ────────────────────────────────────
DATA_DIR             = LIVE_DIR
MODELS_DIR           = MODEL_DIR
MODEL_PATH           = os.path.join(MODEL_DIR, "mlb_model.joblib")
LINES_PATH           = os.path.join(LIVE_DIR, "today_lines.json")
PICKS_PATH           = os.path.join(LIVE_DIR, "picks_today.json")
SEASON_PATH          = os.path.join(LIVE_DIR, "season_2026.parquet")
BET_LOG_PATH         = BET_LOG_CSV
EDGE_1U              = EDGE_1_UNIT
EDGE_2U              = EDGE_2_UNIT
EDGE_3U              = EDGE_3_UNIT

# ── Home-field advantage ───────────────────────────────────────────────────────
# NOTE: This constant is used for DISPLAY purposes only (dashboard, reports).
# Do NOT apply it before comparing model_prob to book_prob in edge calculations.
# The win model was trained with home_field=1 as a feature, so home advantage
# is already embedded in the model's predicted probabilities.  Adding this boost
# pre-edge would double-count it and systematically over-rate away teams.
HOME_FIELD_WIN_PCT_BOOST = 0.02

# ── Kyle's historical odds CSVs ───────────────────────────────────────────────
HIST_ODDS_DIR = os.path.join(HIST_DIR, "odds")

# ── Create dirs ───────────────────────────────────────────────────────────────
for d in [HIST_DIR, LIVE_DIR, LOGS_DIR, MODEL_DIR, HIST_ODDS_DIR]:
    os.makedirs(d, exist_ok=True)
