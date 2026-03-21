import os
from dotenv import load_dotenv
load_dotenv()

# ── Auto-detect machine ───────────────────────────────────────────────────────
USER = os.getenv("USERNAME", os.getenv("USER", "")).lower()

if "kctob" in USER:
    BASE_DIR      = r"C:\Users\Kctob\OneDrive\Documents\MLB Model Files"
    HIST_ODDS_DIR = r"C:\Users\Kctob\OneDrive\Documents\NBA Model Files\Historical Odds"
elif "natha" in USER:
    BASE_DIR      = r"C:\Users\natha\mlb_model\mlb_model"
    HIST_ODDS_DIR = os.path.join(r"C:\Users\natha\mlb_model\mlb_model", "historical_odds")
else:
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    HIST_ODDS_DIR = os.path.join(BASE_DIR, "historical_odds")

# ── API Keys ──────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "baseball_mlb"
TARGET_BOOKS  = ["fanduel", "draftkings"]

# ── Edge Thresholds (tuned from backtest) ─────────────────────────────────────
# Backtest showed 7% is ROI-optimal sweet spot
# Moneyline ROI +35%, Totals +15%, Run Line +3.4%
EDGE_1_UNIT  = 5.0    # >= 5% edge  -> 1 unit  (was 3.0)
EDGE_2_UNIT  = 7.0    # >= 7% edge  -> 2 units (was 6.0, optimal per sweep)
EDGE_3_UNIT  = 10.0   # >= 10% edge -> 3 units (was 9.0, rare elite plays)

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

# ── Create dirs ───────────────────────────────────────────────────────────────
for d in [HIST_DIR, LIVE_DIR, LOGS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)
