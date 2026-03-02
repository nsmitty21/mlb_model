import os
from dotenv import load_dotenv
load_dotenv()

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY     = "baseball_mlb"
TARGET_BOOKS  = ["fanduel", "draftkings"]

SEASON_START = "2026-03-27"

EDGE_1_UNIT   = 3.0
EDGE_2_UNIT   = 6.0
EDGE_3_UNIT   = 9.0

TRAIN_YEARS   = [2021, 2022, 2023, 2024, 2025]
LIVE_SEASON   = 2026
RETRAIN_HOUR  = 3

HIST_DIR      = "data/historical"
LIVE_DIR      = "data/live"
LOGS_DIR      = "data/logs"
MODEL_DIR     = "models"
BET_LOG_CSV   = "data/logs/bet_log.csv"
MARKETS       = ["h2h", "spreads", "totals"]
