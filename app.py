"""
app.py - MLB Betting Model Dashboard
Run with: python app.py
Open: http://localhost:5000
"""

import os, json, threading, time
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request
import schedule
from dotenv import load_dotenv

load_dotenv()

from config import (
    BET_LOG_CSV, LIVE_DIR, LOGS_DIR, MODEL_DIR, RETRAIN_HOUR
)

PICKS_PATH = f"{LIVE_DIR}/picks_today.json"
LINES_PATH = f"{LIVE_DIR}/today_lines.csv"

Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(LIVE_DIR).mkdir(parents=True, exist_ok=True)
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "mlb_2026")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_picks():
    if not Path(PICKS_PATH).exists():
        return []
    try:
        with open(PICKS_PATH) as f:
            data = json.load(f)
        return data.get("picks", [])
    except Exception:
        return []


def load_bet_log():
    if not Path(BET_LOG_CSV).exists():
        return []
    try:
        df = pd.read_csv(BET_LOG_CSV)
        df = df.where(df.notna(), other=None)
        return df.to_dict("records")
    except Exception:
        return []


def save_bet_log(records):
    if not records:
        return
    pd.DataFrame(records).to_csv(BET_LOG_CSV, index=False)


def compute_stats(bet_log):
    if not bet_log:
        return {"total_bets":0,"wins":0,"losses":0,"pending":0,
                "win_pct":0,"total_units":0,"roi":0,"units_by_market":{},"streak":0}

    graded  = [b for b in bet_log if b.get("result") in ["W","L"]]
    pending = sum(1 for b in bet_log if b.get("result") not in ["W","L"])
    wins    = sum(1 for b in graded if b["result"] == "W")
    losses  = len(graded) - wins
    total_pnl     = sum(float(b.get("pnl_units", 0) or 0) for b in graded)
    total_wagered = sum(float(b.get("units", 1) or 1) for b in graded)
    roi = (total_pnl / total_wagered * 100) if total_wagered else 0

    streak = 0
    if graded:
        last = graded[-1]["result"]
        for b in reversed(graded):
            if b["result"] == last:
                streak += 1
            else:
                break
        if last == "L":
            streak = -streak

    by_market = {}
    for b in graded:
        m = b.get("market", "unknown")
        if m not in by_market:
            by_market[m] = {"wins":0,"losses":0,"pnl":0}
        by_market[m]["wins" if b["result"]=="W" else "losses"] += 1
        by_market[m]["pnl"] += float(b.get("pnl_units", 0) or 0)

    return {
        "total_bets":      len(bet_log),
        "wins":            wins,
        "losses":          losses,
        "pending":         pending,
        "win_pct":         round(wins / len(graded) * 100, 1) if graded else 0,
        "total_units":     round(total_pnl, 2),
        "roi":             round(roi, 1),
        "units_by_market": by_market,
        "streak":          streak,
    }


def load_model_meta():
    path = f"{MODEL_DIR}/meta.json"
    if not Path(path).exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/picks")
def api_picks():
    return jsonify(load_picks())


@app.route("/api/betlog")
def api_betlog():
    return jsonify(load_bet_log())


@app.route("/api/stats")
def api_stats():
    return jsonify(compute_stats(load_bet_log()))


@app.route("/api/meta")
def api_meta():
    meta = load_model_meta()
    picks = load_picks()
    meta["picks_today"] = len(picks)
    meta["picks_date"]  = date.today().isoformat()
    if Path(PICKS_PATH).exists():
        with open(PICKS_PATH) as f:
            d = json.load(f)
        meta["generated_at"] = d.get("generated_at", "")
    return jsonify(meta)


@app.route("/api/log_bet", methods=["POST"])
def log_bet():
    data = request.json
    bet_log = load_bet_log()
    for b in bet_log:
        if (b.get("game_id") == data.get("game_id") and
            b.get("market")  == data.get("market")  and
            b.get("side")    == data.get("side")):
            return jsonify({"status": "duplicate"})
    data["logged_at"] = datetime.now().isoformat()
    data["result"]    = "pending"
    data["pnl_units"] = None
    bet_log.append(data)
    save_bet_log(bet_log)
    return jsonify({"status": "ok"})


@app.route("/api/delete_bet/<int:idx>", methods=["DELETE"])
def delete_bet(idx):
    bet_log = load_bet_log()
    if 0 <= idx < len(bet_log):
        bet_log.pop(idx)
        save_bet_log(bet_log)
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 404


@app.route("/api/refresh_picks", methods=["POST"])
def refresh_picks():
    def run():
        import subprocess
        subprocess.run(["python", "pull_lines.py"], capture_output=True)
        subprocess.run(["python", "run_model.py"],  capture_output=True)
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "running"})


@app.route("/api/refresh_results", methods=["POST"])
def refresh_results():
    def run():
        import fetch_results
        fetch_results.run()
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "running"})


# ── Scheduler ─────────────────────────────────────────────────────────────────

def nightly_job():
    print(f"\n[{datetime.now()}] Nightly job starting...")
    import fetch_results, subprocess
    fetch_results.run()
    try:
        import train_model
        train_model.train()
        print("Model retrained")
    except Exception as e:
        print(f"[warn] Retraining failed: {e}")
    subprocess.run(["python", "pull_lines.py"], capture_output=True)
    subprocess.run(["python", "run_model.py"],  capture_output=True)
    print(f"[{datetime.now()}] Nightly job complete.")


def run_scheduler():
    schedule.every().day.at(f"{RETRAIN_HOUR:02d}:00").do(nightly_job)
    while True:
        schedule.run_pending()
        time.sleep(60)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=run_scheduler, daemon=True).start()
    print("\n" + "="*50)
    print("  MLB Edge Dashboard")
    print("  http://localhost:5000")
    print(f"  Nightly retrain at {RETRAIN_HOUR}:00 AM ET")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)