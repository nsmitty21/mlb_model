# MLB Edge — Betting Model

## Every Time You Sit Down to Work

### 1. Pull latest code first (always do this before anything else)
```bash
cd /c/Users/natha/mlb_model/mlb_model  (Wherever the proj is on ur PC)
git pull
```

### 2. Do your work — edit files, test things, etc.

### 3. Push your changes when done
```bash
git add .
git commit -m "describe what you changed"
git push
```

---

## Ktob Setting Up For The First Time
```bash
git clone https://github.com/nsmitty21/mlb_model.git
cd mlb_model
pip install -r requirements.txt --user
pip install flask-socketio python-dotenv schedule apscheduler xgboost --user
```

Then create a `.env` file in the folder with:
```
ODDS_API_KEY=your_key_here
FLASK_SECRET_KEY=mlb_edge_2026
```

Then run the historical data pull and train the model:
```bash
python fetch_historical.py
python train_model.py
```

---

## Daily Workflow (Once Season Starts March 27)
```bash
python pull_lines.py    # grab today's FanDuel + DraftKings lines
python run_model.py     # generate picks
python app.py           # launch dashboard at http://localhost:5000
```

The dashboard **↻ New Picks** button runs pull_lines + run_model automatically.
The nightly scheduler at 3 AM handles results, grading, retraining, and new picks automatically as long as app.py is running.

---

## Manual Commands
```bash
python fetch_results.py          # grade yesterday's bets manually
python train_model.py            # retrain model manually
python pull_lines.py --date 2026-03-28   # pull lines for a specific date
```

---

## Edge Thresholds (config.py)

| Units | Edge Required | Frequency |
|-------|--------------|-----------|
| 1u    | ≥ 3%        | Several/day |
| 2u    | ≥ 6%        | ~1/day    |
| 3u    | ≥ 9%        | Rare      |

---

## File Structure
```
mlb_model/
├── app.py                 # Dashboard + nightly scheduler
├── config.py              # All settings (API key, thresholds, paths)
├── fetch_historical.py    # One-time historical data pull (2021-2025)
├── pull_lines.py          # Daily line pull (FanDuel + DraftKings)
├── train_model.py         # Model training / retraining
├── run_model.py           # Generate picks from today's lines
├── fetch_results.py       # Nightly result fetch + bet grading
├── .env                   # API keys (never committed to GitHub)
├── requirements.txt
├── data/
│   ├── historical/        # 2021-2025 training data
│   ├── live/              # Current season data, today's lines/picks
│   └── logs/              # bet_log.csv
├── models/                # Trained model files
└── templates/
    └── dashboard.html     # Dashboard UI
```