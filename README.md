# MLB Edge — Betting Model

A machine-learning MLB betting model with a live dashboard. Analyzes **Moneyline, Run Line, and Totals** across FanDuel and DraftKings. Picks only suggested when edge threshold is met. Retrains nightly.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Odds API key
Edit `.env`:
```
ODDS_API_KEY=your_actual_key_here
```

### 3. Pull historical data (run once to bootstrap)
```bash
python fetch_historical.py
```
This pulls 2021–2025 game logs (ESPN), pitcher stats, and batter stats (FanGraphs via pybaseball). Takes ~5–10 minutes.

### 4. Train the model
```bash
python train_model.py
```
Trains three gradient-boosted models: Moneyline, Run Line, Totals.

### 5. Pull today's lines
```bash
python pull_lines.py
```

### 6. Generate picks
```bash
python run_model.py
```

### 7. Launch dashboard
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## Daily Workflow (once set up)

The dashboard has a **↻ New Picks** button that runs the full pipeline.  
The scheduler runs **nightly at 3 AM ET** automatically:
1. Fetches yesterday's results from ESPN
2. Grades pending bets, updates P&L
3. Retrains model with new data
4. Pulls today's lines
5. Generates new picks

You can also run manually:
```bash
python pull_lines.py   # get today's lines
python run_model.py    # generate picks
python fetch_results.py  # grade yesterday's bets
```

---

## Edge Thresholds

| Units | Edge Required | Frequency |
|-------|--------------|-----------|
| 1u    | ≥ 3%        | Several/day |
| 2u    | ≥ 6%        | ~1/day    |
| 3u    | ≥ 9%        | Rare      |

---

## Model Architecture

- **Moneyline:** Gradient Boosted Classifier → calibrated win probability
- **Run Line:** Gradient Boosted Classifier → calibrated cover probability  
- **Totals:** Gradient Boosted Regressor → predicted total runs → over/under probability

### Key Features
- Rolling 10/30-game team form (runs scored, allowed, differential, win%)
- Home field advantage per team
- Starting pitcher: ERA, FIP, xFIP, WHIP, K/9, BB/9, HR/9, WAR
- Pitcher differential (home SP vs away SP)
- Season context (avg total runs)

### Learning Loop
Every night after results come in:
1. Game result → updates team rolling form
2. New data appended to `data/live/season_results.csv`
3. Model retrains on all data (2021–2025 historical + 2026 live season)
4. Gets sharper as the season progresses

---

## File Structure

```
mlb_model/
├── app.py                  # Flask dashboard + scheduler
├── config.py               # All settings (API key, thresholds, paths)
├── fetch_historical.py     # One-time historical data pull
├── pull_lines.py           # Daily line pull (FanDuel + DraftKings)
├── train_model.py          # Model training / retraining
├── run_model.py            # Generate picks from today's lines
├── fetch_results.py        # Nightly result fetch + bet grading
├── .env                    # API keys (not committed)
├── requirements.txt
├── data/
│   ├── historical/         # 2021–2025 training data
│   ├── live/               # Current season data, today's lines/picks
│   └── logs/               # bet_log.csv
├── models/                 # Trained model files (.pkl)
└── templates/
    └── dashboard.html      # Dashboard UI
```

---

## Changing Edge Thresholds

Edit `config.py`:
```python
EDGE_1_UNIT = 3.0   # percent
EDGE_2_UNIT = 6.0
EDGE_3_UNIT = 9.0
```

---

## Adding Starting Pitcher Data

The model uses starting pitcher stats from FanGraphs (via pybaseball) for historical seasons. For 2026 live games, you can manually add pitchers to today's lines CSV:

Edit `data/live/today_lines.csv` and add columns:
```
home_pitcher,away_pitcher
"Gerrit Cole","Shohei Ohtani"
```

The model will look up their stats automatically.
