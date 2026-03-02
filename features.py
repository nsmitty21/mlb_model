"""
features.py — MLB Model Feature Engineering
"""
import numpy as np
import pandas as pd

def american_to_prob(odds):
    if pd.isna(odds): return np.nan
    if odds > 0: return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def remove_vig(p_home, p_away):
    total = p_home + p_away
    if total == 0: return p_home, p_away
    return p_home / total, p_away / total

def rolling_mean(s, w=10): return s.rolling(w, min_periods=3).mean()
def expand_mean(s): return s.expanding(min_periods=5).mean()

PARK_FACTORS = {
    "Colorado Rockies":1.18,"Boston Red Sox":1.08,"Cincinnati Reds":1.07,
    "Philadelphia Phillies":1.05,"Chicago Cubs":1.04,"Houston Astros":1.03,
    "Texas Rangers":1.03,"Atlanta Braves":1.02,"New York Yankees":1.01,
    "Toronto Blue Jays":1.00,"Los Angeles Dodgers":1.00,"San Diego Padres":0.99,
    "New York Mets":0.99,"St. Louis Cardinals":0.99,"Baltimore Orioles":0.99,
    "Minnesota Twins":0.98,"Detroit Tigers":0.98,"Kansas City Royals":0.97,
    "Chicago White Sox":0.97,"Cleveland Guardians":0.97,"Seattle Mariners":0.96,
    "Pittsburgh Pirates":0.96,"Tampa Bay Rays":0.96,"Oakland Athletics":0.96,
    "Miami Marlins":0.95,"Washington Nationals":0.95,"San Francisco Giants":0.94,
    "Los Angeles Angels":0.97,"Milwaukee Brewers":0.98,"Arizona Diamondbacks":1.01,
}

def get_park_factor(home_team): return PARK_FACTORS.get(home_team, 1.00)

def compute_team_rolling_features(df, window=15):
    df = df.sort_values(["team","date"]).copy()
    for col in ["runs_scored","runs_allowed"]:
        if col not in df.columns: continue
        df[f"roll_{window}_{col}"] = df.groupby("team")[col].transform(lambda s: rolling_mean(s,window).shift(1))
        df[f"season_{col}_avg"]    = df.groupby("team")[col].transform(lambda s: expand_mean(s).shift(1))
    if "runs_scored" in df and "runs_allowed" in df:
        df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
        df["roll_run_diff"] = df.groupby("team")["run_diff"].transform(lambda s: rolling_mean(s,window).shift(1))
    if "win" in df.columns:
        df["roll_win_pct"]    = df.groupby("team")["win"].transform(lambda s: rolling_mean(s,window).shift(1))
        df["season_win_pct"]  = df.groupby("team")["win"].transform(lambda s: expand_mean(s).shift(1))
    return df

def build_game_features(home_team, away_team, home_pitcher_stats, away_pitcher_stats,
                         home_team_stats, away_team_stats,
                         home_lineup_vs_pitcher=None, away_lineup_vs_pitcher=None,
                         home_rest_days=1, away_rest_days=1):
    park = get_park_factor(home_team)
    def g(d, k, default=np.nan): return d.get(k, default) if d else default
    feats = {
        "home_field_adv":1.0,"park_factor":park,
        "home_rest_days":home_rest_days,"away_rest_days":away_rest_days,
        "rest_advantage":home_rest_days - away_rest_days,
        "home_roll_runs_scored":g(home_team_stats,"roll_runs_scored"),
        "home_roll_runs_allowed":g(home_team_stats,"roll_runs_allowed"),
        "home_roll_run_diff":g(home_team_stats,"roll_run_diff"),
        "home_roll_win_pct":g(home_team_stats,"roll_win_pct"),
        "home_season_win_pct":g(home_team_stats,"season_win_pct"),
        "away_roll_runs_scored":g(away_team_stats,"roll_runs_scored"),
        "away_roll_runs_allowed":g(away_team_stats,"roll_runs_allowed"),
        "away_roll_run_diff":g(away_team_stats,"roll_run_diff"),
        "away_roll_win_pct":g(away_team_stats,"roll_win_pct"),
        "away_season_win_pct":g(away_team_stats,"season_win_pct"),
        "run_diff_diff":g(home_team_stats,"roll_run_diff",0)-g(away_team_stats,"roll_run_diff",0),
        "win_pct_diff":g(home_team_stats,"roll_win_pct",0.5)-g(away_team_stats,"roll_win_pct",0.5),
        "home_sp_era":g(home_pitcher_stats,"roll_era"),
        "home_sp_fip":g(home_pitcher_stats,"roll_fip"),
        "home_sp_whip":g(home_pitcher_stats,"roll_whip"),
        "home_sp_k9":g(home_pitcher_stats,"roll_k_per_9"),
        "home_sp_bb9":g(home_pitcher_stats,"roll_bb_per_9"),
        "home_sp_hr9":g(home_pitcher_stats,"roll_hr_per_9"),
        "home_sp_season_era":g(home_pitcher_stats,"season_era"),
        "home_sp_season_fip":g(home_pitcher_stats,"season_fip"),
        "away_sp_era":g(away_pitcher_stats,"roll_era"),
        "away_sp_fip":g(away_pitcher_stats,"roll_fip"),
        "away_sp_whip":g(away_pitcher_stats,"roll_whip"),
        "away_sp_k9":g(away_pitcher_stats,"roll_k_per_9"),
        "away_sp_bb9":g(away_pitcher_stats,"roll_bb_per_9"),
        "away_sp_hr9":g(away_pitcher_stats,"roll_hr_per_9"),
        "away_sp_season_era":g(away_pitcher_stats,"season_era"),
        "away_sp_season_fip":g(away_pitcher_stats,"season_fip"),
        "sp_era_diff":g(home_pitcher_stats,"roll_era",4.5)-g(away_pitcher_stats,"roll_era",4.5),
        "sp_fip_diff":g(home_pitcher_stats,"roll_fip",4.2)-g(away_pitcher_stats,"roll_fip",4.2),
        "home_lineup_bvp_ops":g(home_lineup_vs_pitcher,"bvp_ops"),
        "home_lineup_bvp_k_pct":g(home_lineup_vs_pitcher,"bvp_k_pct"),
        "away_lineup_bvp_ops":g(away_lineup_vs_pitcher,"bvp_ops"),
        "away_lineup_bvp_k_pct":g(away_lineup_vs_pitcher,"bvp_k_pct"),
        "bvp_ops_diff":g(home_lineup_vs_pitcher,"bvp_ops",0.7)-g(away_lineup_vs_pitcher,"bvp_ops",0.7),
    }
    return feats

FEATURE_COLS = list(build_game_features("","",{},{},{},{}).keys())
