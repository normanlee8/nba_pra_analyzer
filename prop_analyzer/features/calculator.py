import numpy as np
import pandas as pd
import logging
from scipy.stats import nbinom

def calculate_bayesian_std(series, method='neg_binomial', shrinkage_param=10.0, dispersion=0.15):
    """
    Calculates a blended Standard Deviation that shrinks towards a theoretical 
    prior (Negative Binomial) when sample size is small.
    """
    clean_series = series.dropna()
    n = len(clean_series)
    
    if n == 0: return 0.0
    
    actual_mean = clean_series.mean()
    actual_std = clean_series.std(ddof=1) if n > 1 else 0.0
    
    if actual_mean <= 0: return 0.0

    if method == 'poisson':
        theoretical_std = np.sqrt(actual_mean)
    else:
        # Negative Binomial (Over-Dispersed)
        theoretical_var = actual_mean + (dispersion * (actual_mean ** 2))
        theoretical_std = np.sqrt(theoretical_var)
    
    weight = n / (n + shrinkage_param)
    final_std = (weight * actual_std) + ((1.0 - weight) * theoretical_std)
    
    return final_std

def calculate_slope(series):
    y = series.dropna().values
    n = len(y)
    if n < 2: return 0.0
    
    x = np.arange(n)
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope

def calculate_hit_rates(series, lines):
    clean = series.dropna()
    if len(clean) == 0: return 0.0
    
    if isinstance(lines, (list, tuple)):
        results = {}
        for line in lines:
            results[f'hit_{line}'] = (clean >= line).mean()
        return results
    else:
        return (clean >= lines).mean()

def calculate_player_metrics(history_df, stat_col, timeframe=None):
    if history_df is None or history_df.empty or stat_col not in history_df.columns:
        return {'avg': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'trend_slope': 0, 'cv': 0}
    
    data = history_df[stat_col].tail(timeframe) if timeframe else history_df[stat_col]
    data = data.dropna()
    if data.empty: return {'avg': 0, 'std': 0, 'cv': 0}

    avg = data.mean()
    median = data.median()
    std_dev = calculate_bayesian_std(data, shrinkage_param=8.0, method='neg_binomial')
    slope = calculate_slope(data)
    last_3 = data.tail(3)
    recent_avg = last_3.mean() if not last_3.empty else avg
    cv = std_dev / avg if avg > 0 else 0

    return {
        'avg': avg, 'std': std_dev, 'min': data.min(), 'max': data.max(), 
        'median': median, 'trend_slope': slope, 'recent_avg': recent_avg, 
        'count': len(data), 'cv': cv
    }

def calculate_live_vacancy(team_roster_df):
    metrics = {
        'TEAM_MISSING_USG': 0.0, 'TEAM_MISSING_MIN': 0.0,
        'MISSING_USG_G': 0.0, 'MISSING_USG_F': 0.0, 'MISSING_USG_C': 0.0
    }
    
    if team_roster_df is None or team_roster_df.empty: return metrics
    
    def get_injury_weight(status):
        s = str(status).upper().strip()
        if s in ['OUT', 'GTD']: return 1.0  
        if 'DOUBTFUL' in s: return 0.75
        if 'QUESTIONABLE' in s: return 0.50
        return 0.0

    df = team_roster_df.copy()
    df['USG%'] = pd.to_numeric(df.get('USG%', 0), errors='coerce').fillna(0)
    df['MIN'] = pd.to_numeric(df.get('MIN', 0), errors='coerce').fillna(0)
    df['Impact_Weight'] = df.get('STATUS', '').apply(get_injury_weight)
    
    injured_df = df[df['Impact_Weight'] > 0].copy()
    if injured_df.empty: return metrics

    metrics['TEAM_MISSING_USG'] = (injured_df['USG%'] * injured_df['Impact_Weight']).sum()
    metrics['TEAM_MISSING_MIN'] = (injured_df['MIN'] * injured_df['Impact_Weight']).sum()
    
    if 'Pos' in df.columns:
        def cat_pos(p):
            p = str(p).upper()
            return 'G' if 'G' in p else ('F' if 'F' in p else ('C' if 'C' in p else 'X'))
        injured_df['Gen_Pos'] = injured_df['Pos'].apply(cat_pos)
        
        for p in ['G', 'F', 'C']:
            mask = injured_df['Gen_Pos'] == p
            metrics[f'MISSING_USG_{p}'] = (injured_df.loc[mask, 'USG%'] * injured_df.loc[mask, 'Impact_Weight']).sum()

    return metrics

def smooth_projection(raw_proj, season_avg, recent_avg, volatility):
    if pd.isna(raw_proj): raw_proj = season_avg
    if pd.isna(recent_avg): recent_avg = season_avg
    if pd.isna(volatility) or volatility <= 0: volatility = 1.0
    
    trust_recent = 1.0 / (1.0 + (volatility / 5.0))
    final_proj = (0.50 * raw_proj) + (0.50 * trust_recent * recent_avg) + (0.50 * (1 - trust_recent) * season_avg)
    return final_proj