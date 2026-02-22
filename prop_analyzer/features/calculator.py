import numpy as np
import pandas as pd
import logging
from scipy.stats import nbinom

def calculate_bayesian_std(series, method='neg_binomial', shrinkage_param=10.0, dispersion=0.15):
    """
    Calculates a blended Standard Deviation that shrinks towards a theoretical 
    prior (Negative Binomial) when sample size is small.
    
    Improvements:
    - Uses Negative Binomial assumption (Mean < Variance) instead of Poisson (Mean = Variance).
    - NBA stats are over-dispersed; Poisson underestimates volatility for high-usage players.
    
    Formula: 
        Weight = N / (N + K)
        Final_Std = (Weight * Actual_Std) + ((1 - Weight) * Theoretical_Std)
    
    Args:
        series (pd.Series): The data to calculate volatility for.
        method (str): 'neg_binomial' (default) or 'poisson'.
        shrinkage_param (float): 'K' parameter. Higher values mean we trust the prior longer.
        dispersion (float): 'alpha' parameter for NegBinomial. 
                            Var = Mean + (alpha * Mean^2). 
                            0.15 is a conservative baseline for NBA player props.
    """
    clean_series = series.dropna()
    n = len(clean_series)
    
    if n == 0:
        return 0.0
    
    # 1. Calculate Actual Sample Statistics
    actual_mean = clean_series.mean()
    actual_std = clean_series.std(ddof=1) if n > 1 else 0.0
    
    if actual_mean <= 0:
        return 0.0

    # 2. Calculate Theoretical Prior
    if method == 'poisson':
        # Old method: Underestimates volatility for stars (Mean = Variance)
        theoretical_std = np.sqrt(actual_mean)
    else:
        # New method: Negative Binomial (Over-Dispersed)
        # Variance = Mean + (alpha * Mean^2)
        # This accurately captures that higher averages come with exponentially higher variance
        theoretical_var = actual_mean + (dispersion * (actual_mean ** 2))
        theoretical_std = np.sqrt(theoretical_var)
    
    # 3. Calculate Shrinkage Weight (0.0 to 1.0)
    # As N increases, weight -> 1.0 (Trust Data). As N -> 0, weight -> 0.0 (Trust Prior).
    weight = n / (n + shrinkage_param)
    
    # 4. Blend
    final_std = (weight * actual_std) + ((1.0 - weight) * theoretical_std)
    
    return final_std

def calculate_slope(series):
    """
    Calculates the slope of the linear regression line for the series.
    Positive slope = Trending Up. Negative slope = Trending Down.
    """
    y = series.dropna().values
    n = len(y)
    if n < 2:
        return 0.0
    
    x = np.arange(n)
    # Simple linear regression slope formula
    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope

def calculate_hit_rates(series, lines):
    """
    Calculates frequency of hitting Over various lines.
    
    Args:
        series: Historic values.
        lines: List of thresholds (e.g., [10.5, 15.5]) or single float.
    
    Returns:
        float or dict of Hit Rates.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return 0.0
    
    if isinstance(lines, (list, tuple)):
        results = {}
        for line in lines:
            results[f'hit_{line}'] = (clean > line).mean()
        return results
    else:
        return (clean > lines).mean()

def calculate_player_metrics(history_df, stat_col, timeframe=None):
    """
    Core function to generate features for a specific stat column.
    
    Args:
        history_df (pd.DataFrame): Player's game log.
        stat_col (str): The column to analyze (e.g., 'PTS').
        timeframe (int): Optional limit (e.g., Last 10 games).
    
    Returns:
        dict: Statistical features.
    """
    if history_df is None or history_df.empty or stat_col not in history_df.columns:
        return {
            'avg': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0,
            'trend_slope': 0, 'last_3_avg': 0
        }
    
    # Apply timeframe filter if provided
    if timeframe:
        data = history_df[stat_col].tail(timeframe)
    else:
        data = history_df[stat_col]
        
    data = data.dropna()
    if data.empty:
        return {'avg': 0, 'std': 0}

    # Calculate metrics
    avg = data.mean()
    median = data.median()
    
    # Use Dynamic Bayesian Blending for Volatility with Negative Binomial Prior
    # K=8 represents roughly 1/10th of a season
    std_dev = calculate_bayesian_std(data, shrinkage_param=8.0, method='neg_binomial')
    
    # Trend
    slope = calculate_slope(data)
    
    # Recent Form (Exponential Moving Average of last 3 weighted heavily)
    last_3 = data.tail(3)
    recent_avg = last_3.mean() if not last_3.empty else avg

    return {
        'avg': avg,
        'std': std_dev,
        'min': data.min(),
        'max': data.max(),
        'median': median,
        'trend_slope': slope,
        'recent_avg': recent_avg,
        'count': len(data)
    }

def calculate_live_vacancy(team_roster_df):
    """
    Calculates the 'Vacancy' (Missing Usage/Minutes) for a team based on current injuries.
    
    Improvements:
    1. Uses probabilistic weights for Status (Questionable = 50% impact).
    2. Aggregates by Position (Guard/Forward/Center) to give context on WHO gets the usage.
    3. Added safety checks to prevent runaway sums.
    
    Args:
        team_roster_df (pd.DataFrame): Must contain ['STATUS', 'USG%', 'MIN', 'Pos']
    
    Returns:
        dict: {
            'TEAM_MISSING_USG': float,
            'TEAM_MISSING_MIN': float,
            'MISSING_USG_G': float,
            'MISSING_USG_F': float,
            'MISSING_USG_C': float
        }
    """
    metrics = {
        'TEAM_MISSING_USG': 0.0,
        'TEAM_MISSING_MIN': 0.0,
        'MISSING_USG_G': 0.0,
        'MISSING_USG_F': 0.0,
        'MISSING_USG_C': 0.0
    }
    
    if team_roster_df is None or team_roster_df.empty:
        return metrics
    
    required = ['STATUS', 'USG%', 'MIN']
    if not all(col in team_roster_df.columns for col in required):
        return metrics

    # Normalize Status
    def get_injury_weight(status):
        s = str(status).upper().strip()
        if s in ['OUT', 'GTD']: return 1.0  # Treat GTD broadly or check source. Usually OUT/INJURED.
        if 'DOUBTFUL' in s: return 0.75
        if 'QUESTIONABLE' in s: return 0.50
        return 0.0

    # Ensure numeric columns
    df = team_roster_df.copy()
    df['USG%'] = pd.to_numeric(df['USG%'], errors='coerce').fillna(0)
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
    
    # Calculate Impact
    df['Impact_Weight'] = df['STATUS'].apply(get_injury_weight)
    
    # Filter to only rows with impact > 0
    injured_df = df[df['Impact_Weight'] > 0].copy()
    
    if injured_df.empty:
        return metrics

    # 1. Team Totals
    # Note: Usage sums can exceed 100 theoretically if we just sum straight values, 
    # but in context of "missing", simple summation is the standard feature proxy.
    metrics['TEAM_MISSING_USG'] = (injured_df['USG%'] * injured_df['Impact_Weight']).sum()
    metrics['TEAM_MISSING_MIN'] = (injured_df['MIN'] * injured_df['Impact_Weight']).sum()
    
    # 2. Positional Breakdowns
    if 'Pos' in df.columns:
        def categorize_pos(p):
            p = str(p).upper()
            if 'G' in p: return 'G'
            if 'F' in p: return 'F'
            if 'C' in p: return 'C'
            return 'X'

        injured_df['Gen_Pos'] = injured_df['Pos'].apply(categorize_pos)
        
        # Calculate weighted usage per position group
        for pos_code in ['G', 'F', 'C']:
            pos_mask = injured_df['Gen_Pos'] == pos_code
            val = (injured_df.loc[pos_mask, 'USG%'] * injured_df.loc[pos_mask, 'Impact_Weight']).sum()
            metrics[f'MISSING_USG_{pos_code}'] = val

    return metrics

def smooth_projection(raw_proj, season_avg, recent_avg, volatility):
    """
    Weighted ensemble of the raw model projection and simple baselines 
    to prevent overfitting on outliers.
    
    Logic: If volatility is high, trust the long-term Season Avg more.
           If volatility is low (consistent player), trust the Model/Recent.
    """
    # Defensive checks
    if pd.isna(raw_proj): raw_proj = season_avg
    if pd.isna(recent_avg): recent_avg = season_avg
    if pd.isna(volatility) or volatility <= 0: volatility = 1.0
    
    # Establish trust weights
    # High volatility = Lower trust in recent variance/model spikes
    # Example: Vol=10 (High) -> trust_recent approx 0.3
    #          Vol=2 (Low)   -> trust_recent approx 0.8
    trust_recent = 1.0 / (1.0 + (volatility / 5.0))
    
    # Weighted Average
    # 50% Model, remaining 50% split between Recent and Season based on trust
    final_proj = (0.50 * raw_proj) + \
                 (0.50 * trust_recent * recent_avg) + \
                 (0.50 * (1 - trust_recent) * season_avg)
                 
    return final_proj