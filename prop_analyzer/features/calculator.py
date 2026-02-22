import numpy as np
import pandas as pd
import logging
import math
from scipy.stats import nbinom, poisson, norm

def calculate_bayesian_std(series, method='neg_binomial', shrinkage_param=10.0, dispersion=0.15):
    """Calculates a blended Standard Deviation shrinking towards a theoretical prior."""
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

# ====================================================================
# NEW PROBABILISTIC / BETTING FUNCTIONS
# ====================================================================

def estimate_combo_variance(prop_type, proj, std_dev, base_stds=None):
    """Estimates variance for Combo Props using baseline historical covariance matrices."""
    # If standard deviation is directly provided from model history, use it
    if not pd.isna(std_dev) and std_dev > 0:
        return std_dev ** 2
        
    variance = max(proj * 0.25, 1.0) # Fallback variance
    
    if prop_type in ['PRA', 'PR', 'PA', 'RA'] and base_stds:
        # Rough correlation matrices for NBA players
        # PTS & AST generally negatively correlated for high usage (-0.1)
        # PTS & REB generally slightly positive (+0.1)
        # REB & AST generally zero (0.0)
        
        var_pts = base_stds.get('PTS', proj*0.2)**2
        var_reb = base_stds.get('REB', proj*0.1)**2
        var_ast = base_stds.get('AST', proj*0.1)**2
        
        if prop_type == 'PRA':
            variance = var_pts + var_reb + var_ast + 2*(0.1*math.sqrt(var_pts*var_reb)) - 2*(0.1*math.sqrt(var_pts*var_ast))
        elif prop_type == 'PR':
            variance = var_pts + var_reb + 2*(0.1*math.sqrt(var_pts*var_reb))
        elif prop_type == 'PA':
            variance = var_pts + var_ast - 2*(0.1*math.sqrt(var_pts*var_ast))
        elif prop_type == 'RA':
            variance = var_reb + var_ast
            
    return max(variance, 0.5)

def get_discrete_probabilities(proj, line, variance, dist_type='normal'):
    """Calculates Win, Loss, and Push probabilities accurately accounting for whole/half point lines."""
    std_dev = math.sqrt(max(variance, 0.01))
    is_whole_line = (line % 1 == 0)
    win_target = int(math.ceil(line + 0.01)) # e.g. 5.5 -> 6, 5.0 -> 6
    
    if proj <= 0: return {'win': 0.0, 'push': 0.0, 'loss': 1.0}

    try:
        if dist_type in ['poisson', 'nbinom']:
            if variance <= proj:  # Poisson distribution
                p_loss = poisson.cdf(win_target - 1, proj)
                p_push = poisson.pmf(int(line), proj) if is_whole_line else 0.0
            else: # Negative Binomial distribution
                p = proj / variance
                n = (proj ** 2) / (variance - proj)
                p_loss = nbinom.cdf(win_target - 1, n, p)
                p_push = nbinom.pmf(int(line), n, p) if is_whole_line else 0.0
        else: # Normal approximation with continuity correction
            p_loss = norm.cdf(win_target - 0.5, loc=proj, scale=std_dev)
            if is_whole_line:
                p_push = norm.cdf(line + 0.5, loc=proj, scale=std_dev) - norm.cdf(line - 0.5, loc=proj, scale=std_dev)
            else:
                p_push = 0.0
                
        p_win = 1.0 - p_loss - p_push
        return {
            'win': max(min(p_win, 1.0), 0.0), 
            'push': max(min(p_push, 1.0), 0.0), 
            'loss': max(min(p_loss, 1.0), 0.0)
        }
    except Exception as e:
        logging.warning(f"Error calculating distribution prob: {e}")
        return {'win': 0.5, 'push': 0.0, 'loss': 0.5}

def scale_by_pace(player_proj, proj_mins, team_pace, opp_pace):
    """Adjusts projection based on projected game pace versus baseline."""
    if pd.isna(team_pace) or pd.isna(opp_pace) or team_pace <= 0:
        return player_proj
    # Matchup Pace Estimator
    matchup_pace = (team_pace + opp_pace) / 2.0
    pace_modifier = matchup_pace / team_pace
    return player_proj * pace_modifier