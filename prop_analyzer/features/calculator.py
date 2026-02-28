import numpy as np
import pandas as pd
import logging
import math
from scipy.stats import nbinom, poisson, norm

# Silence Pandas 2.1.0+ FutureWarnings regarding silent downcasting
pd.set_option('future.no_silent_downcasting', True)

# ====================================================================
# HELPERS: FEATURE ENGINEERING & INFERENCE
# ====================================================================

def winsorize_series(series, limit=0.10):
    """
    Caps outlier performances using an expanding window.
    This completely eliminates lookahead bias (target leakage) by ensuring
    a player's future performances don't artificially raise their clipping threshold today.
    """
    clean = series.dropna()
    if len(clean) < 5: 
        return series
    
    # Calculate the rolling percentile dynamically (e.g., 90th percentile)
    # Uses data strictly BEFORE or INCLUDING the current row
    expanding_thresholds = clean.expanding(min_periods=5).quantile(1.0 - limit)
    
    # Prevent clipping the first 4 games by setting their threshold to infinity
    expanding_thresholds = expanding_thresholds.fillna(float('inf'))
    
    # Apply the mathematically honest, time-aware clipping
    capped = clean.clip(upper=expanding_thresholds).infer_objects(copy=False)
    
    # Re-align with the original series index to restore any NaNs properly
    return capped.reindex(series.index)

def calculate_dynamic_hit_rates(past_performances, benchmark):
    """
    Calculates hit rates against a specific historical benchmark over multiple windows.
    past_performances should be ordered oldest to newest.
    """
    if not past_performances or pd.isna(benchmark):
        return {
            'L5_HIT_RATE': 0.0, 'L10_HIT_RATE': 0.0, 'L20_HIT_RATE': 0.0, 'SZN_HIT_RATE': 0.0,
            'L5_OVER_RATE': 0.0, 'L10_OVER_RATE': 0.0, 'L20_OVER_RATE': 0.0, 'SZN_OVER_RATE': 0.0,
            'L5_UNDER_RATE': 0.0, 'L10_UNDER_RATE': 0.0, 'L20_UNDER_RATE': 0.0, 'SZN_UNDER_RATE': 0.0
        }
    
    # Reverse so index 0 is the most recent game
    recent = past_performances[::-1]
    
    # Legacy hits for Machine Learning Features (Preserves >=)
    legacy_hits = [1 if x >= benchmark else 0 for x in recent]
    
    # Accurate separated Win Rates (Resolves integer line Push logic)
    over_hits = [1 if x > benchmark else 0 for x in recent]
    under_hits = [1 if x < benchmark else 0 for x in recent]
    
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
        
    return {
        'L5_HIT_RATE': safe_mean(legacy_hits[:5]),
        'L10_HIT_RATE': safe_mean(legacy_hits[:10]),
        'L20_HIT_RATE': safe_mean(legacy_hits[:20]),
        'SZN_HIT_RATE': safe_mean(legacy_hits),
        
        'L5_OVER_RATE': safe_mean(over_hits[:5]),
        'L10_OVER_RATE': safe_mean(over_hits[:10]),
        'L20_OVER_RATE': safe_mean(over_hits[:20]),
        'SZN_OVER_RATE': safe_mean(over_hits),
        
        'L5_UNDER_RATE': safe_mean(under_hits[:5]),
        'L10_UNDER_RATE': safe_mean(under_hits[:10]),
        'L20_UNDER_RATE': safe_mean(under_hits[:20]),
        'SZN_UNDER_RATE': safe_mean(under_hits)
    }

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
        theoretical_var = actual_mean + (dispersion * (actual_mean ** 2))
        theoretical_std = np.sqrt(theoretical_var)
    
    weight = n / (n + shrinkage_param)
    final_std = (weight * actual_std) + ((1.0 - weight) * theoretical_std)
    
    return final_std

def smooth_projection(raw_proj, season_avg, recent_avg, volatility):
    if pd.isna(raw_proj): raw_proj = season_avg
    if pd.isna(recent_avg): recent_avg = season_avg
    if pd.isna(volatility) or volatility <= 0: volatility = 1.0
    
    trust_recent = 1.0 / (1.0 + (volatility / 5.0))
    # Trust the ML model heavily (90%). Only use averages as a 10% smoothing anchor to prevent extreme edge cases.
    final_proj = (0.90 * raw_proj) + (0.10 * trust_recent * recent_avg) + (0.10 * (1 - trust_recent) * season_avg)
    return final_proj

# ====================================================================
# PROBABILISTIC / BETTING FUNCTIONS
# ====================================================================

def estimate_combo_variance(prop_type, proj, std_dev, base_stds=None, correlations=None, sample_size=15):
    """Estimates variance for Combo Props using baseline historical covariance matrices and dynamic correlations."""
    base_variance = max(proj * 0.25, 1.0)
    
    if not correlations:
        correlations = {'PTS_REB': 0.25, 'PTS_AST': 0.25, 'REB_AST': 0.25}
        
    # Calculate Structural Covariance
    if prop_type in ['PRA', 'PR', 'PA', 'RA'] and base_stds:
        var_pts = base_stds.get('PTS', proj*0.2)**2
        var_reb = base_stds.get('REB', proj*0.1)**2
        var_ast = base_stds.get('AST', proj*0.1)**2
        
        cov_pr = correlations.get('PTS_REB', 0.25) * math.sqrt(var_pts * var_reb)
        cov_pa = correlations.get('PTS_AST', 0.25) * math.sqrt(var_pts * var_ast)
        cov_ra = correlations.get('REB_AST', 0.25) * math.sqrt(var_reb * var_ast)
        
        if prop_type == 'PRA':
            base_variance = var_pts + var_reb + var_ast + 2*cov_pr + 2*cov_pa + 2*cov_ra
        elif prop_type == 'PR':
            base_variance = var_pts + var_reb + 2*cov_pr
        elif prop_type == 'PA':
            base_variance = var_pts + var_ast + 2*cov_pa
        elif prop_type == 'RA':
            base_variance = var_reb + var_ast + 2*cov_ra
            
    recent_variance = std_dev ** 2 if not pd.isna(std_dev) and std_dev > 0 else base_variance
    
    # Bayesian Shrinkage. Base the blend on sample size instead of a hardcoded 40/60 split.
    weight = min(sample_size / 20.0, 0.90)  # Max out at 90% trust in recent variance after 18+ games
    
    if prop_type in ['PRA', 'PR', 'PA', 'RA']:
        final_variance = (base_variance * (1.0 - weight)) + (recent_variance * weight)
    else:
        final_variance = recent_variance
        
    return max(final_variance, 0.5)

def get_discrete_probabilities(proj, line, historical_variance, dist_type='normal', tweedie_power=1.5):
    """Calculates probabilities accurately accounting for Tweedie dispersion and whole/half point lines."""
    if proj > 0 and historical_variance > 0:
        phi = historical_variance / (proj ** tweedie_power)
    else:
        phi = 1.0
        
    dynamic_variance = phi * (proj ** tweedie_power)
    variance = max(dynamic_variance, 0.01)
    
    std_dev = math.sqrt(variance)
    is_whole_line = (line % 1 == 0)
    win_target = int(math.ceil(line + 0.01))
    
    if proj <= 0: return {'win': 0.0, 'push': 0.0, 'loss': 1.0}

    try:
        if dist_type in ['poisson', 'nbinom']:
            if variance <= proj:
                # Fallback to Poisson if underdispersed
                p_loss = poisson.cdf(win_target - 1, proj)
                p_push = poisson.pmf(int(line), proj) if is_whole_line else 0.0
            else:
                p = proj / variance
                n = (proj ** 2) / (variance - proj)
                p_loss = nbinom.cdf(win_target - 1, n, p)
                p_push = nbinom.pmf(int(line), n, p) if is_whole_line else 0.0
        else:
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