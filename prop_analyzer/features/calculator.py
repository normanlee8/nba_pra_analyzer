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


# ====================================================================
# PROBABILISTIC / BETTING FUNCTIONS
# ====================================================================

def estimate_combo_variance(prop_type, proj, std_dev, base_stds=None, correlations=None, sample_size=15):
    """Estimates variance for Combo Props using accurate covariance matrices and statistical diversification."""
    
    recent_variance = std_dev ** 2 if not pd.isna(std_dev) and std_dev > 0 else (proj * 0.35) ** 2

    structural_variance = recent_variance  # Default fallback

    # Exact Covariance Matrix Math for Combos
    if prop_type in ['PRA', 'PR', 'PA', 'RA'] and base_stds and correlations:
        std_p = base_stds.get('PTS', proj * 0.4)
        std_r = base_stds.get('REB', proj * 0.3)
        std_a = base_stds.get('AST', proj * 0.3)
        
        var_p = std_p ** 2
        var_r = std_r ** 2
        var_a = std_a ** 2
        
        cov_pr = correlations.get('PTS_REB', 0.15) * std_p * std_r
        cov_pa = correlations.get('PTS_AST', 0.15) * std_p * std_a
        cov_ra = correlations.get('REB_AST', 0.05) * std_r * std_a
        
        if prop_type == 'PRA':
            structural_variance = var_p + var_r + var_a + 2*cov_pr + 2*cov_pa + 2*cov_ra
        elif prop_type == 'PR':
            structural_variance = var_p + var_r + 2*cov_pr
        elif prop_type == 'PA':
            structural_variance = var_p + var_a + 2*cov_pa
        elif prop_type == 'RA':
            structural_variance = var_r + var_a + 2*cov_ra

    # Bayesian Shrinkage (Blend recent observed variance with theoretical structural variance)
    # We require ~30 games to fully trust the player's recent variance over the math model
    weight_recent = min(sample_size / 30.0, 0.80) 
    
    final_variance = (recent_variance * weight_recent) + (structural_variance * (1.0 - weight_recent))
        
    # Ensure strict mathematical overdispersion for discrete modeling (NB requires variance > mean)
    return max(final_variance, proj * 1.05)

def get_discrete_probabilities(proj, line, variance, dist_type='normal', tweedie_power=None):
    """Calculates probabilities directly from exact variance mapping to Negative Binomial parameters."""
    
    # Force strict overdispersion so Negative Binomial mathematically functions without collapsing
    variance = max(variance, proj * 1.05)
    std_dev = math.sqrt(variance)
    
    is_whole_line = (line % 1 == 0)
    loss_threshold = math.floor(line - 0.01)
    
    if proj <= 0: return {'win': 0.0, 'push': 0.0, 'loss': 1.0}

    try:
        if dist_type in ['poisson', 'nbinom']:
            if variance <= proj:
                # Fallback to Poisson if mathematically trapped
                p_loss_strict = poisson.cdf(loss_threshold, proj)
                p_push = poisson.pmf(int(line), proj) if is_whole_line else 0.0
            else:
                # Exact mapping to Negative Binomial distribution
                p = proj / variance
                n = (proj ** 2) / (variance - proj)
                p_loss_strict = nbinom.cdf(loss_threshold, n, p)
                p_push = nbinom.pmf(int(line), n, p) if is_whole_line else 0.0
        else:
            # Continuous Normal approximation for non-count stats (e.g., fractional fantasy points)
            p_loss_strict = norm.cdf(line - 0.5, loc=proj, scale=std_dev) if is_whole_line else norm.cdf(line, loc=proj, scale=std_dev)
            if is_whole_line:
                p_push = norm.cdf(line + 0.5, loc=proj, scale=std_dev) - norm.cdf(line - 0.5, loc=proj, scale=std_dev)
            else:
                p_push = 0.0
                
        p_win = 1.0 - p_loss_strict - p_push
        return {
            'win': max(min(p_win, 1.0), 0.0), 
            'push': max(min(p_push, 1.0), 0.0), 
            'loss': max(min(p_loss_strict, 1.0), 0.0) 
        }
    except Exception as e:
        logging.warning(f"Error calculating distribution prob: {e}")
        return {'win': 0.5, 'push': 0.0, 'loss': 0.5}