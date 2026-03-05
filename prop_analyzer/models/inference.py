import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.features.calculator import (smooth_projection, 
                                               get_discrete_probabilities, 
                                               estimate_combo_variance)

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)

def get_system_learning_maps(days_back=21):
    graded_files = sorted(cfg.GRADED_DIR.glob("graded_*.parquet"), reverse=True)
    if not graded_files: return {}, {}

    recent_dfs = []
    cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
    
    for f in graded_files:
        try:
            file_date_str = f.stem.replace('graded_props_', '').replace('graded_', '')
            file_date = pd.to_datetime(file_date_str)
            if file_date < cutoff_date: continue
            df = pd.read_parquet(f)
            if Cols.ACTUAL_VAL in df.columns and Cols.PREDICTION in df.columns:
                recent_dfs.append(df)
        except Exception: continue
            
    if not recent_dfs: return {}, {}
        
    full_history = pd.concat(recent_dfs, ignore_index=True)
    keep_cols = [c for c in [Cols.PLAYER_ID, Cols.PLAYER_NAME, Cols.PROP_TYPE, Cols.ACTUAL_VAL, Cols.PREDICTION] if c in full_history.columns]
    full_history = full_history[keep_cols].copy()

    full_history[Cols.ACTUAL_VAL] = pd.to_numeric(full_history[Cols.ACTUAL_VAL], errors='coerce')
    full_history[Cols.PREDICTION] = pd.to_numeric(full_history[Cols.PREDICTION], errors='coerce')
    full_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION], inplace=True)
    
    full_history['Error'] = full_history[Cols.ACTUAL_VAL] - full_history[Cols.PREDICTION]
    full_history['Abs_Error'] = full_history['Error'].abs()
    
    full_history['Pct_Error'] = np.where(
        full_history[Cols.PREDICTION] > 0, 
        full_history['Error'] / full_history[Cols.PREDICTION], 
        0.0
    )
    
    full_history['Mapped_Prop'] = full_history[Cols.PROP_TYPE].map(lambda x: cfg.MASTER_PROP_MAP.get(x, x))
    cat_bias_pct = full_history.groupby('Mapped_Prop')['Pct_Error'].mean().to_dict()
    cat_mae = full_history.groupby('Mapped_Prop')['Abs_Error'].mean().to_dict()

    return cat_bias_pct, cat_mae

def determine_confidence_tier(win_prob, pick_type, delta_gap, line, abs_diff, cv, l10_hit_rate, vs_opp_hit_rate, vs_opp_games, s_avg, r_avg):
    tier = 'Pass'
    
    if pick_type == 'Over':
        if cv > cfg.MAX_CV_HARD_PASS_OVER or l10_hit_rate < cfg.MIN_L10_HIT_HARD_PASS_OVER:
            return 'Pass / Too Volatile'
    else:
        dynamic_cv_threshold = getattr(cfg, 'MAX_CV_HARD_PASS_UNDER_LOW_LINE', 0.85) if line < 10.0 else getattr(cfg, 'MAX_CV_HARD_PASS_UNDER_BASE', 0.45)
        if cv > dynamic_cv_threshold:
            return 'Pass / Too Volatile'

    if win_prob >= cfg.MIN_PROB_FOR_S_TIER and cv < cfg.MAX_CV_FOR_S_TIER and (pick_type != 'Over' or l10_hit_rate >= cfg.MIN_L10_HIT_FOR_S_TIER): 
        if abs_diff >= (line * 0.08):
            tier = 'S Tier'
        else:
            tier = 'A Tier' 
    elif win_prob >= cfg.MIN_PROB_FOR_A_TIER and cv < cfg.MAX_CV_FOR_A_TIER: 
        if abs_diff >= (line * 0.04):
            tier = 'A Tier'
        else:
            tier = 'B Tier'
    elif win_prob >= cfg.MIN_PROB_FOR_B_TIER and cv < cfg.MAX_CV_FOR_B_TIER: 
        tier = 'B Tier'
    elif win_prob >= cfg.MIN_PROB_FOR_C_TIER: 
        tier = 'C Tier'
    else: 
        tier = 'Pass'
    
    return tier

def evaluate_prop(proj, line, variance, prop_type, delta_gap, cv, l10_over_rate, l10_under_rate, vs_opp_over_rate, vs_opp_under_rate, vs_opp_games, s_avg, r_avg, tweedie_power=1.5):
    if line <= 0: return None
    
    nbinom_props = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
    dist_type = 'nbinom' if prop_type in nbinom_props else 'normal'
    
    probs_over = get_discrete_probabilities(proj, line, variance, dist_type=dist_type, tweedie_power=tweedie_power)
    probs_under = {'win': probs_over['loss'], 'loss': probs_over['win'], 'push': probs_over['push']}
    
    if probs_over['win'] >= probs_under['win']:
        pick, win_prob = 'Over', probs_over['win']
        active_hit_rate = l10_over_rate
        active_vs_opp_hit_rate = vs_opp_over_rate
    else:
        pick, win_prob = 'Under', probs_under['win']
        active_hit_rate = l10_under_rate
        active_vs_opp_hit_rate = vs_opp_under_rate 
        
    abs_diff = abs(proj - line)
    tier = determine_confidence_tier(win_prob, pick, delta_gap, line, abs_diff, cv, active_hit_rate, active_vs_opp_hit_rate, vs_opp_games, s_avg, r_avg)
        
    return {
        'Pick': pick,
        'Win_Prob': win_prob,
        'Tier': tier,
        'Active_Hit_Rate': active_hit_rate,
        'Active_VS_Opp_Hit_Rate': active_vs_opp_hit_rate
    }

def get_col_safe(df, prop_cat, base_name):
    if base_name in df.columns: return df[base_name]
    if f"{prop_cat}_{base_name}" in df.columns: return df[f"{prop_cat}_{base_name}"]
    return pd.Series(np.nan, index=df.index)

def predict_props(todays_props_df):
    logging.info(f"Starting Probability Inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns: return pd.DataFrame()
    
    try: 
        _, cat_mae = get_system_learning_maps(days_back=21)
    except Exception as e: 
        logging.warning(f"Could not load system learning maps: {e}")
        cat_mae = {}

    # IDEA 2: Predict Minutes first to anchor all feature calculations
    predicted_mins = {}
    min_artifacts = load_artifacts('MIN')
    if min_artifacts:
        logging.info("Standalone Minutes model found. Predicting minutes...")
        min_model = min_artifacts['model']
        min_features = min_artifacts['features']
        min_scaler = min_artifacts['scaler']
        
        X_raw_mins = todays_props_df.copy()
        sanitized_map_mins = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw_mins.columns}
        inv_map_mins = {v: k for k, v in sanitized_map_mins.items()}
        
        new_features_mins = {}
        for f in min_features:
            if f in X_raw_mins.columns: new_features_mins[f] = X_raw_mins[f]
            elif f in inv_map_mins: new_features_mins[f] = X_raw_mins[inv_map_mins[f]]
            else: new_features_mins[f] = np.nan 
            
        X_min_model = pd.DataFrame(new_features_mins, index=X_raw_mins.index)
        try:
            X_min_scaled = min_scaler.transform(X_min_model)
            X_min_scaled_df = pd.DataFrame(X_min_scaled, columns=X_min_model.columns, index=X_min_model.index)
            X_raw_mins['PRED_MIN'] = min_model.predict(X_min_scaled_df)
            predicted_mins = X_raw_mins['PRED_MIN'].to_dict()
        except Exception as e:
            logging.warning(f"Failed to predict standalone minutes: {e}")

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        if prop_cat == 'MIN': continue
        
        artifacts = load_artifacts(prop_cat)
        if not artifacts:
            logging.warning(f"No trained artifacts found for {prop_cat}. Skipping.")
            continue
            
        model, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['features']
        
        X_raw = group.copy()
        
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        new_features = {}
        for f in feature_names:
            if f in X_raw.columns: new_features[f] = X_raw[f]
            elif f in inv_map: new_features[f] = X_raw[inv_map[f]]
            else: new_features[f] = np.nan 

        # IDEA 2: Inject Predicted Minutes into the Feature Engine before scaling
        # This allows the tree models to natively scale rates against the expected playtime
        if 'MIN' in feature_names:
            new_features['MIN'] = pd.Series([predicted_mins.get(idx, 36.0) for idx in X_raw.index], index=X_raw.index)
            
        X_model = pd.DataFrame(new_features, index=X_raw.index)

        try:
            X_scaled = scaler.transform(X_model)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_model.columns, index=X_model.index)
            
            raw_projections = model.predict(X_scaled_df)
            
            szn_avgs = get_col_safe(X_raw, prop_cat, 'SZN_AVG')
            l5_avgs = get_col_safe(X_raw, prop_cat, 'L5_AVG')
            stds = get_col_safe(X_raw, prop_cat, 'L10_STD_DEV')
            cvs = get_col_safe(X_raw, prop_cat, 'L10_CV')
            games_played_col = get_col_safe(X_raw, prop_cat, 'SEASON_G')
            
            hit_rates_legacy = get_col_safe(X_raw, prop_cat, 'L10_HIT_RATE')
            hit_rates_over = get_col_safe(X_raw, prop_cat, 'L10_OVER_RATE')
            hit_rates_under = get_col_safe(X_raw, prop_cat, 'L10_UNDER_RATE')
            
            vs_opp_legacy_rates = get_col_safe(X_raw, prop_cat, 'VS_OPP_HIT_RATE')
            vs_opp_over_rates = get_col_safe(X_raw, prop_cat, 'VS_OPP_OVER_RATE')
            vs_opp_under_rates = get_col_safe(X_raw, prop_cat, 'VS_OPP_UNDER_RATE')
            vs_opp_games_counts = get_col_safe(X_raw, prop_cat, 'VS_OPP_GAMES_COUNT')
            
            corr_pr = get_col_safe(X_raw, prop_cat, 'PTS_REB_CORR')
            corr_pa = get_col_safe(X_raw, prop_cat, 'PTS_AST_CORR')
            corr_ra = get_col_safe(X_raw, prop_cat, 'REB_AST_CORR')
            
            base_stds = {
                'PTS': get_col_safe(X_raw, 'PTS', 'L10_STD_DEV'),
                'REB': get_col_safe(X_raw, 'REB', 'L10_STD_DEV'),
                'AST': get_col_safe(X_raw, 'AST', 'L10_STD_DEV')
            }
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                raw_val = raw_projections[idx]
                
                pred_min = predicted_mins.get(orig_idx, float(row.get('MIN_SZN_AVG', 36.0)))
                hist_mins = float(row.get('MIN_L10_AVG', pred_min))
                
                raw_std = float(stds.iloc[idx]) if not pd.isna(stds.iloc[idx]) else 1.0
                floor_std = max(line * 0.35, np.sqrt(line))
                std_dev = max(raw_std, floor_std)
                
                # IDEA 4: Continuous Bayesian Shrinkage towards Market Anchor
                if line > 0:
                    # Vegas variance proxy: Sharper lines have less variance. We assume ~20% CV for market lines
                    vegas_implied_std = max(line * 0.20, 1.5)
                    var_vegas = vegas_implied_std ** 2
                    var_model = std_dev ** 2
                    
                    # Inverse-Variance Weighting (Posterior Mean calculation)
                    weight_model = var_vegas / (var_model + var_vegas)
                    weight_vegas = var_model / (var_model + var_vegas)
                    
                    adjusted_raw = (raw_val * weight_model) + (line * weight_vegas)
                else:
                    adjusted_raw = raw_val
                
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else adjusted_raw
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else adjusted_raw
                
                cv = float(cvs.iloc[idx]) if not pd.isna(cvs.iloc[idx]) else (std_dev / s_avg if s_avg > 0 else 0.5)
                
                l10_hit_legacy = float(hit_rates_legacy.iloc[idx]) if not pd.isna(hit_rates_legacy.iloc[idx]) else 0.50
                l10_over_rate = float(hit_rates_over.iloc[idx]) if not pd.isna(hit_rates_over.iloc[idx]) else l10_hit_legacy
                l10_under_rate = float(hit_rates_under.iloc[idx]) if not pd.isna(hit_rates_under.iloc[idx]) else (1.0 - l10_hit_legacy)
                
                vs_opp_legacy = float(vs_opp_legacy_rates.iloc[idx]) if not pd.isna(vs_opp_legacy_rates.iloc[idx]) else 0.50
                vs_opp_over_rate = float(vs_opp_over_rates.iloc[idx]) if not pd.isna(vs_opp_over_rates.iloc[idx]) else vs_opp_legacy
                vs_opp_under_rate = float(vs_opp_under_rates.iloc[idx]) if not pd.isna(vs_opp_under_rates.iloc[idx]) else (1.0 - vs_opp_legacy)
                vs_opp_games = int(vs_opp_games_counts.iloc[idx]) if not pd.isna(vs_opp_games_counts.iloc[idx]) else 0

                days_rest = float(row.get(Cols.DAYS_REST, 2.0))
                min_deviation = abs(pred_min - hist_mins) / hist_mins if hist_mins > 0 else 0

                proj = smooth_projection(adjusted_raw, s_avg, r_avg, std_dev)
                
                dynamic_correlations = {
                    'PTS_REB': float(corr_pr.iloc[idx]) if not pd.isna(corr_pr.iloc[idx]) else 0.25,
                    'PTS_AST': float(corr_pa.iloc[idx]) if not pd.isna(corr_pa.iloc[idx]) else 0.25,
                    'REB_AST': float(corr_ra.iloc[idx]) if not pd.isna(corr_ra.iloc[idx]) else 0.25
                }
                
                dynamic_base_stds = {
                    'PTS': float(base_stds['PTS'].iloc[idx]) if not pd.isna(base_stds['PTS'].iloc[idx]) else (proj*0.2),
                    'REB': float(base_stds['REB'].iloc[idx]) if not pd.isna(base_stds['REB'].iloc[idx]) else (proj*0.1),
                    'AST': float(base_stds['AST'].iloc[idx]) if not pd.isna(base_stds['AST'].iloc[idx]) else (proj*0.1)
                }
                
                sample_size = int(games_played_col.iloc[idx]) if not pd.isna(games_played_col.iloc[idx]) else 15
                
                # IDEA 3: Combo Variance Scaling is handled smoothly here by dynamically updating correlations
                variance = estimate_combo_variance(prop_cat, proj, std_dev, base_stds=dynamic_base_stds, correlations=dynamic_correlations, sample_size=sample_size)
                
                historic_mae = cat_mae.get(prop_cat, 1.0)
                if historic_mae > 1.5:  
                    variance_multiplier = min(1.0 + (historic_mae * 0.15), 1.5)
                    variance = variance * variance_multiplier

                delta_gap_final = abs(proj - line) / line if line > 0 else 0
                
                # Default tweedie power param fallback (since we removed tweedie from training)
                eval_res = evaluate_prop(proj, line, variance, prop_cat, delta_gap_final, cv, l10_over_rate, l10_under_rate, vs_opp_over_rate, vs_opp_under_rate, vs_opp_games, s_avg, r_avg, tweedie_power=1.5)
                if not eval_res: continue
                
                if days_rest > 7.0 or min_deviation > 0.30:
                    tier_ladder = ['S Tier', 'A Tier', 'B Tier', 'C Tier', 'Pass']
                    if eval_res['Tier'] in tier_ladder:
                        current_idx = tier_ladder.index(eval_res['Tier'])
                        if current_idx < len(tier_ladder) - 2: 
                            eval_res['Tier'] = tier_ladder[current_idx + 1]
                
                position = row.get('POSITION', row.get('PLAYER_POSITION', row.get('Position', 'UNK')))

                res_dict = {
                    Cols.PLAYER_NAME: row[Cols.PLAYER_NAME],
                    Cols.TEAM: row.get('TEAM_ABBREVIATION', row.get(Cols.TEAM, 'UNK')),
                    Cols.OPPONENT: row.get(Cols.OPPONENT, 'UNK'),
                    'Position': position,
                    Cols.DATE: row[Cols.DATE],
                    Cols.PROP_TYPE: prop_cat,
                    Cols.PROP_LINE: line,
                    'Proj': round(proj, 2),
                    'Prob': round(eval_res['Win_Prob'], 3),
                    'Pick': eval_res['Pick'],
                    'Consistency_CV': round(cv, 3), 
                    'Active_Hit%': round(eval_res['Active_Hit_Rate'] * 100.0, 1), 
                    'Matchup_Hit%': round(eval_res['Active_VS_Opp_Hit_Rate'] * 100.0, 1) if vs_opp_games > 0 else 'N/A',
                    'Tier': eval_res['Tier'],
                    '_Sort_Diff': eval_res['Win_Prob'] 
                }
                results.append(res_dict)

        except Exception as e:
            logging.error(f"Inference error for {prop_cat}: {e}", exc_info=True)
            continue

    final_df = pd.DataFrame(results)
    if final_df.empty: return pd.DataFrame()
        
    return final_df