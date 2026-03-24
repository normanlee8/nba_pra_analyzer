import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.features.calculator import (smooth_projection, 
                                               get_discrete_probabilities, 
                                               estimate_combo_variance)

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)


def determine_confidence_tier(win_prob, pick_type, delta_gap, line, abs_diff, cv, l10_hit_rate, vs_opp_hit_rate, vs_opp_games, s_avg, r_avg):
    """
    Evaluates confidence tiers based strictly on probability, hit rates, volatility, and edge/delta.
    """
    tier = 'Pass'
    
    # 1. TRAP DETECTION (Hard Passes)
    if pick_type == 'Over':
        if cv > cfg.MAX_CV_HARD_PASS_OVER or l10_hit_rate < cfg.MIN_L10_HIT_HARD_PASS_OVER:
            return 'Pass / Too Volatile'
    else:
        dynamic_cv_threshold = getattr(cfg, 'MAX_CV_HARD_PASS_UNDER_LOW_LINE', 0.35) if line < 10.0 else getattr(cfg, 'MAX_CV_HARD_PASS_UNDER_BASE', 0.45)
        min_l10_hit_under = getattr(cfg, 'MIN_L10_HIT_HARD_PASS_UNDER', 0.40) 
        if cv > dynamic_cv_threshold or l10_hit_rate < min_l10_hit_under:
            return 'Pass / Too Volatile'

    # 2. TIER ASSIGNMENT
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
    """Evaluates the win probability of a projection against a specific line using statistical distributions."""
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
        'Pick': pick, 'Win_Prob': win_prob, 'Tier': tier,
        'Active_Hit_Rate': active_hit_rate, 'Active_VS_Opp_Hit_Rate': active_vs_opp_hit_rate
    }


def get_col_safe(df, prop_cat, base_name):
    """Helper to safely extract columns with or without prop prefixes."""
    if base_name in df.columns: return df[base_name]
    if f"{prop_cat}_{base_name}" in df.columns: return df[f"{prop_cat}_{base_name}"]
    return pd.Series(np.nan, index=df.index)


def predict_props(todays_props_df):
    logging.info(f"Starting Quantile Inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns: 
        return pd.DataFrame()
    
    meta_artifacts = load_artifacts('META_CALIBRATOR')
    meta_model = meta_artifacts['model'] if meta_artifacts else None
    meta_features = meta_artifacts['features'] if meta_artifacts else []

    predicted_mins = {}
    min_artifacts = load_artifacts('MIN')
    
    if min_artifacts:
        min_model_dict = min_artifacts['model']
        min_scaler = min_artifacts['scaler']
        min_features_raw = min_artifacts['features']
        
        # Handle dict if quantile trained, else handle direct model
        min_model = min_model_dict['q50'] if isinstance(min_model_dict, dict) else min_model_dict

        X_raw_mins = todays_props_df.copy()
        sanitized_map_mins = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw_mins.columns}
        inv_map_mins = {v: k for k, v in sanitized_map_mins.items()}
        
        new_features_mins = {}
        for f in min_features_raw:
            if f in X_raw_mins.columns: new_features_mins[f] = X_raw_mins[f]
            elif f in inv_map_mins: new_features_mins[f] = X_raw_mins[inv_map_mins[f]]
            else: new_features_mins[f] = np.nan 
            
        X_min_model = pd.DataFrame(new_features_mins, index=X_raw_mins.index)
        try:
            X_min_scaled = min_scaler.transform(X_min_model)
            X_min_scaled_df = pd.DataFrame(X_min_scaled, columns=X_min_model.columns, index=X_min_model.index)
            X_raw_mins['PRED_MIN'] = min_model.predict(X_min_scaled_df)
            predicted_mins = X_raw_mins['PRED_MIN'].to_dict()
            
            for orig_idx, pred_min in predicted_mins.items():
                if pred_min <= 0 or pd.isna(pred_min): continue
                for stat in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'USG_PROXY']:
                    per36_col = f'{stat}_L5_PER36'
                    if per36_col in todays_props_df.columns:
                        val_per36 = todays_props_df.at[orig_idx, per36_col]
                        if not pd.isna(val_per36) and val_per36 > 0:
                            # Dynamically scale the L5 average to the model's expected minute workload
                            todays_props_df.at[orig_idx, f'{stat}_L5_AVG'] = val_per36 * (pred_min / 36.0)
                todays_props_df.at[orig_idx, 'MIN_L5_AVG'] = pred_min

        except Exception as e:
            logging.warning(f"Failed to predict standalone minutes: {e}")

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        if prop_cat == 'MIN': continue
        
        artifacts = load_artifacts(prop_cat)
        if not artifacts: 
            logging.warning(f"Missing artifacts for {prop_cat}, skipping inference.")
            continue
            
        model_dict, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['features']
        
        X_raw = group.copy()
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        new_features = {}
        for f in feature_names:
            if f in X_raw.columns: new_features[f] = X_raw[f]
            elif f in inv_map: new_features[f] = X_raw[inv_map[f]]
            else: new_features[f] = np.nan 
            
        X_model = pd.DataFrame(new_features, index=X_raw.index)

        try:
            X_scaled = scaler.transform(X_model)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_model.columns, index=X_model.index)
            
            # Extract Exact Quantiles
            if isinstance(model_dict, dict) and 'q50' in model_dict:
                pred_q10 = model_dict['q10'].predict(X_scaled_df)
                pred_q50 = model_dict['q50'].predict(X_scaled_df)
                pred_q90 = model_dict['q90'].predict(X_scaled_df)
            else:
                pred_q50 = model_dict.predict(X_scaled_df)
                pred_q10 = pred_q50 * 0.75
                pred_q90 = pred_q50 * 1.25
            
            szn_avgs = get_col_safe(X_raw, prop_cat, 'SZN_AVG')
            l5_avgs = get_col_safe(X_raw, prop_cat, 'L5_AVG')
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
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                
                # EXACT TARGET PREDICTION (Median of the Quantile Spread)
                # Removed artificial 30% Vegas regression. We trust the ML model natively.
                adjusted_raw = pred_q50[idx] 
                
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else adjusted_raw
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else adjusted_raw
                
                floor_std = max(line * 0.35, np.sqrt(line))
                
                # TRUE VARIANCE FROM MACHINE LEARNING QUANTILES
                # The spread between the 10th and 90th percentile gives us an exact, player-specific standard deviation
                # A normal distribution 10th-90th spread is roughly 2.56 standard deviations wide
                model_implied_std = (pred_q90[idx] - pred_q10[idx]) / 2.56
                std_dev = max(model_implied_std, floor_std)
                
                cv = float(cvs.iloc[idx]) if not pd.isna(cvs.iloc[idx]) else (std_dev / s_avg if s_avg > 0 else 0.5)
                
                l10_hit_legacy = float(hit_rates_legacy.iloc[idx]) if not pd.isna(hit_rates_legacy.iloc[idx]) else 0.50
                l10_over_rate = float(hit_rates_over.iloc[idx]) if not pd.isna(hit_rates_over.iloc[idx]) else l10_hit_legacy
                l10_under_rate = float(hit_rates_under.iloc[idx]) if not pd.isna(hit_rates_under.iloc[idx]) else (1.0 - l10_hit_legacy)
                
                vs_opp_legacy = float(vs_opp_legacy_rates.iloc[idx]) if not pd.isna(vs_opp_legacy_rates.iloc[idx]) else 0.50
                vs_opp_over_rate = float(vs_opp_over_rates.iloc[idx]) if not pd.isna(vs_opp_over_rates.iloc[idx]) else vs_opp_legacy
                vs_opp_under_rate = float(vs_opp_under_rates.iloc[idx]) if not pd.isna(vs_opp_under_rates.iloc[idx]) else (1.0 - vs_opp_legacy)
                vs_opp_games = int(vs_opp_games_counts.iloc[idx]) if not pd.isna(vs_opp_games_counts.iloc[idx]) else 0

                days_rest = float(row.get(Cols.DAYS_REST, 2.0))
                
                # Projection is fully driven by the model's exact learned median
                proj = smooth_projection(adjusted_raw, s_avg, r_avg, std_dev)
                
                dynamic_correlations = {
                    'PTS_REB': float(corr_pr.iloc[idx]) if not pd.isna(corr_pr.iloc[idx]) else 0.25,
                    'PTS_AST': float(corr_pa.iloc[idx]) if not pd.isna(corr_pa.iloc[idx]) else 0.25,
                    'REB_AST': float(corr_ra.iloc[idx]) if not pd.isna(corr_ra.iloc[idx]) else 0.25
                }
                
                sample_size = int(games_played_col.iloc[idx]) if not pd.isna(games_played_col.iloc[idx]) else 15
                variance = estimate_combo_variance(prop_cat, proj, std_dev, base_stds={}, correlations=dynamic_correlations, sample_size=sample_size, base_projs={})

                delta_gap_final = abs(proj - line) / line if line > 0 else 0
                
                eval_res = evaluate_prop(proj, line, variance, prop_cat, delta_gap_final, cv, l10_over_rate, l10_under_rate, vs_opp_over_rate, vs_opp_under_rate, vs_opp_games, s_avg, r_avg, tweedie_power=1.5)
                if not eval_res: continue
                
                base_tier = eval_res['Tier']
                final_tier = base_tier
                meta_prob = None

                # Meta-Calibrator adjustments (Detecting Traps)
                if meta_model is not None and len(meta_features) > 0:
                    blowout_pot = float(row.get('BLOWOUT_POTENTIAL', 0.0))
                    matchup_hit_pct = eval_res['Active_VS_Opp_Hit_Rate'] * 100.0 if vs_opp_games > 0 else eval_res['Active_Hit_Rate'] * 100.0
                    
                    meta_input_dict = {
                        'Prob': eval_res['Win_Prob'], 'Consistency_CV': cv, 'Proj': proj,
                        Cols.PROP_LINE: line, 'Active_Hit%': eval_res['Active_Hit_Rate'] * 100.0,
                        'Matchup_Hit%': matchup_hit_pct, 'BLOWOUT_POTENTIAL': blowout_pot, 'Delta_Gap_Pct': delta_gap_final
                    }
                    meta_input = pd.DataFrame([meta_input_dict])
                    for f in meta_features:
                        if f not in meta_input.columns: meta_input[f] = np.nan
                    
                    try:
                        meta_prob = meta_model.predict_proba(meta_input[meta_features])[0][1]
                        if base_tier in ['S Tier', 'A Tier'] and meta_prob < 0.45: final_tier = 'Trap / Fade'
                        elif base_tier == 'Pass' and meta_prob > 0.65: final_tier = 'B Tier (Meta Edge)'
                    except Exception: pass

                position = row.get('POSITION', row.get('PLAYER_POSITION', row.get('Position', 'UNK')))

                results.append({
                    Cols.PLAYER_NAME: row[Cols.PLAYER_NAME], Cols.TEAM: row.get('TEAM_ABBREVIATION', row.get(Cols.TEAM, 'UNK')),
                    Cols.OPPONENT: row.get(Cols.OPPONENT, 'UNK'), 'Position': position,
                    Cols.DATE: row[Cols.DATE], Cols.PROP_TYPE: prop_cat, Cols.PROP_LINE: line,
                    'Proj': round(proj, 2), 'Prob': round(eval_res['Win_Prob'], 3),
                    'Pick': eval_res['Pick'], 'Consistency_CV': round(cv, 3), 
                    'Active_Hit%': round(eval_res['Active_Hit_Rate'] * 100.0, 1), 
                    'Matchup_Hit%': round(eval_res['Active_VS_Opp_Hit_Rate'] * 100.0, 1) if vs_opp_games > 0 else 'N/A',
                    'Tier': final_tier, 'Base_Tier': base_tier,
                    'Meta_Prob': round(meta_prob, 3) if meta_prob is not None else 'N/A',
                    'BLOWOUT_POTENTIAL': round(float(row.get('BLOWOUT_POTENTIAL', 0.0)), 2),
                    '_Sort_Diff': meta_prob if meta_prob is not None else eval_res['Win_Prob'] 
                })

        except Exception as e:
            logging.error(f"Inference error for {prop_cat}: {e}", exc_info=True)

    return pd.DataFrame(results) if results else pd.DataFrame()