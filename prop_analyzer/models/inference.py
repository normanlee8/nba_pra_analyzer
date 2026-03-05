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
from prop_analyzer.features.calculator import (get_discrete_probabilities, 
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

def determine_confidence_tier(win_prob):
    """
    Tiering based strictly on calibrated Win Probability distributions. 
    Accuracy/Winning focused. High variance drops win probability naturally, 
    so we don't need artificial CV filters here.
    """
    if win_prob >= 0.65:
        return 'S Tier'
    elif win_prob >= 0.60:
        return 'A Tier'
    elif win_prob >= 0.56:
        return 'B Tier'
    elif win_prob >= 0.53:
        return 'C Tier'
    else:
        return 'Pass'

def evaluate_prop(proj, line, variance, prop_type, l10_over_rate, l10_under_rate, vs_opp_over_rate, vs_opp_under_rate):
    if line <= 0: return None
    
    nbinom_props = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
    dist_type = 'nbinom' if prop_type in nbinom_props else 'normal'
    
    probs_over = get_discrete_probabilities(proj, line, variance, dist_type=dist_type)
    probs_under = {'win': probs_over['loss'], 'loss': probs_over['win'], 'push': probs_over['push']}
    
    if probs_over['win'] >= probs_under['win']:
        pick, win_prob = 'Over', probs_over['win']
        active_hit_rate = l10_over_rate
        active_vs_opp_hit_rate = vs_opp_over_rate
    else:
        pick, win_prob = 'Under', probs_under['win']
        active_hit_rate = l10_under_rate
        active_vs_opp_hit_rate = vs_opp_under_rate 
        
    tier = determine_confidence_tier(win_prob)
        
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

        if 'MIN' in feature_names:
            new_features['MIN'] = pd.Series([predicted_mins.get(idx, 36.0) for idx in X_raw.index], index=X_raw.index)
            
        X_model = pd.DataFrame(new_features, index=X_raw.index)

        try:
            X_scaled = scaler.transform(X_model)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_model.columns, index=X_model.index)
            
            # The ML Model is the ultimate source of truth. No manual average smoothing.
            raw_projections = model.predict(X_scaled_df)
            
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
                proj = float(raw_projections[idx])  # Strict reliance on Stacking Ensemble
                
                raw_std = float(stds.iloc[idx]) if not pd.isna(stds.iloc[idx]) else 1.0
                floor_std = max(line * 0.25, np.sqrt(line)) # Minimal sanity floor
                std_dev = max(raw_std, floor_std)
                
                cv = float(cvs.iloc[idx]) if not pd.isna(cvs.iloc[idx]) else (std_dev / proj if proj > 0 else 0.5)
                
                l10_hit_legacy = float(hit_rates_legacy.iloc[idx]) if not pd.isna(hit_rates_legacy.iloc[idx]) else 0.50
                l10_over_rate = float(hit_rates_over.iloc[idx]) if not pd.isna(hit_rates_over.iloc[idx]) else l10_hit_legacy
                l10_under_rate = float(hit_rates_under.iloc[idx]) if not pd.isna(hit_rates_under.iloc[idx]) else (1.0 - l10_hit_legacy)
                
                vs_opp_legacy = float(vs_opp_legacy_rates.iloc[idx]) if not pd.isna(vs_opp_legacy_rates.iloc[idx]) else 0.50
                vs_opp_over_rate = float(vs_opp_over_rates.iloc[idx]) if not pd.isna(vs_opp_over_rates.iloc[idx]) else vs_opp_legacy
                vs_opp_under_rate = float(vs_opp_under_rates.iloc[idx]) if not pd.isna(vs_opp_under_rates.iloc[idx]) else (1.0 - vs_opp_legacy)
                vs_opp_games = int(vs_opp_games_counts.iloc[idx]) if not pd.isna(vs_opp_games_counts.iloc[idx]) else 0
                
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
                
                variance = estimate_combo_variance(prop_cat, proj, std_dev, base_stds=dynamic_base_stds, correlations=dynamic_correlations, sample_size=sample_size)
                
                eval_res = evaluate_prop(proj, line, variance, prop_cat, l10_over_rate, l10_under_rate, vs_opp_over_rate, vs_opp_under_rate)
                if not eval_res: continue
                
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