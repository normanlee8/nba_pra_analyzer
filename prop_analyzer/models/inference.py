import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.features.calculator import smooth_projection
from prop_analyzer.models.training import rename_features_for_model

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)

def get_recent_bias_map(days_back=21):
    """Loads graded history and calculates average model error for bias correction."""
    graded_files = sorted(cfg.GRADED_DIR.glob("graded_*.parquet"), reverse=True)
    if not graded_files: return {}

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
            
    if not recent_dfs: return {}
        
    full_history = pd.concat(recent_dfs, ignore_index=True)
    keep_cols = [c for c in [Cols.PLAYER_ID, Cols.PLAYER_NAME, Cols.PROP_TYPE, Cols.ACTUAL_VAL, Cols.PREDICTION] if c in full_history.columns]
    full_history = full_history[keep_cols].copy()

    full_history[Cols.ACTUAL_VAL] = pd.to_numeric(full_history[Cols.ACTUAL_VAL], errors='coerce')
    full_history[Cols.PREDICTION] = pd.to_numeric(full_history[Cols.PREDICTION], errors='coerce')
    full_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION], inplace=True)
    
    full_history['Error'] = full_history[Cols.ACTUAL_VAL] - full_history[Cols.PREDICTION]
    group_cols = [Cols.PLAYER_ID, Cols.PROP_TYPE] if Cols.PLAYER_ID in full_history.columns else [Cols.PLAYER_NAME, Cols.PROP_TYPE]

    bias_series = full_history.groupby(group_cols)['Error'].mean()
    return bias_series.to_dict()

def calculate_implied_probability(proj, line, std_dev=None):
    """
    Since we replaced the Classifier with an Ensemble Regressor, we derive
    the win probability based on the continuous projection using a Normal Distribution.
    """
    if pd.isna(std_dev) or std_dev <= 0:
        std_dev = max(line * 0.20, 1.0) # Fallback: assume variance is ~20% of the line
    
    # Probability of actual going OVER the discrete line
    prob_over = 1.0 - norm.cdf(line, loc=proj, scale=std_dev)
    return prob_over

def determine_tier(prob_over, proj_val, line, prop_type):
    """Calculates confidence Tier based on Continuous Edge and Derived Probability."""
    if pd.isna(prob_over) or pd.isna(proj_val) or pd.isna(line) or line == 0:
        return {'Tier': 'C Tier', 'Best Pick': 'Pass', 'Win_Prob': 0.0, 'Edge': 0.0}

    is_over = prob_over >= 0.50
    
    if is_over:
        win_prob = prob_over
        edge_val = proj_val - line
        pick = 'Over'
        if proj_val < line: return {'Tier': 'C Tier', 'Best Pick': 'Over', 'Win_Prob': win_prob, 'Edge': 0.0}
    else:
        win_prob = 1.0 - prob_over
        edge_val = line - proj_val
        pick = 'Under'
        if proj_val > line: return {'Tier': 'C Tier', 'Best Pick': 'Under', 'Win_Prob': win_prob, 'Edge': 0.0}

    tier = 'C Tier'
    edge_pct = (edge_val / line) if line > 0 else 0.0
    
    if win_prob > 0.60 and edge_pct > 0.10: tier = 'S Tier'
    elif win_prob > 0.56 and edge_pct > 0.05: tier = 'A Tier'
    elif win_prob > 0.53: tier = 'B Tier'
    
    if prop_type in ['BLK', 'STL', 'FG3M'] and tier == 'S Tier' and edge_pct < 0.15:
        tier = 'A Tier'

    return {'Tier': tier, 'Best Pick': pick, 'Win_Prob': win_prob, 'Edge': edge_val}

def predict_props(todays_props_df):
    logging.info(f"Starting batch inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns:
        logging.critical(f"Column '{Cols.PROP_TYPE}' not found in input.")
        return pd.DataFrame()

    logging.info("Loading recent grading history for Bias Correction...")
    try:
        bias_map = get_recent_bias_map(days_back=21)
    except Exception as e:
        logging.warning(f"Failed to load bias map: {e}")
        bias_map = {}

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        logging.info(f"Predicting {len(group)} rows for {prop_cat}...")
        
        # --- 1. DETERMINE MODEL SOURCE (Multi-Output vs Individual) ---
        model_to_load = prop_cat
        is_multi = False
        target_idx = 0

        if prop_cat in ['PTS', 'REB', 'AST']:
            multi_arts = load_artifacts('BASE_MULTI')
            if multi_arts and 'metadata' in multi_arts and prop_cat in multi_arts['metadata']['target_cols']:
                model_to_load = 'BASE_MULTI'
                is_multi = True
                target_idx = multi_arts['metadata']['target_cols'].index(prop_cat)

        artifacts = load_artifacts(model_to_load)
        if not artifacts:
            # Fallback to direct model if MULTI fails
            artifacts = load_artifacts(prop_cat)
            is_multi = False
            if not artifacts:
                logging.warning(f"No model artifacts found for {prop_cat}. Skipping.")
                continue
            
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_names = artifacts['features']
        
        # --- 2. PREPROCESS FEATURES ---
        X_raw = group.copy()
        X_raw = rename_features_for_model(X_raw, prop_cat)
        
        X_model = pd.DataFrame(index=X_raw.index)
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        for f in feature_names:
            if f in X_raw.columns: X_model[f] = X_raw[f]
            elif f in inv_map: X_model[f] = X_raw[inv_map[f]]
            else: X_model[f] = 0.0 

        try:
            X_scaled = scaler.transform(X_model)
            
            # --- 3. GENERATE ENSEMBLE PREDICTIONS ---
            preds = model.predict(X_scaled)
            
            # Extract target array depending on if the model is Multi-Output
            if is_multi:
                raw_proj_values = preds[:, target_idx]
            else:
                raw_proj_values = preds
            
            # Extract Stabilization Factors
            szn_avgs = X_raw.get('SZN Avg', pd.Series(np.nan, index=X_raw.index))
            l5_avgs = X_raw.get('L5 Avg', szn_avgs)
            vols = X_raw.get('L10_STD_DEV', pd.Series(np.nan, index=X_raw.index))
            
            final_proj_values = []
            final_probs = []
            
            # --- 4. STABILIZE AND CALCULATE PROBABILITY ---
            for idx, raw_val in enumerate(raw_proj_values):
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else raw_val
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else raw_val
                vol = float(vols.iloc[idx]) if not pd.isna(vols.iloc[idx]) else None
                
                smoothed = smooth_projection(raw_val, s_avg, r_avg, vol if vol else 1.0)
                final_proj_values.append(smoothed)
                
                line = float(group.iloc[idx][Cols.PROP_LINE])
                prob_over = calculate_implied_probability(smoothed, line, vol)
                final_probs.append(prob_over)
            
            # --- 5. BUILD FINAL RESULTS ---
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                prob = float(final_probs[idx])
                proj = float(final_proj_values[idx])
                
                # Apply Bias Correction
                p_id = row.get(Cols.PLAYER_ID)
                key = (int(p_id), prop_cat) if not pd.isna(p_id) else (row.get(Cols.PLAYER_NAME), prop_cat)
                
                bias = bias_map.get(key, 0.0)
                proj = proj + (bias * 0.5)
                
                # Grade Signal
                analysis = determine_tier(prob, proj, line, prop_cat)
                diff_pct = (analysis['Edge'] / line) * 100.0 if line > 0 else 0.0

                res_dict = {
                    Cols.PLAYER_NAME: row[Cols.PLAYER_NAME],
                    Cols.TEAM: row.get('TEAM_ABBREVIATION', row.get(Cols.TEAM, 'UNK')),
                    Cols.OPPONENT: row.get(Cols.OPPONENT, 'UNK'),
                    Cols.DATE: row[Cols.DATE],
                    Cols.PROP_TYPE: prop_cat,
                    Cols.PROP_LINE: line,
                    'Proj': round(proj, 2),
                    'Prob': round(analysis['Win_Prob'], 3),
                    'Pick': analysis['Best Pick'],
                    'Tier': analysis['Tier'],
                    '_Sort_Diff': diff_pct 
                }
                results.append(res_dict)

        except Exception as e:
            logging.error(f"Inference error for {prop_cat}: {e}", exc_info=True)
            continue

    final_df = pd.DataFrame(results)
    if final_df.empty: return pd.DataFrame()
        
    return final_df