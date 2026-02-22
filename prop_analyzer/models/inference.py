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
from prop_analyzer.features.calculator import smooth_projection
from prop_analyzer.models.training import (
    add_interaction_features, 
    rename_features_for_model, 
    PROP_KEY_MAP
)

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)

def get_recent_bias_map(days_back=21):
    """
    Loads graded history and calculates the average model error per player+prop.
    Returns a dictionary: {(PlayerID, PropType): Bias_Value}
    Positive Bias = Model Undershoots (Actual > Pred) -> We should Add to pred.
    Negative Bias = Model Overshoots (Actual < Pred) -> We should Subtract.
    """
    bias_map = {}
    
    # Find recent graded files
    graded_files = sorted(cfg.GRADED_DIR.glob("graded_*.parquet"), reverse=True)
    if not graded_files:
        return {}

    recent_dfs = []
    cutoff_date = pd.Timestamp.now() - timedelta(days=days_back)
    
    for f in graded_files:
        try:
            # Parse date from filename
            file_date_str = f.stem.replace('graded_props_', '').replace('graded_', '')
            try:
                file_date = pd.to_datetime(file_date_str)
            except:
                continue

            if file_date < cutoff_date:
                continue
                
            df = pd.read_parquet(f)
            # Ensure we have necessary columns
            if Cols.ACTUAL_VAL in df.columns and Cols.PREDICTION in df.columns:
                recent_dfs.append(df)
        except Exception:
            continue
            
    if not recent_dfs:
        return {}
        
    full_history = pd.concat(recent_dfs, ignore_index=True)
    
    # === OPTIMIZATION & CLEANING ===
    # 1. Reduce width to avoid Fragmentation Warning
    # We only need ID, Prop, Actual, and Prediction. Dropping the 1000+ feature columns.
    keep_cols = [c for c in [Cols.PLAYER_ID, Cols.PLAYER_NAME, Cols.PROP_TYPE, Cols.ACTUAL_VAL, Cols.PREDICTION] if c in full_history.columns]
    full_history = full_history[keep_cols].copy()

    # 2. Convert to numeric, coercing errors to NaN
    full_history[Cols.ACTUAL_VAL] = pd.to_numeric(full_history[Cols.ACTUAL_VAL], errors='coerce')
    full_history[Cols.PREDICTION] = pd.to_numeric(full_history[Cols.PREDICTION], errors='coerce')
    
    # 3. Drop rows where we couldn't convert (e.g., 'Actual' was 'INJ' or empty)
    full_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION], inplace=True)
    
    # Calculate Residual: (Actual - Prediction)
    full_history['Error'] = full_history[Cols.ACTUAL_VAL] - full_history[Cols.PREDICTION]
    
    # Group by Player and Prop
    group_cols = [Cols.PLAYER_ID, Cols.PROP_TYPE]
    
    # Fallback to Name if ID missing in history
    if Cols.PLAYER_ID not in full_history.columns:
        if Cols.PLAYER_NAME in full_history.columns:
            group_cols = [Cols.PLAYER_NAME, Cols.PROP_TYPE]
        else:
            return {}

    bias_series = full_history.groupby(group_cols)['Error'].mean()
    
    # Convert to dict for fast lookup
    return bias_series.to_dict()

def determine_tier(prob_over, proj_val, line, prop_type):
    """
    Calculates the confidence Tier (S, A, B, C).
    """
    if pd.isna(prob_over) or pd.isna(proj_val) or pd.isna(line) or line == 0:
        return {'Tier': 'C Tier', 'Best Pick': 'Pass', 'Win_Prob': 0.0, 'Edge': 0.0}

    # 1. Determine Direction & Edge
    is_over = prob_over >= 0.50
    
    if is_over:
        win_prob = prob_over
        edge_val = proj_val - line
        pick = 'Over'
        if proj_val < line:
            return {'Tier': 'C Tier', 'Best Pick': 'Over', 'Win_Prob': win_prob, 'Edge': 0.0}
    else:
        win_prob = 1.0 - prob_over
        edge_val = line - proj_val
        pick = 'Under'
        if proj_val > line:
            return {'Tier': 'C Tier', 'Best Pick': 'Under', 'Win_Prob': win_prob, 'Edge': 0.0}

    # 2. Assign Tier based on Strength of Signal
    tier = 'C Tier'
    edge_pct = (edge_val / line) if line > 0 else 0.0
    
    # Tier Thresholds
    if win_prob > 0.60 and edge_pct > 0.10:
        tier = 'S Tier'
    elif win_prob > 0.56 and edge_pct > 0.05:
        tier = 'A Tier'
    elif win_prob > 0.53:
        tier = 'B Tier'
    
    if prop_type in ['BLK', 'STL', 'FG3M'] and tier == 'S Tier' and edge_pct < 0.15:
        tier = 'A Tier'

    return {
        'Tier': tier,
        'Best Pick': pick,
        'Win_Prob': win_prob,
        'Edge': edge_val
    }

def predict_props(todays_props_df):
    logging.info(f"Starting batch inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns:
        logging.critical(f"Column '{Cols.PROP_TYPE}' not found in input.")
        return pd.DataFrame()

    # --- Load Bias Map ---
    logging.info("Loading recent grading history for Bias Correction...")
    try:
        bias_map = get_recent_bias_map(days_back=21)
        logging.info(f"Loaded bias corrections for {len(bias_map)} player/prop combos.")
    except Exception as e:
        logging.warning(f"Failed to load bias map: {e}")
        bias_map = {}

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        logging.info(f"Predicting {len(group)} rows for {prop_cat}...")
        
        artifacts = load_artifacts(prop_cat)
        if not artifacts: continue
            
        clf = artifacts['clf']
        xgb_q20 = artifacts['q20']['xgb']
        xgb_q80 = artifacts['q80']['xgb']
        scaler = artifacts['scaler']
        feature_names = artifacts['features']
        
        X_raw = group.copy()
        X_raw = add_interaction_features(X_raw)
        X_raw = rename_features_for_model(X_raw, prop_cat)
        
        X_model = pd.DataFrame(index=X_raw.index)
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        for f in feature_names:
            if f in X_raw.columns:
                X_model[f] = X_raw[f]
            elif f in inv_map:
                X_model[f] = X_raw[inv_map[f]]
            else:
                X_model[f] = 0.0 

        try:
            X_scaled = scaler.transform(X_model)
            
            # Raw Predictions
            probs = clf.predict_proba(X_scaled)[:, 1]
            q20_preds = xgb_q20.predict(X_scaled)
            q80_preds = xgb_q80.predict(X_scaled)
            raw_proj_values = (q20_preds + q80_preds) / 2.0
            
            # --- STABILIZATION ---
            szn_avgs = X_raw.get('SZN Avg', pd.Series(np.nan, index=X_raw.index))
            l5_avgs = X_raw.get('L5 Avg', szn_avgs)
            vols = X_raw.get('L10_STD_DEV', pd.Series(1.0, index=X_raw.index))
            
            final_proj_values = []
            
            for idx, raw_val in enumerate(raw_proj_values):
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else raw_val
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else raw_val
                vol = float(vols.iloc[idx]) if not pd.isna(vols.iloc[idx]) else 1.0
                
                smoothed = smooth_projection(raw_val, s_avg, r_avg, vol)
                final_proj_values.append(smoothed)
            
            proj_values = np.array(final_proj_values)
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                prob = float(probs[idx])
                proj = float(proj_values[idx])
                
                # --- Apply Bias Correction ---
                p_id = row.get(Cols.PLAYER_ID)
                if pd.isna(p_id):
                    key = (row.get(Cols.PLAYER_NAME), prop_cat)
                else:
                    key = (int(p_id), prop_cat)
                
                bias = bias_map.get(key, 0.0)
                correction = bias * 0.5
                proj = proj + correction
                # -----------------------------
                
                analysis = determine_tier(prob, proj, line, prop_cat)
                
                diff_pct = 0.0
                if line > 0:
                    diff_pct = (analysis['Edge'] / line) * 100.0

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
    if final_df.empty:
        return pd.DataFrame()
        
    return final_df