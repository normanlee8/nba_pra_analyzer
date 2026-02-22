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
                                               estimate_combo_variance, 
                                               scale_by_pace)

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)

def get_recent_bias_map(days_back=21):
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

    return full_history.groupby(group_cols)['Error'].mean().to_dict()

def calculate_betting_metrics(probs, odds=-110):
    p_win, p_loss = probs['win'], probs['loss']
    decimal_odds = 1.0 + (100.0 / abs(odds)) if odds < 0 else 1.0 + (odds / 100.0)
    b = decimal_odds - 1.0
    ev = (p_win * b) - p_loss
    kelly = ev / b if p_win > 0 and ev > 0 else 0.0
    return {'EV': ev, 'Kelly': kelly, 'Implied_Odds': 1.0 / decimal_odds}

def determine_ev_tier(ev, win_prob, pick_type):
    tier = 'Pass'
    ev_pct = ev * 100.0
    win_pct = win_prob * 100.0
    
    if ev_pct >= 8.0 and win_pct >= 58.0: tier = 'S Tier'
    elif ev_pct >= 4.0 and win_pct >= 54.0: tier = 'A Tier'
    elif ev_pct >= 1.5: tier = 'B Tier'
    else: tier = 'C Tier'
    return tier

def evaluate_prop(proj, line, variance, prop_type):
    if line <= 0: return None
    dist_type = 'nbinom' if prop_type in ['REB', 'AST', 'STL', 'BLK', 'FG3M'] else 'normal'
    
    probs_over = get_discrete_probabilities(proj, line, variance, dist_type=dist_type)
    metrics_over = calculate_betting_metrics(probs_over)
    
    probs_under = {'win': probs_over['loss'], 'loss': probs_over['win'], 'push': probs_over['push']}
    metrics_under = calculate_betting_metrics(probs_under)
    
    if metrics_over['EV'] >= metrics_under['EV']:
        pick, metrics, win_prob = 'Over', metrics_over, probs_over['win']
    else:
        pick, metrics, win_prob = 'Under', metrics_under, probs_under['win']
        
    return {
        'Pick': pick,
        'Win_Prob': win_prob,
        'EV_Pct': metrics['EV'] * 100.0,
        'Kelly': metrics['Kelly'],
        'Tier': determine_ev_tier(metrics['EV'], win_prob, pick)
    }

def get_col_safe(df, prop_cat, base_name):
    if base_name in df.columns: return df[base_name]
    if f"{prop_cat}_{base_name}" in df.columns: return df[f"{prop_cat}_{base_name}"]
    return pd.Series(np.nan, index=df.index)

def predict_props(todays_props_df):
    logging.info(f"Starting EV inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns: return pd.DataFrame()
    try: bias_map = get_recent_bias_map(days_back=21)
    except Exception: bias_map = {}

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
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
            artifacts = load_artifacts(prop_cat)
            is_multi = False
            if not artifacts: continue
            
        model, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['features']
        
        # FIX: Removed the buggy rename wrapper, passing raw columns perfectly mapped
        X_raw = group.copy()
        X_model = pd.DataFrame(index=X_raw.index)
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        for f in feature_names:
            if f in X_raw.columns: X_model[f] = X_raw[f]
            elif f in inv_map: X_model[f] = X_raw[inv_map[f]]
            else: X_model[f] = 0.0 

        try:
            X_scaled = scaler.transform(X_model)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_model.columns, index=X_model.index)
            
            preds = model.predict(X_scaled_df)
            raw_proj_values = preds[:, target_idx] if is_multi else preds
            
            szn_avgs = get_col_safe(X_raw, prop_cat, 'SZN_AVG')
            l5_avgs = get_col_safe(X_raw, prop_cat, 'L5_AVG')
            stds = get_col_safe(X_raw, prop_cat, 'L10_STD_DEV')
            team_pace = get_col_safe(X_raw, prop_cat, 'GAME_PACE')
            opp_pace = get_col_safe(X_raw, prop_cat, 'OPP_GAME_PACE')
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                raw_val = raw_proj_values[idx]
                
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else raw_val
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else raw_val
                std_dev = float(stds.iloc[idx]) if not pd.isna(stds.iloc[idx]) else None
                t_pace = float(team_pace.iloc[idx]) if not pd.isna(team_pace.iloc[idx]) else None
                o_pace = float(opp_pace.iloc[idx]) if not pd.isna(opp_pace.iloc[idx]) else None
                
                proj = smooth_projection(raw_val, s_avg, r_avg, std_dev if std_dev else 1.0)
                proj = scale_by_pace(proj, 36.0, t_pace, o_pace)
                
                p_id = row.get(Cols.PLAYER_ID)
                key = (int(p_id), prop_cat) if not pd.isna(p_id) else (row.get(Cols.PLAYER_NAME), prop_cat)
                proj += (bias_map.get(key, 0.0) * 0.5)
                
                variance = estimate_combo_variance(prop_cat, proj, std_dev)
                eval_res = evaluate_prop(proj, line, variance, prop_cat)
                if not eval_res: continue
                
                res_dict = {
                    Cols.PLAYER_NAME: row[Cols.PLAYER_NAME],
                    Cols.TEAM: row.get('TEAM_ABBREVIATION', row.get(Cols.TEAM, 'UNK')),
                    Cols.OPPONENT: row.get(Cols.OPPONENT, 'UNK'),
                    Cols.DATE: row[Cols.DATE],
                    Cols.PROP_TYPE: prop_cat,
                    Cols.PROP_LINE: line,
                    'Proj': round(proj, 2),
                    'Prob': round(eval_res['Win_Prob'], 3),
                    'Pick': eval_res['Pick'],
                    'EV%': round(eval_res['EV_Pct'], 2),
                    'Kelly': round(eval_res['Kelly'], 3),
                    'Tier': eval_res['Tier'],
                    '_Sort_Diff': eval_res['EV_Pct']
                }
                results.append(res_dict)

        except Exception as e:
            logging.error(f"Inference error for {prop_cat}: {e}", exc_info=True)
            continue

    final_df = pd.DataFrame(results)
    if final_df.empty: return pd.DataFrame()
        
    return final_df