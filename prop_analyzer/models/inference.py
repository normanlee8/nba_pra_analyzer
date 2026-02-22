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

def calculate_betting_metrics(probs, odds=-110):
    """Calculates EV and Kelly Criterion based on sportsbook odds (Standard -110)."""
    p_win, p_loss = probs['win'], probs['loss']
    
    # Calculate Decimal Odds & Payout
    if odds < 0:
        decimal_odds = 1.0 + (100.0 / abs(odds))
    else:
        decimal_odds = 1.0 + (odds / 100.0)
        
    b = decimal_odds - 1.0  # Profit per $1 wagered (Payout)
    
    # Expected Value Formula: EV = (Win Prob * Payout) - (Loss Prob * 1)
    # Note: Push probability returns stake, EV effect is 0.
    ev = (p_win * b) - p_loss
    
    # Kelly Criterion Formula: f* = (bp - q) / b
    # Tells us the optimal % of bankroll to wager.
    if p_win > 0 and ev > 0:
        kelly = ev / b 
    else:
        kelly = 0.0
        
    return {'EV': ev, 'Kelly': kelly, 'Implied_Odds': 1.0 / decimal_odds}

def determine_ev_tier(ev, win_prob, pick_type):
    """Professional Grade Tiering based STRICTLY on Expected Value (EV)."""
    tier = 'Pass'
    
    ev_pct = ev * 100.0
    win_pct = win_prob * 100.0
    
    # S Tier: "Nuke" Play -> High EV, High Certainty. Warrants max bet.
    if ev_pct >= 8.0 and win_pct >= 58.0: tier = 'S Tier'
    
    # A Tier: Strong +EV play.
    elif ev_pct >= 4.0 and win_pct >= 54.0: tier = 'A Tier'
    
    # B Tier: Slight edge, good for parlays or volume.
    elif ev_pct >= 1.5: tier = 'B Tier'
    
    # C Tier: Negative or neutral EV. 
    else: tier = 'C Tier'

    return tier

def evaluate_prop(proj, line, variance, prop_type):
    """Evaluates both sides (Over/Under) to find the best EV play."""
    if line <= 0: return None
    
    # Pick probability distribution type based on prop category
    dist_type = 'nbinom' if prop_type in ['REB', 'AST', 'STL', 'BLK', 'FG3M'] else 'normal'
    
    # Calculate probabilities for the OVER
    probs_over = get_discrete_probabilities(proj, line, variance, dist_type=dist_type)
    metrics_over = calculate_betting_metrics(probs_over)
    
    # Calculate probabilities for the UNDER
    # Win Under = Loss Over. Loss Under = Win Over.
    probs_under = {'win': probs_over['loss'], 'loss': probs_over['win'], 'push': probs_over['push']}
    metrics_under = calculate_betting_metrics(probs_under)
    
    # Select best side based on EV
    if metrics_over['EV'] >= metrics_under['EV']:
        pick = 'Over'
        metrics = metrics_over
        win_prob = probs_over['win']
    else:
        pick = 'Under'
        metrics = metrics_under
        win_prob = probs_under['win']
        
    tier = determine_ev_tier(metrics['EV'], win_prob, pick)
    
    return {
        'Pick': pick,
        'Win_Prob': win_prob,
        'EV_Pct': metrics['EV'] * 100.0,
        'Kelly': metrics['Kelly'],
        'Tier': tier
    }

def predict_props(todays_props_df):
    logging.info(f"Starting EV inference for {len(todays_props_df)} props...")
    results = []
    
    if Cols.PROP_TYPE not in todays_props_df.columns:
        logging.critical(f"Column '{Cols.PROP_TYPE}' not found in input.")
        return pd.DataFrame()

    try:
        bias_map = get_recent_bias_map(days_back=21)
    except Exception as e:
        logging.warning(f"Failed to load bias map: {e}")
        bias_map = {}

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        logging.info(f"Predicting {len(group)} rows for {prop_cat}...")
        
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
            if not artifacts:
                continue
            
        model, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['features']
        
        X_raw = rename_features_for_model(group.copy(), prop_cat)
        X_model = pd.DataFrame(index=X_raw.index)
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        for f in feature_names:
            if f in X_raw.columns: X_model[f] = X_raw[f]
            elif f in inv_map: X_model[f] = X_raw[inv_map[f]]
            else: X_model[f] = 0.0 

        try:
            X_scaled = scaler.transform(X_model)
            preds = model.predict(X_scaled)
            raw_proj_values = preds[:, target_idx] if is_multi else preds
            
            szn_avgs = X_raw.get('SZN Avg', pd.Series(np.nan, index=X_raw.index))
            l5_avgs = X_raw.get('L5 Avg', szn_avgs)
            stds = X_raw.get('L10_STD_DEV', pd.Series(np.nan, index=X_raw.index))
            team_pace = X_raw.get('GAME_PACE', pd.Series(np.nan, index=X_raw.index))
            opp_pace = X_raw.get('OPP_GAME_PACE', pd.Series(np.nan, index=X_raw.index))
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                raw_val = raw_proj_values[idx]
                
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else raw_val
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else raw_val
                std_dev = float(stds.iloc[idx]) if not pd.isna(stds.iloc[idx]) else None
                t_pace = float(team_pace.iloc[idx]) if not pd.isna(team_pace.iloc[idx]) else None
                o_pace = float(opp_pace.iloc[idx]) if not pd.isna(opp_pace.iloc[idx]) else None
                
                # 1. Base Smoothed Projection
                proj = smooth_projection(raw_val, s_avg, r_avg, std_dev if std_dev else 1.0)
                
                # 2. Scale via Matchup Pace
                proj = scale_by_pace(proj, 36.0, t_pace, o_pace)
                
                # 3. Apply Bias Correction
                p_id = row.get(Cols.PLAYER_ID)
                key = (int(p_id), prop_cat) if not pd.isna(p_id) else (row.get(Cols.PLAYER_NAME), prop_cat)
                proj += (bias_map.get(key, 0.0) * 0.5)
                
                # 4. Calculate Variance (Handling Combos)
                variance = estimate_combo_variance(prop_cat, proj, std_dev)
                
                # 5. Evaluate Expected Value & Probability
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
                    '_Sort_Diff': eval_res['EV_Pct'] # Used by run_analysis for sorting
                }
                results.append(res_dict)

        except Exception as e:
            logging.error(f"Inference error for {prop_cat}: {e}", exc_info=True)
            continue

    final_df = pd.DataFrame(results)
    if final_df.empty: return pd.DataFrame()
        
    return final_df