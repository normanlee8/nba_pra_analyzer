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
                                               scale_by_pace,
                                               calculate_implied_minutes)

def load_artifacts(prop_cat):
    return registry.load_artifacts(prop_cat)

def get_system_learning_maps(days_back=21):
    graded_files = sorted(cfg.GRADED_DIR.glob("graded_*.parquet"), reverse=True)
    if not graded_files: return {}, {}, {}

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
            
    if not recent_dfs: return {}, {}, {}
        
    full_history = pd.concat(recent_dfs, ignore_index=True)
    keep_cols = [c for c in [Cols.PLAYER_ID, Cols.PLAYER_NAME, Cols.PROP_TYPE, Cols.ACTUAL_VAL, Cols.PREDICTION] if c in full_history.columns]
    full_history = full_history[keep_cols].copy()

    full_history[Cols.ACTUAL_VAL] = pd.to_numeric(full_history[Cols.ACTUAL_VAL], errors='coerce')
    full_history[Cols.PREDICTION] = pd.to_numeric(full_history[Cols.PREDICTION], errors='coerce')
    full_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION], inplace=True)
    
    full_history['Error'] = full_history[Cols.ACTUAL_VAL] - full_history[Cols.PREDICTION]
    full_history['Abs_Error'] = full_history['Error'].abs()
    
    group_cols = [Cols.PLAYER_ID, Cols.PROP_TYPE] if Cols.PLAYER_ID in full_history.columns else [Cols.PLAYER_NAME, Cols.PROP_TYPE]

    player_bias = full_history.groupby(group_cols)['Error'].mean().to_dict()
    full_history['Mapped_Prop'] = full_history[Cols.PROP_TYPE].map(lambda x: cfg.MASTER_PROP_MAP.get(x, x))
    cat_bias = full_history.groupby('Mapped_Prop')['Error'].mean().to_dict()
    cat_mae = full_history.groupby('Mapped_Prop')['Abs_Error'].mean().to_dict()

    return player_bias, cat_bias, cat_mae

def determine_confidence_tier(win_prob, pick_type, delta_gap, line, abs_diff, cv, l10_hit_rate, vs_opp_hit_rate, vs_opp_games):
    """
    Strictly evaluates based on Win Probability, Hit Rates, low Volatility, and Matchup History.
    """
    tier = 'Pass'
    win_pct = win_prob * 100.0
    
    # Dynamic Vegas Trap Check based on Line Size (Refined for extreme low lines)
    is_trap = False
    if line <= 2.5:
        # Lines of 1.5 and 2.5 are highly volatile, require severe gap to be considered a trap
        if delta_gap > 0.60 and abs_diff > 1.5: is_trap = True
    elif line <= 4.5:
        if delta_gap > 0.45 and abs_diff > 1.35: is_trap = True
    elif line <= 12.5:
        if delta_gap > 0.28 and abs_diff > 2.0: is_trap = True
    else:
        if delta_gap > 0.20 and abs_diff > 3.5: is_trap = True

    if is_trap:
        return 'Trap / High Variance'

    # Filter out highly volatile players entirely
    if cv > 0.40 or (pick_type == 'Over' and l10_hit_rate < 0.40):
        return 'Pass / Too Volatile'

    # Volume-Scaled Matchup Filter
    if vs_opp_games >= 5:
        # High confidence sample: Require at least a 40% hit rate to play an Over
        if pick_type == 'Over' and vs_opp_hit_rate < 0.40:
            return 'Pass / Owned by Opponent'
        if pick_type == 'Under' and vs_opp_hit_rate > 0.60:
            return 'Pass / Owns Opponent'
            
    elif vs_opp_games >= 3:
        # Low confidence sample (Noise): Only veto on absolute extremes (0% or 100%)
        if pick_type == 'Over' and vs_opp_hit_rate == 0.0:
            return 'Pass / Bad Matchup History'
        if pick_type == 'Under' and vs_opp_hit_rate == 1.0:
            return 'Pass / Bad Matchup History'

    # Probability-Driven Tiers
    if win_pct >= 70.0 and cv < 0.25 and (pick_type == 'Over' and l10_hit_rate >= 0.70): 
        tier = 'S Tier'
    elif win_pct >= 64.0 and cv < 0.30: 
        tier = 'A Tier'
    elif win_pct >= 58.0 and cv < 0.35: 
        tier = 'B Tier'
    elif win_pct >= 54.0: 
        tier = 'C Tier'
    else: 
        tier = 'Pass'
    
    return tier

def evaluate_prop(proj, line, variance, prop_type, delta_gap, cv, l10_hit_rate, vs_opp_hit_rate, vs_opp_games):
    """Evaluates probability of outcome based on distribution modeling."""
    if line <= 0: return None
    dist_type = 'nbinom' if prop_type in ['REB', 'AST', 'STL', 'BLK', 'FG3M'] else 'normal'
    
    probs_over = get_discrete_probabilities(proj, line, variance, dist_type=dist_type)
    probs_under = {'win': probs_over['loss'], 'loss': probs_over['win'], 'push': probs_over['push']}
    
    # Find the Maximum Probability Side
    if probs_over['win'] >= probs_under['win']:
        pick, win_prob = 'Over', probs_over['win']
        active_hit_rate = l10_hit_rate
    else:
        pick, win_prob = 'Under', probs_under['win']
        active_hit_rate = 1.0 - l10_hit_rate 
        
    abs_diff = abs(proj - line)
    tier = determine_confidence_tier(win_prob, pick, delta_gap, line, abs_diff, cv, active_hit_rate, vs_opp_hit_rate, vs_opp_games)
        
    return {
        'Pick': pick,
        'Win_Prob': win_prob,
        'Tier': tier,
        'Active_Hit_Rate': active_hit_rate
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
        player_bias, cat_bias, cat_mae = get_system_learning_maps(days_back=21)
    except Exception as e: 
        logging.warning(f"Could not load system learning maps: {e}")
        player_bias, cat_bias, cat_mae = {}, {}, {}

    grouped = todays_props_df.groupby(Cols.PROP_TYPE)
    
    for prop_cat, group in grouped:
        artifacts = load_artifacts(prop_cat)
        if not artifacts:
            logging.warning(f"No trained artifacts found for {prop_cat}. Skipping.")
            continue
            
        model, scaler, feature_names = artifacts['model'], artifacts['scaler'], artifacts['features']
        
        X_raw = group.copy()
        
        sanitized_map = {c: re.sub(r'[^\w\s]', '_', str(c)).replace(' ', '_') for c in X_raw.columns}
        inv_map = {v: k for k, v in sanitized_map.items()}
        
        # Batch column addition to prevent DataFrame fragmentation warnings
        new_features = {}
        for f in feature_names:
            if f in X_raw.columns: new_features[f] = X_raw[f]
            elif f in inv_map: new_features[f] = X_raw[inv_map[f]]
            else: new_features[f] = 0.0 
            
        X_model = pd.DataFrame(new_features, index=X_raw.index)

        try:
            # Pass through scaler (imputation & standard scaling removed during training update)
            X_scaled = scaler.transform(X_model)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_model.columns, index=X_model.index)
            
            # MODEL PREDICTS RAW TARGET DIRECTLY NOW
            raw_projections = model.predict(X_scaled_df)
            
            szn_avgs = get_col_safe(X_raw, prop_cat, 'SZN_AVG')
            l5_avgs = get_col_safe(X_raw, prop_cat, 'L5_AVG')
            stds = get_col_safe(X_raw, prop_cat, 'L10_STD_DEV')
            cvs = get_col_safe(X_raw, prop_cat, 'L10_CV')
            hit_rates = get_col_safe(X_raw, prop_cat, 'L10_HIT_RATE')
            
            # Extract Matchup Stats
            vs_opp_hit_rates = get_col_safe(X_raw, prop_cat, 'VS_OPP_HIT_RATE')
            vs_opp_games_counts = get_col_safe(X_raw, prop_cat, 'VS_OPP_GAMES_COUNT')
            
            team_pace = get_col_safe(X_raw, prop_cat, 'GAME_PACE')
            opp_pace = get_col_safe(X_raw, prop_cat, 'OPP_GAME_PACE')
            
            for idx, (orig_idx, row) in enumerate(group.iterrows()):
                line = float(row[Cols.PROP_LINE])
                
                # Assign the direct model output instead of adding to the line
                raw_val = raw_projections[idx]
                
                abs_diff_raw = abs(raw_val - line)
                delta_gap_raw = abs_diff_raw / line if line > 0 else 0
                
                decay_threshold = 0.45 if line <= 4.5 else (0.28 if line <= 12.5 else 0.20)
                
                if delta_gap_raw > decay_threshold and abs_diff_raw > 1.2:
                    raw_val = (raw_val * 0.5) + (line * 0.5)
                
                s_avg = float(szn_avgs.iloc[idx]) if not pd.isna(szn_avgs.iloc[idx]) else raw_val
                r_avg = float(l5_avgs.iloc[idx]) if not pd.isna(l5_avgs.iloc[idx]) else raw_val
                raw_std = float(stds.iloc[idx]) if not pd.isna(stds.iloc[idx]) else 1.0
                
                # THE FIX: Prevent Variance Suppression on Small Samples
                # Establish a mathematical floor using the season average and line.
                baseline_std = max(s_avg, line) * 0.15 # Assume at least a 15% standard deviation baseline
                std_dev = max(raw_std, baseline_std)
                
                cv = float(cvs.iloc[idx]) if not pd.isna(cvs.iloc[idx]) else (std_dev / s_avg if s_avg > 0 else 0.5)
                l10_hit_rate = float(hit_rates.iloc[idx]) if not pd.isna(hit_rates.iloc[idx]) else 0.50
                
                vs_opp_hit_rate = float(vs_opp_hit_rates.iloc[idx]) if not pd.isna(vs_opp_hit_rates.iloc[idx]) else 0.50
                vs_opp_games = int(vs_opp_games_counts.iloc[idx]) if not pd.isna(vs_opp_games_counts.iloc[idx]) else 0

                t_pace = float(team_pace.iloc[idx]) if not pd.isna(team_pace.iloc[idx]) else None
                o_pace = float(opp_pace.iloc[idx]) if not pd.isna(opp_pace.iloc[idx]) else None
                days_rest = float(row.get(Cols.DAYS_REST, 2.0))
                
                per_36 = (s_avg / float(row.get('MIN_SZN_AVG', 36.0))) * 36.0 if row.get('MIN_SZN_AVG', 0) > 0 else 0
                implied_mins = calculate_implied_minutes(line, per_36)
                hist_mins = float(row.get('MIN_L10_AVG', implied_mins))
                min_deviation = abs(implied_mins - hist_mins) / hist_mins if hist_mins > 0 else 0

                proj = smooth_projection(raw_val, s_avg, r_avg, std_dev)
                proj = scale_by_pace(proj, 36.0, t_pace, o_pace, prop_type=prop_cat)
                
                c_bias = cat_bias.get(prop_cat, 0.0)
                proj += (c_bias * 0.30)
                
                p_id = row.get(Cols.PLAYER_ID)
                key = (int(p_id), prop_cat) if not pd.isna(p_id) else (row.get(Cols.PLAYER_NAME), prop_cat)
                proj += (player_bias.get(key, 0.0) * 0.5)
                
                variance = estimate_combo_variance(prop_cat, proj, std_dev)
                
                historic_mae = cat_mae.get(prop_cat, 1.0)
                if historic_mae > 1.5:  
                    variance = variance * (1.0 + (historic_mae * 0.35))

                # ---------------------------------------------------------
                # MATHEMATICAL MATCHUP ADJUSTMENT 
                # ---------------------------------------------------------
                if vs_opp_games >= 4: # Require at least 4 games to care
                    if vs_opp_hit_rate >= 0.75:
                        # Player dominates this matchup. Nudge the projection up.
                        matchup_boost = min((vs_opp_games * 0.015), 0.10) # Max 10% boost
                        proj = proj * (1.0 + matchup_boost)
                        
                    elif vs_opp_hit_rate <= 0.25:
                        # Player struggles in this matchup. Nudge the projection down.
                        matchup_penalty = min((vs_opp_games * 0.015), 0.10)
                        proj = proj * (1.0 - matchup_penalty)
                # ---------------------------------------------------------

                delta_gap_final = abs(proj - line) / line if line > 0 else 0
                
                eval_res = evaluate_prop(proj, line, variance, prop_cat, delta_gap_final, cv, l10_hit_rate, vs_opp_hit_rate, vs_opp_games)
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
                    'Matchup_Hit%': round(vs_opp_hit_rate * 100.0, 1) if vs_opp_games > 0 else 'N/A',
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