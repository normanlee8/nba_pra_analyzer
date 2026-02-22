import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def add_rolling_stats_history(df, stats_to_roll=None):
    """
    Calculates historical rolling features on a dataset.
    CRITICAL: All rolling stats are shifted by 1 to represent "stats entering the game".
    """
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    # Strict Multi-Level Sort to prevent leakage
    sort_cols = [Cols.PLAYER_ID, Cols.DATE]
    if Cols.GAME_ID in df.columns:
        sort_cols.append(Cols.GAME_ID)
        
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'USG_PROXY', 'TS_PCT', 'MIN']
        
    for col in stats_to_roll:
        if col not in df.columns: 
            df[col] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)

    # 1. Base Rolling Averages & Volatility
    for col in stats_to_roll:
        # Expanding (Season) Avg
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().shift(1).values
        
        # Short, Medium, Long Form
        df[f'{col}_L5_AVG'] = grouped[col].rolling(window=5, min_periods=1).mean().shift(1).values
        df[f'{col}_L10_AVG'] = grouped[col].rolling(window=10, min_periods=1).mean().shift(1).values
        df[f'{col}_L20_AVG'] = grouped[col].rolling(window=20, min_periods=1).mean().shift(1).values
        
        # Volatility (Std Dev and Coefficient of Variation)
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().shift(1).values
        # CV normalizes volatility so a 30ppg and 10ppg scorer can be compared
        df[f'{col}_L10_CV'] = np.where(df[f'{col}_L10_AVG'] > 0, 
                                       df[f'{col}_L10_STD'] / df[f'{col}_L10_AVG'], 
                                       0.0)
        
        # Form vs Baseline Ratios (Is player hot or cold?)
        df[f'{col}_FORM_RATIO'] = np.where(df[f'{col}_{Cols.SZN_AVG}'] > 0, 
                                           df[f'{col}_L5_AVG'] / df[f'{col}_{Cols.SZN_AVG}'], 
                                           1.0)

    # 2. Contextual Splits (Home/Away & Rest)
    split_targets = ['PTS', 'REB', 'AST', 'PRA', 'USG_PROXY', 'MIN']
    
    if 'IS_HOME' in df.columns:
        for col in split_targets:
            if col in df.columns:
                # Transform guarantees index alignment. Fallback to SZN Avg if split has no history.
                df[f'{col}_HOME_AWAY_AVG'] = df.groupby([Cols.PLAYER_ID, 'IS_HOME'])[col].transform(
                    lambda x: x.expanding().mean().shift(1)
                ).fillna(df[f'{col}_{Cols.SZN_AVG}'])
                
    if 'Rest_Category' in df.columns:
        for col in split_targets:
            if col in df.columns:
                df[f'{col}_REST_SPLIT_AVG'] = df.groupby([Cols.PLAYER_ID, 'Rest_Category'])[col].transform(
                    lambda x: x.expanding().mean().shift(1)
                ).fillna(df[f'{col}_{Cols.SZN_AVG}'])

    # 3. Rolling Hit Rates (Floor / Ceiling Detection)
    hit_targets = {
        'PTS': [10, 15, 20, 25, 30],
        'REB': [6, 8, 10, 12],
        'AST': [4, 6, 8, 10],
        'PRA': [20, 25, 30, 35, 40]
    }
    
    for stat, thresholds in hit_targets.items():
        if stat in df.columns:
            for t in thresholds:
                temp_hit = (df[stat] >= t).astype(float)
                # Calculates % of games over threshold in the last 10 games
                df[f'{stat}_L10_HitRate_{t}'] = temp_hit.groupby(df[Cols.PLAYER_ID]).transform(
                    lambda x: x.rolling(10, min_periods=1).mean().shift(1)
                )

    return df

def build_feature_set(props_df):
    logging.info("Building feature set with Point-in-Time safety (Leakage Fixed)...")
    
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    if props_df.empty:
        return pd.DataFrame()

    box_scores = loader.load_box_scores()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        try:
            dvp_df = pd.read_parquet(cfg.MASTER_DVP_FILE)
        except Exception as e:
            logging.error(f"Failed to read DVP Parquet: {e}")
            dvp_df = None

    if Cols.PLAYER_ID not in props_df.columns:
        if player_stats_static is not None:
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
            manual_map = {
                'deuce mcbride': 'miles mcbride', 'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort', 'pj washington': 'p.j. washington',
                'jimmy butler': 'jimmy butler iii', 'herb jones': 'herbert jones',
                'robert williams': 'robert williams iii', 'trey murphy': 'trey murphy iii',
                'kelly oubre': 'kelly oubre jr.', 'michael porter': 'michael porter jr.',
                'nick richards': 'nick richards', 'gg jackson': 'gg jackson ii'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
            
            props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            if props_df.empty: 
                logging.warning("No players matched ID map.")
                return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')

    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating Full Game rolling stats...")
        
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        
        if Cols.DATE in box_scores.columns:
            box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])

        history_df = add_rolling_stats_history(box_scores.copy())
        
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        features_df = pd.merge_asof(
            props_df, history_df, on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward', suffixes=('', '_hist')
        )
        
        if player_stats_static is not None:
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')
    else:
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
        
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    if dvp_df is not None:
        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        features_df['Primary_Pos'] = features_df.get('Pos', 'PG').apply(normalize_pos).astype(str)
        if 'Primary_Pos' in dvp_df.columns: dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)
        
        if 'SEASON_ID' not in features_df.columns:
             features_df['yr'] = features_df[Cols.DATE].dt.year
             features_df['mo'] = features_df[Cols.DATE].dt.month
             features_df['season_start'] = np.where(features_df['mo'] > 8, features_df['yr'], features_df['yr'] - 1)
             features_df['SEASON_ID'] = features_df['season_start'].astype(str) + "-" + (features_df['season_start'] + 1).astype(str).str[-2:]
             features_df.drop(columns=['yr', 'mo', 'season_start'], inplace=True)

        if 'SEASON_ID' in dvp_df.columns:
            features_df = pd.merge(
                features_df, dvp_df, 
                left_on=['SEASON_ID', Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['SEASON_ID', 'OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )

    if vs_opp_df is not None and not vs_opp_df.empty:
        features_df = pd.merge(
            features_df, vs_opp_df,
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left'
        )

    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df