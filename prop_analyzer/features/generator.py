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
    CRITICAL FIX: All rolling stats are shifted by 1 to represent "stats entering the game".
    """
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    # CRITICAL FIX: Strict Multi-Level Sort to prevent leakage
    # We sort by Player -> Date -> GameID (if avail) to ensure deterministic order
    sort_cols = [Cols.PLAYER_ID, Cols.DATE]
    if Cols.GAME_ID in df.columns:
        sort_cols.append(Cols.GAME_ID)
        
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = [
            'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA'
        ]
        
    # Ensure stats exist (fill missing with 0 to prevent errors)
    for col in stats_to_roll:
        if col not in df.columns: 
            df[col] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)

    for col in stats_to_roll:
        # --- CRITICAL FIX: .shift(1) applied to all aggregations ---
        # This ensures the feature for Game X only contains data from Games 1 to X-1.
        
        # SZN Avg (Expanding Mean)
        # Note: min_periods=1 allows the first game to have a value (likely NaN after shift, handled by Imputer)
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().shift(1).values
        
        # L5 Avg (Rolling 5)
        df[f'{col}_{Cols.L5_AVG}'] = grouped[col].rolling(window=5, min_periods=1).mean().shift(1).values
        
        # L10 Std Dev
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().shift(1).values
        
        # EWMA (Exponential Weighted Moving Average)
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().shift(1).values

    # Advanced Stats (Only if present)
    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = grouped['USG_PROXY'].expanding().mean().shift(1).values
        df['L5_USG_PROXY'] = grouped['USG_PROXY'].rolling(window=5).mean().shift(1).values
        
    if 'TS_PCT' in df.columns:
        df['SZN_TS_PCT'] = grouped['TS_PCT'].expanding().mean().shift(1).values
        
    return df

def build_feature_set(props_df):
    logging.info("Building feature set with Point-in-Time safety (Leakage Fixed)...")
    
    # 1. Load Data (Optimized: Load once)
    # Note: loader functions should already implement caching, but we ensure explicit single calls here.
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    # Only load heavy history files if we actually have props to process
    if props_df.empty:
        return pd.DataFrame()

    box_scores = loader.load_box_scores()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        try:
            dvp_df = pd.read_parquet(cfg.MASTER_DVP_FILE)
            # Ensure Season ID exists for correct merging
            if 'SEASON_ID' not in dvp_df.columns:
                # If missing (legacy file), assume current season or drop
                logging.warning("DVP file missing SEASON_ID. DVP merging might be inaccurate.")
        except Exception as e:
            logging.error(f"Failed to read DVP Parquet: {e}")
            dvp_df = None

    # 2. Map Player Names to IDs
    if Cols.PLAYER_ID not in props_df.columns:
        if player_stats_static is not None:
            # Create cleaner name map
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
            # Manual Mapping overrides
            manual_map = {
                'deuce mcbride': 'miles mcbride',
                'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort',
                'pj washington': 'p.j. washington',
                'jimmy butler': 'jimmy butler iii',
                'herb jones': 'herbert jones',
                'robert williams': 'robert williams iii',
                'trey murphy': 'trey murphy iii',
                'kelly oubre': 'kelly oubre jr.',
                'michael porter': 'michael porter jr.',
                'nick richards': 'nick richards',
                'gg jackson': 'gg jackson ii'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
            
            props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            if props_df.empty: 
                logging.warning("No players matched ID map. Check naming conventions.")
                return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        else:
            logging.error("Static player stats missing. Cannot map IDs.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering (Full Game)
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating Full Game rolling stats...")
        
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        
        if Cols.DATE in box_scores.columns:
            box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])
        elif 'GAME_DATE' in box_scores.columns:
             box_scores[Cols.DATE] = pd.to_datetime(box_scores['GAME_DATE'])

        # Calculate history with shifts
        history_df = add_rolling_stats_history(box_scores.copy())
        
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        # Merge point-in-time stats
        features_df = pd.merge_asof(
            props_df, history_df, on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward', suffixes=('', '_hist')
        )
        
        # Merge Static Stats (Season Avg, etc from current season file)
        if player_stats_static is not None:
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')
    else:
        # Fallback if no history
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    # 4. Merge Team/Opponent Stats (Season-Aware)
    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
        
        # Merge Team Stats
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        # Merge Opponent Stats
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    # 5. Merge DVP (Season-Aware)
    if dvp_df is not None:
        # Standardize Position
        if 'Pos' not in features_df.columns and player_stats_static is not None:
             if Cols.PLAYER_ID in player_stats_static.columns:
                 pos_map = player_stats_static.set_index(Cols.PLAYER_ID)['Pos'].to_dict()
                 features_df['Pos'] = features_df[Cols.PLAYER_ID].map(pos_map).fillna('PG')

        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        features_df['Primary_Pos'] = features_df.get('Pos', 'PG').apply(normalize_pos)
        features_df['Primary_Pos'] = features_df['Primary_Pos'].astype(str)
        
        # Normalize columns for merge
        if 'Primary_Pos' in dvp_df.columns:
            dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)
        
        # Ensure props_df has SEASON_ID
        if 'SEASON_ID' not in features_df.columns:
             features_df['yr'] = features_df[Cols.DATE].dt.year
             features_df['mo'] = features_df[Cols.DATE].dt.month
             features_df['season_start'] = np.where(features_df['mo'] > 8, features_df['yr'], features_df['yr'] - 1)
             features_df['SEASON_ID'] = features_df['season_start'].astype(str) + "-" + (features_df['season_start'] + 1).astype(str).str[-2:]
             features_df.drop(columns=['yr', 'mo', 'season_start'], inplace=True)

        if 'SEASON_ID' in dvp_df.columns:
            # Merge on Season + Opponent + Position
            features_df = pd.merge(
                features_df, dvp_df, 
                left_on=['SEASON_ID', Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['SEASON_ID', 'OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )
        else:
            # Fallback legacy merge
            features_df = pd.merge(
                features_df, dvp_df, 
                left_on=[Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )

    # 6. Merge H2H (Head to Head)
    if vs_opp_df is not None and not vs_opp_df.empty:
        features_df = pd.merge(
            features_df, vs_opp_df,
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left'
        )

    # 7. Final Polish / Fill Vacancy
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_dfs