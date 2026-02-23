import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader
from prop_analyzer.features.calculator import winsorize_series, calculate_implied_minutes

def add_rolling_stats_history(df, stats_to_roll=None):
    """Calculates historical rolling features on a dataset."""
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    sort_cols = [Cols.PLAYER_ID, Cols.DATE]
    if Cols.GAME_ID in df.columns:
        sort_cols.append(Cols.GAME_ID)
        
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'USG_PROXY', 'TS_PCT', 'MIN']
        
    for col in stats_to_roll:
        if col not in df.columns: 
            df[col] = 0.0

    # Calculate Days Rest
    df[Cols.DAYS_REST] = df.groupby(Cols.PLAYER_ID)[Cols.DATE].diff().dt.days.fillna(7.0)

    grouped = df.groupby(Cols.PLAYER_ID)

    # 1. Base Rolling Averages & Volatility
    for col in stats_to_roll:
        df[f'{col}_WINSOR'] = grouped[col].transform(lambda x: winsorize_series(x, limit=0.10))
        grouped_winsor = df.groupby(Cols.PLAYER_ID)[f'{col}_WINSOR']
        
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().shift(1).values
        df[f'{col}_L5_AVG'] = grouped_winsor.rolling(window=5, min_periods=1).median().shift(1).values
        df[f'{col}_L10_AVG'] = grouped_winsor.rolling(window=10, min_periods=1).median().shift(1).values
        df[f'{col}_L20_AVG'] = grouped_winsor.rolling(window=20, min_periods=1).median().shift(1).values
        
        df[f'{col}_L10_STD_DEV'] = grouped[col].rolling(window=10, min_periods=3).std().shift(1).values
        df[f'{col}_L10_CV'] = np.where(df[f'{col}_L10_AVG'] > 0, 
                                       df[f'{col}_L10_STD_DEV'] / df[f'{col}_L10_AVG'], 
                                       0.0)
        
        df[f'{col}_FORM_RATIO'] = np.where(df[f'{col}_{Cols.SZN_AVG}'] > 0, 
                                           df[f'{col}_L5_AVG'] / df[f'{col}_{Cols.SZN_AVG}'], 
                                           1.0)

    # 2. Contextual Splits
    split_targets = ['PTS', 'REB', 'AST', 'PRA', 'USG_PROXY', 'MIN']
    if 'IS_HOME' in df.columns:
        for col in split_targets:
            if col in df.columns:
                df[f'{col}_HOME_AWAY_AVG'] = df.groupby([Cols.PLAYER_ID, 'IS_HOME'])[col].transform(
                    lambda x: x.expanding().mean().shift(1)
                ).fillna(df[f'{col}_{Cols.SZN_AVG}'])
                
    if 'Rest_Category' in df.columns:
        for col in split_targets:
            if col in df.columns:
                df[f'{col}_REST_SPLIT_AVG'] = df.groupby([Cols.PLAYER_ID, 'Rest_Category'])[col].transform(
                    lambda x: x.expanding().mean().shift(1)
                ).fillna(df[f'{col}_{Cols.SZN_AVG}'])

    drop_cols = [c for c in df.columns if c.endswith('_WINSOR')]
    df.drop(columns=drop_cols, inplace=True)

    return df

def build_feature_set(props_df):
    logging.info("Building EV feature set...")
    
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    box_scores = loader.load_box_scores()
    
    if props_df.empty: return pd.DataFrame()

    if Cols.PLAYER_ID not in props_df.columns and player_stats_static is not None:
        name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
        props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
        props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
        props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
        if not props_df.empty: props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')

    if box_scores is not None and not box_scores.empty:
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        if Cols.DATE in box_scores.columns: box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])

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
            cols_to_use = [c for c in player_stats_static.columns if c not in features_df.columns or c == Cols.PLAYER_ID]
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

    # Mapping Pace
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
    if 'OPP_Possessions per Game' in features_df.columns:
        features_df['OPP_GAME_PACE'] = features_df['OPP_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    # --- NEW: Safe Imputation for Advanced Stats ---
    # Fills missing values with the league median so ML Models never crash on missing data
    advanced_stats = [
        'OPP_Opponent Effective Field Goal %', 'OPP_Opponent True Shooting %',
        'TEAM_Field Goals Attempted per Game', 'OPP_Field Goals Attempted per Game',
        'TEAM_Three Pointers Attempted per Game', 'OPP_Three Pointers Attempted per Game',
        'OPP_Opponent Offensive Rebounding %', 'TEAM_Assists per FGM', 
        'OPP_Opponent Assists per FGM', 'TEAM_Assist to Turnover Ratio',
        'OPP_Opponent Points in Paint per Game', 'OPP_Opponent Percent of Points from 3 Pointers',
        'OPP_Opponent Personal Fouls per Game', 'OPP_Opponent Fastbreak Points per Game',
        'TEAM_Extra Scoring Chances per Game', 'OPP_Extra Scoring Chances per Game',
        'OPP_Opponent Points + Rebounds + Assists per Game', 'OPP_Opponent Points + Assists per Game'
    ]
    
    for col in advanced_stats:
        if col not in features_df.columns:
            features_df[col] = 0.0
        else:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            median_val = features_df[col].median()
            features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

    # Standardize Position
    if 'Position' not in features_df.columns:
        if 'Pos' in features_df.columns:
            features_df['Position'] = features_df['Pos']
        elif 'POSITION' in features_df.columns:
            features_df['Position'] = features_df['POSITION']
        else:
            features_df['Position'] = 'UNK'
            
    features_df['Position'] = features_df['Position'].astype(str).apply(lambda x: x.split('-')[0] if '-' in x else x)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df