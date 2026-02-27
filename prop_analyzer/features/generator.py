import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader
from prop_analyzer.features.calculator import (
    winsorize_series, calculate_dynamic_hit_rates
)

def add_rolling_stats_history(df, stats_to_roll=None):
    """Calculates historical rolling features, including new CV and Volatility metrics for consistency."""
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    sort_cols = [Cols.PLAYER_ID, Cols.DATE]
    if Cols.GAME_ID in df.columns:
        sort_cols.append(Cols.GAME_ID)
        
    df[Cols.DATE] = pd.to_datetime(df[Cols.DATE])
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'USG_PROXY', 'TS_PCT', 'MIN']
        
    for col in stats_to_roll:
        if col not in df.columns: 
            df[col] = 0.0

    df[Cols.DAYS_REST] = df.groupby(Cols.PLAYER_ID)[Cols.DATE].diff().dt.days.fillna(7.0)

    df_temp = df[[Cols.PLAYER_ID, Cols.DATE]].copy().set_index(Cols.DATE)
    rolling_counts = df_temp.groupby(Cols.PLAYER_ID)[Cols.PLAYER_ID].rolling('7D').count() - 1
    df['Games_in_Last_7_Days'] = rolling_counts.values
    df['Games_in_Last_7_Days'] = df['Games_in_Last_7_Days'].clip(lower=0)

    grouped = df.groupby(Cols.PLAYER_ID)
    new_cols = {}

    for col in stats_to_roll:
        winsorized = grouped[col].transform(lambda x: winsorize_series(x, limit=0.10))
        grouped_winsor = winsorized.groupby(df[Cols.PLAYER_ID])
        
        szn_avg = grouped[col].expanding().mean().shift(1).values
        l5_avg = grouped_winsor.rolling(window=5, min_periods=1).median().shift(1).values
        l10_avg = grouped_winsor.rolling(window=10, min_periods=1).median().shift(1).values
        l20_avg = grouped_winsor.rolling(window=20, min_periods=1).median().shift(1).values
        
        new_cols[f'{col}_{Cols.SZN_AVG}'] = szn_avg
        new_cols[f'{col}_L5_AVG'] = l5_avg
        new_cols[f'{col}_L10_AVG'] = l10_avg
        new_cols[f'{col}_L20_AVG'] = l20_avg
        
        l5_std = grouped[col].rolling(window=5, min_periods=2).std().shift(1).values
        l10_std = grouped[col].rolling(window=10, min_periods=3).std().shift(1).values
        l20_std = grouped[col].rolling(window=20, min_periods=5).std().shift(1).values

        new_cols[f'{col}_L5_STD_DEV'] = l5_std
        new_cols[f'{col}_L10_STD_DEV'] = l10_std
        new_cols[f'{col}_L20_STD_DEV'] = l20_std

        new_cols[f'{col}_L5_CV'] = np.divide(l5_std, l5_avg, out=np.zeros_like(l5_std), where=(l5_avg > 0))
        new_cols[f'{col}_L10_CV'] = np.divide(l10_std, l10_avg, out=np.zeros_like(l10_std), where=(l10_avg > 0))
        new_cols[f'{col}_L20_CV'] = np.divide(l20_std, l20_avg, out=np.zeros_like(l20_std), where=(l20_avg > 0))
        
        form_out = np.ones_like(l5_avg)
        new_cols[f'{col}_FORM_RATIO'] = np.divide(l5_avg, szn_avg, out=form_out, where=(szn_avg > 0))

    new_cols['PTS_REB_CORR'] = grouped.apply(lambda x: x['PTS'].rolling(50, min_periods=5).corr(x['REB']).shift(1), include_groups=False).reset_index(level=0, drop=True).fillna(0.1).values
    new_cols['PTS_AST_CORR'] = grouped.apply(lambda x: x['PTS'].rolling(50, min_periods=5).corr(x['AST']).shift(1), include_groups=False).reset_index(level=0, drop=True).fillna(0.1).values
    new_cols['REB_AST_CORR'] = grouped.apply(lambda x: x['REB'].rolling(50, min_periods=5).corr(x['AST']).shift(1), include_groups=False).reset_index(level=0, drop=True).fillna(0.1).values

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    split_targets = ['PTS', 'REB', 'AST', 'PRA', 'USG_PROXY', 'MIN']
    new_split_cols = {}
    if 'Rest_Category' in df.columns:
        for col in split_targets:
            if col in df.columns:
                val = df.groupby([Cols.PLAYER_ID, 'Rest_Category'])[col].transform(lambda x: x.expanding().mean().shift(1))
                new_split_cols[f'{col}_REST_SPLIT_AVG'] = val.fillna(df[f'{col}_{Cols.SZN_AVG}'])

    if new_split_cols:
        df = pd.concat([df, pd.DataFrame(new_split_cols, index=df.index)], axis=1)

    return df


def build_feature_set(props_df):
    logging.info("Building Probability-Optimized feature set...")
    
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    dvp_df = loader.load_dvp_stats()
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

    if 'Position' not in features_df.columns:
        if 'Pos' in features_df.columns:
            features_df['Position'] = features_df['Pos']
        elif 'POSITION' in features_df.columns:
            features_df['Position'] = features_df['POSITION']
        else:
            features_df['Position'] = 'UNK'
            
    features_df['Position'] = features_df['Position'].astype(str).apply(lambda x: x.split('-')[0] if '-' in x else x)

    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating form hit rates and opponent matchup history...")
        bs_sorted = box_scores.sort_values(Cols.DATE)
        
        player_histories = bs_sorted.groupby(Cols.PLAYER_ID).apply(
            lambda df: df.to_dict('records'), include_groups=False
        ).to_dict()

        def compute_row_hit_rates(row, stat_col, benchmark_col):
            pid = row.get(Cols.PLAYER_ID)
            dt = row.get(Cols.DATE)
            benchmark = row.get(benchmark_col)
            opp = row.get(Cols.OPPONENT)
            
            if pd.isna(pid) or pd.isna(benchmark) or pid not in player_histories:
                return pd.Series([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
                
            hist = player_histories[pid]
            past_games = [g for g in hist if g[Cols.DATE] < dt]
            if not past_games:
                 return pd.Series([0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
                 
            stats = [g[stat_col] for g in past_games if stat_col in g and not pd.isna(g[stat_col])]
            rates = calculate_dynamic_hit_rates(stats, benchmark)
            
            past_matchups = [g for g in past_games if g.get(Cols.OPPONENT) == opp]
            matchup_stats = [g[stat_col] for g in past_matchups if stat_col in g and not pd.isna(g[stat_col])]
            matchup_games_count = len(matchup_stats)
            vs_opp_hit_rate = sum(1 for x in matchup_stats if x >= benchmark) / matchup_games_count if matchup_games_count > 0 else 0.50
            
            return pd.Series([
                rates['L5_HIT_RATE'], rates['L10_HIT_RATE'], rates['L20_HIT_RATE'], rates['SZN_HIT_RATE'],
                vs_opp_hit_rate, float(matchup_games_count)
            ])

        if 'PROP_TYPE' in features_df.columns:
            hr_cols = ['L5_HIT_RATE', 'L10_HIT_RATE', 'L20_HIT_RATE', 'SZN_HIT_RATE', 'VS_OPP_HIT_RATE', 'VS_OPP_GAMES_COUNT']
            features_df[hr_cols] = features_df.apply(
                lambda row: compute_row_hit_rates(row, row.get('PROP_TYPE'), f"{row.get('PROP_TYPE')}_{Cols.SZN_AVG}"), axis=1
            )
            
        for stat in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']:
            benchmark_col = f'{stat}_{Cols.SZN_AVG}'
            if benchmark_col in features_df.columns:
                hr_cols = [f'{stat}_L5_HIT_RATE', f'{stat}_L10_HIT_RATE', f'{stat}_L20_HIT_RATE', f'{stat}_SZN_HIT_RATE', f'{stat}_VS_OPP_HIT_RATE', f'{stat}_VS_OPP_GAMES_COUNT']
                features_df[hr_cols] = features_df.apply(
                    lambda row: compute_row_hit_rates(row, stat, benchmark_col), axis=1
                )

    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
        
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    if dvp_df is not None and not dvp_df.empty:
        logging.info("Merging Defense vs Position (DvP) Stats...")
        features_df['OPPONENT_ABBREV'] = features_df.get(Cols.OPPONENT, 'UNK')
        
        def normalize_pos(pos):
            if not isinstance(pos, str): return 'UNKNOWN'
            p = pos.split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p
            
        features_df['Primary_Pos'] = features_df['Position'].apply(normalize_pos)
        
        if 'SEASON_ID' in dvp_df.columns:
            latest_szn = dvp_df['SEASON_ID'].max()
            dvp_to_merge = dvp_df[dvp_df['SEASON_ID'] == latest_szn].drop(columns=['SEASON_ID'])
        else:
            dvp_to_merge = dvp_df
            
        features_df = pd.merge(features_df, dvp_to_merge, on=['OPPONENT_ABBREV', 'Primary_Pos'], how='left')
        
        dvp_cols = [c for c in features_df.columns if c.startswith('DVP_') and 'MULTIPLIER' in c]
        for c in dvp_cols:
            features_df[c] = features_df[c].fillna(1.0)

    if vs_opp_df is not None and not vs_opp_df.empty:
        logging.info("Merging historical VS Opponent stats...")
        if Cols.OPPONENT in features_df.columns and Cols.OPPONENT in vs_opp_df.columns:
            features_df = pd.merge(features_df, vs_opp_df, on=[Cols.PLAYER_ID, Cols.OPPONENT], how='left')

    if 'MATCHUP' in features_df.columns and 'IS_HOME' not in features_df.columns:
        features_df['IS_HOME'] = np.where(features_df['MATCHUP'].str.contains('@'), 0, 1)
    elif 'IS_HOME' not in features_df.columns:
        features_df['IS_HOME'] = 1  

    if Cols.OPPONENT in features_df.columns:
        features_df['IS_ALTITUDE'] = np.where(features_df[Cols.OPPONENT].isin(['DEN', 'UTA']), 1.0, 0.0)

    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
    if 'OPP_Possessions per Game' in features_df.columns:
        features_df['OPP_GAME_PACE'] = features_df['OPP_Possessions per Game']
        
    if 'OPP_GAME_PACE' in features_df.columns and 'Primary_Pos' in features_df.columns:
        features_df['PACE_PG_INTERACTION'] = np.where(features_df['Primary_Pos'] == 'PG', features_df['OPP_GAME_PACE'], 0.0)

    # --- NEW: Safely Merge Active Daily Vacancy (No Historical Leakage) ---
    vacancy_path = cfg.DATA_DIR / "master_daily_vacancy.parquet"
    if vacancy_path.exists():
        vacancy_df = pd.read_parquet(vacancy_path)
        
        if Cols.DATE in features_df.columns:
            # We ONLY apply today's injury report to today's lines (Target Leakage Fix)
            max_date = features_df[Cols.DATE].max()
            today_mask = features_df[Cols.DATE] == max_date
            
            drop_cols = [c for c in vacancy_df.columns if c in features_df.columns and c != 'TEAM_ABBREVIATION']
            features_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            
            features_with_vacancy = pd.merge(features_df[today_mask], vacancy_df, on='TEAM_ABBREVIATION', how='left')
            features_df = pd.concat([features_df[~today_mask], features_with_vacancy], ignore_index=True)
        else:
            drop_cols = [c for c in vacancy_df.columns if c in features_df.columns and c != 'TEAM_ABBREVIATION']
            features_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            features_df = pd.merge(features_df, vacancy_df, on='TEAM_ABBREVIATION', how='left')

    cols_to_fill = [
        'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F',
        'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT'
    ]
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    if 'OPP_DAYS_REST' not in features_df.columns: features_df['OPP_DAYS_REST'] = 2.0
    features_df['OPP_DAYS_REST'] = features_df['OPP_DAYS_REST'].fillna(2.0)
    
    if 'OPP_IS_B2B' not in features_df.columns: features_df['OPP_IS_B2B'] = 0.0
    features_df['OPP_IS_B2B'] = features_df['OPP_IS_B2B'].fillna(0.0)

    splits_path = cfg.DATA_DIR / "master_home_away_splits.parquet"
    if splits_path.exists():
        splits_df = pd.read_parquet(splits_path)
        if 'SEASON_ID' in splits_df.columns:
            latest_szn = splits_df['SEASON_ID'].max()
            splits_df = splits_df[splits_df['SEASON_ID'] == latest_szn].drop(columns=['SEASON_ID', Cols.PLAYER_NAME], errors='ignore')
            
        features_df = pd.merge(features_df, splits_df, on=Cols.PLAYER_ID, how='left')
        
        stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'MIN']
        for col in stat_cols:
            if f'{col}_HOME' in features_df.columns and f'{col}_AWAY' in features_df.columns:
                features_df[f'{col}_HOME'] = features_df[f'{col}_HOME'].fillna(0.0)
                features_df[f'{col}_AWAY'] = features_df[f'{col}_AWAY'].fillna(0.0)
                features_df[f'{col}_DIFF'] = features_df[f'{col}_DIFF'].fillna(0.0)
                
                features_df[f'{col}_SPLIT_AVG'] = np.where(
                    features_df['IS_HOME'] == 1,
                    features_df[f'{col}_HOME'],
                    features_df[f'{col}_AWAY']
                )

        features_df['PR_SPLIT_AVG'] = features_df.get('PTS_SPLIT_AVG', 0) + features_df.get('REB_SPLIT_AVG', 0)
        features_df['PA_SPLIT_AVG'] = features_df.get('PTS_SPLIT_AVG', 0) + features_df.get('AST_SPLIT_AVG', 0)
        features_df['RA_SPLIT_AVG'] = features_df.get('REB_SPLIT_AVG', 0) + features_df.get('AST_SPLIT_AVG', 0)

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

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df