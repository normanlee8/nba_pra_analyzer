import pandas as pd
import numpy as np
import logging
import re
from unidecode import unidecode
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader
from prop_analyzer.features.calculator import (
    winsorize_series, calculate_dynamic_hit_rates
)
from prop_analyzer.features.geography import NBA_LOCATIONS, haversine_distance

def add_rolling_stats_history(df, stats_to_roll=None):
    """Calculates historical rolling features, including new CV, Volatility, L1/L3, and Per-36 metrics."""
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
            df[col] = np.nan

    df[Cols.DAYS_REST] = df.groupby(Cols.PLAYER_ID)[Cols.DATE].diff().dt.days.fillna(7.0)

    df_temp = df[[Cols.PLAYER_ID, Cols.DATE]].copy().set_index(Cols.DATE)
    rolling_counts = df_temp.groupby(Cols.PLAYER_ID)[Cols.PLAYER_ID].rolling('7D').count() - 1
    df['Games_in_Last_7_Days'] = rolling_counts.values
    df['Games_in_Last_7_Days'] = df['Games_in_Last_7_Days'].clip(lower=0)

    grouped = df.groupby(Cols.PLAYER_ID)
    new_cols = {}

    for col in stats_to_roll:
        # TARGET LEAKAGE FIX: Shift the series by 1 so we only use past games
        shifted_col = grouped[col].shift(1)
        shifted_grouped = shifted_col.groupby(df[Cols.PLAYER_ID])

        winsorized = shifted_grouped.transform(lambda x: winsorize_series(x, limit=0.10))
        grouped_winsor = winsorized.groupby(df[Cols.PLAYER_ID])
        
        szn_avg = shifted_grouped.expanding().mean().values
        l5_mean = grouped_winsor.rolling(window=5, min_periods=1).mean().values
        
        l1_avg = grouped_winsor.rolling(window=1, min_periods=1).mean().values
        l3_avg = grouped_winsor.rolling(window=3, min_periods=1).median().values
        l5_avg = grouped_winsor.rolling(window=5, min_periods=1).median().values
        l10_avg = grouped_winsor.rolling(window=10, min_periods=1).median().values
        l20_avg = grouped_winsor.rolling(window=20, min_periods=1).median().values
        
        new_cols[f'{col}_{Cols.SZN_AVG}'] = szn_avg
        new_cols[f'{col}_L1_AVG'] = l1_avg
        new_cols[f'{col}_L3_AVG'] = l3_avg
        new_cols[f'{col}_L5_AVG'] = l5_avg
        new_cols[f'{col}_L10_AVG'] = l10_avg
        new_cols[f'{col}_L20_AVG'] = l20_avg
        
        l3_std = shifted_grouped.rolling(window=3, min_periods=2).std().values
        l5_std = shifted_grouped.rolling(window=5, min_periods=2).std().values
        l10_std = shifted_grouped.rolling(window=10, min_periods=3).std().values
        l20_std = shifted_grouped.rolling(window=20, min_periods=5).std().values

        new_cols[f'{col}_L3_STD_DEV'] = l3_std
        new_cols[f'{col}_L5_STD_DEV'] = l5_std
        new_cols[f'{col}_L10_STD_DEV'] = l10_std
        new_cols[f'{col}_L20_STD_DEV'] = l20_std

        new_cols[f'{col}_L3_CV'] = np.divide(l3_std, l3_avg, out=np.zeros_like(l3_std), where=(l3_avg > 0))
        new_cols[f'{col}_L5_CV'] = np.divide(l5_std, l5_avg, out=np.zeros_like(l5_std), where=(l5_avg > 0))
        new_cols[f'{col}_L10_CV'] = np.divide(l10_std, l10_avg, out=np.zeros_like(l10_std), where=(l10_avg > 0))
        new_cols[f'{col}_L20_CV'] = np.divide(l20_std, l20_avg, out=np.zeros_like(l20_std), where=(l20_avg > 0))
        
        form_out = np.ones_like(l5_mean)
        new_cols[f'{col}_FORM_RATIO'] = np.divide(l5_mean, szn_avg, out=form_out, where=(szn_avg > 0))

    # Calculate L3 Minute Deltas (Leading indicator for role increase)
    if f'MIN_L3_AVG' in new_cols and f'MIN_{Cols.SZN_AVG}' in new_cols:
        new_cols['MIN_L3_DELTA'] = new_cols['MIN_L3_AVG'] - new_cols[f'MIN_{Cols.SZN_AVG}']
        
    # Calculate Minute-Normalized Per 36 L5 Stats
    if 'MIN_L5_AVG' in new_cols:
        min_l5 = new_cols['MIN_L5_AVG']
        for col in stats_to_roll:
            if col not in ['MIN', 'USG_PROXY', 'TS_PCT']:
                # SAFE DIVISION: Use np.divide to prevent division by zero evaluation
                base_per_min = np.divide(new_cols[f'{col}_L5_AVG'], min_l5, out=np.zeros_like(min_l5), where=(min_l5 > 0))
                new_cols[f'{col}_L5_PER36'] = base_per_min * 36.0

    # TARGET LEAKAGE FIX: Shift values for correlations as well
    new_cols['PTS_REB_CORR'] = grouped.apply(lambda x: x['PTS'].shift(1).rolling(50, min_periods=5).corr(x['REB'].shift(1)), include_groups=False).reset_index(level=0, drop=True).values
    new_cols['PTS_AST_CORR'] = grouped.apply(lambda x: x['PTS'].shift(1).rolling(50, min_periods=5).corr(x['AST'].shift(1)), include_groups=False).reset_index(level=0, drop=True).values
    new_cols['REB_AST_CORR'] = grouped.apply(lambda x: x['REB'].shift(1).rolling(50, min_periods=5).corr(x['AST'].shift(1)), include_groups=False).reset_index(level=0, drop=True).values

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    split_targets = ['PTS', 'REB', 'AST', 'PRA', 'USG_PROXY', 'MIN']
    new_split_cols = {}
    if 'Rest_Category' in df.columns:
        for col in split_targets:
            if col in df.columns:
                # TARGET LEAKAGE FIX: Shift for Expanding Splits
                val = df.groupby([Cols.PLAYER_ID, 'Rest_Category'])[col].transform(lambda x: x.shift(1).expanding().mean())
                new_split_cols[f'{col}_REST_SPLIT_AVG'] = val.fillna(df[f'{col}_{Cols.SZN_AVG}'])

    if new_split_cols:
        df = pd.concat([df, pd.DataFrame(new_split_cols, index=df.index)], axis=1)

    return df

def add_team_fatigue_and_travel(df):
    """Calculates schedule density, flight miles, and time zone shifts."""
    if 'IS_HOME' not in df.columns:
        df['IS_HOME'] = np.where(df['MATCHUP'].str.contains('@'), 0, 1)
        
    df['GAME_LOCATION_TEAM'] = np.where(df['IS_HOME'] == 1, df['TEAM_ABBREVIATION'], df[Cols.OPPONENT])
    
    # 1. Extract unique team games to calculate schedule metrics without player duplicates
    team_games = df[['TEAM_ABBREVIATION', Cols.DATE, 'GAME_LOCATION_TEAM']].drop_duplicates().sort_values(['TEAM_ABBREVIATION', Cols.DATE])
    
    # 2. Calculate Travel & Timezone Shifts
    team_games['PREV_LOCATION'] = team_games.groupby('TEAM_ABBREVIATION')['GAME_LOCATION_TEAM'].shift(1)
    
    def calculate_travel(row):
        loc1 = row['PREV_LOCATION']
        loc2 = row['GAME_LOCATION_TEAM']
        if pd.isna(loc1) or loc1 not in NBA_LOCATIONS or loc2 not in NBA_LOCATIONS:
            return pd.Series([0.0, 0.0])
            
        c1, c2 = NBA_LOCATIONS[loc1], NBA_LOCATIONS[loc2]
        dist = haversine_distance(c1['lat'], c1['lon'], c2['lat'], c2['lon'])
        tz_shift = c2['tz'] - c1['tz']  # Positive means traveling East (lose sleep), Negative means West
        return pd.Series([dist, tz_shift])
        
    team_games[['FLIGHT_MILES', 'TZ_SHIFT']] = team_games.apply(calculate_travel, axis=1)
    
    # 3. Calculate rolling schedule density (3-in-4, 4-in-6)
    team_games = team_games.set_index(Cols.DATE)
    grouped = team_games.groupby('TEAM_ABBREVIATION')
    
    # How many games were played in the trailing 4, 6, and 7 days (including today)
    team_games['TEAM_GAMES_L4'] = grouped['TEAM_ABBREVIATION'].rolling('4D').count().values
    team_games['TEAM_GAMES_L6'] = grouped['TEAM_ABBREVIATION'].rolling('6D').count().values
    team_games['TEAM_GAMES_L7'] = grouped['TEAM_ABBREVIATION'].rolling('7D').count().values
    
    # Binary fatigue flags
    team_games['IS_3_IN_4'] = np.where(team_games['TEAM_GAMES_L4'] >= 3, 1, 0)
    team_games['IS_4_IN_6'] = np.where(team_games['TEAM_GAMES_L6'] >= 4, 1, 0)
    
    # Flag severe scenarios: Playing on a back-to-back AND changing time zones Eastward
    team_games['IS_TZ_SHOCK'] = np.where((team_games['TEAM_GAMES_L4'] >= 2) & (team_games['TZ_SHIFT'] > 0), 1, 0)
    
    team_games = team_games.reset_index()
    
    # 4. Merge back to main DataFrame
    merge_cols = ['TEAM_ABBREVIATION', Cols.DATE]
    fatigue_features = ['FLIGHT_MILES', 'TZ_SHIFT', 'TEAM_GAMES_L4', 'TEAM_GAMES_L6', 'TEAM_GAMES_L7', 'IS_3_IN_4', 'IS_4_IN_6', 'IS_TZ_SHOCK']
    
    df = pd.merge(df, team_games[merge_cols + fatigue_features], on=merge_cols, how='left')
    
    # Fill NAs for first games of season
    for col in fatigue_features:
        df[col] = df[col].fillna(0)
        
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
            direction='backward', allow_exact_matches=False, suffixes=('', '_hist')
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

    # =============== PLAY-BY-PLAY (PBPStats) DATA INTEGRATION ===============
    season_folders = sorted([f for f in cfg.DATA_DIR.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)])
    if season_folders:
        latest_folder = season_folders[-1]
        szn_id = latest_folder.name
        
        # 1. FOUL TROUBLE & ROTATION TRUST (From PBPStats Totals)
        pbp_tot_file = cfg.DATA_DIR / f"master_pbp_player_totals_{szn_id}.parquet"
        if pbp_tot_file.exists():
            pt_df = pd.read_parquet(pbp_tot_file)
            
            # Foul Trouble Risk
            if 'PersonalFouls' in pt_df.columns and 'OffPoss' in pt_df.columns:
                pt_df['FOUL_RISK_PER_100'] = np.where(pt_df['OffPoss'] > 0, (pt_df['PersonalFouls'] / pt_df['OffPoss']) * 100, 0.0)
            else:
                pt_df['FOUL_RISK_PER_100'] = 0.0
            
            # Coach Trust metric (PBP Minutes / Games Played matching)
            if 'Minutes' in pt_df.columns and 'GamesPlayed' in pt_df.columns:
                pt_df['COACH_TRUST_MINS'] = np.where(pt_df['GamesPlayed'] > 0, pt_df['Minutes'] / pt_df['GamesPlayed'], 0.0)
            else:
                pt_df['COACH_TRUST_MINS'] = 0.0

            merge_cols = [c for c in [Cols.PLAYER_ID, 'FOUL_RISK_PER_100', 'COACH_TRUST_MINS', 'SecondChancePoints'] if c in pt_df.columns]
            if len(merge_cols) > 1:
                features_df = pd.merge(features_df, pt_df[merge_cols].drop_duplicates(subset=[Cols.PLAYER_ID]), on=Cols.PLAYER_ID, how='left')

        # 2. ASSIST NETWORK & WITH/WITHOUT YOU (WOWY) DYNAMIC INJURY LOGIC
        ast_file = cfg.DATA_DIR / f"master_assist_networks_{szn_id}.parquet"
        inj_file = latest_folder / "daily_injuries.parquet"
        lu_file = cfg.DATA_DIR / f"master_pbp_lineups_{szn_id}.parquet"
        
        if inj_file.exists():
            inj_df = pd.read_parquet(inj_file)
            inj_df['Date'] = pd.to_datetime(inj_df.get('Date', pd.Timestamp.today().normalize()))
            out_players = inj_df[inj_df['Status_Clean'].isin(['OUT', 'GTD'])].copy()
            out_players['Name_Clean'] = out_players['Player'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
            
            # -> ASSIST NETWORK DISRUPTION (Point-In-Time Merge)
            if ast_file.exists() and not out_players.empty:
                ast_df = pd.read_parquet(ast_file)
                if 'SHOOTER_ID' in ast_df.columns:
                    ast_df['Shooter_Clean'] = ast_df['Shooter'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
                    total_asts = ast_df.groupby(Cols.PLAYER_ID)['Asts'].sum().reset_index(name='Total_Asts')
                    
                    ast_impact_records = []
                    for dt, grp in out_players.groupby('Date'):
                        out_asts = ast_df[ast_df['Shooter_Clean'].isin(grp['Name_Clean'])]
                        lost_asts = out_asts.groupby(Cols.PLAYER_ID)['Asts'].sum().reset_index(name='Lost_Asts')
                        ast_impact = pd.merge(total_asts, lost_asts, on=Cols.PLAYER_ID, how='left').fillna(0)
                        ast_impact['LOST_AST_SHARE'] = np.where(ast_impact['Total_Asts'] > 0, ast_impact['Lost_Asts'] / ast_impact['Total_Asts'], 0.0)
                        ast_impact['Date'] = dt
                        ast_impact_records.append(ast_impact[[Cols.PLAYER_ID, 'Date', 'LOST_AST_SHARE']])
                    
                    if ast_impact_records:
                        ast_history = pd.concat(ast_impact_records, ignore_index=True).sort_values('Date')
                        features_df = pd.merge_asof(
                            features_df.sort_values(Cols.DATE), 
                            ast_history, 
                            left_on=Cols.DATE, right_on='Date', 
                            by=Cols.PLAYER_ID, direction='backward'
                        )
                        features_df.drop(columns=['Date'], inplace=True, errors='ignore')
                        features_df['LOST_AST_SHARE'] = features_df['LOST_AST_SHARE'].fillna(0.0)
            else:
                features_df['LOST_AST_SHARE'] = 0.0
                
            # -> WOWY TEAM EFFICIENCY IMPACT (Filtering Lineups per Date)
            if lu_file.exists() and pbp_tot_file.exists() and not out_players.empty:
                lu_df = pd.read_parquet(lu_file)
                pt_df = pd.read_parquet(pbp_tot_file)
                pt_df['Name_Clean'] = pt_df['Name'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
                out_mapped = pd.merge(out_players, pt_df[['Name_Clean', 'EntityId']], on='Name_Clean', how='inner')
                
                wowy_records = []
                for dt, grp in out_mapped.groupby('Date'):
                    out_ids = grp['EntityId'].astype(str).tolist()
                    if not out_ids: continue
                    
                    if 'LineupId' in lu_df.columns and 'TeamAbbreviation' in lu_df.columns:
                        def is_lineup_wowy(lineup_id_str):
                            ids_in_lineup = str(lineup_id_str).split('-')
                            # Exact string ID match to prevent substring bugs
                            return not any(out_id in ids_in_lineup for out_id in out_ids)
                        
                        wowy_lineups = lu_df[lu_df['LineupId'].apply(is_lineup_wowy)]
                        wowy_team_stats = wowy_lineups.groupby('TeamAbbreviation').apply(
                            lambda x: pd.Series({
                                'WOWY_OFF_EFF': x['OffPoss'].sum() > 0 and (x['Pts'].sum() / x['OffPoss'].sum()) * 100 or 0,
                                'WOWY_DEF_EFF': x['DefPoss'].sum() > 0 and (x['OppPts'].sum() / x['DefPoss'].sum()) * 100 or 0
                            })
                        ).reset_index()
                        
                        wowy_team_stats.rename(columns={'TeamAbbreviation': 'TEAM_ABBREVIATION'}, inplace=True)
                        wowy_team_stats['Date'] = dt
                        wowy_records.append(wowy_team_stats)
                
                if wowy_records:
                    wowy_history = pd.concat(wowy_records, ignore_index=True).sort_values('Date')
                    features_df = pd.merge_asof(
                        features_df.sort_values(Cols.DATE), 
                        wowy_history, 
                        left_on=Cols.DATE, right_on='Date', 
                        by='TEAM_ABBREVIATION', direction='backward'
                    )
                    features_df.drop(columns=['Date'], inplace=True, errors='ignore')
                else:
                    features_df['WOWY_OFF_EFF'] = np.nan
                    features_df['WOWY_DEF_EFF'] = np.nan
            else:
                features_df['WOWY_OFF_EFF'] = np.nan
                features_df['WOWY_DEF_EFF'] = np.nan
        else:
            features_df['LOST_AST_SHARE'] = 0.0
            features_df['WOWY_OFF_EFF'] = np.nan
            features_df['WOWY_DEF_EFF'] = np.nan
    # ========================================================================

    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating form hit rates and opponent matchup history...")
        bs_sorted = box_scores.sort_values(Cols.DATE)
        
        player_histories = bs_sorted.groupby(Cols.PLAYER_ID).apply(
            lambda df: df.to_dict('records'), include_groups=False
        ).to_dict()

        def compute_row_hit_rates(row, stat_col, default_benchmark_col):
            pid = row.get(Cols.PLAYER_ID)
            dt = row.get(Cols.DATE)
            
            prop_line = row.get(Cols.PROP_LINE)
            if not pd.isna(prop_line) and prop_line > 0 and row.get('PROP_TYPE') == stat_col:
                benchmark = prop_line
            else:
                benchmark = row.get(default_benchmark_col)
                
            opp = row.get(Cols.OPPONENT)
            
            if pd.isna(pid) or pd.isna(benchmark) or pid not in player_histories:
                return pd.Series([np.nan] * 10)
                
            hist = player_histories[pid]
            past_games = [g for g in hist if g[Cols.DATE] < dt]
            if not past_games:
                 return pd.Series([np.nan] * 10)
                 
            stats = [g[stat_col] for g in past_games if stat_col in g and not pd.isna(g[stat_col])]
            rates = calculate_dynamic_hit_rates(stats, benchmark)
            
            past_matchups = [g for g in past_games if g.get(Cols.OPPONENT) == opp]
            matchup_stats = [g[stat_col] for g in past_matchups if stat_col in g and not pd.isna(g[stat_col])]
            matchup_games_count = len(matchup_stats)
            
            vs_opp_legacy_rate = sum(1 for x in matchup_stats if x >= benchmark) / matchup_games_count if matchup_games_count > 0 else np.nan
            vs_opp_over_rate = sum(1 for x in matchup_stats if x > benchmark) / matchup_games_count if matchup_games_count > 0 else np.nan
            vs_opp_under_rate = sum(1 for x in matchup_stats if x < benchmark) / matchup_games_count if matchup_games_count > 0 else np.nan
            
            return pd.Series([
                rates['L5_HIT_RATE'], rates['L10_HIT_RATE'], rates['L20_HIT_RATE'], rates['SZN_HIT_RATE'],
                rates['L10_OVER_RATE'], rates['L10_UNDER_RATE'],
                vs_opp_legacy_rate, vs_opp_over_rate, vs_opp_under_rate, float(matchup_games_count)
            ])

        if 'PROP_TYPE' in features_df.columns:
            hr_cols = ['L5_HIT_RATE', 'L10_HIT_RATE', 'L20_HIT_RATE', 'SZN_HIT_RATE', 'L10_OVER_RATE', 'L10_UNDER_RATE', 'VS_OPP_HIT_RATE', 'VS_OPP_OVER_RATE', 'VS_OPP_UNDER_RATE', 'VS_OPP_GAMES_COUNT']
            features_df[hr_cols] = features_df.apply(
                lambda row: compute_row_hit_rates(row, row.get('PROP_TYPE'), f"{row.get('PROP_TYPE')}_{Cols.SZN_AVG}"), axis=1
            )
            
        for stat in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']:
            benchmark_col = f'{stat}_{Cols.SZN_AVG}'
            if benchmark_col in features_df.columns:
                hr_cols = [f'{stat}_L5_HIT_RATE', f'{stat}_L10_HIT_RATE', f'{stat}_L20_HIT_RATE', f'{stat}_SZN_HIT_RATE', f'{stat}_L10_OVER_RATE', f'{stat}_L10_UNDER_RATE', f'{stat}_VS_OPP_HIT_RATE', f'{stat}_VS_OPP_OVER_RATE', f'{stat}_VS_OPP_UNDER_RATE', f'{stat}_VS_OPP_GAMES_COUNT']
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
        logging.info("Merging Defense vs Position (DvP) Stats safely to avoid leakage...")
        features_df['OPPONENT_ABBREV'] = features_df.get(Cols.OPPONENT, 'UNK')
        
        def normalize_pos(pos):
            if not isinstance(pos, str): return 'UNKNOWN'
            p = pos.split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p
            
        features_df['Primary_Pos'] = features_df['Position'].apply(normalize_pos)
        
        if Cols.DATE in dvp_df.columns and Cols.DATE in features_df.columns:
            dvp_df[Cols.DATE] = pd.to_datetime(dvp_df[Cols.DATE])
            features_df[Cols.DATE] = pd.to_datetime(features_df[Cols.DATE])
            
            dvp_sorted = dvp_df.sort_values(Cols.DATE)
            feat_sorted = features_df.sort_values(Cols.DATE)
            
            features_df = pd.merge_asof(
                feat_sorted, dvp_sorted,
                on=Cols.DATE,
                by=['OPPONENT_ABBREV', 'Primary_Pos'],
                direction='backward',
                allow_exact_matches=False
            )
        else:
            features_df = pd.merge(features_df, dvp_df, on=['OPPONENT_ABBREV', 'Primary_Pos'], how='left')
        
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

    # TIME SERIES FIX: Merge Vacancy point-in-time
    vacancy_path = cfg.DATA_DIR / "master_daily_vacancy.parquet"
    if vacancy_path.exists():
        vacancy_df = pd.read_parquet(vacancy_path)
        if 'Date' in vacancy_df.columns and Cols.DATE in features_df.columns:
            vacancy_df['Date'] = pd.to_datetime(vacancy_df['Date'])
            features_df[Cols.DATE] = pd.to_datetime(features_df[Cols.DATE])
            
            features_df = features_df.sort_values(Cols.DATE)
            vacancy_df = vacancy_df.sort_values('Date')
            
            # Remove any overlapping older columns to prevent _x, _y duplication
            drop_cols = [c for c in vacancy_df.columns if c in features_df.columns and c not in ['TEAM_ABBREVIATION', 'Date']]
            features_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            
            features_df = pd.merge_asof(
                features_df, vacancy_df,
                left_on=Cols.DATE, right_on='Date',
                by='TEAM_ABBREVIATION',
                direction='backward'
            )
            features_df.drop(columns=['Date'], inplace=True, errors='ignore')

    cols_to_fill = [
        'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F',
        'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT'
    ]
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = np.nan

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
                features_df[f'{col}_SPLIT_AVG'] = np.where(
                    features_df['IS_HOME'] == 1,
                    features_df[f'{col}_HOME'],
                    features_df[f'{col}_AWAY']
                )

        features_df['PR_SPLIT_AVG'] = features_df.get('PTS_SPLIT_AVG', np.nan) + features_df.get('REB_SPLIT_AVG', np.nan)
        features_df['PA_SPLIT_AVG'] = features_df.get('PTS_SPLIT_AVG', np.nan) + features_df.get('AST_SPLIT_AVG', np.nan)
        features_df['RA_SPLIT_AVG'] = features_df.get('REB_SPLIT_AVG', np.nan) + features_df.get('AST_SPLIT_AVG', np.nan)

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
            features_df[col] = np.nan
        else:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            median_val = features_df[col].median()
            features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else np.nan)

    # ========================================================================
    # SCHEME SYNERGY MATCHUP LOGIC (SHOT LOCATION ENGINE)
    # ========================================================================
    logging.info("Merging Shot Location Synergy Stats...")
    if season_folders:
        latest_folder = season_folders[-1]
        szn_id = latest_folder.name
        shooting_stats_path = cfg.DATA_DIR / f"{szn_id}/NBA Player Shooting Stats.parquet"
        
        if shooting_stats_path.exists():
            shoot_df = pd.read_parquet(shooting_stats_path)
            
            shoot_cols = {
                Cols.PLAYER_ID: Cols.PLAYER_ID, 
                '0-3': 'FREQ_PAINT',
                '3P': 'FREQ_3PT'
            }
            
            cols_to_keep = [c for c in shoot_cols.keys() if c in shoot_df.columns]
            
            if Cols.PLAYER_ID not in cols_to_keep and 'clean_name' in features_df.columns:
                if 'Player' in shoot_df.columns:
                    shoot_df['clean_name'] = shoot_df['Player'].apply(lambda x: str(x).lower().strip())
                    features_df = pd.merge(features_df, shoot_df[['clean_name'] + [c for c in cols_to_keep if c != Cols.PLAYER_ID]].rename(columns=shoot_cols), on='clean_name', how='left')
            else:
                features_df = pd.merge(features_df, shoot_df[cols_to_keep].rename(columns=shoot_cols), on=Cols.PLAYER_ID, how='left')
                
            features_df['FREQ_PAINT'] = pd.to_numeric(features_df.get('FREQ_PAINT', 0), errors='coerce').fillna(0.0)
            features_df['FREQ_3PT'] = pd.to_numeric(features_df.get('FREQ_3PT', 0), errors='coerce').fillna(0.0)

            # 1. Paint Synergy (Player drives vs Team allows Paint Points)
            if 'OPP_Opponent Points in Paint per Game' in features_df.columns:
                opp_paint_pts = pd.to_numeric(features_df['OPP_Opponent Points in Paint per Game'], errors='coerce').fillna(45.0)
                paint_def_multiplier = opp_paint_pts / opp_paint_pts.median() 
                features_df['SYNERGY_PAINT_EDGE'] = features_df['FREQ_PAINT'] * paint_def_multiplier
            else:
                features_df['SYNERGY_PAINT_EDGE'] = 0.0

            # 2. 3PT Synergy (Player shoots 3s vs Team allows 3s)
            if 'OPP_Opponent Three Pointers Attempted per Game' in features_df.columns:
                opp_3pa = pd.to_numeric(features_df['OPP_Opponent Three Pointers Attempted per Game'], errors='coerce').fillna(35.0)
                perimeter_def_multiplier = opp_3pa / opp_3pa.median()
                features_df['SYNERGY_3PT_EDGE'] = features_df['FREQ_3PT'] * perimeter_def_multiplier
            else:
                features_df['SYNERGY_3PT_EDGE'] = 0.0
                
            # 3. Overall Scheme Fit
            features_df['SCHEME_SYNERGY_SCORE'] = features_df['SYNERGY_PAINT_EDGE'] + features_df['SYNERGY_3PT_EDGE']
        else:
            features_df['SYNERGY_PAINT_EDGE'] = 0.0
            features_df['SYNERGY_3PT_EDGE'] = 0.0
            features_df['SCHEME_SYNERGY_SCORE'] = 0.0
    else:
        features_df['SYNERGY_PAINT_EDGE'] = 0.0
        features_df['SYNERGY_3PT_EDGE'] = 0.0
        features_df['SCHEME_SYNERGY_SCORE'] = 0.0

    # ========================================================================
    # SCHEDULE DENSITY AND TRAVEL FATIGUE ENGINE
    # ========================================================================
    logging.info("Calculating Schedule Density and Travel Fatigue...")
    features_df = add_team_fatigue_and_travel(features_df)

    # 1. Blowout Potential (Net Rating Mismatch) - NON-LINEAR SCALING
    has_eff = all(c in features_df.columns for c in [
        'TEAM_Offensive Efficiency', 'TEAM_Defensive Efficiency', 
        'OPP_Offensive Efficiency', 'OPP_Defensive Efficiency'
    ])
    if has_eff:
        team_net = pd.to_numeric(features_df['TEAM_Offensive Efficiency'], errors='coerce') - pd.to_numeric(features_df['TEAM_Defensive Efficiency'], errors='coerce')
        opp_net = pd.to_numeric(features_df['OPP_Offensive Efficiency'], errors='coerce') - pd.to_numeric(features_df['OPP_Defensive Efficiency'], errors='coerce')
        net_diff = abs(team_net - opp_net).fillna(0.0)
        
        # Override with pure WOWY efficiency differential if available (meaning a star is injured)
        if 'WOWY_OFF_EFF' in features_df.columns and 'WOWY_DEF_EFF' in features_df.columns:
            wowy_net = features_df['WOWY_OFF_EFF'].fillna(team_net) - features_df['WOWY_DEF_EFF'].fillna(team_net)
            # Find the shift the injury caused in the blowout dynamic
            net_diff = abs(wowy_net - opp_net).fillna(0.0)
            
        features_df['BLOWOUT_POTENTIAL'] = np.where(net_diff > 10.0, net_diff ** 2, 0.0)
    else:
        features_df['BLOWOUT_POTENTIAL'] = np.nan

    # 2. Foul Trouble Risk & Matchup Interaction
    if 'OPP_Opponent Personal Fouls per Game' in features_df.columns:
        features_df['OPP_FOUL_DRAW_RATE'] = pd.to_numeric(features_df['OPP_Opponent Personal Fouls per Game'], errors='coerce').fillna(np.nan)
    else:
        features_df['OPP_FOUL_DRAW_RATE'] = np.nan

    if 'FOUL_RISK_PER_100' in features_df.columns and 'OPP_FOUL_DRAW_RATE' in features_df.columns:
        # Calculate the median foul draw rate to create a baseline multiplier
        median_foul_draw = features_df['OPP_FOUL_DRAW_RATE'].median()
        if pd.isna(median_foul_draw) or median_foul_draw <= 0:
            median_foul_draw = 20.0 # Standard NBA fallback average
            
        # Create an opponent multiplier (e.g., Embiid's 76ers might be 1.15, a passive team 0.85)
        foul_draw_multiplier = features_df['OPP_FOUL_DRAW_RATE'] / median_foul_draw
        
        # Interaction Term: Player's inherent foul risk scaled by the opponent's whistle tendency
        features_df['FOUL_TROUBLE_VULNERABILITY'] = features_df['FOUL_RISK_PER_100'].fillna(0.0) * foul_draw_multiplier.fillna(1.0)
    else:
        features_df['FOUL_TROUBLE_VULNERABILITY'] = 0.0

    # 3. Conditioned Usage Proxy (Proportional Distribution)
    usg_col = f'USG_PROXY_{Cols.SZN_AVG}'
    if 'TEAM_MISSING_USG' in features_df.columns and usg_col in features_df.columns:
        missing_usg = pd.to_numeric(features_df['TEAM_MISSING_USG'], errors='coerce').fillna(0.0)
        team_active_usg = np.maximum(100.0 - missing_usg, 20.0)
        absorption_rate = features_df[usg_col] / team_active_usg
        features_df['EXPECTED_USG_SHIFT'] = absorption_rate * missing_usg
    else:
        features_df['EXPECTED_USG_SHIFT'] = 0.0

    # ========================================================================
    # 4. PLAYER-SPECIFIC HISTORICAL MODEL ERROR (BIAS & MAE)
    # ========================================================================
    try:
        # Load all graded history files
        graded_files_dir = cfg.GRADED_PROPS_PARQUET_DIR
        if graded_files_dir.exists():
            graded_files = sorted(graded_files_dir.glob("graded_props_*.parquet"))
            
            if graded_files:
                hist_dfs = [pd.read_parquet(f) for f in graded_files]
                graded_history = pd.concat(hist_dfs, ignore_index=True)
                
                req_cols = [Cols.DATE, Cols.PLAYER_ID, Cols.PROP_TYPE, Cols.PREDICTION, Cols.ACTUAL_VAL]
                if all(c in graded_history.columns for c in req_cols):
                    graded_history[Cols.DATE] = pd.to_datetime(graded_history[Cols.DATE])
                    graded_history[Cols.ACTUAL_VAL] = pd.to_numeric(graded_history[Cols.ACTUAL_VAL], errors='coerce')
                    graded_history[Cols.PREDICTION] = pd.to_numeric(graded_history[Cols.PREDICTION], errors='coerce')
                    
                    # Drop rows missing actual results or predictions
                    graded_history = graded_history.dropna(subset=[Cols.ACTUAL_VAL, Cols.PREDICTION])
                    
                    # Calculate Error: Positive = Model Over-predicted, Negative = Model Under-predicted
                    graded_history['MODEL_ERROR'] = graded_history[Cols.PREDICTION] - graded_history[Cols.ACTUAL_VAL]
                    graded_history['MODEL_ABS_ERROR'] = graded_history['MODEL_ERROR'].abs()
                    
                    graded_history = graded_history.sort_values(Cols.DATE)
                    
                    # Group by Player and Prop Type
                    grp = graded_history.groupby([Cols.PLAYER_ID, Cols.PROP_TYPE])
                    
                    # Calculate 10-game rolling MAE and Bias.
                    # CRITICAL: .shift(1) prevents lookahead leakage (do not use today's error to predict today)
                    graded_history['PLAYER_HISTORIC_MODEL_BIAS'] = grp['MODEL_ERROR'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                    graded_history['PLAYER_HISTORIC_MODEL_MAE'] = grp['MODEL_ABS_ERROR'].transform(lambda x: x.rolling(10, min_periods=1).mean().shift(1))
                    
                    # Pivot so we can merge safely into features_df (which may have multiple stats per row during training)
                    pivot_history = graded_history.pivot_table(
                        index=[Cols.DATE, Cols.PLAYER_ID],
                        columns=Cols.PROP_TYPE,
                        values=['PLAYER_HISTORIC_MODEL_BIAS', 'PLAYER_HISTORIC_MODEL_MAE']
                    ).reset_index()
                    
                    # Flatten MultiIndex columns -> Creates 'PTS_PLAYER_HISTORIC_MODEL_BIAS', etc.
                    pivot_history.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in pivot_history.columns]
                    
                    # Merge point-in-time into the main features_df
                    pivot_history[Cols.DATE] = pd.to_datetime(pivot_history[Cols.DATE])
                    pivot_history = pivot_history.sort_values(Cols.DATE)
                    
                    features_df = pd.merge_asof(
                        features_df.sort_values(Cols.DATE),
                        pivot_history,
                        on=Cols.DATE,
                        by=Cols.PLAYER_ID,
                        direction='backward'
                    )
                    logging.info("Successfully merged Player-Specific Model Error (Bias & MAE).")
    except Exception as e:
        logging.warning(f"Failed to process historical model bias: {e}")

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df