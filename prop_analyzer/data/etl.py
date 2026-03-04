import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from rapidfuzz import process, fuzz
from unidecode import unidecode
import warnings

# Import config and Data Contract
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TEAM_NAME_MAP = {
    "Atlanta": "ATL", "Atlanta Hawks": "ATL", "Boston": "BOS", "Boston Celtics": "BOS",
    "Brooklyn": "BKN", "Brooklyn Nets": "BKN", "Charlotte": "CHA", "Charlotte Hornets": "CHA",
    "Chicago": "CHI", "Chicago Bulls": "CHI", "Cleveland": "CLE", "Cleveland Cavaliers": "CLE",
    "Dallas": "DAL", "Dallas Mavericks": "DAL", "Denver": "DEN", "Denver Nuggets": "DEN",
    "Detroit": "DET", "Detroit Pistons": "DET", "Golden State": "GSW", "Golden State Warriors": "GSW",
    "Houston": "HOU", "Houston Rockets": "HOU", "Indiana": "IND", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "Memphis": "MEM", "Memphis Grizzlies": "MEM", "Miami": "MIA", "Miami Heat": "MIA",
    "Milwaukee": "MIL", "Milwaukee Bucks": "MIL", "Minnesota": "MIN", "Minnesota Timberwolves": "MIN",
    "New Orleans": "NOP", "New Orleans Pelicans": "NOP", "New York": "NYK", "New York Knicks": "NYK",
    "Okla City": "OKC", "Oklahoma City Thunder": "OKC", "Orlando": "ORL", "Orlando Magic": "ORL",
    "Philadelphia": "PHI", "Philadelphia 76ers": "PHI", "Phoenix": "PHX", "Phoenix Suns": "PHX",
    "Portland": "POR", "Portland Trail Blazers": "POR", "Sacramento": "SAC", "Sacramento Kings": "SAC",
    "San Antonio": "SAS", "San Antonio Spurs": "SAS", "Toronto": "TOR", "Toronto Raptors": "TOR",
    "Utah": "UTA", "Utah Jazz": "UTA", "Washington": "WAS", "Washington Wizards": "WAS",
}

BBREF_COLUMN_MAP = {
    'G': 'SEASON_G', 'PTS': 'SEASON_PTS', 'TRB': 'SEASON_TRB', 'AST': 'SEASON_AST', 'Pos': 'Position'
}

def get_season_folders(data_dir):
    folders = [f for f in data_dir.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)]
    return sorted(folders)

def load_clean_data(filepath_stem, required_cols=[]):
    if isinstance(filepath_stem, Path):
        path_str = str(filepath_stem)
        base = re.sub(r'\.(csv|parquet)$', '', path_str)
    else:
        base = str(filepath_stem)
    
    parquet_path = Path(base + ".parquet")
    csv_path = Path(base + ".csv")

    try:
        df = None
        if parquet_path.exists(): df = pd.read_parquet(parquet_path)
        elif csv_path.exists(): df = pd.read_csv(csv_path, low_memory=False)
        else: return None
            
        if df is not None and not df.empty and required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing: return None 
        return df
    except Exception as e:
        logging.error(f"Error loading {base}: {e}")
        return None

def get_metric_from_filename(filename, prefix="NBA Team "):
    clean_name = re.sub(r'\.(csv|parquet)$', '', filename)
    if not clean_name.startswith(prefix): return None
    return clean_name[len(prefix):]

def create_player_id_map(data_dir, season_folders):
    logging.info("Creating Player ID Map from ESPN Box Scores...")
    all_player_dfs = []

    for folder in season_folders:
        file_stem = folder / "NBA Player Box Scores"
        df = load_clean_data(file_stem, required_cols=['ESPN_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION'])
        if df is not None:
            players = df[['ESPN_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates(subset=['ESPN_ID'], keep='last')
            all_player_dfs.append(players)
    
    if not all_player_dfs:
        logging.critical("CRITICAL: No valid ESPN box scores found to build ID map.")
        return None
        
    player_map_df = pd.concat(all_player_dfs)
    player_map_df.drop_duplicates(subset=['ESPN_ID'], keep='last', inplace=True)
    player_map_df.rename(columns={'ESPN_ID': Cols.PLAYER_ID}, inplace=True)
    player_map_df[Cols.PLAYER_ID] = pd.to_numeric(player_map_df[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
    player_map_df['Player_Clean'] = player_map_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().strip())
    
    return player_map_df

def process_master_player_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_player_stats (BBref & ID Sync) ---")
    
    id_map_clean = player_id_map[[Cols.PLAYER_ID, 'Player_Clean', 'TEAM_ABBREVIATION', 'PLAYER_NAME']].drop_duplicates(subset=['Player_Clean'])
    name_to_id = id_map_clean.set_index('Player_Clean')[Cols.PLAYER_ID].to_dict()

    def find_match(name):
        if not name: return None
        match = process.extractOne(name, name_to_id.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=88)
        return name_to_id.get(match[0]) if match else None

    for folder in season_folders:
        season_id = folder.name
        try:
            season_player_df = id_map_clean.copy()
            season_player_df['SEASON_ID'] = season_id

            bball_ref_stem = folder / "NBA Player Per Game Averages"
            bball_ref_df = load_clean_data(bball_ref_stem, required_cols=['Player', 'PTS'])
            
            if bball_ref_df is not None:
                bball_ref_df['Player_Clean'] = bball_ref_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                bball_ref_df = bball_ref_df.rename(columns=BBREF_COLUMN_MAP)
                
                if 'Position' in bball_ref_df.columns:
                    bball_ref_df['Position'] = bball_ref_df['Position'].astype(str).apply(lambda x: x.split('-')[0] if '-' in x else x)

                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df['Player_Clean'].apply(find_match)
                bball_ref_df = bball_ref_df[bball_ref_df[Cols.PLAYER_ID].notna()]
                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df[Cols.PLAYER_ID].astype(int)
                bball_ref_df.drop_duplicates(subset=[Cols.PLAYER_ID], keep='first', inplace=True)
                
                season_cols = [Cols.PLAYER_ID, 'Position', 'SEASON_G', 'SEASON_PTS', 'SEASON_TRB', 'SEASON_AST', 'MIN']
                cols_exist = [col for col in season_cols if col in bball_ref_df.columns]
                season_player_df = pd.merge(season_player_df, bball_ref_df[cols_exist], on=Cols.PLAYER_ID, how="left")
                
                adv_stem = folder / "NBA Player Advanced Stats"
                adv_df = load_clean_data(adv_stem, required_cols=['Player', 'USG%'])
                if adv_df is not None:
                    adv_df['Player_Clean'] = adv_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                    adv_df[Cols.PLAYER_ID] = adv_df['Player_Clean'].apply(find_match)
                    adv_df = adv_df[adv_df[Cols.PLAYER_ID].notna()]
                    adv_df[Cols.PLAYER_ID] = adv_df[Cols.PLAYER_ID].astype(int)
                    adv_df.drop_duplicates(subset=[Cols.PLAYER_ID], keep='first', inplace=True)
                    
                    adv_cols = [c for c in [Cols.PLAYER_ID, 'TS%', 'USG%', 'PER', 'AST%', 'TRB%'] if c in adv_df.columns]
                    season_player_df = pd.merge(season_player_df, adv_df[adv_cols], on=Cols.PLAYER_ID, how="left", suffixes=('', '_adv'))

            season_player_df.rename(columns={'Player_Clean': 'clean_name'}, inplace=True)
            out_name = f"master_player_stats_{season_id}.parquet"
            season_player_df.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name}")
            
        except Exception as e:
            logging.error(f"Error processing player stats for {folder}: {e}", exc_info=True)

def process_master_team_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_team_stats ---")
    unique_teams = player_id_map['TEAM_ABBREVIATION'].unique()

    for folder in season_folders:
        season_id = folder.name
        season_team_dfs = []
        files = list(folder.glob("NBA Team *.csv")) + list(folder.glob("NBA Team *.parquet"))
        
        for filepath in files:
            df = load_clean_data(filepath.parent / filepath.stem)
            if df is None or 'Team' not in df.columns: continue
            metric_name = get_metric_from_filename(filepath.name)
            if not metric_name: continue
            
            year_cols = [col for col in df.columns if re.match(r'202\d', str(col))]
            val_col = max(year_cols, key=lambda x: str(x)) if year_cols else (df.columns[2] if len(df.columns) > 2 else None)

            if not val_col: continue

            df['TEAM_ABBREVIATION'] = df['Team'].map(TEAM_NAME_MAP)
            df = df[df['TEAM_ABBREVIATION'].notna()]
            df[metric_name] = pd.to_numeric(df[val_col].astype(str).str.replace(r'[%,]', '', regex=True), errors='coerce')
            season_team_dfs.append(df[['TEAM_ABBREVIATION', metric_name]])

        if season_team_dfs:
            season_master = pd.DataFrame(unique_teams, columns=['TEAM_ABBREVIATION']).dropna()
            for df in season_team_dfs:
                season_master = pd.merge(season_master, df, on='TEAM_ABBREVIATION', how='left')
            
            season_master['SEASON_ID'] = season_id
            out_name = f"master_team_stats_{season_id}.parquet"
            season_master.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name}")

def calculate_historical_vacancy(bs_df, player_df):
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F', 'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT']
    for c in vacancy_cols:
        if c not in bs_df.columns: bs_df[c] = 0.0
        else: bs_df[c] = bs_df[c].fillna(0.0)
    return bs_df

def process_master_box_scores(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_box_scores (Advanced Features) ---")
    
    for folder in season_folders:
        season_id = folder.name
        try:
            file_stem = folder / "NBA Player Box Scores"
            bs_df = load_clean_data(file_stem)
            
            if bs_df is None or bs_df.empty: continue

            if 'ESPN_ID' in bs_df.columns:
                bs_df.rename(columns={'ESPN_ID': Cols.PLAYER_ID}, inplace=True)
            
            if 'PLAYER_NAME' in bs_df.columns:
                bs_df.rename(columns={'PLAYER_NAME': Cols.PLAYER_NAME}, inplace=True)
            
            bs_df.dropna(subset=[Cols.PLAYER_ID], inplace=True)
            bs_df[Cols.PLAYER_ID] = bs_df[Cols.PLAYER_ID].astype(int)
            
            if Cols.GAME_ID in bs_df.columns:
                bs_df[Cols.GAME_ID] = pd.to_numeric(bs_df[Cols.GAME_ID], errors='coerce').fillna(0).astype(int)
            if Cols.DATE in bs_df.columns: 
                bs_df[Cols.DATE] = pd.to_datetime(bs_df[Cols.DATE], errors='coerce')

            if 'MATCHUP' in bs_df.columns:
                bs_df['IS_HOME'] = np.where(bs_df['MATCHUP'].str.contains('@'), 0, 1)
            else:
                bs_df['IS_HOME'] = 1 
            
            bs_df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE], inplace=True)
            bs_df['Days_Between_Games'] = bs_df.groupby(Cols.PLAYER_ID)[Cols.DATE].diff().dt.days
            bs_df['Days_Rest'] = (bs_df['Days_Between_Games'] - 1).fillna(3)
            bs_df['Days_Rest'] = bs_df['Days_Rest'].clip(lower=0, upper=5)
            
            conditions = [bs_df['Days_Rest'] == 0, bs_df['Days_Rest'] == 1, bs_df['Days_Rest'] >= 2]
            choices = ['B2B', '1_Day', '2_Plus_Days']
            bs_df['Rest_Category'] = np.select(conditions, choices, default='Unknown')
            
            if 'TEAM_ABBREVIATION' in bs_df.columns and 'OPPONENT_ABBREV' in bs_df.columns and Cols.DATE in bs_df.columns:
                team_games = bs_df[['TEAM_ABBREVIATION', Cols.DATE]].drop_duplicates().sort_values(['TEAM_ABBREVIATION', Cols.DATE])
                team_games['OPP_DAYS_REST'] = team_games.groupby('TEAM_ABBREVIATION')[Cols.DATE].diff().dt.days.fillna(3.0)
                team_games['OPP_DAYS_REST'] = team_games['OPP_DAYS_REST'].clip(upper=7.0) 
                team_games['OPP_IS_B2B'] = np.where(team_games['OPP_DAYS_REST'] <= 1, 1.0, 0.0)
                
                bs_df = pd.merge(bs_df, team_games, left_on=['OPPONENT_ABBREV', Cols.DATE], right_on=['TEAM_ABBREVIATION', Cols.DATE], how='left', suffixes=('', '_opp_drop'))
                if 'TEAM_ABBREVIATION_opp_drop' in bs_df.columns:
                    bs_df.drop(columns=['TEAM_ABBREVIATION_opp_drop'], inplace=True)
                
                bs_df['OPP_DAYS_REST'] = bs_df['OPP_DAYS_REST'].fillna(3.0)
                bs_df['OPP_IS_B2B'] = bs_df['OPP_IS_B2B'].fillna(0.0)

            p_stats_path = output_dir / f"master_player_stats_{season_id}.parquet"
            if p_stats_path.exists():
                p_stats = pd.read_parquet(p_stats_path)
                if Cols.PLAYER_ID in p_stats.columns:
                    p_stats[Cols.PLAYER_ID] = pd.to_numeric(p_stats[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                    pos_col = 'Position' if 'Position' in p_stats.columns else ('Pos' if 'Pos' in p_stats.columns else None)
                    if pos_col:
                        p_stats_szn = p_stats[[Cols.PLAYER_ID, pos_col]].drop_duplicates(subset=[Cols.PLAYER_ID])
                        if pos_col != 'Position':
                            p_stats_szn.rename(columns={pos_col: 'Position'}, inplace=True)
                        
                        if 'Position' not in bs_df.columns:
                            bs_df = pd.merge(bs_df, p_stats_szn, on=Cols.PLAYER_ID, how='left')

            numeric_cols = ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'TOV', 'MIN', 'STL', 'BLK']
            for col in numeric_cols:
                if col in bs_df.columns: bs_df[col] = pd.to_numeric(bs_df[col], errors='coerce').fillna(0)
            
            bs_df['PRA'] = bs_df['PTS'] + bs_df['REB'] + bs_df['AST']
            bs_df['PR'] = bs_df['PTS'] + bs_df['REB']
            bs_df['PA'] = bs_df['PTS'] + bs_df['AST']
            bs_df['RA'] = bs_df['REB'] + bs_df['AST']
            bs_df['STOCKS'] = bs_df.get('STL', 0) + bs_df.get('BLK', 0)

            ts_denom = 2 * (bs_df['FGA'] + 0.44 * bs_df['FTA'])
            bs_df['TS_PCT'] = np.where(ts_denom > 0, bs_df['PTS'] / ts_denom, 0.0)
            
            usg_num = (bs_df['FGA'] + 0.44 * bs_df['FTA'] + bs_df['TOV'])
            bs_df['USG_PROXY'] = np.where(bs_df['MIN'] > 0, (usg_num / bs_df['MIN']) * 100, 0.0)

            per_36_cols = ['PTS', 'REB', 'AST', 'PRA', 'USG_PROXY']
            for col in per_36_cols:
                if col in bs_df.columns:
                    bs_df[f'{col}_PER36'] = np.where(bs_df['MIN'] >= 5, (bs_df[col] / bs_df['MIN']) * 36, 0.0).round(2)

            bs_df['SEASON_ID'] = season_id
            if p_stats_path.exists():
                bs_df = calculate_historical_vacancy(bs_df, pd.read_parquet(p_stats_path))

            split_stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'MIN']
            for col in split_stat_cols:
                if col in bs_df.columns:
                    bs_df[f'{col}_SPLIT_AVG'] = bs_df.groupby([Cols.PLAYER_ID, 'IS_HOME'])[col].transform(
                        lambda x: x.expanding().mean().shift(1)
                    ).fillna(0.0).round(2)

            if 'OPPONENT_ABBREV' not in bs_df.columns:
                 bs_df['OPPONENT_ABBREV'] = "UNK"

            subset_cols = [Cols.PLAYER_ID, Cols.DATE]
            if Cols.GAME_ID in bs_df.columns: subset_cols.insert(1, Cols.GAME_ID)
            
            bs_df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
            out_name = f"master_box_scores_{season_id}.parquet"
            bs_df.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name} ({len(bs_df)} rows)")
            
        except Exception as e:
            logging.error(f"Error processing box scores for {season_id}: {e}", exc_info=True)

def process_vs_opponent_stats(data_dir, output_dir):
    logging.info("--- Starting: process_vs_opponent_stats ---")
    all_files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not all_files: return

    dfs = []
    for f in all_files:
        try: dfs.append(pd.read_parquet(f))
        except: pass
    
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    agg_cols = {k: 'mean' for k in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'MIN'] if k in df.columns}
    if Cols.GAME_ID in df.columns: agg_cols[Cols.GAME_ID] = 'count'
    
    name_col = Cols.PLAYER_NAME if Cols.PLAYER_NAME in df.columns else 'PLAYER_NAME'
    
    vs_opp_df = df.groupby([Cols.PLAYER_ID, name_col, 'OPPONENT_ABBREV']).agg(agg_cols).reset_index()
    if Cols.GAME_ID in vs_opp_df.columns: vs_opp_df.rename(columns={Cols.GAME_ID: 'GAMES_PLAYED'}, inplace=True)
    
    vs_opp_df.round(2).to_parquet(output_dir / "master_vs_opponent.parquet", index=False)
    logging.info("Saved master_vs_opponent.parquet")

def process_dvp_stats(output_dir):
    logging.info("--- Starting: process_dvp_stats (Advanced Composite - Fixed Target Leakage) ---")
    files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not files: return
    
    all_dvp_dfs = []
    for file_path in files:
        try:
            match = re.search(r'\d{4}-\d{2}', file_path.name)
            season_id = match.group(0) if match else "UNKNOWN"
            
            df = pd.read_parquet(file_path)
            
            team_stats_file = output_dir / f"master_team_stats_{season_id}.parquet"
            team_def_rtg = None
            if team_stats_file.exists():
                ts_df = pd.read_parquet(team_stats_file)
                if 'Defensive Efficiency' in ts_df.columns:
                    team_def_rtg = ts_df[['TEAM_ABBREVIATION', 'Defensive Efficiency']].set_index('TEAM_ABBREVIATION').to_dict()['Defensive Efficiency']

            pos_col = 'Position' if 'Position' in df.columns else ('Pos' if 'Pos' in df.columns else None)
            
            required = [pos_col, 'OPPONENT_ABBREV', Cols.DATE] if pos_col else ['OPPONENT_ABBREV', Cols.DATE]
            if not all(c in df.columns for c in required): continue

            def normalize_pos(pos):
                if not isinstance(pos, str): return 'UNKNOWN'
                p = pos.split('-')[0].upper().strip()
                if p == 'G': return 'SG' 
                if p == 'F': return 'PF' 
                return p
            
            if pos_col:
                df['Primary_Pos'] = df[pos_col].apply(normalize_pos)
            else:
                df['Primary_Pos'] = 'UNKNOWN'
                
            valid_positions = ['PG', 'SG', 'SF', 'PF', 'C']
            df = df[df['Primary_Pos'].isin(valid_positions)].copy()

            df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE], inplace=True)

            stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
            
            for col in stat_cols:
                if col in df.columns:
                    exp_series = df.groupby(Cols.PLAYER_ID)[col].expanding().mean()
                    df[f'{col}_AVG'] = exp_series.groupby(level=0).shift(1).reset_index(level=0, drop=True)
                    
            df.dropna(subset=[f'{c}_AVG' for c in stat_cols if c in df.columns], inplace=True)

            for col in stat_cols:
                if col in df.columns:
                    df[f'{col}_PCT_DIFF'] = np.where(df[f'{col}_AVG'] > 0, 
                                                     (df[col] - df[f'{col}_AVG']) / df[f'{col}_AVG'], 0.0)

            df.sort_values(by=Cols.DATE, inplace=True)
            dvp_group = df.groupby(['OPPONENT_ABBREV', 'Primary_Pos'])
            
            for col in stat_cols:
                if f'{col}_PCT_DIFF' in df.columns:
                    df[f'EXP_MEAN_{col}_DIFF'] = dvp_group[f'{col}_PCT_DIFF'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0.0)
            
            league_pos_baselines = df.groupby('Primary_Pos')[stat_cols].mean().reset_index()
            rename_map = {c: f"{c}_BASE" for c in stat_cols}
            league_pos_baselines.rename(columns=rename_map, inplace=True)
            
            season_dvp = df[[Cols.DATE, 'OPPONENT_ABBREV', 'Primary_Pos']].copy().drop_duplicates(subset=['OPPONENT_ABBREV', 'Primary_Pos', Cols.DATE])
            
            season_dvp = pd.merge(season_dvp, league_pos_baselines, on='Primary_Pos', how='left')
            
            for col in stat_cols:
                if f'EXP_MEAN_{col}_DIFF' in df.columns:
                    diff_map = df.drop_duplicates(subset=['OPPONENT_ABBREV', 'Primary_Pos', Cols.DATE]).set_index(['OPPONENT_ABBREV', 'Primary_Pos', Cols.DATE])[f'EXP_MEAN_{col}_DIFF']
                    season_dvp[f'{col}_PCT_DIFF'] = season_dvp.set_index(['OPPONENT_ABBREV', 'Primary_Pos', Cols.DATE]).index.map(diff_map)
                    
                    season_dvp[f'DVP_{col}_MULTIPLIER'] = 1.0 + season_dvp[f'{col}_PCT_DIFF'].fillna(0.0)
                    season_dvp[f'DVP_{col}'] = season_dvp[f'{col}_BASE'] * season_dvp[f'DVP_{col}_MULTIPLIER']
            
            if team_def_rtg:
                season_dvp['OPP_DEF_EFF'] = season_dvp['OPPONENT_ABBREV'].map(team_def_rtg).fillna(110.0)

            season_dvp['SEASON_ID'] = season_id
            all_dvp_dfs.append(season_dvp)
            
        except Exception as e:
            logging.error(f"Error processing DVP for {file_path.name}: {e}", exc_info=True)

    if all_dvp_dfs:
        final_dvp_all = pd.concat(all_dvp_dfs, ignore_index=True)
        final_dvp_all.sort_values(by=['OPPONENT_ABBREV', 'Primary_Pos', Cols.DATE], inplace=True)
        final_dvp_all.round(3).to_parquet(output_dir / "master_dvp_stats.parquet", index=False)
        logging.info(f"Saved master_dvp_stats.parquet (Multi-Season Time-Series: {len(final_dvp_all)} rows)")

def process_home_away_splits(output_dir):
    logging.info("--- Starting: process_home_away_splits ---")
    all_files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not all_files: return

    dfs = []
    for f in all_files:
        try: 
            df = pd.read_parquet(f)
            match = re.search(r'\d{4}-\d{2}', f.name)
            if match and 'SEASON_ID' not in df.columns:
                df['SEASON_ID'] = match.group(0)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading {f.name} for home/away splits: {e}")
            
    if not dfs: return
    combined_df = pd.concat(dfs, ignore_index=True)
    
    stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'MIN']
    valid_cols = [c for c in stat_cols if c in combined_df.columns]
    
    if not valid_cols: return
    
    name_col = Cols.PLAYER_NAME if Cols.PLAYER_NAME in combined_df.columns else 'PLAYER_NAME'
    
    splits_df = combined_df.groupby(['SEASON_ID', Cols.PLAYER_ID, name_col, 'IS_HOME'])[valid_cols].mean().reset_index()
    
    counts_df = combined_df.groupby(['SEASON_ID', Cols.PLAYER_ID, 'IS_HOME']).size().reset_index(name='GP_SPLIT')
    splits_df = pd.merge(splits_df, counts_df, on=['SEASON_ID', Cols.PLAYER_ID, 'IS_HOME'], how='left')
    
    pivot_cols = valid_cols + ['GP_SPLIT']
    splits_pivot = splits_df.pivot(index=['SEASON_ID', Cols.PLAYER_ID, name_col], columns='IS_HOME', values=pivot_cols)
    
    splits_pivot.columns = [f"{col}_{'HOME' if is_home == 1 else 'AWAY'}" for col, is_home in splits_pivot.columns]
    splits_pivot = splits_pivot.reset_index()
    
    for col in valid_cols:
        home_col = f"{col}_HOME"
        away_col = f"{col}_AWAY"
        if home_col in splits_pivot.columns and away_col in splits_pivot.columns:
            splits_pivot[f"{col}_DIFF"] = splits_pivot[home_col] - splits_pivot[away_col]
            
    splits_pivot.round(2).to_parquet(output_dir / "master_home_away_splits.parquet", index=False)
    logging.info(f"Saved master_home_away_splits.parquet ({len(splits_pivot)} rows)")

def process_pbpstats_data(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_pbpstats_data (Mapping Assist Networks, Foul Risk & WOWY Data) ---")
    id_map_clean = player_id_map[[Cols.PLAYER_ID, 'PLAYER_NAME']].drop_duplicates()
    id_map_clean['Name_Clean'] = id_map_clean['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))

    for folder in season_folders:
        season_id = folder.name
        
        # 1. Map Assist Networks to exact Player IDs
        ast_file = folder / "PBPStats Assist Networks.parquet"
        if ast_file.exists():
            ast_df = pd.read_parquet(ast_file)
            if 'Passer' in ast_df.columns and 'Shooter' in ast_df.columns:
                ast_df['Passer_Clean'] = ast_df['Passer'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
                ast_df['Shooter_Clean'] = ast_df['Shooter'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
                
                ast_df = pd.merge(ast_df, id_map_clean.rename(columns={'Name_Clean': 'Passer_Clean', Cols.PLAYER_ID: 'PASSER_ID'}), on='Passer_Clean', how='left')
                ast_df = pd.merge(ast_df, id_map_clean.rename(columns={'Name_Clean': 'Shooter_Clean', Cols.PLAYER_ID: 'SHOOTER_ID'}), on='Shooter_Clean', how='left')
                
                if 'PASSER_ID' in ast_df.columns:
                    ast_df[Cols.PLAYER_ID] = ast_df['PASSER_ID']
                    
                ast_df.to_parquet(output_dir / f"master_assist_networks_{season_id}.parquet", index=False)
                
        # 2. Map Player Totals for Rotation Trust and Foul Troubleshooting
        pt_file = folder / "PBPStats Player Totals.parquet"
        if pt_file.exists():
            pt_df = pd.read_parquet(pt_file)
            if 'Name' in pt_df.columns:
                pt_df['Name_Clean'] = pt_df['Name'].apply(lambda x: unidecode(str(x)).lower().replace(" ", ""))
                pt_df = pd.merge(pt_df, id_map_clean, on='Name_Clean', how='left')
                pt_df.to_parquet(output_dir / f"master_pbp_player_totals_{season_id}.parquet", index=False)
                
        # 3. Expose Lineup data for WOWY parsing in generator
        lu_file = folder / "PBPStats Lineup Totals.parquet"
        if lu_file.exists():
            pd.read_parquet(lu_file).to_parquet(output_dir / f"master_pbp_lineups_{season_id}.parquet", index=False)

def process_daily_vacancy(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_daily_vacancy (Time-Series) ---")
    if not season_folders: return
    latest_folder = season_folders[-1]
    season_id = latest_folder.name
    
    injuries_path = latest_folder / "daily_injuries.parquet"
    if not injuries_path.exists(): 
        logging.info("No daily injuries report found for current season.")
        return
        
    inj_df = pd.read_parquet(injuries_path)
    if inj_df is None or inj_df.empty: return
    
    if 'Status_Clean' not in inj_df.columns: return
    out_df = inj_df[inj_df['Status_Clean'] == 'OUT'].copy()
    if out_df.empty: return
    
    # TIME SERIES FIX: Ensure Date exists and calculate grouped by Date
    if 'Date' not in out_df.columns:
        out_df['Date'] = pd.Timestamp.today().normalize()
    out_df['Date'] = pd.to_datetime(out_df['Date'])
    
    out_df['clean_name'] = out_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
    id_map_clean = player_id_map[['Player_Clean', Cols.PLAYER_ID]].drop_duplicates()
    out_df = pd.merge(out_df, id_map_clean, left_on='clean_name', right_on='Player_Clean', how='left')
    
    p_stats_path = output_dir / f"master_player_stats_{season_id}.parquet"
    if p_stats_path.exists():
        p_stats = pd.read_parquet(p_stats_path)
        
        stats_to_get = [c for c in ['USG%', 'AST%', 'TRB%', 'MIN', 'Position'] if c in p_stats.columns]
        
        if stats_to_get:
            # FIX: Drop overlapping columns BEFORE merge to prevent _x and _y suffixes
            cols_to_drop = [c for c in stats_to_get if c in out_df.columns]
            if cols_to_drop:
                out_df.drop(columns=cols_to_drop, inplace=True)
                
            out_df = pd.merge(out_df, p_stats[[Cols.PLAYER_ID] + stats_to_get], on=Cols.PLAYER_ID, how='left')
            
            for c in ['USG%', 'AST%', 'TRB%', 'MIN']:
                if c in out_df.columns:
                    out_df[c] = pd.to_numeric(out_df[c], errors='coerce').fillna(0.0)
                    
            min_ratio = out_df.get('MIN', 0.0) / 48.0
            out_df['TEAM_MISSING_USG'] = out_df.get('USG%', 0.0) * min_ratio
            out_df['TEAM_MISSING_AST_PCT'] = out_df.get('AST%', 0.0) * min_ratio
            out_df['TEAM_MISSING_REB_PCT'] = out_df.get('TRB%', 0.0) * min_ratio
            out_df['TEAM_MISSING_MIN'] = out_df.get('MIN', 0.0)
            
            # FIX: Ensure column exists as a Series before astype
            if 'Position' not in out_df.columns:
                out_df['Position'] = 'UNK'
            out_df['Position'] = out_df['Position'].astype(str).str.upper()
            
            out_df['MISSING_USG_G'] = np.where(out_df['Position'].str.contains('G'), out_df['TEAM_MISSING_USG'], 0.0)
            out_df['MISSING_USG_F'] = np.where(out_df['Position'].str.contains('F|C'), out_df['TEAM_MISSING_USG'], 0.0)
                
            agg_dict = {
                'TEAM_MISSING_USG': 'sum',
                'TEAM_MISSING_AST_PCT': 'sum',
                'TEAM_MISSING_REB_PCT': 'sum',
                'TEAM_MISSING_MIN': 'sum',
                'MISSING_USG_G': 'sum',
                'MISSING_USG_F': 'sum'
            }
            
            # Grouping by both Date and Team prevents target leakage across games
            team_vacancy = out_df.groupby(['Date', 'Team']).agg(agg_dict).reset_index()
            team_vacancy.rename(columns={'Team': 'TEAM_ABBREVIATION'}, inplace=True)
            
            team_vacancy.to_parquet(output_dir / "master_daily_vacancy.parquet", index=False)
            logging.info(f"Saved time-series master_daily_vacancy.parquet with {len(team_vacancy)} records.")

    # Hook the Play-By-Play parsing pipeline immediately following the vacancy builder
    process_pbpstats_data(player_id_map, season_folders, output_dir)