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
    "Atlanta": "ATL", "Atlanta Hawks": "ATL",
    "Boston": "BOS", "Boston Celtics": "BOS",
    "Brooklyn": "BKN", "Brooklyn Nets": "BKN",
    "Charlotte": "CHA", "Charlotte Hornets": "CHA",
    "Chicago": "CHI", "Chicago Bulls": "CHI",
    "Cleveland": "CLE", "Cleveland Cavaliers": "CLE",
    "Dallas": "DAL", "Dallas Mavericks": "DAL",
    "Denver": "DEN", "Denver Nuggets": "DEN",
    "Detroit": "DET", "Detroit Pistons": "DET",
    "Golden State": "GSW", "Golden State Warriors": "GSW",
    "Houston": "HOU", "Houston Rockets": "HOU",
    "Indiana": "IND", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "Memphis": "MEM", "Memphis Grizzlies": "MEM",
    "Miami": "MIA", "Miami Heat": "MIA",
    "Milwaukee": "MIL", "Milwaukee Bucks": "MIL",
    "Minnesota": "MIN", "Minnesota Timberwolves": "MIN",
    "New Orleans": "NOP", "New Orleans Pelicans": "NOP",
    "New York": "NYK", "New York Knicks": "NYK",
    "Okla City": "OKC", "Oklahoma City Thunder": "OKC",
    "Orlando": "ORL", "Orlando Magic": "ORL",
    "Philadelphia": "PHI", "Philadelphia 76ers": "PHI",
    "Phoenix": "PHX", "Phoenix Suns": "PHX",
    "Portland": "POR", "Portland Trail Blazers": "POR",
    "Sacramento": "SAC", "Sacramento Kings": "SAC",
    "San Antonio": "SAS", "San Antonio Spurs": "SAS",
    "Toronto": "TOR", "Toronto Raptors": "TOR",
    "Utah": "UTA", "Utah Jazz": "UTA",
    "Washington": "WAS", "Washington Wizards": "WAS",
}

BBREF_COLUMN_MAP = {
    'G': 'SEASON_G',
    'PTS': 'SEASON_PTS',
    'TRB': 'SEASON_TRB', 
    'AST': 'SEASON_AST'
}

def get_season_folders(data_dir):
    """
    Finds all season subfolders (e.g., '2024-25', '2025-26') in the data directory.
    Returns sorted list of Path objects.
    """
    folders = [f for f in data_dir.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)]
    return sorted(folders)

def load_clean_data(filepath_stem, required_cols=[]):
    """
    Smart loader: Prefers Parquet, falls back to CSV.
    filepath_stem: Path object without extension OR string.
    """
    if isinstance(filepath_stem, Path):
        path_str = str(filepath_stem)
        base = re.sub(r'\.(csv|parquet)$', '', path_str)
    else:
        base = str(filepath_stem)
    
    parquet_path = Path(base + ".parquet")
    csv_path = Path(base + ".csv")

    try:
        df = None
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            return None
            
        if df is not None and not df.empty and required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return None 
        
        return df
    except Exception as e:
        logging.error(f"Error loading {base}: {e}")
        return None

def get_metric_from_filename(filename, prefix="NBA Team "):
    clean_name = re.sub(r'\.(csv|parquet)$', '', filename)
    if not clean_name.startswith(prefix):
        return None
    return clean_name[len(prefix):]

def create_player_id_map(data_dir, season_folders):
    """
    Builds a global player ID map using the ESPN IDs from the box scores.
    Replaces the old nba_api logic.
    """
    logging.info("Creating Player ID Map from ESPN Box Scores...")
    all_player_dfs = []

    for folder in season_folders:
        file_stem = folder / "NBA Player Box Scores"
        df = load_clean_data(file_stem, required_cols=['ESPN_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION'])
        
        if df is not None:
            # Drop duplicates to just get unique players per season
            players = df[['ESPN_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']].drop_duplicates(subset=['ESPN_ID'], keep='last')
            all_player_dfs.append(players)
    
    if not all_player_dfs:
        logging.critical("CRITICAL: No valid ESPN box scores found to build ID map.")
        return None
        
    player_map_df = pd.concat(all_player_dfs)
    # Keep the most recent team for each player if they changed teams
    player_map_df.drop_duplicates(subset=['ESPN_ID'], keep='last', inplace=True)
    
    # Rename ESPN_ID to our system's global PLAYER_ID
    player_map_df.rename(columns={'ESPN_ID': Cols.PLAYER_ID}, inplace=True)
    player_map_df[Cols.PLAYER_ID] = pd.to_numeric(player_map_df[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
    
    # Create clean name for fuzzy matching with BBRef later
    player_map_df['Player_Clean'] = player_map_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().strip())
    
    return player_map_df

def process_master_player_stats(player_id_map, season_folders, output_dir):
    """
    Merges Basketball Reference Season Averages & Advanced stats into the Master Player File.
    Matches names to the ESPN IDs via fuzzy matching.
    """
    logging.info("--- Starting: process_master_player_stats (BBref Only) ---")
    
    id_map_clean = player_id_map[[Cols.PLAYER_ID, 'Player_Clean', 'TEAM_ABBREVIATION', 'PLAYER_NAME']].drop_duplicates(subset=['Player_Clean'])
    name_to_id = id_map_clean.set_index('Player_Clean')[Cols.PLAYER_ID].to_dict()

    def find_match(name):
        if not name: return None
        match = process.extractOne(name, name_to_id.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=88)
        return name_to_id.get(match[0]) if match else None

    for folder in season_folders:
        season_id = folder.name
        try:
            # Start with all players known to exist this season (from ID map)
            season_player_df = id_map_clean.copy()
            season_player_df['SEASON_ID'] = season_id

            # 1. Bball-Ref Averages
            bball_ref_stem = folder / "NBA Player Per Game Averages"
            bball_ref_df = load_clean_data(bball_ref_stem, required_cols=['Player', 'PTS'])
            
            if bball_ref_df is not None:
                bball_ref_df['Player_Clean'] = bball_ref_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                bball_ref_df = bball_ref_df.rename(columns=BBREF_COLUMN_MAP)
                
                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df['Player_Clean'].apply(find_match)
                bball_ref_df = bball_ref_df[bball_ref_df[Cols.PLAYER_ID].notna()]
                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df[Cols.PLAYER_ID].astype(int)
                bball_ref_df.drop_duplicates(subset=[Cols.PLAYER_ID], keep='first', inplace=True)
                
                season_cols = [Cols.PLAYER_ID, 'Pos', 'SEASON_G', 'SEASON_PTS', 'SEASON_TRB', 'SEASON_AST']
                cols_exist = [col for col in season_cols if col in bball_ref_df.columns]
                season_player_df = pd.merge(season_player_df, bball_ref_df[cols_exist], on=Cols.PLAYER_ID, how="left")
                
                # 2. Bball-Ref Advanced Stats
                adv_stem = folder / "NBA Player Advanced Stats"
                adv_df = load_clean_data(adv_stem, required_cols=['Player', 'USG%'])
                if adv_df is not None:
                    adv_df['Player_Clean'] = adv_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                    adv_df[Cols.PLAYER_ID] = adv_df['Player_Clean'].apply(find_match)
                    adv_df = adv_df[adv_df[Cols.PLAYER_ID].notna()]
                    adv_df[Cols.PLAYER_ID] = adv_df[Cols.PLAYER_ID].astype(int)
                    adv_df.drop_duplicates(subset=[Cols.PLAYER_ID], keep='first', inplace=True)
                    
                    adv_cols = [c for c in [Cols.PLAYER_ID, 'TS%', 'USG%', 'PER'] if c in adv_df.columns]
                    season_player_df = pd.merge(season_player_df, adv_df[adv_cols], on=Cols.PLAYER_ID, how="left", suffixes=('', '_adv'))

            season_player_df.rename(columns={'Player_Clean': 'clean_name'}, inplace=True)
            
            out_name = f"master_player_stats_{season_id}.parquet"
            season_player_df.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name}")
            
        except Exception as e:
            logging.error(f"Error processing player stats for {folder}: {e}", exc_info=True)

def process_master_team_stats(player_id_map, season_folders, output_dir):
    """
    Builds Master Team Stats purely from TeamRankings scraped files.
    """
    logging.info("--- Starting: process_master_team_stats (TeamRankings Only) ---")
    
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
            
            # TR Data has columns like ["Rank", "Team", "2024", "Last 3", "Last 1", "Home", "Away"]
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
    logging.info("--- Initializing Historical Vacancy Columns (Placeholder) ---")
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in vacancy_cols:
        if c not in bs_df.columns:
            bs_df[c] = 0.0
        else:
            bs_df[c] = bs_df[c].fillna(0.0)
    return bs_df

def process_master_box_scores(player_id_map, season_folders, output_dir):
    """
    Cleans and processes the ESPN Box Scores.
    """
    logging.info("--- Starting: process_master_box_scores (ESPN Data) ---")
    
    for folder in season_folders:
        season_id = folder.name
        try:
            file_stem = folder / "NBA Player Box Scores"
            bs_df = load_clean_data(file_stem)
            
            if bs_df is None or bs_df.empty: 
                continue

            # Standardize ID column from ESPN
            if 'ESPN_ID' in bs_df.columns:
                bs_df.rename(columns={'ESPN_ID': Cols.PLAYER_ID}, inplace=True)
            
            bs_df.dropna(subset=[Cols.PLAYER_ID], inplace=True)
            bs_df[Cols.PLAYER_ID] = bs_df[Cols.PLAYER_ID].astype(int)
            
            if Cols.GAME_ID in bs_df.columns:
                bs_df[Cols.GAME_ID] = pd.to_numeric(bs_df[Cols.GAME_ID], errors='coerce').fillna(0).astype(int)
            
            if Cols.DATE in bs_df.columns: 
                bs_df[Cols.DATE] = pd.to_datetime(bs_df[Cols.DATE], errors='coerce')

            # Merge Position Info from BBRef master stats
            p_stats_path = output_dir / f"master_player_stats_{season_id}.parquet"
            if p_stats_path.exists():
                p_stats = pd.read_parquet(p_stats_path)
                if Cols.PLAYER_ID in p_stats.columns:
                    p_stats[Cols.PLAYER_ID] = pd.to_numeric(p_stats[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                    p_stats_szn = p_stats[[Cols.PLAYER_ID, 'Pos']].drop_duplicates(subset=[Cols.PLAYER_ID])
                    bs_df = pd.merge(bs_df, p_stats_szn, on=Cols.PLAYER_ID, how='left')

            # Numeric Conversions
            numeric_cols = ['PTS', 'REB', 'AST', 'FGA', 'FTA', 'TOV', 'MIN']
            for col in numeric_cols:
                if col in bs_df.columns: 
                    bs_df[col] = pd.to_numeric(bs_df[col], errors='coerce').fillna(0)
            
            # Derived Stats (PRA focuses)
            bs_df['PRA'] = bs_df['PTS'] + bs_df['REB'] + bs_df['AST']
            bs_df['PR'] = bs_df['PTS'] + bs_df['REB']
            bs_df['PA'] = bs_df['PTS'] + bs_df['AST']
            bs_df['RA'] = bs_df['REB'] + bs_df['AST']

            # Advanced Metrics (TS and USG Proxies)
            ts_denom = 2 * (bs_df['FGA'] + 0.44 * bs_df['FTA'])
            bs_df['TS_PCT'] = np.where(ts_denom > 0, bs_df['PTS'] / ts_denom, 0.0)
            
            usg_num = (bs_df['FGA'] + 0.44 * bs_df['FTA'] + bs_df['TOV'])
            bs_df['USG_PROXY'] = np.where(bs_df['MIN'] > 0, usg_num / bs_df['MIN'], 0.0)

            per_36_cols = ['PTS', 'REB', 'AST', 'PRA']
            for col in per_36_cols:
                if col in bs_df.columns:
                    bs_df[f'{col}_PER36'] = np.where(bs_df['MIN'] > 0, (bs_df[col] / bs_df['MIN']) * 36, 0.0).round(2)

            bs_df['SEASON_ID'] = season_id
            
            if p_stats_path.exists():
                bs_df = calculate_historical_vacancy(bs_df, pd.read_parquet(p_stats_path))

            # Opponent is pre-processed by ESPN Scraper (OPPONENT_ABBREV), 
            # just ensure it's there.
            if 'OPPONENT_ABBREV' not in bs_df.columns:
                 bs_df['OPPONENT_ABBREV'] = "UNK"

            subset_cols = [Cols.PLAYER_ID, Cols.DATE]
            if Cols.GAME_ID in bs_df.columns:
                subset_cols.insert(1, Cols.GAME_ID)
            
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
        try:
            dfs.append(pd.read_parquet(f))
        except: pass
    
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    agg_cols = {k: 'mean' for k in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'MIN'] if k in df.columns}
    
    if Cols.GAME_ID in df.columns: 
        agg_cols[Cols.GAME_ID] = 'count'
    
    vs_opp_df = df.groupby([Cols.PLAYER_ID, 'PLAYER_NAME', 'OPPONENT_ABBREV']).agg(agg_cols).reset_index()
    if Cols.GAME_ID in vs_opp_df.columns: 
        vs_opp_df.rename(columns={Cols.GAME_ID: 'GAMES_PLAYED'}, inplace=True)
    
    vs_opp_df.round(2).to_parquet(output_dir / "master_vs_opponent.parquet", index=False)
    logging.info("Saved master_vs_opponent.parquet")

def process_dvp_stats(output_dir):
    """
    Calculates Defense vs Position (DvP) stats PER SEASON to avoid historical leakage.
    """
    logging.info("--- Starting: process_dvp_stats (Season-Aware) ---")
    files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not files: return
    
    all_dvp_dfs = []

    for file_path in files:
        try:
            match = re.search(r'\d{4}-\d{2}', file_path.name)
            season_id = match.group(0) if match else "UNKNOWN"
            
            df = pd.read_parquet(file_path)
            
            required = ['Pos', 'OPPONENT_ABBREV']
            if Cols.DATE in df.columns: required.append(Cols.DATE)
            
            if not all(c in df.columns for c in required):
                continue

            def normalize_pos(pos):
                if not isinstance(pos, str): return 'UNKNOWN'
                p = pos.split('-')[0].upper().strip()
                if p == 'G': return 'SG' 
                if p == 'F': return 'PF' 
                return p
            
            df['Primary_Pos'] = df['Pos'].apply(normalize_pos)
            valid_positions = ['PG', 'SG', 'SF', 'PF', 'C']
            df = df[df['Primary_Pos'].isin(valid_positions)].copy()

            if Cols.DATE in df.columns:
                df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE], inplace=True)

            stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
            
            for col in stat_cols:
                if col in df.columns:
                    exp_series = df.groupby(Cols.PLAYER_ID)[col].expanding().mean()
                    df[f'{col}_AVG'] = exp_series.groupby(level=0).shift(1).reset_index(level=0, drop=True)
                    
            df.dropna(subset=[f'{c}_AVG' for c in stat_cols if c in df.columns], inplace=True)

            for col in stat_cols:
                if col in df.columns:
                    df[f'{col}_DIFF'] = df[col] - df[f'{col}_AVG']

            diff_cols = {f'{col}_DIFF': 'mean' for col in stat_cols if col in df.columns}
            if not diff_cols: continue
            
            dvp_diffs = df.groupby(['OPPONENT_ABBREV', 'Primary_Pos']).agg(diff_cols).reset_index()
            league_pos_baselines = df.groupby('Primary_Pos')[stat_cols].mean().reset_index()
            
            rename_map = {c: f"{c}_BASE" for c in stat_cols}
            league_pos_baselines.rename(columns=rename_map, inplace=True)
            
            merged_dvp = pd.merge(dvp_diffs, league_pos_baselines, on='Primary_Pos', how='inner')
            
            season_dvp = pd.DataFrame()
            season_dvp['SEASON_ID'] = season_id
            season_dvp['OPPONENT_ABBREV'] = merged_dvp['OPPONENT_ABBREV']
            season_dvp['Primary_Pos'] = merged_dvp['Primary_Pos']
            
            for col in stat_cols:
                if f'{col}_DIFF' in merged_dvp.columns and f'{col}_BASE' in merged_dvp.columns:
                    season_dvp[f'DVP_{col}'] = merged_dvp[f'{col}_BASE'] + merged_dvp[f'{col}_DIFF']
            
            all_dvp_dfs.append(season_dvp)
            
        except Exception as e:
            logging.error(f"Error processing DVP for {file_path.name}: {e}", exc_info=True)

    if all_dvp_dfs:
        final_dvp_all = pd.concat(all_dvp_dfs, ignore_index=True)
        final_dvp_all.round(2).to_parquet(output_dir / "master_dvp_stats.parquet", index=False)
        logging.info(f"Saved master_dvp_stats.parquet (Multi-Season: {len(final_dvp_all)} rows)")