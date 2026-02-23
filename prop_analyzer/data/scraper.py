import pandas as pd
import requests
import time
import logging
import sys
import re
import io
import random
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import project config
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

# --- DYNAMIC CONFIGURATION ---
def get_season_config():
    now = datetime.now()
    if now.month >= 10:
        current_start_year = now.year
    else:
        current_start_year = now.year - 1
        
    current_end_year = current_start_year + 1
    prev_start_year = current_start_year - 1
    prev_end_year = current_start_year

    curr_season_str = f"{current_start_year}-{str(current_end_year)[-2:]}"
    prev_season_str = f"{prev_start_year}-{str(prev_end_year)[-2:]}"

    return [
        {
            "id": "last_season",
            "season_str": prev_season_str,
            "bball_ref_year": prev_end_year,
            "is_current": False,
            "tr_date_param": f"{prev_end_year}-07-01" 
        },
        {
            "id": "current_season",
            "season_str": curr_season_str,
            "bball_ref_year": current_end_year,
            "is_current": True,
            "tr_date_param": None
        }
    ]

# --- TUNING SETTINGS ---
MAX_WORKERS = 4               # For general scraping
MAX_ESPN_WORKERS = 8          # Higher concurrency for ESPN API
ESPN_API_TIMEOUT = 15

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

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

# Strictly filtered to PRA + Pace/Efficiency metrics (UPDATED WITH NEW STATS)
TEAMRANKINGS_SLUG_MAP = {
    # Original Base Stats
    "Points per Game": "points-per-game",
    "Offensive Efficiency": "offensive-efficiency",
    "Defensive Efficiency": "defensive-efficiency",
    "Shooting %": "shooting-pct",
    "Effective Field Goal %": "effective-field-goal-pct",
    "True Shooting %": "true-shooting-percentage",
    "Total Rebounds per Game": "total-rebounds-per-game",
    "Assists per Game": "assists-per-game",
    "Opponent Points per Game": "opponent-points-per-game",
    "Opponent Total Rebounds per Game": "opponent-total-rebounds-per-game",
    "Opponent Assists per Game": "opponent-assists-per-game",
    "Possessions per Game": "possessions-per-game",
    
    # --- NEW: Rebounding Optimization ---
    "Opponent Effective Field Goal %": "opponent-effective-field-goal-pct",
    "Opponent True Shooting %": "opponent-true-shooting-percentage",
    "Field Goals Attempted per Game": "field-goals-attempted-per-game",
    "Opponent Field Goals Attempted per Game": "opponent-field-goals-attempted-per-game",
    "Three Pointers Attempted per Game": "three-pointers-attempted-per-game",
    "Opponent Three Pointers Attempted per Game": "opponent-three-pointers-attempted-per-game",
    "Opponent Offensive Rebounding %": "opponent-offensive-rebounding-pct",

    # --- NEW: Assist Optimization ---
    "Assists per FGM": "assists-per-fgm",
    "Opponent Assists per FGM": "opponent-assists-per-fgm",
    "Assist to Turnover Ratio": "assist-turnover-ratio",

    # --- NEW: Points Optimization ---
    "Opponent Points in Paint per Game": "opponent-points-in-paint-per-game",
    "Opponent Percent of Points from 3 Pointers": "opponent-percent-of-points-from-3-pointers",
    "Opponent Personal Fouls per Game": "opponent-personal-fouls-per-game",
    "Opponent Fastbreak Points per Game": "opponent-fastbreak-points-per-game",

    # --- NEW: Combination / Overall Volume Optimization ---
    "Extra Scoring Chances per Game": "extra-scoring-chances-per-game",
    "Opponent Points + Rebounds + Assists per Game": "opponent-points-rebounds-assists-per-game",
    "Opponent Points + Assists per Game": "opponent-points-assists-per-game"
}

MASTER_FILE_MAP = {
    "NBA Player Per Game Averages.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_per_game.html", "per_game_stats")),
    "NBA Player Advanced Stats.csv": ("bball_ref", ("https://www.basketball-reference.com/leagues/NBA_{YEAR}_advanced.html", "advanced")),
}

def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=MAX_ESPN_WORKERS, pool_maxsize=MAX_ESPN_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session

def save_clean_parquet(df, filename_stem, output_dir):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        clean_name = filename_stem.replace('.csv', '') + ".parquet"
        file_path = output_dir / clean_name
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        df.to_parquet(file_path, index=False)
        logging.info(f"Successfully saved {file_path.name}")
    except Exception as e:
        logging.error(f"FAILED to save {filename_stem}: {e}")

def deduplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [
            dup if i == 0 else f"{dup}_{i}" 
            for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def scrape_daily_injuries(session, output_dir):
    logging.info("--- Scraping Daily Injury Report (CBS Sports) ---")
    url = "https://www.cbssports.com/nba/injuries/"
    filename = "daily_injuries" 
    
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            logging.warning("No injury tables found on CBS Sports.")
            return

        all_rows = []
        for table in tables:
            team_abbr = "UNK"
            team_header = table.find_previous(class_="TeamLogoNameLockup-name")
            if team_header:
                raw_team_name = team_header.get_text(strip=True)
                team_abbr = TEAM_NAME_MAP.get(raw_team_name, "UNK")
            
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if not cols: continue 
                
                name_cell = cols[0]
                long_name_span = name_cell.find('span', class_=lambda x: x and 'long' in x)
                player_text = long_name_span.get_text(strip=True) if long_name_span else name_cell.get_text(strip=True)

                try:
                    status = cols[4].get_text(strip=True)
                    all_rows.append({
                        "Team": team_abbr,
                        "Player": player_text,
                        "Position": cols[1].get_text(strip=True),
                        "Injury Status": status
                    })
                except IndexError:
                    continue

        if not all_rows: return

        injury_df = pd.DataFrame(all_rows)
        def clean_status(val):
            s = str(val).lower()
            if 'out' in s: return 'OUT'
            if 'doubtful' in s: return 'DOUBTFUL'
            if 'questionable' in s or 'game time decision' in s: return 'GTD'
            return 'UNKNOWN'
        
        injury_df['Status_Clean'] = injury_df['Injury Status'].apply(clean_status)
        save_clean_parquet(injury_df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape injuries: {e}")

def scrape_teamrankings(session, slug, filename, season_cfg, output_dir):
    url = f"https://www.teamrankings.com/nba/stat/{slug}"
    if season_cfg['tr_date_param']:
        url += f"?date={season_cfg['tr_date_param']}"
        
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table') 
        
        if not table: return

        dfs = pd.read_html(io.StringIO(str(table)))
        if not dfs: return
            
        df = dfs[0]
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = [col[1] if len(col) > 1 else col[0] for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        df = deduplicate_columns(df)

        if len(df.columns) >= 8:
            df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6]].copy()
            df.columns = ["Rank", "Team", str(season_cfg['bball_ref_year']), "Last 3", "Last 1", "Home", "Away"]

        if 'Team' in df.columns:
            df['Team'] = df['Team'].apply(lambda x: str(x).split('(')[0].strip())
        
        save_clean_parquet(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
    finally:
        time.sleep(0.5) 

def scrape_bball_ref(session, url_template, table_id, filename, season_cfg, output_dir):
    url = url_template.replace("{YEAR}", str(season_cfg['bball_ref_year']))
    
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', id=table_id)
        
        if not table:
            comment = soup.find(string=lambda text: isinstance(text, Comment) and table_id in text)
            if comment:
                table = BeautifulSoup(comment, 'html.parser').find('table', id=table_id)
        
        if not table: return
            
        df = pd.read_html(io.StringIO(str(table)))[0]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[1] if len(col) > 1 else col[0]) for col in df.columns]
        else:
            df.columns = [str(col) for col in df.columns]

        df = deduplicate_columns(df)
        if 'Rk' in df.columns: df = df[df['Rk'] != 'Rk']
        if 'Awards' in df.columns: df = df.drop(columns=['Awards'])
            
        save_clean_parquet(df, filename, output_dir)
        
    except Exception as e:
        logging.error(f"Failed to scrape {url}: {e}")
    finally:
        time.sleep(1) 

def get_season_dates(season_str):
    start_year = int(season_str.split('-')[0])
    start_date = datetime(start_year, 10, 20) 
    end_date = datetime(start_year + 1, 4, 20)
    if end_date > datetime.now():
        end_date = datetime.now() - timedelta(days=1)
    return start_date, end_date

def fetch_espn_daily_box_scores(session, target_date):
    date_str = target_date.strftime('%Y%m%d')
    scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    
    try:
        sb_resp = session.get(scoreboard_url, timeout=ESPN_API_TIMEOUT)
        sb_resp.raise_for_status()
        sb_data = sb_resp.json()
        
        events = sb_data.get('events', [])
        if not events:
            return pd.DataFrame()
            
        daily_records = []
        for event in events:
            game_id = event['id']
            status = event.get('status', {}).get('type', {})
            if not status.get('completed', False):
                continue
                
            summary_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
            
            try:
                sum_resp = session.get(summary_url, timeout=ESPN_API_TIMEOUT)
                sum_resp.raise_for_status()
                sum_data = sum_resp.json()
                
                boxscore = sum_data.get('boxscore', {})
                players_data = boxscore.get('players', [])
                
                if len(players_data) == 2:
                    team1 = players_data[0]['team']['abbreviation']
                    team2 = players_data[1]['team']['abbreviation']
                    
                    for t_data in players_data:
                        current_team = t_data['team']['abbreviation']
                        opp_team = team2 if current_team == team1 else team1
                        
                        stats_blocks = t_data.get('statistics', [])
                        if not stats_blocks: continue
                        
                        labels = stats_blocks[0].get('labels', [])
                        athletes = stats_blocks[0].get('athletes', [])
                        
                        for ath in athletes:
                            if ath.get('didNotPlay') or not ath.get('stats'): continue
                            
                            stat_dict = dict(zip(labels, ath['stats']))
                            
                            fg_split = stat_dict.get('FG', '0-0').split('-')
                            fga = fg_split[1] if len(fg_split) > 1 else '0'
                            
                            ft_split = stat_dict.get('FT', '0-0').split('-')
                            fta = ft_split[1] if len(ft_split) > 1 else '0'
                            
                            pos_dict = ath.get('athlete', {}).get('position', {})
                            position_str = pos_dict.get('abbreviation', 'UNK')
                            
                            record = {
                                'GAME_ID': game_id,
                                Cols.DATE: target_date.strftime('%Y-%m-%d'),
                                'ESPN_ID': ath['athlete']['id'],
                                'PLAYER_NAME': ath['athlete']['displayName'],
                                'TEAM_ABBREVIATION': current_team,
                                'OPPONENT_ABBREV': opp_team,
                                'Position': position_str,  
                                'MIN': stat_dict.get('MIN', '0'),
                                'PTS': stat_dict.get('PTS', '0'),
                                'REB': stat_dict.get('REB', '0'),
                                'AST': stat_dict.get('AST', '0'),
                                'FGA': fga,
                                'FTA': fta,
                                'TOV': stat_dict.get('TO', '0')
                            }
                            daily_records.append(record)
            except Exception as e:
                logging.error(f"Error fetching summary for game {game_id}: {e}")
                
        return pd.DataFrame(daily_records)
        
    except Exception as e:
        logging.error(f"Error fetching ESPN scoreboard for {date_str}: {e}")
        return pd.DataFrame()

def scrape_espn_box_scores_incremental(session, season_cfg, output_dir):
    target_season = season_cfg['season_str']
    logging.info(f"--- Gap Detection: ESPN Box Scores for {target_season} ---")
    
    box_scores_file = output_dir / "NBA Player Box Scores.parquet"
    existing_df = pd.DataFrame()
    existing_dates_set = set()
    
    start_date, end_date = get_season_dates(target_season)

    if box_scores_file.exists():
        try:
            existing_df = pd.read_parquet(box_scores_file)
            if Cols.DATE in existing_df.columns and not existing_df.empty:
                existing_dates_set = set(existing_df[Cols.DATE].unique())
        except Exception as e:
            logging.warning(f"Could not read existing box scores ({e}). Re-scraping full season.")

    dates_to_fetch = []
    current = start_date
    while current <= end_date:
        date_str = current.strftime('%Y-%m-%d')
        if date_str not in existing_dates_set:
            dates_to_fetch.append(current)
        current += timedelta(days=1)

    if not dates_to_fetch:
        logging.info("Box scores are completely up to date. No missing gaps.")
        return

    logging.info(f"Identified {len(dates_to_fetch)} missing days to fetch concurrently.")
    
    all_daily_stats = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_ESPN_WORKERS) as executor:
        future_to_date = {executor.submit(fetch_espn_daily_box_scores, session, d): d for d in dates_to_fetch}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_date):
            target_dt = future_to_date[future]
            completed += 1
            try:
                df = future.result()
                if not df.empty:
                    all_daily_stats.append(df)
            except Exception as exc:
                logging.error(f"Date {target_dt.strftime('%Y-%m-%d')} generated an exception: {exc}")
                
            if completed % 10 == 0 or completed == len(dates_to_fetch):
                logging.info(f"  Fetches complete: {completed}/{len(dates_to_fetch)}")

    if all_daily_stats:
        new_data_df = pd.concat(all_daily_stats, ignore_index=True)
        final_df = pd.concat([existing_df, new_data_df], ignore_index=True) if not existing_df.empty else new_data_df
        
        final_df['MIN'] = final_df['MIN'].astype(str).str.replace(r'[a-zA-Z]', '', regex=True)
        final_df['MIN'] = pd.to_numeric(final_df['MIN'], errors='coerce').fillna(0).astype(int)
        
        subset_cols = ['ESPN_ID', Cols.DATE, 'GAME_ID']
        final_df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
        
        final_df.sort_values(by=[Cols.DATE, 'GAME_ID', 'TEAM_ABBREVIATION'], inplace=True)
        
        save_clean_parquet(final_df, "NBA Player Box Scores", output_dir)
        logging.info(f"Updated aggregated box scores. Added {len(new_data_df)} rows. Total rows: {len(final_df)}")
    else:
        logging.info("No new games found in the requested dates.")

def should_skip_season_file(output_dir, filename_stem, is_current_season):
    if is_current_season: return False 
    clean_name = filename_stem.replace('.csv', '') + ".parquet"
    file_path = output_dir / clean_name
    return file_path.exists() and file_path.stat().st_size > 0

def main():
    start_time = time.time()
    logging.info("========= STARTING NBA DATA SCRAPER (FAST CONCURRENT MODE) =========")
    
    session = create_robust_session()
    seasons_to_scrape = get_season_config()
    
    for season_cfg in seasons_to_scrape:
        season_str = season_cfg['season_str']
        is_current = season_cfg['is_current']
        
        output_dir = cfg.DATA_DIR / season_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"--- Processing Season: {season_str} (Live: {is_current}) ---")
        
        if is_current:
            scrape_daily_injuries(session, output_dir)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for filename, (source, data) in MASTER_FILE_MAP.items():
                if should_skip_season_file(output_dir, filename, is_current): continue
                url_template, table_id = data
                futures.append(executor.submit(scrape_bball_ref, session, url_template, table_id, filename, season_cfg, output_dir))
            for future in concurrent.futures.as_completed(futures): future.result() 
                
        scrape_espn_box_scores_incremental(session, season_cfg, output_dir)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for friendly_name, slug in TEAMRANKINGS_SLUG_MAP.items():
                sanitized_name = re.sub(r"\(.*\)", "", friendly_name).strip().replace(" / ", " per ").replace("/", " per ")
                filename = f"NBA Team {sanitized_name}" 
                if should_skip_season_file(output_dir, filename, is_current): continue
                futures.append(executor.submit(scrape_teamrankings, session, slug, filename, season_cfg, output_dir))
            for future in concurrent.futures.as_completed(futures): future.result()
            
    session.close()
    
    elapsed = time.time() - start_time
    logging.info(f"========= NBA DATA SCRAPER FINISHED in {int(elapsed // 60)}:{int(elapsed % 60):02d} minutes =========")

if __name__ == "__main__":
    main()