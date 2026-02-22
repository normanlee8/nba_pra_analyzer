import csv
import re
import datetime
import logging
import pandas as pd
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

# Expanded map to catch common abbreviations
DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6,
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
    'FRI': 4, 'SAT': 5, 'SUN': 6
}

class SmartDateDetector:
    """
    1. Checks the date based on the text Day (e.g., "Mon") if provided.
    2. Checks the NBA Schedule for Today and Tomorrow.
    3. Checks historical box scores (Backtesting).
    """
    def __init__(self, lookback_days=3):
        self.history_map = {}
        self.lookback_days = lookback_days
        self.schedule_cache = {}  # Cache schedule to avoid spamming API
        self.team_id_map = self._build_team_map()
        self._load_history()

    def _build_team_map(self):
        """Builds a dictionary mapping abbreviations to Team IDs."""
        try:
            nba_teams = teams.get_teams()
            id_map = {t['abbreviation']: t['id'] for t in nba_teams}
            
            # Add common variants/legacy codes manually
            if 'NOP' in id_map: id_map['NO'] = id_map['NOP']
            if 'UTA' in id_map: id_map['UTAH'] = id_map['UTA']
            if 'SAS' in id_map: id_map['SA'] = id_map['SAS']
            if 'GSW' in id_map: id_map['GS'] = id_map['GSW']
            if 'NYK' in id_map: id_map['NY'] = id_map['NYK']
            if 'WAS' in id_map: id_map['WSH'] = id_map['WAS']
            if 'PHX' in id_map: id_map['PHO'] = id_map['PHX']
            
            return id_map
        except Exception as e:
            logging.error(f"Failed to load team ID map: {e}")
            return {}

    def _load_history(self):
        """Loads the last N days of matchups from Master Parquet files."""
        files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
        if not files: return

        try:
            dfs = []
            for f in files:
                try:
                    # Read Parquet instead of CSV
                    d = pd.read_parquet(f, columns=['TEAM_ABBREVIATION', 'OPPONENT_ABBREV', 'GAME_DATE'])
                    dfs.append(d)
                except Exception as e: 
                    continue
            
            if not dfs: return

            df = pd.concat(dfs, ignore_index=True)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Dynamic cutoff based on lookback_days
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.lookback_days)
            recent = df[df['GAME_DATE'] >= cutoff].sort_values('GAME_DATE', ascending=True)
            
            for _, row in recent.iterrows():
                t1 = row['TEAM_ABBREVIATION']
                t2 = row['OPPONENT_ABBREV']
                date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
                self.history_map[(t1, t2)] = date_str
                self.history_map[(t2, t1)] = date_str
                
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")

    def _check_nba_schedule(self, team, opponent, check_date):
        """Checks if a specific matchup is scheduled for a specific date string using IDs."""
        t1_id = self.team_id_map.get(team.upper())
        t2_id = self.team_id_map.get(opponent.upper())

        if not t1_id or not t2_id:
            return False

        if check_date in self.schedule_cache:
            games_df = self.schedule_cache[check_date]
        else:
            try:
                board = scoreboardv2.ScoreboardV2(game_date=check_date, timeout=5)
                games_df = board.game_header.get_data_frame()
                self.schedule_cache[check_date] = games_df
            except Exception as e:
                logging.warning(f"Could not fetch schedule for {check_date}: {e}")
                self.schedule_cache[check_date] = pd.DataFrame()
                return False

        if games_df is None or games_df.empty:
            return False
            
        try:
            match = games_df[
                ((games_df['HOME_TEAM_ID'] == t1_id) & (games_df['VISITOR_TEAM_ID'] == t2_id)) |
                ((games_df['HOME_TEAM_ID'] == t2_id) & (games_df['VISITOR_TEAM_ID'] == t1_id))
            ]
            return not match.empty
        except KeyError:
            return False

    def get_date_from_day(self, day_str):
        """Calculates the nearest date for a given day string."""
        if not day_str or day_str not in DAYS_MAP:
            return None
            
        target_weekday = DAYS_MAP[day_str]
        today = datetime.datetime.now()
        current_weekday = today.weekday()
        
        diff = target_weekday - current_weekday
        if diff < -2:
             diff += 7
        elif diff > 4:
             diff -= 7
             
        target_date = today + datetime.timedelta(days=diff)
        return target_date.strftime("%Y-%m-%d")

    def find_date(self, team, opponent, day_str=None):
        today = datetime.datetime.now()
        tomorrow = today + datetime.timedelta(days=1)
        
        str_today = today.strftime("%Y-%m-%d")
        str_tomorrow = tomorrow.strftime("%Y-%m-%d")

        if day_str:
            calculated_date = self.get_date_from_day(day_str)
            if calculated_date:
                return calculated_date

        if self._check_nba_schedule(team, opponent, str_today):
            return str_today
            
        if self._check_nba_schedule(team, opponent, str_tomorrow):
            return str_tomorrow

        if (team, opponent) in self.history_map:
            return self.history_map[(team, opponent)]

        return str_today

def clean_prop_line(text):
    """Robustly extracts a numeric value from a line string."""
    s = text.strip().upper()
    s = s.replace(',', '.') 
    s = re.sub(r'^[OU]\s+', '', s)
    s = s.replace('OVER', '').replace('UNDER', '').strip()
    try:
        val = float(s)
        return str(val)
    except ValueError:
        return None

def parse_matchup(matchup_line):
    """Extracts Team abbreviations and Day from a matchup line."""
    day_match = re.search(r'-\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', matchup_line, re.IGNORECASE)
    day_str = day_match.group(1) if day_match else None

    line = matchup_line.replace(' vs ', ' @ ').replace(' vs. ', ' @ ').replace('-', ' @ ')
    match = re.search(r'\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b', line) 
    
    if match:
        team1 = match.group(1)
        team2 = match.group(2)
        full_matchup_string = f"{team1} vs. {team2}"
        return team1, team2, full_matchup_string, day_str
    return None, None, None, None


def update_master_prop_history(data_to_write, header):
    """
    Appends new daily props to the master history tracking file.
    This creates a historical database of Vegas lines for model training.
    """
    if not data_to_write:
        return
        
    new_df = pd.DataFrame(data_to_write, columns=header)
    history_file = cfg.MASTER_PROP_HISTORY_FILE
    
    try:
        if history_file.exists():
            hist_df = pd.read_parquet(history_file)
            # Combine old and new
            combined_df = pd.concat([hist_df, new_df], ignore_index=True)
            # Drop duplicates in case the converter is run multiple times on the same day
            # Keep the last entry so line movement (updates) overwrite earlier parses
            combined_df = combined_df.drop_duplicates(
                subset=[Cols.PLAYER_NAME, Cols.DATE, Cols.PROP_TYPE], 
                keep='last'
            )
        else:
            combined_df = new_df
            
        # Ensure the directory exists
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save back to parquet
        combined_df.to_parquet(history_file, index=False)
        logging.info(f"Updated Master Prop History: {history_file.name} (Total records: {len(combined_df)})")
        
    except Exception as e:
        logging.error(f"Failed to update master prop history: {e}")


def parse_text_to_csv(input_path=None, output_path=None):
    if input_path is None: input_path = cfg.INPUT_PROPS_TXT
    if output_path is None: output_path = cfg.PROPS_FILE
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    date_detector = SmartDateDetector(lookback_days=3)
    
    current_player = None
    current_team = None
    current_opponent = None
    current_matchup = None
    current_game_date = None 
    
    data_to_write = [] 
    
    lines_processed = 0
    props_parsed = 0
    suspicious_lines = 0

    # List of keywords to explicitly ignore
    IGNORED_PHRASES = {
        'HIGHER', 'LOWER', 'FEWER PICKS', 'MORE PICKS', 'DRAFTS', 
        'PICK\'EM', 'LIVE', 'RESULTS', 'RANKINGS', 'NEWS FEED', 
        '$0.00', 'ALL NBA', 'COLLAPSE ALL', 'ADD PICKS', 'ENTRY AMOUNT',
        'REWARDS', 'ENTER AMOUNT', 'STANDARD', 'FLEX', 'PLAY',
        'FIND NBA TEAMS', 'PRE-GAME & IN-GAME', 'PICK\'EM TIPS'
    }

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            lines_processed += 1

            # 1. Check for Time/Matchup Line
            t1, t2, full_matchup, day_str = parse_matchup(line)
            
            if full_matchup:
                current_matchup = full_matchup
                current_team = t1
                current_opponent = t2
                current_game_date = date_detector.find_date(t1, t2, day_str)
                continue 

            # 2. Check for Prop Line Value
            cleaned_val = clean_prop_line(line)
            if cleaned_val:
                prop_line_value = cleaned_val
                continue

            # 3. Check for Category
            if 'prop_line_value' in locals() and prop_line_value is not None:
                prop_category_str = line
                prop_category_std = cfg.MASTER_PROP_MAP.get(prop_category_str, None)
                
                if not prop_category_std:
                    for k, v in cfg.MASTER_PROP_MAP.items():
                        if k.lower() == prop_category_str.lower():
                            prop_category_std = v
                            break
                
                if prop_category_std and current_player and current_matchup:
                    data_to_write.append([
                        current_player, current_team, current_opponent,
                        current_matchup, prop_category_std, prop_line_value,
                        current_game_date 
                    ])
                    props_parsed += 1
                else:
                    logging.warning(f"Skipped incomplete prop: Player={current_player}, Prop={line}, Line={prop_line_value}")
                
                prop_line_value = None 
                continue

            # 4. Fallback: Player Name
            # Skip known garbage lines before assigning them as Player Name
            upper_line = line.upper()
            if upper_line in IGNORED_PHRASES or any(upper_line.startswith(p) for p in ['MORE PICKS', 'GET UP TO', 'CLAIM YOUR']):
                continue
            
            # Validation
            if any(char.isdigit() for char in line) or 'OVER' in upper_line or 'UNDER' in upper_line:
                logging.warning(f"SUSPICIOUS: Line interpreted as Player Name but contains digits/keywords: '{line}'")
                suspicious_lines += 1
            
            current_player = line
            prop_line_value = None 
            continue

        if not data_to_write:
            logging.warning("No valid props parsed. Check input format.")
            return

        header = Cols.get_required_input_cols()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        print("-" * 40)
        print(f"PARSING COMPLETE: {output_path}")
        print(f"Lines Processed: {lines_processed}")
        print(f"Props Extracted: {props_parsed}")
        if suspicious_lines > 0:
            print(f"WARNING: {suspicious_lines} lines looked suspicious (Check logs).")
        print("-" * 40)
        
        logging.info(f"Successfully converted props to {output_path} ({len(data_to_write)} rows)")
        
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rec_path = cfg.INPUT_DIR / "records" / f"{now_ts}.csv"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, 'w', newline='', encoding='utf-8') as f_rec:
            writer = csv.writer(f_rec)
            writer.writerow(header)
            writer.writerows(data_to_write)

        # UPDATE THE MASTER HISTORY FILE
        update_master_prop_history(data_to_write, header)

    except Exception as e:
        logging.error(f"Error parsing props: {e}", exc_info=True)