import csv
import re
import datetime
import logging
import pandas as pd
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6,
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
    'FRI': 4, 'SAT': 5, 'SUN': 6
}

class SmartDateDetector:
    """
    Determines game dates purely through text parsing and local history files.
    No external API calls are used.
    """
    def __init__(self, lookback_days=3):
        self.history_map = {}
        self.lookback_days = lookback_days
        self._load_history()

    def _load_history(self):
        """Loads the last N days of matchups from Master Parquet files as a fallback."""
        files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
        if not files: return
        try:
            dfs = []
            for f in files:
                try:
                    d = pd.read_parquet(f, columns=['TEAM_ABBREVIATION', 'OPPONENT_ABBREV', 'GAME_DATE'])
                    dfs.append(d)
                except Exception: 
                    continue
            if not dfs: return
            
            df = pd.concat(dfs, ignore_index=True)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
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

    def get_date_from_day(self, day_str):
        if not day_str or day_str not in DAYS_MAP: return None
        target_weekday = DAYS_MAP[day_str]
        today = datetime.datetime.now()
        current_weekday = today.weekday()
        diff = target_weekday - current_weekday
        
        if diff < -2: diff += 7
        elif diff > 4: diff -= 7
        
        target_date = today + datetime.timedelta(days=diff)
        return target_date.strftime("%Y-%m-%d")

    def find_date(self, team, opponent, day_str=None):
        today = datetime.datetime.now()
        
        if day_str:
            calculated_date = self.get_date_from_day(day_str)
            if calculated_date: return calculated_date
            
        if (team, opponent) in self.history_map: 
            return self.history_map[(team, opponent)]
            
        return today.strftime("%Y-%m-%d")

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

def update_master_prop_history(data_to_write, header):
    if not data_to_write: return
    new_df = pd.DataFrame(data_to_write, columns=header)
    history_file = cfg.MASTER_PROP_HISTORY_FILE
    try:
        if history_file.exists():
            hist_df = pd.read_parquet(history_file)
            combined_df = pd.concat([hist_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=[Cols.PLAYER_NAME, Cols.DATE, Cols.PROP_TYPE], 
                keep='last'
            )
        else:
            combined_df = new_df
        history_file.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(history_file, index=False)
        logging.info(f"Updated Master Prop History: {history_file.name} (Total records: {len(combined_df)})")
    except Exception as e:
        logging.error(f"Failed to update master prop history: {e}")

# =========================================================
# PARSER ENGINES
# =========================================================

def _parse_prizepicks(lines, date_detector):
    data_to_write = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # PrizePicks Anchor: "TEAM - POS"
        anchor_match = re.match(r'^([A-Z]{2,3})\s*-\s*[A-Z\-]+$', line)
        
        if anchor_match and i + 4 < len(lines):
            current_team = anchor_match.group(1)
            current_player = lines[i+1]
            matchup_line = lines[i+2]
            prop_line_str = lines[i+3]
            prop_category_str = lines[i+4]
            
            opp_match = re.search(r'(@|vs)\s+([A-Z]{2,3})', matchup_line)
            if opp_match:
                current_opponent = opp_match.group(2)
                current_matchup = f"{current_team} vs. {current_opponent}"
            else:
                current_opponent = "UNK"
                current_matchup = f"{current_team} vs. UNK"
                
            day_match = re.search(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', matchup_line, re.IGNORECASE)
            day_str = day_match.group(1) if day_match else None
            current_game_date = date_detector.find_date(current_team, current_opponent, day_str)
            
            prop_line_value = clean_prop_line(prop_line_str)
            
            prop_category_std = None
            for k, v in cfg.MASTER_PROP_MAP.items():
                if k.lower() == prop_category_str.lower():
                    prop_category_std = v
                    break
                    
            if prop_category_std and prop_line_value:
                data_to_write.append([
                    current_player, current_team, current_opponent,
                    current_matchup, prop_category_std, prop_line_value,
                    current_game_date 
                ])
            i += 5
        else:
            i += 1
            
    return data_to_write, "PrizePicks"

def _parse_underdog(lines, date_detector):
    data_to_write = []
    current_player = current_team = current_opponent = current_matchup = current_game_date = prop_line_value = None
    
    IGNORED_PHRASES = {
        'HIGHER', 'LOWER', 'FEWER PICKS', 'MORE PICKS', 'DRAFTS', 
        'PICK\'EM', 'LIVE', 'RESULTS', 'RANKINGS', 'NEWS FEED', 
        '$0.00', 'ALL NBA', 'COLLAPSE ALL', 'ADD PICKS', 'ENTRY AMOUNT',
        'REWARDS', 'ENTER AMOUNT', 'STANDARD', 'FLEX', 'PLAY',
        'FIND NBA TEAMS', 'PRE-GAME & IN-GAME', 'PICK\'EM TIPS'
    }

    for line in lines:
        # 1. Matchup Line Check
        day_match = re.search(r'-\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)', line, re.IGNORECASE)
        day_str = day_match.group(1) if day_match else None
        
        m_line = line.replace(' vs ', ' @ ').replace(' vs. ', ' @ ').replace('-', ' @ ')
        match = re.search(r'\b([A-Z]{2,3})\s*@\s*([A-Z]{2,3})\b', m_line) 
        
        if match:
            current_team = match.group(1)
            current_opponent = match.group(2)
            current_matchup = f"{current_team} vs. {current_opponent}"
            current_game_date = date_detector.find_date(current_team, current_opponent, day_str)
            continue 

        # 2. Line Value Check
        cleaned_val = clean_prop_line(line)
        if cleaned_val:
            prop_line_value = cleaned_val
            continue

        # 3. Prop Category Check (Only processes if we have a floating line value)
        if prop_line_value is not None:
            prop_category_std = None
            for k, v in cfg.MASTER_PROP_MAP.items():
                if k.lower() == line.lower():
                    prop_category_std = v
                    break
            
            if prop_category_std and current_player and current_matchup:
                data_to_write.append([
                    current_player, current_team, current_opponent,
                    current_matchup, prop_category_std, prop_line_value,
                    current_game_date 
                ])
                prop_line_value = None 
                continue
            elif prop_category_std:
                prop_line_value = None
                continue

        # 4. Fallback Player Name
        upper_line = line.upper()
        if upper_line in IGNORED_PHRASES or any(upper_line.startswith(p) for p in ['MORE PICKS', 'GET UP TO', 'CLAIM YOUR']):
            continue
        if any(char.isdigit() for char in line) or 'OVER' in upper_line or 'UNDER' in upper_line:
            continue
        
        current_player = line
        prop_line_value = None 
        
    return data_to_write, "Underdog"

# =========================================================
# MAIN PARSER LOGIC
# =========================================================

def parse_text_to_csv(input_path=None, output_path=None):
    if input_path is None: input_path = cfg.INPUT_PROPS_TXT
    if output_path is None: output_path = cfg.PROPS_FILE
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    date_detector = SmartDateDetector(lookback_days=3)

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = [line.strip() for line in f_in.readlines() if line.strip()]

        if not lines:
            logging.warning("Input file is empty.")
            return

        # Auto-Detect Platform Format based on the first 100 lines
        is_prizepicks = any(re.match(r'^([A-Z]{2,3})\s*-\s*[A-Z\-]+$', line) for line in lines[:100])
        
        if is_prizepicks:
            data_to_write, detected_platform = _parse_prizepicks(lines, date_detector)
        else:
            data_to_write, detected_platform = _parse_underdog(lines, date_detector)

        if not data_to_write:
            logging.warning(f"No valid props parsed. Platform detected: {detected_platform}. Check input format.")
            return

        header = Cols.get_required_input_cols()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        print("-" * 40)
        print(f"PARSING COMPLETE: {output_path}")
        print(f"Platform Detected: {detected_platform}")
        print(f"Total Lines Scanned: {len(lines)}")
        print(f"Valid Props Extracted: {len(data_to_write)}")
        print("-" * 40)
        
        logging.info(f"Successfully converted {detected_platform} props to {output_path} ({len(data_to_write)} rows)")
        
        # Save historical record
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rec_path = cfg.INPUT_DIR / "records" / f"{now_ts}.csv"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, 'w', newline='', encoding='utf-8') as f_rec:
            writer = csv.writer(f_rec)
            writer.writerow(header)
            writer.writerows(data_to_write)

        # Update historical master 
        update_master_prop_history(data_to_write, header)

    except Exception as e:
        logging.error(f"Error parsing props: {e}", exc_info=True)