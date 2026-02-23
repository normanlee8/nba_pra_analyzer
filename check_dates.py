import pandas as pd
from pathlib import Path
import sys

# Add project root to path so we can import your config
sys.path.append(str(Path(__file__).resolve().parent))
from prop_analyzer.config import Cols

def check_parquet_dates():
    # 1. Check the raw scraped data
    raw_path = Path("prop_data/2025-26/NBA Player Box Scores.parquet")
    if raw_path.exists():
        df_raw = pd.read_parquet(raw_path)
        
        # Determine the correct date column
        date_col = Cols.DATE if Cols.DATE in df_raw.columns else 'GAME_DATE'
        
        if date_col not in df_raw.columns:
            print(f"--- SCRAPER OUTPUT ---")
            print(f"Error: Could not find date column. Available columns: {df_raw.columns.tolist()}\n")
        else:
            max_date = df_raw[date_col].max()
            games_today = df_raw[df_raw[date_col] == max_date]['GAME_ID'].nunique()
            print(f"--- SCRAPER OUTPUT ---")
            print(f"Latest Date Scraped: {max_date}")
            print(f"Number of completed games logged for this date: {games_today}\n")
    else:
        print(f"File not found: {raw_path}\n")

    # 2. Check the master database file (what the grader actually uses)
    master_path = Path("prop_data/master_box_scores_2025-26.parquet")
    if master_path.exists():
        df_master = pd.read_parquet(master_path)
        
        # Determine the correct date column
        date_col = Cols.DATE if Cols.DATE in df_master.columns else 'GAME_DATE'
        
        if date_col not in df_master.columns:
            print(f"--- MASTER DB OUTPUT ---")
            print(f"Error: Could not find date column. Available columns: {df_master.columns.tolist()}")
        else:
            print(f"--- MASTER DB OUTPUT ---")
            print(f"Latest Date in Master DB: {df_master[date_col].max()}")
    else:
        print(f"File not found: {master_path}")

if __name__ == "__main__":
    check_parquet_dates()