import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import etl, dataset
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="build_db")
    
    try:
        logging.info(">>> STARTING ETL PIPELINE (Parquet Optimized) <<<")
        
        # --- PHASE 1: ETL (Extract, Transform, Load) ---
        logging.info("Step 1: Aggregating Master Files from Season Folders...")
        
        # 1. Identify Season Folders
        season_folders = etl.get_season_folders(cfg.DATA_DIR)
        if not season_folders:
            logging.critical(f"No season folders (e.g., 2024-25) found in {cfg.DATA_DIR}")
            return
            
        logging.info(f"Found Seasons: {[f.name for f in season_folders]}")

        # 2. Build ID Map
        # Scans all years to map Player Names to unique IDs
        player_id_map = etl.create_player_id_map(cfg.DATA_DIR, season_folders)
        if player_id_map is None:
            logging.critical("Failed to create Player ID Map. Aborting.")
            return

        # 3. Process Master Stats (Saves as .parquet)
        etl.process_master_player_stats(player_id_map, season_folders, cfg.DATA_DIR)
        etl.process_master_team_stats(player_id_map, season_folders, cfg.DATA_DIR)
        
        # Note: Box Scores depends on Player Stats existing first
        etl.process_master_box_scores(player_id_map, season_folders, cfg.DATA_DIR)
        
        # 4. Derivative Stats (Vs Opponent & DVP)
        etl.process_vs_opponent_stats(cfg.DATA_DIR, cfg.DATA_DIR)
        etl.process_dvp_stats(cfg.DATA_DIR)

        # 5. NEW: Process Q1 History
        # This aggregates the daily Q1 scrapes into master_q1_stats.parquet
        etl.process_q1_history(cfg.DATA_DIR)

        # --- PHASE 2: Dataset Creation ---
        logging.info("Step 2: Building Final Training Dataset...")
        
        # This reads the parquet master files and adds rolling features (SZN_AVG, L5, etc.)
        dataset.create_training_dataset()
        
        logging.info("<<< DATABASE BUILD COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Database Build: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()