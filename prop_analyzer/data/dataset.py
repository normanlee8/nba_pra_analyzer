import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.data import loader
from prop_analyzer.features import generator

def create_training_dataset():
    logging.info("--- Building Final Training Dataset ---")
    
    # 1. Load Base Box Scores
    box_scores = loader.load_box_scores()
    if box_scores is None or box_scores.empty:
        logging.error("No box scores available. Cannot build training set.")
        return

    # Ensure Date Standardization for merging
    if Cols.DATE in box_scores.columns:
        box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE]).dt.normalize()

    # 2. Merge Real Vegas Lines (History)
    # This allows training to use actual historical lines instead of synthetic ones
    prop_hist_path = cfg.MASTER_PROP_HISTORY_FILE
    if prop_hist_path.exists():
        logging.info("Merging real historical Vegas lines...")
        try:
            prop_hist = pd.read_parquet(prop_hist_path)
            
            # Normalize dates to match box_scores
            if Cols.DATE in prop_hist.columns:
                prop_hist[Cols.DATE] = pd.to_datetime(prop_hist[Cols.DATE]).dt.normalize()
            
            # We pivot the data to get one row per game with columns like 'Line_PTS', 'Line_REB'
            if Cols.PROP_TYPE in prop_hist.columns and Cols.PROP_LINE in prop_hist.columns:
                # Deduplicate before pivot
                prop_hist = prop_hist.drop_duplicates(subset=[Cols.PLAYER_NAME, Cols.DATE, Cols.PROP_TYPE])
                
                # Pivot
                pivoted = prop_hist.pivot(
                    index=[Cols.PLAYER_NAME, Cols.DATE], 
                    columns=Cols.PROP_TYPE, 
                    values=Cols.PROP_LINE
                ).reset_index()
                
                # Rename columns: PTS -> Line_PTS
                pivoted.columns = [
                    f"Line_{c}" if c not in [Cols.PLAYER_NAME, Cols.DATE] else c 
                    for c in pivoted.columns
                ]
                
                # Merge into box scores
                if Cols.PLAYER_NAME in box_scores.columns:
                    box_scores = pd.merge(
                        box_scores,
                        pivoted,
                        on=[Cols.PLAYER_NAME, Cols.DATE],
                        how='left'
                    )
                    logging.info(f"Merged lines. Columns added: {[c for c in pivoted.columns if 'Line_' in c]}")
                else:
                    logging.warning("Box scores missing Player Name, cannot merge lines by name.")
                    
        except Exception as e:
            logging.warning(f"Failed to merge prop history: {e}")

    # 3. Generate Features (Rolling Averages, etc.)
    # This adds the PRE-GAME context (e.g., L5_AVG) needed for training
    logging.info("Calculating advanced features for training set...")
    training_df = generator.add_rolling_stats_history(box_scores.copy())

    # 4. Save Final Dataset
    logging.info(f"Saving training set with {training_df.shape[1]} columns...")
    training_df.to_parquet(cfg.MASTER_TRAINING_FILE, index=False)
    logging.info(f"Saved to {cfg.MASTER_TRAINING_FILE}")

if __name__ == "__main__":
    # Setup simple console logging if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_training_dataset()