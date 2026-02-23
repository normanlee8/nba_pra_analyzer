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

    if 'PLAYER_NAME' in box_scores.columns and Cols.PLAYER_NAME not in box_scores.columns:
        box_scores.rename(columns={'PLAYER_NAME': Cols.PLAYER_NAME}, inplace=True)

    # --- NEW: Calculate Historical Home/Away Differentials (No Data Leakage) ---
    logging.info("Calculating historical point-in-time Home/Away differentials...")
    stat_cols = ['PTS', 'REB', 'AST', 'PRA', 'MIN']
    
    # Sort chronologically to strictly enforce point-in-time calculations
    box_scores = box_scores.sort_values(by=[Cols.PLAYER_ID, Cols.DATE])
    
    for col in stat_cols:
        if col in box_scores.columns:
            # Isolate Home and Away stats into temporary columns
            box_scores[f'{col}_TEMP_HOME'] = np.where(box_scores['IS_HOME'] == 1, box_scores[col], np.nan)
            box_scores[f'{col}_TEMP_AWAY'] = np.where(box_scores['IS_HOME'] == 0, box_scores[col], np.nan)
            
            # Forward fill the expanding means so an Away game still knows what the current Home average is
            box_scores[f'{col}_RUNNING_HOME'] = box_scores.groupby(Cols.PLAYER_ID)[f'{col}_TEMP_HOME'].transform(
                lambda x: x.expanding().mean().shift(1).ffill()
            ).fillna(0.0)
            
            box_scores[f'{col}_RUNNING_AWAY'] = box_scores.groupby(Cols.PLAYER_ID)[f'{col}_TEMP_AWAY'].transform(
                lambda x: x.expanding().mean().shift(1).ffill()
            ).fillna(0.0)
            
            # Calculate the point-in-time differential (Home - Away)
            box_scores[f'{col}_DIFF'] = box_scores[f'{col}_RUNNING_HOME'] - box_scores[f'{col}_RUNNING_AWAY']
            
            # Drop the temporary calculation columns
            box_scores.drop(columns=[
                f'{col}_TEMP_HOME', f'{col}_TEMP_AWAY', 
                f'{col}_RUNNING_HOME', f'{col}_RUNNING_AWAY'
            ], inplace=True)

    # Composite Combo Splits (PR, PA, RA)
    box_scores['PR_SPLIT_AVG'] = box_scores.get('PTS_SPLIT_AVG', 0) + box_scores.get('REB_SPLIT_AVG', 0)
    box_scores['PA_SPLIT_AVG'] = box_scores.get('PTS_SPLIT_AVG', 0) + box_scores.get('AST_SPLIT_AVG', 0)
    box_scores['RA_SPLIT_AVG'] = box_scores.get('REB_SPLIT_AVG', 0) + box_scores.get('AST_SPLIT_AVG', 0)
    
    box_scores['PR_DIFF'] = box_scores.get('PTS_DIFF', 0) + box_scores.get('REB_DIFF', 0)
    box_scores['PA_DIFF'] = box_scores.get('PTS_DIFF', 0) + box_scores.get('AST_DIFF', 0)
    box_scores['RA_DIFF'] = box_scores.get('REB_DIFF', 0) + box_scores.get('AST_DIFF', 0)

    # 2. Merge Real Vegas Lines (History)
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
    logging.info("Calculating advanced features for training set...")
    training_df = generator.add_rolling_stats_history(box_scores.copy())

    # 4. Save Final Dataset
    logging.info(f"Saving training set with {training_df.shape[1]} columns...")
    training_df.to_parquet(cfg.MASTER_TRAINING_FILE, index=False)
    logging.info(f"Saved to {cfg.MASTER_TRAINING_FILE}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_training_dataset()