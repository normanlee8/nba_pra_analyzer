import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.data import loader
from prop_analyzer.features import generator

def create_training_dataset():
    """
    Builds the final training dataset (master_training_dataset.parquet).
    Combines:
    1. Master Box Scores (Full Game)
    2. Master Q1 Stats (Targets for Q1 props)
    3. Master 1H Stats (Targets for 1H props)
    4. Real Vegas Lines (Merged from master_prop_history.parquet)
    5. Rolling Features (SZN_AVG, L5, etc.) calculated by generator.py
    """
    logging.info("--- Building Final Training Dataset ---")
    
    # 1. Load Base Box Scores
    box_scores = loader.load_box_scores()
    if box_scores is None or box_scores.empty:
        logging.error("No box scores available. Cannot build training set.")
        return

    # Ensure Date Standardization for merging
    if Cols.DATE in box_scores.columns:
        box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE]).dt.normalize()

    # Check for GAME_ID presence to improve merge accuracy
    has_game_id = Cols.GAME_ID in box_scores.columns

    # 2. Load Quarter/Half Targets
    q1_df = loader.load_master_q1_history()
    h1_df = loader.load_master_1h_history()
    
    # 3. Merge Q1 Data (Target Columns)
    if not q1_df.empty:
        logging.info(f"Merging {len(q1_df)} Q1 records...")
        
        if Cols.DATE in q1_df.columns:
            q1_df[Cols.DATE] = pd.to_datetime(q1_df[Cols.DATE]).dt.normalize()
        
        # Ensure numeric types
        for c in ['PTS', 'REB', 'AST']:
            if c in q1_df.columns:
                q1_df[c] = pd.to_numeric(q1_df[c], errors='coerce').fillna(0)

        # Calculate Combo Stats (if missing)
        q1_df['PRA'] = q1_df['PTS'] + q1_df['REB'] + q1_df['AST']
        q1_df['PR'] = q1_df['PTS'] + q1_df['REB']
        q1_df['PA'] = q1_df['PTS'] + q1_df['AST']
        q1_df['RA'] = q1_df['REB'] + q1_df['AST']
        
        # Prepare for merge
        q1_df = q1_df.rename(columns={
            'PTS': 'Q1_PTS', 'REB': 'Q1_REB', 'AST': 'Q1_AST',
            'FG3M': 'Q1_FG3M', 
            'PRA': 'Q1_PRA', 'PR': 'Q1_PR', 'PA': 'Q1_PA', 'RA': 'Q1_RA'
        })
        
        # Select only necessary columns to avoid conflicts
        cols_to_merge = [
            Cols.PLAYER_ID, Cols.DATE, 
            'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_FG3M', 
            'Q1_PRA', 'Q1_PR', 'Q1_PA', 'Q1_RA'
        ]
        
        # Add GAME_ID to merge keys if available in both
        merge_keys = [Cols.PLAYER_ID, Cols.DATE]
        if has_game_id and Cols.GAME_ID in q1_df.columns:
            cols_to_merge.append(Cols.GAME_ID)
            merge_keys.append(Cols.GAME_ID)
        
        q1_subset = q1_df[[c for c in cols_to_merge if c in q1_df.columns]]
        
        # Merge onto Box Scores
        box_scores = pd.merge(
            box_scores, 
            q1_subset,
            on=merge_keys,
            how='left'
        )
    
    # 4. Merge 1H Data (Target Columns)
    if not h1_df.empty:
        logging.info(f"Merging {len(h1_df)} 1H records...")
        
        if Cols.DATE in h1_df.columns:
            h1_df[Cols.DATE] = pd.to_datetime(h1_df[Cols.DATE]).dt.normalize()
            
        for c in ['PTS', 'REB', 'AST']:
            if c in h1_df.columns:
                h1_df[c] = pd.to_numeric(h1_df[c], errors='coerce').fillna(0)

        # Calculate Combo Stats
        h1_df['PRA'] = h1_df['PTS'] + h1_df['REB'] + h1_df['AST']
        h1_df['PR'] = h1_df['PTS'] + h1_df['REB']
        h1_df['PA'] = h1_df['PTS'] + h1_df['AST']
        h1_df['RA'] = h1_df['REB'] + h1_df['AST']
        
        h1_df = h1_df.rename(columns={
            'PTS': '1H_PTS', 'REB': '1H_REB', 'AST': '1H_AST',
            'FG3M': '1H_FG3M', 
            'PRA': '1H_PRA', 'PR': '1H_PR', 'PA': '1H_PA', 'RA': '1H_RA'
        })
        
        cols_to_merge = [
            Cols.PLAYER_ID, Cols.DATE, 
            '1H_PTS', '1H_REB', '1H_AST', '1H_FG3M', 
            '1H_PRA', '1H_PR', '1H_PA', '1H_RA'
        ]
        
        # Add GAME_ID to merge keys if available in both
        merge_keys = [Cols.PLAYER_ID, Cols.DATE]
        if has_game_id and Cols.GAME_ID in h1_df.columns:
            cols_to_merge.append(Cols.GAME_ID)
            merge_keys.append(Cols.GAME_ID)
            
        h1_subset = h1_df[[c for c in cols_to_merge if c in h1_df.columns]]
        
        box_scores = pd.merge(
            box_scores,
            h1_subset,
            on=merge_keys,
            how='left'
        )

    # 5. Merge Real Vegas Lines (History)
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
                # Note: We merge on Name + Date because History file might not have IDs
                if Cols.PLAYER_NAME in box_scores.columns:
                    start_len = len(box_scores)
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

    # 6. Generate Features (Rolling Averages, etc.)
    # This adds the PRE-GAME context (e.g., L5_AVG) needed for training
    logging.info("Calculating features for training set...")
    
    # A. Full Game Rolling
    training_df = generator.add_rolling_stats_history(box_scores.copy())
    
    # B. Q1 Rolling (Calculate rolling stats of the Q1 targets)
    if 'Q1_PTS' in training_df.columns:
        stats_to_roll = ['Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_FG3M', 'Q1_PRA', 'Q1_PR', 'Q1_PA', 'Q1_RA']
        training_df = generator.add_rolling_stats_history(
            training_df, 
            stats_to_roll=[c for c in stats_to_roll if c in training_df.columns]
        )
        
    # C. 1H Rolling
    if '1H_PTS' in training_df.columns:
        stats_to_roll = ['1H_PTS', '1H_REB', '1H_AST', '1H_FG3M', '1H_PRA', '1H_PR', '1H_PA', '1H_RA']
        training_df = generator.add_rolling_stats_history(
            training_df, 
            stats_to_roll=[c for c in stats_to_roll if c in training_df.columns]
        )

    # 7. Save Final Dataset
    logging.info(f"Saving training set with {training_df.shape[1]} columns...")
    training_df.to_parquet(cfg.MASTER_TRAINING_FILE, index=False)
    logging.info(f"Saved to {cfg.MASTER_TRAINING_FILE}")

if __name__ == "__main__":
    # Setup simple console logging if run directly
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    create_training_dataset()