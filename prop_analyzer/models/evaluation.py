import pandas as pd
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import text

def calculate_derived_stats(df):
    """
    Calculates composite stats (PRA, PA, etc.) from raw box score columns.
    Includes Quarters and Halves.
    """
    # Derived Quarter Stats
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pts, reb, ast = f'{q}_PTS', f'{q}_REB', f'{q}_AST'
        # Only calculate if all components exist (prevent partial sums)
        if pts in df.columns and reb in df.columns and ast in df.columns:
            df[f'{q}_PRA'] = df[pts] + df[reb] + df[ast]
            df[f'{q}_PR'] = df[pts] + df[reb]
            df[f'{q}_PA'] = df[pts] + df[ast]
            df[f'{q}_RA'] = df[reb] + df[ast]

    # Derived Halves (1H = Q1 + Q2)
    base_stats = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV', 'FG3M']
    for stat in base_stats:
        q1_col = f'Q1_{stat}'
        q2_col = f'Q2_{stat}'
        
        if q1_col in df.columns and q2_col in df.columns:
            df[f'1H_{stat}'] = df[q1_col] + df[q2_col]
            
    return df

def check_prop_row(row):
    """
    Compares the prediction against the actual value to determine correctness.
    Uses strictly standardized columns from Cols class.
    """
    # 1. Get Prop Category (e.g., 'Points')
    prop_cat_clean = str(row.get(Cols.PROP_TYPE, '')).strip()
    
    # 2. Map to DB Column (e.g., 'Points' -> 'PTS')
    prop_map_lookup = cfg.MASTER_PROP_MAP.get(prop_cat_clean, prop_cat_clean)
    
    try:
        # 3. Get Line
        line_val = row.get(Cols.PROP_LINE)
        if pd.isna(line_val):
            return pd.Series([None, 'Error', None])
        line = float(line_val)
        
        # 4. Get Actual Value from merged Box Score data
        # The merge in grade_predictions suffixes the box score cols (usually no suffix or handled there)
        # But here 'row' contains the merged data.
        actual = row.get(prop_map_lookup)
        
        # If lookup failed (maybe it's a raw column like 'PTS' already), try direct
        if pd.isna(actual):
            actual = row.get(prop_cat_clean)
            
        if actual is not None:
            actual = float(actual)
            
    except (ValueError, TypeError):
        return pd.Series([None, 'Error', None])
        
    if pd.isna(actual): 
        return pd.Series([None, 'Missing Data', None])
    
    # 5. Determine Result (Over/Under/Push)
    if actual > line: 
        res = 'Over'
    elif actual < line: 
        res = 'Under'
    else:
        res = 'Push'
    
    # 6. Determine Correctness
    # We compare the Result against our Pick (Cols.EDGE_TYPE)
    my_pick = row.get(Cols.EDGE_TYPE) # Expected: 'Over' or 'Under'
    
    correctness = 'Incorrect'
    if res == 'Push':
        correctness = 'Push'
    elif res == my_pick:
        correctness = 'Correct'
    
    return pd.Series([actual, res, correctness])

def grade_predictions():
    logging.info("--- Grading Predictions vs Actuals ---")
    
    # 1. Load Data
    try:
        props_file = cfg.PROCESSED_OUTPUT
        if not props_file.exists():
            logging.warning(f"No processed props file found at {props_file}")
            return
            
        df_props = pd.read_csv(props_file)
        
        if not cfg.MASTER_BOX_SCORES_FILE.exists():
            logging.warning("No master box scores found. Cannot grade.")
            return
            
        df_box = pd.read_csv(cfg.MASTER_BOX_SCORES_FILE, low_memory=False)
    except Exception as e:
        logging.error(f"Error loading files for grading: {e}")
        return

    if df_props.empty:
        logging.warning("Props file is empty.")
        return

    # 2. Prep & Normalize
    # Standardize Player Names for Join
    if Cols.PLAYER_NAME not in df_props.columns:
        logging.error(f"Schema mismatch: {Cols.PLAYER_NAME} missing from props file.")
        return

    df_props['join_player'] = df_props[Cols.PLAYER_NAME].apply(text.preprocess_name_for_fuzzy_match)
    
    # Ensure DATE exists and is formatted
    if Cols.DATE not in df_props.columns:
        logging.error(f"Schema mismatch: {Cols.DATE} missing from props file.")
        return

    df_props['join_date'] = pd.to_datetime(df_props[Cols.DATE], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Prep Box Scores
    # Note: MASTER_BOX_SCORES has standard columns from etl.py
    df_box['join_player'] = df_box['PLAYER_NAME'].apply(text.preprocess_name_for_fuzzy_match)
    
    # Check if Cols.DATE is in box scores (standardized in etl.py) or fallback to GAME_DATE
    date_col_box = Cols.DATE if Cols.DATE in df_box.columns else 'GAME_DATE'
    df_box['join_date'] = pd.to_datetime(df_box[date_col_box], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Calculate derived stats (Q1, 1H, etc.) so we can grade those props
    df_box = calculate_derived_stats(df_box)

    # 3. Merge
    # Left merge preserves the order and rows of our predictions
    df_merged = pd.merge(
        df_props, 
        df_box, 
        on=['join_player', 'join_date'], 
        how='left', 
        suffixes=('', '_box')
    )

    # 4. Grade
    # Apply grading logic row-by-row
    out_cols = [Cols.ACTUAL_VAL, Cols.RESULT, Cols.CORRECTNESS]
    df_merged[out_cols] = df_merged.apply(check_prop_row, axis=1)
    
    # 5. Save Results
    # Overwrite the file with graded columns appended
    try:
        df_merged.to_csv(props_file, index=False)
        logging.info(f"Graded results saved to {props_file}")
    except Exception as e:
        logging.error(f"Failed to save grading results: {e}")
    
    # 6. Performance Summary Report
    # Filter strictly for Correct/Incorrect (Exclude Pushes/Missing)
    if Cols.CORRECTNESS in df_merged.columns:
        graded = df_merged[df_merged[Cols.CORRECTNESS].isin(['Correct', 'Incorrect'])]
        
        def log_accuracy(subset, label):
            total = len(subset)
            if total > 0:
                correct = len(subset[subset[Cols.CORRECTNESS] == 'Correct'])
                acc = (correct / total) * 100
                logging.info(f"Accuracy on {total} {label}: {acc:.2f}% ({correct}/{total})")
            else:
                logging.info(f"Accuracy on 0 {label}: N/A")

        logging.info("-" * 40)
        logging.info("PERFORMANCE SUMMARY")
        
        log_accuracy(graded, "Total Graded Props")
        
        # S-Tier Stats (if Tier column exists)
        if Cols.TIER in graded.columns:
            s_tier = graded[graded[Cols.TIER] == 'S Tier']
            log_accuracy(s_tier, "S-Tier Props")
            
            a_tier = graded[graded[Cols.TIER] == 'A Tier']
            log_accuracy(a_tier, "A-Tier Props")
        
        logging.info("-" * 40)
    else:
        logging.warning("Skipping summary: Correctness column not generated.")