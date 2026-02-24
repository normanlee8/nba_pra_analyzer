import pandas as pd
import numpy as np
import logging
from sklearn.metrics import brier_score_loss
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import text

def calculate_derived_stats(df):
    """Calculates composite stats (PRA, PA, etc.) from raw box score columns."""
    for stat in ['PTS', 'REB', 'AST']:
        if stat in df.columns:
            df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
        else:
            df[stat] = 0

    df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    df['PR'] = df['PTS'] + df['REB']
    df['PA'] = df['PTS'] + df['AST']
    df['RA'] = df['REB'] + df['AST']
            
    return df

def check_prop_row(row):
    """Compares the prediction against actual value to determine correctness and error."""
    prop_cat_clean = str(row.get(Cols.PROP_TYPE, '')).strip()
    prop_map_lookup = cfg.MASTER_PROP_MAP.get(prop_cat_clean, prop_cat_clean)
    
    try:
        line_val = row.get(Cols.PROP_LINE)
        if pd.isna(line_val): return pd.Series([None, 'Error', None, None])
        line = float(line_val)
        
        actual = row.get(prop_map_lookup)
        if pd.isna(actual): actual = row.get(prop_cat_clean)
            
        if actual is not None: actual = float(actual)
            
    except (ValueError, TypeError):
        return pd.Series([None, 'Error', None, None])
        
    if pd.isna(actual): 
        return pd.Series([None, 'Missing Data', None, None])
    
    res = 'Over' if actual > line else ('Under' if actual < line else 'Push')
    
    my_pick = row.get('Pick', row.get(Cols.EDGE_TYPE)) 
    
    correctness = 'Incorrect'
    if res == 'Push': correctness = 'Push'
    elif res == my_pick: correctness = 'Correct'

    proj = row.get('Proj')
    error = np.nan
    if not pd.isna(proj):
        error = actual - float(proj)
    
    return pd.Series([actual, res, correctness, error])

def grade_predictions():
    logging.info("--- Grading Predictions vs Actuals ---")
    
    try:
        props_file = cfg.PROCESSED_OUTPUT_SYSTEM
        if not props_file.exists():
            props_file = cfg.PROCESSED_OUTPUT_XLSX.with_suffix('.csv') 
            if not props_file.exists():
                logging.warning(f"No processed props file found to grade.")
                return
            
        df_props = pd.read_parquet(props_file) if props_file.suffix == '.parquet' else pd.read_csv(props_file)
        
        if not cfg.MASTER_BOX_SCORES_FILE.exists():
            logging.warning("No master box scores found. Cannot grade.")
            return
            
        df_box = pd.read_parquet(cfg.MASTER_BOX_SCORES_FILE)
    except Exception as e:
        logging.error(f"Error loading files for grading: {e}")
        return

    if df_props.empty: return

    if Cols.PLAYER_NAME not in df_props.columns or Cols.DATE not in df_props.columns:
        logging.error("Schema mismatch: Missing Name or Date columns.")
        return

    df_props['join_player'] = df_props[Cols.PLAYER_NAME].apply(text.preprocess_name_for_fuzzy_match)
    df_props['join_date'] = pd.to_datetime(df_props[Cols.DATE], errors='coerce').dt.strftime('%Y-%m-%d')
    
    df_box['join_player'] = df_box['PLAYER_NAME'].apply(text.preprocess_name_for_fuzzy_match)
    date_col_box = Cols.DATE if Cols.DATE in df_box.columns else 'GAME_DATE'
    df_box['join_date'] = pd.to_datetime(df_box[date_col_box], errors='coerce').dt.strftime('%Y-%m-%d')
    
    df_box = calculate_derived_stats(df_box)
    df_merged = pd.merge(df_props, df_box, on=['join_player', 'join_date'], how='left', suffixes=('', '_box'))

    out_cols = [Cols.ACTUAL_VAL, Cols.RESULT, Cols.CORRECTNESS, 'Proj_Error']
    df_merged[out_cols] = df_merged.apply(check_prop_row, axis=1)
    
    try:
        save_path = cfg.GRADED_DIR / f"graded_props_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet"
        df_merged.to_parquet(save_path, index=False)
        logging.info(f"Graded results saved to {save_path.name}")
    except Exception as e:
        logging.error(f"Failed to save grading results: {e}")
    
    if Cols.CORRECTNESS in df_merged.columns:
        graded = df_merged[df_merged[Cols.CORRECTNESS].isin(['Correct', 'Incorrect'])].copy()
        
        def log_performance(subset, label):
            total = len(subset)
            if total > 0:
                correct = len(subset[subset[Cols.CORRECTNESS] == 'Correct'])
                acc = (correct / total) * 100
                
                # Brier Score for Probability Accuracy
                brier_str = "N/A"
                if 'Prob' in subset.columns:
                    y_true = (subset[Cols.CORRECTNESS] == 'Correct').astype(int)
                    y_prob = pd.to_numeric(subset['Prob'], errors='coerce').fillna(0.5)
                    brier = brier_score_loss(y_true, y_prob)
                    brier_str = f"{brier:.3f}"
                
                logging.info(f"[{label}] Acc: {acc:.2f}% ({correct}/{total}) | Brier: {brier_str}")
            else:
                logging.info(f"[{label}] No graded data available.")

        logging.info("-" * 50)
        logging.info("PERFORMANCE SUMMARY (Win Rate & Probability Accuracy)")
        
        log_performance(graded, "Total Graded Props")
        
        if 'Tier' in graded.columns:
            for tier in ['S Tier', 'A Tier', 'B Tier']:
                tier_df = graded[graded['Tier'] == tier]
                if not tier_df.empty:
                    log_performance(tier_df, f"{tier} Props")
                    
        logging.info("--- PERFORMANCE BY CATEGORY ---")
        graded['Mapped_Prop'] = graded[Cols.PROP_TYPE].map(lambda x: cfg.MASTER_PROP_MAP.get(x, x))
        categories = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
        
        for cat in categories:
            cat_df = graded[graded['Mapped_Prop'] == cat]
            if not cat_df.empty:
                log_performance(cat_df, f"{cat} Props")
        
        logging.info("-" * 50)