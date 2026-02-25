import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import common

def print_accuracy_report(df, label="Total"):
    """Helper to print formatted percentage stats"""
    total = len(df)
    if total == 0:
        return

    wins = len(df[df[Cols.RESULT] == 'WIN'])
    losses = len(df[df[Cols.RESULT] == 'LOSS'])
    pushes = len(df[df[Cols.RESULT] == 'PUSH'])
    
    decided = wins + losses
    if decided > 0:
        acc = (wins / decided) * 100
        logging.info(f"{label}: {acc:.1f}% ({wins}/{decided}) [Pushes: {pushes}]")
    else:
        logging.info(f"{label}: N/A (Only Pushes)")

def save_user_scorecard(df, date_str):
    """
    Saves a clean, human-readable Excel file for the user.
    Applies conditional formatting: WIN = Green, LOSS = Red.
    """
    col_map = {
        Cols.PLAYER_NAME: 'Player',
        'Team': 'Team',
        'Opponent': 'Opponent',
        Cols.PROP_TYPE: 'Prop',
        Cols.PROP_LINE: 'Line',
        Cols.DATE: 'Date',
        Cols.PREDICTION: 'Proj',
        Cols.CONFIDENCE: 'Prob',
        Cols.EDGE_TYPE: 'Pick',
        Cols.TIER: 'Tier',
        Cols.ACTUAL_VAL: 'Actual',
        Cols.RESULT: 'Result'
    }
    
    optional_map = {
        'Units': 'Units',
        'Diff%': 'Diff%',
        'L5': 'L5',
        'SZN': 'SZN'
    }
    
    final_rename = {}
    for sys_col, user_col in col_map.items():
        if sys_col in df.columns:
            final_rename[sys_col] = user_col
            
    for sys_col, user_col in optional_map.items():
        if sys_col in df.columns:
            final_rename[sys_col] = user_col

    if not final_rename:
        logging.warning("No matching columns found for scorecard.")
        return

    clean_df = df[list(final_rename.keys())].rename(columns=final_rename).copy()
    
    scorecard_dir = cfg.GRADED_DIR / "user_scorecards"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    
    scorecard_path = scorecard_dir / f"{date_str}.xlsx"
    
    try:
        def color_result(val):
            if val == 'WIN':
                return 'color: #008000; font-weight: bold' # Green
            elif val == 'LOSS':
                return 'color: #FF0000; font-weight: bold' # Red
            elif val == 'PUSH':
                return 'color: #808080; font-weight: bold' # Gray
            return ''

        if 'Result' in clean_df.columns:
            try:
                styler = clean_df.style.map(color_result, subset=['Result'])
            except AttributeError:
                styler = clean_df.style.applymap(color_result, subset=['Result'])
        else:
            styler = clean_df.style

        styler.to_excel(scorecard_path, index=False, engine='openpyxl')
        
    except Exception as e:
        logging.error(f"Failed to save user scorecard: {e}")

def grade_predictions():
    # 1. Load Predictions
    preds_path = cfg.PROCESSED_OUTPUT_SYSTEM
    if not preds_path.exists():
        logging.critical(f"No predictions file found at {preds_path}")
        return pd.DataFrame(), None 

    try:
        preds_df = pd.read_parquet(preds_path)
        if preds_df.empty:
            logging.warning("Predictions file is empty.")
            return pd.DataFrame(), None
            
        clean_map = {
            'Player': Cols.PLAYER_NAME,
            'Prop': Cols.PROP_TYPE,
            'Line': Cols.PROP_LINE,
            'Date': Cols.DATE,
            'Pick': Cols.EDGE_TYPE,
            'Prob': Cols.CONFIDENCE,
            'Proj': Cols.PREDICTION,
            'Tier': Cols.TIER
        }
        actual_rename = {k: v for k, v in clean_map.items() if k in preds_df.columns and v not in preds_df.columns}
        if actual_rename:
            preds_df.rename(columns=actual_rename, inplace=True)
            
    except Exception as e:
        logging.critical(f"Failed to load predictions: {e}")
        return pd.DataFrame(), None

    # 2. Load Truth Data
    logging.info("Loading historical data for grading...")
    
    # Load raw box scores instead of master DB to ensure PLAYER_NAME is present
    raw_files = list(cfg.DATA_DIR.glob("*/NBA Player Box Scores.parquet"))
    if not raw_files:
        logging.warning("No raw box scores found. Props cannot be graded.")
        return pd.DataFrame(), None
        
    full_game_df = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
    if 'GAME_DATE' in full_game_df.columns and Cols.DATE not in full_game_df.columns:
        full_game_df.rename(columns={'GAME_DATE': Cols.DATE}, inplace=True)

    # --- Combo Stats Calculation (Numeric forced for ESPN API) ---
    for col in ['PTS', 'REB', 'AST']:
        if col not in full_game_df.columns: 
            full_game_df[col] = 0
        else:
            full_game_df[col] = pd.to_numeric(full_game_df[col], errors='coerce').fillna(0)

    full_game_df['PRA'] = full_game_df['PTS'] + full_game_df['REB'] + full_game_df['AST']
    full_game_df['PR'] = full_game_df['PTS'] + full_game_df['REB']
    full_game_df['PA'] = full_game_df['PTS'] + full_game_df['AST']
    full_game_df['RA'] = full_game_df['REB'] + full_game_df['AST']

    logging.info(f"Grading {len(preds_df)} predictions...")

    # Normalize Dates
    if Cols.DATE in full_game_df.columns:
        full_game_df[Cols.DATE] = pd.to_datetime(full_game_df[Cols.DATE]).dt.normalize()

    preds_df['Match_Date'] = pd.to_datetime(preds_df[Cols.DATE]).dt.normalize()
    prop_map = cfg.MASTER_PROP_MAP
    
    results = []
    
    for idx, row in preds_df.iterrows():
        prop_type = row[Cols.PROP_TYPE]
        p_date = row['Match_Date']
        
        prop_key = str(prop_map.get(prop_type, prop_type))
        truth_df = full_game_df
        data_col = prop_key

        mask = None
        # Try finding by ID
        if Cols.PLAYER_ID in row and pd.notna(row[Cols.PLAYER_ID]):
             p_id = int(row[Cols.PLAYER_ID])
             if Cols.PLAYER_ID in truth_df.columns:
                 mask = (truth_df[Cols.PLAYER_ID] == p_id) & (truth_df[Cols.DATE] == p_date)
        
        # Fallback to Name match (Robust)
        if mask is None or mask.sum() == 0:
            p_name = str(row.get(Cols.PLAYER_NAME, '')).lower().strip()
             
             # Try common name columns to guarantee a match
            name_col = None
            for col in ['PLAYER_NAME', Cols.PLAYER_NAME, 'Player', 'Player_Name']:
                 if col in truth_df.columns:
                     name_col = col
                     break
                     
            if name_col:
                # Allow a 1-day window FORWARD in case props were fetched the night before,
                # but strictly prevent looking backwards at yesterday's games.
                date_diff = (truth_df[Cols.DATE] - p_date).dt.days
                mask = (truth_df[name_col].astype(str).str.lower().str.strip() == p_name) & (date_diff >= 0) & (date_diff <= 1)
             
        if mask is not None:
            match = truth_df[mask]
        else:
            match = pd.DataFrame()
        
        if match.empty:
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Pending / Not Found'
            results.append(row)
            continue
            
        if data_col not in match.columns:
             if 'Points' in prop_type: data_col = 'PTS'
             elif 'Rebounds' in prop_type: data_col = 'REB'
             elif 'Assists' in prop_type: data_col = 'AST'

        if data_col not in match.columns:
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = f'Stat {data_col} Missing'
            results.append(row)
            continue

        actual = match.iloc[0].get(data_col)
        
        if pd.isna(actual):
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Stat is NaN'
            results.append(row)
            continue
            
        row[Cols.ACTUAL_VAL] = actual
        
        try:
            line = float(row[Cols.PROP_LINE])
            pick = row[Cols.EDGE_TYPE]
            
            if pick == 'Over':
                if actual > line: res = 'WIN'
                elif actual < line: res = 'LOSS'
                else: res = 'PUSH'
            elif pick == 'Under':
                if actual < line: res = 'WIN'
                elif actual > line: res = 'LOSS'
                else: res = 'PUSH'
            else:
                res = 'ERROR'
        except Exception:
            res = 'Error Grading'
            
        row[Cols.RESULT] = res
        row[Cols.CORRECTNESS] = 1 if res == 'WIN' else 0
        
        # Capture error difference for analytics
        if pd.notna(row.get(Cols.PREDICTION)):
            row['Proj_Error'] = float(actual) - float(row[Cols.PREDICTION])
        else:
            row['Proj_Error'] = None

        results.append(row)

    graded_df = pd.DataFrame(results)
    
    # Determine the actual game date based on the data
    if not graded_df.empty and 'Match_Date' in graded_df.columns:
        try:
            # Get the most common date in the dataset
            game_date = pd.to_datetime(graded_df['Match_Date']).dt.date.mode()[0]
            game_date_str = game_date.strftime("%Y-%m-%d")
        except Exception:
            game_date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        game_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # 7. Reporting
    logging.info("-" * 40)
    logging.info(">>> GRADING REPORT <<<")
    
    if graded_df.empty:
        logging.warning("No results to grade.")
        return pd.DataFrame(), game_date_str

    # Add mapping to graded_df BEFORE creating the finished subset
    graded_df['Mapped_Prop'] = graded_df[Cols.PROP_TYPE].map(lambda x: cfg.MASTER_PROP_MAP.get(x, x))
    
    finished = graded_df[graded_df[Cols.RESULT].isin(['WIN', 'LOSS', 'PUSH'])]
    
    print_accuracy_report(finished, "Total Props")
    
    if Cols.TIER in finished.columns:
        # Include our newly established tier 'Trap / High Variance' if any fell into it
        for tier in ['S Tier', 'A Tier', 'B Tier', 'Trap / High Variance']:
            tier_df = finished[finished[Cols.TIER] == tier]
            if not tier_df.empty:
                print_accuracy_report(tier_df, f"{tier} Props")

    # --- CATEGORY REPORTING ---
    logging.info(">>> CATEGORY HIT RATES <<<")
    categories = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']
    
    for cat in categories:
        cat_df = finished[finished['Mapped_Prop'] == cat]
        if not cat_df.empty:
            print_accuracy_report(cat_df, f"{cat} Props")

    logging.info("-" * 40)
    
    # 8. Save Outputs
    parquet_path = cfg.GRADED_DIR / f"graded_props_{game_date_str}.parquet"
    csv_path = cfg.GRADED_DIR / f"graded_{game_date_str}.csv"
    
    try:
        # Convert objects to string for saving
        for col in graded_df.select_dtypes(include=['object']).columns:
            graded_df[col] = graded_df[col].astype(str)
            
        graded_df.to_parquet(parquet_path, index=False)
        graded_df.to_csv(csv_path, index=False)
        
        save_user_scorecard(graded_df, game_date_str)
        logging.info(f"Saved graded results for game date: {game_date_str}")
        
    except Exception as e:
        logging.error(f"Failed to save output: {e}")
        
    return graded_df, game_date_str

def analyze_strengths_and_weaknesses(graded_df):
    """Provides a deep dive into where the model leaks value."""
    logging.info(">>> POST-MORTEM SYSTEM ANALYSIS <<<")
    
    if 'Proj_Error' not in graded_df.columns:
        return
        
    graded_df['Proj_Error'] = pd.to_numeric(graded_df['Proj_Error'], errors='coerce')
    graded_df['Abs_Error'] = graded_df['Proj_Error'].abs()
    
    # Where is the model missing by the widest margin?
    worst_props = graded_df.groupby(Cols.PROP_TYPE)['Abs_Error'].mean().sort_values(ascending=False)
    logging.info(f"\nHighest Average Error by Stat Type:\n{worst_props.head(3).to_string()}")
    
    # Are we systematically over-projecting or under-projecting?
    bias_props = graded_df.groupby(Cols.PROP_TYPE)['Proj_Error'].mean()
    logging.info(f"\nSystematic Bias (Positive = Under-projecting, Negative = Over-projecting):\n{bias_props.to_string()}")

def grade_parlays(graded_props_df, game_date_str):
    """Grades historical parlays by parsing the 'Picks' column string."""
    parlay_path = cfg.OUTPUT_DIR / "processed_parlays.csv"
    if not parlay_path.exists():
        logging.warning("No processed_parlays.csv found to grade.")
        return
        
    logging.info(">>> GRADING PARLAYS <<<")
    parlays_df = pd.read_csv(parlay_path)
    
    # Ensure Mapped_Prop exists
    if 'Mapped_Prop' not in graded_props_df.columns:
        graded_props_df['Mapped_Prop'] = graded_props_df[Cols.PROP_TYPE].map(lambda x: cfg.MASTER_PROP_MAP.get(x, x))
        
    graded_props_df['Lookup_Key'] = graded_props_df[Cols.PLAYER_NAME].str.lower() + "_" + graded_props_df['Mapped_Prop']
    result_map = dict(zip(graded_props_df['Lookup_Key'], graded_props_df[Cols.RESULT]))
    
    parlay_results = []
    total_wagered = 0
    total_won = 0
    
    for idx, row in parlays_df.iterrows():
        try:
            ticket_str = str(row.get('Picks', ''))
            payout_str = str(row.get('Payout', '0')).replace('x', '')
            payout = float(payout_str)
            
            # The CSV outputs legs separated by " | "
            legs = ticket_str.split(' | ')
            leg_results = []
            
            for leg in legs:
                parts = leg.strip().split()
                # Expected string format: "Aaron Wiggins Under 14.5 PTS"
                # So we need at least 4 parts: [First, Last, Pick, Line, Prop]
                if len(parts) >= 4:
                    prop = parts[-1] 
                    # Everything before the last 3 items (Pick, Line, Prop) is the name
                    player_name = " ".join(parts[:-3]).strip().lower()
                    key = f"{player_name}_{prop}"
                    
                    leg_results.append(result_map.get(key, 'PENDING'))
                else:
                    leg_results.append('PARSE_ERROR')
                    
            if 'PENDING' in leg_results or 'PARSE_ERROR' in leg_results or 'ERROR' in leg_results:
                status = 'PENDING/ERROR'
            elif 'LOSS' in leg_results:
                status = 'LOSS'
            elif all(res == 'WIN' for res in leg_results):
                status = 'WIN'
            else:
                # E.g., Wins and a Push = Downgraded Parlay
                status = 'PUSH/DOWNGRADE'
                
            parlay_results.append(status)
            
            # ROI Calculations (Assuming 1 Unit flat bet per ticket)
            if status in ['WIN', 'LOSS', 'PUSH/DOWNGRADE']:
                total_wagered += 1 
            if status == 'WIN':
                total_won += payout
                
        except Exception as e:
            parlay_results.append('PARSE_ERROR')
            
    parlays_df['Result'] = parlay_results
    
    wins = len(parlays_df[parlays_df['Result'] == 'WIN'])
    losses = len(parlays_df[parlays_df['Result'] == 'LOSS'])
    total_decided = wins + losses
    
    if total_decided > 0:
        roi = ((total_won - total_wagered) / total_wagered) * 100
        logging.info(f"Parlay Win Rate: {(wins/total_decided)*100:.1f}% ({wins}/{total_decided})")
        logging.info(f"Parlay ROI: {roi:.2f}% (Units Wagered: {total_wagered}, Units Won: {total_won:.2f})")
    else:
        logging.info("No parlays fully completed yet.")
    
    # Save graded parlays using the game date instead of system date
    parlays_df.to_csv(cfg.GRADED_DIR / f"graded_parlays_{game_date_str}.csv", index=False)

def main():
    common.setup_logging(name="grading")
    
    # 1. Grade individual props
    graded_df, game_date_str = grade_predictions() 
    
    # 2. Grade Parlays & Generate Insights
    if not graded_df.empty:
        grade_parlays(graded_df, game_date_str)
        analyze_strengths_and_weaknesses(graded_df)
    else:
        # Fallback: Peek at PROCESSED_OUTPUT_SYSTEM to find the target date
        try:
            if cfg.PROCESSED_OUTPUT_SYSTEM.exists():
                preds = pd.read_parquet(cfg.PROCESSED_OUTPUT_SYSTEM)
                date_col = 'Date' if 'Date' in preds.columns else Cols.DATE
                fallback_date_str = pd.to_datetime(preds[date_col]).dt.date.mode()[0].strftime("%Y-%m-%d")
            else:
                fallback_date_str = datetime.now().strftime("%Y-%m-%d")
        except Exception:
            fallback_date_str = datetime.now().strftime("%Y-%m-%d")

        parquet_path = cfg.GRADED_DIR / f"graded_props_{fallback_date_str}.parquet"
        
        if parquet_path.exists():
            graded_df = pd.read_parquet(parquet_path)
            grade_parlays(graded_df, fallback_date_str)
            analyze_strengths_and_weaknesses(graded_df)

if __name__ == "__main__":
    main()