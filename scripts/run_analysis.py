import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import generator
from prop_analyzer.models import inference
from prop_analyzer.utils import common

def save_pretty_excel(df, output_path):
    """
    Saves the dataframe to Excel with Autosizing and Conditional Formatting.
    """
    try:
        if df.empty: return

        has_xlsxwriter = False
        try:
            import xlsxwriter
            has_xlsxwriter = True
        except ImportError:
            logging.warning("XlsxWriter not installed. Saving standard CSV-style Excel.")

        if has_xlsxwriter:
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Picks', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Picks']
            
            # Formats
            pct_fmt = workbook.add_format({'num_format': '0.0%'})
            header_fmt = workbook.add_format({'bold': True, 'bottom': 1, 'bg_color': '#F0F0F0'})
            
            # Tier Colors
            s_tier_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'}) # Green
            a_tier_fmt = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'}) # Yellow
            b_tier_fmt = workbook.add_format({'bg_color': '#E0E0E0', 'font_color': '#333333'}) # Grey
            c_tier_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'}) # Red

            # Write Headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

            # Auto-fit Columns
            for i, col in enumerate(df.columns):
                # Calculate width based on max length of data + header
                # Limit sample to 50 rows for speed optimization
                sample_values = df[col].astype(str).head(50)
                max_len = max(sample_values.map(len).max(), len(str(col)))
                width = min(max_len + 4, 40)
                
                if col == 'Prob':
                    worksheet.set_column(i, i, width, pct_fmt)
                else:
                    worksheet.set_column(i, i, width)

            # Conditional Formatting for Tiers
            tier_col_idx = df.columns.get_loc('Tier') if 'Tier' in df.columns else -1
            if tier_col_idx != -1:
                # Apply format to the whole column (rows 1 to N)
                rng = f"{xlsxwriter.utility.xl_col_to_name(tier_col_idx)}2:{xlsxwriter.utility.xl_col_to_name(tier_col_idx)}{len(df)+1}"
                
                worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'S Tier', 'format': s_tier_fmt})
                worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'A Tier', 'format': a_tier_fmt})
                worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'B Tier', 'format': b_tier_fmt})
                worksheet.conditional_format(rng, {'type': 'text', 'criteria': 'containing', 'value': 'C Tier', 'format': c_tier_fmt})

            writer.close()
        else:
            df.to_excel(output_path, index=False)

        logging.info(f"Saved Excel analysis to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")

def print_tier_summary(df):
    """Logs a summary of findings."""
    if 'Tier' not in df.columns: return
    
    counts = df['Tier'].value_counts()
    logging.info("--- ANALYSIS SUMMARY ---")
    for tier in ['S Tier', 'A Tier', 'B Tier', 'C Tier']:
        count = counts.get(tier, 0)
        logging.info(f"  {tier}: {count} props")
    logging.info("------------------------")

def print_pretty_table(df, title="TOP 15 DISCOVERED EDGES"):
    if df.empty:
        print("No results to display.")
        return

    # Convert to string for display
    df_str = df.astype(str)
    
    widths = []
    for col in df.columns:
        max_len = max(df_str[col].map(len).max(), len(col))
        widths.append(max_len + 2)

    fmt = "| " + " | ".join([f"{{:<{w}}}" for w in widths]) + " |"
    total_width = sum(widths) + (3 * len(widths)) + 1
    sep_line = "=" * total_width

    try:
        print(f"\n{title}")
        print(sep_line)
        print(fmt.format(*df.columns))
        print("-" * total_width)

        for _, row in df.iterrows():
            print(fmt.format(*row.values))

        print(sep_line + "\n")
    except Exception:
        print(df.head(15))

def main():
    common.setup_logging(name="analysis_pregame")
    logging.info(">>> STARTING PRE-GAME PROP ANALYSIS <<<")
    
    try:
        # 1. Load Props
        props_path = cfg.PROPS_FILE
        if not props_path.exists():
            logging.critical(f"Props file not found: {props_path}")
            logging.critical("Please put 'props_today.csv' in the input folder.")
            return

        try:
            props_df = pd.read_csv(props_path)
            props_df.columns = props_df.columns.str.strip()
            
            # --- Robust Date Parsing ---
            # Ensure we have valid dates for the time-travel feature generation
            if Cols.DATE in props_df.columns:
                props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE], errors='coerce')
                # Fill missing dates with today (assuming input is for today's games)
                if props_df[Cols.DATE].isna().any():
                    today = pd.Timestamp.now().normalize()
                    props_df[Cols.DATE] = props_df[Cols.DATE].fillna(today)
            else:
                logging.warning(f"'{Cols.DATE}' column missing. Assuming today's date.")
                props_df[Cols.DATE] = pd.Timestamp.now().normalize()
                
            logging.info(f"Loaded {len(props_df)} props.")

            # =========================================================================
            # NEW: AUTO-SAVE TO HISTORY (Learning Loop)
            # =========================================================================
            try:
                history_path = cfg.MASTER_PROP_HISTORY_FILE
                
                # Create a clean copy for storage
                history_entry = props_df.copy()
                
                # Ensure consistent string types for key columns to prevent merge errors
                if Cols.PLAYER_NAME in history_entry.columns:
                    history_entry[Cols.PLAYER_NAME] = history_entry[Cols.PLAYER_NAME].astype(str)
                if Cols.PROP_TYPE in history_entry.columns:
                    history_entry[Cols.PROP_TYPE] = history_entry[Cols.PROP_TYPE].astype(str)
                
                if history_path.exists():
                    existing_hist = pd.read_parquet(history_path)
                    combined_hist = pd.concat([existing_hist, history_entry], ignore_index=True)
                    
                    # Deduplicate: Keep the LATEST entry for a specific player/date/prop
                    # This allows you to update lines during the day and keep the final closing line
                    dedup_cols = [c for c in [Cols.PLAYER_NAME, Cols.DATE, Cols.PROP_TYPE] if c in combined_hist.columns]
                    if dedup_cols:
                        combined_hist.drop_duplicates(subset=dedup_cols, keep='last', inplace=True)
                    
                    combined_hist.to_parquet(history_path, index=False)
                    logging.info(f"Updated Prop History. Total records: {len(combined_hist)}")
                else:
                    history_entry.to_parquet(history_path, index=False)
                    logging.info(f"Created new Prop History file at {history_path}")
                    
            except Exception as e:
                # Don't crash the analysis if history saving fails, just warn
                logging.warning(f"Failed to save prop history (Learning loop will be unaffected for this run): {e}")
            # =========================================================================
            
        except Exception as e:
            logging.critical(f"Failed to read props file: {e}")
            return

        # 2. Build Features
        features_df = generator.build_feature_set(props_df)
        if features_df.empty: 
            logging.error("Feature generation returned empty dataframe.")
            return

        # 3. Run Inference
        logging.info("Running Machine Learning Inference...")
        results_df = inference.predict_props(features_df)
        
        if results_df is None or results_df.empty:
            logging.warning("No predictions were generated. Check model artifacts or input data.")
            return

        # 4. Sorting & Ranking
        if '_Sort_Diff' not in results_df.columns: results_df['_Sort_Diff'] = 0.0
        if 'Tier' not in results_df.columns: results_df['Tier'] = 'C Tier'
        
        # S Tier = 0, A Tier = 1, etc.
        tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
        results_df['Tier_Rank'] = results_df['Tier'].map(tier_map).fillna(99)
        
        # Sort by Tier (asc) then by Edge Size (desc)
        results_df.sort_values(by=['Tier_Rank', '_Sort_Diff'], ascending=[True, False], inplace=True)
        
        # Log Summary before formatting destroys numeric types
        print_tier_summary(results_df)

        # 5. Format Output
        # Clean Date String for display
        if Cols.DATE in results_df.columns:
            results_df[Cols.DATE] = pd.to_datetime(results_df[Cols.DATE]).dt.strftime('%Y-%m-%d')

        # Rename to Final Output Columns
        rename_map = {
            Cols.PLAYER_NAME: 'Player',
            Cols.PROP_TYPE: 'Prop',
            Cols.PROP_LINE: 'Line',
            Cols.DATE: 'Date',
        }
        results_df.rename(columns=rename_map, inplace=True)

        # Select Columns (Strictly keeping user preferred format)
        keep_cols = [
            'Player', 'Team', 'Opponent', 'Prop', 'Line', 
            'Proj', 'Prob', 'Pick', 'Tier', 
            'Date'
        ]
        
        final_cols = [c for c in keep_cols if c in results_df.columns]
        final_output = results_df[final_cols].copy()

        # 6. Save Files
        # Save Parquet (System)
        final_output.to_parquet(cfg.PROCESSED_OUTPUT_SYSTEM, index=False)
        logging.info(f"Saved system results to {cfg.PROCESSED_OUTPUT_SYSTEM}")
        
        # Save Excel (User - Pretty)
        save_pretty_excel(final_output, cfg.PROCESSED_OUTPUT_XLSX)
        
        # 7. Console Display
        console_output = final_output.copy()
        if 'Prob' in console_output.columns:
            # Format Prob as % string for console only
            if pd.api.types.is_numeric_dtype(console_output['Prob']):
                console_output['Prob'] = console_output['Prob'].apply(lambda x: f"{x*100:.1f}%")
            
        print_pretty_table(console_output.head(15))

        logging.info("<<< ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()