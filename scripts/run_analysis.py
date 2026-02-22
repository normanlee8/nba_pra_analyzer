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

def filter_cannibalized_props(df):
    """
    Prevents over-exposure to a single player by filtering out redundant Combo props.
    If a player has multiple +EV Over props (e.g., PTS and PRA), it keeps only the highest EV play.
    """
    if df.empty or 'EV%' not in df.columns: return df
    
    filtered_rows = []
    
    # Group by Player, Date, and Pick Direction (Over/Under)
    grouped = df.groupby(['Player', 'Date', 'Pick'])
    
    for _, group in grouped:
        # Sort by Expected Value descending
        sorted_group = group.sort_values(by='EV%', ascending=False)
        # Keep only the highest EV prop for this player/direction combination
        filtered_rows.append(sorted_group.iloc[0:1])
        
    if not filtered_rows: return df
    
    filtered_df = pd.concat(filtered_rows, ignore_index=True)
    # Re-sort to maintain overall tier hierarchy
    filtered_df.sort_values(by=['Tier_Rank', '_Sort_Diff'], ascending=[True, False], inplace=True)
    return filtered_df

def save_pretty_excel(df, output_path):
    try:
        if df.empty: return

        has_xlsxwriter = False
        try:
            import xlsxwriter
            has_xlsxwriter = True
        except ImportError:
            pass

        if has_xlsxwriter:
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='EV_Picks', index=False)
            workbook = writer.book
            worksheet = writer.sheets['EV_Picks']
            
            pct_fmt = workbook.add_format({'num_format': '0.0%'})
            kelly_fmt = workbook.add_format({'num_format': '0.00%'})
            header_fmt = workbook.add_format({'bold': True, 'bottom': 1, 'bg_color': '#F0F0F0'})
            
            s_tier_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            a_tier_fmt = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
            b_tier_fmt = workbook.add_format({'bg_color': '#E0E0E0', 'font_color': '#333333'})
            c_tier_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)

            for i, col in enumerate(df.columns):
                sample_values = df[col].astype(str).head(50)
                max_len = max(sample_values.map(len).max(), len(str(col)))
                width = min(max_len + 4, 40)
                
                if col in ['Prob']: worksheet.set_column(i, i, width, pct_fmt)
                elif col in ['EV%', 'Kelly']: worksheet.set_column(i, i, width, kelly_fmt)
                else: worksheet.set_column(i, i, width)

            tier_col_idx = df.columns.get_loc('Tier') if 'Tier' in df.columns else -1
            if tier_col_idx != -1:
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
    if 'Tier' not in df.columns: return
    counts = df['Tier'].value_counts()
    logging.info("--- EV ANALYSIS SUMMARY ---")
    for tier in ['S Tier', 'A Tier', 'B Tier', 'C Tier', 'Pass']:
        count = counts.get(tier, 0)
        logging.info(f"  {tier}: {count} props")
    logging.info("---------------------------")

def print_pretty_table(df, title="TOP 15 +EV DISCOVERIES"):
    if df.empty:
        print("No results to display.")
        return

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
    logging.info(">>> STARTING PRE-GAME EV PROP ANALYSIS <<<")
    
    try:
        props_path = cfg.PROPS_FILE
        if not props_path.exists():
            logging.critical(f"Props file not found: {props_path}")
            return

        props_df = pd.read_csv(props_path)
        props_df.columns = props_df.columns.str.strip()
        
        if Cols.DATE in props_df.columns:
            props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE], errors='coerce')
            if props_df[Cols.DATE].isna().any():
                props_df[Cols.DATE] = props_df[Cols.DATE].fillna(pd.Timestamp.now().normalize())
        else:
            props_df[Cols.DATE] = pd.Timestamp.now().normalize()
            
        logging.info(f"Loaded {len(props_df)} props.")

        features_df = generator.build_feature_set(props_df)
        if features_df.empty: 
            logging.error("Feature generation returned empty dataframe.")
            return

        logging.info("Running Probabilistic EV Inference...")
        results_df = inference.predict_props(features_df)
        
        if results_df is None or results_df.empty:
            logging.warning("No EV predictions generated.")
            return

        if '_Sort_Diff' not in results_df.columns: results_df['_Sort_Diff'] = 0.0
        if 'Tier' not in results_df.columns: results_df['Tier'] = 'Pass'
        
        tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3, 'Pass': 4}
        results_df['Tier_Rank'] = results_df['Tier'].map(tier_map).fillna(99)
        results_df.sort_values(by=['Tier_Rank', '_Sort_Diff'], ascending=[True, False], inplace=True)
        
        # Format Date and Rename Cols
        if Cols.DATE in results_df.columns:
            results_df[Cols.DATE] = pd.to_datetime(results_df[Cols.DATE]).dt.strftime('%Y-%m-%d')

        rename_map = {Cols.PLAYER_NAME: 'Player', Cols.PROP_TYPE: 'Prop', Cols.PROP_LINE: 'Line', Cols.DATE: 'Date'}
        results_df.rename(columns=rename_map, inplace=True)

        # Apply Cannibalization Filter to isolate best risk
        results_df = filter_cannibalized_props(results_df)

        print_tier_summary(results_df)

        keep_cols = ['Player', 'Team', 'Opponent', 'Prop', 'Line', 'Proj', 'Prob', 'Pick', 'EV%', 'Kelly', 'Tier', 'Date']
        final_cols = [c for c in keep_cols if c in results_df.columns]
        final_output = results_df[final_cols].copy()

        final_output.to_parquet(cfg.PROCESSED_OUTPUT_SYSTEM, index=False)
        save_pretty_excel(final_output, cfg.PROCESSED_OUTPUT_XLSX)
        
        console_output = final_output.copy()
        if 'Prob' in console_output.columns:
            console_output['Prob'] = console_output['Prob'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else x)
        if 'EV%' in console_output.columns:
            console_output['EV%'] = console_output['EV%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else x)
        if 'Kelly' in console_output.columns:
            console_output['Kelly'] = console_output['Kelly'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else x)
            
        print_pretty_table(console_output.head(15))
        logging.info("<<< EV ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()