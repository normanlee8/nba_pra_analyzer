# scripts/run_analysis.py

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
from prop_analyzer.models.parlay_optimizer import ParlayOptimizer
from prop_analyzer.utils import common

def save_pretty_excel(df, output_path, sheet_name='EV_Picks'):
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
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
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
                
                if col in ['Prob', 'Win Prob', 'Joint Prob']: worksheet.set_column(i, i, width, pct_fmt)
                elif col in ['EV%', 'Kelly', 'Ticket EV%']: worksheet.set_column(i, i, width, kelly_fmt)
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
        print(f"\nNo results to display for {title}.")
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

def print_stacked_parlays(df, title):
    """
    Prints parlays in a vertical, stacked bullet-point format to prevent 
    console line-wrapping on high leg-count tickets.
    """
    if df.empty: return
    
    print(f"\n{title}")
    print("=" * 65)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        prob = row['Joint Prob']
        ev = row['Ticket EV%']
        payout = row['Payout']
        
        # Print the Ticket Header
        print(f" OPTION {i}  |  Win Prob: {prob}  |  EV: {ev}  |  Payout: {payout}")
        
        # Split the string of picks and print each leg vertically
        picks = str(row['Picks']).split(' | ')
        for pick in picks:
            print(f"   [+] {pick}")
            
        print("-" * 65)

def format_parlays_for_output(parlays):
    """Converts the raw parlay dicts into a printable/saveable DataFrame"""
    rows = []
    for p in parlays:
        legs_str = " | ".join([f"{leg['player_name']} {leg['pick']} {leg['line']} {leg['stat_type']}" for leg in p['ticket']])
        rows.append({
            'Legs': p['legs'],
            'Joint Prob': p['joint_prob'],
            'Ticket EV%': p['ev'],
            'Payout': f"{p['payout_multiplier']}x",
            'Picks': legs_str
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort strictly by Highest Probability of Winning
        df = df.sort_values(by=['Joint Prob', 'Ticket EV%'], ascending=[False, False])
    return df

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

        # Removed the Cannibalization Filter here so all props remain in the DataFrame

        print_tier_summary(results_df)

        # --- PARLAY OPTIMIZER INTEGRATION ---
        logging.info("Initializing Underdog Parlay Optimizer...")
        try:
            hist_df = pd.read_parquet(cfg.MASTER_BOX_SCORES_FILE) 
            optimizer = ParlayOptimizer(historical_data=hist_df)
            
            # Map inference results to optimizer schema
            daily_props_for_parlays = []
            for _, row in results_df.iterrows():
                # We need raw win probability, not the string format.
                prob_val = row['Prob'] 
                
                # Construct pseudo-game_id to ensure games are isolated correctly
                teams = sorted([str(row['Team']), str(row['Opponent'])])
                game_id = f"{row['Date']}_{teams[0]}_{teams[1]}"
                
                daily_props_for_parlays.append({
                    'player_name': row['Player'],
                    'team': row['Team'],
                    'opponent': row['Opponent'],
                    'game_id': game_id,
                    'position': row.get('Position', 'UNK'),
                    'stat_type': row['Prop'],
                    'win_prob': prob_val,
                    'pick': row['Pick'],
                    'line': row['Line'],
                    'Tier': row['Tier'] 
                })
                
            # Run Combinatorial Search for 2 to 8 legs
            top_parlays = optimizer.optimize_parlays(daily_props_for_parlays, min_legs=2, max_legs=8, top_n=10)
            parlays_df = format_parlays_for_output(top_parlays)
            
        except Exception as e:
            logging.error(f"Parlay Optimizer failed: {e}", exc_info=True)
            parlays_df = pd.DataFrame()


        # --- FINAL CONSOLE AND FILE OUTPUTS ---
        keep_cols = ['Player', 'Team', 'Opponent', 'Prop', 'Line', 'Proj', 'Prob', 'Pick', 'EV%', 'Kelly', 'Tier', 'Date']
        final_cols = [c for c in keep_cols if c in results_df.columns]
        final_output = results_df[final_cols].copy()

        final_output.to_parquet(cfg.PROCESSED_OUTPUT_SYSTEM, index=False)
        save_pretty_excel(final_output, cfg.PROCESSED_OUTPUT_XLSX, sheet_name='Straight_Picks')
        
        # Save parlays if generated
        if not parlays_df.empty:
            parlay_output_path = cfg.OUTPUT_DIR / "EV_Parlays.csv"
            parlays_df.to_csv(parlay_output_path, index=False)
            logging.info(f"Saved {len(parlays_df)} parlay combinations to {parlay_output_path}")

        # Format Straight Picks for Console
        console_output = final_output.copy()
        if 'Prob' in console_output.columns:
            console_output['Prob'] = console_output['Prob'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else x)
        if 'EV%' in console_output.columns:
            console_output['EV%'] = console_output['EV%'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else x)
        if 'Kelly' in console_output.columns:
            console_output['Kelly'] = console_output['Kelly'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else x)
            
        print_pretty_table(console_output.head(15), title="TOP 15 STRAIGHT +EV PICKS")
        
        # Format Parlays for Console (Stacked List Format)
        if not parlays_df.empty:
            console_parlays = parlays_df.copy()
            console_parlays['Ticket EV%'] = console_parlays['Ticket EV%'].apply(lambda x: f"{x*100:.1f}%")
            console_parlays['Joint Prob'] = console_parlays['Joint Prob'].apply(lambda x: f"{x*100:.2f}%")
            
            print("\n" + "="*80)
            print(" TOP UNDERDOG PARLAYS BY WIN PROBABILITY (PER LEG COUNT)")
            print("="*80)
            
            for leg_count in range(2, 9):
                leg_df = console_parlays[console_parlays['Legs'] == leg_count]
                if not leg_df.empty:
                    print_stacked_parlays(leg_df.head(3), title=f"TOP {leg_count}-LEG PARLAYS")
            
        logging.info("<<< EV ANALYSIS COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Analysis Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()