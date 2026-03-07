# --- FEATURE GROUPS ---

# Core Base Features (Advanced ETL Output)
BASE_FEATURE_COLS = [
    # General Averages
    'SZN_AVG', 'L5_AVG', 'L10_AVG', 'L20_AVG',
    
    # Form vs Baseline
    'FORM_RATIO', 
    
    # Model Residual Features (NEW)
    'PLAYER_HISTORIC_MODEL_BIAS', 'PLAYER_HISTORIC_MODEL_MAE',
    
    # Volatility & Distribution Metrics (Crucial for EV Math)
    'L10_STD_DEV', 'L10_CV',
    'L10_HitRate_10', 'L10_HitRate_15', 'L10_HitRate_20', 'L10_HitRate_25', 'L10_HitRate_30',
    'VS_OPP_HIT_RATE', 'VS_OPP_GAMES_COUNT', 
    
    # Contextual Splits
    'REST_SPLIT_AVG', 'IS_HOME', 'Days_Rest',
    
    # Rate & Advanced
    'USG_PROXY_PER36', 'TS_PCT', 'USG_PROXY', 
    'L5_USG_PROXY', 'SZN_USG_PROXY',
    
    # Opponent & Game Context (Pace Scaling)
    'DVP_PTS_MULTIPLIER', 'DVP_REB_MULTIPLIER', 'DVP_AST_MULTIPLIER', 
    'DVP_PRA_MULTIPLIER', 'DVP_PR_MULTIPLIER', 'DVP_PA_MULTIPLIER', 'DVP_RA_MULTIPLIER',
    'OPP_DEF_EFF', 'GAME_PACE', 'OPP_GAME_PACE', 'OPP_DAYS_REST', 'OPP_IS_B2B',
    
    # ---> NEW: Schedule Density & Travel Fatigue <---
    'FLIGHT_MILES', 'TZ_SHIFT', 'TEAM_GAMES_L4', 'TEAM_GAMES_L6', 'TEAM_GAMES_L7', 
    'IS_3_IN_4', 'IS_4_IN_6', 'IS_TZ_SHOCK',
    
    # Vacancy (Missing Usage/Minutes/Rates on Team)
    'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 
    'MISSING_USG_G', 'MISSING_USG_F', 'MISSING_USG_C',
    'TEAM_MISSING_AST_PCT', 'TEAM_MISSING_REB_PCT',

    # --- ADVANCED SCRAPED STATS ---
    # Rebounding Context
    'OPP_Opponent Effective Field Goal %', 'OPP_Opponent True Shooting %',
    'TEAM_Field Goals Attempted per Game', 'OPP_Field Goals Attempted per Game',
    'TEAM_Three Pointers Attempted per Game', 'OPP_Three Pointers Attempted per Game',
    'OPP_Opponent Offensive Rebounding %', 
    'TEAM_Total Rebounds per Game', 'OPP_Opponent Total Rebounds per Game', 
    
    # Assist Context
    'TEAM_Assists per FGM', 'OPP_Opponent Assists per FGM', 'TEAM_Assist to Turnover Ratio',
    
    # Scoring Context
    'OPP_Opponent Points in Paint per Game', 'OPP_Opponent Percent of Points from 3 Pointers',
    'OPP_Opponent Personal Fouls per Game', 'OPP_Opponent Fastbreak Points per Game',
    'TEAM_Points per Game', 'OPP_Opponent Points per Game', 
    
    # Combo & Volume Context
    'TEAM_Extra Scoring Chances per Game', 'OPP_Extra Scoring Chances per Game',
    'OPP_Opponent Points + Rebounds + Assists per Game', 'OPP_Opponent Points + Assists per Game',

    # Tactical & Scheme Matchup (Shot Location Engine)
    'FREQ_PAINT', 'FREQ_3PT',
    'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE', 'SCHEME_SYNERGY_SCORE'
]

VS_OPP_FEATURES = [
    'VS_OPP_PTS', 'VS_OPP_REB', 'VS_OPP_AST', 
    'VS_OPP_PRA', 'VS_OPP_PR', 'VS_OPP_PA', 'VS_OPP_RA', 
    'VS_OPP_MIN', 'VS_OPP_GAMES_PLAYED'
]

HIST_FEATURES = [
    'HIST_VS_OPP_PTS_AVG', 'HIST_VS_OPP_REB_AVG', 'HIST_VS_OPP_AST_AVG',
    'HIST_VS_OPP_PRA_AVG', 'HIST_VS_OPP_PR_AVG', 'HIST_VS_OPP_PA_AVG',
    'HIST_VS_OPP_RA_AVG', 'HIST_VS_OPP_GAMES'
]

# --- MAPPINGS ---
PROP_FEATURE_MAP = {
    'PTS': ['PTS', 'PRA', 'PR', 'PA', 'USG_PROXY', 'TS_PCT', 'GAME_PACE', 'OPP_GAME_PACE',
            'PTS_SPLIT_AVG', 'PTS_DIFF', 'MIN_SPLIT_AVG',
            'OPP_Opponent Points in Paint per Game', 'OPP_Opponent Percent of Points from 3 Pointers',
            'OPP_Opponent Personal Fouls per Game', 'OPP_Opponent Fastbreak Points per Game',
            'TEAM_Extra Scoring Chances per Game', 'OPP_Extra Scoring Chances per Game',
            'TEAM_Points per Game', 'OPP_Opponent Points per Game',
            'SCHEME_SYNERGY_SCORE', 'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE'], 
            
    'REB': ['REB', 'PRA', 'PR', 'RA', 'GAME_PACE', 'OPP_GAME_PACE',
            'REB_SPLIT_AVG', 'REB_DIFF', 'MIN_SPLIT_AVG',
            'OPP_Opponent Effective Field Goal %', 'OPP_Opponent True Shooting %',
            'TEAM_Field Goals Attempted per Game', 'OPP_Field Goals Attempted per Game',
            'TEAM_Three Pointers Attempted per Game', 'OPP_Three Pointers Attempted per Game',
            'OPP_Opponent Offensive Rebounding %',
            'TEAM_Total Rebounds per Game', 'OPP_Opponent Total Rebounds per Game',
            'FREQ_PAINT'], 
            
    'AST': ['AST', 'PRA', 'PA', 'RA', 'GAME_PACE', 'OPP_GAME_PACE',
            'AST_SPLIT_AVG', 'AST_DIFF', 'MIN_SPLIT_AVG',
            'TEAM_Assists per FGM', 'OPP_Opponent Assists per FGM', 'TEAM_Assist to Turnover Ratio',
            'TEAM_Points per Game', 'OPP_Opponent Points per Game',
            'FREQ_PAINT'], 
            
    'PRA': ['PRA', 'PTS', 'REB', 'AST', 'PR', 'PA', 'RA', 'GAME_PACE', 'OPP_GAME_PACE',
            'PRA_SPLIT_AVG', 'PRA_DIFF', 'MIN_SPLIT_AVG',
            'TEAM_Extra Scoring Chances per Game', 'OPP_Extra Scoring Chances per Game',
            'OPP_Opponent Points + Rebounds + Assists per Game',
            'TEAM_Points per Game', 'OPP_Opponent Points per Game',
            'TEAM_Total Rebounds per Game', 'OPP_Opponent Total Rebounds per Game',
            'SCHEME_SYNERGY_SCORE', 'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE'],
            
    'PR':  ['PR', 'PTS', 'REB', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE',
            'PR_SPLIT_AVG', 'PR_DIFF', 'MIN_SPLIT_AVG',
            'OPP_Opponent Points in Paint per Game', 'OPP_Opponent Effective Field Goal %',
            'TEAM_Extra Scoring Chances per Game',
            'TEAM_Points per Game', 'OPP_Opponent Points per Game',
            'TEAM_Total Rebounds per Game', 'OPP_Opponent Total Rebounds per Game',
            'SCHEME_SYNERGY_SCORE', 'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE'],
            
    'PA':  ['PA', 'PTS', 'AST', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE',
            'PA_SPLIT_AVG', 'PA_DIFF', 'MIN_SPLIT_AVG',
            'OPP_Opponent Points + Assists per Game', 'OPP_Opponent Assists per FGM',
            'TEAM_Extra Scoring Chances per Game',
            'TEAM_Points per Game', 'OPP_Opponent Points per Game',
            'SCHEME_SYNERGY_SCORE', 'SYNERGY_PAINT_EDGE', 'SYNERGY_3PT_EDGE'],
            
    'RA':  ['RA', 'REB', 'AST', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE',
            'RA_SPLIT_AVG', 'RA_DIFF', 'MIN_SPLIT_AVG',
            'OPP_Opponent Effective Field Goal %', 'OPP_Opponent Assists per FGM',
            'TEAM_Total Rebounds per Game', 'OPP_Opponent Total Rebounds per Game',
            'FREQ_PAINT'],
}

RELEVANT_KEYWORDS = {
    'PTS': ['Points', 'PTS', 'Offensive Efficiency', 'True Shooting'],
    'REB': ['Rebound', 'REB', 'Opponent Total Rebounds'],
    'AST': ['Assist', 'AST'],
    'PRA': ['Points', 'Rebound', 'Assist', 'Offensive Efficiency'],
    'PR':  ['Points', 'Rebound'],
    'PA':  ['Points', 'Assist'],
    'RA':  ['Rebound', 'Assist'],
}