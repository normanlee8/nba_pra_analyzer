# --- FEATURE GROUPS ---

# Features that should ALWAYS be included if available
BASE_FEATURE_COLS = [
    # Trend & Average
    'SZN Avg', 'L3 Avg', 'L5 EWMA', 'Loc Avg', 'CoV %',
    'SZN Games', 'Selected Std Dev', 'L10_STD_DEV',
    
    # Context
    'Prop Line',
    
    # Fatigue / Schedule
    'GAMES_IN_L5', 'IS_B2B', 'Days Rest',
    
    # Vacancy (The new split logic)
    'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 
    'MISSING_USG_G', 'MISSING_USG_F',
    
    # Advanced
    'SZN_TS_PCT', 'SZN_EFG_PCT', 'SZN_USG_PROXY',
    'L5_TS_PCT', 'L5_EFG_PCT', 'L5_USG_PROXY',
    'LOC_TS_PCT', 'LOC_EFG_PCT', 'LOC_USG_PROXY',
    'TS_DIFF', 'EFG_DIFF', 'USG_DIFF',
    
    # PBP / OnCourt
    'PBP_OnCourt_PlusMinus', 'PBP_OnCourt_USG_PCT'
]

# Features that are specific to Matchup History (Prefix: VS_OPP_)
VS_OPP_FEATURES = [
    'VS_OPP_PTS', 'VS_OPP_REB', 'VS_OPP_AST', 
    'VS_OPP_PRA', 'VS_OPP_PR', 'VS_OPP_PA', 'VS_OPP_RA', 
    'VS_OPP_MIN', 'VS_OPP_GAMES_PLAYED'
]

# Features regarding historical averages (Prefix: HIST_)
HIST_FEATURES = [
    'HIST_VS_OPP_PTS_AVG', 'HIST_VS_OPP_REB_AVG', 'HIST_VS_OPP_AST_AVG',
    'HIST_VS_OPP_PRA_AVG', 'HIST_VS_OPP_PR_AVG', 'HIST_VS_OPP_PA_AVG',
    'HIST_VS_OPP_RA_AVG', 'HIST_VS_OPP_GAMES'
]

# --- MAPPINGS ---

# For Feature Pruning: Defines which VS_OPP and HIST stats are relevant 
# for a specific target prop. 
PROP_FEATURE_MAP = {
    # Full Game
    'PTS': ['PTS', 'PRA', 'PR', 'PA', 'USG_PROXY', 'TS_PCT'],
    'REB': ['REB', 'PRA', 'PR', 'RA'],
    'AST': ['AST', 'PRA', 'PA', 'RA'],
    'PRA': ['PRA', 'PTS', 'REB', 'AST', 'PR', 'PA', 'RA'],
    'PR':  ['PR', 'PTS', 'REB', 'PRA'],
    'PA':  ['PA', 'PTS', 'AST', 'PRA'],
    'RA':  ['RA', 'REB', 'AST', 'PRA'],
}

# Keywords to look for in "Team Stats" columns (e.g. Rank columns)
RELEVANT_KEYWORDS = {
    'PTS': ['Points', 'PTS', 'Offensive Efficiency', 'True Shooting'],
    'REB': ['Rebound', 'REB', 'Opponent Total Rebounds'],
    'AST': ['Assist', 'AST'],
    'PRA': ['Points', 'Rebound', 'Assist', 'Offensive Efficiency'],
    'PR': ['Points', 'Rebound'],
    'PA': ['Points', 'Assist'],
    'RA': ['Rebound', 'Assist'],
}