# --- FEATURE GROUPS ---

# Core Base Features (Advanced ETL Output)
BASE_FEATURE_COLS = [
    # General Averages
    'SZN_AVG', 'L5_AVG', 'L10_AVG', 'L20_AVG',
    
    # Form vs Baseline
    'FORM_RATIO', 
    
    # Volatility & Distribution Metrics (Crucial for EV Math)
    'L10_STD_DEV', 'L10_CV',
    'L10_HitRate_10', 'L10_HitRate_15', 'L10_HitRate_20', 'L10_HitRate_25', 'L10_HitRate_30',
    
    # Contextual Splits
    'HOME_AWAY_AVG', 'REST_SPLIT_AVG', 'IS_HOME', 'Days_Rest',
    
    # Rate & Advanced
    'USG_PROXY_PER36', 'TS_PCT', 'USG_PROXY', 
    'L5_USG_PROXY', 'SZN_USG_PROXY',
    
    # Opponent & Game Context (Pace Scaling)
    'DVP_MULTIPLIER', 'OPP_DEF_EFF',
    'GAME_PACE', 'OPP_GAME_PACE',
    
    # Vacancy (Missing Usage/Minutes on Team)
    'TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 
    'MISSING_USG_G', 'MISSING_USG_F', 'MISSING_USG_C'
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
    'PTS': ['PTS', 'PRA', 'PR', 'PA', 'USG_PROXY', 'TS_PCT', 'GAME_PACE', 'OPP_GAME_PACE'],
    'REB': ['REB', 'PRA', 'PR', 'RA', 'GAME_PACE', 'OPP_GAME_PACE'],
    'AST': ['AST', 'PRA', 'PA', 'RA', 'GAME_PACE', 'OPP_GAME_PACE'],
    'PRA': ['PRA', 'PTS', 'REB', 'AST', 'PR', 'PA', 'RA', 'GAME_PACE', 'OPP_GAME_PACE'],
    'PR':  ['PR', 'PTS', 'REB', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE'],
    'PA':  ['PA', 'PTS', 'AST', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE'],
    'RA':  ['RA', 'REB', 'AST', 'PRA', 'GAME_PACE', 'OPP_GAME_PACE'],
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