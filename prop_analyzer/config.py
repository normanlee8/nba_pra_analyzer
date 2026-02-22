from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GRADED_DIR = OUTPUT_DIR / "graded_history"

# Ensure key directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(INPUT_DIR / "records").mkdir(parents=True, exist_ok=True)
GRADED_DIR.mkdir(parents=True, exist_ok=True)

# Specific File Paths
INPUT_PROPS_TXT = INPUT_DIR / "props_input.txt"
PROPS_FILE = INPUT_DIR / "props_today.csv"

# Final results
PROCESSED_OUTPUT_SYSTEM = OUTPUT_DIR / "processed_props.parquet" 
PROCESSED_OUTPUT_XLSX = OUTPUT_DIR / "processed_props.xlsx"

# --- MASTER DATA FILES (ALL PARQUET) ---
MASTER_PLAYER_FILE = DATA_DIR / "master_player_stats_2025-26.parquet"
MASTER_PLAYER_PATTERN = "master_player_stats_*.parquet"

MASTER_TEAM_FILE = DATA_DIR / "master_team_stats_2025-26.parquet"
MASTER_TEAM_PATTERN = "master_team_stats_*.parquet"

MASTER_BOX_SCORES_FILE = DATA_DIR / "master_box_scores_2025-26.parquet"
MASTER_BOX_SCORES_PATTERN = "master_box_scores_*.parquet"

# NEW: Master Prop History (Real Vegas Lines)
MASTER_PROP_HISTORY_FILE = DATA_DIR / "master_prop_history.parquet"

MASTER_VS_OPP_FILE = DATA_DIR / "master_vs_opponent.parquet"
MASTER_DVP_FILE = DATA_DIR / "master_dvp_stats.parquet"
MASTER_TRAINING_FILE = DATA_DIR / "master_training_dataset.parquet"

# --- DATA CONTRACT (SCHEMA) ---
class Cols:
    PLAYER_NAME = 'Player Name'
    PLAYER_ID = 'PLAYER_ID'
    GAME_ID = 'GAME_ID'  
    TEAM = 'Team'
    OPPONENT = 'Opponent'
    MATCHUP = 'Matchup'
    DATE = 'GAME_DATE'
    
    PROP_TYPE = 'Prop Category'
    PROP_LINE = 'Prop Line'
    
    PREDICTION = 'Model_Pred'
    CONFIDENCE = 'Model_Conf'
    EDGE_TYPE = 'Edge_Type'
    TIER = 'Tier'
    
    ACTUAL_VAL = 'Actual Value'
    RESULT = 'Result'
    CORRECTNESS = 'Correctness'
    
    SZN_AVG = 'SZN_AVG'
    L5_AVG = 'L5_AVG'
    
    @classmethod
    def get_required_input_cols(cls):
        return [cls.PLAYER_NAME, cls.TEAM, cls.OPPONENT, cls.MATCHUP, cls.PROP_TYPE, cls.PROP_LINE, cls.DATE]

# --- THRESHOLDS ---
MIN_PROB_FOR_S_TIER = 0.59  
MIN_EDGE_FOR_S_TIER = 1.5
MIN_EDGE_FOR_A_TIER = 1.0
LIVE_MIN_PROB_THRESHOLD = 0.65
LIVE_BLOWOUT_THRESHOLD = 20
BAYESIAN_PRIOR_WEIGHT = 6.0  
EWMA_DECAY_FACTOR = 0.80     
MIN_GAMES_FOR_ANALYSIS = 5

# --- PRIORS ---
BAYESIAN_PRIORS = {
    'PTS': 12.0, 'REB': 4.0, 'AST': 3.0, 'PRA': 18.0,
    'PR': 16.0, 'PA': 15.0, 'RA': 7.0
}

# --- PROP MAPPING ---
MASTER_PROP_MAP = {
    # Core
    'Points': 'PTS', 'pts': 'PTS',
    'Rebounds': 'REB', 'reb': 'REB',
    'Assists': 'AST', 'ast': 'AST',
    
    # Combos
    'Pts + Rebs + Asts': 'PRA', 'Pts+Rebs+Asts': 'PRA', 'pra': 'PRA',
    'Rebounds + Assists': 'RA', 'ra': 'RA',
    'Points + Rebounds': 'PR', 'pr': 'PR',
    'Points + Assists': 'PA', 'pa': 'PA'
}

SUPPORTED_PROPS = [
    'PTS', 'REB', 'AST',
    'PRA', 'PR', 'PA', 'RA'
]