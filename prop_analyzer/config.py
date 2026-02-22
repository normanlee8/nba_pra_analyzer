from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GRADED_DIR = OUTPUT_DIR / "graded_history"

# Advanced Modeling Directories
MODEL_VERSIONS_DIR = MODEL_DIR / "versions"
MODEL_METADATA_DIR = MODEL_DIR / "metadata"

# Ensure key directories exist
for d in [DATA_DIR, MODEL_DIR, INPUT_DIR, OUTPUT_DIR, GRADED_DIR, 
          INPUT_DIR / "records", MODEL_VERSIONS_DIR, MODEL_METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

# --- ADVANCED TRAINING CONFIGURATION ---
CV_TIME_SPLITS = 5
OPTUNA_N_TRIALS_XGB = 20  # Set to 50+ for production runs
OPTUNA_N_TRIALS_LGB = 20  # Set to 50+ for production runs
OPTUNA_N_TRIALS_RF = 10

# --- PRIORS ---
BAYESIAN_PRIORS = {
    'PTS': 12.0, 'REB': 4.0, 'AST': 3.0, 'PRA': 18.0,
    'PR': 16.0, 'PA': 15.0, 'RA': 7.0
}

# --- PROP MAPPING ---
MASTER_PROP_MAP = {
    'Points': 'PTS', 'pts': 'PTS',
    'Rebounds': 'REB', 'reb': 'REB',
    'Assists': 'AST', 'ast': 'AST',
    'Pts + Rebs + Asts': 'PRA', 'Pts+Rebs+Asts': 'PRA', 'pra': 'PRA',
    'Rebounds + Assists': 'RA', 'ra': 'RA',
    'Points + Rebounds': 'PR', 'pr': 'PR',
    'Points + Assists': 'PA', 'pa': 'PA'
}

BASE_TARGETS = ['PTS', 'REB', 'AST']
COMPOSITE_TARGETS = ['PRA', 'PR', 'PA', 'RA']
SUPPORTED_PROPS = BASE_TARGETS + COMPOSITE_TARGETS