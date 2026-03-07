from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Graded Output Directories
GRADED_DIR = OUTPUT_DIR / "graded_history"
GRADED_PROPS_CSV_DIR = GRADED_DIR / "props_csv"
GRADED_PROPS_PARQUET_DIR = GRADED_DIR / "props_parquet"
GRADED_PROPS_EXCEL_DIR = GRADED_DIR / "props_excel"
GRADED_PARLAYS_DIR = GRADED_DIR / "parlays"

# Advanced Modeling Directories
MODEL_VERSIONS_DIR = MODEL_DIR / "versions"
MODEL_METADATA_DIR = MODEL_DIR / "metadata"

# Ensure key directories exist
for d in [DATA_DIR, MODEL_DIR, INPUT_DIR, OUTPUT_DIR, GRADED_DIR, 
          GRADED_PROPS_CSV_DIR, GRADED_PROPS_PARQUET_DIR, 
          GRADED_PROPS_EXCEL_DIR, GRADED_PARLAYS_DIR,
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
    
    DAYS_REST = 'DAYS_REST'
    IMPLIED_MINS = 'IMPLIED_MINS'
    RESIDUAL = 'RESIDUAL'
    
    @classmethod
    def get_required_input_cols(cls):
        return [cls.PLAYER_NAME, cls.TEAM, cls.OPPONENT, cls.MATCHUP, cls.PROP_TYPE, cls.PROP_LINE, cls.DATE]

# --- THRESHOLDS & TIERING LOGIC ---
MIN_PROB_FOR_S_TIER = 0.70  
MAX_CV_FOR_S_TIER = 0.25
MIN_L10_HIT_FOR_S_TIER = 0.70

MIN_PROB_FOR_A_TIER = 0.64
MAX_CV_FOR_A_TIER = 0.30

MIN_PROB_FOR_B_TIER = 0.58
MAX_CV_FOR_B_TIER = 0.35

MIN_PROB_FOR_C_TIER = 0.54

MAX_CV_HARD_PASS_OVER = 0.40
MIN_L10_HIT_HARD_PASS_OVER = 0.40

# NEW: Volatility checks for Unders (Prevents High-Variance Ceiling Risk)
# We now use dynamic CV handling for Unders based on line size, and we drop the L10 hit rate 
# filter for Unders entirely to capitalize on situations where minutes are projected to drop.
MAX_CV_HARD_PASS_UNDER_BASE = 0.45  
MAX_CV_HARD_PASS_UNDER_LOW_LINE = 0.85

LIVE_MIN_PROB_THRESHOLD = 0.65
LIVE_BLOWOUT_THRESHOLD = 20
BAYESIAN_PRIOR_WEIGHT = 6.0  
EWMA_DECAY_FACTOR = 0.80     
MIN_GAMES_FOR_ANALYSIS = 5
SKEPTICISM_DECAY_THRESHOLD = 0.40  

# --- ADVANCED TRAINING CONFIGURATION ---
CV_TIME_SPLITS = 5
OPTUNA_N_TRIALS_XGB = 20  
OPTUNA_N_TRIALS_LGB = 20  
OPTUNA_N_TRIALS_RF = 10

# --- PRIORS ---
BAYESIAN_PRIORS = {
    'PTS': 12.0, 'REB': 4.0, 'AST': 3.0, 'PRA': 18.0,
    'PR': 16.0, 'PA': 15.0, 'RA': 7.0
}

# --- PROP MAPPING ---
MASTER_PROP_MAP = {
    # Standard / Underdog
    'Points': 'PTS', 'pts': 'PTS',
    'Rebounds': 'REB', 'reb': 'REB',
    'Assists': 'AST', 'ast': 'AST',
    'Pts + Rebs + Asts': 'PRA', 'pra': 'PRA',
    'Rebounds + Assists': 'RA', 'ra': 'RA',
    'Points + Rebounds': 'PR', 'pr': 'PR',
    'Points + Assists': 'PA', 'pa': 'PA',
    
    # NEW: PrizePicks Specific
    'Pts+Rebs': 'PR',
    'Pts+Asts': 'PA',
    'Rebs+Asts': 'RA',
    'Pts+Rebs+Asts': 'PRA',
}

BASE_TARGETS = ['PTS', 'REB', 'AST']
COMPOSITE_TARGETS = ['PRA', 'PR', 'PA', 'RA']
SUPPORTED_PROPS = BASE_TARGETS + COMPOSITE_TARGETS