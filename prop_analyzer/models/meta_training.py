import sys
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.models import registry
from prop_analyzer.utils import common

def build_meta_dataset(days_back=45):
    """Loads graded history to create the training set for the meta-model."""
    # Chronologically sort files to prevent time-series data leakage
    graded_files = sorted(cfg.GRADED_PROPS_PARQUET_DIR.glob("graded_props_*.parquet"))
    
    if not graded_files:
        logging.warning("No graded history found to train meta-model.")
        return None, None, None

    # Implement a rolling window to prevent concept drift (e.g., only last 45 days)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    recent_files = []
    for f in graded_files:
        try:
            date_str = f.stem.replace('graded_props_', '')
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date >= cutoff_date:
                recent_files.append(f)
        except ValueError:
            recent_files.append(f) # Fallback if naming convention varies

    if not recent_files:
        logging.warning(f"No graded history found in the last {days_back} days.")
        return None, None, None

    dfs = [pd.read_parquet(f) for f in recent_files]
    df = pd.concat(dfs, ignore_index=True)

    # --- MAP LEGACY SCHEMA TO CURRENT FEATURES ---
    schema_mapping = {
        'Edge_Type': 'Pick',
        'Model_Pred': 'Proj',
        'Model_Conf': 'Prob'
    }
    
    for old_col, new_col in schema_mapping.items():
        if old_col in old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
            
    if Cols.PREDICTION not in df.columns and 'Proj' in df.columns:
        df[Cols.PREDICTION] = df['Proj']

    req_cols = [Cols.ACTUAL_VAL, Cols.PREDICTION, 'Pick', Cols.PROP_LINE]
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        logging.warning(f"Graded history missing critical columns: {missing_cols}")
        return None, None, None
        
    df = df.dropna(subset=req_cols).copy()
    df = df[df[Cols.ACTUAL_VAL] != df[Cols.PROP_LINE]] # Exclude pushes

    if df.empty:
        logging.warning("No valid graded data after filtering.")
        return None, None, None

    # Define the binary target: Did the primary model's pick win?
    df['IS_WIN'] = 0
    df.loc[(df['Pick'] == 'Over') & (df[Cols.ACTUAL_VAL] > df[Cols.PROP_LINE]), 'IS_WIN'] = 1
    df.loc[(df['Pick'] == 'Under') & (df[Cols.ACTUAL_VAL] < df[Cols.PROP_LINE]), 'IS_WIN'] = 1

    # Calculate Delta Gap Pct
    df['Delta_Gap_Pct'] = np.where(df[Cols.PROP_LINE] > 0, abs(df['Proj'] - df[Cols.PROP_LINE]) / df[Cols.PROP_LINE], 0)

    # Handle optional context metrics with safe defaults
    if 'BLOWOUT_POTENTIAL' not in df.columns: df['BLOWOUT_POTENTIAL'] = 0.0
    else: df['BLOWOUT_POTENTIAL'] = df['BLOWOUT_POTENTIAL'].fillna(0.0)
        
    if 'Consistency_CV' not in df.columns: df['Consistency_CV'] = 0.5 
    if 'Active_Hit%' not in df.columns: df['Active_Hit%'] = 50.0
    if 'Matchup_Hit%' not in df.columns: df['Matchup_Hit%'] = df['Active_Hit%']
        
    df['Matchup_Hit%'] = pd.to_numeric(df['Matchup_Hit%'], errors='coerce').fillna(df['Active_Hit%'])

    # Extract Meta-Features
    meta_features = [
        'Prob', 'Consistency_CV', 'Proj', Cols.PROP_LINE, 
        'Active_Hit%', 'Matchup_Hit%', 'BLOWOUT_POTENTIAL', 'Delta_Gap_Pct'
    ]
    
    available_features = [f for f in meta_features if f in df.columns]
    df = df.dropna(subset=available_features)
    
    if df.empty or len(df) < 50:
         logging.warning("Not enough clean data to train meta-model. Need at least 50 records.")
         return None, None, None

    return df[available_features], df['IS_WIN'], available_features

def train_meta_classifier():
    logging.info("Training Error-Prediction Meta-Model...")
    X, y, feature_names = build_meta_dataset(days_back=45) # 45-day rolling window
    
    if X is None or X.empty:
        logging.info("Meta-Model training aborted due to lack of data.")
        return

    # CHRONOLOGICAL SPLIT: Fixes time-series data leakage
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # 2-Leg Optimizer Alignment: Create sample weights to hyper-focus on the 55% - 65% boundary
    # This prevents the model from wasting learning capacity on >80% or <40% locks, 
    # ensuring the calibrator is deadly accurate right at the 58% parlay inclusion threshold.
    sample_weights = np.ones(len(y_train))
    prob_series = X_train['Prob']
    sample_weights[(prob_series >= 0.55) & (prob_series <= 0.65)] = 2.0

    # Base XGBoost Classifier
    base_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='auc',
        random_state=42
    )

    # PROBABILITY CALIBRATION: Fixes uncalibrated tree probabilities
    calib_method = 'isotonic' if len(X_train) >= 1000 else 'sigmoid'
    
    meta_model = CalibratedClassifierCV(estimator=base_model, method=calib_method, cv=5)
    
    # Fit with the newly engineered parlay-threshold sample weights
    meta_model.fit(X_train, y_train, sample_weight=sample_weights)
    logging.info(f"Calibrating probabilities using method: {calib_method} (Training size: {len(X_train)})")

    # Evaluate
    y_pred_proba = meta_model.predict_proba(X_test)[:, 1]
    y_pred = meta_model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f"Meta-Model AUC: {auc:.3f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    # Save artifacts
    artifacts = {
        'model': meta_model,
        'scaler': None,
        'features': feature_names,
        'metadata': {'type': 'meta_classifier', 'auc': float(auc)}
    }
    registry.save_artifacts('META_CALIBRATOR', artifacts)
    logging.info("Meta-model saved successfully.")

if __name__ == "__main__":
    common.setup_logging(name="meta_training")
    train_meta_classifier()