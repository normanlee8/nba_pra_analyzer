import sys
import pandas as pd
import numpy as np
import logging
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import re
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.models import registry
from prop_analyzer.utils import common

# Map Prop Categories to Data Column Prefixes
PROP_KEY_MAP = {
    'Points': 'PTS', 'Rebounds': 'REB', 'Assists': 'AST',
    'PRA': 'PRA', 'Pts+Reb+Ast': 'PRA', 
    'Pts+Reb': 'PR', 'Pts+Ast': 'PA', 'Reb+Ast': 'RA',
    'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST',
    'PRA': 'PRA', 'PR': 'PR', 'PA': 'PA', 'RA': 'RA'
}

def rename_features_for_model(df, prop_cat):
    """Maps specific column names to generic definition names."""
    prefix = PROP_KEY_MAP.get(prop_cat, prop_cat)
    
    mapping = {
        f'{prefix}_{Cols.SZN_AVG}': 'SZN Avg',
        f'{prefix}_{Cols.L5_AVG}': 'L5 Avg',  
        f'{prefix}_L5_EWMA': 'L5 EWMA',
        f'{prefix}_L3_AVG': 'L3 Avg',
        f'{prefix}_L10_STD': 'L10_STD_DEV',
        f'SZN_TS_PCT': 'SZN_TS_PCT',
        f'SZN_USG_PROXY': 'SZN_USG_PROXY'
    }
    
    actual_rename = {k: v for k, v in mapping.items() if k in df.columns}
    if actual_rename:
        df = df.rename(columns=actual_rename)
    return df

def get_feature_cols(prop_cat, all_columns):
    """Determines which columns to use for training based on definitions."""
    relevant = feat_defs.BASE_FEATURE_COLS.copy()
    
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for vc in vacancy_cols:
        if vc in all_columns and vc not in relevant:
            relevant.append(vc)

    keywords = feat_defs.RELEVANT_KEYWORDS.get(prop_cat, [])
    rank_cols = [
        c for c in all_columns 
        if ('_RANK' in c or 'TEAM_' in c or 'OPP_' in c or 'DVP_' in c)
        and c not in relevant
        and 'NAME' not in c and 'ABBREV' not in c and Cols.DATE not in c
        and 'SEASON_ID' not in c and Cols.PLAYER_ID not in c
        and c not in vacancy_cols 
    ]

    if keywords:
        filtered_ranks = [
            c for c in rank_cols 
            if any(k in c for k in keywords) 
            or 'PACE' in c or 'EFF' in c or 'DVP_' in c
        ]
        relevant.extend(filtered_ranks)
    else:
        relevant.extend(rank_cols)
    
    allowed_suffixes = feat_defs.PROP_FEATURE_MAP.get(prop_cat, [])
    final_features = set(relevant)
    
    always_keep = ['VS_OPP_GAMES_PLAYED', 'VS_OPP_MIN']
    for f in feat_defs.VS_OPP_FEATURES:
        if f in always_keep: continue
        is_valid = any(f == f"VS_OPP_{s}" for s in allowed_suffixes)
        if not is_valid and f in final_features:
            final_features.remove(f)
            
    for f in feat_defs.HIST_FEATURES:
        if f == 'HIST_VS_OPP_GAMES': continue
        is_valid = any(f.startswith(f"HIST_VS_OPP_{s}_") for s in allowed_suffixes)
        if not is_valid and f in final_features:
            final_features.remove(f)
            
    return [c for c in list(final_features) if c in all_columns]

def backfill_missing_cols(df, cols):
    """Ensures all feature columns exist, setting to NaN for Imputer."""
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan 
    return df

def train_ensemble_model(df, target_cols, prop_name):
    """
    Core function for training optimized models. Supports both Single Target (Composites) 
    and Multi-Output Regression (P, R, A simultaneously).
    """
    logging.info(f"Training Professional Ensemble for {prop_name} (Targets: {target_cols})...")

    # --- 1. TIME SERIES SPLIT PROTECTION (Walk-Forward Setup) ---
    date_col = Cols.DATE if Cols.DATE in df.columns else 'GAME_DATE'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col, ascending=True).reset_index(drop=True)
    else:
        logging.warning(f"[{prop_name}] Date column missing. Cannot guarantee strict walk-forward validation!")

    # --- 2. PREPARE TARGETS AND FEATURES ---
    df = df.dropna(subset=target_cols).copy()
    
    feature_list = []
    for t in target_cols:
        df = rename_features_for_model(df, t)
        feature_list.extend(get_feature_cols(t, df.columns))
    
    feature_list = list(set(feature_list)) 
    
    if len(feature_list) < 5:
        logging.warning(f"[{prop_name}] Not enough features found. Skipping.")
        return

    df = backfill_missing_cols(df, feature_list)
    sanitized_cols = [re.sub(r'[^\w\s]', '_', str(col)).replace(' ', '_') for col in feature_list]
    
    X = df[feature_list].copy()
    X.columns = sanitized_cols
    y = df[target_cols].copy()

    is_multi_output = len(target_cols) > 1
    if not is_multi_output:
        y = y.iloc[:, 0]

    # --- 3. PREPROCESSING PIPELINE ---
    zero_impute_keywords = ['HIST_', 'VS_OPP_', 'DVP_', 'MISSING']
    hist_cols = [c for c in X.columns if any(k in c for k in zero_impute_keywords)]
    base_cols = [c for c in X.columns if c not in hist_cols]
    
    preprocessor = ColumnTransformer([
        ('zero_fill', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
            ('scaler', StandardScaler())
        ]), hist_cols),
        ('median_fill', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), base_cols)
    ], remainder='passthrough')

    X_proc = preprocessor.fit_transform(X)
    X_proc_df = pd.DataFrame(X_proc, columns=sanitized_cols)

    # --- 4. TIME-SERIES CROSS VALIDATION & OPTUNA ---
    tscv = TimeSeriesSplit(n_splits=5) 

    def optimize_base_models():
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # XGBoost Optuna
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            base_mod = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, objective='reg:squarederror')
            mod = MultiOutputRegressor(base_mod) if is_multi_output else base_mod
            
            maes = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                mod.fit(X_proc_df.iloc[train_idx], y.iloc[train_idx])
                preds = mod.predict(X_proc_df.iloc[val_idx])
                maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            return np.mean(maes)

        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(xgb_objective, n_trials=10) # Set higher (e.g., 50) for deep production runs
        
        # LightGBM Optuna
        def lgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            }
            base_mod = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            mod = MultiOutputRegressor(base_mod) if is_multi_output else base_mod
            
            maes = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                mod.fit(X_proc_df.iloc[train_idx], y.iloc[train_idx])
                preds = mod.predict(X_proc_df.iloc[val_idx])
                maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            return np.mean(maes)

        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lgb_objective, n_trials=10)

        # Random Forest Optuna
        def rf_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
            }
            base_mod = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            mod = MultiOutputRegressor(base_mod) if is_multi_output else base_mod
            
            maes = []
            for train_idx, val_idx in tscv.split(X_proc_df):
                mod.fit(X_proc_df.iloc[train_idx], y.iloc[train_idx])
                preds = mod.predict(X_proc_df.iloc[val_idx])
                maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            return np.mean(maes)

        study_rf = optuna.create_study(direction='minimize')
        study_rf.optimize(rf_objective, n_trials=5)

        return study_xgb.best_params, study_lgb.best_params, study_rf.best_params

    logging.info(f"[{prop_name}] Running Hyperparameter Optimization (Walk-Forward CV)...")
    xgb_params, lgb_params, rf_params = optimize_base_models()

    # --- 5. FINAL ENSEMBLE TRAINING ---
    xgb_best = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
    lgb_best = lgb.LGBMRegressor(**lgb_params, random_state=42, n_jobs=-1, verbose=-1)
    rf_best = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)

    # Wrap inside Voting Regressor
    ensemble = VotingRegressor(estimators=[
        ('xgb', xgb_best),
        ('lgb', lgb_best),
        ('rf', rf_best)
    ])

    # If Multi-Output, wrap the ensemble itself
    final_model = MultiOutputRegressor(ensemble) if is_multi_output else ensemble

    logging.info(f"[{prop_name}] Fitting Final Ensemble Model on full training data...")
    final_model.fit(X_proc_df, y)

    # --- 6. METRICS EVALUATION (Holdout of Final 15%) ---
    split_idx = int(len(X_proc_df) * 0.85)
    X_val, y_val = X_proc_df.iloc[split_idx:], y.iloc[split_idx:]
    val_preds = final_model.predict(X_val)

    mae = mean_absolute_error(y_val, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    r2 = r2_score(y_val, val_preds)
    logging.info(f"[{prop_name}] Holdout Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # --- 7. SHAP FEATURE IMPORTANCE ---
    logging.info(f"[{prop_name}] Calculating SHAP Feature Importance...")
    shap_importance = []
    try:
        # Extract the XGB component from the ensemble for clean TreeExplainer interpretation
        if is_multi_output:
            explainer_model = final_model.estimators_[0].named_estimators_['xgb']
        else:
            explainer_model = final_model.named_estimators_['xgb']

        explainer = shap.TreeExplainer(explainer_model)
        X_sample = X_proc_df.sample(n=min(1500, len(X_proc_df)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list): 
            shap_values = shap_values[0]
        
        vals = np.abs(shap_values).mean(0)
        feat_imp_df = pd.DataFrame({'Feature': sanitized_cols, 'Importance': vals})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        shap_importance = feat_imp_df.head(20).to_dict(orient='records')
        
        logging.info(f"[{prop_name}] Top Feature: {shap_importance[0]['Feature']} (Score: {shap_importance[0]['Importance']:.4f})")
    except Exception as e:
        logging.warning(f"SHAP extraction failed: {e}")

    # --- 8. SAVE ARTIFACTS AND METADATA ---
    metadata = {
        'training_date': datetime.now().isoformat(),
        'target_cols': target_cols,
        'is_multi_output': is_multi_output,
        'metrics': {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)},
        'hyperparameters': {'xgb': xgb_params, 'lgb': lgb_params, 'rf': rf_params},
        'top_features': shap_importance
    }

    artifacts = {
        'scaler': preprocessor,
        'features': sanitized_cols,
        'model': final_model,
        'metadata': metadata
    }
    
    registry.save_artifacts(prop_name, artifacts)
    logging.info(f"[{prop_name}] Successfully trained and saved.")

def main():
    common.setup_logging(name="train_models")
    logging.info(">>> STARTING ADVANCED MODEL TRAINING PIPELINE")

    train_file = cfg.MASTER_TRAINING_FILE
    if not train_file.exists():
        logging.critical(f"Training dataset not found at {train_file}")
        return

    df = pd.read_parquet(train_file)
    if df.empty:
        logging.critical("Training dataset is empty.")
        return

    # 1. Train Joint Multi-Output Engine for Base Stats (P, R, A simultaneously)
    base_targets = ['PTS', 'REB', 'AST']
    available_base = [t for t in base_targets if t in df.columns]
    
    if len(available_base) == 3:
        logging.info("--- Training Multi-Output Engine (PTS, REB, AST) ---")
        train_ensemble_model(df, target_cols=available_base, prop_name='BASE_MULTI')
    else:
        logging.warning("Missing base columns. Cannot train joint BASE_MULTI model.")

    # 2. Train Individual Ensemble Models for Composites (PRA, PR, etc.)
    composite_props = ['PRA', 'PR', 'PA', 'RA']
    available_composites = [p for p in composite_props if p in df.columns]
    
    for prop in available_composites:
        logging.info(f"--- Training Composite Engine: {prop} ---")
        train_ensemble_model(df, target_cols=[prop], prop_name=prop)

    logging.info("<<< TRAINING COMPLETE.")

if __name__ == "__main__":
    main()